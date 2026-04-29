from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from coded_tools.tolstoy.parsing import (
    ConsolidationPlan,
    ParsedAnswer,
    Proposal,
    extract_failures,
    normalize_answer,
    parse_answer,
    parse_consolidation_plan,
    parse_equivalence_report,
    parse_gc_response,
    parse_proposal,
    parse_validator_report,
)
from coded_tools.tolstoy.types import DagState, Node, NodeStatus
from coded_tools.tools.agent_caller import AgentCaller

LOGGER = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return os.environ.get("NS_TOLSTOY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug(message: str) -> None:
    if _debug_enabled():
        print(f"[tolstoy.engine] {message}", flush=True)


@dataclass
class TolstoyRunConfig:
    k_answer: int = 3
    k_validator: int = 3
    k_gc: int = 3
    max_iter: int = 50
    max_active_nodes: int | None = 10
    max_proposal_retries: int = 5
    use_gc: bool = False
    use_reasons: bool = False
    show_nc_answers: bool = False
    use_scratchpad: bool = False
    cite_problem: bool = True
    answer_temperature: float = 0.7
    frames_path: str | None = None
    result_path: str | None = None


class TolstoyEngine:
    """DAG orchestrator that delegates LLM work to in-network Neuro-SAN agents."""

    def __init__(self, callers: dict[str, AgentCaller], config: TolstoyRunConfig):
        self.callers = callers
        self.config = config

    async def run(self, problem: str) -> dict:
        state = DagState(problem=problem, frames_path=self.config.frames_path)
        _debug(f"run start max_iter={self.config.max_iter} k_answer={self.config.k_answer} k_validator={self.config.k_validator}")
        if self.config.cite_problem:
            fact0 = Node(
                id=0,
                question="What is the original problem statement?",
                answer=problem,
                parent_ids=[],
                status=NodeStatus.ANSWERED,
            )
            state.add_node(fact0)
            state.snapshot("node 0 added [problem]")

        final_node_id: int | None = None

        for iteration in range(self.config.max_iter):
            LOGGER.info("Tolstoy iteration %s/%s", iteration + 1, self.config.max_iter)
            _debug(f"iteration {iteration + 1}/{self.config.max_iter}")

            if await self._maybe_consolidate(state):
                _debug("consolidated active nodes")
                continue

            proposal = await self._get_next_proposal(state)
            if proposal is None:
                _debug("no proposal returned")
                continue

            if proposal.final_answer_node_id is not None:
                final = state.get(proposal.final_answer_node_id)
                if final and final.status == NodeStatus.ANSWERED:
                    _debug(f"final answer via node {final.id}")
                    final_node_id = final.id
                    state.snapshot(f"done via node {final.id}", final_node_id=final.id)
                    return self._finalize(state, final.answer, iteration + 1, final_node_id)
                continue

            if proposal.question is None:
                _debug("proposal had no question")
                continue
            _debug(f"proposal parents={proposal.parent_ids} question={proposal.question[:160]!r}")

            accepted, reasoning = await self._validate(state, proposal.question, proposal.parent_ids)
            if not accepted:
                _debug(f"proposal rejected {reasoning[:200]!r}")
                rejected = Node(
                    id=len(state.nodes),
                    question=proposal.question,
                    answer=reasoning,
                    parent_ids=proposal.parent_ids,
                    status=NodeStatus.REJECTED,
                    reason=proposal.reason if self.config.use_reasons else None,
                )
                state.add_node(rejected)
                await self._maybe_update_scratchpad(state, rejected)
                state.snapshot(f"node {rejected.id} added [rejected]")
                continue

            status, answer_text, raw_answers = await self._answer(state, proposal.question, proposal.parent_ids)
            _debug(f"answer status={status.value} text={answer_text[:200]!r}")
            new_node = Node(
                id=len(state.nodes),
                question=proposal.question,
                answer=answer_text,
                parent_ids=proposal.parent_ids,
                status=status,
                reason=proposal.reason if self.config.use_reasons else None,
                raw_answers=raw_answers,
            )
            state.add_node(new_node)

            if status == NodeStatus.ANSWERED:
                if proposal.retire_ids:
                    reasons = {
                        node_id: f"retired by proposer after node {new_node.id}"
                        for node_id in proposal.retire_ids
                    }
                    state.retire(proposal.retire_ids, reasons)
                if self.config.use_gc:
                    await self._run_gc(state, new_node)

            await self._maybe_update_scratchpad(state, new_node)
            state.snapshot(f"node {new_node.id} added [{new_node.status.value}]")

        synthesized = await self._synthesize_final_answer(state)
        if synthesized is not None:
            _debug(f"synthesized final answer via node {synthesized.id}")
            state.snapshot(f"done via synthesized node {synthesized.id}", final_node_id=synthesized.id)
            return self._finalize(state, synthesized.answer, self.config.max_iter, synthesized.id)

        best = self._latest_answered(state)
        final_answer = best.answer if best else ""
        final_node_id = best.id if best else None
        _debug(f"fallback final_node_id={final_node_id} answer={final_answer[:200]!r}")
        state.snapshot("done via fallback", final_node_id=final_node_id)
        return self._finalize(state, final_answer, self.config.max_iter, final_node_id)

    async def _maybe_consolidate(self, state: DagState) -> bool:
        caller = self.callers.get("consolidator")
        max_active = self.config.max_active_nodes
        if not caller or not max_active:
            return False

        active_answered = state.active_answered(include_fact0=False)
        if len(active_answered) < max_active:
            return False

        raw = await caller.call_agent(
            {
                "problem": state.problem,
                "active_facts": self._format_active_nodes(state, answered_only=True),
            }
        )
        plan: ConsolidationPlan = parse_consolidation_plan(raw)
        if plan.error:
            LOGGER.warning("Consolidation skipped: %s", plan.error)
            return False

        invalid = [
            node_id
            for node_id in plan.node_ids
            if node_id not in state.active_ids
            or state.get(node_id) is None
            or state.get(node_id).status != NodeStatus.ANSWERED
        ]
        if invalid:
            LOGGER.warning("Consolidation skipped due to invalid nodes: %s", invalid)
            return False

        accepted, reasoning = await self._validate(state, plan.merge_question or "", plan.node_ids)
        if not accepted:
            LOGGER.warning("Consolidation question rejected: %s", reasoning)
            return False

        status, answer_text, raw_answers = await self._answer(state, plan.merge_question or "", plan.node_ids)
        if status != NodeStatus.ANSWERED:
            return False

        merged = Node(
            id=len(state.nodes),
            question=plan.merge_question or "",
            answer=answer_text,
            parent_ids=plan.node_ids,
            status=NodeStatus.ANSWERED,
            raw_answers=raw_answers,
        )
        state.add_node(merged)
        reasons = {node_id: f"consolidated into node {merged.id}" for node_id in plan.node_ids}
        state.retire(plan.node_ids, reasons)
        state.snapshot(f"node {merged.id} added [consolidated]")
        return True

    async def _get_next_proposal(self, state: DagState) -> Proposal | None:
        caller = self.callers["proposer"]
        feedback: str | None = None

        for attempt in range(self.config.max_proposal_retries):
            _debug(f"proposer attempt {attempt + 1}")
            raw = await caller.call_agent(
                {
                    "problem": state.problem,
                    "active_facts": self._format_active_nodes(state),
                    "scratchpad": state.scratchpad or "(empty)",
                    "feedback": feedback or "",
                    "use_reasons": self.config.use_reasons,
                    "cite_problem": self.config.cite_problem,
                }
            )
            proposal = parse_proposal(raw, cite_problem=self.config.cite_problem)
            if proposal.error:
                _debug(f"proposal parse error={proposal.error}")
                feedback = proposal.error
                continue

            if proposal.final_answer_node_id is not None:
                return proposal

            invalid = [
                node_id
                for node_id in proposal.parent_ids
                if node_id not in state.active_ids
                or state.get(node_id) is None
                or state.get(node_id).status != NodeStatus.ANSWERED
            ]
            if invalid:
                _debug(f"proposal invalid parents={invalid}")
                feedback = f"Only active ANSWERED facts may be cited. Invalid ids: {invalid}."
                continue
            return proposal

        return None

    async def _validate(self, state: DagState, question: str, parent_ids: list[int]) -> tuple[bool, str]:
        caller = self.callers["validator"]
        source_facts = self._format_source_nodes(state, parent_ids)
        _debug(f"validate start parents={parent_ids}")
        raw_reports = await self._call_many(
            caller,
            self.config.k_validator,
            {
                "problem": state.problem,
                "source_facts": source_facts,
                "question": question,
            },
        )
        reports = [parse_validator_report(raw) for raw in raw_reports]
        if all(report.accepted for report in reports):
            _debug("validate accepted")
            return True, "\n---\n".join(report.details for report in reports)
        _debug("validate rejected")
        return False, "\n---\n".join(extract_failures(report.details) for report in reports)

    async def _answer(self, state: DagState, question: str, parent_ids: list[int]) -> tuple[NodeStatus, str, list[str]]:
        caller = self.callers["answerer"]
        _debug(f"answer start question={question[:160]!r}")
        raw_answers = await self._call_many(
            caller,
            self.config.k_answer,
            {
                "question": question,
                "answer_temperature": self.config.answer_temperature,
            },
        )
        parsed = [parse_answer(raw) for raw in raw_answers]

        final_answers = [item.value for item in parsed if item.kind == "final" and item.value]
        contradictions = [item.value for item in parsed if item.kind == "contradiction" and item.value]

        same, canonical = await self._judge_equivalence(question, final_answers)
        if same and canonical is not None:
            return NodeStatus.ANSWERED, canonical, raw_answers

        if contradictions and len(contradictions) == len([item for item in parsed if item.kind != "invalid"]):
            return NodeStatus.CONTRADICTION, contradictions[0], raw_answers

        if self.config.show_nc_answers:
            lines = ["NO CONSENSUS"]
            for index, answer in enumerate(final_answers or [item.value for item in parsed], start=1):
                if answer:
                    lines.append(f"Answerer {index}: {answer}")
            return NodeStatus.NO_CONSENSUS, "\n".join(lines), raw_answers
        return NodeStatus.NO_CONSENSUS, "NO CONSENSUS", raw_answers

    async def _run_gc(self, state: DagState, new_node: Node) -> None:
        caller = self.callers.get("gc")
        if not caller:
            return

        raw_responses = await self._call_many(
            caller,
            self.config.k_gc,
            {
                "problem": state.problem,
                "active_facts": self._format_active_nodes(state, answered_only=True),
                "new_fact": self._format_node(new_node),
            },
        )

        retire_sets: list[set[int]] = []
        reason_maps: list[dict[int, str]] = []
        for raw in raw_responses:
            retire_ids, reasons = parse_gc_response(raw)
            retire_sets.append(set(retire_ids))
            reason_maps.append(reasons)

        if not retire_sets:
            return

        intersection = retire_sets[0]
        for retire_set in retire_sets[1:]:
            intersection &= retire_set
        retired = sorted(intersection)
        if not retired:
            return

        reasons_out: dict[int, str] = {}
        for node_id in retired:
            for reason_map in reason_maps:
                if node_id in reason_map:
                    reasons_out[node_id] = reason_map[node_id]
                    break
        state.retire(retired, reasons_out)

    async def _maybe_update_scratchpad(self, state: DagState, node: Node) -> None:
        caller = self.callers.get("scratchpad")
        if not caller or not self.config.use_scratchpad:
            return
        if node.status not in {NodeStatus.REJECTED, NodeStatus.NO_CONSENSUS, NodeStatus.CONTRADICTION}:
            return

        raw = await caller.call_agent(
            {
                "problem": state.problem,
                "scratchpad": state.scratchpad or "(empty)",
                "event_type": node.status.value,
                "question": node.question,
                "result": node.answer,
            }
        )
        state.scratchpad = raw.strip() or state.scratchpad

    async def _judge_equivalence(self, question: str, answers: list[str]) -> tuple[bool, str | None]:
        if not answers:
            return False, None
        normalized = {normalize_answer(answer) for answer in answers}
        if len(normalized) == 1:
            return True, answers[0].strip()

        caller = self.callers.get("equivalence_judge")
        if not caller:
            return False, None

        raw = await caller.call_agent({"question": question, "answers": answers})
        same, canonical = parse_equivalence_report(raw)
        if same:
            return True, (canonical or answers[0].strip())
        return False, None

    async def _synthesize_final_answer(self, state: DagState) -> Node | None:
        if "answerer" not in self.callers:
            return None

        parent_ids = [node.id for node in state.active_answered(include_fact0=False)]
        if not parent_ids:
            return None

        question = self._build_final_synthesis_question(state, parent_ids)
        status, answer_text, raw_answers = await self._answer(state, question, parent_ids)
        if status != NodeStatus.ANSWERED:
            _debug(f"final synthesis failed status={status.value}")
            return None

        node = Node(
            id=len(state.nodes),
            question=question,
            answer=answer_text,
            parent_ids=parent_ids,
            status=NodeStatus.ANSWERED,
            raw_answers=raw_answers,
        )
        state.add_node(node)
        return node

    async def _call_many(self, caller: AgentCaller, count: int, tool_args: dict) -> list[str]:
        _debug(f"dispatch {max(count, 1)} call(s) to {caller.get_name()}")
        coroutines = [caller.call_agent(tool_args) for _ in range(max(count, 1))]
        results = list(await asyncio.gather(*coroutines))
        _debug(f"completed {len(results)} call(s) to {caller.get_name()}")
        return results

    def _build_final_synthesis_question(self, state: DagState, parent_ids: list[int]) -> str:
        facts = self._format_source_nodes(state, parent_ids)
        return (
            "Use only the original problem statement and the cited facts below.\n\n"
            f"Original problem:\n{state.problem}\n\n"
            "Cited facts:\n"
            f"{facts}\n\n"
            "Solve the original problem. Return the answer in exactly the format requested "
            "inside the original problem statement."
        )

    def _format_node(self, node: Node) -> str:
        status = "" if node.status == NodeStatus.ANSWERED else f" [{node.status.value}]"
        reason = f"\nReason: {node.reason}" if node.reason else ""
        return (
            f"Fact {node.id}{status}\n"
            f"Question: {node.question}{reason}\n"
            f"Answer: {node.answer}"
        )

    def _format_active_nodes(self, state: DagState, answered_only: bool = False) -> str:
        nodes = state.active_answered(include_fact0=False) if answered_only else state.active_nodes
        if not nodes:
            return "(none)"
        return "\n\n".join(self._format_node(node) for node in nodes)

    def _format_source_nodes(self, state: DagState, parent_ids: list[int]) -> str:
        nodes = [state.get(node_id) for node_id in parent_ids]
        resolved = [node for node in nodes if node is not None]
        if not resolved:
            return "(none)"
        return "\n\n".join(self._format_node(node) for node in resolved)

    def _latest_answered(self, state: DagState) -> Node | None:
        answered = [node for node in state.nodes if node.status == NodeStatus.ANSWERED and node.id != 0]
        return answered[-1] if answered else None

    def _finalize(self, state: DagState, answer: str, iterations: int, final_node_id: int | None) -> dict:
        result = {
            "answer": answer,
            "iterations": iterations,
            "nodes": len(state.nodes),
            "final_node_id": final_node_id,
            "scratchpad": state.scratchpad,
            "frames": state.frames,
        }
        if self.config.result_path:
            Path(self.config.result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
