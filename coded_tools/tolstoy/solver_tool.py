from __future__ import annotations

import os
from typing import Any

from neuro_san.interfaces.coded_tool import CodedTool
from neuro_san.internals.graph.activations.branch_activation import BranchActivation

from coded_tools.tools.coded_tool_agent_caller import CodedToolAgentCaller
from coded_tools.tolstoy.engine import TolstoyEngine, TolstoyRunConfig


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _debug_enabled() -> bool:
    return os.environ.get("NS_TOLSTOY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug(message: str) -> None:
    if _debug_enabled():
        print(f"[tolstoy.tool] {message}", flush=True)


class TolstoySolverTool(BranchActivation, CodedTool):
    """Coded tool wrapper around the Tolstoy DAG engine."""

    def invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> Any:
        # Neuro-SAN may call async_invoke directly. Keep invoke as a no-op shim.
        return None

    async def async_invoke(self, args: dict[str, Any], sly_data: dict[str, Any]) -> Any:
        problem = str(args.get("problem") or "").strip()
        if not problem:
            return "A non-empty problem is required."
        _debug(f"start problem={problem[:120]!r}")

        tools = args.get("tools") or {}
        callers = {
            role: CodedToolAgentCaller(self, name)
            for role, name in tools.items()
            if isinstance(name, str) and name
        }

        engine = TolstoyEngine(
            callers=callers,
            config=TolstoyRunConfig(
                k_answer=int(args.get("k_answer", 3) or 3),
                k_validator=int(args.get("k_validator", 3) or 3),
                k_gc=int(args.get("k_gc", 3) or 3),
                max_iter=int(args.get("max_iter", 50) or 50),
                max_active_nodes=(
                    None if args.get("max_active_nodes") in (None, "", 0, "0") else int(args.get("max_active_nodes"))
                ),
                max_proposal_retries=int(args.get("max_proposal_retries", 5) or 5),
                use_gc=_coerce_bool(args.get("use_gc"), False),
                use_reasons=_coerce_bool(args.get("use_reasons"), False),
                show_nc_answers=_coerce_bool(args.get("show_nc_answers"), False),
                use_scratchpad=_coerce_bool(args.get("use_scratchpad"), False),
                cite_problem=_coerce_bool(args.get("cite_problem"), True),
                answer_temperature=float(args.get("answer_temperature", 0.7) or 0.7),
                frames_path=args.get("frames_path"),
                result_path=args.get("result_path"),
            ),
        )

        result = await engine.run(problem)
        _debug(f"done answer={result.get('answer', '')!r} iterations={result.get('iterations')}")
        sly_data["tolstoy_result"] = result
        return result["answer"]
