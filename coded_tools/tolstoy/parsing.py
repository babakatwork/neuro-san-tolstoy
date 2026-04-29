from __future__ import annotations

import re
from dataclasses import dataclass


_KEYWORD_RE = re.compile(
    r"^(PARENT FACTS|PROPOSED QUESTION|RETIRE|REASON|FINAL ANSWER NODE|CONSOLIDATE|MERGE QUESTION)\s*:",
    re.IGNORECASE,
)


@dataclass
class Proposal:
    question: str | None
    parent_ids: list[int]
    retire_ids: list[int]
    final_answer_node_id: int | None
    reason: str | None
    error: str | None


@dataclass
class ConsolidationPlan:
    node_ids: list[int]
    merge_question: str | None
    error: str | None


@dataclass
class ParsedAnswer:
    kind: str
    value: str


@dataclass
class ValidatorReport:
    accepted: bool
    details: str


def extract_fact_ids(text: str) -> list[int]:
    return sorted({int(match) for match in re.findall(r"\bfact\s+(\d+)\b", text or "", flags=re.IGNORECASE)})


def parse_id_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip().isdigit()]


def _extract_multiline_value(raw: str, label: str) -> str | None:
    lines = raw.splitlines()
    prefix = f"{label}:"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith(prefix.upper()):
            parts = [stripped.split(":", 1)[1].strip()]
            for continuation in lines[index + 1 :]:
                if _KEYWORD_RE.match(continuation.strip()):
                    break
                parts.append(continuation.rstrip())
            return "\n".join(parts).strip()
    return None


def parse_proposal(raw: str, cite_problem: bool) -> Proposal:
    final_answer_node_id = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FINAL ANSWER NODE:"):
            value = stripped.split(":", 1)[1].strip()
            match = re.search(r"\d+", value)
            if not match:
                return Proposal(None, [], [], None, None, "FINAL ANSWER NODE must include an integer id.")
            final_answer_node_id = int(match.group(0))
            return Proposal(None, [], [], final_answer_node_id, None, None)

    question = _extract_multiline_value(raw, "PROPOSED QUESTION")
    if not question:
        return Proposal(None, [], [], None, None, "Missing PROPOSED QUESTION line.")

    parent_ids: list[int] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("PARENT FACTS:"):
            value = stripped.split(":", 1)[1].strip()
            if value.upper() != "NONE":
                parent_ids = parse_id_list(value)
            break
    parent_ids = sorted(set(parent_ids) | set(extract_fact_ids(question)))
    if cite_problem and not parent_ids:
        return Proposal(None, [], [], None, None, "At least one parent fact must be declared when cite_problem is enabled.")

    retire_ids: list[int] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("RETIRE:"):
            value = stripped.split(":", 1)[1].strip()
            if value and value.upper() != "NONE":
                retire_ids = parse_id_list(value)
            break

    reason = _extract_multiline_value(raw, "REASON")
    return Proposal(question, parent_ids, retire_ids, None, reason, None)


def parse_validator_report(raw: str) -> ValidatorReport:
    verdict = None
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            verdict = stripped.split(":", 1)[1].strip().upper()
            break
    accepted = verdict == "ACCEPT"
    details = raw.strip() or "Validator returned an empty response."
    return ValidatorReport(accepted=accepted, details=details)


def extract_failures(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    failures = [line for line in lines if line.upper().startswith("ITEM ") and "FAIL" in line.upper()]
    verdict = next((line for line in lines if line.upper().startswith("VERDICT:")), None)
    payload = failures + ([verdict] if verdict else [])
    return "\n".join(payload) if payload else (raw.strip() or "Validator rejected the question.")


def parse_answer(raw: str) -> ParsedAnswer:
    lines = raw.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip()
        if stripped.upper().startswith("CONTRADICTION:"):
            value = stripped.split(":", 1)[1].strip()
            rest = "\n".join(part.strip() for part in lines[index + 1 :] if part.strip())
            return ParsedAnswer("contradiction", f"{value}\n{rest}".strip() if rest else value)
        if stripped.upper().startswith("FINAL ANSWER:"):
            value = stripped.split(":", 1)[1].strip()
            rest = "\n".join(part.rstrip() for part in lines[index + 1 :] if part.strip())
            return ParsedAnswer("final", f"{value}\n{rest}".strip() if rest else value)
    fallback = next((line.strip() for line in reversed(lines) if line.strip()), "")
    return ParsedAnswer("invalid", fallback)


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def parse_equivalence_report(raw: str) -> tuple[bool, str | None]:
    verdict = None
    canonical = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            verdict = stripped.split(":", 1)[1].strip().upper()
        elif stripped.upper().startswith("CANONICAL:"):
            canonical = stripped.split(":", 1)[1].strip() or None
    return verdict == "SAME", canonical


def parse_gc_response(raw: str) -> tuple[list[int], dict[int, str]]:
    retire_ids: list[int] = []
    reasons: dict[int, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("RETIRE:"):
            value = stripped.split(":", 1)[1].strip()
            if value.upper() != "NONE":
                retire_ids = parse_id_list(value)
        else:
            match = re.match(r"REASON\s+(\d+)\s*:\s*(.+)", stripped, flags=re.IGNORECASE)
            if match:
                reasons[int(match.group(1))] = match.group(2).strip()
    return retire_ids, reasons


def parse_consolidation_plan(raw: str) -> ConsolidationPlan:
    node_ids: list[int] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CONSOLIDATE:"):
            node_ids = parse_id_list(stripped.split(":", 1)[1].strip())
            break
    merge_question = _extract_multiline_value(raw, "MERGE QUESTION")
    if len(node_ids) < 2:
        return ConsolidationPlan(node_ids, merge_question, "CONSOLIDATE must name at least two answered nodes.")
    if not merge_question:
        return ConsolidationPlan(node_ids, None, "Missing MERGE QUESTION line.")
    return ConsolidationPlan(node_ids, merge_question, None)
