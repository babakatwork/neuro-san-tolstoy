from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from longcot import load_questions, verify
from longcot._parsing import extract_last_balanced_brackets, extract_solution
from longcot._types import ChemistryVerifyOptions, MathVerifyOptions, VerifyOptions
from leaf_common.time.timeout_reached_exception import TimeoutReachedException

from neuro_san.client.agent_session_factory import AgentSessionFactory
from neuro_san.client.streaming_input_processor import StreamingInputProcessor
from neuro_san.internals.chat.data_driven_chat_session import DataDrivenChatSession

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AGENT_MANIFEST_FILE", str(REPO_ROOT / "registries" / "manifest.hocon"))
os.environ.setdefault("AGENT_TOOL_PATH", str(REPO_ROOT / "coded_tools"))

RESULTS_DIR = REPO_ROOT / "results" / "longcot"
FRAMES_DIR = RESULTS_DIR / "frames"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AGENT = "tolstoy/tolstoy_reasoner"
DEFAULT_SESSION_TYPE = os.environ.get("NS_SESSION_TYPE", "http")
SOLUTION_FORMAT_RE = re.compile(r"format\s*:\s*solution\s*=\s*(.+)", re.IGNORECASE)

NO_FALLBACK = VerifyOptions(
    math=MathVerifyOptions(enable_fallback=False),
    chemistry=ChemistryVerifyOptions(enable_fallback=False),
)

_LOCK = threading.Lock()
_COMPLETED = 0
_TOTAL = 0


def make_thread(timeout_ms: float) -> dict[str, Any]:
    return {
        "last_chat_response": None,
        "prompt": "",
        "timeout": timeout_ms,
        "num_input": 0,
        "user_input": None,
        "sly_data": {},
        "chat_filter": {"chat_filter_type": "MAXIMAL"},
    }


def normalize_agent_name(session_type: str, agent_name: str) -> str:
    if not agent_name:
        return agent_name
    if session_type == "direct":
        if "/" not in agent_name and not agent_name.endswith((".hocon", ".json")):
            return f"tolstoy/{agent_name}"
        return agent_name
    if "/" in agent_name and not agent_name.endswith((".hocon", ".json")):
        return agent_name.rsplit("/", 1)[-1]
    return agent_name


def create_session(session_type: str, agent_name: str, host: str, port: int, timeout_ms: float):
    factory = AgentSessionFactory()
    return factory.create_session(
        session_type,
        agent_name,
        host,
        port,
        False,
        {"user_id": os.environ.get("USER") or "benchmark"},
        timeout_ms / 1000.0,
    )


def _log_line(message: str) -> None:
    with _LOCK:
        print(message, flush=True)


def _assert_direct_session_initializable_sync(session) -> None:
    async def _run() -> None:
        invocation_context = session.invocation_context.safe_shallow_copy()
        chat_session = DataDrivenChatSession(agent_network=session.agent_network)
        try:
            await chat_session.set_up(invocation_context, {})
        finally:
            with suppress(Exception):
                await chat_session.delete_resources()
            with suppress(Exception):
                invocation_context.close()

    asyncio.run(_run())


def run_request(session, thread: dict[str, Any], request: str, thinking_file: str) -> tuple[dict[str, Any], dict[str, Any]]:
    processor = StreamingInputProcessor("DEFAULT", thinking_file, session, None)
    next_thread = dict(thread)
    next_thread["user_input"] = request
    next_thread = processor.process_once(next_thread)
    sly_data = next_thread.get("sly_data") or {}
    return next_thread, sly_data


def run_request_with_progress(
    session,
    thread: dict[str, Any],
    request: str,
    heartbeat_s: float,
    question_id: str,
    thinking_file: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if heartbeat_s <= 0:
        return run_request(session, thread, request, thinking_file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_request, session, thread, request, thinking_file)
        start = time.time()
        while True:
            try:
                return future.result(timeout=heartbeat_s)
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - start
                _log_line(f"  [{question_id}] still working: {elapsed:.1f}s")


def build_request(question_text: str, args, frames_path: str, result_path: str) -> str:
    payload = {
        "problem": question_text,
        "max_iter": args.max_iter,
        "max_active_nodes": args.max_active_nodes,
        "k_answer": args.k_answer,
        "k_validator": args.k_validator,
        "k_gc": args.k_gc,
        "max_proposal_retries": args.max_proposal_retries,
        "use_gc": args.use_gc,
        "use_reasons": args.use_reasons,
        "show_nc_answers": args.show_nc_answers,
        "use_scratchpad": args.use_scratchpad,
        "cite_problem": args.cite_problem,
        "answer_temperature": args.answer_temperature,
        "frames_path": frames_path,
        "result_path": result_path,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _infer_expected_solution_shape(prompt: str) -> str:
    match = SOLUTION_FORMAT_RE.search(prompt or "")
    if not match:
        return "unknown"
    template = match.group(1).strip()
    if template.startswith("["):
        return "list"
    return "scalar"


def format_benchmark_response(question, response: str) -> str:
    raw = (response or "").strip()
    if not raw:
        return ""

    body = (extract_solution(raw) or raw).strip()
    shape = _infer_expected_solution_shape(question.prompt or "")

    if shape == "list":
        bracketed = extract_last_balanced_brackets(body)
        if bracketed:
            body = bracketed
        elif not body.startswith("["):
            body = f"[{body}]"
    elif shape == "scalar":
        body = next((line.strip() for line in body.splitlines() if line.strip()), body)

    return f"solution = {body}"


def _verification_options(disable_fallback: bool) -> VerifyOptions | None:
    if not disable_fallback:
        return None
    return NO_FALLBACK


def filter_questions_by_difficulty(questions, difficulty: str | None):
    if not difficulty:
        return list(questions)

    normalized = str(difficulty).strip().lower()
    if normalized == "longcot-mini":
        allowed = {"easy"}
    elif normalized == "longcot":
        allowed = {"medium", "hard"}
    else:
        allowed = {normalized}
    return [question for question in questions if str(question.difficulty).strip().lower() in allowed]


def _looks_like_transport_repr(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return "tool_result_origin=" in stripped or stripped.startswith("content='' additional_kwargs={}")


def extract_response_text(thread: dict[str, Any], sly_data: dict[str, Any]) -> str:
    result = sly_data.get("tolstoy_result") or {}
    canonical = str(result.get("answer") or "").strip()
    if canonical:
        return canonical

    transcript = str(thread.get("last_chat_response") or "").strip()
    if _looks_like_transport_repr(transcript):
        return ""
    return transcript


def verify_answer(question, answer: str, disable_fallback: bool) -> bool:
    if not answer:
        return False
    try:
        return verify(question, answer, options=_verification_options(disable_fallback))
    except Exception:
        return False


def format_exception(exc: Exception, timeout_ms: float) -> str:
    if isinstance(exc, TimeoutReachedException):
        return (
            f"{exc.__class__.__name__}: question exceeded the configured timeout of "
            f"{timeout_ms / 1000.0:.1f}s"
        )
    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def print_progress(question_id: str, is_correct: bool, iterations: int | None, error: str | None = None) -> None:
    global _COMPLETED
    with _LOCK:
        _COMPLETED += 1
        if error:
            status = "ERROR    "
        else:
            status = "CORRECT  " if is_correct else "INCORRECT"
        iter_text = f"  iters={iterations:3d}" if iterations is not None else ""
        detail = f"  {error}" if error else ""
        print(f"  [{_COMPLETED:3d}/{_TOTAL}]  {status}{iter_text}  {question_id}{detail}")


def run_one(question, args, run_id: str) -> dict[str, Any]:
    qid = str(question.question_id)
    suffix = f"_{args.tag}" if args.tag else ""
    suffix += f"_{run_id}" if run_id else ""
    frames_path = str(FRAMES_DIR / f"{qid}_frames{suffix}.json")
    result_path = str(RESULTS_DIR / f"{qid}_result{suffix}.json")
    thinking_file = f"/tmp/neuro_san_tolstoy_longcot_{qid}{suffix}.txt"

    session = create_session(args.session_type, args.agent_name, args.host, args.port, args.timeout_ms)
    thread = make_thread(args.timeout_ms)
    request = build_request(question.prompt, args, frames_path, result_path)
    start = time.time()

    response = ""
    formatted_response = ""
    sly_data: dict[str, Any] = {}
    result: dict[str, Any] = {}
    error: str | None = None

    try:
        thread, sly_data = run_request_with_progress(
            session,
            thread,
            request,
            args.heartbeat_s,
            qid,
            thinking_file,
        )
        response = extract_response_text(thread, sly_data)
        formatted_response = format_benchmark_response(question, response)
        result = sly_data.get("tolstoy_result") or {}
        if not response:
            error = "empty response from agent network"
    except Exception as exc:
        error = format_exception(exc, args.timeout_ms)
    finally:
        session.close()

    elapsed = time.time() - start
    is_correct = verify_answer(question, formatted_response, args.disable_verifier_fallback) if not error else False
    print_progress(qid, is_correct, result.get("iterations"), error)

    return {
        "question_id": question.question_id,
        "domain": question.domain,
        "difficulty": question.difficulty,
        "template": (question.problem or {}).get("template"),
        "gold_answer": question.answer,
        "raw_prediction": response,
        "predicted_answer": formatted_response,
        "correct": is_correct,
        "error": error,
        "elapsed_seconds": elapsed,
        "prompt_chars": len(question.prompt or ""),
        "iterations": result.get("iterations"),
        "nodes": result.get("nodes"),
        "final_node_id": result.get("final_node_id"),
        "frames_path": frames_path,
        "result_path": result_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Neuro-SAN Tolstoy agent on the LongCoT benchmark.")
    parser.add_argument("--domain", help="Optional LongCoT domain filter.")
    parser.add_argument("--difficulty", help="Optional LongCoT difficulty filter. Accepts easy|medium|hard plus longcot-mini and longcot aliases.")
    parser.add_argument("--n", type=int, help="Run only the first N filtered questions.")
    parser.add_argument("--index", type=int, help="Run one question by 0-based index in the filtered set.")
    parser.add_argument("--question-id", help="Run one question by LongCoT question id.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent question workers.")
    parser.add_argument("--shortest-first", action="store_true", help="Sort the filtered question set by prompt length before slicing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing JSONL log.")
    parser.add_argument("--tag", default="", help="Optional tag appended to result files.")

    parser.add_argument("--agent-name", default=os.environ.get("TOLSTOY_AGENT", "tolstoy_reasoner"))
    parser.add_argument(
        "--session-type",
        default=DEFAULT_SESSION_TYPE,
        choices=["direct", "grpc", "service", "http", "https"],
        help="Neuro-SAN session type.",
    )
    parser.add_argument("--host", default=os.environ.get("NS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NS_PORT", "30011")))
    parser.add_argument("--timeout-ms", type=float, default=180000.0)
    parser.add_argument("--heartbeat-s", type=float, default=15.0, help="Per-question progress heartbeat in seconds; set <= 0 to disable.")
    parser.add_argument("--verbose", action="store_true", help="Enable Tolstoy debug logging for direct runs.")
    parser.add_argument(
        "--disable-verifier-fallback",
        action="store_true",
        help="Disable LongCoT's default math/chem fallback verification. Leave this off for leaderboard-comparable runs.",
    )

    parser.add_argument("--max-iter", type=int, default=24)
    parser.add_argument("--max-active-nodes", type=int, default=10)
    parser.add_argument("--k-answer", type=int, default=3)
    parser.add_argument("--k-validator", type=int, default=3)
    parser.add_argument("--k-gc", type=int, default=3)
    parser.add_argument("--max-proposal-retries", type=int, default=5)
    parser.add_argument("--use-gc", action="store_true")
    parser.add_argument("--use-reasons", action="store_true")
    parser.add_argument("--show-nc-answers", action="store_true")
    parser.add_argument("--use-scratchpad", action="store_true")
    parser.add_argument("--cite-problem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--answer-temperature", type=float, default=0.7)

    args = parser.parse_args()
    args.agent_name = normalize_agent_name(args.session_type, args.agent_name)
    if args.verbose:
        os.environ["NS_TOLSTOY_DEBUG"] = "1"

    questions = load_questions()
    if args.domain:
        questions = [question for question in questions if question.domain == args.domain]
    questions = filter_questions_by_difficulty(questions, args.difficulty)
    if args.question_id:
        questions = [question for question in questions if str(question.question_id) == str(args.question_id)]
    if args.shortest_first:
        questions = sorted(questions, key=lambda question: len(question.prompt or ""))
    if args.index is not None:
        questions = [questions[args.index]]
    if args.n is not None:
        questions = questions[: args.n]

    if args.session_type == "direct":
        initial_session = create_session(args.session_type, args.agent_name, args.host, args.port, args.timeout_ms)
        try:
            _assert_direct_session_initializable_sync(initial_session)
        finally:
            initial_session.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"longcot_tolstoy_{timestamp}.jsonl"
    if log_path.exists() and not args.overwrite:
        raise FileExistsError(f"{log_path} already exists; use --overwrite to replace it.")

    global _COMPLETED, _TOTAL
    _COMPLETED = 0
    _TOTAL = len(questions)

    entries: list[dict[str, Any]] = []
    run_id = timestamp

    with open(log_path, "w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
            futures = {executor.submit(run_one, question, args, run_id): question for question in questions}
            for future in concurrent.futures.as_completed(futures):
                question = futures[future]
                try:
                    entry = future.result()
                    entries.append(entry)
                    with _LOCK:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        handle.flush()
                except Exception as exc:  # pragma: no cover - integration runner
                    print(f"  ERROR on {question.question_id}: {format_exception(exc, args.timeout_ms)}")

    correct = sum(1 for entry in entries if entry["correct"])
    total = len(entries)
    accuracy = (correct / total) if total else 0.0
    print()
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}" if total else "No results.")
    print(f"Results logged to {log_path}")


if __name__ == "__main__":
    main()
