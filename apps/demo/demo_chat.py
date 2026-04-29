from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import time
from contextlib import suppress
from pathlib import Path

from neuro_san.client.agent_session_factory import AgentSessionFactory
from neuro_san.client.streaming_input_processor import StreamingInputProcessor
from neuro_san.internals.chat.data_driven_chat_session import DataDrivenChatSession
from leaf_common.time.timeout_reached_exception import TimeoutReachedException

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AGENT_MANIFEST_FILE", str(REPO_ROOT / "registries" / "manifest.hocon"))
os.environ.setdefault("AGENT_TOOL_PATH", str(REPO_ROOT / "coded_tools"))

DEFAULT_AGENT = os.environ.get("TOLSTOY_AGENT", "tolstoy/tolstoy_reasoner")
DEFAULT_SESSION_TYPE = os.environ.get("NS_SESSION_TYPE", "direct")
DEFAULT_NS_HOST = os.environ.get("NS_HOST", "localhost")
DEFAULT_NS_PORT = int(os.environ.get("NS_PORT", "30011"))

HELP_TEXT = """
Commands:
  help
  quit | exit

You can send either:
  1. a plain problem statement
  2. a JSON object containing solver arguments

Example plain problem:
  What is 46048 x 42098?

Example JSON:
  {"problem": "What is 46048 x 42098?", "max_iter": 24, "k_answer": 3}
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat client for the Neuro-SAN Tolstoy agent.")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="Runtime agent name.")
    parser.add_argument("--host", default=DEFAULT_NS_HOST, help="Neuro-SAN server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_NS_PORT, help="Neuro-SAN server port.")
    parser.add_argument("--timeout-ms", type=float, default=120000.0, help="Per-turn timeout in milliseconds.")
    parser.add_argument("--heartbeat-s", type=float, default=5.0, help="Progress print interval while waiting.")
    parser.add_argument("--verbose", action="store_true", help="Enable local debug logging for direct Tolstoy runs.")
    parser.add_argument(
        "--session-type",
        default=DEFAULT_SESSION_TYPE,
        choices=["direct", "grpc", "service", "http", "https"],
        help="Neuro-SAN session type.",
    )
    return parser.parse_args()


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
        {"user_id": os.environ.get("USER") or "demo"},
        timeout_ms / 1000.0,
    )


def assert_server_reachable(host: str, port: int):
    if os.environ.get("NS_SKIP_PREFLIGHT", "").strip().lower() in {"1", "true", "yes"}:
        return

    last_error = None
    attempted = []

    candidate_hosts = [host]
    if host in {"localhost", "127.0.0.1", "::1"}:
        for extra in ["127.0.0.1", "::1", "localhost"]:
            if extra not in candidate_hosts:
                candidate_hosts.append(extra)

    for candidate_host in candidate_hosts:
        try:
            with socket.create_connection((candidate_host, port), timeout=2.0):
                return
        except OSError as exc:
            attempted.append(f"{candidate_host}:{port} -> {exc}")
            last_error = exc
    raise RuntimeError(
        f"Cannot reach Neuro-SAN server at {host}:{port}. "
        f"Attempts: {'; '.join(attempted)}. Last socket error: {last_error}"
    )


def make_thread(timeout_ms: float):
    return {
        "last_chat_response": None,
        "prompt": "",
        "timeout": timeout_ms,
        "num_input": 0,
        "user_input": None,
        "sly_data": {},
        "chat_filter": {"chat_filter_type": "MAXIMAL"},
    }


def run_turn(session, thread, user_input: str):
    processor = StreamingInputProcessor("DEFAULT", "/tmp/neuro_san_tolstoy_demo.txt", session, None)
    thread["user_input"] = user_input
    return processor.process_once(thread)


def _looks_like_transport_repr(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return "tool_result_origin=" in stripped or stripped.startswith("content='' additional_kwargs={}")


def extract_response_text(thread) -> str:
    sly_data = thread.get("sly_data") or {}
    result = sly_data.get("tolstoy_result") or {}
    canonical = str(result.get("answer") or "").strip()
    if canonical:
        return canonical

    transcript = str(thread.get("last_chat_response") or "").strip()
    if _looks_like_transport_repr(transcript):
        return ""
    return transcript


async def assert_direct_session_initializable(session) -> None:
    invocation_context = session.invocation_context.safe_shallow_copy()
    chat_session = DataDrivenChatSession(agent_network=session.agent_network)
    try:
        await chat_session.set_up(invocation_context, {})
    finally:
        with suppress(Exception):
            await chat_session.delete_resources()
        with suppress(Exception):
            invocation_context.close()


async def run_turn_with_progress(session, thread, user_input: str, heartbeat_s: float):
    worker = asyncio.create_task(asyncio.to_thread(run_turn, session, thread, user_input))
    start = time.time()
    while True:
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=heartbeat_s)
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"[still working: {elapsed:.1f}s]", flush=True)


def format_exception(exc: Exception, timeout_ms: float) -> str:
    if isinstance(exc, TimeoutReachedException):
        return (
            f"{exc.__class__.__name__}: turn exceeded the configured timeout of "
            f"{timeout_ms / 1000.0:.1f}s"
        )
    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


async def main():
    args = parse_args()
    args.agent = normalize_agent_name(args.session_type, args.agent)
    if args.verbose:
        os.environ["NS_TOLSTOY_DEBUG"] = "1"

    print(f"Top agent: {args.agent}")
    print(f"Session type: {args.session_type}")
    print(f"Neuro-SAN: {args.host}:{args.port}")
    print(f"Timeout: {args.timeout_ms} ms")
    print()
    print(HELP_TEXT)
    print()

    if args.session_type != "direct":
        assert_server_reachable(args.host, args.port)
    thread = make_thread(args.timeout_ms)
    initial_session = create_session(args.session_type, args.agent, args.host, args.port, args.timeout_ms)
    try:
        if args.session_type == "direct":
            await assert_direct_session_initializable(initial_session)
    finally:
        initial_session.close()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            break
        if raw.lower() == "help":
            print(HELP_TEXT)
            continue

        try:
            print("[sending request]")
            start = time.time()
            session = create_session(args.session_type, args.agent, args.host, args.port, args.timeout_ms)
            try:
                thread = await run_turn_with_progress(session, thread, raw, args.heartbeat_s)
            finally:
                session.close()
            elapsed = time.time() - start
            print(f"[completed in {elapsed:.2f}s]")
            print(extract_response_text(thread))
        except Exception as exc:  # pragma: no cover - interactive convenience
            print(f"ERROR: {format_exception(exc, args.timeout_ms)}")


if __name__ == "__main__":
    asyncio.run(main())
