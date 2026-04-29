from __future__ import annotations

import json
import logging
import os
from typing import Any

from neuro_san.internals.graph.activations.branch_activation import BranchActivation

from coded_tools.tools.agent_caller import AgentCaller

LOGGER = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return os.environ.get("NS_TOLSTOY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug(message: str) -> None:
    if _debug_enabled():
        print(f"[tolstoy.agent] {message}", flush=True)


class CodedToolAgentCaller(AgentCaller):
    """Calls an in-network agent from a coded tool via BranchActivation.use_tool()."""

    def __init__(self, branch_activation: BranchActivation, name: str):
        self._branch_activation = branch_activation
        self._name = name

    def get_name(self) -> str:
        return self._name

    async def call_agent(self, tool_args: dict[str, Any], sly_data: dict[str, Any] | None = None) -> str:
        use_sly_data = sly_data or {}
        LOGGER.debug("Calling agent %s with args=%s", self._name, json.dumps(tool_args, ensure_ascii=True))
        _debug(f"calling {self._name}")
        response = await self._branch_activation.use_tool(self._name, tool_args, sly_data=use_sly_data)
        _debug(f"finished {self._name}")
        return "" if response is None else str(response)
