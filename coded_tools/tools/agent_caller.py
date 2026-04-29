from __future__ import annotations

from typing import Any


class AgentCaller:
    """Minimal interface for coded tools that call back into Neuro-SAN agents."""

    def get_name(self) -> str:
        raise NotImplementedError

    async def call_agent(self, tool_args: dict[str, Any], sly_data: dict[str, Any] | None = None) -> str:
        raise NotImplementedError
