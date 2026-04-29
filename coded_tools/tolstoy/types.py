from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeStatus(str, Enum):
    ANSWERED = "answered"
    NO_CONSENSUS = "no_consensus"
    REJECTED = "rejected"
    CONTRADICTION = "contradiction"


@dataclass
class Node:
    id: int
    question: str
    answer: str
    parent_ids: list[int]
    status: NodeStatus
    reason: str | None = None
    raw_answers: list[str] = field(default_factory=list)


@dataclass
class DagState:
    problem: str
    nodes: list[Node] = field(default_factory=list)
    active_ids: set[int] = field(default_factory=set)
    gc_reasons: dict[int, str] = field(default_factory=dict)
    scratchpad: str = ""
    frames: list[dict[str, Any]] = field(default_factory=list)
    frames_path: str | None = None

    def get(self, node_id: int) -> Node | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    @property
    def active_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.id in self.active_ids]

    def active_answered(self, include_fact0: bool = False) -> list[Node]:
        nodes = [
            node
            for node in self.active_nodes
            if node.status == NodeStatus.ANSWERED and (include_fact0 or node.id != 0)
        ]
        return nodes

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.active_ids.add(node.id)

    def retire(self, node_ids: list[int], reasons: dict[int, str] | None = None) -> None:
        for node_id in node_ids:
            self.active_ids.discard(node_id)
            if reasons and node_id in reasons:
                self.gc_reasons[node_id] = reasons[node_id]

    def snapshot(self, description: str, final_node_id: int | None = None) -> None:
        frame = {
            "problem": self.problem,
            "description": description,
            "nodes": [
                {
                    "id": node.id,
                    "question": node.question,
                    "answer": node.answer,
                    "parent_ids": list(node.parent_ids),
                    "status": node.status.value,
                    "active": node.id in self.active_ids,
                    "gc_reason": self.gc_reasons.get(node.id),
                    "reason": node.reason,
                    "raw_answers": list(node.raw_answers),
                }
                for node in self.nodes
            ],
            "active_ids": sorted(self.active_ids),
            "scratchpad": self.scratchpad,
        }
        if final_node_id is not None:
            frame["final_node_id"] = final_node_id
        self.frames.append(frame)
        if self.frames_path:
            Path(self.frames_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.frames_path).write_text(json.dumps(self.frames, indent=2), encoding="utf-8")
