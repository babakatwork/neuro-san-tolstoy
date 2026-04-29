import asyncio

from coded_tools.tolstoy.engine import TolstoyEngine, TolstoyRunConfig
from coded_tools.tools.agent_caller import AgentCaller


class SequenceCaller(AgentCaller):
    def __init__(self, name: str, responses: list[str]):
        self._name = name
        self._responses = list(responses)

    def get_name(self) -> str:
        return self._name

    async def call_agent(self, tool_args, sly_data=None) -> str:
        assert self._responses, f"No responses left for {self._name}"
        return self._responses.pop(0)


def test_engine_can_solve_simple_problem():
    callers = {
        "proposer": SequenceCaller(
            "proposer",
            [
                """PARENT FACTS: 0
PROPOSED QUESTION: From the original problem statement "What is 2 + 2?", compute the sum and return only the integer.
RETIRE: NONE
REASON: Ask for the single arithmetic result.""",
                "FINAL ANSWER NODE: 1",
            ],
        ),
        "validator": SequenceCaller(
            "validator",
            [
                "ITEM 1: PASS - self-contained\nITEM 2: PASS - cited\nITEM 3: PASS - unambiguous\nVERDICT: ACCEPT",
                "ITEM 1: PASS - self-contained\nITEM 2: PASS - cited\nITEM 3: PASS - unambiguous\nVERDICT: ACCEPT",
                "ITEM 1: PASS - self-contained\nITEM 2: PASS - cited\nITEM 3: PASS - unambiguous\nVERDICT: ACCEPT",
            ],
        ),
        "answerer": SequenceCaller(
            "answerer",
            [
                "FINAL ANSWER: 4",
                "FINAL ANSWER: 4",
                "FINAL ANSWER: 4",
            ],
        ),
    }

    engine = TolstoyEngine(
        callers=callers,
        config=TolstoyRunConfig(
            max_iter=3,
            cite_problem=True,
        ),
    )

    result = asyncio.run(engine.run("What is 2 + 2?"))

    assert result["answer"] == "4"
    assert result["final_node_id"] == 1
    assert result["nodes"] == 2
    assert any(frame.get("final_node_id") == 1 for frame in result["frames"])
