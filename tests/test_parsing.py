from coded_tools.tolstoy.parsing import (
    extract_fact_ids,
    normalize_answer,
    parse_answer,
    parse_consolidation_plan,
    parse_proposal,
)


def test_parse_proposal_with_multiline_question():
    raw = """PARENT FACTS: 0, 3
PROPOSED QUESTION: Using Fact 3 and the original problem,
compute the final value and return it as an integer.
RETIRE: 3
REASON: We can now collapse the intermediate step."""

    proposal = parse_proposal(raw, cite_problem=True)

    assert proposal.error is None
    assert proposal.parent_ids == [0, 3]
    assert proposal.retire_ids == [3]
    assert proposal.reason == "We can now collapse the intermediate step."
    assert "compute the final value" in proposal.question


def test_parse_answer_variants():
    final = parse_answer("Some reasoning\nFINAL ANSWER: 42")
    contradiction = parse_answer("CONTRADICTION: Fact 3 says the opposite.")

    assert final.kind == "final"
    assert final.value == "42"
    assert contradiction.kind == "contradiction"
    assert "Fact 3" in contradiction.value


def test_parse_consolidation_plan():
    plan = parse_consolidation_plan(
        "CONSOLIDATE: 4, 6, 8\nMERGE QUESTION: Rewrite Facts 4, 6, and 8 as one summary table."
    )

    assert plan.error is None
    assert plan.node_ids == [4, 6, 8]
    assert plan.merge_question == "Rewrite Facts 4, 6, and 8 as one summary table."


def test_helpers_normalize_and_extract():
    assert extract_fact_ids("Fact 3 depends on fact 12.") == [3, 12]
    assert normalize_answer("  12 \n  34 ") == "12 34"
