from types import SimpleNamespace

from apps.benchmarking.run_longcot import extract_response_text, format_benchmark_response


def _question(prompt: str):
    return SimpleNamespace(prompt=prompt)


def test_format_benchmark_response_wraps_scalar_answers():
    question = _question("Return your answer in the format: solution = <integer>")

    assert format_benchmark_response(question, "1938500704") == "solution = 1938500704"


def test_format_benchmark_response_extracts_list_payloads():
    question = _question("Return your answer in the format: solution = [move1, move2, move3]")

    assert (
        format_benchmark_response(question, "PIECES=[BK@f4,BP@e2,BR@g2]")
        == "solution = [BK@f4,BP@e2,BR@g2]"
    )


def test_format_benchmark_response_preserves_existing_solution_marker():
    question = _question("Return your answer in the format: solution = [a, b, c]")

    assert (
        format_benchmark_response(question, "Reasoning\nsolution = [1, 2, 3]")
        == "solution = [1, 2, 3]"
    )


def test_extract_response_text_prefers_structured_answer_and_drops_transport_repr():
    thread = {
        "last_chat_response": "content='' additional_kwargs={} tool_result_origin=[{'tool': 'tolstoy_reasoner'}]",
        "sly_data": {"tolstoy_result": {"answer": ""}},
    }

    assert extract_response_text(thread, thread["sly_data"]) == ""

    thread["sly_data"]["tolstoy_result"]["answer"] = "solution = 4"
    assert extract_response_text(thread, thread["sly_data"]) == "solution = 4"
