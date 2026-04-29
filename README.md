# Neuro-SAN Tolstoy

`neuro-san-tolstoy` is a Neuro-SAN implementation of the Tolstoy DAG-style
reasoning pattern. It follows the same overall shape as the existing
MAKER-style `mdap_decomposer` example:

- a top-level Neuro-SAN agent network
- a coded tool that owns the orchestration state
- supporting specialist agents for proposing, validating, answering, merging,
  and pruning DAG facts

The coded tool grows a graph of verified facts until one node can be declared
the final answer to the original problem.

## Repo layout

```text
apps/
  benchmarking/
    run_longcot.py
  demo/
    demo_chat.py

coded_tools/
  tolstoy/
    engine.py
    parsing.py
    solver_tool.py
    types.py
  tools/
    agent_caller.py
    coded_tool_agent_caller.py

registries/
  llm_config.hocon
  manifest.hocon
  tolstoy/
    manifest.hocon
    tolstoy_reasoner.hocon

tests/
  test_engine.py
  test_parsing.py
```

## Installation

Run everything from the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="your-api-key"
export AGENT_MANIFEST_FILE="$(pwd)/registries/manifest.hocon"
export AGENT_TOOL_PATH="$(pwd)/coded_tools"
```

`longcot` is installed from GitHub rather than PyPI. That is intentional: the
benchmark package exists in `LongHorizonReasoning/longcot`, but it is not
currently published on PyPI.

## Model configuration

The default model config lives in
[registries/llm_config.hocon](registries/llm_config.hocon). By default this
repo is configured for OpenAI with `gpt-5.2`.

## Running the demo

The interactive demo defaults to `direct` mode, so it does not require a
separate Neuro-SAN server:

```bash
python apps/demo/demo_chat.py
```

For debug output from the proposer / validator / answerer / coded tool path:

```bash
python apps/demo/demo_chat.py --verbose
```

You can send either a plain problem:

```text
What is 46048 x 42098?
```

or structured JSON:

```json
{
  "problem": "What is 46048 x 42098?",
  "max_iter": 24,
  "k_answer": 3,
  "use_scratchpad": true
}
```

Useful options:

```bash
python apps/demo/demo_chat.py \
  --verbose \
  --timeout-ms 240000 \
  --heartbeat-s 5
```

Notes:

- In direct mode, the demo creates a fresh Neuro-SAN session per turn so the
  timeout budget is per request rather than shared across the whole REPL.
- If a run exceeds the configured timeout, the demo now reports a concrete
  timeout error instead of a blank `ERROR:`.
- Higher fan-out settings such as larger `k_answer` values often need a larger
  `--timeout-ms`.

## Running with a Neuro-SAN HTTP server

If you want to talk to a separately running Neuro-SAN service, start it first:

```bash
python -m neuro_san.service.main_loop.server_main_loop --http_port 30021
```

Then point the demo at it:

```bash
python apps/demo/demo_chat.py --session-type http --port 30021
```

The demo normalizes the agent naming differences between direct and service
sessions for you.

## LongCoT benchmark harness

This repo includes a benchmark runner that drives the Tolstoy agent and grades
its returned answer with the `longcot` package.

By default, the runner uses LongCoT's normal verifier settings. That makes the
default path the right one for leaderboard-style evaluation. The optional
`--disable-verifier-fallback` flag is only for ablations; it turns off the
math/chem fallback judges and will generally make those domains score worse.

Good smoke-test pattern for a genuinely small sample:

```bash
python apps/benchmarking/run_longcot.py \
  --session-type direct \
  --domain chess \
  --shortest-first \
  --n 1 \
  --workers 1 \
  --max-iter 8 \
  --k-answer 1 \
  --k-validator 1 \
  --timeout-ms 120000
```

HTTP mode example:

```bash
python apps/benchmarking/run_longcot.py \
  --session-type http \
  --port 30021 \
  --domain math \
  --n 5 \
  --workers 2 \
  --max-iter 24 \
  --k-answer 3
```

Direct mode example:

```bash
python apps/benchmarking/run_longcot.py \
  --session-type direct \
  --domain math \
  --n 5 \
  --workers 1 \
  --max-iter 24 \
  --k-answer 3
```

Higher-`k` example:

```bash
python apps/benchmarking/run_longcot.py \
  --session-type direct \
  --domain chess \
  --shortest-first \
  --n 1 \
  --workers 1 \
  --max-iter 8 \
  --k-answer 5 \
  --k-validator 1 \
  --timeout-ms 120000 \
  --tag smoke_shortest_k5
```

Useful benchmark flags:

- `--shortest-first`: sort the filtered question set by prompt length before
  applying `--n` or `--index`; this is the easiest way to get a true smoke
  test instead of accidentally picking a very large prompt.
- `--heartbeat-s`: print per-question progress while a sample is still running.
- `--verbose`: enable Tolstoy debug logs during the benchmark run.
- `--timeout-ms`: per-question timeout budget.
- `--disable-verifier-fallback`: disable LongCoT's default fallback verification
  for math and chemistry; keep this off unless you are doing a controlled
  ablation.
- `--k-answer`, `--k-validator`, `--k-gc`: fan-out controls for the Tolstoy
  agent.

Notes:

- In direct mode, the benchmark runner does a direct-session initialization
  check before launching the worker pool, so schema/setup issues fail early.
- Each benchmark entry now records `raw_prediction`, normalized
  `predicted_answer`, `error`, `elapsed_seconds`, and `prompt_chars` in
  addition to the Tolstoy trace metadata.
- If the DAG reaches its iteration limit without an explicit final-answer node,
  the engine now makes one last synthesis pass over the active answered facts
  before falling back to the latest answered node.
- Larger `k_answer` values work, but they often need more time. Increase
  `--timeout-ms` accordingly.

Results are written under `results/longcot/`, including:

- per-question JSONL benchmark logs
- per-question DAG frame captures
- per-question structured result payloads

## Recommended benchmark workflow

If your goal is a public LongCoT submission, a good sequence is:

1. Start with `easy` only (the `LongCoT-Mini` style pass), not the full set.
2. Compare one or two candidate `k_answer` values on a short slice, usually
   something like `k_answer=3` versus `k_answer=5`.
3. Keep `k_validator` small during tuning; `1` is a good cheap baseline.
4. Once one configuration is clearly better, run the full `easy` split.
5. Only then spend the money on the full `medium` + `hard` evaluation.

Example pilot:

```bash
python apps/benchmarking/run_longcot.py \
  --session-type direct \
  --domain chess \
  --shortest-first \
  --n 10 \
  --workers 1 \
  --max-iter 8 \
  --k-answer 3 \
  --k-validator 1 \
  --timeout-ms 180000 \
  --tag pilot_chess_k3
```

Then rerun the same slice with `--k-answer 5` and compare:

- accuracy
- per-question error rate
- elapsed time
- token/cost usage from your model provider

## Verification

```bash
python -m pytest tests
```

## Current notes

- The main agent network is
  [registries/tolstoy/tolstoy_reasoner.hocon](registries/tolstoy/tolstoy_reasoner.hocon).
- The coded-tool entry point is
  [coded_tools/tolstoy/solver_tool.py](coded_tools/tolstoy/solver_tool.py).
- The DAG orchestration logic lives in
  [coded_tools/tolstoy/engine.py](coded_tools/tolstoy/engine.py).
- The implementation is functional, but prompt quality is still being tuned.
  Arithmetic-style tasks can still show `NO CONSENSUS`, contradictions, or
  incorrect final answers under some settings.
