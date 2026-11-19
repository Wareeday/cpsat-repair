# Formalize, then repair

Can a language model write a constraint model for a problem, and can a checker's counterexample help it fix that model when it is wrong?

The LLM reads a plain-English problem statement and writes an [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver) model. The solver solves it. A separate checker, which knows nothing about the model and does not import OR-Tools, verifies the answer. If the answer is wrong, the checker's counterexample (for example "operation 2 of job 4 starts at 7 but operation 1 of the same job only finishes at 9") goes back to the LLM, which gets at most three attempts to repair its model. I compare this against a baseline where the LLM writes its own search code in plain Python.

I built this as a small, fully free experiment (Python, OR-Tools, any local or free-tier model) around one question I care about: **when an LLM translates a problem into a solver, where does the translation break, and does feedback from a verifier fix it?**

Everything here is open source and free to run: Python, OR-Tools (Apache 2.0), pandas, matplotlib, pytest. There is no external dataset to download or license: the benchmark instances are produced by the generators in this repository. The language model is whatever you point the script at. For a fully open pipeline use a local open-weight model through Ollama; hosted free tiers also serve open-weight models (see the free-tier section), although the hosting itself is a commercial service.

## What is compared

| Condition | What the model does | Feedback |
|---|---|---|
| `python` | writes plain-Python search code (standard library only) | none |
| `cpsat` | writes a CP-SAT model | none |
| `cpsat_repair` | writes a CP-SAT model | up to 3 rounds of checker counterexamples |

`cpsat` and `cpsat_repair` start from the same round-0 program, so the repair effect is a paired comparison and not a difference between two lucky or unlucky samples. Every condition gets the same problem text and the same time limit: the prompts ask for an answer within about 20 s and a run is killed after 30 s.

The model never sees the test instance. It gets the problem statement, the input and output format, and one tiny example, and has to write a general `solve(instance)` function. That function is then run on hidden instances of five sizes.

## Problems

Eight families, each with a generator, a trusted reference model, and an independent checker. Each family has five size levels; the default dataset has 2 instances per size, so 80 instances.

| Family | Type | Sizes (smallest to largest) |
|---|---|---|
| `nqueens_blocked` | satisfaction | n = 8, 12, 16, 24, 32, with 2n blocked squares |
| `latin_completion` | satisfaction | n = 5, 8, 10, 14, 20, 42% of cells given |
| `graph_coloring` | minimise colours | 10 to 60 vertices, edge probability 0.35 down to 0.25 |
| `knapsack_conflicts` | maximise value | 10, 20, 40, 100, 200 items, with conflicting pairs |
| `bin_packing` | minimise bins | 8, 12, 20, 30, 45 items |
| `tsp` | minimise tour length | 6, 8, 11, 15, 20 cities |
| `job_shop` | minimise makespan | 3x3, 4x4, 5x5, 7x6, 10x8 (jobs x machines) |
| `shift_scheduling` | minimise violated day-off requests | 4 employees x 7 days up to 20 x 21 |

For the optimisation families an answer is only **correct** if it is valid *and* optimal. Valid-but-suboptimal is reported separately.

## How an answer is judged

1. The generated code runs in a child process with a wall-clock limit (and a memory cap on Linux/macOS).
2. `Family.evaluate()` checks the returned dict in plain Python: format, every constraint, and it recomputes the objective itself. It does not trust any number the model reports.
3. For optimisation problems the recomputed objective is compared with the proven optimum stored with the instance.

Every run ends in one of: `ok`, `suboptimal`, `invalid` (a constraint is violated), `format_error`, `runtime_error`, `timeout`, `no_code`. These are the failure categories used in the tables below.

The ground-truth optimum comes from a hand-written CP-SAT model per family (`reference()`), solved to proven optimality. The unit tests cross-check these optima against brute force on tiny TSP, knapsack, bin-packing and colouring instances, and check that the reference solutions pass the checkers and that deliberately broken solutions are rejected.

For satisfaction problems the generator plants a solution first, so every instance is guaranteed to be solvable.

One more check guards against a cheap way to "pass" repair. The counterexample contains the failing instance, so a model could patch that single instance and nothing else. Whenever a repaired program passes, it is run again on a sibling instance (same family and size level, different seed) that the model never saw, and the result is stored as `heldout` and reported.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # 2 to 3 minutes
```

The dataset is already in `data/instances.jsonl`. To regenerate it: `python -m cpsr.generate --per-size 2`.

**With a local model (Ollama):**

```bash
ollama pull qwen2.5-coder:7b
python -m cpsr.run --backend ollama --model qwen2.5-coder:7b --out results/qwen7b
```

**With a free-tier API.** Any provider that speaks the OpenAI chat format works. Check the key, URL and model name first, then run:

```bash
export OPENAI_API_KEY=...          # Windows PowerShell: $env:OPENAI_API_KEY="..."
python -m cpsr.ping --backend openai --base-url https://api.groq.com/openai/v1 --list-models
python -m cpsr.ping --backend openai --base-url https://api.groq.com/openai/v1 --model openai/gpt-oss-120b
python -m cpsr.run  --backend openai --base-url https://api.groq.com/openai/v1 \
    --model openai/gpt-oss-120b --sleep 25 --out results/gptoss120b
```

**Then the report:**

```bash
python -m cpsr.report --results results/gptoss120b --update-readme README.md
```

This writes the tables and four figures into the run folder and fills in the Results section below.

Options worth knowing: `--samples` (independent programs per family and condition, default 3), `--max-size 3` (only the three smallest size levels), `--families tsp,job_shop`, `--temperature`, `--repair-rounds`, `--timeout`, `--sleep` (seconds between model calls), `--max-tokens` (cap on reply length, default 4096; reasoning tokens count against it), `--no-seed` (if a provider rejects the seed field) and `--extra-body '{"reasoning_effort": "low"}'` (extra JSON fields for every request, if the provider supports them).

### Free tiers

Limits below are what I found in November 2026 in provider documentation as quoted by third-party trackers. They change often and each provider's own console shows the numbers that apply to your account, so check there before a long run.

| Provider | Free limits | Open-weight models | Notes |
|---|---|---|---|
| Groq | `openai/gpt-oss-120b`: 30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day | gpt-oss-120b, Qwen, others | Very fast. The daily token cap is the bottleneck. No card. |
| Mistral (free mode) | no card; exact limits only on the console's Limits page (older docs said about 1 request/s and 1 billion tokens/month) | Mistral Small and Devstral Small are open weights; which models the free mode may call is not published | Prompts may be used for training, which does not matter here because the instances are synthetic. Phone verification may be needed. |
| OpenRouter, `:free` models | 20 requests/min and 50 requests/day (1,000/day after a one-time $10 purchase) | `qwen/qwen3-coder:free`, `openai/gpt-oss-120b:free` | The free roster changes monthly and the daily cap is shared by all free models. |
| Google Gemini API | free tier with limits that have changed several times | Gemma is open; Gemini is not | Fine for a closed-model comparison, not for an open-source claim. |
| Cerebras | needs a card; $5 credit that expires after 30 days | gpt-oss-120b, Qwen | Not free in practice. |
| Ollama (local) | none | any model your machine can run | Slower and weaker with a 7B-8B model, but unlimited and fully open. |

A rough budget, so there are no surprises. Each first program costs about 500 tokens in and several hundred out (more if the model reasons first). A repair call also carries the earlier conversation, so it costs more, roughly 2,500 to 5,000 tokens in total. With the default of 3 samples a run needs 48 first-program calls plus 1 to 3 repair calls for every (instance, sample) pair that fails, so something like 0.5 to 1.5 million tokens depending on how often the model fails. On Groq's 200,000 tokens/day that is several days, not hours. The run is built for that: finished work is saved and cached, and when the daily limit is hit the script stops with a message and continues from the same place when you run the same command the next day. To finish sooner, use `--samples 2`, or `--max-size 4`.

### Testing the pipeline without a model

`--backend mock` replaces the model with scripted programs (some deliberately wrong at round 0 so the repair loop gets exercised). It exists to test the code. Numbers from it say nothing about any language model.

```bash
python -m cpsr.run --backend mock --mock-mode flaky --max-size 2 --out results/mock_test
```

The tests also run the real command line against a small fake server that answers in the OpenAI format, with model-style CP-SAT and brute-force programs. No real provider is contacted by the tests.

## Layout

```
cpsr/
  families/        one file per problem family: generator, reference model, checker
  prompts.py       prompts, code extraction, counterexample messages
  llm.py           Ollama / OpenAI-compatible / mock backends and the reply cache
  ping.py          checks a key, URL and model name before a long run
  sandbox.py       runs generated code in a child process with a time limit
  pipeline.py      the three conditions and the repair loop
  generate.py      builds data/instances.jsonl
  run.py           command line for experiments
  report.py        tables, figures, README update
data/instances.jsonl
tests/
results/           one folder per run (runs.jsonl, programs/, figures, summary.md)
```

Each run folder keeps every program the model wrote (`programs/<family>/`), including each repaired version, so any result can be inspected by hand. `report.py` also writes `failures_for_annotation.csv`, one row per distinct failure of the one-shot CP-SAT programs with an empty `my_category` column. It is meant for labelling failures by hand (missing constraint, wrong objective, off-by-one indexing, solver API misuse, wrong output format, ...).

## Results

<!-- RESULTS:START -->
*No run has been recorded yet.* Run the commands above, then `python -m cpsr.report --results results/<run> --update-readme README.md` replaces this block with the tables and figures.
<!-- RESULTS:END -->

## What I found

*To be written after the first real run. Things worth checking: whether repair helps more on some families than others; which failure categories repair fixes and which it does not; whether the gap between `python` and `cpsat` grows with instance size; whether the model fixes the instance it was shown or the underlying mistake (see the first limitation below).*

## Limitations

- **Repair is judged on the instance that produced the counterexample, plus one sibling.** A repaired program is re-run on one sibling instance of the same family and size, and that rate is reported, but one sibling is a weak test of generality. The counterexample also shows the model the failing instance when it is short, which makes patching that single instance easier.
- **The baseline gets no feedback.** `python` is one-shot. A fairer and harder baseline would be plain Python with the same repair loop. I left it out to keep the first version small.
- **Small sample.** 2 instances per size and 3 programs per family and condition. The confidence intervals are Wilson intervals that treat runs as independent, but runs of one program on different instances are correlated, so the intervals are too narrow.
- **Instance selection.** If the reference CP-SAT model could not prove optimality in 30 s, the instance was discarded and re-drawn (this happened once in the shipped dataset). That biases the set slightly towards instances CP-SAT finds easy.
- **Shared solver.** The ground truth and the model's answer both rely on CP-SAT. The checkers are independent of it, and the brute-force tests help, but a solver bug would not be caught.
- **Counterexamples for suboptimal answers** only say that a better solution exists. They do not reveal the optimal value unless you pass `--reveal-optimum`.
- **The sandbox is a resource guard, not a security boundary.** Generated code runs with your user's permissions. Use a container if you do not trust the model.
- **Free-tier limits move.** The numbers in the free-tier table are a snapshot. Reasoning models also spend hidden tokens that count towards the limits and towards `--max-tokens`, so a reply can be cut off before the code; such replies are recorded as `no_code`.
- Results depend on the model, temperature, and prompt wording. I report one model per run folder and do not claim anything beyond it.

## License

MIT
