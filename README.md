# Formalize, then repair

Can a language model write a constraint model for a problem, and can a checker's counterexample help it fix that model when it is wrong?

The LLM reads a plain-English problem statement and writes an [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver) model. The solver solves it. A separate checker, which knows nothing about the model and does not import OR-Tools, verifies the answer. If the answer is wrong, the checker's counterexample (for example "operation 2 of job 4 starts at 7 but operation 1 of the same job only finishes at 9") goes back to the LLM, which gets at most three attempts to repair its model. I compare this against a baseline where the LLM writes its own search code in plain Python.

I built this as a small, fully free experiment (Python, OR-Tools, any local or free-tier model) around one question I care about: **when an LLM translates a problem into a solver, where does the translation break, and does feedback from a verifier fix it?**

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

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # about a minute
```

The dataset is already in `data/instances.jsonl`. To regenerate it: `python -m cpsr.generate --per-size 2`.

**With a local model (Ollama):**

```bash
ollama pull qwen2.5-coder:7b
python -m cpsr.run --backend ollama --model qwen2.5-coder:7b --out results/qwen7b
```

**With a free-tier API** (anything that speaks the OpenAI chat format: Groq, OpenRouter, Gemini's compatibility endpoint, ...):

```bash
export OPENAI_API_KEY=...          # Windows PowerShell: $env:OPENAI_API_KEY="..."
python -m cpsr.run --backend openai --base-url https://api.groq.com/openai/v1 \
    --model llama-3.3-70b-versatile --sleep 2 --out results/llama70b
```

**Then the report:**

```bash
python -m cpsr.report --results results/qwen7b --update-readme README.md
```

This writes the tables and four figures into the run folder and fills in the Results section below.

Useful options: `--samples` (independent programs per family and condition, default 3), `--max-size 3` (only the three smallest size levels, good for a quick trial), `--families tsp,job_shop`, `--temperature`, `--repair-rounds`, `--timeout`. Runs are resumable: model replies are cached in `llm_cache.jsonl`, and finished runs are skipped if you start the same command again.

The number of model calls depends on how often the first attempt fails. With 3 samples it is 48 calls for the first programs, plus up to 3 per failing (instance, sample) pair: between about 50 and 770 calls in total.

### Testing the pipeline without a model

`--backend mock` replaces the model with scripted programs (some deliberately wrong at round 0 so the repair loop gets exercised). It exists to test the code. Numbers from it say nothing about any language model.

```bash
python -m cpsr.run --backend mock --mock-mode flaky --max-size 2 --out results/mock_test
```

## Layout

```
cpsr/
  families/        one file per problem family: generator, reference model, checker
  prompts.py       prompts, code extraction, counterexample messages
  llm.py           Ollama / OpenAI-compatible / mock backends and the reply cache
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

- **Repair is evaluated on the instance that produced the counterexample.** The repaired program is not re-run on other instances. So "repair succeeded" means the model fixed that instance; it does not prove it fixed the general bug. Running repaired programs on held-out instances is the obvious next step.
- **The baseline gets no feedback.** `python` is one-shot. A fairer and harder baseline would be plain Python with the same repair loop. I left it out to keep the first version small.
- **Small sample.** 2 instances per size and 3 programs per family and condition. The confidence intervals are Wilson intervals that treat runs as independent, but runs of one program on different instances are correlated, so the intervals are too narrow.
- **Instance selection.** If the reference CP-SAT model could not prove optimality in 30 s, the instance was discarded and re-drawn (this happened once in the shipped dataset). That biases the set slightly towards instances CP-SAT finds easy.
- **Shared solver.** The ground truth and the model's answer both rely on CP-SAT. The checkers are independent of it, and the brute-force tests help, but a solver bug would not be caught.
- **Counterexamples for suboptimal answers** only say that a better solution exists. They do not reveal the optimal value unless you pass `--reveal-optimum`.
- **The sandbox is a resource guard, not a security boundary.** Generated code runs with your user's permissions. Use a container if you do not trust the model.
- Results depend on the model, temperature, and prompt wording. I report one model per run folder and do not claim anything beyond it.

## License

MIT
