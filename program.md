# autoresearch — stateset-agents

Autonomous RL research on the [stateset-agents](../stateset-agents) framework. An AI agent iterates on GSPO/GRPO training configurations to maximize conversational agent reward.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, evaluation scenarios, evaluation harness. Do not modify.
   - `train.py` — the file you modify. Training hyperparameters, reward config, model config, RL algorithm settings.
4. **Verify setup**: Run `uv run prepare.py` to verify dependencies are installed and GPU is available.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Platform-powered mode (alternative)

Instead of the manual agent loop below, you can use the **platform-powered auto-research loop** built into stateset-agents. This automates everything — proposing experiments, training, evaluating, keeping/reverting — without needing an LLM agent to drive the loop.

```bash
# Default: perturbation proposer, 5-min time budget, runs forever
uv run auto_research.py

# Bayesian optimization with early stopping
uv run auto_research.py --proposer bayesian --patience 10

# LLM-driven proposals (requires ANTHROPIC_API_KEY)
uv run auto_research.py --proposer llm --max-experiments 50

# Try different training algorithms automatically
uv run auto_research.py --search-space multi_algorithm --algorithm auto

# Quick test run
uv run auto_research.py --max-experiments 5 --time-budget 60
```

Results are saved to `./auto_research_results/` (JSONL + TSV). The loop **resumes automatically** if restarted — it reads `experiments.jsonl`, restores the best checkpoint, and continues from where it left off.

**Proposer strategies:** `perturbation` (default), `smart` (learns which params matter), `adaptive` (starts broad, narrows down), `random`, `grid`, `bayesian` (needs `pip install optuna`), `llm` (needs `ANTHROPIC_API_KEY`).

**Analyzing results after a run:**

```python
from stateset_agents.training.auto_research import ExperimentTracker, compare_runs

# Load and inspect
tracker = ExperimentTracker.load("./auto_research_results")
tracker.print_summary()  # Includes ASCII convergence chart

# Structured analysis (for Jupyter)
analysis = tracker.get_analysis()
# analysis["parameter_importance"], analysis["convergence_curve"], analysis["experiments"]

# Compare two runs
print(compare_runs("./run_perturbation", "./run_smart"))

# Import existing results.tsv from manual experiments
tracker = ExperimentTracker.from_legacy_tsv("results.tsv")
tracker.print_summary()
```

See `auto_research.py` for full configuration.

## What you're optimizing

You are training a **conversational customer service agent** using the stateset-agents GSPO (Group Sequence Policy Optimization) framework. The agent learns to have multi-turn conversations with customers — answering questions, resolving issues, and providing support.

The **metric is `avg_reward`** — higher is better. This is computed by the fixed evaluation harness in `prepare.py` which runs the trained agent through 8 held-out customer service scenarios and computes a multi-objective reward score.

## Experimentation

Each experiment runs on a single GPU. Training is invoked simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - **Hyperparameters**: learning rate, batch size, warmup, gradient accumulation, etc.
  - **RL algorithm settings**: clip ratio, entropy coefficient, KL penalty, advantage normalization, baseline type, GSPO clip ranges, number of outer iterations.
  - **Model configuration**: model name/size, LoRA rank, quantization, gradient checkpointing.
  - **Reward configuration**: domain, custom reward weights, or even custom reward components.
  - **Training scenarios**: the scenarios the agent trains on (distinct from eval scenarios).
  - **System prompt**: the agent's persona and instructions.
  - **Architectural changes**: you can restructure train.py, add custom training loops, use different stateset-agents trainers (GRPO, DAPO, VAPO), or compose reward functions differently.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed evaluation harness, evaluation scenarios, and constants.
- Modify files in the `stateset-agents` package directory (../stateset-agents/).
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the highest avg_reward.** Everything in train.py is fair game.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful avg_reward gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so run `train.py` as-is.

## Output format

Once the script finishes it prints a summary:

```
---
avg_reward:       0.654321
reward_std:       0.123456
success_rate:     0.7500
avg_ep_length:    4.2
training_seconds: 285.3
total_seconds:    310.7
peak_vram_mb:     4500.2
num_episodes:     5
num_generations:  4
learning_rate:    5e-06
lora_r:           8
model:            gpt2
reward_domain:    customer_service
```

Extract the key metric from the log:

```
grep "^avg_reward:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	avg_reward	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. avg_reward achieved (e.g. 0.654321) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	avg_reward	memory_gb	status	description
a1b2c3d	0.654321	4.4	keep	baseline
b2c3d4e	0.712345	4.5	keep	increase LR to 1e-5
c3d4e5f	0.643210	4.4	discard	add KL penalty beta=0.1
d4e5f6g	0.000000	0.0	crash	switch to 7B model (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar13`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit
2. Modify `train.py` with an experimental idea
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read the results: `grep "^avg_reward:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback.
7. Record the results in `results.tsv` (do NOT commit results.tsv — leave it untracked)
8. If avg_reward **improved** (higher), keep the commit and advance the branch
9. If avg_reward is equal or worse, `git reset --hard` back to where you started

**Timeout**: Each run should take roughly 5 minutes for training + evaluation overhead. If a run exceeds 10 minutes, kill it and treat as a failure.

**Crashes**: If it's a simple bug (typo, import error), fix and re-run. If the idea itself is broken (OOM, fundamental incompatibility), log "crash" and move on.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask. The human may be asleep. You are autonomous. If you run out of ideas, think harder — try combining things that worked, try more radical changes, revisit ideas that were close, try different reward functions, try different models. The loop runs until manually stopped.

## Ideas to explore

Here's a non-exhaustive list of experimental directions:

**Quick wins (try first):**
- Increase learning rate (1e-5, 2e-5)
- Increase num_generations (8, 16)
- Increase num_outer_iterations (3, 5)
- Try different warmup ratios (0.05, 0.2)
- Adjust GSPO clip ranges

**Reward engineering:**
- Try "sales" or "technical_support" reward domains
- Set custom reward weights emphasizing different qualities
- Compose multiple reward functions

**Model changes:**
- Try distilgpt2 (smaller, faster, more iterations)
- Try gpt2-medium (larger, fewer iterations, potentially better)
- Adjust LoRA rank (4, 16, 32)
- Change LoRA alpha ratio

**Algorithm changes:**
- Adjust entropy coefficient
- Add KL penalty (beta > 0)
- Try different baseline types ("group_median")
- Change advantage normalization
- Use the high-level `train()` function with different profiles

**Training data:**
- Add more diverse training scenarios
- Adjust scenario complexity
- Change system prompt to be more detailed

**Advanced:**
- Use DAPO or VAPO trainers from stateset-agents
- Implement curriculum learning (easy scenarios first)
- Try different gradient accumulation strategies
