# zkautoresearch

STARK-verified autonomous research for `stateset-agents`.

This repo now combines two related pieces of work:

- `auto_research.py`: a subprocess-backed autonomous RL experiment runner with hard time budgets, resumable runs, runtime fingerprints, signed provenance, STARK proof emission for real improvements, audit reports, best-artifact repair, and append-only repair history.
- `run_network.py` plus `stark_autoresearch/`: the decentralized multi-agent demo that submits proof-backed improvements to a coordinator without rerunning experiments.

## Main workflows

### 1. Autonomous research runner

```bash
uv sync
python auto_research.py --runtime-profile smoke --max-experiments 2 --time-budget 60
python auto_research.py --analyze --output-dir ./auto_research_results
python auto_research.py --audit-run ./auto_research_results --require-signature
python auto_research.py --repair-best-artifacts ./auto_research_results --require-signature
```

Useful verification commands:

```bash
python auto_research.py --verify-provenance ./auto_research_results --require-signature
python auto_research.py --verify-proof ./auto_research_results --require-signature
python auto_research.py --verify-repair-report ./auto_research_results --require-signature
python auto_research.py --verify-repair-history ./auto_research_results --require-signature
```

The runner operates on the fixed evaluation harness in `prepare.py` and the editable training entrypoint in `train.py`. Results are written under `auto_research_results/` with per-run summaries, provenance envelopes, proof artifacts, `audit_report.json`, `attestation_summary.json`, `repair_report.json`, and `repair_history/repair_*.json`.

### 2. Decentralized network demo

```bash
python run_network.py --agents 2 --experiments 5 --seed 42
python run_network.py --agents 4 --experiments 20 --export results.json
```

This path keeps the original STARK-backed coordinator demo and reproducibility tests intact.

## Environment

The repo expects local checkouts of:

- `../stateset-agents`
- `../icommerce-app/stateset-stark/crates/ves-stark-python`

Dependency status can be checked with:

```bash
python -m stark_autoresearch.environment
zkautoresearch-doctor
```

## Tests

```bash
python -m pytest -q
make test-core
make test-autoresearch
make demo-seeded
```

## Layout

```text
stark_autoresearch/          STARK proof, agent, coordinator, environment code
auto_research.py             Main CLI for autonomous research / audit / repair
subprocess_auto_research.py  Subprocess experiment runner
attestation_audit.py         Provenance, proof, best-artifact, and repair verification
experiment_runtime.py        Runtime detection, hashing, signing, provenance helpers
prepare.py                   Locked evaluation harness and scenarios
train.py                     Single experiment entrypoint
program.md                   Human/agent operating instructions
```

## License

MIT
