# zkautoresearch

Decentralized autoresearch network with STARK cryptographic verification.

Multiple autoresearch agents run in parallel optimizing hyperparameters. Every accepted metric improvement is proven via a STARK proof that is bound to the experiment metadata, config hash, baseline, claimed reward, and improvement delta, so verifiers can confirm results without re-running experiments.

## How it works

1. **Parallel agents** each propose and run hyperparameter experiments
2. **When a metric improves**, the agent encodes `delta = (new_reward - best_reward) × 1,000,000` as a u64 and generates a STARK proof that `delta < 1,000,000` using the [stateset-stark](https://github.com/stateset/stateset-stark) circuit
3. **The proof payload binds** the experiment id, agent id, config hash, baseline reward, claimed new reward, sequence number, and canonical amount binding for `delta`
4. **The coordinator verifies proofs** in <1ms without re-running any training, and rejects mismatched rewards/configs before accepting an update
5. **Verified improvements are broadcast** to all agents, who update their baselines and learn

## Performance

| Metric | Value |
|---|---|
| Proof size | ~75 KB |
| Proving time | ~30ms |
| Verification time | <1ms |
| Throughput | ~40 experiments/sec (6 agents) |

## Usage

Requires `ves_stark` Python bindings built from [stateset-stark](https://github.com/stateset/stateset-stark).

Inspect the current environment:

```bash
python -m stark_autoresearch.environment
zkautoresearch-doctor
```

Install the bindings from a local checkout of `stateset-stark`:

```bash
python scripts/install_ves_stark.py --stateset-stark-dir /path/to/stateset-stark
make install-ves-stark STATESET_STARK_DIR=/path/to/stateset-stark
```

Run the network:

```bash
python run_network.py                                    # 4 agents, 20 experiments each
python run_network.py --agents 8 --experiments 50        # scale up
python run_network.py --initial-best 0.523               # start from a baseline
python run_network.py --strategy adaptive                # all agents use adaptive
python run_network.py --seed 42                          # reproducible search trajectory
python run_network.py --export results.json              # export results
```

Run integrity tests:

```bash
python -m pytest -q
python test_stark_integrity.py
make test
```

## Automation

- `make doctor` prints the current dependency status
- `make test` runs the full test suite
- `make demo-seeded` runs a reproducible demo trajectory
- `.github/workflows/ci.yml` runs a dependency-light packaging job and a full `ves_stark` integration job

## Reproducibility

- Use `--seed` to make agent proposals, synthetic rewards, and experiment ids deterministic across runs
- The exported proof metadata includes config hashes and sequence numbers so accepted trajectories can be audited

## Architecture

```
stark_autoresearch/
├── proof.py         — STARK proof generation/verification for metric improvement
├── experiment.py    — Experiment configs + hyperparameter perturbation
├── agent.py         — Parallel agent with adaptive self-learning strategy
├── coordinator.py   — Network coordinator: proof verification + broadcasting
├── environment.py   — Dependency doctor / runtime environment checks
└── __init__.py

scripts/install_ves_stark.py  — Install the local ves_stark bindings
run_network.py                — Run the decentralized network
test_stark_integrity.py       — Cryptographic soundness tests
test_environment.py           — Packaging / dependency-light tests
test_network_behavior.py      — Reproducibility + export regression tests
```

## STARK integrity guarantees

- **Valid proofs verify** — genuine improvements are accepted
- **Tampered proofs are rejected** — verification fails on altered bytes
- **Non-improvements can't generate proofs** — the circuit is unsatisfiable
- **Wrong baselines or claimed rewards are rejected** — proof-bound scaled values must match the submission
- **Config mismatches are rejected** — the proof is tied to the experiment's config hash
- **Later proofs remain verifiable** — sequence numbers are part of the bound public inputs

## License

MIT
