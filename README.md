# zkautoresearch

Decentralized autoresearch network with STARK cryptographic verification.

Multiple autoresearch agents run in parallel optimizing hyperparameters. Every metric improvement is proven via a STARK proof — verifiers can confirm results without re-running experiments.

## How it works

1. **Parallel agents** each propose and run hyperparameter experiments
2. **When a metric improves**, the agent encodes `delta = (new_reward - best_reward) × 1,000,000` as a u64 and generates a STARK proof that `delta < 1,000,000` using the [stateset-stark](https://github.com/stateset/stateset-stark) circuit
3. **If the metric didn't improve**, delta wraps to ~2^64 and the proof is mathematically impossible to generate
4. **The coordinator verifies proofs** in <1ms without re-running any training
5. **Verified improvements are broadcast** to all agents, who update their baselines and learn

## Performance

| Metric | Value |
|---|---|
| Proof size | ~75 KB |
| Proving time | ~30ms |
| Verification time | <1ms |
| Throughput | ~40 experiments/sec (6 agents) |

## Usage

Requires `ves_stark` Python bindings built from [stateset-stark](https://github.com/stateset/stateset-stark):

```bash
cd /path/to/stateset-stark/crates/ves-stark-python
maturin develop --release
```

Run the network:

```bash
python run_network.py                                    # 4 agents, 20 experiments each
python run_network.py --agents 8 --experiments 50        # scale up
python run_network.py --initial-best 0.523               # start from a baseline
python run_network.py --strategy adaptive                # all agents use adaptive
python run_network.py --export results.json              # export results
```

Run integrity tests:

```bash
python test_stark_integrity.py
```

## Architecture

```
stark_autoresearch/
├── proof.py         — STARK proof generation/verification for metric improvement
├── experiment.py    — Experiment configs + hyperparameter perturbation
├── agent.py         — Parallel agent with adaptive self-learning strategy
├── coordinator.py   — Network coordinator: proof verification + broadcasting
└── __init__.py

run_network.py             — Run the decentralized network
test_stark_integrity.py    — Cryptographic soundness tests
```

## STARK integrity guarantees

- **Valid proofs verify** — genuine improvements are accepted
- **Tampered proofs are rejected** — FRI verification catches any bit flip
- **Non-improvements can't generate proofs** — the circuit is unsatisfiable
- **Wrong baselines are detected** — public input mismatch caught before verification

## License

MIT
