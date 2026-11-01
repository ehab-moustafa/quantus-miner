# Optimization phases (sequential)

Each phase is 12 agents, each writing **only** its own test and result row. The shell runner measures after the files exist. It does not launch agents.

| Phase | What changes | Gate |
|-------|----------------|------|
| 1 | Custom copies of `qp-poseidon-core` 3.1.0 and `qpow-math` hot functions, with a faster variant that must match the library | `./scripts/opt-phases/run-phase.sh 1` |
| 2 | Move passing phase-1 functions into `pow-core/src/custom/` and call them from the hash path | `run-phase.sh 2` |
| 3 | CUDA `field.cuh` only: incremental adds that match `gf64_add_model` | `run-phase.sh 3` |
| 4 | CUDA `poseidon.cuh` / `nonce_line.cuh` only | `run-phase.sh 4` |
| 5 | `mining_entry.cu` only | `run-phase.sh 5` |
| 6 | `engine-cuda/src/lib.rs` host batching only | `run-phase.sh 6` |
| 7 | Build `RESULTS.md` from `scripts/opt-phases/results/*.md` | `run-phase.sh 7` |

Library sources:

- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/qp-poseidon-core-3.1.0/`
- `qpow-math` git checkout under `~/.cargo/git/checkouts/chain-*/qpow-math/`

Golden rule: new code must `assert_eq` the library on edges and random inputs before any timing row is kept.
