# p1_02 — Goldilocks mul / reduce128

Source: `qp-poseidon-core` 3.1.0 `goldilocks.rs` (`Mul`, `reduce128`). New path strength-reduces `(2^32 - 1) * x` to `(x << 32) - x` for 32-bit limbs.

| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| p1_02 mul/reduce128 | 1.33 ns/op | 1.22 ns/op | 1.09 | mul/reduce only. **Release** re-run (`--release`, 1M muls). **Adopt** the shift-subtract fold in phase 2. |

Timing from `cargo test -p pow-core --test p1_02_mul -- --nocapture` (1M muls; dev profile opt-level 1). Re-run test for machine-local `OLD_NS` / `NEW_NS` / `RATIO` stdout.
