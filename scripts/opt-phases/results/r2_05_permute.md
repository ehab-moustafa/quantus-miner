# r2_05 — whole Poseidon2 permute_mut

Release timing of qp-poseidon-core 3.1.0 `Poseidon2::permute_mut` against `custom::permute_mut` and one new schedule in `pow-core/tests/r2_05_permute.rs`.

The new permute is not the unrolled copy. External rounds still add the round constant, S-box all twelve lanes, then `custom::external_linear_layer`. The 22 internal rounds follow the CUDA `int_round_p` schedule: lane 0 of the diffusion, with the next round constant already added, is formed first, and the next S-box (square, then x^3 and x^4) runs while lanes 1..11 are multiplied. The last internal round stores lane 0 and does not start another S-box.

Canonical equality vs the library held for the fixed zero state (library output matches the p3 zero vector) and for 64 splitmix states. `custom::permute_mut` matched the library on those same inputs. Asserts passed.

Command: `cargo test -p pow-core --test r2_05_permute --release -- --nocapture` (200_000 iterations, median of 7 trials, performance-core QoS). A repeat of that run produced medians 693.995 / 621.451 / 581.183 ns. On that run the fastest library sample was still slower than the slowest custom sample (692.706 vs 628.648, ratio 1.102) and the slowest pipelined sample (692.706 vs 585.891, ratio 1.182).

| Item | Old ns/op | New ns/op | Ratio | Verdict |
|------|----------:|----------:|------:|---------|
| library vs custom::permute_mut | 693.995 | 621.451 | 1.117 | Adopt |
| library vs pipelined permute | 693.995 | 581.183 | 1.194 | Adopt |
