# Phase 1.08 — Poseidon2 full permute (width 12)

Measured on `cargo test -p pow-core --test p1_08_permute -- --nocapture` (5_000 permutes; 200 random + P3 zero/seq parity).

| Item | Old ns/op | New ns/op | Ratio | Hash-rate note |
|------|-----------|-----------|-------|----------------|
| Poseidon2::permute_mut vs inlined unrolled fast_permute | 869.083 | 872.942 | 0.996 | Full permute microbench only (~1×); mining hash-rate gain depends on fraction of time in permute (dominated by S-box muls). **Keep library `permute_mut`.** |
