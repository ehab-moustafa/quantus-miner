# Phase 1.07 — Poseidon2 external round (width 12)

Measured on `cargo test -p pow-core --test p1_07_ext_round -- --nocapture` (20_000 rounds; 1k random states + P3 zero-state vector).

| Item | Old ns/op | New ns/op | Ratio | Hash-rate note |
|------|-----------|-----------|-------|----------------|
| External round (RC + S-box×12 + ext linear) reference vs unrolled fast | 66.64 | 53.28 | 1.251 | One of eight external rounds per permute; microbench only. **Adopt** the unrolled round in phase 2. |
