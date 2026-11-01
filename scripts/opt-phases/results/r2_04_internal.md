# r2_04 — Poseidon2 internal layer and internal rounds

qp-poseidon-core 3.1.0. Release: `cargo test -p pow-core --test r2_04_internal --release`.

Old is the library body (`internal_linear_layer`, lane-0 RC + `exp7` + that layer, and the 22-round internal section of `permute_mut`). New is a const unrolled diagonal using the same field add, sum, and mul. ns/op are medians of 7 trials. Ratio is old/new. Adopt only when canonical results match and the ratio is clearly above 1.03.

Canonical equality passed: library sequential vector, edge states, random states, and full permute against `Poseidon2::permute_mut`.

| Item | Old ns/op | New ns/op | Ratio | Verdict |
|------|----------:|----------:|------:|---------|
| internal linear layer | 9.35 | 9.30 | 1.005 | Reject |
| one internal round | 14.74 | 13.30 | 1.108 | Adopt |
| 22 internal rounds | 340.52 | 286.17 | 1.190 | Adopt |

Trial ratios: layer 0.986–1.017, one round 1.089–1.150, 22 rounds 1.123–1.196. The layer alone ties under release optimization. The same unroll is faster once it sits in a round and in the 22-round section.
