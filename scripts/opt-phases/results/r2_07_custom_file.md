# r2_07 — `custom.rs` vs qp-poseidon-core 3.1.0

Release on Apple M4 Pro: `cargo test -p pow-core --test r2_07_custom_file --release -- --nocapture`. Two runs. Old is the library. New is `pow_core::custom`. Ratio is old/new. Each cell is the mean of the two run medians.

Asserts passed. Canonical equality held for every edge product (20×20) and 4096 random products, the external-layer vector `[144, 156, …]`, 64 random states (external layer and permute), the P3 zero and sequential permute vectors, and 2000 chained permutes from three starts.

The earlier 573 ns/op figure was one 2000-call custom sample with no library pair. The same shape of sample here was 596 ns and 615 ns. The 2000-call custom median is 619 ns/op.

`fast_mul` is not used by `permute_mut`. Private helpers are only reached through the public functions below.

| Item | Old ns/op | New ns/op | Ratio | Verdict |
|------|-----------|-----------|-------|---------|
| fast_mul vs Goldilocks multiply | 0.641 | 0.775 | 0.827 | Prefer library |
| external_linear_layer vs library | 13.22 | 11.37 | 1.163 | Keep |
| permute_mut vs Poseidon2::permute_mut | 698.6 | 620.5 | 1.126 | Keep |
| whole file, 2000× permute_mut | 701.9 | 619.0 | 1.134 | Keep |

Keep only when custom matched and the median was more than 3% faster. Both runs agreed on every verdict.

Run medians (old / new): mul 0.639/0.770 and 0.642/0.780; external layer 13.14/11.36 and 13.30/11.39; permute 702.1/617.7 and 695.0/623.3; 2000× permute 699.0/622.3 and 704.9/615.8.
