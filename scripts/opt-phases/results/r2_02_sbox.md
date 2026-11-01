# r2_02 — Goldilocks S-box x^7

Source: qp-poseidon-core 3.1.0 `Goldilocks::exp7` (`x^2`, `x^3`, `x^4`, `x^7`). Release throughput on 2000 random inputs (median of 8 trials). Old is library `exp7`. Ratio is old/new. Canonical equality held on 44 edges and 2000 random inputs before timing. A repeat run stayed within about 0.01 on every ratio.

| Item | Old ns/op | New ns/op | Ratio | Verdict |
|------|-----------|-----------|-------|---------|
| x2,x4,x6,x7 | 1.8679 | 2.1212 | 0.8806 | Reject |
| x2,x3,x4,x7 | 1.8679 | 1.8709 | 0.9984 | Reject |
| x2,x3,x6,x7 | 1.8679 | 2.1075 | 0.8863 | Reject |
| x2,x4,x5,x7 | 1.8679 | 2.1102 | 0.8852 | Reject |
| x2,x3,x5,x7 | 1.8679 | 2.1270 | 0.8782 | Reject |
| x2,x4,x3,x7 | 1.8679 | 1.8821 | 0.9925 | Reject |
