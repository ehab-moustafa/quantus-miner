| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| External linear layer (width 12) | 24.05 ns/op | 171.10 ns/op | 0.1406 | Cached 12×12 circulant–MDS matrix (F_p-linear); full mat–vec is slower on CPU than chunk MDS + fold. **Keep chunk MDS.** Hoist `M·Δ` only on the increment path. |
