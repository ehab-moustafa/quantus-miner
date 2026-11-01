| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| internal linear layer (WIDTH=12) | 14.28 ns/op | 12.17 ns/op | 1.17 | Internal-round diffusion only; hash-rate impact is the fraction of permute time in this layer (not extrapolated MH/s). **Adopt** the unrolled diag in phase 2. |
