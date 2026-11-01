| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| Goldilocks S-box x^7 | 2.15 ns/op | 2.25 ns/op | 0.9549 | S-box is the bulk of permute time; this row is the S-box microbench only (not extrapolated MH/s). **Keep library `exp7`.** The x²·x²·x²·x chain matched values and was slower. |
