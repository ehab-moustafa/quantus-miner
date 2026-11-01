| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| `mining_main_nonce_line` inner loop | recompute `35*t` every nonce | carry `u,v` and step ±35 | same `u,v` as the single-shot formula | Flag stays off. Default `mining_main` hash path is unchanged except the shared external layer. |
