| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| External-layer column sum | 3 chunk M4s, then sum | 1 wide M4 on summed lanes | bit-identical on 256 random states plus all-`u64::MAX` | Default kernel. Every external layer does 4 M4s instead of 6. GPU MH/s not measured on this machine. |
| Nonce-line round-1 mix | inner multiply loop `{2,1,1,3}` | explicit double and triple | same field values on 128 random lanes | Same add count, no loop. |
