| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| Nonce-line pre-S-box u,v | `35*t` then two adds from pk | running `u-=35`, `v+=35` via `gf64_add` | matches rebuild on windows through `2^26` and near `u32::MAX` | Nonce-line kernel only. Drops one integer multiply after the first nonce in a thread. S-boxes unchanged. |
