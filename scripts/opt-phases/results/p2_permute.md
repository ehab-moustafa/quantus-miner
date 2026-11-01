| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| custom permute wired into mining_midstate | library permute | 573 ns/op release | matches library on 64 random states and NONCE_HASH_KVS midstates | CPU midstate path. GPU kernel is unchanged until phases 3–5. |
