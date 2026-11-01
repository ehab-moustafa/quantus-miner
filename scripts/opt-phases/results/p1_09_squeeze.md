| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| `hash_squeeze_twice` 96B preimage (library vs custom sponge, full 64B out) | 90755084 | 90798250 | 0.9995 | PoW reference is `get_nonce_hash` → library squeeze; custom high-only first 32B matches NONCE_HASH for early reject. **Keep library squeeze.** GPU already skips the second squeeze on reject. |
