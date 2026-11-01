| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| `is_valid_nonce` verify (20_000 calls, total ns) | 93129125 | 94396458 | 0.9866 | MSB-first limb `hash < target`; parity on `NONCE_HASH_KVS` + 200 random difficulties. Hash dominates; compare tweak alone is not a hash-rate win yet. **Keep library verify.** |
