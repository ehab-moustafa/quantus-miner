| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| bytes_to_felts (32 B) | 100.061 ns/op | 8.105 ns/op | 12.3450 | Injective 4-byte encoding + terminator; not used on mining hot path. **Do not wire into mining.** |

**Mining vs serialization (5 lines)**

1. Hot path (`mining_midstate`, GPU launch): absorb exactly **8** little-endian u32 limbs from the 32-byte header (and again for nonce high), then `permute_mut` — **no** terminator felt.
2. `qpow_math::get_nonce_hash` / `hash_squeeze_twice` stream **`bytes_to_felts_iter`** over the full variable-length preimage (96 B → 24 data limbs + terminator); that is verifier/regression, not per-nonce mining.
3. `bytes_to_felts` / `bytes_to_u64s` **Vec** APIs are for storage and tests; sponge hashing uses the iterator to avoid allocation.
4. **Unused on mining:** `felts_to_bytes`, `bytes_to_felts_compact`, `digest_to_bytes` / `bytes_to_digest`, `u128_to_felts`, quantized amount helpers.
5. Phase-1 win here is micro-ops on fixed 32/64 B limb packing if we ever wire injective encoding; midstate already uses the simpler 8-limb header layout.
