# r2_08 `lib.rs` mining functions

Release `cargo test -p pow-core --test r2_08_lib_file --release`. Median of 5 trials. Old is one `hash_from_nonce` (`qpow_math::get_nonce_hash`). New is the narrower call. Ratio = Old / New. CPU nanoseconds per call only.

Golden asserts passed. All 5 `NONCE_HASH_KVS` entries: `mining_midstate` equals stored `kv.mid`, and `hash_from_nonce` equals stored `kv.hash`. The same 5, plus 32 random headers, also match a library Poseidon2 two-permute midstate, a library external-linear prestate, and `get_nonce_hash`. Flipping the low 64 nonce bits leaves midstate and prestate unchanged and changes the full hash.

`mining_prestate_low64` already calls `mining_midstate`. The group row is one midstate call plus one prestate call.

| Item | Old ns/op | New ns/op | Ratio | Verdict |
|------|-----------|-----------|-------|---------|
| `mining_midstate`, first `NONCE_HASH_KVS` vector | 3750.1 | 1276.6 | 2.94 | Faster prefix (two permutations). |
| `mining_prestate_low64`, first vector | 3750.1 | 1290.4 | 2.91 | Faster prefix. About 14 ns above midstate; the extra linear layer and round constant are cheap. |
| midstate + prestate together, first vector | 3750.1 | 2514.8 | 1.49 | Pair is still under one full hash, and it matches the sum of the two calls. One prestate call already includes the midstate. |
| `mining_midstate`, 32 random headers | 3589.1 | 1204.7 | 2.98 | Faster prefix (two permutations). |
| `mining_prestate_low64`, 32 random headers | 3589.1 | 1267.7 | 2.83 | Faster prefix. Still only a few dozen nanoseconds above midstate. |
| midstate + prestate together, 32 random headers | 3589.1 | 2504.7 | 1.43 | Pair stays under one full hash and again tracks midstate plus prestate. Use prestate alone when the next step is the low-64 resume. |
