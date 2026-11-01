# r2_10 CPU hash path and CUDA host model

Release `cargo test -p pow-core --test r2_10_cross --release -- --nocapture`. One test, asserts passed. CPU times are the median of 5 trials, 8000 repetitions, over all 5 `NONCE_HASH_KVS`. Ratio = Old / New. The nanosecond figures are CPU time per call.

The two `custom::permute_mut` calls are timed on the states `mining_midstate` builds: header absorbed, then the high half of the nonce absorbed. `mining_midstate` itself is the same two permutes plus those absorbs and the canonical reduction.

Field adds are `gf64_add` / `GF64_ADD_NONCE_STEP` calls. An M4 call is one `ext_layer64_chunk4` or one `ext_layer64_chunk4_wide`. The current `ext_layer64` does 4: one wide column sum, then three lane chunks. Flat `mining_main` counts are 12 inject adds plus `permute64_twice_after_initial` per nonce (high half only). Nonce-line counts are one thread walking N nonces in one k: one `uv_at`, then a hash and a ±35 step per nonce, including the step after the last nonce. The host prestate and `pk[14]` sit outside both totals.

The running ±35 step uses strict `gf64_add`. It matches `gf64_add(pk, ±35*t)` for every `t` in `0..2^26` on 3 pk pairs, and on 4096-step windows through `u32::MAX` for 13 further pairs (edges plus 8 random). A step from `t = u32::MAX` lands on the `u64` value at `t = 2^32`. The lazy fold, which drops the carry out of the low half, leaves the `35*t` value within 64 steps from `pk = 0`.

The column sum was bit-identical on 276 states (8 edges, 12 single-lane spikes, 256 random): 828 three-chunk M4 calls against 276 wide M4 calls. Both residues match the Goldilocks mat4.

| Item | Old | New | Ratio | Verdict |
|------|-----|-----|-------|---------|
| Golden midstate and full hash, 5 `NONCE_HASH_KVS` | stored `kv.mid`, `kv.hash` | `mining_midstate`, `get_nonce_hash`, `hash_from_nonce`, library resume | 5/5 | Pass. Each `custom::permute_mut` inside the midstate matches `Poseidon2::permute_mut`. Resume from that midstate matches `get_nonce_hash`. |
| Two `custom::permute_mut` vs `get_nonce_hash` | 3619.4 ns | 1241.0 ns | 2.9164 | CPU median on the five vectors. The pair is the midstate prefix (about 620.5 ns per permute). |
| Two `custom::permute_mut` vs `hash_from_nonce` | 3573.9 ns | 1241.0 ns | 2.8798 | `hash_from_nonce` calls `get_nonce_hash` after `to_big_endian`. Same five vectors, same median spread. |
| `mining_midstate` vs `get_nonce_hash` | 3619.4 ns | 1269.4 ns | 2.8514 | Whole function, absorbs included. 28.4 ns above the two permutes alone. |
| Running ±35 `u,v` vs `35*t` | `gf64_add(pk, ±35*t)` | `u += P−35`, `v += 35` | match | Pass on the full `t < 2^26` domain and on the windows above. Strict `gf64_add`. |
| Nonce-line `35*t` multiplies, N=1000, one thread | 1000 | 1 | 1000 | Base `t` is multiplied once. Later nonces are the ±35 step. |
| Nonce-line `u,v` field adds, N=1000 | 2000 | 2002 | 0.999 | Two `gf64_add`s per nonce either way. The extra pair is the step after the last nonce. |
| External-layer column sum | 3 chunk M4s, then add | 1 wide M4 | 3 | Bit-identical on 276 states, and the same residue as the Goldilocks mat4. |
| Field adds, flat `mining_main` vs nonce-line, N=1 | 38 | 52 | 0.731 | Nonce-line spends 21 `gf64_add`s in the round-1 mix. Flat spends 12 on the inject. Both then finish the high half. |
| Field adds, N=1000 | 38000 | 50002 | 0.760 | One thread, one k. Nonce-line total is one `uv_at` plus 1000 hashes and 1000 steps. |
| M4 calls, N=1 | 68 | 64 | 1.0625 | 17 external layers vs 16. The missing layer is `RC_INITIAL[1]`, already folded into `pk`. |
| M4 calls, N=1000 | 68000 | 64000 | 1.0625 | Same 64/68 ratio across the batch. |

No GPU hash rate was measured.
