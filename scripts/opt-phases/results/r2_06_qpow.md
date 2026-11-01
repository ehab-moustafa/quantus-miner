# r2_06 qpow-math hash re-challenge

Release `cargo test -p pow-core --test r2_06_qpow --release`. Median of 5 paired rounds. Asserts passed: every alternative hash matched `get_nonce_hash` on `NONCE_HASH_KVS` and on 20 random header/nonce pairs, including a 2^256 carry. `mine_range` results matched on easy, zero, max, and random difficulties.

Old is the library call. New is a test-only alternative. Adopt only when outputs match and every round's ratio is above 1.03.

The mining group is absorb (header + nonce, high half cached), permute, squeeze, compare. Its target is `U512::MAX / 2^40`. All 32 sampled hashes rejected on the high half, so the new group skips the second squeeze.

| Item | Old ns/op | New ns/op | Ratio | Verdict |
| --- | --- | --- | --- | --- |
| `bytes_to_felts_iter` (96 B) | 87.5 | 1.1 | 79.371 | Reject — not on mining hot path |
| `hash_squeeze_twice`, fused absorb + `custom::permute_mut` | 3540.4 | 3012.4 | 1.175 | Adopt |
| `hash_squeeze_twice`, fused absorb + hoisted library permute | 3401.0 | 3452.7 | 0.985 | Reject |
| `get_nonce_hash` single-shot (fused + custom permute) | 3553.6 | 3031.4 | 1.172 | Adopt |
| `get_nonce_hash` sequential midstate, library permute, full hash | 3612.5 | 2115.5 | 1.708 | Adopt — prefer the custom-permute row |
| `get_nonce_hash` sequential midstate, custom permute, full hash | 3541.3 | 1802.6 | 1.965 | Adopt |
| `is_valid_nonce` | 3648.5 | 3218.1 | 1.134 | Adopt — not the search hot path |
| `hash < target` limb walk | 1.923 | 1.934 | 0.995 | Reject — not on mining hot path |
| `achieved_difficulty_from_hash` | 19.4 | 18.6 | 1.041 | Reject — not on mining hot path |
| Group vs `get_nonce_hash`, library permute | 3454.5 | 1334.5 | 2.589 | Adopt — prefer the custom-permute row |
| Group vs `get_nonce_hash`, custom permute | 3559.9 | 1222.4 | 2.912 | Adopt |
| Group vs `hash_from_nonce`, custom permute | 3576.7 | 1234.2 | 2.898 | Adopt |
| Hash file `mine_range`, custom permute | 3526.6 | 1213.7 | 2.906 | Adopt |

Search path to wire: resume after header and nonce-high, run `custom::permute_mut`, and skip the second squeeze when the high 32 bytes already miss the target. That is 2.91× `get_nonce_hash` and 2.90× `hash_from_nonce`. Single-shot `get_nonce_hash` / `hash_squeeze_twice` still gains 1.17× from the fused custom permute. Hoisting library `Poseidon2` alone does not.

`bytes_to_felts_iter`, the limb compare, and `achieved_difficulty_from_hash` are not the per-nonce search. The achieved-difficulty ratio is also not stable (slowest round 0.87).
