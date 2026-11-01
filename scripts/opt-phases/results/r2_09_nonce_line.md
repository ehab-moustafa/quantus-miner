# r2_09 nonce_line

`cargo test -p pow-core --test r2_09_nonce_line --release -- --nocapture`

4000 counters, release, fat LTO. Each ns/op is the median of 3 trials of at least 50 ms. The timed loop folds the entire return value. Alternatives live only in `crates/pow-core/tests/r2_09_nonce_line.rs` and `assert_eq` the library on every walking bit of the low half, the edge counters, and all 4000 samples.

Round-trip asserts passed for 100/100 counters. `nonce_from_absolute_t(counter, t_of)` equals `map_counter(counter_with_t)`, `counter_with_t` is idempotent and keeps `k` and `t`, and absorb word 0 recovers `t` as `(w0 - B0) / 4`. Adding `t_room_in_k` lands on `t = 2^26 - 1` with the same `k`; one more step wraps `t` to 0 and increments that 64-bit `k`.

| Item | Old ns/op | New ns/op | Ratio | Verdict |
| --- | --- | --- | --- | --- |
| t_of | 4.007 | 0.372 | 10.7799 | Faster equal form |
| k_of | 0.429 | 0.400 | 1.0725 | Faster equal form |
| base_words | 74.334 | 2.713 | 27.3998 | Faster equal form |
| map_counter | 77.789 | 2.963 | 26.2547 | Faster equal form |
| launch_params | 128.304 | 58.418 | 2.1963 | Faster equal form |
| segment_launch_params | 1364.594 | 1307.483 | 1.0437 | Faster equal form |
| counter_add_t | 7.485 | 1.010 | 7.4145 | Faster equal form |
| nonce_from_absolute_t | 79.875 | 1.681 | 47.5141 | Faster equal form |
| t_room_in_k | 1.831 | 0.381 | 4.8105 | Faster equal form |
| full path | 289.783 | 59.829 | 4.8435 | Faster equal form |

Equal forms, same outputs:

- `t_of` masks the low 26 bits of the last big-endian u32.
- `k_of` is `(low 128 bits >> 26)` as `u64`. That is the same truncation as `(U512 >> 26).low_u64()`: bits `[26, 90)` of the low half. Fat LTO already reduced the library body to a short shift, so the remaining gap is about 7%.
- `base_words` is eight unrolled 29-bit windows from four big-endian u64 limbs, each OR'd with `0x80000000`. Windows start at bits 26, 55, 84, 113, 142, 171, 200, 229. Bits past 255 stay 0.
- `map_counter` uses those words and `wrapping_add` of `V[i] as u32 * t`.
- `launch_params` keeps the same external linear layer, round constants, and `exp7`. Only base-word extraction changes. The field work is about 58 ns; the old bit loop was the rest.
- `segment_launch_params` is the same swap inside `launch_params`. `mining_midstate` is almost the whole call, so the win is about 4%.
- `counter_add_t`, `nonce_from_absolute_t`, and `t_room_in_k` use the little-endian `U512` limbs. `(k << 26) + t` fits in a `u128`. Bits above the 64-bit `k` window are cleared, matching `counter_with_t`. The nonce packs the eight little-endian absorb words with a byte swap into limbs 0..3 and leaves the high half alone.
- The full path is `base_words` + `t_of` + `launch_params` + `nonce_from_absolute_t`, with the midstate already computed. The fast path feeds the extracted words straight into the launch and builds the nonce from limbs. `launch_params` still sees the original counter; the nonce sees the truncated `counter_with_t` counter, which is what the library does.
