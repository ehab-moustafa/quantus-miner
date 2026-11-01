# Measured changes

| Item | Old | New | Ratio | Hash-rate note |
|------|-----|-----|-------|----------------|
| p1_01 Goldilocks field add | 0.545 ns/op | 0.571 ns/op | 0.95× | Field-add microbench only (2M iters). **Keep library add.** |
| p1_02 mul/reduce128 | 1.33 ns/op | 1.22 ns/op | 1.09 | mul/reduce only. **Release** re-run (`--release`, 1M muls). **Adopt** the shift-subtract fold in phase 2. |
| Goldilocks S-box x^7 | 2.15 ns/op | 2.25 ns/op | 0.9549 | S-box is the bulk of permute time; this row is the S-box microbench only (not extrapolated MH/s). **Keep library `exp7`.** The x²·x²·x²·x chain matched values and was slower. |
| Poseidon2 apply_mat4 (4×4 MDS) | 6.12 ns/op | 5.42 ns/op | 1.1282 | External linear layer chunk; mat4 microbench only (not extrapolated MH/s). **Adopt** factored sum in phase 2. |
| External linear layer (width 12) | 24.05 ns/op | 171.10 ns/op | 0.1406 | Cached 12×12 circulant–MDS matrix (F_p-linear); full mat–vec is slower on CPU than chunk MDS + fold. **Keep chunk MDS.** Hoist `M·Δ` only on the increment path. |
| internal linear layer (WIDTH=12) | 14.28 ns/op | 12.17 ns/op | 1.17 | Internal-round diffusion only; hash-rate impact is the fraction of permute time in this layer (not extrapolated MH/s). **Adopt** the unrolled diag in phase 2. |
| External round (RC + S-box×12 + ext linear) reference vs unrolled fast | 66.64 | 53.28 | 1.251 | One of eight external rounds per permute; microbench only. **Adopt** the unrolled round in phase 2. |
| Poseidon2::permute_mut vs inlined unrolled fast_permute | 869.083 | 872.942 | 0.996 | Full permute microbench only (~1×); mining hash-rate gain depends on fraction of time in permute (dominated by S-box muls). **Keep library `permute_mut`.** |
| `hash_squeeze_twice` 96B preimage (library vs custom sponge, full 64B out) | 90755084 | 90798250 | 0.9995 | PoW reference is `get_nonce_hash` → library squeeze; custom high-only first 32B matches NONCE_HASH for early reject. **Keep library squeeze.** GPU already skips the second squeeze on reject. |
| `get_nonce_hash` vs midstate resume (3 permutes) | 4003.2 ns/nonce | 2255.2 ns/nonce | 1.78 | measured CPU ratio only. **Already the GPU prestate path.** Use it for any new CPU reference. |
| `is_valid_nonce` verify (20_000 calls, total ns) | 93129125 | 94396458 | 0.9866 | MSB-first limb `hash < target`; parity on `NONCE_HASH_KVS` + 200 random difficulties. Hash dominates; compare tweak alone is not a hash-rate win yet. **Keep library verify.** |
| bytes_to_felts (32 B) | 100.061 ns/op | 8.105 ns/op | 12.3450 | Injective 4-byte encoding + terminator; not used on mining hot path. **Do not wire into mining.** |
| custom permute wired into mining_midstate | library permute | 573 ns/op release | matches library on 64 random states and NONCE_HASH_KVS midstates | CPU midstate path. GPU kernel is unchanged until phases 3–5. |
| Nonce-line pre-S-box u,v | `35*t` then two adds from pk | running `u-=35`, `v+=35` via `gf64_add` | matches rebuild on windows through `2^26` and near `u32::MAX` | Nonce-line kernel only. Drops one integer multiply after the first nonce in a thread. S-boxes unchanged. |
| External-layer column sum | 3 chunk M4s, then sum | 1 wide M4 on summed lanes | bit-identical on 256 random states plus all-`u64::MAX` | Default kernel. Every external layer does 4 M4s instead of 6. GPU MH/s not measured on this machine. |
| Nonce-line round-1 mix | inner multiply loop `{2,1,1,3}` | explicit double and triple | same field values on 128 random lanes | Same add count, no loop. |
| `mining_main_nonce_line` inner loop | recompute `35*t` every nonce | carry `u,v` and step ±35 | same `u,v` as the single-shot formula | Flag stays off. Default `mining_main` hash path is unchanged except the shared external layer. |
| Host batching | prestate, k-segments, pipeline already in `engine-cuda` | no further host change | — | CPU midstate already uses `custom::permute_mut`. GPU batch shape unchanged. |
