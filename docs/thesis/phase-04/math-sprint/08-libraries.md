# Phase 04 — Library APIs vs miner hot path

Dependencies (workspace `Cargo.toml`): **`qp-poseidon-core` 3.1.0**, **`qpow-math`** (chain tag `v0.7.1-q-day-2`). The reference verifier path is five Poseidon2 permutations per nonce (`get_nonce_hash` → `hash_squeeze_twice`). Production mining already skips two permutations via midstate on CPU/GPU, but several crate APIs that implement the same digest with less work are not used where `hash_from_nonce` is still called.

## Unused or under-used APIs (same PoW hash, less per-nonce work)

| Crate path | Function / type | Why it reduces work | Miner usage today |
|------------|-----------------|---------------------|-------------------|
| `pow_core` | `mining_midstate` | Sponge after header + nonce high 256 bits (2 permutations done once per batch). | Used by GPU/CUDA and `MiningHasher`; **not** used by `hash_from_nonce`. |
| `pow_core` | `MiningHasher::hash_if_valid` | Reuses cached midstate + `Poseidon2::permute_mut` ×3 (early exit before 3rd permute when high hash words exceed target). | **CPU engine only**; benchmarks and CUDA verification still call `hash_from_nonce`. |
| `pow_core` | `mining_prestate_low64` / `mining_prestate_low64_u32s` | State after partial absorb of nonce low 192 bits + first external linear layer + RC₀; GPU injects only low 64 bits per nonce. | CUDA/GPU host setup; **not** on CPU reference loop. |
| `pow_core` | `mining_midstate_u32s` | Packed midstate for GPU buffers (avoid host re-pack). | GPU/CUDA; not on `hash_from_nonce` path. |
| `pow_core` | `nonce_line::launch_params` / `segment_launch_params` | Precomputes 14 felts for nonce-line kernel (skips re-deriving external layer + first round from midstate each launch). | CUDA nonce-line; **not** linear CPU nonce scan. |
| `pow_core` | `nonce_line::map_counter_u512` | Maps pool counter → PoW nonce without re-encoding header. | Pool/nonce-line; **not** `step_nonce` linear scan. |
| `qp_poseidon_core` | `Poseidon2::new` + `permute_mut` | Resume sponge from midstate (3 permutations vs 5 inside `hash_squeeze_twice`). | `pow-core` internals; **`qpow_math::get_nonce_hash` always full sponge**. |
| `qp_poseidon_core` | `hash_squeeze_twice` | Single entry for 96-byte header‖nonce preimage (streaming absorb via internal `bytes_to_felts_iter`). | Called only inside **`qpow_math::get_nonce_hash`**; miner could call with prebuilt `[u8; 96]` and skip `U512` conversion on hot path. |
| `qp_poseidon_core::serialization` | `bytes_to_felts_iter` | Incremental 4-byte/felt absorb without allocating preimage `Vec`. | **Private to sponge**; miner re-implements manual u32 absorb in `mining_midstate` / `MiningHasher`. |
| `qp_poseidon_core::serialization` | `digest_to_bytes` | Pack first 4 state felts after permute (squeeze) without full `hash_squeeze_twice` restart. | **Not called** on mining path (manual `as_canonical_u64().to_le_bytes()` in `pow-core`). |
| `qp_poseidon_core::goldilocks` | `Goldilocks` field `Add` / `AddAssign` chain | Update low-nonce absorb with +1 carry across u32 limbs instead of re-parsing 64-byte nonce. | Available; **`hash_from_nonce` rebuilds via `U512::to_big_endian` every nonce**. |
| `qp_poseidon_core::goldilocks` | `Goldilocks::Sum` | `external_linear_layer` could use iterator sum for chunk totals (minor). | Miner uses manual sums in `external_linear_layer`. |
| `qpow_math` | `mine_range` | Same hash as `get_nonce_hash` but inlined loop (no `JobContext` / `U512` wrapper). | Exported from `pow-core`; **CPU engine uses `MiningHasher`, not `mine_range`**. |
| `qpow_math` | `is_valid_nonce` | One hash + target check (avoids separate hash + compare in callers). | WASM solver only; **not** miner-cli engines. |
| `qpow_math` | `achieved_difficulty_from_hash` | Target math from existing hash (no re-hash). | **Unused** in quantus-miner. |

## Not present in these crates (do not wait on them)

- **Batch hash** (many nonces, one syscall): neither `qp-poseidon-core` nor `qpow-math` exposes batched Poseidon2; batching is implemented in **GPU/CUDA kernels** and host midstate upload.
- **SIMD field ops**: `pow-core` feature `simd-poseidon2` is an empty hook; `Goldilocks` uses scalar `x86_64` `add_no_canonicalize_trashing_input` only inside **`qp_poseidon_core::goldilocks`**, not a separate SIMD batch API.
- **Incremental sponge type**: `Poseidon2State` in `qp-poseidon-core` is **private**; there is no public “resume after partial absorb” besides re-implementing midstate in `pow-core`.

## Recommended wiring (already proven in-tree)

1. **Verifier / regression**: keep `hash_from_nonce` → `qpow_math::get_nonce_hash`.
2. **Throughput**: prefer `MiningHasher` or `mining_midstate` + three `permute_mut` (see `pow-core` `test_midstate_resumes_to_full_hash`).
3. **GPU**: `mining_prestate_low64_u32s` + kernel low-64 injection (see `HOT_PATH_MATRIX.md`).
4. **Nonce line**: `segment_launch_params` + `launch_params` (see `nonce_line.rs`).

Benchmark: `cargo run -p pow-core --release --example math_nonce_delta_bench` compares (A) `hash_from_nonce` vs (B) cached midstate + low-nonce absorb chain.
