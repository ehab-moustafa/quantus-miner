# qPoW Hot Path Matrix

Canonical PoW is `get_nonce_hash` (five Poseidon2 permutations over header + 64-byte big-endian nonce, then double squeeze). Production mining skips the first two permutations by resuming from a **batch midstate** computed on the host, then runs **three permutations per nonce** on CPU or GPU. This document maps that optimized path from `pow-core` through `engine-cpu`, `engine-gpu` (host + `mining_u64_apple.wgsl`), and the `miner-cli` benchmark loop.

**Reference hash (full 5 permutes, not on hot path):** `crates/pow-core/src/lib.rs:284-287` → `qpow_math::get_nonce_hash`.

---

## End-to-end batch flow (GPU)

```
miner-cli benchmark thread
  └─ MinerEngine::search_range (GpuEngine)
       ├─ once/job: target upload (compare-ready u32s on Apple)
       └─ per batch loop:
            ├─ [host upload] midstate + start_nonce + dispatch (+ clear results)
            ├─ compute pass: mining_main @ mining_u64_apple.wgsl
            ├─ copy results → staging
            ├─ [map_async] + poll (overlap: queue *next* batch inputs here)
            └─ host read flag/nonce/hash → hash_count / Found
```

CPU workers in the same benchmark call `FastCpuEngine::search_range` with no GPU upload or `map_async`; midstate is cached inside `MiningHasher` per high-nonce half.

---

## Stage matrix (primary)

| Stage | What happens | CPU | GPU device | GPU host | Key refs |
| --- | --- | --- | --- | --- | --- |
| **Job / target setup** | `target = U512::MAX / difficulty`; Apple path byte-swaps each target u32 before upload | Both (in-memory) | — | GPU only | `pow-core/src/lib.rs:260-269`; `engine-gpu/src/kernels/mod.rs:75-94`; `engine-gpu/src/lib.rs:700-710` |
| **Precompute (midstate)** | Absorb 32 B header + high 32 B of nonce BE; **2×** `permute_mut`; export 12 canonical u64 felts (24 LE u32 limbs) | Per nonce batch when high half changes | Read uniform/storage midstate once per thread (hoisted) | **Per batch** (or pipelined during prior `map_async` poll) | `pow-core/src/lib.rs:98-110`, `28-35`; `engine-gpu/src/lib.rs:239-246`; `mining_u64_apple.wgsl:5-8`, `277-281`, `317-322` |
| **Precompute (optional prestate)** | Low-64 specialization path (`mining_prestate_low64*`) — **not** used by current CPU/GPU mining engines | — | — | — | `pow-core/src/lib.rs:45-70` (spec only) |
| **Host upload** | `write_buffer`: dispatch config, start nonce (LE u32 limbs), midstate u32s; zero results; (once) difficulty target | — | — | **Yes** | `engine-gpu/src/lib.rs:214-247`, `706-710`, `914-917`, `961-965` |
| **Nonce iteration** | Increment low 256 bits of nonce; batch capped so high half never carries | `step_nonce` / range loop | Inner loop `j` over `nonces_per_thread` | Schedules `total_threads`, `nonces_per_thread`, `batch_size` | `engine-cpu/src/lib.rs:175-195`; `engine-gpu/src/lib.rs:178-191`, `193-211`; `mining_u64_apple.wgsl:266-315` |
| **Absorb low nonce + pad** | Add low 8× u32 (bswap) into sponge state[0..7] | `bytes[32..]` chunks | Same in WGSL | — | `pow-core/src/lib.rs:326-328`; `mining_u64_apple.wgsl:320-322` |
| **Permute** | Poseidon2: **3×** `permute64` / `permute_mut` (squeeze-twice path: 2 before first compare, 1 after pass) | Yes | Yes | — | `pow-core/src/lib.rs:329-341`; `mining_u64_apple.wgsl:245-260`, `325-331`, `355` |
| **Compare (early)** | Reject on first hash word / first 32 B BE without full squeeze | `hash_bytes[..32] > target_bytes[..32]` | `first > tgt[15]` (Apple: host-pre-bswapped target) | — | `pow-core/src/lib.rs:338-339`; `mining_u64_apple.wgsl:332-336` |
| **Compare (full)** | Lexicographic u32 limbs vs target; second squeeze half if first half ≤ target | `hash < target` after full 64 B | 8+8 limb loops, `below` flag | — | `pow-core/src/lib.rs:341-346`; `mining_u64_apple.wgsl:337-371` |
| **Solution publish** | Store nonce + hash limbs atomically | `EngineStatus::Found` | `atomicExchange` on `results[0]` | Parse mapped staging buffer | `engine-cpu/src/lib.rs:178-188`; `mining_u64_apple.wgsl:373-380`; `engine-gpu/src/lib.rs:1004-1025` |
| **Compute dispatch** | `set_pipeline`, `set_bind_group`, `dispatch_workgroups` | — | — | **Yes** | `engine-gpu/src/lib.rs:921-931` |
| **map_async + poll** | Map staging buffer; `device.poll(Wait)`; queue next batch inputs while waiting | — | — | **Yes** | `engine-gpu/src/lib.rs:943-993`, `717-772` (`inputs_prequeued` / `next_batch`) |

---

## Kernel selection (GPU device code path)

| Backend / features | WGSL | Precompute comment | Compare notes |
| --- | --- | --- | --- |
| Metal + `SHADER_INT64` | `mining_u64_apple.wgsl` | `@binding(1)` uniform midstate | Host `pre_bswap_target_upload`; GPU compares without `bswap` on target |
| Other + `SHADER_INT64` | `mining_u64.wgsl` | storage midstate | Target uploaded as raw LE u32s; compare uses `bswap32` on hash limbs |
| No `SHADER_INT64` | `mining.wgsl` | storage midstate | Goldilocks field path |

Selector: `engine-gpu/src/kernels/mod.rs:96-109`. Pipeline entry: `mining_main` at `engine-gpu/src/lib.rs:523-524`.

---

## Crate-by-crate hot path

### `pow-core`

| Stage | Location | Executor |
| --- | --- | --- |
| Midstate (2 permutes) | `lib.rs:98-110` | CPU (host); called from GPU upload |
| Midstate → u32 pack | `lib.rs:28-35` | CPU (host) |
| `MiningHasher` midstate cache | `lib.rs:318-324` | CPU |
| Low-nonce absorb + permute ×3 + compare | `lib.rs:326-346` | CPU |
| `JobContext` / target | `lib.rs:247-270` | CPU (both engines) |

### `engine-cpu`

| Stage | Location | Executor |
| --- | --- | --- |
| `prepare_context` | `lib.rs:138-140` | CPU |
| Hot loop: cancel batching → `hash_if_valid` → `step_nonce` | `lib.rs:165-195` | CPU |
| `hash_if_valid` delegates to `pow-core` | via `MiningHasher` | CPU |

### `engine-gpu` (host)

| Stage | Location | Executor |
| --- | --- | --- |
| `search_range` batch loop | `lib.rs:634-876` | CPU (orchestration) |
| Target upload (job scope) | `lib.rs:700-710` | CPU → GPU queue |
| `write_batch_inputs` | `lib.rs:214-247` | CPU → GPU queue |
| `run_single_batch` upload / dispatch / copy | `lib.rs:910-941` | CPU → GPU queue |
| `map_async` + overlapped next upload | `lib.rs:943-965` | CPU (async map + poll) |
| Result decode / hash accounting | `lib.rs:997-1033` | CPU |
| Batch size clamp (no carry into high nonce) | `lib.rs:178-191` | CPU |

### `mining_u64_apple.wgsl` (device)

| Stage | Location | Executor |
| --- | --- | --- |
| Entry / dispatch guards | `266-275` | GPU |
| Hoist midstate, target, nonce_base | `277-289` | GPU |
| Nonce increment (low 256 bits) | `300-315` | GPU |
| Resume sponge + absorb low half | `317-322` | GPU |
| `permute64` definition | `245-260` | GPU |
| Squeeze phase 0–1 (2 permutes) + early word test | `325-336` | GPU |
| Full first-half compare | `337-354` | GPU |
| Third permute + second-half compare | `355-371` | GPU |
| Atomic solution write | `373-380` | GPU |

### `miner-cli` benchmark loop

| Stage | Location | Executor |
| --- | --- | --- |
| Engine resolve (GPU/CUDA) | `main.rs:365-386` | CPU |
| `FastCpuEngine::new` / chunk sizes | `main.rs:413-417`, `444-445` | CPU |
| `prepare_context` (shared `JobContext`) | `main.rs:427-428` | CPU |
| Worker thread: stride ranges, `search_range` | `main.rs:470-514` | CPU (+ GPU inside engine) |
| Cancel + hash aggregation | `main.rs:527-553` | CPU |
| Per-thread GPU teardown | `main.rs:517-518` (`GpuEngine::clear_worker_resources`) | CPU |

Benchmark intentionally uses `U512::MAX` difficulty (`main.rs:425`) so the hot path is almost always **permute + compare reject**, not solution handling.

---

## Overlap: precompute vs host upload vs map_async

| When | Action | Ref |
| --- | --- | --- |
| First batch in `search_range` | `write_batch_inputs` then dispatch | `lib.rs:910-912` |
| Steady state | Previous batch’s poll loop calls `write_batch_inputs` for **next** start nonce while current staging `map_async` completes | `lib.rs:894-896`, `961-965` |
| Same poll loop | `device.poll(Wait { timeout: 10ms })` until map callback | `lib.rs:968-971` |
| Midstate CPU work | Runs inside `write_batch_inputs` via `mining_midstate_u32s` | `lib.rs:239-246` |

---

## Related entry points (same hot path, not benchmark)

| Caller | Role | Ref |
| --- | --- | --- |
| `miner-service` worker pool | `engine.search_range` with production batch sizes | `miner-service/src/lib.rs:468` |
| `engine-gpu/benches/gpu_engine_bench.rs` | Criterion → `GpuEngine::search_range` | benches file |
| `engine-cpu/benches/cpu_engine_bench.rs` | Criterion → `FastCpuEngine::search_range` | benches file |

---

## Invariants tying stages together

1. **Midstate validity:** GPU batches must not increment nonce across a carry into the high 256 bits (`batch_nonce_count` headroom + kernel assumption `mining_u64_apple.wgsl:300-301`).
2. **Bit-exactness:** Apple u64 kernel must match `pow-core` / `mining.wgsl` (`kernels/mod.rs:3-4`).
3. **Compare layout:** Apple uploads compare-ready target u32s (`kernels/mod.rs:37-40`); kernel comment at `mining_u64_apple.wgsl:332-333`.

---

*Generated for deep-sprint thesis work (agent R01). Repository root: `quantus-miner`.*
