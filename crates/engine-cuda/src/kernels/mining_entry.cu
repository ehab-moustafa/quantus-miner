#include "poseidon.cuh"
#include "nonce_line.cuh"

#define MAX_HITS 8

struct MiningParams {
    u32 prestate[24];
    u32 start_nonce[16];
    u32 difficulty_target[16];
    u32 dispatch_config[3];
    /* Nonce line (#110): host fills pk per k-segment (see lib.rs A2). */
    u32 nonce_line_k_base[8];
    i32 nonce_line_v[8];
    u64 nonce_line_pk[NONCE_LINE_K_PARAMS];
    u32 nonce_line_t_base;
    u32 nonce_line_pad;
};

extern "C" __global__ void __launch_bounds__(256, 4) hash_nonces(u32 *hashes, const u32 *midstate,
                                       const u32 *start_nonce, u32 count) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }
    u64 mid[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        mid[i] = ((u64)midstate[2 * i + 1] << 32) | (u64)midstate[2 * i];
    }
    u32 nonce_base[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        nonce_base[i] = start_nonce[i];
    }
    u32 current_nonce[16];
    nonce_from_index(nonce_base, tid, current_nonce);
    u32 hash_le[16];
    hash_from_midstate(mid, current_nonce, hash_le);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        hashes[tid * 16u + (u32)i] = hash_le[i];
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) mining_main(u32 *results,
                                       const MiningParams *params_dev) {
    const MiningParams params = *params_dev;
    u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    u32 total_threads = params.dispatch_config[0];
    u32 nonces_per_thread = params.dispatch_config[1];
    u32 total_nonces = params.dispatch_config[2];
    if (thread_id >= total_threads) {
        return;
    }
    u32 base_index = thread_id * nonces_per_thread;

    u64 mid[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        mid[i] = ((u64)params.prestate[2 * i + 1] << 32) | (u64)params.prestate[2 * i];
    }
    u32 tgt_hi[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        tgt_hi[i] = params.difficulty_target[8 + i];
    }
    u64 nonce_base_low = ((u64)params.start_nonce[1] << 32) |
                         (u64)params.start_nonce[0];

    #pragma unroll

    for (u32 j = 0; j < nonces_per_thread; j++) {
        u32 logical_index = base_index + j;
        if (logical_index >= total_nonces) {
            break;
        }
        u64 st[12];
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            st[i] = mid[i];
        }
        u64 nonce_low = nonce_base_low + (u64)logical_index;
        u64 x6 = (u64)bswap32((u32)(nonce_low >> 32));
        u64 x7 = (u64)bswap32((u32)nonce_low);
        u64 x6_2 = x6 + x6;
        u64 x6_3 = x6_2 + x6;
        u64 x6_4 = x6_2 + x6_2;
        u64 x6_6 = x6_3 + x6_3;
        u64 x7_2 = x7 + x7;
        u64 x7_3 = x7_2 + x7;
        u64 x7_4 = x7_2 + x7_2;
        u64 x7_6 = x7_3 + x7_3;
        u64 c0 = x6 + x7;
        u64 c1 = x6_3 + x7;
        u64 c2 = x6_2 + x7_3;
        u64 c3 = x6 + x7_2;
        st[0] = GF64_ADD_NONCE_STEP(st[0], c0);
        st[1] = GF64_ADD_NONCE_STEP(st[1], c1);
        st[2] = GF64_ADD_NONCE_STEP(st[2], c2);
        st[3] = GF64_ADD_NONCE_STEP(st[3], c3);
        st[4] = GF64_ADD_NONCE_STEP(st[4], x6_2 + x7_2);
        st[5] = GF64_ADD_NONCE_STEP(st[5], x6_6 + x7_2);
        st[6] = GF64_ADD_NONCE_STEP(st[6], x6_4 + x7_6);
        st[7] = GF64_ADD_NONCE_STEP(st[7], x6_2 + x7_4);
        st[8] = GF64_ADD_NONCE_STEP(st[8], c0);
        st[9] = GF64_ADD_NONCE_STEP(st[9], c1);
        st[10] = GF64_ADD_NONCE_STEP(st[10], c2);
        st[11] = GF64_ADD_NONCE_STEP(st[11], c3);
        permute64_twice_after_initial(st);

        u32 first[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            u64 c = gf64_canon(st[i]);
            first[2 * i] = (u32)(c & EPS64);
            first[2 * i + 1] = (u32)(c >> 32);
        }
        u32 cmp = 0u;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u32 h = bswap32(first[i]);
            u32 t = tgt_hi[7 - i];
            if (h != t) {
                cmp = (h > t) ? 1u : 2u;
                break;
            }
        }
        if (cmp == 1u) {
            continue;
        }
        // Record the candidate and keep going: no thread ever stops early, so a
        // launch always evaluates its whole rectangle. The host verifies every
        // recorded index on the CPU.
        u32 slot = atomicAdd(&results[0], 1u);
        if (slot < MAX_HITS) {
            results[1 + slot] = logical_index;
        }
    }
}

/* Issue #110: threads walk t along v; blocks share one pk[14] (same k). */
extern "C" __global__ void __launch_bounds__(256, 4)
    mining_main_nonce_line(u32 *results, const MiningParams *params_dev) {
    const MiningParams params = *params_dev;
    u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    u32 total_threads = params.dispatch_config[0];
    u32 t_per_thread = params.dispatch_config[1];
    u32 t_end_exclusive = params.dispatch_config[2];
    if (thread_id >= total_threads) {
        return;
    }
    u64 pk[NONCE_LINE_K_PARAMS];
#pragma unroll
    for (int i = 0; i < NONCE_LINE_K_PARAMS; i++) {
        pk[i] = params.nonce_line_pk[i];
    }

    u32 tgt_hi[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        tgt_hi[i] = params.difficulty_target[8 + i];
    }

    u32 base_t = params.nonce_line_t_base + thread_id * t_per_thread;
    u64 line_u, line_v;
    gf64_nonce_line_uv_at(pk[0], pk[1], base_t, line_u, line_v);
#pragma unroll 1
    for (u32 j = 0; j < t_per_thread; j++) {
        u32 t = base_t + j;
        if (t >= t_end_exclusive) {
            break;
        }
        u32 first_be[8];
        nonce_line_hash_high_half_from_uv(pk, line_u, line_v, first_be);
        gf64_nonce_line_uv_step(line_u, line_v);
        u32 cmp = 0u;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            u32 h = first_be[i];
            u32 tg = tgt_hi[7 - i];
            if (h != tg) {
                cmp = (h > tg) ? 1u : 2u;
                break;
            }
        }
        if (cmp == 1u) {
            continue;
        }
        u32 slot = atomicAdd(&results[0], 1u);
        if (slot < MAX_HITS) {
            results[1 + slot] = t;
        }
    }
}
