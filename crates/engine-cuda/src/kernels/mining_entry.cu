#include "poseidon.cuh"
#if defined(MINER_CUDA_CPP_OUT0)
#include "cpp_out0.cuh"
#endif
#include "nonce_line.cuh"
#include "target_cmp.cuh"

#define MAX_HITS 8

#ifndef MINER_CUDA_LAUNCH_THREADS
#define MINER_CUDA_LAUNCH_THREADS 256
#endif
#ifndef MINER_CUDA_MIN_BLOCKS
#define MINER_CUDA_MIN_BLOCKS 4
#endif
#ifndef MINER_CUDA_NONCE_LINE_LAUNCH_THREADS
#define MINER_CUDA_NONCE_LINE_LAUNCH_THREADS MINER_CUDA_LAUNCH_THREADS
#endif
#ifndef MINER_CUDA_NONCE_LINE_MIN_BLOCKS
#define MINER_CUDA_NONCE_LINE_MIN_BLOCKS 2
#endif
#ifndef MINER_CUDA_NONCE_LINE_SH_PK
#define MINER_CUDA_NONCE_LINE_SH_PK 1
#endif
#ifndef MINER_CUDA_NONCE_LINE_T_UNROLL
#define MINER_CUDA_NONCE_LINE_T_UNROLL 1
#endif

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

extern "C" __global__ void __launch_bounds__(MINER_CUDA_LAUNCH_THREADS, MINER_CUDA_MIN_BLOCKS) hash_nonces(u32 *hashes, const u32 *midstate,
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

#ifdef MINER_CUDA_DUAL
__device__ __forceinline__ void apply_nonce_low(u64 *st, const u64 *mid, u64 nonce_low) {
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        st[i] = mid[i];
    }
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
}

// Two independent hashes, round by round, so the S-box chains overlap.
__device__ __forceinline__ void permute64_after_initial_dual(u64 *a, u64 *b) {
    #pragma unroll 1
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            a[i] = gf64_sbox(a[i]);
        }
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            b[i] = gf64_sbox(b[i]);
        }
        ext_layer64(a, RC_INITIAL[r + 1]);
        ext_layer64(b, RC_INITIAL[r + 1]);
    }
    u64 xa = gf64_sbox(a[0]);
    u64 xb = gf64_sbox(b[0]);
    #pragma unroll 1
    for (int r = 0; r < 21; r++) {
        u64 rc = RC_INTERNAL[r + 1];
        u64 out_a = int_round_head(a, xa, rc);
        u64 out_b = int_round_head(b, xb, rc);
        u64 a2 = gf64_sqr(out_a);
        u64 b2 = gf64_sqr(out_b);
        int_round_tail(a, xa);
        int_round_tail(b, xb);
        xa = gf64_sbox_after_sqr(out_a, a2);
        xb = gf64_sbox_after_sqr(out_b, b2);
    }
    a[0] = int_round_p(a, xa, 0ULL);
    b[0] = int_round_p(b, xb, 0ULL);
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        a[i] = gf64_add(a[i], RC_TERMINAL[0][i]);
        b[i] = gf64_add(b[i], RC_TERMINAL[0][i]);
    }
    #pragma unroll 1
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            a[i] = gf64_sbox(a[i]);
        }
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            b[i] = gf64_sbox(b[i]);
        }
        ext_layer64(a, RC_TERMINAL[r + 1]);
        ext_layer64(b, RC_TERMINAL[r + 1]);
    }
}

__device__ __forceinline__ void permute64_twice_dual(u64 *a, u64 *b) {
    permute64_after_initial_dual(a, b);
    a[0] = gf64_add(a[0], 1ULL);
    a[1] = gf64_add(a[1], 1ULL);
    b[0] = gf64_add(b[0], 1ULL);
    b[1] = gf64_add(b[1], 1ULL);
    ext_layer64(a, RC_INITIAL[0]);
    ext_layer64(b, RC_INITIAL[0]);
    permute64_after_initial_dual(a, b);
}

__device__ __forceinline__ void record_if_candidate(u32 *results, u64 *st, const u32 *tgt_hi,
                                                    u32 logical_index) {
    u32 first[8];
    state_to_first_limbs(st, first);
    u32 cmp = hash_high_half_cmp(first, tgt_hi);
    if (cmp == 1u) {
        return;
    }
    u32 slot = atomicAdd(&results[0], 1u);
    if (slot < MAX_HITS) {
        results[1 + slot] = logical_index;
    }
}
#endif

extern "C" __global__ void __launch_bounds__(MINER_CUDA_LAUNCH_THREADS, MINER_CUDA_MIN_BLOCKS) mining_main(u32 *results,
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

#ifdef MINER_CUDA_DUAL
    for (u32 j = 0; j < nonces_per_thread;) {
        u32 logical0 = base_index + j;
        if (logical0 >= total_nonces) {
            break;
        }
        u32 logical1 = logical0 + 1u;
        if (j + 1u < nonces_per_thread && logical1 < total_nonces) {
            u64 st0[12];
            u64 st1[12];
            apply_nonce_low(st0, mid, nonce_base_low + (u64)logical0);
            apply_nonce_low(st1, mid, nonce_base_low + (u64)logical1);
            permute64_twice_dual(st0, st1);
            record_if_candidate(results, st0, tgt_hi, logical0);
            record_if_candidate(results, st1, tgt_hi, logical1);
            j += 2u;
            continue;
        }
        u64 st[12];
        apply_nonce_low(st, mid, nonce_base_low + (u64)logical0);
        permute64_twice_after_initial(st);
        record_if_candidate(results, st, tgt_hi, logical0);
        j += 1u;
    }
#else
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
        state_to_first_limbs(st, first);
        u32 cmp = hash_high_half_cmp(first, tgt_hi);
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
#endif
}

/* Issue #110: threads walk t along v; blocks share one pk[14] (same k). */
extern "C" __global__ void __launch_bounds__(MINER_CUDA_NONCE_LINE_LAUNCH_THREADS,
                                             MINER_CUDA_NONCE_LINE_MIN_BLOCKS)
    mining_main_nonce_line(u32 *results, const MiningParams *params_dev) {
    const MiningParams params = *params_dev;
    u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    u32 total_threads = params.dispatch_config[0];
    u32 t_per_thread = params.dispatch_config[1];
    u32 t_end_exclusive = params.dispatch_config[2];
    if (thread_id >= total_threads) {
        return;
    }
#if MINER_CUDA_NONCE_LINE_SH_PK
    __shared__ u64 sh_pk[NONCE_LINE_K_PARAMS];
    if (threadIdx.x < NONCE_LINE_K_PARAMS) {
        sh_pk[threadIdx.x] = params.nonce_line_pk[threadIdx.x];
    }
    __syncthreads();
    const u64 *pk = sh_pk;
#else
    u64 pk_reg[NONCE_LINE_K_PARAMS];
#pragma unroll
    for (int i = 0; i < NONCE_LINE_K_PARAMS; i++) {
        pk_reg[i] = params.nonce_line_pk[i];
    }
    const u64 *pk = pk_reg;
#endif

    u32 tgt_hi[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        tgt_hi[i] = params.difficulty_target[8 + i];
    }

    u32 base_t = params.nonce_line_t_base + thread_id * t_per_thread;
    u64 line_u, line_v;
    gf64_nonce_line_uv_at(pk[0], pk[1], base_t, line_u, line_v);
#pragma unroll MINER_CUDA_NONCE_LINE_T_UNROLL
    for (u32 j = 0; j < t_per_thread; j++) {
        u32 t = base_t + j;
        if (t >= t_end_exclusive) {
            break;
        }
        u32 first_be[8];
        nonce_line_hash_high_half_from_uv_prestate(pk, line_u, line_v, params.prestate,
                                                   params.nonce_line_k_base, t, first_be);
        gf64_nonce_line_uv_step(line_u, line_v);
        u32 cmp = hash_high_half_cmp_be(first_be, tgt_hi);
        if (cmp == 1u) {
            continue;
        }
        u32 slot = atomicAdd(&results[0], 1u);
        if (slot < MAX_HITS) {
            results[1 + slot] = t;
        }
    }
}
