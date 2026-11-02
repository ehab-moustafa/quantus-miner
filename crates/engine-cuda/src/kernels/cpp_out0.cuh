/*
 * CPPminer partial second-squeeze filter (hash_out0 schedule).
 * Reference: danifest751/CPPminer cp_qpow_cuda.cu — hash_out0(), upload_constants() c_post.
 * Wired from nonce_line when MINER_CUDA_CPP_OUT0=1 (replaces full permute64 tail).
 */
#pragma once

#if defined(MINER_CUDA_CPP_OUT0)

/* Build c_post[g] from poseidon RC tables (same logic as CPPminer upload_constants). */
__device__ __forceinline__ void cpp_out0_c_post_row(int g, u64 row[12]) {
#pragma unroll
    for (int i = 0; i < 12; i++) {
        row[i] = 0ULL;
    }
    const int gg = g & 7;
    if (gg <= 2) {
#pragma unroll
        for (int i = 0; i < 12; i++) {
            row[i] = RC_INITIAL[gg + 1][i];
        }
    } else if (gg == 3) {
        row[0] = RC_INTERNAL[0];
    } else if (gg <= 6) {
#pragma unroll
        for (int i = 0; i < 12; i++) {
            row[i] = RC_TERMINAL[gg - 3][i];
        }
    } else {
        row[0] = 1ULL;
        row[1] = 1ULL;
    }
}

__device__ __forceinline__ void cpp_out0_c_post_row15(u64 row[12]) {
#pragma unroll
    for (int i = 0; i < 12; i++) {
        row[i] = RC_INITIAL[0][i];
    }
}

__device__ __forceinline__ void cpp_out0_internal22(u64 s[12]) {
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
        u64 x = gf64_sbox(s[0]);
        x = int_round_p(s, x, RC_INTERNAL[r + 1]);
    }
#pragma unroll
    for (int i = 0; i < 12; i++) {
        s[i] = CPP_OUT0_GF64_ADD(s[i], RC_TERMINAL[0][i]);
    }
}

#if defined(MINER_CUDA_OUT0_LAZY_W) && MINER_CUDA_OUT0_LAZY_W
/* Experimental: lazy wide reduce on dedicated out0 int rounds (not wired — T-126). */
__device__ __forceinline__ u64 cpp_out0_int_round_head(u64 *state, u64 x, u64 rc0) {
    Wide s0 = int_row_sum(state, x);
    wide_add(s0, rc0);
    u32 r0, r1, r2, r3;
    mul_diag_add(x, MDS_DIAG[0], s0, r0, r1, r2, r3);
    return CPP_OUT0_REDUCE128(r0, r1, r2, r3);
}
#endif

/* One g iteration in hash_out0 (sboxes, optional ext / internal22). */
#if defined(MINER_CUDA_OUT0_EXT_UNROLL) && MINER_CUDA_OUT0_EXT_UNROLL
__device__ __forceinline__ void cpp_out0_ext_layer64(u64 *state, const u64 *rc12) {
    Wide col_sum[4];
    ext_layer64_cols(state, col_sum);
    ext_layer64_reduce_chunk4(state[0], state[1], state[2], state[3], col_sum, rc12[0], rc12[1],
                              rc12[2], rc12[3], &state[0], &state[1], &state[2], &state[3]);
    ext_layer64_reduce_chunk4(state[4], state[5], state[6], state[7], col_sum, rc12[4], rc12[5],
                              rc12[6], rc12[7], &state[4], &state[5], &state[6], &state[7]);
    ext_layer64_reduce_chunk4(state[8], state[9], state[10], state[11], col_sum, rc12[8], rc12[9],
                              rc12[10], rc12[11], &state[8], &state[9], &state[10], &state[11]);
}
#define CPP_OUT0_EXT_LAYER cpp_out0_ext_layer64
#else
#define CPP_OUT0_EXT_LAYER ext_layer64
#endif

__device__ __forceinline__ void cpp_out0_g_step(u64 s[12], int g, bool final_g) {
#pragma unroll
    for (int i = 0; i < 12; i++) {
        s[i] = gf64_sbox(s[i]);
    }
    if (final_g) {
        return;
    }
    u64 post[12];
    cpp_out0_c_post_row(g, post);
    CPP_OUT0_EXT_LAYER(s, post);
    if (g == 7) {
        cpp_out0_c_post_row15(post);
        CPP_OUT0_EXT_LAYER(s, post);
    }
    if ((g & 7) == 3) {
        cpp_out0_internal22(s);
    }
}

#if defined(MINER_CUDA_OUT0_G_UNROLL) && MINER_CUDA_OUT0_G_UNROLL
/* blackbeard-style: fully unrolled g=1..15 (no back-edge branch). */
__device__ __forceinline__ void cpp_out0_second_permute_loops(u64 s[12]) {
    cpp_out0_g_step(s, 1, false);
    cpp_out0_g_step(s, 2, false);
    cpp_out0_g_step(s, 3, false);
    cpp_out0_g_step(s, 4, false);
    cpp_out0_g_step(s, 5, false);
    cpp_out0_g_step(s, 6, false);
    cpp_out0_g_step(s, 7, false);
    cpp_out0_g_step(s, 8, false);
    cpp_out0_g_step(s, 9, false);
    cpp_out0_g_step(s, 10, false);
    cpp_out0_g_step(s, 11, false);
    cpp_out0_g_step(s, 12, false);
    cpp_out0_g_step(s, 13, false);
    cpp_out0_g_step(s, 14, false);
    cpp_out0_g_step(s, 15, true);
}
#elif defined(MINER_CUDA_OUT0_G_PARTIAL9) && MINER_CUDA_OUT0_G_PARTIAL9
__device__ __forceinline__ void cpp_out0_second_permute_loops(u64 s[12]) {
    cpp_out0_g_step(s, 1, false);
    cpp_out0_g_step(s, 2, false);
    cpp_out0_g_step(s, 3, false);
    cpp_out0_g_step(s, 4, false);
    cpp_out0_g_step(s, 5, false);
    cpp_out0_g_step(s, 6, false);
    cpp_out0_g_step(s, 7, false);
    cpp_out0_g_step(s, 8, false);
    cpp_out0_g_step(s, 9, false);
#pragma unroll 1
    for (int g = 10; g <= 15; g++) {
        cpp_out0_g_step(s, g, g == 15);
    }
}
#elif defined(MINER_CUDA_OUT0_G_PARTIAL) && MINER_CUDA_OUT0_G_PARTIAL
/* T-131: unroll g=1..7 (heavy internal22 at g=3,7), loop g=8..15. */
__device__ __forceinline__ void cpp_out0_second_permute_loops(u64 s[12]) {
    cpp_out0_g_step(s, 1, false);
    cpp_out0_g_step(s, 2, false);
    cpp_out0_g_step(s, 3, false);
    cpp_out0_g_step(s, 4, false);
    cpp_out0_g_step(s, 5, false);
    cpp_out0_g_step(s, 6, false);
    cpp_out0_g_step(s, 7, false);
#pragma unroll 1
    for (int g = 8; g <= 15; g++) {
        cpp_out0_g_step(s, g, g == 15);
    }
}
#else
/* Second permute rounds g=1..15 (CPPminer hash_out0 inner loop). */
__device__ __forceinline__ void cpp_out0_second_permute_loops(u64 s[12]) {
#pragma unroll 1
    for (int g = 1;; g++) {
        const bool final_g = (g == 15);
        cpp_out0_g_step(s, g, final_g);
        if (final_g) {
            break;
        }
    }
}
#endif

/* out0 = (M_ext s)[0] — first row of external MDS fold (CPPminer tail). */
__device__ __forceinline__ u64 cpp_out0_ext_row0_canon(const u64 s[12]) {
    Wide acc = wide_from(s[0]);
    wide_add(acc, s[0]);
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int rep_m = (k == 0) ? 2 : 1;
#pragma unroll
        for (int rep = 0; rep < rep_m; rep++) {
            if (!(k == 0 && rep == 0)) {
                wide_add(acc, s[4 * k]);
                wide_add(acc, s[4 * k]);
            }
            wide_add(acc, s[4 * k + 1]);
            wide_add(acc, s[4 * k + 1]);
            wide_add(acc, s[4 * k + 1]);
            wide_add(acc, s[4 * k + 2]);
            wide_add(acc, s[4 * k + 3]);
        }
    }
    return gf64_canon(wide_reduce(acc));
}

#endif /* MINER_CUDA_CPP_OUT0 */
