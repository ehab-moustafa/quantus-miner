/* Structured nonce search ("nonce line") — Quantus-Network/quantus-miner#110
 *
 * Reference: CPPminer nonce_line.hpp
 * https://github.com/danifest751/CPPminer/blob/release/fork/src/qpow/include/qpow/nonce_line.hpp
 *
 * Walk w = B(k) + t*v so M_ext*v is zero except lanes 0 and 4. Round 1 needs two
 * S-boxes; the other ten lanes fold into pk[14] per k (launch_params).
 *
 * Hash dispatch (merge M3):
 *   MINER_CUDA_CPP_OUT0 unset/0 — pow-oracle path (midstate + permute64_pow) for G0A tests.
 *   MINER_CUDA_CPP_OUT0=1       — UV round-1 + cpp_out0.cuh out0 schedule (T-123 champion).
 */

#pragma once

#if defined(MINER_CUDA_CPP_OUT0)
#include "cpp_out0.cuh"
#endif

#define NONCE_LINE_K_T_BITS 26
#define NONCE_LINE_K_PARAMS 14

#ifndef MINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL
#define MINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL 1
#endif
#ifndef MINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL
#define MINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL 1
#endif

__device__ __constant__ int NONCE_LINE_K_V[8] = {4, -17, 11, -3, -4, 17, -11, 3};

/* bits [pos, pos + len) of the 256-bit big-endian number b[0..31] (8 x u32 BE limbs). */
__device__ __forceinline__ u32 nonce_line_bits256(const u32 b_be[8], int pos, int len) {
    u32 r = 0;
    for (int i = len - 1; i >= 0; i--) {
        const int bit = pos + i;
        u32 byte_val = 0u;
        if (bit < 256) {
            const int byte_idx = 31 - bit / 8;
            const int limb = byte_idx / 4;
            const int off = 3 - (byte_idx % 4);
            byte_val = (b_be[limb] >> (off * 8)) & 0xFFu;
        }
        r = (r << 1) | (byte_val >> (bit % 8)) & 1u;
    }
    return r;
}

__device__ __forceinline__ void nonce_line_base_words(const u32 ctr_hi_be[8], u32 b[8]) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        b[i] = 0x80000000u | nonce_line_bits256(ctr_hi_be, NONCE_LINE_K_T_BITS + 29 * i, 29);
    }
}

__device__ __forceinline__ void nonce_line_external_round_sboxes(u64 *state) {
#if MINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL
    state[0] = gf64_sbox(state[0]);
    state[1] = gf64_sbox(state[1]);
    state[2] = gf64_sbox(state[2]);
    state[3] = gf64_sbox(state[3]);
    state[4] = gf64_sbox(state[4]);
    state[5] = gf64_sbox(state[5]);
    state[6] = gf64_sbox(state[6]);
    state[7] = gf64_sbox(state[7]);
    state[8] = gf64_sbox(state[8]);
    state[9] = gf64_sbox(state[9]);
    state[10] = gf64_sbox(state[10]);
    state[11] = gf64_sbox(state[11]);
#else
#pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = gf64_sbox(state[i]);
    }
#endif
}

__device__ __forceinline__ void nonce_line_external_round(u64 *state, const u64 *rc12) {
    nonce_line_external_round_sboxes(state);
    ext_layer64(state, rc12);
}

__device__ __forceinline__ void nonce_line_terminal_external_rounds(u64 *state) {
#if MINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL
    nonce_line_external_round(state, RC_TERMINAL[1]);
    nonce_line_external_round(state, RC_TERMINAL[2]);
    nonce_line_external_round(state, RC_TERMINAL[3]);
    nonce_line_external_round(state, RC_TERMINAL[4]);
#else
#pragma unroll
    for (int r = 0; r < 4; r++) {
        nonce_line_external_round(state, RC_TERMINAL[r + 1]);
    }
#endif
}

/* Constants for counters sharing k: pk[0], pk[1] = lanes 0/4 after ext_layer(mid+B)+RC_INITIAL[0];
 * pk[2..] = folded round-0 on the other ten lanes + RC_INITIAL[1]. */
__device__ __forceinline__ void nonce_line_launch_params(const u64 mid[12],
                                                         const u32 ctr_hi_be[8],
                                                         u64 pk[NONCE_LINE_K_PARAMS]) {
    u32 b[8];
    nonce_line_base_words(ctr_hi_be, b);
    u64 s[12];
#pragma unroll
    for (int i = 0; i < 12; i++) {
        s[i] = mid[i];
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        s[i] = gf64_add(s[i], (u64)b[i]);
    }
    ext_layer64(s, RC_INITIAL[0]);
#pragma unroll
    for (int i = 0; i < 12; i++) {
        s[i] = gf64_canon(s[i]);
    }
    pk[0] = s[0];
    pk[1] = s[4];
    u64 kfold[12];
#pragma unroll
    for (int i = 0; i < 12; i++) {
        kfold[i] = (i == 0 || i == 4) ? 0ULL : gf64_sbox(s[i]);
    }
    ext_layer64(kfold, RC_INITIAL[1]);
#pragma unroll
    for (int i = 0; i < 12; i++) {
        pk[2 + i] = gf64_canon(kfold[i]);
    }
}

__device__ __forceinline__ void nonce_line_launch_params_from_b(const u64 mid[12], const u32 b[8],
                                                                u64 pk[NONCE_LINE_K_PARAMS]) {
    u64 s[12];
#pragma unroll
    for (int i = 0; i < 12; i++) {
        s[i] = mid[i];
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        s[i] = gf64_add(s[i], (u64)b[i]);
    }
    ext_layer64(s, RC_INITIAL[0]);
#pragma unroll
    for (int i = 0; i < 12; i++) {
        s[i] = gf64_canon(s[i]);
    }
    pk[0] = s[0];
    pk[1] = s[4];
    u64 kfold[12];
#pragma unroll
    for (int i = 0; i < 12; i++) {
        kfold[i] = (i == 0 || i == 4) ? 0ULL : gf64_sbox(s[i]);
    }
    ext_layer64(kfold, RC_INITIAL[1]);
#pragma unroll
    for (int i = 0; i < 12; i++) {
        pk[2 + i] = gf64_canon(kfold[i]);
    }
}

__device__ __forceinline__ void nonce_line_mix_cblk(u64 *dst, const u64 *pk_lane, u64 cblk) {
    const u64 twice = gf64_add(cblk, cblk);
    const u64 thrice = gf64_add(twice, cblk);
    dst[0] = gf64_add(pk_lane[0], twice);
    dst[1] = gf64_add(pk_lane[1], cblk);
    dst[2] = gf64_add(pk_lane[2], cblk);
    dst[3] = gf64_add(pk_lane[3], thrice);
}

/* State entering round 1 from pre-S-box lanes u, v (see gf64_nonce_line_uv_at). */
__device__ __forceinline__ void nonce_line_round1_state_from_uv(const u64 pk[NONCE_LINE_K_PARAMS],
                                                                u64 u, u64 v, u64 s[12]) {
    u64 f, g;
    gf64_sbox_pair(u, v, f, g);
    const u64 c2 = gf64_add(f, g);
    const u64 c0 = gf64_add(f, c2);
    const u64 c1 = gf64_add(g, c2);
    nonce_line_mix_cblk(s, pk + 2, c0);
    nonce_line_mix_cblk(s + 4, pk + 6, c1);
    nonce_line_mix_cblk(s + 8, pk + 10, c2);
}

/* State entering round 1 (round constants included) for line position t. */
__device__ __forceinline__ void nonce_line_round1_state(const u64 pk[NONCE_LINE_K_PARAMS], u32 t,
                                                        u64 s[12]) {
    u64 u, v;
    gf64_nonce_line_uv_at(pk[0], pk[1], t, u, v);
    nonce_line_round1_state_from_uv(pk, u, v, s);
}

/* Remainder of the first permutation from the round-1 state. */
__device__ __forceinline__ void nonce_line_permute_from_round1(u64 *state) {
#pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = gf64_sbox(state[i]);
    }
    /* pk[2..] already include RC_INITIAL[1]; finish round-1 linear layer without RC. */
    ext_layer64(state, RC_TERMINAL[4]);
    nonce_line_external_round(state, RC_INITIAL[2]);
    nonce_line_external_round(state, RC_INITIAL[3]);
    u64 x = gf64_sbox(gf64_add(state[0], RC_INTERNAL[0]));
    /* PILOT-002 G0-A: piped internal rounds here diverged from scalar `int_round_p` vs full hash prefix. */
#pragma unroll 1
    for (int r = 0; r < 21; r++) {
        x = gf64_sbox(int_round_p(state, x, RC_INTERNAL[r + 1]));
    }
    state[0] = int_round_p(state, x, 0ULL);
#pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = gf64_add(state[i], RC_TERMINAL[0][i]);
    }
    nonce_line_terminal_external_rounds(state);
}

__device__ __forceinline__ void nonce_line_absorb_mapped_low(u64 *state, const u32 k_base[8],
                                                             u32 t) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int vi = NONCE_LINE_K_V[i];
        const u32 w = k_base[i] + (u32)((long long)vi * (long long)t);
        state[i] = gf64_add(state[i], (u64)w);
    }
}

__device__ __forceinline__ void nonce_line_pack_first_be8(const u64 st[12], u32 *first_be) {
    u32 first[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 c = gf64_canon(st[i]);
        first[2 * i] = (u32)(c & EPS64);
        first[2 * i + 1] = (u32)(c >> 32);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        first_be[i] = bswap32(first[i]);
    }
}

#if defined(MINER_CUDA_CPP_OUT0)
/* V2_OUT0 filter shape: BE limbs 0..1 from out0 row0; limbs 2..7 zero. */
__device__ __forceinline__ void nonce_line_pack_cpp_out0_first_be8(u64 out0_canon, u32 *first_be) {
    u32 first[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        first[i] = 0u;
    }
    u64 c = gf64_canon(out0_canon);
    first[0] = (u32)(c & EPS64);
    first[1] = (u32)(c >> 32);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        first_be[i] = bswap32(first[i]);
    }
}
#endif

/* Host mirror: `goldilocks_state_after_third_permute` + capacity + scalar permute. */
__device__ __forceinline__ void nonce_line_hash_high_half_pow(const u64 mid[12],
                                                              const u32 k_base[8], u32 t,
                                                              u32 *first_be) {
    u64 st[12];
#pragma unroll
    for (int i = 0; i < 12; i++) {
        st[i] = mid[i];
    }
    nonce_line_absorb_mapped_low(st, k_base, t);
    permute64_pow(st);
    st[0] = gf64_add(st[0], 1ULL);
    st[1] = gf64_add(st[1], 1ULL);
    permute64_pow(st);
    nonce_line_pack_first_be8(st, first_be);
}

__device__ __forceinline__ void nonce_line_load_mid_from_prestate(const u32 prestate[24],
                                                                  u64 mid[12]) {
#pragma unroll
    for (int i = 0; i < 12; i++) {
        mid[i] = ((u64)prestate[2 * i + 1] << 32) | (u64)prestate[2 * i];
    }
}

__device__ __forceinline__ void nonce_line_hash_high_half_from_uv(const u64 pk[NONCE_LINE_K_PARAMS],
                                                                  u64 u, u64 v, const u64 mid[12],
                                                                  const u32 k_base[8], u32 t,
                                                                  u32 *first_be) {
#if defined(MINER_CUDA_CPP_OUT0)
    (void)mid;
    (void)k_base;
    (void)t;
    u64 st[12];
    nonce_line_round1_state_from_uv(pk, u, v, st);
    cpp_out0_second_permute_loops(st);
    const u64 o = cpp_out0_ext_row0_canon(st);
    nonce_line_pack_cpp_out0_first_be8(o, first_be);
#else
    (void)pk;
    (void)u;
    (void)v;
    nonce_line_hash_high_half_pow(mid, k_base, t, first_be);
#endif
}

__device__ __forceinline__ void nonce_line_hash_high_half_from_uv_prestate(
    const u64 pk[NONCE_LINE_K_PARAMS], u64 u, u64 v, const u32 prestate[24],
    const u32 k_base[8], u32 t, u32 *first_be) {
    u64 mid[12];
    nonce_line_load_mid_from_prestate(prestate, mid);
    nonce_line_hash_high_half_from_uv(pk, u, v, mid, k_base, t, first_be);
}

__device__ __forceinline__ void nonce_line_hash_high_half(const u64 mid[12], const u32 k_base[8],
                                                          u32 t, u32 *first_be) {
    nonce_line_hash_high_half_pow(mid, k_base, t, first_be);
}
