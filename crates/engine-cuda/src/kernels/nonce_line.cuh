/* Structured nonce search ("nonce line") — Quantus-Network/quantus-miner#110
 *
 * Reference: CPPminer nonce_line.hpp
 * https://github.com/danifest751/CPPminer/blob/release/fork/src/qpow/include/qpow/nonce_line.hpp
 *
 * Walk w = B(k) + t*v so M_ext*v is zero except lanes 0 and 4. Round 1 needs two
 * S-boxes; the other ten lanes fold into pk[14] per k (launch_params).
 */

#pragma once

#define NONCE_LINE_K_T_BITS 26
#define NONCE_LINE_K_PARAMS 14

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
    for (int i = 0; i < 8; i++) {
        b[i] = 0x80000000u | nonce_line_bits256(ctr_hi_be, NONCE_LINE_K_T_BITS + 29 * i, 29);
    }
}

__device__ __forceinline__ void nonce_line_external_round(u64 *state, const u64 *rc12) {
#pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = gf64_sbox(state[i]);
    }
    ext_layer64(state, rc12);
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
    const u64 f = gf64_sbox(u);
    const u64 g = gf64_sbox(v);
    const u64 c2 = gf64_add(f, g);
    const u64 c0 = gf64_add(c2, f);
    const u64 c1 = gf64_add(c2, g);
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
#pragma unroll 1
    for (int r = 0; r < 21; r++) {
        x = gf64_sbox(int_round_p(state, x, RC_INTERNAL[r + 1]));
    }
    state[0] = int_round_p(state, x, 0ULL);
#pragma unroll
    for (int i = 0; i < 12; i++) {
        state[i] = gf64_add(state[i], RC_TERMINAL[0][i]);
    }
#pragma unroll 1
    for (int r = 0; r < 4; r++) {
        nonce_line_external_round(state, RC_TERMINAL[r + 1]);
    }
}

__device__ __forceinline__ void nonce_line_hash_high_half_from_uv(const u64 pk[NONCE_LINE_K_PARAMS],
                                                                  u64 u, u64 v, u32 *first_be) {
    u64 st[12];
    nonce_line_round1_state_from_uv(pk, u, v, st);
    nonce_line_permute_from_round1(st);
    st[0] = gf64_add(st[0], 1ULL);
    st[1] = gf64_add(st[1], 1ULL);
    permute64(st);
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

__device__ __forceinline__ void nonce_line_hash_high_half(const u64 pk[NONCE_LINE_K_PARAMS], u32 t,
                                                          u32 *first_be) {
    u64 u, v;
    gf64_nonce_line_uv_at(pk[0], pk[1], t, u, v);
    nonce_line_hash_high_half_from_uv(pk, u, v, first_be);
}
