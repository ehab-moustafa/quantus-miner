#pragma once

#ifndef MINER_CUDA_CMP_LIMBS
#define MINER_CUDA_CMP_LIMBS 8
#endif

__device__ __forceinline__ void state_to_first_limbs(const u64 *st, u32 first[8]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 c = gf64_canon(st[i]);
        first[2 * i] = (u32)(c & EPS64);
        first[2 * i + 1] = (u32)(c >> 32);
    }
}

// Returns 0 if hash <= target on compared limbs, 1 if hash > target (reject), 2 if hash < target (candidate).
__device__ __forceinline__ u32 hash_high_half_cmp(const u32 first[8], const u32 *tgt_hi) {
    u32 cmp = 0u;
    const int n = MINER_CUDA_CMP_LIMBS;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i >= n) {
            break;
        }
        u32 h = bswap32(first[i]);
        u32 t = tgt_hi[7 - i];
        if (h != t) {
            cmp = (h > t) ? 1u : 2u;
            break;
        }
    }
    if (cmp == 0u && n < 8) {
#pragma unroll
        for (int i = n; i < 8; i++) {
            u32 h = bswap32(first[i]);
            u32 t = tgt_hi[7 - i];
            if (h != t) {
                cmp = (h > t) ? 1u : 2u;
                break;
            }
        }
    }
    return cmp;
}

// `first_be[i]` already bswap32(first[i]) — nonce_line path.
__device__ __forceinline__ u32 hash_high_half_cmp_be(const u32 first_be[8], const u32 *tgt_hi) {
    u32 cmp = 0u;
    const int n = MINER_CUDA_CMP_LIMBS;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i >= n) {
            break;
        }
        u32 h = first_be[i];
        u32 t = tgt_hi[7 - i];
        if (h != t) {
            cmp = (h > t) ? 1u : 2u;
            break;
        }
    }
    if (cmp == 0u && n < 8) {
#pragma unroll
        for (int i = n; i < 8; i++) {
            u32 h = first_be[i];
            u32 t = tgt_hi[7 - i];
            if (h != t) {
                cmp = (h > t) ? 1u : 2u;
                break;
            }
        }
    }
    return cmp;
}
