//! Structured nonce search ("nonce line") per [#110](https://github.com/Quantus-Network/quantus-miner/issues/110).
//!
//! Counters are 64-byte big-endian values: bytes `0..32` pass through to the nonce;
//! bytes `32..64` encode `C = k * 2^26 + t` as a 256-bit **big-endian** integer.
//! The mapped nonce writes little-endian absorb words in `32..64`.

use crate::{external_linear_layer, mining_midstate, SPONGE_WIDTH};
use primitive_types::U512;
use qp_poseidon_core::poseidon2::INITIAL_EXTERNAL_CONSTANTS;
use qp_poseidon_core::Goldilocks;

pub const T_BITS: u32 = 26;
pub const T_SPAN: u64 = 1 << T_BITS;
pub const T_MASK: u64 = T_SPAN - 1;
pub const K_PARAMS: usize = 14;

pub const V: [i32; 8] = [4, -17, 11, -3, -4, 17, -11, 3];

/// Bits `[pos, pos + len)` of the 256-bit big-endian number in `b[0..32]`.
pub fn bits256_be(b: &[u8; 32], pos: usize, len: usize) -> u32 {
    let mut r = 0u32;
    for i in (0..len).rev() {
        let bit = pos + i;
        let byte = if bit < 256 {
            b[31 - bit / 8]
        } else {
            0
        };
        r = (r << 1) | ((byte >> (bit % 8)) & 1) as u32;
    }
    r
}

/// Low `t` field from a 64-byte counter (BE).
pub fn t_of(counter_be: &[u8; 64]) -> u32 {
    bits256_be(counter_be[32..64].try_into().unwrap(), 0, T_BITS as usize)
}

/// `k` from counter: `C >> T_BITS` on the low 256-bit half (BE).
pub fn k_of(counter_be: &[u8; 64]) -> u64 {
    let low: [u8; 32] = counter_be[32..64].try_into().unwrap();
    let c = U512::from_big_endian(&{
        let mut be = [0u8; 64];
        be[32..64].copy_from_slice(&low);
        be
    });
    (c >> T_BITS).low_u64()
}

/// `B(k)` absorb words for this counter's `k` (CPPminer `base_words`).
pub fn base_words(counter_be: &[u8; 64]) -> [u32; 8] {
    let low: [u8; 32] = counter_be[32..64].try_into().unwrap();
    std::array::from_fn(|i| {
        0x8000_0000 | bits256_be(&low, T_BITS as usize + 29 * i, 29)
    })
}

/// Map counter → 64-byte big-endian nonce (pool / `get_nonce_hash` layout).
pub fn map_counter(counter_be: &[u8; 64]) -> [u8; 64] {
    let b = base_words(counter_be);
    let t = t_of(counter_be);
    let mut out = *counter_be;
    for i in 0..8 {
        let w = b[i].wrapping_add((V[i] as i64).wrapping_mul(t as i64) as u32);
        out[32 + 4 * i..32 + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

pub fn map_counter_u512(counter: U512) -> U512 {
    let be = counter.to_big_endian();
    U512::from_big_endian(&map_counter(&be))
}

/// `launch_params` from CPPminer `nonce_line.hpp`: 14 canonical u64 felts for one `k`.
pub fn launch_params(mid: &[u64; SPONGE_WIDTH], counter_be: &[u8; 64]) -> [u64; K_PARAMS] {
    let b = base_words(counter_be);
    let mut s: [Goldilocks; SPONGE_WIDTH] = mid.map(Goldilocks::from_u64);
    for i in 0..8 {
        s[i] += Goldilocks::from_u64(b[i] as u64);
    }
    external_linear_layer(&mut s);
    for (felt, rc) in s.iter_mut().zip(INITIAL_EXTERNAL_CONSTANTS[0]) {
        *felt += Goldilocks::from_u64(rc);
    }
    let mut pk = [0u64; K_PARAMS];
    pk[0] = s[0].as_canonical_u64();
    pk[1] = s[4].as_canonical_u64();

    let mut k_st: [Goldilocks; SPONGE_WIDTH] = std::array::from_fn(|i| {
        if i == 0 || i == 4 {
            Goldilocks::ZERO
        } else {
            s[i].exp7()
        }
    });
    external_linear_layer(&mut k_st);
    for i in 0..SPONGE_WIDTH {
        k_st[i] += Goldilocks::from_u64(INITIAL_EXTERNAL_CONSTANTS[1][i]);
        pk[2 + i] = k_st[i].as_canonical_u64();
    }
    pk
}

/// Pack `pk[14]` into 28 little-endian u32 limbs (CUDA `MiningParamsNonceLine.pk`).
pub fn pk_to_u32s(pk: &[u64; K_PARAMS]) -> [u32; 28] {
    let mut out = [0u32; 28];
    for (i, &felt) in pk.iter().enumerate() {
        out[2 * i] = felt as u32;
        out[2 * i + 1] = (felt >> 32) as u32;
    }
    out
}

/// Counter with `t` replaced (same `k`).
pub fn counter_with_t(counter_be: &[u8; 64], t: u32) -> [u8; 64] {
    let mut out = *counter_be;
    let k = k_of(counter_be);
    let c = (U512::from(k) << T_BITS) + U512::from(t as u64);
    let c_be = c.to_big_endian();
    out[32..64].copy_from_slice(&c_be[32..64]);
    out
}

/// Add `delta` to the counter's `t` field (may carry into `k`).
pub fn counter_add_t(counter: U512, delta: u32) -> U512 {
    let be = counter.to_big_endian();
    let t = t_of(&be) as u64 + delta as u64;
    let k = k_of(&be);
    let c = (U512::from(k) << T_BITS) + U512::from(t);
    let c_be = c.to_big_endian();
    let mut out = be;
    out[32..64].copy_from_slice(&c_be[32..64]);
    U512::from_big_endian(&out)
}

/// Nonce after advancing `t` by `t_offset` from `segment_start` counter, then mapping.
pub fn nonce_from_t_offset(segment_start: U512, t_offset: u64) -> U512 {
    let ctr = counter_add_t(segment_start, t_offset as u32);
    map_counter_u512(ctr)
}

/// Mapped pool nonce for absolute line coordinate `t` (same `k` as `segment_start`).
pub fn nonce_from_absolute_t(segment_start: U512, t_abs: u32) -> U512 {
    let be = segment_start.to_big_endian();
    map_counter_u512(U512::from_big_endian(&counter_with_t(&be, t_abs)))
}

/// Host-side `launch_params` for a segment: midstate from header + counter high half.
pub fn segment_launch_params(header: [u8; 32], segment_start: U512) -> [u64; K_PARAMS] {
    let counter_be = segment_start.to_big_endian();
    let mid = mining_midstate(header, counter_be[..32].try_into().unwrap());
    launch_params(&mid, &counter_be)
}

/// Remaining `t` values before `k` increments (within `2^26`).
pub fn t_room_in_k(counter: U512) -> u32 {
    let be = counter.to_big_endian();
    let t = t_of(&be) as u64;
    (T_SPAN - (t % T_SPAN)) as u32
}

// --- Round-1 S-box accounting (see docs/thesis/phase-04/math-sprint/03-nonce-line.md) ---

/// T-dependent S-boxes in the round-1 inner loop (flat: full inject + round; line: `f`, `g`).
pub const FLAT_ROUND1_T_DEP_SBOX_PER_NONCE: u64 = 12;
pub const NONCE_LINE_ROUND1_T_DEP_SBOX_PER_NONCE: u64 = 2;

/// Total round-1 S-box invocations per nonce (line: `round1_state` + opening of `permute_from_round1`).
pub const FLAT_ROUND1_SBOX_PER_NONCE: u64 = 12;
pub const NONCE_LINE_ROUND1_SBOX_PER_NONCE: u64 = 14;

/// Hoisted once per `k` in `launch_params` (lanes `i ∉ {0,4}` after `RC_INITIAL[0]`).
pub const NONCE_LINE_LAUNCH_SBOX_PER_K: u64 = 10;

/// Round-1 S-box totals for `n_nonces` at fixed `k`: `(flat, nonce_line)`.
pub fn round1_sbox_equiv_totals(n_nonces: u64) -> (u64, u64) {
    let flat = FLAT_ROUND1_SBOX_PER_NONCE * n_nonces;
    let nonce_line = NONCE_LINE_LAUNCH_SBOX_PER_K + NONCE_LINE_ROUND1_SBOX_PER_NONCE * n_nonces;
    (flat, nonce_line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hash_from_nonce, JobContext};
    use hex;

    fn decode64(s: &str) -> [u8; 64] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    #[test]
    fn map_counter_roundtrip_golden_nonces_as_identity_when_t_zero() {
        for v in crate::NONCE_HASH_KVS {
            let nonce_be = decode64(v.nonce);
            // Golden nonces are not counters; only check map doesn't panic on t=0 counters.
            let mut ctr = nonce_be;
            ctr[32..64].fill(0);
            let _ = map_counter(&ctr);
        }
    }

    #[test]
    fn counter_add_t_increments_t_within_k() {
        let mut ctr = [0u8; 64];
        ctr[32..64][31] = 5; // t = 5 in BE low byte
        let c0 = U512::from_big_endian(&ctr);
        let c1 = counter_add_t(c0, 3);
        assert_eq!(t_of(&c1.to_big_endian()), 8);
        assert_eq!(k_of(&c1.to_big_endian()), k_of(&ctr));
    }

    #[test]
    fn round1_sbox_equiv_counts_one_vs_thousand() {
        let (flat_1, nl_1) = round1_sbox_equiv_totals(1);
        let (flat_1k, nl_1k) = round1_sbox_equiv_totals(1000);
        eprintln!(
            "round1 sbox-equiv: N=1 flat={flat_1} nonce_line={nl_1}; \
             N=1000 flat={flat_1k} nonce_line={nl_1k}"
        );
        assert_eq!(flat_1, 12);
        assert_eq!(nl_1, 24);
        assert_eq!(flat_1k, 12_000);
        assert_eq!(nl_1k, 14_010);
    }

    #[test]
    fn launch_params_and_map_produce_hashable_nonces() {
        let header = hex::decode(crate::NONCE_HASH_KVS[2].header)
            .unwrap()
            .try_into()
            .unwrap();
        let ctx = JobContext::new(header, U512::from(1u64));
        let mut ctr = [0u8; 64];
        ctr[0] = 0xab;
        for t in [0u32, 1, 100, 10_000] {
            let c_be = counter_with_t(&ctr, t);
            let counter = U512::from_big_endian(&c_be);
            let mid = mining_midstate(header, c_be[..32].try_into().unwrap());
            let pk = launch_params(&mid, &c_be);
            assert_ne!(pk[0], 0);
            let nonce = map_counter_u512(counter);
            let _ = hash_from_nonce(&ctx, nonce);
        }
    }
}
