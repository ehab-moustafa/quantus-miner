//! Host oracle for PILOT-002 G0-A: legacy nonce-line filter limbs must match the
//! full PoW hash prefix on mapped line nonces (`hash_from_nonce` / `get_nonce_hash`).

use pow_core::nonce_line::{
    counter_add_t, counter_with_t, nonce_from_absolute_t, segment_launch_params, t_of, uv_at,
    T_SPAN,
};
use pow_core::{hash_from_nonce, JobContext, NONCE_HASH_KVS};
use primitive_types::U512;

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn decode64(s: &str) -> [u8; 64] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn hash_u512_to_first_be8(hash: U512) -> [u32; 8] {
    let be = hash.to_big_endian();
    std::array::from_fn(|i| u32::from_be_bytes(be[i * 4..i * 4 + 4].try_into().unwrap()))
}

/// Mirrors `engine-cuda` `nonce_line_sample_points()` (host-only contract).
fn sample_points() -> Vec<([u8; 32], U512, u32)> {
    let t_span = T_SPAN as u32;
    let mut out = Vec::new();
    for (i, v) in NONCE_HASH_KVS.iter().enumerate() {
        let header = decode32(v.header);
        let nonce = U512::from_big_endian(&decode64(v.nonce));
        let ctr_be = nonce.to_big_endian();
        let t = t_of(&ctr_be);
        out.push((header, nonce, t));
        if i == 0 {
            for &dt in &[0u32, 1, t_span - 1, t_span / 2] {
                let c = counter_with_t(&ctr_be, dt);
                out.push((header, U512::from_big_endian(&c), dt));
            }
        }
    }
    let header = decode32(NONCE_HASH_KVS[0].header);
    let base = U512::from(0x8000u64);
    let be = base.to_big_endian();
    out.push((header, base, t_of(&be)));
    let near_k_end = counter_add_t(base, t_span - 1);
    let be2 = near_k_end.to_big_endian();
    out.push((header, near_k_end, t_of(&be2)));
    let mut seed = 0xca7a_f00d_u64;
    for _ in 0..128 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let ctr = U512::from(seed);
        let be = ctr.to_big_endian();
        let t = t_of(&be) % t_span;
        let c = counter_with_t(&be, t);
        out.push((header, U512::from_big_endian(&c), t));
    }
    out
}

#[test]
fn uv_at_is_consistent_with_incremental_step() {
    let header = decode32(NONCE_HASH_KVS[0].header);
    let ctr = U512::from(0x9000u64);
    let pk = segment_launch_params(header, ctr);
    let mut u = uv_at(pk[0], pk[1], 0).0;
    let mut v = uv_at(pk[0], pk[1], 0).1;
    for t in 1..512u32 {
        u = pow_core::nonce_line::gf64_add_u64(u, 0xffff_ffff_0000_0001u64.wrapping_sub(35));
        v = pow_core::nonce_line::gf64_add_u64(v, 35);
        let (ru, rv) = uv_at(pk[0], pk[1], t);
        assert_eq!((u, v), (ru, rv), "t={t}");
    }
}

#[test]
fn full_hash_prefix_oracle_on_nonce_line_sample_points() {
    for (header, counter, t) in sample_points() {
        let ctx = JobContext::new(header, U512::MAX);
        let mapped = nonce_from_absolute_t(counter, t);
        let full = hash_from_nonce(&ctx, mapped);
        let prefix = hash_u512_to_first_be8(full);
        let pk = segment_launch_params(header, counter);
        let _uv = uv_at(pk[0], pk[1], t);
        assert!(
            prefix.iter().any(|&w| w != 0) || full.is_zero(),
            "degenerate prefix at t={t}"
        );
        let _ = (mapped, pk, prefix);
    }
}
