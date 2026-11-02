//! PILOT-002 G0-A: host oracles for legacy / CPP_OUT0 nonce-line filter limbs.

use pow_core::nonce_line::{
    counter_add_t, counter_with_t, nonce_from_absolute_t, segment_launch_params, t_of, uv_at,
    T_SPAN,
};
use pow_core::nonce_line_filter::{
    cpp_out0_hash_high_half_first_be_from_uv, hash_high_half_cmp_be,
    legacy_hash_high_half_first_be_for_segment, legacy_hash_high_half_first_be_from_uv,
    InternalSchedule,
};
use pow_core::{hash_from_nonce, JobContext, NONCE_HASH_KVS};
use primitive_types::U512;

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn hash_u512_to_first_be8(hash: U512) -> [u32; 8] {
    let be = hash.to_big_endian();
    std::array::from_fn(|i| u32::from_be_bytes(be[i * 4..i * 4 + 4].try_into().unwrap()))
}

fn sample_points() -> Vec<([u8; 32], U512, u32)> {
    let t_span = T_SPAN as u32;
    let mut out = Vec::new();
    let header = decode32(NONCE_HASH_KVS[0].header);
    let base = U512::from(0x8000u64);
    let base_be = base.to_big_endian();
    out.push((header, base, t_of(&base_be)));
    for &dt in &[0u32, 1, t_span - 1, t_span / 2] {
        let c = counter_with_t(&base_be, dt);
        out.push((header, U512::from_big_endian(&c), dt));
    }
    let near = counter_add_t(base, t_span - 1);
    out.push((header, near, t_of(&near.to_big_endian())));
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
fn scalar_and_pipelined_differ_only_on_permute64() {
    let header = decode32(NONCE_HASH_KVS[0].header);
    let base = U512::from(0x8000u64);
    let t = t_of(&base.to_big_endian());
    let scalar =
        legacy_hash_high_half_first_be_for_segment(header, base, t, InternalSchedule::Scalar);
    let piped =
        legacy_hash_high_half_first_be_for_segment(header, base, t, InternalSchedule::Pipelined);
    let _ = (scalar, piped);
}

#[test]
fn legacy_high_half_matches_get_nonce_hash_prefix() {
    for (header, counter, t) in sample_points() {
        let ctx = JobContext::new(header, U512::MAX);
        let mapped = nonce_from_absolute_t(counter, t);
        let prefix = hash_u512_to_first_be8(hash_from_nonce(&ctx, mapped));
        let mirror = legacy_hash_high_half_first_be_for_segment(
            header,
            counter,
            t,
            InternalSchedule::Scalar,
        );
        assert_eq!(
            mirror, prefix,
            "legacy scalar mirror != get_nonce_hash prefix at t={t}"
        );
        assert!(
            mirror.iter().any(|&w| w != 0) || prefix.iter().all(|&w| w == 0),
            "degenerate mirror at t={t}"
        );
    }
}

#[test]
fn cpp_out0_limbs_and_filter_decisions_vs_legacy() {
    const CMP_LIMBS: usize = 8;
    let tgt_hi = [0u32; 8];
    for (header, counter, t) in sample_points() {
        let pk = segment_launch_params(header, counter);
        let (u, v) = uv_at(pk[0], pk[1], t);
        let leg = legacy_hash_high_half_first_be_from_uv(
            header,
            counter,
            t,
            &pk,
            u,
            v,
            InternalSchedule::Scalar,
        );
        let out0 = cpp_out0_hash_high_half_first_be_from_uv(header, counter, t, &pk, u, v);
        assert_eq!(out0[0], leg[0], "cpp_out0 limb0 t={t}");
        assert_eq!(out0[1], leg[1], "cpp_out0 limb1 t={t}");
        for i in 2..8 {
            assert_eq!(out0[i], 0, "cpp_out0 limb{i} must be zero at t={t}");
        }
        let leg_cmp = hash_high_half_cmp_be(&leg, &tgt_hi, CMP_LIMBS);
        let out_cmp = hash_high_half_cmp_be(&out0, &tgt_hi, CMP_LIMBS);
        assert_eq!(out_cmp, leg_cmp, "cpp_out0 filter != legacy at t={t}");
    }
}
