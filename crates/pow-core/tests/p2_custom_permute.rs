//! Phase 2: custom permute and field mul match the library, and midstate still matches golden vectors.

use pow_core::custom::{self, fast_mul};
use pow_core::NONCE_HASH_KVS;
use qp_poseidon_core::{Goldilocks, Poseidon2, SPONGE_WIDTH};
use std::time::Instant;

fn canon(state: &[Goldilocks; SPONGE_WIDTH]) -> [u64; SPONGE_WIDTH] {
    state.map(|g| g.as_canonical_u64())
}

#[test]
fn custom_permute_matches_library_on_random_states() {
    let poseidon = Poseidon2::new();
    let mut seed = 0xC0FFEE_u64;
    for _ in 0..64 {
        let mut state = [Goldilocks::ZERO; SPONGE_WIDTH];
        for lane in &mut state {
            seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *lane = Goldilocks::from_u64(seed);
        }
        let mut lib = state;
        let mut fast = state;
        poseidon.permute_mut(&mut lib);
        custom::permute_mut(&mut fast);
        assert_eq!(canon(&lib), canon(&fast));
    }
}

#[test]
fn custom_midstate_matches_golden_vectors() {
    for kv in NONCE_HASH_KVS {
        let header: [u8; 32] = hex::decode(kv.header).unwrap().try_into().unwrap();
        let nonce: [u8; 64] = hex::decode(kv.nonce).unwrap().try_into().unwrap();
        let mid = pow_core::mining_midstate(header, nonce[..32].try_into().unwrap());
        assert_eq!(mid, kv.mid, "midstate drifted for header {}", kv.header);
    }
}

#[test]
fn fast_mul_matches_goldilocks_edges() {
    let samples = [0u64, 1, 2, u32::MAX as u64, u64::MAX, 0xFFFF_FFFF_0000_0001];
    for &a in &samples {
        for &b in &samples {
            let expect = (Goldilocks::from_u64(a) * Goldilocks::from_u64(b)).as_canonical_u64();
            let got = Goldilocks::from_u64(fast_mul(a, b)).as_canonical_u64();
            assert_eq!(got, expect, "{a} * {b}");
        }
    }
}

#[test]
fn custom_permute_is_not_slower_than_a_sanity_budget() {
    let mut state = [Goldilocks::from_u64(1); SPONGE_WIDTH];
    let start = Instant::now();
    for _ in 0..2_000 {
        custom::permute_mut(&mut state);
    }
    let ns = start.elapsed().as_nanos() / 2_000;
    eprintln!("custom permute {ns} ns/op");
    assert!(ns < 50_000, "permute unexpectedly slow: {ns} ns");
}
