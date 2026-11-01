//! Phase 1 agent 11: `is_valid_nonce` — qpow_math vs MSB-first U512 limb compare.

use pow_core::NONCE_HASH_KVS;
use primitive_types::U512;
use qpow_math::{get_nonce_hash, is_valid_nonce};
use std::hint::black_box;
use std::time::Instant;

#[inline(always)]
fn old_verify(block_hash: [u8; 32], nonce: [u8; 64], difficulty: U512) -> (bool, U512) {
    is_valid_nonce(block_hash, nonce, difficulty)
}

/// Strict `hash < target` with u64 limbs compared MSB → LSB (index 7 → 0).
#[inline(always)]
fn hash_lt_target_msb_first(hash: U512, target: U512) -> bool {
    let h = hash.0;
    let t = target.0;
    for i in (0..8).rev() {
        if h[i] != t[i] {
            return h[i] < t[i];
        }
    }
    false
}

#[inline(always)]
fn fast_verify(block_hash: [u8; 32], nonce: [u8; 64], difficulty: U512) -> (bool, U512) {
    if difficulty.is_zero() {
        return (false, U512::zero());
    }
    let hash = get_nonce_hash(block_hash, nonce);
    let target = U512::MAX / difficulty;
    (hash_lt_target_msb_first(hash, target), hash)
}

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn decode64(s: &str) -> [u8; 64] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn assert_verify_parity(block_hash: [u8; 32], nonce: [u8; 64], difficulty: U512) {
    let (valid_old, hash_old) = old_verify(block_hash, nonce, difficulty);
    let (valid_new, hash_new) = fast_verify(block_hash, nonce, difficulty);
    assert_eq!(valid_old, valid_new, "valid flag mismatch");
    assert_eq!(hash_old, hash_new, "hash mismatch");
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_nonzero_difficulty(rng: &mut u64) -> U512 {
    loop {
        let mut limbs = [0u64; 8];
        for limb in &mut limbs {
            *limb = splitmix64(rng);
        }
        let d = U512(limbs);
        if !d.is_zero() {
            return d;
        }
    }
}

#[test]
fn p1_11_verify_parity_and_bench() {
    for v in NONCE_HASH_KVS {
        let header = decode32(v.header);
        let nonce = decode64(v.nonce);
        for difficulty in [
            U512::one(),
            U512::from(256u64),
            U512::from(1_000_000u64),
            U512::MAX,
        ] {
            assert_verify_parity(header, nonce, difficulty);
        }
    }

    let mut rng = 0xC0FFEE_u64;
    for _ in 0..200 {
        let mut header = [0u8; 32];
        let mut nonce = [0u8; 64];
        for b in header.iter_mut() {
            *b = splitmix64(&mut rng) as u8;
        }
        for b in nonce.iter_mut() {
            *b = splitmix64(&mut rng) as u8;
        }
        let difficulty = random_nonzero_difficulty(&mut rng);
        assert_verify_parity(header, nonce, difficulty);
    }

    const ITERS: u32 = 20_000;
    let header = decode32(NONCE_HASH_KVS[2].header);
    let difficulty = U512::from(65_536u64);
    let mut nonce = decode64(NONCE_HASH_KVS[2].nonce);
    let mut acc_old = 0u8;

    let t0 = Instant::now();
    for i in 0..ITERS {
        nonce[63] = (i & 0xFF) as u8;
        nonce[62] = ((i >> 8) & 0xFF) as u8;
        let (valid, hash) = old_verify(black_box(header), black_box(nonce), black_box(difficulty));
        acc_old ^= u8::from(valid);
        acc_old ^= hash.0[0] as u8;
    }
    let old_ns = t0.elapsed().as_nanos();

    nonce = decode64(NONCE_HASH_KVS[2].nonce);
    let mut acc_new = 0u8;
    let t1 = Instant::now();
    for i in 0..ITERS {
        nonce[63] = (i & 0xFF) as u8;
        nonce[62] = ((i >> 8) & 0xFF) as u8;
        let (valid, hash) = fast_verify(black_box(header), black_box(nonce), black_box(difficulty));
        acc_new ^= u8::from(valid);
        acc_new ^= hash.0[0] as u8;
    }
    let new_ns = t1.elapsed().as_nanos();

    assert_eq!(acc_old, acc_new, "timing loops diverged");
    let ratio = old_ns as f64 / new_ns as f64;
    println!("OLD_NS {old_ns} NEW_NS {new_ns} RATIO {ratio:.4}");
    black_box(acc_new);
}
