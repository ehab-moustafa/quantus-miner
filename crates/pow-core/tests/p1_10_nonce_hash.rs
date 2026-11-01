//! Phase 1 agent 10: `qpow_math::get_nonce_hash` vs midstate-resume (3 permutes).

use pow_core::{mining_midstate, JobContext, NONCE_HASH_KVS, SPONGE_WIDTH};
use primitive_types::U512;
use qp_poseidon_core::{Goldilocks, Poseidon2};
use std::hint::black_box;
use std::time::Instant;

/// Reference: `qpow-math` `get_nonce_hash` (five permutations via `hash_squeeze_twice`).
#[inline(always)]
fn old_nonce_hash(header: [u8; 32], nonce_be: [u8; 64]) -> U512 {
    qpow_math::get_nonce_hash(header, nonce_be)
}

/// Fast path: sponge after header + nonce high half, then three permutations per nonce.
fn new_nonce_hash(
    poseidon: &Poseidon2,
    header: [u8; 32],
    nonce_be: [u8; 64],
    mid_cache: &mut Option<([u8; 32], [Goldilocks; SPONGE_WIDTH])>,
) -> U512 {
    let high: [u8; 32] = nonce_be[..32].try_into().unwrap();
    let mut state: [Goldilocks; SPONGE_WIDTH] = match mid_cache {
        Some((cached_high, state)) if *cached_high == high => *state,
        _ => {
            let state = mining_midstate(header, high).map(Goldilocks::from_u64);
            *mid_cache = Some((high, state));
            state
        }
    };
    for (felt, chunk) in state.iter_mut().zip(nonce_be[32..].chunks_exact(4)) {
        *felt += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
    poseidon.permute_mut(&mut state);
    state[0] += Goldilocks::ONE;
    state[1] += Goldilocks::ONE;
    poseidon.permute_mut(&mut state);

    let mut hash_bytes = [0u8; 64];
    for (felt, chunk) in state.iter().zip(hash_bytes[..32].chunks_exact_mut(8)) {
        chunk.copy_from_slice(&felt.as_canonical_u64().to_le_bytes());
    }
    poseidon.permute_mut(&mut state);
    for (felt, chunk) in state.iter().zip(hash_bytes[32..].chunks_exact_mut(8)) {
        chunk.copy_from_slice(&felt.as_canonical_u64().to_le_bytes());
    }
    U512::from_big_endian(&hash_bytes)
}

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn decode64(s: &str) -> [u8; 64] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn assert_parity(header: [u8; 32], nonce_be: [u8; 64]) {
    let old = old_nonce_hash(header, nonce_be);
    let poseidon = Poseidon2::new();
    let mut cache = None;
    let new = new_nonce_hash(&poseidon, header, nonce_be, &mut cache);
    assert_eq!(old, new, "midstate path must match get_nonce_hash");
}

#[test]
fn p1_10_nonce_hash_parity_and_bench() {
    for (i, kv) in NONCE_HASH_KVS.iter().enumerate() {
        let header = decode32(kv.header);
        let nonce_be = decode64(kv.nonce);
        assert_parity(header, nonce_be);
        let want = decode64(kv.hash);
        assert_eq!(
            old_nonce_hash(header, nonce_be).to_big_endian(),
            want,
            "kv {i} golden hash"
        );
    }

    let header = [0x2a; 32];
    let start = U512::from(0x1_0000_0000u64);
    let mut nonce = start;
    for step in 0..1000 {
        let nonce_be = nonce.to_big_endian();
        assert_parity(header, nonce_be);
        nonce = nonce.saturating_add(U512::one());
        black_box(step);
    }

    const ITERS: u64 = 10_000;
    let ctx = JobContext::new(header, U512::from(1u64));
    let bench_start = start;

    let t0 = Instant::now();
    let mut acc_old = U512::zero();
    let mut nonce = bench_start;
    for _ in 0..ITERS {
        let nonce_be = nonce.to_big_endian();
        acc_old = black_box(old_nonce_hash(ctx.header, nonce_be));
        nonce = nonce.saturating_add(U512::one());
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t1 = Instant::now();
    let mut acc_new = U512::zero();
    let poseidon = Poseidon2::new();
    let mut mid_cache = None;
    nonce = bench_start;
    for _ in 0..ITERS {
        let nonce_be = nonce.to_big_endian();
        acc_new = black_box(new_nonce_hash(
            &poseidon,
            ctx.header,
            nonce_be,
            &mut mid_cache,
        ));
        nonce = nonce.saturating_add(U512::one());
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;

    assert_eq!(acc_old, acc_new, "timing loops must end on same hash");
    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.3} NEW_NS {new_ns:.3} RATIO {ratio:.4}");
}
