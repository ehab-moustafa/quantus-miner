//! Compare reference PoW hashing vs midstate-resume + incremental nonce absorb.
//!
//! Run: `cargo run -p pow-core --release --example math_nonce_delta_bench`

use pow_core::{hash_from_nonce, mining_midstate, JobContext};
use primitive_types::U512;
use qp_poseidon_core::{Goldilocks, Poseidon2, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

const ITERS: u64 = 10_000;

fn absorb_low_nonce_words(nonce_be: &[u8; 64]) -> [Goldilocks; 8] {
    std::array::from_fn(|i| {
        let off = 32 + 4 * i;
        Goldilocks::from_u64(u32::from_le_bytes(nonce_be[off..off + 4].try_into().unwrap()) as u64)
    })
}

/// Big-endian +1 on the 512-bit nonce (same layout as `U512::to_big_endian`).
fn increment_nonce_be(nonce_be: &mut [u8; 64]) {
    for i in (0..64).rev() {
        nonce_be[i] = nonce_be[i].wrapping_add(1);
        if nonce_be[i] != 0 {
            return;
        }
    }
}

/// After `increment_nonce_be`, patch only absorb limbs whose u32 encoding changed.
fn increment_absorb_chain(absorb: &mut [Goldilocks; 8], prev_be: &[u8; 64], next_be: &[u8; 64]) {
    for (i, felt) in absorb.iter_mut().enumerate() {
        let off = 32 + 4 * i;
        if prev_be[off..off + 4] != next_be[off..off + 4] {
            *felt += Goldilocks::from_u64(u32::from_le_bytes(
                next_be[off..off + 4].try_into().unwrap(),
            ) as u64);
            *felt -= Goldilocks::from_u64(u32::from_le_bytes(
                prev_be[off..off + 4].try_into().unwrap(),
            ) as u64);
        }
    }
}

fn hash_from_midstate(
    poseidon: &Poseidon2,
    mid: &[Goldilocks; SPONGE_WIDTH],
    absorb: &[Goldilocks; 8],
) -> U512 {
    let mut state = *mid;
    for (felt, &add) in state.iter_mut().zip(absorb) {
        *felt += add;
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

fn bench_reference(ctx: &JobContext, start: U512) -> (u128, U512) {
    let t0 = Instant::now();
    let mut nonce = start;
    let mut last = U512::zero();
    for _ in 0..ITERS {
        last = hash_from_nonce(ctx, nonce);
        black_box(last);
        nonce = nonce.saturating_add(U512::one());
    }
    let ns = t0.elapsed().as_nanos();
    (ns / ITERS as u128, last)
}

fn bench_incremental(ctx: &JobContext, start: U512) -> (u128, U512) {
    let mut nonce_be = start.to_big_endian();
    let high: [u8; 32] = nonce_be[..32].try_into().unwrap();
    let mid_u64 = mining_midstate(ctx.header, high);
    let mid: [Goldilocks; SPONGE_WIDTH] = mid_u64.map(Goldilocks::from_u64);
    let poseidon = Poseidon2::new();
    let mut absorb = absorb_low_nonce_words(&nonce_be);

    let t0 = Instant::now();
    let mut last = U512::zero();
    for _ in 0..ITERS {
        last = hash_from_midstate(&poseidon, &mid, &absorb);
        black_box(last);
        let prev = nonce_be;
        increment_nonce_be(&mut nonce_be);
        increment_absorb_chain(&mut absorb, &prev, &nonce_be);
    }
    let ns = t0.elapsed().as_nanos();
    (ns / ITERS as u128, last)
}

fn main() {
    let header = [0x2a; 32];
    let ctx = JobContext::new(header, U512::from(1u64));
    let start = U512::from(0x1_0000_0000u64); // high half stable for 10k +1 steps

    let (ns_a, ref_a) = bench_reference(&ctx, start);
    let (ns_b, ref_b) = bench_incremental(&ctx, start);

    assert_eq!(ref_a, ref_b, "incremental path must match hash_from_nonce");

    let ratio = ns_a as f64 / ns_b as f64;
    println!("hash_from_nonce (A):     {ns_a} ns/nonce");
    println!("midstate + absorb (B):     {ns_b} ns/nonce");
    println!("A/B ratio:                 {ratio:.3}x");
}
