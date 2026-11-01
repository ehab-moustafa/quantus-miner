//! Phase 1 agent 07: one Poseidon2 external (full) round — qp-poseidon-core 3.1.0.
//!
//! Library `external_round`: add RC, S-box all 12 lanes, external linear layer.
//! `fast_round` is the same schedule with inlined S-box / MDS.

use qp_poseidon_core::poseidon2::{INITIAL_EXTERNAL_CONSTANTS, TERMINAL_EXTERNAL_CONSTANTS};
use qp_poseidon_core::{Goldilocks, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

fn rc_table(round: usize) -> [Goldilocks; SPONGE_WIDTH] {
    core::array::from_fn(|i| Goldilocks::new(INITIAL_EXTERNAL_CONSTANTS[round][i]))
}

fn rc_table_term(round: usize) -> [Goldilocks; SPONGE_WIDTH] {
    core::array::from_fn(|i| Goldilocks::new(TERMINAL_EXTERNAL_CONSTANTS[round][i]))
}

#[inline(always)]
fn sbox7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    x6 * x
}

#[inline(never)]
fn apply_mat4(x: &mut [Goldilocks; 4]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    x[3] = t01233 + x[0].double();
    x[1] = t01123 + x[2].double();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

/// Copied from qp-poseidon-core-3.1.0 `external_linear_layer`.
#[inline(never)]
fn external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        apply_mat4(chunk.try_into().unwrap());
    }
    let sums: [Goldilocks; 4] =
        core::array::from_fn(|k| (0..SPONGE_WIDTH).step_by(4).map(|j| state[j + k]).sum());
    for (i, elem) in state.iter_mut().enumerate() {
        *elem += sums[i % 4];
    }
}

/// Reference: qp-poseidon-core `Poseidon2::external_round` body (RC → S-box×12 → ext linear).
#[inline(never)]
fn reference_round(state: &mut [Goldilocks; SPONGE_WIDTH], rc: &[Goldilocks; SPONGE_WIDTH]) {
    for i in 0..SPONGE_WIDTH {
        state[i] += rc[i];
        state[i] = state[i].exp7();
    }
    external_linear_layer(state);
}

#[inline(never)]
fn fast_external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    apply_mat4((&mut state[0..4]).try_into().unwrap());
    apply_mat4((&mut state[4..8]).try_into().unwrap());
    apply_mat4((&mut state[8..12]).try_into().unwrap());
    let s0 = state[0] + state[4] + state[8];
    let s1 = state[1] + state[5] + state[9];
    let s2 = state[2] + state[6] + state[10];
    let s3 = state[3] + state[7] + state[11];
    state[0] += s0;
    state[1] += s1;
    state[2] += s2;
    state[3] += s3;
    state[4] += s0;
    state[5] += s1;
    state[6] += s2;
    state[7] += s3;
    state[8] += s0;
    state[9] += s1;
    state[10] += s2;
    state[11] += s3;
}

/// Same external round as reference; unrolled S-box and MDS.
#[inline(never)]
fn fast_round(state: &mut [Goldilocks; SPONGE_WIDTH], rc: &[Goldilocks; SPONGE_WIDTH]) {
    state[0] = sbox7(state[0] + rc[0]);
    state[1] = sbox7(state[1] + rc[1]);
    state[2] = sbox7(state[2] + rc[2]);
    state[3] = sbox7(state[3] + rc[3]);
    state[4] = sbox7(state[4] + rc[4]);
    state[5] = sbox7(state[5] + rc[5]);
    state[6] = sbox7(state[6] + rc[6]);
    state[7] = sbox7(state[7] + rc[7]);
    state[8] = sbox7(state[8] + rc[8]);
    state[9] = sbox7(state[9] + rc[9]);
    state[10] = sbox7(state[10] + rc[10]);
    state[11] = sbox7(state[11] + rc[11]);
    fast_external_linear_layer(state);
}

fn assert_round_pair(
    mut state: [Goldilocks; SPONGE_WIDTH],
    rc: [Goldilocks; SPONGE_WIDTH],
) {
    let mut reference = state;
    let mut fast = state;
    reference_round(&mut reference, &rc);
    fast_round(&mut fast, &rc);
    assert_eq!(
        reference.map(|g| g.as_canonical_u64()),
        fast.map(|g| g.as_canonical_u64()),
        "external round mismatch"
    );
    black_box(&mut state);
}

#[test]
fn p1_07_external_round_parity_and_bench() {
    // qp-poseidon-core: first external round on zero state after initial ext linear.
    let rc0 = rc_table(0);
    let mut lib_step = [Goldilocks::ZERO; SPONGE_WIDTH];
    reference_round(&mut lib_step, &rc0);
    let expected_after_round = [
        15870188304392300026u64,
        5420667415882104228,
        4061330223075420786,
        3907156211996553271,
        7220184397568305579,
        6276399041015305555,
        10112551490188661599,
        920706293309282174,
        6439956850909369582,
        10360554408705021829,
        3933312188334710888,
        6787120046090971660,
    ];
    assert_eq!(lib_step.map(|g| g.as_canonical_u64()), expected_after_round);

    let mut rng_state = 0x243f6a8885a308d3u64;
    let mut next_u64 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for round in 0..4 {
        let rc = rc_table(round);
        for _ in 0..250 {
            let state: [Goldilocks; SPONGE_WIDTH] =
                core::array::from_fn(|_| Goldilocks::from_u64(next_u64()));
            assert_round_pair(state, rc);
        }
    }

    for round in 0..4 {
        let rc = rc_table_term(round);
        for _ in 0..250 {
            let state: [Goldilocks; SPONGE_WIDTH] =
                core::array::from_fn(|_| Goldilocks::from_u64(next_u64()));
            assert_round_pair(state, rc);
        }
    }

    const ITERS: u32 = 20_000;
    let base: [Goldilocks; SPONGE_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64(0x7d5c08eef608e382_u64.wrapping_add(i as u64)));
    let rc = rc_table(2);
    let mut acc = 0u64;

    let t0 = Instant::now();
    for i in 0..ITERS {
        let mut s = base;
        s[0] += Goldilocks::from_u64(i as u64);
        reference_round(black_box(&mut s), black_box(&rc));
        acc ^= black_box(s[0].as_canonical_u64());
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t1 = Instant::now();
    for i in 0..ITERS {
        let mut s = base;
        s[0] += Goldilocks::from_u64(i as u64);
        fast_round(black_box(&mut s), black_box(&rc));
        acc ^= black_box(s[0].as_canonical_u64());
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;

    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.2} NEW_NS {new_ns:.2} RATIO {ratio:.4}");
    black_box(acc);
}
