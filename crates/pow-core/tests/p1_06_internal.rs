//! Phase 1 agent 06: internal linear layer (WIDTH=12) vs qp-poseidon-core 3.1.0.
//! Source: qp-poseidon-core-3.1.0/src/poseidon2.rs

use qp_poseidon_core::poseidon2::MATRIX_DIAG;
use qp_poseidon_core::{Goldilocks, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

/// Diagonal scalars baked in once — same on every internal round (22× per permute).
const MATRIX_DIAG_G: [Goldilocks; SPONGE_WIDTH] = [
    Goldilocks::new(MATRIX_DIAG[0]),
    Goldilocks::new(MATRIX_DIAG[1]),
    Goldilocks::new(MATRIX_DIAG[2]),
    Goldilocks::new(MATRIX_DIAG[3]),
    Goldilocks::new(MATRIX_DIAG[4]),
    Goldilocks::new(MATRIX_DIAG[5]),
    Goldilocks::new(MATRIX_DIAG[6]),
    Goldilocks::new(MATRIX_DIAG[7]),
    Goldilocks::new(MATRIX_DIAG[8]),
    Goldilocks::new(MATRIX_DIAG[9]),
    Goldilocks::new(MATRIX_DIAG[10]),
    Goldilocks::new(MATRIX_DIAG[11]),
];

/// Copied from qp-poseidon-core-3.1.0 `internal_linear_layer` (parameterized diag).
#[inline(never)]
fn internal_linear_layer(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    matrix_diag: &[Goldilocks; SPONGE_WIDTH],
) {
    let sum: Goldilocks = state.iter().copied().sum();
    for i in 0..SPONGE_WIDTH {
        state[i] = sum + state[i] * matrix_diag[i];
    }
}

/// Static diagonal + unrolled scale-add (diag repeats every internal round).
#[inline(never)]
fn fast_internal_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    let sum: Goldilocks = state.iter().copied().sum();
    state[0] = sum + state[0] * MATRIX_DIAG_G[0];
    state[1] = sum + state[1] * MATRIX_DIAG_G[1];
    state[2] = sum + state[2] * MATRIX_DIAG_G[2];
    state[3] = sum + state[3] * MATRIX_DIAG_G[3];
    state[4] = sum + state[4] * MATRIX_DIAG_G[4];
    state[5] = sum + state[5] * MATRIX_DIAG_G[5];
    state[6] = sum + state[6] * MATRIX_DIAG_G[6];
    state[7] = sum + state[7] * MATRIX_DIAG_G[7];
    state[8] = sum + state[8] * MATRIX_DIAG_G[8];
    state[9] = sum + state[9] * MATRIX_DIAG_G[9];
    state[10] = sum + state[10] * MATRIX_DIAG_G[10];
    state[11] = sum + state[11] * MATRIX_DIAG_G[11];
}

fn assert_internal_pair(state: [Goldilocks; SPONGE_WIDTH], diag: &[Goldilocks; SPONGE_WIDTH]) {
    let mut old = state;
    let mut fast = state;
    internal_linear_layer(&mut old, diag);
    fast_internal_linear_layer(&mut fast);
    assert_eq!(
        old.map(|g| g.as_canonical_u64()),
        fast.map(|g| g.as_canonical_u64()),
        "internal linear layer mismatch"
    );
}

#[test]
fn p1_06_internal_linear_parity_and_bench() {
    let diag = &MATRIX_DIAG_G;

    // qp-poseidon-core unit test vector (sequential 1..=12 input).
    let mut seq: [Goldilocks; SPONGE_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64((i + 1) as u64));
    internal_linear_layer(&mut seq, diag);
    let expected = [
        14102670999874605902u64,
        12724564314584031161,
        2820561051427350843,
        16542800896675938601,
        15306463738286039492,
        7752500014678011478,
        6061018803481092188,
        7074894295206835361,
        12845226504185330854,
        9784840285965170449,
        9819782087245364719,
        15989125545469266117,
    ];
    assert_eq!(seq.map(|g| g.as_canonical_u64()), expected);

    let mut rng_state = 0x6a09e667f3bcc908u64;
    let mut next_u64 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for _ in 0..10_000 {
        let state: [Goldilocks; SPONGE_WIDTH] =
            core::array::from_fn(|_| Goldilocks::from_u64(next_u64()));
        assert_internal_pair(state, diag);
    }

    const ITERS: u32 = 100_000;
    let base: [Goldilocks; SPONGE_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64(0x486dcb55419d475a_u64.wrapping_add(i as u64)));
    let mut acc = 0u64;

    let t0 = Instant::now();
    for i in 0..ITERS {
        let mut s = base;
        s[0] += Goldilocks::from_u64(i as u64);
        internal_linear_layer(black_box(&mut s), diag);
        acc ^= black_box(s[0].as_canonical_u64());
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t1 = Instant::now();
    for i in 0..ITERS {
        let mut s = base;
        s[0] += Goldilocks::from_u64(i as u64);
        fast_internal_linear_layer(black_box(&mut s));
        acc ^= black_box(s[0].as_canonical_u64());
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;

    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.2} NEW_NS {new_ns:.2} RATIO {ratio:.4}");
    black_box(acc);
}
