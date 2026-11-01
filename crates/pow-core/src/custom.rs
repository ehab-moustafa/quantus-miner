//! Phase-2 copies of the phase-1 paths that matched the library and were faster.
//! Field add, S-box chain, full 12×12 matrix, and the squeeze reimplementation stay out.

use qp_poseidon_core::poseidon2::{
    INITIAL_EXTERNAL_CONSTANTS, INTERNAL_CONSTANTS, MATRIX_DIAG, TERMINAL_EXTERNAL_CONSTANTS,
};
use qp_poseidon_core::{Goldilocks, SPONGE_WIDTH};

const NEG_ORDER: u64 = 0xFFFF_FFFF; // 2^32 - 1

/// `(2^32 - 1) * x` as a shift-subtract. Same residue as `reduce128` in qp-poseidon-core 3.1.0.
#[inline(always)]
pub fn fast_mul(a: u64, b: u64) -> u64 {
    let prod = (a as u128).wrapping_mul(b as u128);
    let x_lo = prod as u64;
    let x_hi = (prod >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi as u32 as u64;
    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        t0 = t0.wrapping_sub(NEG_ORDER);
    }
    let t1 = x_hi_lo.wrapping_shl(32).wrapping_sub(x_hi_lo);
    let (sum, over) = t0.overflowing_add(t1);
    let (sum, over2) = sum.overflowing_add(over as u64 * NEG_ORDER);
    sum.wrapping_add(over2 as u64 * NEG_ORDER)
}

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

#[inline(always)]
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

/// Unrolled external linear layer. Same values as the library chunk loop.
pub fn external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
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

fn external_round(state: &mut [Goldilocks; SPONGE_WIDTH], rc: &[u64; SPONGE_WIDTH]) {
    for i in 0..SPONGE_WIDTH {
        state[i] += Goldilocks::new(rc[i]);
        state[i] = state[i].exp7();
    }
    external_linear_layer(state);
}

fn internal_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
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

/// Full Poseidon2 permute using the unrolled external layer and internal diagonal.
pub fn permute_mut(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    external_linear_layer(state);
    for rc in &INITIAL_EXTERNAL_CONSTANTS {
        external_round(state, rc);
    }
    for &rc in &INTERNAL_CONSTANTS {
        state[0] += Goldilocks::new(rc);
        state[0] = state[0].exp7();
        internal_linear_layer(state);
    }
    for rc in &TERMINAL_EXTERNAL_CONSTANTS {
        external_round(state, rc);
    }
}
