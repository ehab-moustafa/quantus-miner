//! Phase 1 agent 05: external linear layer (WIDTH=12) vs qp-poseidon-core 3.1.0.
//! Source: qp-poseidon-core-3.1.0/src/poseidon2.rs

use qp_poseidon_core::{Goldilocks, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

/// Integer coefficients of the 12×12 external linear map over F_p (circulant ⊗ M_4).
const EXT_COEFF: [[u8; SPONGE_WIDTH]; SPONGE_WIDTH] = [
    [4, 6, 2, 2, 2, 3, 1, 1, 2, 3, 1, 1],
    [2, 4, 6, 2, 1, 2, 3, 1, 1, 2, 3, 1],
    [2, 2, 4, 6, 1, 1, 2, 3, 1, 1, 2, 3],
    [6, 2, 2, 4, 3, 1, 1, 2, 3, 1, 1, 2],
    [2, 3, 1, 1, 4, 6, 2, 2, 2, 3, 1, 1],
    [1, 2, 3, 1, 2, 4, 6, 2, 1, 2, 3, 1],
    [1, 1, 2, 3, 2, 2, 4, 6, 1, 1, 2, 3],
    [3, 1, 1, 2, 6, 2, 2, 4, 3, 1, 1, 2],
    [2, 3, 1, 1, 2, 3, 1, 1, 4, 6, 2, 2],
    [1, 2, 3, 1, 1, 2, 3, 1, 2, 4, 6, 2],
    [1, 1, 2, 3, 1, 1, 2, 3, 2, 2, 4, 6],
    [3, 1, 1, 2, 3, 1, 1, 2, 6, 2, 2, 4],
];

/// Copied from qp-poseidon-core-3.1.0 `apply_mat4`.
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

#[inline(always)]
fn scale(c: u8, x: Goldilocks) -> Goldilocks {
    match c {
        0 => Goldilocks::ZERO,
        1 => x,
        2 => x.double(),
        3 => x.double() + x,
        4 => x.double().double(),
        5 => x.double().double() + x,
        6 => x.double().double() + x.double(),
        _ => unreachable!(),
    }
}

/// Single fused pass using the cached circulant–MDS integer matrix (F_p-linear).
#[inline(never)]
fn fast_external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    let input = *state;
    for (i, out) in state.iter_mut().enumerate() {
        let row = &EXT_COEFF[i];
        let mut acc = Goldilocks::ZERO;
        for j in 0..SPONGE_WIDTH {
            acc += scale(row[j], input[j]);
        }
        *out = acc;
    }
}

fn assert_ext_pair_from(input: [Goldilocks; SPONGE_WIDTH]) {
    let mut old = input;
    let mut fast = input;
    external_linear_layer(&mut old);
    fast_external_linear_layer(&mut fast);
    assert_eq!(
        old.map(|g| g.as_canonical_u64()),
        fast.map(|g| g.as_canonical_u64()),
        "external linear layer mismatch"
    );
}

#[test]
fn p1_05_external_linear_parity_and_bench() {
    // qp-poseidon-core unit test vector (sequential 1..=12 input).
    let mut seq: [Goldilocks; SPONGE_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64((i + 1) as u64));
    external_linear_layer(&mut seq);
    let expected = [
        144u64, 156, 168, 148, 172, 184, 196, 176, 200, 212, 224, 204,
    ];
    assert_eq!(seq.map(|g| g.as_canonical_u64()), expected);

    let mut fast_seq: [Goldilocks; SPONGE_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64((i + 1) as u64));
    fast_external_linear_layer(&mut fast_seq);
    assert_eq!(fast_seq.map(|g| g.as_canonical_u64()), expected);

    let mut rng_state = 0x517cc1b727220a95u64;
    let mut next_u64 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for _ in 0..10_000 {
        let state: [Goldilocks; SPONGE_WIDTH] =
            core::array::from_fn(|_| Goldilocks::from_u64(next_u64()));
        assert_ext_pair_from(state);
    }

    // Linearity spot-check: ext(s + d) == ext(s) + ext(d) on random small deltas.
    for _ in 0..64 {
        let s: [Goldilocks; SPONGE_WIDTH] =
            core::array::from_fn(|_| Goldilocks::from_u64(next_u64()));
        let d: [Goldilocks; SPONGE_WIDTH] =
            core::array::from_fn(|_| Goldilocks::from_u64(next_u64() & 0xffff));
        let mut sum = core::array::from_fn(|i| s[i] + d[i]);
        external_linear_layer(&mut sum);
        let mut es = s;
        let mut ed = d;
        external_linear_layer(&mut es);
        external_linear_layer(&mut ed);
        let right: [Goldilocks; SPONGE_WIDTH] = core::array::from_fn(|i| es[i] + ed[i]);
        assert_eq!(
            sum.map(|g| g.as_canonical_u64()),
            right.map(|g| g.as_canonical_u64()),
            "external layer must be F_p-linear"
        );
    }

    const ITERS: u32 = 100_000;
    let base: [Goldilocks; SPONGE_WIDTH] = core::array::from_fn(|i| {
        Goldilocks::from_u64(0x486dcb55419d475a_u64.wrapping_add(i as u64))
    });
    let mut acc = 0u64;

    let t0 = Instant::now();
    for i in 0..ITERS {
        let mut s = base;
        s[0] += Goldilocks::from_u64(i as u64);
        external_linear_layer(black_box(&mut s));
        acc ^= black_box(s[0].as_canonical_u64());
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t1 = Instant::now();
    for i in 0..ITERS {
        let mut s = base;
        s[0] += Goldilocks::from_u64(i as u64);
        fast_external_linear_layer(black_box(&mut s));
        acc ^= black_box(s[0].as_canonical_u64());
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;

    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.2} NEW_NS {new_ns:.2} RATIO {ratio:.4}");
    black_box(acc);
}
