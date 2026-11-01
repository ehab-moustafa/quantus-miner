//! Phase 1.08: full Poseidon2 width-12 permute — library vs inlined fast path.

use qp_poseidon_core::poseidon2::{
    HALF_EXTERNAL_ROUNDS, INITIAL_EXTERNAL_CONSTANTS, INTERNAL_CONSTANTS, INTERNAL_ROUNDS,
    MATRIX_DIAG, SPONGE_WIDTH, TERMINAL_EXTERNAL_CONSTANTS,
};
use qp_poseidon_core::{Goldilocks, Poseidon2};
use std::hint::black_box;
use std::sync::OnceLock;
use std::time::Instant;

struct PermuteTables {
    internal: [Goldilocks; INTERNAL_ROUNDS],
    diag: [Goldilocks; SPONGE_WIDTH],
    init_ext: [[Goldilocks; SPONGE_WIDTH]; HALF_EXTERNAL_ROUNDS],
    term_ext: [[Goldilocks; SPONGE_WIDTH]; HALF_EXTERNAL_ROUNDS],
}

fn tables() -> &'static PermuteTables {
    static T: OnceLock<PermuteTables> = OnceLock::new();
    T.get_or_init(|| PermuteTables {
        internal: core::array::from_fn(|i| Goldilocks::new(INTERNAL_CONSTANTS[i])),
        diag: core::array::from_fn(|i| Goldilocks::new(MATRIX_DIAG[i])),
        init_ext: core::array::from_fn(|r| {
            core::array::from_fn(|i| Goldilocks::new(INITIAL_EXTERNAL_CONSTANTS[r][i]))
        }),
        term_ext: core::array::from_fn(|r| {
            core::array::from_fn(|i| Goldilocks::new(TERMINAL_EXTERNAL_CONSTANTS[r][i]))
        }),
    })
}

fn ref_permute(poseidon: &Poseidon2, state: &mut [Goldilocks; SPONGE_WIDTH]) {
    poseidon.permute_mut(state);
}

#[inline(always)]
fn sbox7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    x3 * x4
}

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

#[inline(always)]
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
fn internal_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH], matrix_diag: &[Goldilocks; SPONGE_WIDTH]) {
    let sum: Goldilocks = state.iter().copied().sum();
    for i in 0..SPONGE_WIDTH {
        state[i] = sum + state[i] * matrix_diag[i];
    }
}

#[inline(always)]
fn external_round(state: &mut [Goldilocks; SPONGE_WIDTH], rc: &[Goldilocks; SPONGE_WIDTH]) {
    for i in 0..SPONGE_WIDTH {
        state[i] += rc[i];
        state[i] = sbox7(state[i]);
    }
    external_linear_layer(state);
}

#[inline(always)]
fn internal_round(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    rc: Goldilocks,
    matrix_diag: &[Goldilocks; SPONGE_WIDTH],
) {
    state[0] += rc;
    state[0] = sbox7(state[0]);
    internal_linear_layer(state, matrix_diag);
}

/// Inlined copy of `Poseidon2::permute_mut` (qp-poseidon-core 3.1.0); rounds fully unrolled.
#[inline(always)]
fn fast_permute(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    let t = tables();
    external_linear_layer(state);
    external_round(state, &t.init_ext[0]);
    external_round(state, &t.init_ext[1]);
    external_round(state, &t.init_ext[2]);
    external_round(state, &t.init_ext[3]);
    internal_round(state, t.internal[0], &t.diag);
    internal_round(state, t.internal[1], &t.diag);
    internal_round(state, t.internal[2], &t.diag);
    internal_round(state, t.internal[3], &t.diag);
    internal_round(state, t.internal[4], &t.diag);
    internal_round(state, t.internal[5], &t.diag);
    internal_round(state, t.internal[6], &t.diag);
    internal_round(state, t.internal[7], &t.diag);
    internal_round(state, t.internal[8], &t.diag);
    internal_round(state, t.internal[9], &t.diag);
    internal_round(state, t.internal[10], &t.diag);
    internal_round(state, t.internal[11], &t.diag);
    internal_round(state, t.internal[12], &t.diag);
    internal_round(state, t.internal[13], &t.diag);
    internal_round(state, t.internal[14], &t.diag);
    internal_round(state, t.internal[15], &t.diag);
    internal_round(state, t.internal[16], &t.diag);
    internal_round(state, t.internal[17], &t.diag);
    internal_round(state, t.internal[18], &t.diag);
    internal_round(state, t.internal[19], &t.diag);
    internal_round(state, t.internal[20], &t.diag);
    internal_round(state, t.internal[21], &t.diag);
    external_round(state, &t.term_ext[0]);
    external_round(state, &t.term_ext[1]);
    external_round(state, &t.term_ext[2]);
    external_round(state, &t.term_ext[3]);
}

fn state_from_u64s(limbs: [u64; SPONGE_WIDTH]) -> [Goldilocks; SPONGE_WIDTH] {
    core::array::from_fn(|i| Goldilocks::new(limbs[i]))
}

fn assert_permute_matches(input: [Goldilocks; SPONGE_WIDTH], poseidon: &Poseidon2) {
    let mut expected = input;
    let mut got = input;
    ref_permute(poseidon, &mut expected);
    fast_permute(&mut got);
    assert_eq!(expected, got, "permute mismatch");
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn p1_08_poseidon2_permute_parity_and_bench() {
    let poseidon = Poseidon2::new();
    let _ = tables();

    const P3_ZERO: [u64; 12] = [
        0xc9bc9432e1686884,
        0x03ecbab0dcdd2189,
        0x5e7ac885b3dc1215,
        0x6ac07513801d191f,
        0xca5c593fb184dcfc,
        0x414dec5f3e455287,
        0x1a17df170127ae41,
        0xe7e592bd0af9b0a5,
        0xc71a9b27edc66a4c,
        0x2728671759ac43c2,
        0xb9969c20f7f672f9,
        0xc5140b586823b92f,
    ];
    const P3_SEQ: [u64; 12] = [
        0x7e9574e2a3d6c48b,
        0x9d7bc16d282d2f2b,
        0x798826626d94a498,
        0x0831011bb22304c7,
        0xbdccb5931fffd16c,
        0xe98687714dacbefc,
        0xc6a1ed29dd75e027,
        0x1aec96681d15f765,
        0xc74b2c710b170a23,
        0x5fb4aff45e9c24fb,
        0x1fb3d228db0127eb,
        0xe201a7e214b16e74,
    ];

    for limbs in [P3_ZERO, P3_SEQ] {
        let input = state_from_u64s(limbs);
        assert_permute_matches(input, &poseidon);
    }

    let mut rng = 0xA5A5_1771_0800_0008u64;
    for _ in 0..200 {
        let input = state_from_u64s(core::array::from_fn(|_| splitmix64(&mut rng)));
        assert_permute_matches(input, &poseidon);
    }

    const ITERS: u64 = 5_000;
    let seed = state_from_u64s([
        0x486dcb55419d475a,
        0x6c1cb2a358cc24f1,
        0xe3f30d509a1436bb,
        0xd9a64f068dca7c29,
        0xe59b3f57aabba1ae,
        0x2a3dd4505b478fdc,
        0xada1f8dc7676ed25,
        0x2711aa8b5509d516,
        0x4ae6acd0c9c92897,
        0x56eb3d6b5256d67a,
        0x1f7a9d55923bf51e,
        0x3600427d397a7f68,
    ]);
    let mut acc = 0u64;

    let t0 = Instant::now();
    for i in 0..ITERS {
        let mut st = seed;
        st[0] = Goldilocks::new(black_box(st[0].as_canonical_u64().wrapping_add(i)));
        ref_permute(&poseidon, &mut st);
        acc ^= black_box(st[0].as_canonical_u64());
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t1 = Instant::now();
    for i in 0..ITERS {
        let mut st = seed;
        st[0] = Goldilocks::new(black_box(st[0].as_canonical_u64().wrapping_add(i)));
        fast_permute(&mut st);
        acc ^= black_box(st[0].as_canonical_u64());
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;

    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.3} NEW_NS {new_ns:.3} RATIO {ratio:.4}");
    black_box(acc);
}
