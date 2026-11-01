//! Whole-permute re-challenge of `Poseidon2::permute_mut` (qp-poseidon-core 3.1.0).
//!
//! `pipelined_permute` is the same mathematical permutation with a different
//! internal-round schedule: lane 0 of the diffusion (plus the next round
//! constant) is formed first, and the next S-box starts before lanes 1..11
//! are multiplied. That is the CPU analogue of CUDA `int_round_p`.

use pow_core::custom;
use qp_poseidon_core::poseidon2::{
    INITIAL_EXTERNAL_CONSTANTS, INTERNAL_CONSTANTS, MATRIX_DIAG, TERMINAL_EXTERNAL_CONSTANTS,
};
use qp_poseidon_core::{Goldilocks, Poseidon2, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

const INIT_RC: [[Goldilocks; SPONGE_WIDTH]; 4] = [
    row(INITIAL_EXTERNAL_CONSTANTS[0]),
    row(INITIAL_EXTERNAL_CONSTANTS[1]),
    row(INITIAL_EXTERNAL_CONSTANTS[2]),
    row(INITIAL_EXTERNAL_CONSTANTS[3]),
];

const TERM_RC: [[Goldilocks; SPONGE_WIDTH]; 4] = [
    row(TERMINAL_EXTERNAL_CONSTANTS[0]),
    row(TERMINAL_EXTERNAL_CONSTANTS[1]),
    row(TERMINAL_EXTERNAL_CONSTANTS[2]),
    row(TERMINAL_EXTERNAL_CONSTANTS[3]),
];

const INT_RC: [Goldilocks; 22] = [
    Goldilocks::new(INTERNAL_CONSTANTS[0]),
    Goldilocks::new(INTERNAL_CONSTANTS[1]),
    Goldilocks::new(INTERNAL_CONSTANTS[2]),
    Goldilocks::new(INTERNAL_CONSTANTS[3]),
    Goldilocks::new(INTERNAL_CONSTANTS[4]),
    Goldilocks::new(INTERNAL_CONSTANTS[5]),
    Goldilocks::new(INTERNAL_CONSTANTS[6]),
    Goldilocks::new(INTERNAL_CONSTANTS[7]),
    Goldilocks::new(INTERNAL_CONSTANTS[8]),
    Goldilocks::new(INTERNAL_CONSTANTS[9]),
    Goldilocks::new(INTERNAL_CONSTANTS[10]),
    Goldilocks::new(INTERNAL_CONSTANTS[11]),
    Goldilocks::new(INTERNAL_CONSTANTS[12]),
    Goldilocks::new(INTERNAL_CONSTANTS[13]),
    Goldilocks::new(INTERNAL_CONSTANTS[14]),
    Goldilocks::new(INTERNAL_CONSTANTS[15]),
    Goldilocks::new(INTERNAL_CONSTANTS[16]),
    Goldilocks::new(INTERNAL_CONSTANTS[17]),
    Goldilocks::new(INTERNAL_CONSTANTS[18]),
    Goldilocks::new(INTERNAL_CONSTANTS[19]),
    Goldilocks::new(INTERNAL_CONSTANTS[20]),
    Goldilocks::new(INTERNAL_CONSTANTS[21]),
];

const DIAG: [Goldilocks; SPONGE_WIDTH] = [
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

const fn row(rc: [u64; SPONGE_WIDTH]) -> [Goldilocks; SPONGE_WIDTH] {
    [
        Goldilocks::new(rc[0]),
        Goldilocks::new(rc[1]),
        Goldilocks::new(rc[2]),
        Goldilocks::new(rc[3]),
        Goldilocks::new(rc[4]),
        Goldilocks::new(rc[5]),
        Goldilocks::new(rc[6]),
        Goldilocks::new(rc[7]),
        Goldilocks::new(rc[8]),
        Goldilocks::new(rc[9]),
        Goldilocks::new(rc[10]),
        Goldilocks::new(rc[11]),
    ]
}

#[inline(always)]
fn sbox7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    x3 * x4
}

#[inline(always)]
fn external_sbox_round(state: &mut [Goldilocks; SPONGE_WIDTH], rc: &[Goldilocks; SPONGE_WIDTH]) {
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
    custom::external_linear_layer(state);
}

/// One internal round. `x` is this round's S-boxed lane 0.
///
/// Lane 0 (with the next round constant already added) is ready before the
/// other diagonal products. The next S-box's square is issued in that window,
/// then its x^3 / x^4 steps retire alongside lanes 1..11. `state[0]` is left
/// stale; the returned value is the next round's S-boxed lane 0.
#[inline(always)]
fn int_round_pipe(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    x: Goldilocks,
    rc_next: Goldilocks,
) -> Goldilocks {
    let p0 = x * DIAG[0];
    let rest: Goldilocks = state[1..].iter().copied().sum();
    let sum = rest + x;
    let out0 = sum + p0 + rc_next;

    let n2 = out0 * out0;

    state[1] = sum + state[1] * DIAG[1];
    let n3 = n2 * out0;
    state[2] = sum + state[2] * DIAG[2];
    state[3] = sum + state[3] * DIAG[3];
    let n4 = n2 * n2;
    state[4] = sum + state[4] * DIAG[4];
    state[5] = sum + state[5] * DIAG[5];
    state[6] = sum + state[6] * DIAG[6];
    state[7] = sum + state[7] * DIAG[7];
    state[8] = sum + state[8] * DIAG[8];
    state[9] = sum + state[9] * DIAG[9];
    state[10] = sum + state[10] * DIAG[10];
    state[11] = sum + state[11] * DIAG[11];
    n3 * n4
}

/// Last internal round: same diffusion, no following S-box.
#[inline(always)]
fn int_round_finish(state: &mut [Goldilocks; SPONGE_WIDTH], x: Goldilocks) {
    let p0 = x * DIAG[0];
    let rest: Goldilocks = state[1..].iter().copied().sum();
    let sum = rest + x;
    state[0] = sum + p0;
    state[1] = sum + state[1] * DIAG[1];
    state[2] = sum + state[2] * DIAG[2];
    state[3] = sum + state[3] * DIAG[3];
    state[4] = sum + state[4] * DIAG[4];
    state[5] = sum + state[5] * DIAG[5];
    state[6] = sum + state[6] * DIAG[6];
    state[7] = sum + state[7] * DIAG[7];
    state[8] = sum + state[8] * DIAG[8];
    state[9] = sum + state[9] * DIAG[9];
    state[10] = sum + state[10] * DIAG[10];
    state[11] = sum + state[11] * DIAG[11];
}

/// Full width-12 Poseidon2 permutation.
#[inline(always)]
fn pipelined_permute(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    custom::external_linear_layer(state);
    external_sbox_round(state, &INIT_RC[0]);
    external_sbox_round(state, &INIT_RC[1]);
    external_sbox_round(state, &INIT_RC[2]);
    external_sbox_round(state, &INIT_RC[3]);

    let mut x = sbox7(state[0] + INT_RC[0]);
    x = int_round_pipe(state, x, INT_RC[1]);
    x = int_round_pipe(state, x, INT_RC[2]);
    x = int_round_pipe(state, x, INT_RC[3]);
    x = int_round_pipe(state, x, INT_RC[4]);
    x = int_round_pipe(state, x, INT_RC[5]);
    x = int_round_pipe(state, x, INT_RC[6]);
    x = int_round_pipe(state, x, INT_RC[7]);
    x = int_round_pipe(state, x, INT_RC[8]);
    x = int_round_pipe(state, x, INT_RC[9]);
    x = int_round_pipe(state, x, INT_RC[10]);
    x = int_round_pipe(state, x, INT_RC[11]);
    x = int_round_pipe(state, x, INT_RC[12]);
    x = int_round_pipe(state, x, INT_RC[13]);
    x = int_round_pipe(state, x, INT_RC[14]);
    x = int_round_pipe(state, x, INT_RC[15]);
    x = int_round_pipe(state, x, INT_RC[16]);
    x = int_round_pipe(state, x, INT_RC[17]);
    x = int_round_pipe(state, x, INT_RC[18]);
    x = int_round_pipe(state, x, INT_RC[19]);
    x = int_round_pipe(state, x, INT_RC[20]);
    x = int_round_pipe(state, x, INT_RC[21]);
    int_round_finish(state, x);

    external_sbox_round(state, &TERM_RC[0]);
    external_sbox_round(state, &TERM_RC[1]);
    external_sbox_round(state, &TERM_RC[2]);
    external_sbox_round(state, &TERM_RC[3]);
}

fn canon(state: &[Goldilocks; SPONGE_WIDTH]) -> [u64; SPONGE_WIDTH] {
    state.map(|g| g.as_canonical_u64())
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(never)]
fn timed_lib(poseidon: &Poseidon2, state: &mut [Goldilocks; SPONGE_WIDTH]) {
    poseidon.permute_mut(state);
}

#[inline(never)]
fn timed_custom(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    custom::permute_mut(state);
}

#[inline(never)]
fn timed_pipe(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    pipelined_permute(state);
}

const SEED: [Goldilocks; SPONGE_WIDTH] = [
    Goldilocks::new(0x486dcb55419d475a),
    Goldilocks::new(0x6c1cb2a358cc24f1),
    Goldilocks::new(0xe3f30d509a1436bb),
    Goldilocks::new(0xd9a64f068dca7c29),
    Goldilocks::new(0xe59b3f57aabba1ae),
    Goldilocks::new(0x2a3dd4505b478fdc),
    Goldilocks::new(0xada1f8dc7676ed25),
    Goldilocks::new(0x2711aa8b5509d516),
    Goldilocks::new(0x4ae6acd0c9c92897),
    Goldilocks::new(0x56eb3d6b5256d67a),
    Goldilocks::new(0x1f7a9d55923bf51e),
    Goldilocks::new(0x3600427d397a7f68),
];

fn ns_per(iters: u64, mut permute: impl FnMut(&mut [Goldilocks; SPONGE_WIDTH])) -> f64 {
    const WARMUP: u64 = 2_000;
    let mut acc = 0u64;
    for i in 0..WARMUP {
        let mut st = SEED;
        st[1] = Goldilocks::new(st[1].as_canonical_u64().wrapping_add(i));
        permute(&mut st);
        acc ^= st[0].as_canonical_u64();
    }
    black_box(acc);

    acc = 0;
    let started = Instant::now();
    for i in 0..iters {
        let mut st = SEED;
        st[1] = Goldilocks::new(black_box(st[1].as_canonical_u64().wrapping_add(i)));
        permute(&mut st);
        acc ^= black_box(st[0].as_canonical_u64());
    }
    let ns = started.elapsed().as_nanos() as f64 / iters as f64;
    black_box(acc);
    ns
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn prefer_performance_cores() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        // QOS_CLASS_USER_INTERACTIVE — keep the timed thread off efficiency cores.
        let rc = pthread_set_qos_class_self_np(0x21, 0);
        assert_eq!(rc, 0, "pthread_set_qos_class_self_np");
    }
}

#[test]
fn r2_05_permute_parity_and_bench() {
    prefer_performance_cores();
    let poseidon = Poseidon2::new();

    let mut zero = [Goldilocks::ZERO; SPONGE_WIDTH];
    let mut zero_pipe = zero;
    let mut zero_custom = zero;
    poseidon.permute_mut(&mut zero);
    pipelined_permute(&mut zero_pipe);
    custom::permute_mut(&mut zero_custom);
    let zero_canon = canon(&zero);
    assert_eq!(
        zero_canon,
        [
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
        ],
        "library zero-state permute drifted from the p3 vector"
    );
    assert_eq!(canon(&zero_pipe), zero_canon, "pipelined zero state");
    assert_eq!(canon(&zero_custom), zero_canon, "custom zero state");

    let mut rng = 0xA5A5_0517_0000_0005u64;
    for n in 0..64 {
        let input = core::array::from_fn(|_| Goldilocks::new(splitmix64(&mut rng)));
        let mut lib = input;
        let mut pipe = input;
        let mut cust = input;
        poseidon.permute_mut(&mut lib);
        pipelined_permute(&mut pipe);
        custom::permute_mut(&mut cust);
        let expect = canon(&lib);
        assert_eq!(canon(&pipe), expect, "pipelined random state {n}");
        assert_eq!(canon(&cust), expect, "custom random state {n}");
    }

    const ITERS: u64 = 200_000;
    const TRIALS: usize = 7;
    let mut lib_samples = [0.0; TRIALS];
    let mut custom_samples = [0.0; TRIALS];
    let mut pipe_samples = [0.0; TRIALS];

    for trial in 0..TRIALS {
        let start = trial % 3;
        for step in 0..3 {
            match (start + step) % 3 {
                0 => {
                    lib_samples[trial] = ns_per(ITERS, |st| timed_lib(&poseidon, st));
                    println!("trial {trial} lib_ns {:.3}", lib_samples[trial]);
                }
                1 => {
                    custom_samples[trial] = ns_per(ITERS, timed_custom);
                    println!("trial {trial} custom_ns {:.3}", custom_samples[trial]);
                }
                _ => {
                    pipe_samples[trial] = ns_per(ITERS, timed_pipe);
                    println!("trial {trial} pipe_ns {:.3}", pipe_samples[trial]);
                }
            }
        }
    }

    let lib_ns = median(&mut lib_samples);
    let custom_ns = median(&mut custom_samples);
    let pipe_ns = median(&mut pipe_samples);
    let ratio_custom = lib_ns / custom_ns;
    let ratio_pipe = lib_ns / pipe_ns;
    println!("MEDIAN lib {lib_ns:.3} custom {custom_ns:.3} pipe {pipe_ns:.3}");
    println!("RATIO custom {ratio_custom:.4} pipe {ratio_pipe:.4}");
}
