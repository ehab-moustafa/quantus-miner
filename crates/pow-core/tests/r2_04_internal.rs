//! Re-challenge: Poseidon2 internal linear layer and internal rounds.
//! Library source: qp-poseidon-core 3.1.0 `poseidon2.rs` (`internal_linear_layer`,
//! `internal_round`, and the internal loop inside `permute_mut`).
//!
//! Alternative: const diagonal, unrolled `sum + diag[i] * state[i]`.
//! Field add, sum, and multiply stay the library implementations.
//! Round form is lane-0 RC add, library `exp7`, then that layer.
//! The 22-round form is that round, looped, against the library internal section.

use qp_poseidon_core::poseidon2::{
    INITIAL_EXTERNAL_CONSTANTS, INTERNAL_CONSTANTS, INTERNAL_ROUNDS, MATRIX_DIAG,
    TERMINAL_EXTERNAL_CONSTANTS,
};
use qp_poseidon_core::{Goldilocks, Poseidon2, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

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

const INTERNAL_G: [Goldilocks; INTERNAL_ROUNDS] = [
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

fn ext_table(src: &[[u64; SPONGE_WIDTH]; 4]) -> [[Goldilocks; SPONGE_WIDTH]; 4] {
    core::array::from_fn(|r| core::array::from_fn(|i| Goldilocks::new(src[r][i])))
}

const LAYER_ITERS: u64 = 2_000_000;
const ROUND_ITERS: u64 = 1_000_000;
const SECTION_ITERS: u64 = 40_000;
const TRIALS: usize = 7;

fn canon(state: &[Goldilocks; SPONGE_WIDTH]) -> [u64; SPONGE_WIDTH] {
    state.map(|g| g.as_canonical_u64())
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_state(rng: &mut u64) -> [Goldilocks; SPONGE_WIDTH] {
    core::array::from_fn(|_| Goldilocks::new(xorshift(rng)))
}

// ---------------------------------------------------------------------------
// Library copies. Bodies match qp-poseidon-core 3.1.0.
// `*_body` is inlined the way `internal_linear_layer` / `internal_round` are.
// `*_never` is only the isolated timer entry so the bench loop cannot inline
// the op into itself.
// ---------------------------------------------------------------------------

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
fn external_round(state: &mut [Goldilocks; SPONGE_WIDTH], rc: &[Goldilocks; SPONGE_WIDTH]) {
    for i in 0..SPONGE_WIDTH {
        state[i] += rc[i];
        state[i] = state[i].exp7();
    }
    external_linear_layer(state);
}

/// `internal_linear_layer`: `state[i] = sum(state) + diag[i] * state[i]`.
#[inline(always)]
fn lib_layer_body(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    matrix_diag: &[Goldilocks; SPONGE_WIDTH],
) {
    let sum: Goldilocks = state.iter().copied().sum();
    for i in 0..SPONGE_WIDTH {
        state[i] = sum + state[i] * matrix_diag[i];
    }
}

#[inline(never)]
fn lib_internal_linear_layer(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    matrix_diag: &[Goldilocks; SPONGE_WIDTH],
) {
    lib_layer_body(state, matrix_diag);
}

#[inline(always)]
fn lib_round_body(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    rc: Goldilocks,
    matrix_diag: &[Goldilocks; SPONGE_WIDTH],
) {
    state[0] += rc;
    state[0] = state[0].exp7();
    lib_layer_body(state, matrix_diag);
}

#[inline(never)]
fn lib_internal_round(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    rc: Goldilocks,
    matrix_diag: &[Goldilocks; SPONGE_WIDTH],
) {
    lib_round_body(state, rc, matrix_diag);
}

/// Internal section of `permute_mut`: 22 inlined internal rounds.
#[inline(never)]
fn lib_internal_section(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    constants: &[Goldilocks; INTERNAL_ROUNDS],
    matrix_diag: &[Goldilocks; SPONGE_WIDTH],
) {
    for round in 0..INTERNAL_ROUNDS {
        lib_round_body(state, constants[round], matrix_diag);
    }
}

// ---------------------------------------------------------------------------
// Fresh unrolled diagonal. Same sum and the same Goldilocks mul/add.
// ---------------------------------------------------------------------------

#[inline(always)]
fn alt_layer_body(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    let sum: Goldilocks = state.iter().copied().sum();
    state[0] = sum + state[0] * DIAG[0];
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

#[inline(never)]
fn alt_internal_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    alt_layer_body(state);
}

#[inline(always)]
fn alt_round_body(state: &mut [Goldilocks; SPONGE_WIDTH], rc: Goldilocks) {
    state[0] += rc;
    state[0] = state[0].exp7();
    alt_layer_body(state);
}

#[inline(never)]
fn alt_internal_round(state: &mut [Goldilocks; SPONGE_WIDTH], rc: Goldilocks) {
    alt_round_body(state, rc);
}

#[inline(never)]
fn alt_internal_section(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    constants: &[Goldilocks; INTERNAL_ROUNDS],
) {
    for round in 0..INTERNAL_ROUNDS {
        alt_round_body(state, constants[round]);
    }
}

fn permute_with_lib_internal(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    init: &[[Goldilocks; SPONGE_WIDTH]; 4],
    term: &[[Goldilocks; SPONGE_WIDTH]; 4],
) {
    external_linear_layer(state);
    for rc in init {
        external_round(state, rc);
    }
    lib_internal_section(state, &INTERNAL_G, &DIAG);
    for rc in term {
        external_round(state, rc);
    }
}

fn permute_with_alt_internal(
    state: &mut [Goldilocks; SPONGE_WIDTH],
    init: &[[Goldilocks; SPONGE_WIDTH]; 4],
    term: &[[Goldilocks; SPONGE_WIDTH]; 4],
) {
    external_linear_layer(state);
    for rc in init {
        external_round(state, rc);
    }
    alt_internal_section(state, &INTERNAL_G);
    for rc in term {
        external_round(state, rc);
    }
}

fn assert_same(tag: &str, left: &[Goldilocks; SPONGE_WIDTH], right: &[Goldilocks; SPONGE_WIDTH]) {
    assert_eq!(canon(left), canon(right), "{tag}");
}

fn edge_states() -> Vec<[Goldilocks; SPONGE_WIDTH]> {
    vec![
        [Goldilocks::ZERO; SPONGE_WIDTH],
        core::array::from_fn(|i| Goldilocks::from_u64((i + 1) as u64)),
        [Goldilocks::new(1); SPONGE_WIDTH],
        [Goldilocks::new(u64::MAX); SPONGE_WIDTH],
        [Goldilocks::new(0xFFFF_FFFF_0000_0001); SPONGE_WIDTH],
        [Goldilocks::new(0xFFFF_FFFF_0000_0000); SPONGE_WIDTH],
        core::array::from_fn(|i| Goldilocks::new(0xFFFF_FFFF_0000_0001u64.wrapping_add(i as u64 * 17))),
        core::array::from_fn(|i| Goldilocks::new(INTERNAL_CONSTANTS[i % INTERNAL_ROUNDS].wrapping_add(i as u64))),
    ]
}

fn assert_canonical_equality() {
    // Library unit vector: internal linear layer on 1..=12.
    let sequential = core::array::from_fn(|i| Goldilocks::from_u64((i + 1) as u64));
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
    let mut seq_lib = sequential;
    let mut seq_alt = sequential;
    lib_internal_linear_layer(&mut seq_lib, &DIAG);
    alt_internal_linear_layer(&mut seq_alt);
    assert_eq!(canon(&seq_lib), expected, "library layer drifted from qp-poseidon-core vector");
    assert_eq!(canon(&seq_alt), expected, "unrolled diag drifted from qp-poseidon-core vector");

    let mut rng = 0x6a09e667f3bcc908u64;
    for n in 0..10_000 {
        let state = random_state(&mut rng);
        let mut old = state;
        let mut new = state;
        lib_internal_linear_layer(&mut old, &DIAG);
        alt_internal_linear_layer(&mut new);
        assert_same(&format!("layer random {n}"), &old, &new);
    }
    for (n, state) in edge_states().into_iter().enumerate() {
        let mut old = state;
        let mut new = state;
        lib_internal_linear_layer(&mut old, &DIAG);
        alt_internal_linear_layer(&mut new);
        assert_same(&format!("layer edge {n}"), &old, &new);
    }

    for (n, state) in edge_states().into_iter().enumerate() {
        for (r, &rc) in INTERNAL_G.iter().enumerate() {
            let mut old = state;
            let mut new = state;
            lib_internal_round(&mut old, rc, &DIAG);
            alt_internal_round(&mut new, rc);
            assert_same(&format!("round edge {n} rc {r}"), &old, &new);
        }
    }
    for n in 0..2_000 {
        let state = random_state(&mut rng);
        let rc = Goldilocks::new(xorshift(&mut rng));
        let mut old = state;
        let mut new = state;
        lib_internal_round(&mut old, rc, &DIAG);
        alt_internal_round(&mut new, rc);
        assert_same(&format!("round random {n}"), &old, &new);
    }

    for (n, state) in edge_states().into_iter().enumerate() {
        let mut old = state;
        let mut new = state;
        lib_internal_section(&mut old, &INTERNAL_G, &DIAG);
        alt_internal_section(&mut new, &INTERNAL_G);
        assert_same(&format!("22-round edge {n}"), &old, &new);
    }
    for n in 0..2_000 {
        let state = random_state(&mut rng);
        let mut old = state;
        let mut new = state;
        lib_internal_section(&mut old, &INTERNAL_G, &DIAG);
        alt_internal_section(&mut new, &INTERNAL_G);
        assert_same(&format!("22-round random {n}"), &old, &new);
    }

    // Sandwich both internal sections in the library external schedule and
    // compare to `Poseidon2::permute_mut` (zero, sequential, random).
    let init = ext_table(&INITIAL_EXTERNAL_CONSTANTS);
    let term = ext_table(&TERMINAL_EXTERNAL_CONSTANTS);
    let poseidon = Poseidon2::new();
    let p3_zero = [
        0xc9bc9432e1686884u64,
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
    let p3_seq = [
        0x7e9574e2a3d6c48bu64,
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

    let mut zero = [Goldilocks::ZERO; SPONGE_WIDTH];
    poseidon.permute_mut(&mut zero);
    assert_eq!(canon(&zero), p3_zero, "Poseidon2 zero vector");

    let mut seq = sequential;
    poseidon.permute_mut(&mut seq);
    assert_eq!(canon(&seq), p3_seq, "Poseidon2 sequential vector");

    for (n, state) in edge_states().into_iter().enumerate() {
        let mut lib = state;
        let mut via_lib = state;
        let mut via_alt = state;
        poseidon.permute_mut(&mut lib);
        permute_with_lib_internal(&mut via_lib, &init, &term);
        permute_with_alt_internal(&mut via_alt, &init, &term);
        assert_same(&format!("permute lib-internal edge {n}"), &lib, &via_lib);
        assert_same(&format!("permute alt-internal edge {n}"), &lib, &via_alt);
    }
    for n in 0..256 {
        let state = random_state(&mut rng);
        let mut lib = state;
        let mut via_lib = state;
        let mut via_alt = state;
        poseidon.permute_mut(&mut lib);
        permute_with_lib_internal(&mut via_lib, &init, &term);
        permute_with_alt_internal(&mut via_alt, &init, &term);
        assert_same(&format!("permute lib-internal random {n}"), &lib, &via_lib);
        assert_same(&format!("permute alt-internal random {n}"), &lib, &via_alt);
    }
}

fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_iters(iters: u64, mut body: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        body();
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

struct TrialStats {
    old_ns: f64,
    new_ns: f64,
    ratio: f64,
    min_ratio: f64,
    max_ratio: f64,
}

fn finish(label: &str, olds: &[f64], news: &[f64]) -> TrialStats {
    let ratios: Vec<f64> = olds.iter().zip(news).map(|(o, n)| o / n).collect();
    let stats = TrialStats {
        old_ns: median(olds),
        new_ns: median(news),
        ratio: median(&ratios),
        min_ratio: ratios.iter().copied().fold(f64::INFINITY, f64::min),
        max_ratio: ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    };
    println!(
        "{label} OLD_NS {:.2} NEW_NS {:.2} RATIO {:.4} MIN_RATIO {:.4} MAX_RATIO {:.4}",
        stats.old_ns, stats.new_ns, stats.ratio, stats.min_ratio, stats.max_ratio
    );
    for (i, (o, n)) in olds.iter().zip(news).enumerate() {
        println!(
            "{label} trial {i} old {:.2} new {:.2} ratio {:.4}",
            o,
            n,
            o / n
        );
    }
    stats
}

fn bench_base() -> [Goldilocks; SPONGE_WIDTH] {
    core::array::from_fn(|i| Goldilocks::new(0x9E37_79B9_7F4A_7C15u64.wrapping_mul((i as u64) + 3)))
}

fn bench_layer() -> TrialStats {
    let base = bench_base();
    let mut sold = base;
    let mut snew = base;
    for _ in 0..100_000 {
        lib_internal_linear_layer(black_box(&mut sold), &DIAG);
        alt_internal_linear_layer(black_box(&mut snew));
    }
    let mut olds = [0f64; TRIALS];
    let mut news = [0f64; TRIALS];
    for trial in 0..TRIALS {
        let mut left = base;
        let mut right = base;
        let old_first = trial % 2 == 0;
        let run_old = |state: &mut [Goldilocks; SPONGE_WIDTH]| {
            time_iters(LAYER_ITERS, || {
                lib_internal_linear_layer(black_box(state), &DIAG);
            })
        };
        let run_new = |state: &mut [Goldilocks; SPONGE_WIDTH]| {
            time_iters(LAYER_ITERS, || {
                alt_internal_linear_layer(black_box(state));
            })
        };
        if old_first {
            olds[trial] = run_old(&mut left);
            news[trial] = run_new(&mut right);
        } else {
            news[trial] = run_new(&mut right);
            olds[trial] = run_old(&mut left);
        }
        black_box(canon(&left));
        black_box(canon(&right));
    }
    finish("LAYER", &olds, &news)
}

fn bench_round() -> TrialStats {
    let base = bench_base();
    let rc = INTERNAL_G[7];
    let mut sold = base;
    let mut snew = base;
    for _ in 0..50_000 {
        lib_internal_round(black_box(&mut sold), rc, &DIAG);
        alt_internal_round(black_box(&mut snew), rc);
    }
    let mut olds = [0f64; TRIALS];
    let mut news = [0f64; TRIALS];
    for trial in 0..TRIALS {
        let mut left = base;
        let mut right = base;
        let run_old = |state: &mut [Goldilocks; SPONGE_WIDTH]| {
            time_iters(ROUND_ITERS, || {
                lib_internal_round(black_box(state), rc, &DIAG);
            })
        };
        let run_new = |state: &mut [Goldilocks; SPONGE_WIDTH]| {
            time_iters(ROUND_ITERS, || {
                alt_internal_round(black_box(state), rc);
            })
        };
        if trial % 2 == 0 {
            olds[trial] = run_old(&mut left);
            news[trial] = run_new(&mut right);
        } else {
            news[trial] = run_new(&mut right);
            olds[trial] = run_old(&mut left);
        }
        black_box(canon(&left));
        black_box(canon(&right));
    }
    finish("ROUND", &olds, &news)
}

fn bench_section() -> TrialStats {
    let base = bench_base();
    let mut sold = base;
    let mut snew = base;
    for _ in 0..2_000 {
        lib_internal_section(black_box(&mut sold), &INTERNAL_G, &DIAG);
        alt_internal_section(black_box(&mut snew), &INTERNAL_G);
    }
    let mut olds = [0f64; TRIALS];
    let mut news = [0f64; TRIALS];
    for trial in 0..TRIALS {
        let mut left = base;
        let mut right = base;
        let run_old = |state: &mut [Goldilocks; SPONGE_WIDTH]| {
            time_iters(SECTION_ITERS, || {
                lib_internal_section(black_box(state), &INTERNAL_G, &DIAG);
            })
        };
        let run_new = |state: &mut [Goldilocks; SPONGE_WIDTH]| {
            time_iters(SECTION_ITERS, || {
                alt_internal_section(black_box(state), &INTERNAL_G);
            })
        };
        if trial % 2 == 0 {
            olds[trial] = run_old(&mut left);
            news[trial] = run_new(&mut right);
        } else {
            news[trial] = run_new(&mut right);
            olds[trial] = run_old(&mut left);
        }
        black_box(canon(&left));
        black_box(canon(&right));
    }
    finish("SECTION", &olds, &news)
}

#[test]
fn r2_04_internal_parity_and_bench() {
    assert_canonical_equality();
    println!("ASSERTS canonical equality passed");
    let layer = bench_layer();
    let round = bench_round();
    let section = bench_section();
    black_box((layer.ratio, round.ratio, section.ratio));
}
