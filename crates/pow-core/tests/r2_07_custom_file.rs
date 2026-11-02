//! Re-time `pow_core::custom` against qp-poseidon-core 3.1.0 in release.
//!
//! `external_linear_layer` is private in the library. The copy below is the
//! qp-poseidon-core 3.1.0 body (`poseidon2.rs`), including its `Sum` reduction.
//! Private helpers in `custom.rs` (`apply_mat4`, `external_round`,
//! `internal_linear_layer`) are not visible here; they are covered by
//! `external_linear_layer` and `permute_mut`.

use pow_core::custom::{self, fast_mul};
use qp_poseidon_core::{Goldilocks, Poseidon2, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

const P: u64 = 0xFFFF_FFFF_0000_0001;

const MUL_ITERS: u64 = 16_777_216;
const MUL_TRIALS: usize = 9;
const EXT_STATES: usize = 1_048_576;
const EXT_TRIALS: usize = 9;
const PERM_ITERS: u32 = 20_000;
const PERM_TRIALS: usize = 9;
const FILE_ITERS: u32 = 2_000;
const FILE_TRIALS: usize = 31;

const P3_ZERO_RESULT: [u64; 12] = [
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

const P3_SEQ_RESULT: [u64; 12] = [
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

#[inline(always)]
fn raw_u64(g: Goldilocks) -> u64 {
    // `Goldilocks` is `#[repr(transparent)]` over `u64`. Reading the raw residue
    // keeps canonicalization out of the multiply timer.
    unsafe { core::mem::transmute(g) }
}

fn canon(state: &[Goldilocks; SPONGE_WIDTH]) -> [u64; SPONGE_WIDTH] {
    state.map(|g| g.as_canonical_u64())
}

fn prefer_performance_core() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        // QOS_CLASS_USER_INTERACTIVE, so the trial runs on a performance core.
        let _ = pthread_set_qos_class_self_np(0x21, 0);
    }
}

/// Verbatim `apply_mat4` from qp-poseidon-core 3.1.0.
#[inline(always)]
fn library_apply_mat4(x: &mut [Goldilocks; 4]) {
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

/// Verbatim `external_linear_layer` from qp-poseidon-core 3.1.0.
#[inline(always)]
fn library_external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        library_apply_mat4(chunk.try_into().unwrap());
    }

    let sums: [Goldilocks; 4] =
        core::array::from_fn(|k| (0..SPONGE_WIDTH).step_by(4).map(|j| state[j + k]).sum());

    for (i, elem) in state.iter_mut().enumerate() {
        *elem += sums[i % 4];
    }
}

fn assert_mul(a: u64, b: u64) {
    let expect = (Goldilocks::new(a) * Goldilocks::new(b)).as_canonical_u64();
    let got = Goldilocks::new(fast_mul(a, b)).as_canonical_u64();
    assert_eq!(got, expect, "fast_mul({a:#x}, {b:#x})");
}

fn mul_edges() -> &'static [u64] {
    &[
        0,
        1,
        2,
        u32::MAX as u64,
        (1u64 << 32) - 1,
        1u64 << 32,
        (1u64 << 32) + 1,
        1u64 << 63,
        P - 2,
        P - 1,
        P,
        P + 1,
        P + 2,
        u64::MAX - 1,
        u64::MAX,
        0xc002_e770_975b_1607,
        0xbca5_1a8d_fe14_593a,
        18_446_744_069_414_584_319,
        9_223_372_034_707_292_161,
        0xFFFF_FFFF_0000_0000,
    ]
}

fn assert_ext(input: [Goldilocks; SPONGE_WIDTH]) {
    let mut library = input;
    let mut custom = input;
    library_external_linear_layer(&mut library);
    custom::external_linear_layer(&mut custom);
    assert_eq!(canon(&library), canon(&custom), "external linear layer");
}

fn assert_permute(poseidon: &Poseidon2, input: [Goldilocks; SPONGE_WIDTH]) {
    let mut library = input;
    let mut custom = input;
    poseidon.permute_mut(&mut library);
    custom::permute_mut(&mut custom);
    assert_eq!(canon(&library), canon(&custom), "permute_mut");
}

fn random_state(seed: &mut u64) -> [Goldilocks; SPONGE_WIDTH] {
    let mut state = [Goldilocks::ZERO; SPONGE_WIDTH];
    for lane in &mut state {
        *seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        *lane = Goldilocks::from_u64(*seed);
    }
    state
}

fn fill_ext_states() -> Vec<[Goldilocks; SPONGE_WIDTH]> {
    let mut seed = 0xA5A5_5A5A_1234_5678u64;
    (0..EXT_STATES).map(|_| random_state(&mut seed)).collect()
}

#[inline(never)]
fn time_library_mul() -> f64 {
    let mut acc = [0u64; 8];
    let start = Instant::now();
    let mut i = 0u64;
    while i < MUL_ITERS {
        let lane = (i as usize) & 7;
        let a = i ^ 0xA5A5_A5A5_A5A5_A5A5;
        let b = i.wrapping_mul(0xD1B5_4A32_D192_ED03);
        let prod = Goldilocks::new(a) * Goldilocks::new(b);
        acc[lane] = raw_u64(prod).wrapping_add(acc[lane]);
        i += 1;
    }
    black_box(acc);
    start.elapsed().as_nanos() as f64 / MUL_ITERS as f64
}

#[inline(never)]
fn time_fast_mul() -> f64 {
    let mut acc = [0u64; 8];
    let start = Instant::now();
    let mut i = 0u64;
    while i < MUL_ITERS {
        let lane = (i as usize) & 7;
        let a = i ^ 0xA5A5_A5A5_A5A5_A5A5;
        let b = i.wrapping_mul(0xD1B5_4A32_D192_ED03);
        acc[lane] = fast_mul(a, b).wrapping_add(acc[lane]);
        i += 1;
    }
    black_box(acc);
    start.elapsed().as_nanos() as f64 / MUL_ITERS as f64
}

#[inline(never)]
fn time_library_ext(
    states: &[[Goldilocks; SPONGE_WIDTH]],
    buf: &mut [[Goldilocks; SPONGE_WIDTH]],
) -> f64 {
    buf.copy_from_slice(states);
    black_box(buf.as_mut_ptr());
    let mut sink = 0u64;
    let start = Instant::now();
    for state in buf.iter_mut() {
        library_external_linear_layer(state);
        sink ^= raw_u64(state[0]);
    }
    black_box(sink);
    start.elapsed().as_nanos() as f64 / buf.len() as f64
}

#[inline(never)]
fn time_custom_ext(
    states: &[[Goldilocks; SPONGE_WIDTH]],
    buf: &mut [[Goldilocks; SPONGE_WIDTH]],
) -> f64 {
    buf.copy_from_slice(states);
    black_box(buf.as_mut_ptr());
    let mut sink = 0u64;
    let start = Instant::now();
    for state in buf.iter_mut() {
        custom::external_linear_layer(state);
        sink ^= raw_u64(state[0]);
    }
    black_box(sink);
    start.elapsed().as_nanos() as f64 / buf.len() as f64
}

#[inline(never)]
fn time_library_perm(poseidon: &Poseidon2, iters: u32) -> f64 {
    let mut state = [Goldilocks::new(1); SPONGE_WIDTH];
    let start = Instant::now();
    for _ in 0..iters {
        poseidon.permute_mut(&mut state);
    }
    black_box(state);
    start.elapsed().as_nanos() as f64 / f64::from(iters)
}

#[inline(never)]
fn time_custom_perm(iters: u32) -> f64 {
    let mut state = [Goldilocks::new(1); SPONGE_WIDTH];
    let start = Instant::now();
    for _ in 0..iters {
        custom::permute_mut(&mut state);
    }
    black_box(state);
    start.elapsed().as_nanos() as f64 / f64::from(iters)
}

fn paired_trials(
    n: usize,
    mut library: impl FnMut() -> f64,
    mut custom: impl FnMut() -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut old = Vec::with_capacity(n);
    let mut new = Vec::with_capacity(n);
    for i in 0..n {
        if i % 2 == 0 {
            old.push(library());
            new.push(custom());
        } else {
            new.push(custom());
            old.push(library());
        }
    }
    (old, new)
}

fn stats(xs: &mut [f64]) -> (f64, f64, f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let at = |p: f64| xs[((n - 1) as f64 * p).round() as usize];
    (xs[0], at(0.10), xs[n / 2], at(0.90), xs[n - 1])
}

fn report(item: &str, mut old: Vec<f64>, mut new: Vec<f64>) {
    let (old_min, old_p10, old_med, old_p90, old_max) = stats(&mut old);
    let (new_min, new_p10, new_med, new_p90, new_max) = stats(&mut new);
    let ratio = old_med / new_med;
    let faster_pct = (old_med - new_med) / old_med * 100.0;
    // Equal is established by the asserts before timing. Keep only if the
    // median custom time is more than 3% below the library median.
    let verdict = if faster_pct > 3.0 {
        "Keep"
    } else {
        "Prefer library"
    };
    println!(
        "R2_ROW item={item} old_ns={old_med:.4} new_ns={new_med:.4} ratio={ratio:.4} faster_pct={faster_pct:.4} verdict={verdict} trials={} old_min={old_min:.4} old_p10={old_p10:.4} old_p90={old_p90:.4} old_max={old_max:.4} new_min={new_min:.4} new_p10={new_p10:.4} new_p90={new_p90:.4} new_max={new_max:.4}",
        old.len()
    );
    assert!(
        old_med > 0.05 && new_med > 0.05,
        "{item} collapsed (old {old_med}, new {new_med}); timer loop was removed"
    );
    if item.contains("permute") || item.contains("whole") {
        assert!(
            old_med < 20_000.0 && new_med < 20_000.0,
            "{item} unexpectedly slow"
        );
    }
}

#[test]
fn r2_07_custom_file() {
    prefer_performance_core();
    let poseidon = Poseidon2::new();

    let edges = mul_edges();
    for &a in edges {
        for &b in edges {
            assert_mul(a, b);
        }
    }
    let mut mul_seed = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..4_096 {
        mul_seed = mul_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x517C_C1B7_2722_0A95);
        let a = mul_seed;
        mul_seed = mul_seed
            .wrapping_mul(0x5851_F42D_4C95_7F2D)
            .wrapping_add(0xA5A5_A5A5_A5A5_A5A5);
        assert_mul(a, mul_seed);
    }

    let sequential: [Goldilocks; SPONGE_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64((i + 1) as u64));
    let mut library_seq = sequential;
    library_external_linear_layer(&mut library_seq);
    let ext_expected = [
        144u64, 156, 168, 148, 172, 184, 196, 176, 200, 212, 224, 204,
    ];
    assert_eq!(
        canon(&library_seq),
        ext_expected,
        "library external layer vector"
    );
    let mut custom_seq = sequential;
    custom::external_linear_layer(&mut custom_seq);
    assert_eq!(
        canon(&custom_seq),
        ext_expected,
        "custom external layer vector"
    );

    assert_ext([Goldilocks::ZERO; SPONGE_WIDTH]);
    assert_ext([Goldilocks::new(1); SPONGE_WIDTH]);
    assert_ext([Goldilocks::new(P - 1); SPONGE_WIDTH]);
    assert_ext([Goldilocks::new(P); SPONGE_WIDTH]);
    assert_ext([Goldilocks::new(u64::MAX); SPONGE_WIDTH]);
    assert_ext(sequential);

    let mut zero = [Goldilocks::ZERO; SPONGE_WIDTH];
    poseidon.permute_mut(&mut zero);
    assert_eq!(canon(&zero), P3_ZERO_RESULT, "library permute zero vector");
    let mut custom_zero = [Goldilocks::ZERO; SPONGE_WIDTH];
    custom::permute_mut(&mut custom_zero);
    assert_eq!(
        canon(&custom_zero),
        P3_ZERO_RESULT,
        "custom permute zero vector"
    );

    let mut seq_perm = sequential;
    poseidon.permute_mut(&mut seq_perm);
    assert_eq!(
        canon(&seq_perm),
        P3_SEQ_RESULT,
        "library permute sequential vector"
    );
    let mut custom_seq_perm = sequential;
    custom::permute_mut(&mut custom_seq_perm);
    assert_eq!(
        canon(&custom_seq_perm),
        P3_SEQ_RESULT,
        "custom permute sequential vector"
    );

    let mut seed = 0xC0FFEE_u64;
    for _ in 0..64 {
        let state = random_state(&mut seed);
        assert_ext(state);
        assert_permute(&poseidon, state);
    }

    for start in [
        [Goldilocks::ZERO; SPONGE_WIDTH],
        sequential,
        [Goldilocks::new(P); SPONGE_WIDTH],
    ] {
        let mut library = start;
        let mut custom = start;
        for _ in 0..FILE_ITERS {
            poseidon.permute_mut(&mut library);
            custom::permute_mut(&mut custom);
        }
        assert_eq!(canon(&library), canon(&custom), "2000 chained permutes");
    }

    println!("R2_ASSERTS pass");

    let ext_states = fill_ext_states();
    let mut ext_buf = ext_states.clone();
    // Warm both paths before the recorded trials.
    black_box(time_library_mul());
    black_box(time_fast_mul());
    black_box(time_library_ext(&ext_states, &mut ext_buf));
    black_box(time_custom_ext(&ext_states, &mut ext_buf));
    black_box(time_library_perm(&poseidon, 4_000));
    black_box(time_custom_perm(4_000));

    let (mul_old, mul_new) = paired_trials(MUL_TRIALS, time_library_mul, time_fast_mul);
    report("fast_mul", mul_old, mul_new);

    let mut ext_old = Vec::with_capacity(EXT_TRIALS);
    let mut ext_new = Vec::with_capacity(EXT_TRIALS);
    for i in 0..EXT_TRIALS {
        if i % 2 == 0 {
            ext_old.push(time_library_ext(&ext_states, &mut ext_buf));
            ext_new.push(time_custom_ext(&ext_states, &mut ext_buf));
        } else {
            ext_new.push(time_custom_ext(&ext_states, &mut ext_buf));
            ext_old.push(time_library_ext(&ext_states, &mut ext_buf));
        }
    }
    report("external_linear_layer", ext_old, ext_new);

    let (perm_old, perm_new) = paired_trials(
        PERM_TRIALS,
        || time_library_perm(&poseidon, PERM_ITERS),
        || time_custom_perm(PERM_ITERS),
    );
    report("permute_mut", perm_old, perm_new);

    let (file_old, file_new) = paired_trials(
        FILE_TRIALS,
        || time_library_perm(&poseidon, FILE_ITERS),
        || time_custom_perm(FILE_ITERS),
    );
    report("whole_file_2000_permute", file_old, file_new);

    // One unrepeated 2000-call sample, same shape as the old 573 ns/op figure.
    let single_old = time_library_perm(&poseidon, FILE_ITERS);
    let single_new = time_custom_perm(FILE_ITERS);
    println!("R2_SINGLE_SHOT old_ns={single_old:.4} new_ns={single_new:.4}");
}
