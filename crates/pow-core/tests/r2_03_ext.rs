//! Re-challenge of Poseidon2 external linear algebra (qp-poseidon-core 3.1.0).
//!
//! Timed separately:
//! 1. one 4-lane `apply_mat4` versus a fresh factored sum (branchless limbs, and
//!    the same factorization accumulated in `u128` with one reduction per lane)
//! 2. the width-12 external layer versus that alternative composed as three mat4s
//!    plus the circulant column fold, versus `custom::external_linear_layer`
//!
//! The fold is not timed on its own and added to mat4. Each external candidate is
//! one inlined call. Adopt only when values match and the robust ratio is clearly
//! above 1.03 (median and second-smallest-old / second-largest-new).

use pow_core::custom;
use qp_poseidon_core::{Goldilocks, SPONGE_WIDTH};
use std::hint::black_box;
use std::time::Instant;

const P: u64 = 0xFFFF_FFFF_0000_0001;
const NEG_ORDER: u64 = P.wrapping_neg(); // 2^32 - 1
const BATCH: usize = 256;
const TRIALS: usize = 5;
const MAT4_ITERS: u64 = 40_000_000;
const EXT_ITERS: u64 = 12_000_000;

/// `Goldilocks` is `#[repr(transparent)]` over its internal limb.
#[inline(always)]
fn limb(g: Goldilocks) -> u64 {
    unsafe { std::mem::transmute(g) }
}

#[inline(always)]
fn pack(v: u64) -> Goldilocks {
    Goldilocks::new(v)
}

/// Branchless Goldilocks add. Same residue as `Goldilocks::add` for every limb pair.
#[inline(always)]
fn gadd(x: u64, y: u64) -> u64 {
    let (sum, c0) = x.overflowing_add(y);
    let (sum, c1) = sum.overflowing_add(if c0 { NEG_ORDER } else { 0 });
    sum.wrapping_add(if c1 { NEG_ORDER } else { 0 })
}

/// Integer value `hi * 2^64 + lo` reduced mod P. Exact for `hi <= 28`.
#[inline(always)]
fn reduce_factored(acc: u128) -> u64 {
    let lo = acc as u64;
    let hi = (acc >> 64) as u64;
    // 2^64 ≡ 2^32 - 1 (mod P), and hi is far below 2^32 on these matrices.
    let hi_term = hi.wrapping_shl(32).wrapping_sub(hi);
    let (sum, carry) = lo.overflowing_add(hi_term);
    sum.wrapping_add(if carry { NEG_ORDER } else { 0 })
}

/// Library `apply_mat4` (qp-poseidon-core 3.1.0). Matrix is circ(2, 3, 1, 1).
#[inline(always)]
fn lib_apply_mat4(x: &mut [Goldilocks; 4]) {
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

/// Fresh factored sum on limbs: share `t01`, `t23`, and the total sum.
#[inline(always)]
fn limb_apply_mat4(x: &mut [Goldilocks; 4]) {
    let a = limb(x[0]);
    let b = limb(x[1]);
    let c = limb(x[2]);
    let d = limb(x[3]);
    let t01 = gadd(a, b);
    let t23 = gadd(c, d);
    let sum = gadd(t01, t23);
    let t01123 = gadd(sum, b);
    let t01233 = gadd(sum, d);
    x[0] = pack(gadd(t01123, t01));
    x[1] = pack(gadd(t01123, gadd(c, c)));
    x[2] = pack(gadd(t01233, t23));
    x[3] = pack(gadd(t01233, gadd(a, a)));
}

/// Same factorization over the integers. One modular reduction per output lane.
#[inline(always)]
fn partial_mat4(a: u128, b: u128, c: u128, d: u128) -> [u128; 4] {
    let t01 = a + b;
    let t23 = c + d;
    let sum = t01 + t23;
    let t01123 = sum + b;
    let t01233 = sum + d;
    [
        t01123 + t01,
        t01123 + (c << 1),
        t01233 + t23,
        t01233 + (a << 1),
    ]
}

#[inline(always)]
fn u128_apply_mat4(x: &mut [Goldilocks; 4]) {
    let y = partial_mat4(
        limb(x[0]) as u128,
        limb(x[1]) as u128,
        limb(x[2]) as u128,
        limb(x[3]) as u128,
    );
    x[0] = pack(reduce_factored(y[0]));
    x[1] = pack(reduce_factored(y[1]));
    x[2] = pack(reduce_factored(y[2]));
    x[3] = pack(reduce_factored(y[3]));
}

/// Library `external_linear_layer`: three chunk mat4s, then the circulant sums.
#[inline(always)]
fn lib_external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        lib_apply_mat4(chunk.try_into().unwrap());
    }
    let sums: [Goldilocks; 4] =
        core::array::from_fn(|k| (0..SPONGE_WIDTH).step_by(4).map(|j| state[j + k]).sum());
    for (i, elem) in state.iter_mut().enumerate() {
        *elem += sums[i % 4];
    }
}

/// Three limb mat4s plus the circulant column fold, one function.
#[inline(always)]
fn limb_external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    limb_apply_mat4((&mut state[0..4]).try_into().unwrap());
    limb_apply_mat4((&mut state[4..8]).try_into().unwrap());
    limb_apply_mat4((&mut state[8..12]).try_into().unwrap());
    column_fold(state);
}

/// Three integer mat4s plus the column fold, reduced only on the twelve outputs.
#[inline(always)]
fn u128_external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    let y0 = partial_mat4(
        limb(state[0]) as u128,
        limb(state[1]) as u128,
        limb(state[2]) as u128,
        limb(state[3]) as u128,
    );
    let y1 = partial_mat4(
        limb(state[4]) as u128,
        limb(state[5]) as u128,
        limb(state[6]) as u128,
        limb(state[7]) as u128,
    );
    let y2 = partial_mat4(
        limb(state[8]) as u128,
        limb(state[9]) as u128,
        limb(state[10]) as u128,
        limb(state[11]) as u128,
    );
    let s = [
        y0[0] + y1[0] + y2[0],
        y0[1] + y1[1] + y2[1],
        y0[2] + y1[2] + y2[2],
        y0[3] + y1[3] + y2[3],
    ];
    let rows = [y0, y1, y2];
    for chunk in 0..3 {
        for lane in 0..4 {
            state[chunk * 4 + lane] = pack(reduce_factored(rows[chunk][lane] + s[lane]));
        }
    }
}

#[inline(always)]
fn column_fold(state: &mut [Goldilocks; SPONGE_WIDTH]) {
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

fn canon4(x: [Goldilocks; 4]) -> [u64; 4] {
    x.map(|g| g.as_canonical_u64())
}

fn canon12(x: [Goldilocks; SPONGE_WIDTH]) -> [u64; SPONGE_WIDTH] {
    x.map(|g| g.as_canonical_u64())
}

fn check_mat4(input: [Goldilocks; 4]) {
    let mut lib = input;
    let mut limb_out = input;
    let mut wide = input;
    lib_apply_mat4(&mut lib);
    limb_apply_mat4(&mut limb_out);
    u128_apply_mat4(&mut wide);
    let expect = canon4(lib);
    assert_eq!(canon4(limb_out), expect, "limb mat4 {input:?}");
    assert_eq!(canon4(wide), expect, "u128 mat4 {input:?}");
}

fn check_ext(input: [Goldilocks; SPONGE_WIDTH]) {
    let mut lib = input;
    let mut limb_out = input;
    let mut wide = input;
    let mut custom_out = input;
    lib_external_linear_layer(&mut lib);
    limb_external_linear_layer(&mut limb_out);
    u128_external_linear_layer(&mut wide);
    custom::external_linear_layer(&mut custom_out);
    let expect = canon12(lib);
    assert_eq!(canon12(limb_out), expect, "limb external {input:?}");
    assert_eq!(canon12(wide), expect, "u128 external {input:?}");
    assert_eq!(canon12(custom_out), expect, "custom external {input:?}");
}

fn xorshift(state: &mut u64) -> u64 {
    let mut z = *state;
    z ^= z << 13;
    z ^= z >> 7;
    z ^= z << 17;
    *state = z;
    z
}

fn set_perf_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(class: u32, relative: i32) -> i32;
    }
    // QOS_CLASS_USER_INTERACTIVE, so samples stay on a performance core.
    unsafe {
        pthread_set_qos_class_self_np(0x21, 0);
    }
}

#[inline(never)]
fn bench_mat4_lib(batch: &[[Goldilocks; 4]; BATCH], iters: u64) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    for _ in 0..iters {
        let mut s = black_box(batch[i]);
        lib_apply_mat4(&mut s);
        acc ^= fold_lanes4(black_box(s));
        i = (i + 1) & (BATCH - 1);
    }
    acc
}

#[inline(never)]
fn bench_mat4_limb(batch: &[[Goldilocks; 4]; BATCH], iters: u64) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    for _ in 0..iters {
        let mut s = black_box(batch[i]);
        limb_apply_mat4(&mut s);
        acc ^= fold_lanes4(black_box(s));
        i = (i + 1) & (BATCH - 1);
    }
    acc
}

#[inline(never)]
fn bench_mat4_u128(batch: &[[Goldilocks; 4]; BATCH], iters: u64) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    for _ in 0..iters {
        let mut s = black_box(batch[i]);
        u128_apply_mat4(&mut s);
        acc ^= fold_lanes4(black_box(s));
        i = (i + 1) & (BATCH - 1);
    }
    acc
}

#[inline(never)]
fn bench_ext_lib(batch: &[[Goldilocks; SPONGE_WIDTH]; BATCH], iters: u64) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    for _ in 0..iters {
        let mut s = black_box(batch[i]);
        lib_external_linear_layer(&mut s);
        acc ^= fold_lanes12(black_box(s));
        i = (i + 1) & (BATCH - 1);
    }
    acc
}

#[inline(never)]
fn bench_ext_limb(batch: &[[Goldilocks; SPONGE_WIDTH]; BATCH], iters: u64) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    for _ in 0..iters {
        let mut s = black_box(batch[i]);
        limb_external_linear_layer(&mut s);
        acc ^= fold_lanes12(black_box(s));
        i = (i + 1) & (BATCH - 1);
    }
    acc
}

#[inline(never)]
fn bench_ext_u128(batch: &[[Goldilocks; SPONGE_WIDTH]; BATCH], iters: u64) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    for _ in 0..iters {
        let mut s = black_box(batch[i]);
        u128_external_linear_layer(&mut s);
        acc ^= fold_lanes12(black_box(s));
        i = (i + 1) & (BATCH - 1);
    }
    acc
}

#[inline(never)]
fn bench_ext_custom(batch: &[[Goldilocks; SPONGE_WIDTH]; BATCH], iters: u64) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;
    for _ in 0..iters {
        let mut s = black_box(batch[i]);
        custom::external_linear_layer(&mut s);
        acc ^= fold_lanes12(black_box(s));
        i = (i + 1) & (BATCH - 1);
    }
    acc
}

#[inline(always)]
fn fold_lanes4(s: [Goldilocks; 4]) -> u64 {
    limb(s[0]) ^ limb(s[1]) ^ limb(s[2]) ^ limb(s[3])
}

#[inline(always)]
fn fold_lanes12(s: [Goldilocks; SPONGE_WIDTH]) -> u64 {
    let mut acc = 0u64;
    for lane in s {
        acc ^= limb(lane);
    }
    acc
}

fn one_sample(mut once: impl FnMut() -> u64, iters: u64) -> f64 {
    let start = Instant::now();
    let acc = once();
    let elapsed = start.elapsed();
    black_box(acc);
    elapsed.as_nanos() as f64 / iters as f64
}

fn median(samples: &[f64; TRIALS]) -> f64 {
    let mut sorted = *samples;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[TRIALS / 2]
}

fn second_smallest(samples: &[f64; TRIALS]) -> f64 {
    let mut sorted = *samples;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[1]
}

fn second_largest(samples: &[f64; TRIALS]) -> f64 {
    let mut sorted = *samples;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[TRIALS - 2]
}

fn report(item: &str, old: [f64; TRIALS], new: [f64; TRIALS]) {
    let old_ns = median(&old);
    let new_ns = median(&new);
    let ratio = old_ns / new_ns;
    let robust = second_smallest(&old) / second_largest(&new);
    let verdict = if ratio > 1.03 && robust > 1.03 {
        "Adopt"
    } else {
        "Reject"
    };
    println!(
        "ITEM {item} OLD_NS {old_ns:.4} NEW_NS {new_ns:.4} RATIO {ratio:.4} ROBUST {robust:.4} VERDICT {verdict}"
    );
    println!("SAMPLES_OLD {item} {old:.4?}");
    println!("SAMPLES_NEW {item} {new:.4?}");
}

#[test]
fn r2_03_ext_parity_and_bench() {
    check_mat4([Goldilocks::ZERO; 4]);
    check_mat4([Goldilocks::ONE; 4]);
    let known4 = [Goldilocks::new(1), Goldilocks::new(2), Goldilocks::new(3), Goldilocks::new(4)];
    check_mat4(known4);
    let mut known4_lib = known4;
    lib_apply_mat4(&mut known4_lib);
    assert_eq!(canon4(known4_lib), [15, 18, 21, 16]);

    check_ext([Goldilocks::ZERO; SPONGE_WIDTH]);
    check_ext([Goldilocks::ONE; SPONGE_WIDTH]);
    let mut known12: [Goldilocks; SPONGE_WIDTH] =
        core::array::from_fn(|i| Goldilocks::from_u64((i + 1) as u64));
    let known_input = known12;
    check_ext(known_input);
    lib_external_linear_layer(&mut known12);
    assert_eq!(
        canon12(known12),
        [144, 156, 168, 148, 172, 184, 196, 176, 200, 212, 224, 204]
    );

    check_mat4([Goldilocks::new(u64::MAX); 4]);
    check_mat4([Goldilocks::new(P - 1); 4]);
    check_ext([Goldilocks::new(u64::MAX); SPONGE_WIDTH]);
    check_ext([Goldilocks::new(P - 1); SPONGE_WIDTH]);

    let mut rng = 0x3141_5926_5358_9793_u64;
    for _ in 0..50_000 {
        check_mat4(core::array::from_fn(|_| Goldilocks::new(xorshift(&mut rng))));
    }
    for _ in 0..20_000 {
        check_ext(core::array::from_fn(|_| Goldilocks::new(xorshift(&mut rng))));
    }
    println!("ASSERTS passed (random, all-zero, all-one, edges, known vectors)");

    set_perf_qos();

    let mut rng = 0xA24B_AED4_963E_E407_u64;
    let mat_batch: [[Goldilocks; 4]; BATCH] =
        core::array::from_fn(|_| core::array::from_fn(|_| Goldilocks::new(xorshift(&mut rng))));
    let ext_batch: [[Goldilocks; SPONGE_WIDTH]; BATCH] =
        core::array::from_fn(|_| core::array::from_fn(|_| Goldilocks::new(xorshift(&mut rng))));

    black_box(bench_mat4_lib(&mat_batch, MAT4_ITERS / 8));
    black_box(bench_mat4_limb(&mat_batch, MAT4_ITERS / 8));
    black_box(bench_mat4_u128(&mat_batch, MAT4_ITERS / 8));
    let mut mat_lib = [0.0; TRIALS];
    let mut mat_limb = [0.0; TRIALS];
    let mut mat_u128 = [0.0; TRIALS];
    for trial in 0..TRIALS {
        // Rotate which kernel runs first so a warm or hot core does not stick to one side.
        let order = trial % 3;
        for step in 0..3 {
            let kind = (order + step) % 3;
            let sample = match kind {
                0 => one_sample(|| bench_mat4_lib(&mat_batch, MAT4_ITERS), MAT4_ITERS),
                1 => one_sample(|| bench_mat4_limb(&mat_batch, MAT4_ITERS), MAT4_ITERS),
                _ => one_sample(|| bench_mat4_u128(&mat_batch, MAT4_ITERS), MAT4_ITERS),
            };
            match kind {
                0 => mat_lib[trial] = sample,
                1 => mat_limb[trial] = sample,
                _ => mat_u128[trial] = sample,
            }
        }
    }

    black_box(bench_ext_lib(&ext_batch, EXT_ITERS / 8));
    black_box(bench_ext_limb(&ext_batch, EXT_ITERS / 8));
    black_box(bench_ext_u128(&ext_batch, EXT_ITERS / 8));
    black_box(bench_ext_custom(&ext_batch, EXT_ITERS / 8));
    let mut ext_lib = [0.0; TRIALS];
    let mut ext_limb = [0.0; TRIALS];
    let mut ext_u128 = [0.0; TRIALS];
    let mut ext_custom = [0.0; TRIALS];
    for trial in 0..TRIALS {
        let order = trial % 4;
        for step in 0..4 {
            let kind = (order + step) % 4;
            let sample = match kind {
                0 => one_sample(|| bench_ext_lib(&ext_batch, EXT_ITERS), EXT_ITERS),
                1 => one_sample(|| bench_ext_limb(&ext_batch, EXT_ITERS), EXT_ITERS),
                2 => one_sample(|| bench_ext_u128(&ext_batch, EXT_ITERS), EXT_ITERS),
                _ => one_sample(|| bench_ext_custom(&ext_batch, EXT_ITERS), EXT_ITERS),
            };
            match kind {
                0 => ext_lib[trial] = sample,
                1 => ext_limb[trial] = sample,
                2 => ext_u128[trial] = sample,
                _ => ext_custom[trial] = sample,
            }
        }
    }

    report("4-lane mat4 limb factored-sum", mat_lib, mat_limb);
    report("4-lane mat4 u128 factored-sum", mat_lib, mat_u128);
    report(
        "width-12 external 3x limb mat4 + circulant fold",
        ext_lib,
        ext_limb,
    );
    report(
        "width-12 external fused u128 mat4x3 + circulant fold",
        ext_lib,
        ext_u128,
    );
    report(
        "width-12 external custom::external_linear_layer",
        ext_lib,
        ext_custom,
    );
}
