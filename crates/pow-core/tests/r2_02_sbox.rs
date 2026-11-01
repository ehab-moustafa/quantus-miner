//! Round-2 rebench of the Goldilocks S-box x^7.
//! Library: qp-poseidon-core 3.1.0 `Goldilocks::exp7` (`src/goldilocks.rs`).
//!
//! `exp7` is the addition chain x^2, x^3, x^4, x^7 (four field muls; critical path 3).
//! Every minimal chain for exponent 7 has four multiplications. These runners time
//! that library function and alternate chains on the same inputs.

use qp_poseidon_core::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

const P: u64 = 0xFFFF_FFFF_0000_0001;

#[inline(always)]
fn limb(g: Goldilocks) -> u64 {
    // `Goldilocks` is `#[repr(transparent)]` over its internal u64 limb.
    unsafe { std::mem::transmute(g) }
}

#[inline(always)]
fn lib_exp7(x: Goldilocks) -> Goldilocks {
    x.exp7()
}

/// 1-2-4-6-7. Critical path length 4.
#[inline(always)]
fn chain_x2_x4_x6_x7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    x6 * x
}

/// 1-2-3-4-7. Same algebra as `Goldilocks::exp7`. Critical path length 3.
#[inline(always)]
fn chain_x2_x3_x4_x7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    x3 * x4
}

/// 1-2-3-6-7 (binary exponentiation). Critical path length 4.
#[inline(always)]
fn chain_x2_x3_x6_x7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x3 = x2 * x;
    let x6 = x3 * x3;
    x6 * x
}

/// 1-2-4-5-7. Critical path length 4.
#[inline(always)]
fn chain_x2_x4_x5_x7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x5 = x4 * x;
    x5 * x2
}

/// 1-2-3-5-7. Critical path length 4.
#[inline(always)]
fn chain_x2_x3_x5_x7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    x5 * x2
}

/// Library schedule with the two independent multiplies swapped (x^4 before x^3).
#[inline(always)]
fn chain_x2_x4_x3_x7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x3 = x2 * x;
    x3 * x4
}

macro_rules! define_runner {
    ($fn_name:ident, $map:path) => {
        #[inline(never)]
        fn $fn_name(xs: &[Goldilocks]) -> u64 {
            let mut acc = 0u64;
            for &x in xs {
                acc ^= limb($map(x));
            }
            acc
        }
    };
}

define_runner!(run_exp7, lib_exp7);
define_runner!(run_x2_x4_x6_x7, chain_x2_x4_x6_x7);
define_runner!(run_x2_x3_x4_x7, chain_x2_x3_x4_x7);
define_runner!(run_x2_x3_x6_x7, chain_x2_x3_x6_x7);
define_runner!(run_x2_x4_x5_x7, chain_x2_x4_x5_x7);
define_runner!(run_x2_x3_x5_x7, chain_x2_x3_x5_x7);
define_runner!(run_x2_x4_x3_x7, chain_x2_x4_x3_x7);

fn assert_chains(x: Goldilocks) {
    let raw = limb(x);
    let expect = x.exp7().as_canonical_u64();
    let chains: [(&str, fn(Goldilocks) -> Goldilocks); 6] = [
        ("x2,x4,x6,x7", chain_x2_x4_x6_x7),
        ("x2,x3,x4,x7", chain_x2_x3_x4_x7),
        ("x2,x3,x6,x7", chain_x2_x3_x6_x7),
        ("x2,x4,x5,x7", chain_x2_x4_x5_x7),
        ("x2,x3,x5,x7", chain_x2_x3_x5_x7),
        ("x2,x4,x3,x7", chain_x2_x4_x3_x7),
    ];
    for (name, chain) in chains {
        let got = chain(x).as_canonical_u64();
        assert_eq!(
            got, expect,
            "{name} canonical mismatch on raw input {raw:#x}: got {got:#x} expect {expect:#x}"
        );
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    }
}

fn once(f: fn(&[Goldilocks]) -> u64, xs: &[Goldilocks], reps: u32) -> f64 {
    let mut acc = 0u64;
    let start = Instant::now();
    for _ in 0..reps {
        acc ^= f(black_box(xs));
    }
    let elapsed = start.elapsed().as_nanos() as f64;
    black_box(acc);
    elapsed / (reps as f64 * xs.len() as f64)
}

#[test]
fn r2_02_sbox_parity_and_bench() {
    assert!(
        cfg!(not(debug_assertions)),
        "r2_02_sbox timings require --release"
    );

    const EDGES: &[u64] = &[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        15,
        16,
        31,
        32,
        63,
        64,
        127,
        128,
        255,
        256,
        1_000,
        1 << 16,
        (1 << 16) - 1,
        1 << 31,
        (1 << 31) - 1,
        1 << 32,
        (1 << 32) - 1,
        1 << 33,
        P - 3,
        P - 2,
        P - 1,
        P,
        P + 1,
        P + 2,
        u64::MAX - 1,
        u64::MAX,
        P.wrapping_neg(),
        P.wrapping_neg() - 1,
        0xc002_e770_975b_1607,
        0xbca5_1a8d_fe14_593a,
        13_835_875_475_997_267_463,
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x19a0_71bd_ddc1_6e57,
    ];

    for &v in EDGES {
        assert_chains(Goldilocks::new(v));
    }

    // Library self-check (qp-poseidon-core tests): 2^7 = 128, and the p3 vector.
    assert_eq!(Goldilocks::new(2).exp7().as_canonical_u64(), 128);
    assert_eq!(
        Goldilocks::new(13_835_875_475_997_267_463)
            .exp7()
            .as_canonical_u64(),
        5_716_687_150_516_714_629
    );

    let mut rng = 0x7202_5B0C_E770_7E71u64;
    let mut inputs = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let v = splitmix64(&mut rng);
        let x = Goldilocks::new(v);
        assert_chains(x);
        inputs.push(x);
    }

    println!(
        "ASSERTS_PASSED edges={} random={}",
        EDGES.len(),
        inputs.len()
    );

    let candidates: [(&str, fn(&[Goldilocks]) -> u64); 6] = [
        ("x2,x4,x6,x7", run_x2_x4_x6_x7),
        ("x2,x3,x4,x7", run_x2_x3_x4_x7),
        ("x2,x3,x6,x7", run_x2_x3_x6_x7),
        ("x2,x4,x5,x7", run_x2_x4_x5_x7),
        ("x2,x3,x5,x7", run_x2_x3_x5_x7),
        ("x2,x4,x3,x7", run_x2_x4_x3_x7),
    ];

    // Warm the code and the P-core before choosing a repetition count.
    for _ in 0..3 {
        black_box(run_exp7(black_box(inputs.as_slice())));
        for (_, run) in candidates {
            black_box(run(black_box(inputs.as_slice())));
        }
    }

    let probe_reps = 80u32;
    let probe_start = Instant::now();
    let mut probe_acc = 0u64;
    for _ in 0..probe_reps {
        probe_acc ^= run_exp7(black_box(inputs.as_slice()));
    }
    let probe_ns = probe_start.elapsed().as_nanos().max(1);
    black_box(probe_acc);
    let per_rep_ns = probe_ns as f64 / f64::from(probe_reps);
    // ~40 ms per sample, clock read outside the inner loop.
    let reps = ((40_000_000.0 / per_rep_ns).round() as u32).clamp(30, 20_000);

    const TRIALS: usize = 8;
    let mut exp7_samples = Vec::with_capacity(TRIALS);
    let mut chain_samples = vec![Vec::with_capacity(TRIALS); candidates.len()];

    for trial in 0..TRIALS {
        // Rotate which routine runs first so a warm-up edge is not stuck on exp7.
        let start = trial % (candidates.len() + 1);
        for step in 0..(candidates.len() + 1) {
            let slot = (start + step) % (candidates.len() + 1);
            if slot == 0 {
                exp7_samples.push(once(run_exp7, &inputs, reps));
            } else {
                let idx = slot - 1;
                chain_samples[idx].push(once(candidates[idx].1, &inputs, reps));
            }
        }
    }

    let old_ns = median(&exp7_samples);
    let old_min = exp7_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let old_max = exp7_samples.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        old_ns > 0.05,
        "exp7 timing collapsed ({old_ns} ns/op); measurement is not live"
    );
    println!(
        "R2_EXP7 ns/op median={old_ns:.4} min={old_min:.4} max={old_max:.4} reps={reps} inputs={}",
        inputs.len()
    );

    for (i, (name, _)) in candidates.iter().enumerate() {
        let samples = &chain_samples[i];
        let new_ns = median(samples);
        assert!(
            new_ns > 0.05,
            "{name} timing collapsed ({new_ns} ns/op); measurement is not live"
        );
        let ratio = old_ns / new_ns;
        let mut paired = Vec::with_capacity(TRIALS);
        for t in 0..TRIALS {
            paired.push(exp7_samples[t] / samples[t]);
        }
        let paired_med = median(&paired);
        let paired_min = paired.iter().copied().fold(f64::INFINITY, f64::min);
        let paired_max = paired.iter().copied().fold(0.0_f64, f64::max);
        println!(
            "R2_CHAIN {name} ns/op median={new_ns:.4} ratio_medians={ratio:.4} paired_median={paired_med:.4} paired_min={paired_min:.4} paired_max={paired_max:.4}"
        );
    }
}
