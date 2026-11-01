//! Phase 1 agent 03: Goldilocks S-box x^7 vs qp-poseidon-core `exp7`.
//! Source: qp-poseidon-core-3.1.0/src/goldilocks.rs

use qp_poseidon_core::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Library path (baseline timing).
#[inline(never)]
fn old_sbox(x: Goldilocks) -> Goldilocks {
    x.exp7()
}

/// Copied multiply chain from qp-poseidon-core-3.1.0 `Goldilocks::exp7`.
#[inline(never)]
fn ref_exp7(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    x3 * x4
}

/// x^7 = x^2 * x^2 * x^2 * x (four field muls; addition-chain length 4 is minimal for 7).
#[inline(never)]
fn fast_sbox(x: Goldilocks) -> Goldilocks {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    x6 * x
}

fn assert_sbox_pair(x: Goldilocks) {
    let y = old_sbox(x);
    assert_eq!(ref_exp7(x), y, "ref_exp7 diverged from library for {x:?}");
    assert_eq!(fast_sbox(x), y, "fast_sbox diverged for input {}", x.as_canonical_u64());
}

#[test]
fn p1_03_sbox_parity_and_bench() {
    for &v in &[
        0u64,
        1,
        2,
        3,
        7,
        0xffff_ffff,
        P - 1,
        P,
        P + 1,
        u64::MAX,
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
    ] {
        assert_sbox_pair(Goldilocks::from_u64(v));
    }

    let mut rng_state = 0x9e3779b97f4a7c15u64;
    let mut next_u64 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    for _ in 0..50_000 {
        assert_sbox_pair(Goldilocks::from_u64(next_u64()));
    }

    let bench_x = Goldilocks::from_u64(0x19a071bdddc16e57);

    let t0 = Instant::now();
    for _ in 0..500_000 {
        black_box(old_sbox(black_box(bench_x)));
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / 500_000.0;

    let t1 = Instant::now();
    for _ in 0..500_000 {
        black_box(fast_sbox(black_box(bench_x)));
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / 500_000.0;

    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.2} NEW_NS {new_ns:.2} RATIO {ratio:.4}");
}
