//! Phase 1.01: Goldilocks field add — library vs branchless carry chain.

use qp_poseidon_core::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

/// Goldilocks prime p = 2^64 - 2^32 + 1
const P: u64 = 0xFFFF_FFFF_0000_0001;
/// Two's complement of P: 2^64 - P = 2^32 - 1
const NEG_ORDER: u64 = P.wrapping_neg();

/// Reference: `qp-poseidon-core` `Goldilocks` addition.
#[inline(always)]
fn old_add(x: u64, y: u64) -> Goldilocks {
    Goldilocks::new(x) + Goldilocks::new(y)
}

/// Branchless copy of Goldilocks `Add` (no `assume` / rare double-overflow branch).
#[inline(always)]
fn fast_add(x: u64, y: u64) -> u64 {
    let (sum, over) = x.overflowing_add(y);
    let (sum, over) = sum.overflowing_add(over as u64 * NEG_ORDER);
    sum.wrapping_add(over as u64 * NEG_ORDER)
}

/// Branchless copy of Goldilocks `Sub` (parity helper; not timed here).
#[inline(always)]
fn fast_sub(x: u64, y: u64) -> u64 {
    let (diff, under) = x.overflowing_sub(y);
    let (diff, under) = diff.overflowing_sub(under as u64 * NEG_ORDER);
    diff.wrapping_sub(under as u64 * NEG_ORDER)
}

fn assert_add_matches(x: u64, y: u64) {
    let expected = old_add(x, y);
    let got = Goldilocks::new(fast_add(x, y));
    assert_eq!(expected, got, "add mismatch for x={x:#x} y={y:#x}");
    let sub_expected = Goldilocks::new(x) - Goldilocks::new(y);
    let sub_got = Goldilocks::new(fast_sub(x, y));
    assert_eq!(sub_expected, sub_got, "sub mismatch for x={x:#x} y={y:#x}");
}

fn edge_inputs() -> Vec<(u64, u64)> {
    let anchors = [
        0u64,
        1,
        P - 1,
        P,
        P + 1,
        u64::MAX,
        NEG_ORDER,
        NEG_ORDER - 1,
        P - 2,
        1 << 32,
        (1 << 32) - 1,
        0xc002e770975b1607,
        0xbca51a8dfe14593a,
    ];
    let mut pairs = Vec::new();
    for &a in &anchors {
        for &b in &anchors {
            pairs.push((a, b));
        }
    }
    pairs
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn p1_01_goldilocks_add_parity_and_bench() {
    for (x, y) in edge_inputs() {
        assert_add_matches(x, y);
    }

    let mut rng = 0xDEAD_BEEF_CAFE_BABEu64;
    for _ in 0..200_000 {
        let x = splitmix64(&mut rng);
        let y = splitmix64(&mut rng);
        assert_add_matches(x, y);
    }

    const ITERS: u64 = 2_000_000;
    let mut acc = 0u64;
    let x0 = 0x486dcb55419d475a_u64;
    let y0 = 0x6c1cb2a358cc24f1_u64;

    let t0 = Instant::now();
    for i in 0..ITERS {
        let x = black_box(x0.wrapping_add(i));
        let y = black_box(y0 ^ i.rotate_left(17));
        acc ^= black_box(old_add(x, y).as_canonical_u64());
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let t1 = Instant::now();
    for i in 0..ITERS {
        let x = black_box(x0.wrapping_add(i));
        let y = black_box(y0 ^ i.rotate_left(17));
        acc ^= black_box(fast_add(x, y));
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;

    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.3} NEW_NS {new_ns:.3} RATIO {ratio:.4}");
    black_box(acc);
}
