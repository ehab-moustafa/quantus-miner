//! Phase 1 agent 02: Goldilocks mul / reduce128 parity vs qp-poseidon-core 3.1.0,
//! with a strength-reduced reduction (p = 2^64 - 2^32 + 1).

use qp_poseidon_core::goldilocks::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

const P: u64 = 0xFFFF_FFFF_0000_0001;
const NEG_ORDER: u64 = P.wrapping_neg(); // 2^32 - 1

#[inline(always)]
fn branch_hint() {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "x86",
        target_arch = "x86_64",
    ))]
    unsafe {
        core::arch::asm!("", options(nomem, nostack, preserves_flags));
    }
}

#[inline]
const fn split(x: u128) -> (u64, u64) {
    (x as u64, (x >> 64) as u64)
}

/// Copied from qp-poseidon-core 3.1.0 `goldilocks.rs` (`reduce128`).
#[inline(always)]
fn reduce128_old(x: u128) -> u64 {
    let (x_lo, x_hi) = split(x);
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & NEG_ORDER;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint();
        t0 = t0.wrapping_sub(NEG_ORDER);
    }
    let t1 = x_hi_lo.wrapping_mul(NEG_ORDER);
    add_no_canonicalize(t0, t1)
}

#[inline(always)]
fn mul_old(a: u64, b: u64) -> u64 {
    reduce128_old(u128::from(a) * u128::from(b))
}

/// Strength-reduced fold: `(2^32 - 1) * x = (x << 32) - x` for 32-bit `x`.
#[inline(always)]
fn fast_mul(a: u64, b: u64) -> u64 {
    let prod = (a as u128).wrapping_mul(b as u128);
    let x_lo = prod as u64;
    let x_hi = (prod >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi as u32 as u64;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint();
        t0 = t0.wrapping_sub(NEG_ORDER);
    }
    let t1 = x_hi_lo.wrapping_shl(32).wrapping_sub(x_hi_lo);
    add_no_canonicalize(t0, t1)
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn add_no_canonicalize(x: u64, y: u64) -> u64 {
    let res_wrapped: u64;
    let adjustment: u64;
    unsafe {
        core::arch::asm!(
            "add {0}, {1}",
            "sbb {1:e}, {1:e}",
            inlateout(reg) x => res_wrapped,
            inlateout(reg) y => adjustment,
            options(pure, nomem, nostack),
        );
    }
    res_wrapped.wrapping_add(adjustment)
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
fn add_no_canonicalize(x: u64, y: u64) -> u64 {
    let (res_wrapped, carry) = x.overflowing_add(y);
    res_wrapped.wrapping_add(NEG_ORDER * u64::from(carry))
}

fn assert_fast_eq_goldilocks(a: u64, b: u64) {
    let expect = Goldilocks::new(a) * Goldilocks::new(b);
    assert_eq!(
        Goldilocks::new(fast_mul(a, b)),
        expect,
        "fast_mul({a}, {b})"
    );
    assert_eq!(
        Goldilocks::new(mul_old(a, b)),
        expect,
        "mul_old({a}, {b})"
    );
}

#[test]
fn p1_02_mul_parity_and_timing() {
    let edges: &[u64] = &[
        0,
        1,
        2,
        P - 1,
        P,
        P + 1,
        u64::MAX,
        u64::MAX - 1,
        1 << 32,
        (1 << 32) - 1,
        1 << 63,
        0xc002e770975b1607,
        0xbca51a8dfe14593a,
        18446744069414584319,
        9223372034707292161,
    ];
    for &a in edges {
        for &b in edges {
            assert_fast_eq_goldilocks(a, b);
        }
    }

    let mut x: u64 = 0xdead_beef_cafe_babe;
    for _ in 0..100_000 {
        let a = x;
        x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(0x517c_c1b7_2722_0a95);
        let b = x.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(a);
        assert_fast_eq_goldilocks(a, b);
    }

    const ITERS: u32 = 1_000_000;
    for i in 0..50_000u32 {
        let a = i as u64;
        let b = (i as u64).wrapping_mul(7);
        black_box(mul_old(a, b));
        black_box(fast_mul(a, b));
    }

    let mut acc_old = 0u64;
    let t0 = Instant::now();
    for i in 0..ITERS {
        let a = i as u64 ^ 0xA5A5_A5A5_A5A5_A5A5;
        let b = (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        acc_old ^= black_box(mul_old(a, b));
    }
    let old_ns = t0.elapsed().as_nanos();

    let mut acc_new = 0u64;
    let t1 = Instant::now();
    for i in 0..ITERS {
        let a = i as u64 ^ 0xA5A5_A5A5_A5A5_A5A5;
        let b = (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        acc_new ^= black_box(fast_mul(a, b));
    }
    let new_ns = t1.elapsed().as_nanos();

    assert_eq!(acc_old, acc_new, "timing loops diverged");
    let ratio = old_ns as f64 / new_ns as f64;
    println!("OLD_NS {old_ns} NEW_NS {new_ns} RATIO {ratio:.4}");
}
