//! Phase 1 agent 04: Poseidon2 external 4×4 MDS (`apply_mat4`) vs factored fast path.
//! Source: qp-poseidon-core-3.1.0/src/poseidon2.rs

use qp_poseidon_core::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

const P: u64 = 0xFFFF_FFFF_0000_0001;
const NEG_ORDER: u64 = P.wrapping_neg();

#[inline(always)]
fn limb(g: Goldilocks) -> u64 {
    // `Goldilocks` is `#[repr(transparent)]` over the internal limb.
    unsafe { std::mem::transmute(g) }
}

#[inline(always)]
fn pack(v: u64) -> Goldilocks {
    Goldilocks::new(v)
}

/// Branchless Goldilocks add on internal limbs (matches `Goldilocks` `Add`).
#[inline(always)]
fn fast_add(x: u64, y: u64) -> u64 {
    let (sum, over) = x.overflowing_add(y);
    let (sum, over) = sum.overflowing_add(over as u64 * NEG_ORDER);
    sum.wrapping_add(over as u64 * NEG_ORDER)
}

/// Copied from qp-poseidon-core-3.1.0/src/poseidon2.rs `apply_mat4`.
#[inline(never)]
fn apply_mat4(x: &mut [Goldilocks; 4]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    // Order matters: overwrite x[0] and x[2] after using x[1] and x[3]
    x[3] = t01233 + x[0].double(); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123 + x[2].double(); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
}

#[inline(never)]
fn old_mat4(mut x: [Goldilocks; 4]) -> [Goldilocks; 4] {
    apply_mat4(&mut x);
    x
}

/// Same MDS, factoring the common row sum `s = x0+x1+x2+x3` and pairwise sums `t01`, `t23`.
#[inline(always)]
fn fast_mat4(x: [Goldilocks; 4]) -> [Goldilocks; 4] {
    let a = limb(x[0]);
    let b = limb(x[1]);
    let c = limb(x[2]);
    let d = limb(x[3]);

    let t01 = fast_add(a, b);
    let t23 = fast_add(c, d);
    let s = fast_add(t01, t23);
    let t01123 = fast_add(s, b);
    let t01233 = fast_add(s, d);
    let d0 = fast_add(a, a);
    let d2 = fast_add(c, c);

    [
        pack(fast_add(t01123, t01)),
        pack(fast_add(t01123, d2)),
        pack(fast_add(t01233, t23)),
        pack(fast_add(t01233, d0)),
    ]
}

fn assert_mat4_pair(state: [Goldilocks; 4]) {
    let expect = old_mat4(state);
    let got = fast_mat4(state);
    assert_eq!(expect, got, "fast_mat4 diverged for state {:?}", state);
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn p1_04_mat4_parity_and_bench() {
    for &v in &[
        0u64,
        1,
        2,
        P - 1,
        P,
        P + 1,
        u64::MAX,
        0xc002e770975b1607,
        0xbca51a8dfe14593a,
    ] {
        assert_mat4_pair([Goldilocks::new(v); 4]);
        assert_mat4_pair([
            Goldilocks::new(v),
            Goldilocks::new(v.wrapping_mul(3)),
            Goldilocks::new(v ^ 0xA5A5_A5A5_A5A5_A5A5),
            Goldilocks::new(v.wrapping_add(P)),
        ]);
    }

    let mut rng = 0x3141592653589793u64;
    for _ in 0..50_000 {
        let state = [
            Goldilocks::new(splitmix64(&mut rng)),
            Goldilocks::new(splitmix64(&mut rng)),
            Goldilocks::new(splitmix64(&mut rng)),
            Goldilocks::new(splitmix64(&mut rng)),
        ];
        assert_mat4_pair(state);
    }

    let bench_state = [
        Goldilocks::new(0x486dcb55419d475a),
        Goldilocks::new(0x6c1cb2a358cc24f1),
        Goldilocks::new(0xe3f30d509a1436bb),
        Goldilocks::new(0xd9a64f068dca7c29),
    ];

    const ITERS: u32 = 200_000;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        black_box(old_mat4(black_box(bench_state)));
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / f64::from(ITERS);

    let t1 = Instant::now();
    for _ in 0..ITERS {
        black_box(fast_mat4(black_box(bench_state)));
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / f64::from(ITERS);

    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.2} NEW_NS {new_ns:.2} RATIO {ratio:.4}");
}
