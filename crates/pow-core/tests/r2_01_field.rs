//! Re-challenge qp-poseidon-core 3.1.0 Goldilocks add, sub, mul, square, and reduce128.
//!
//! `reduce128` is private, so the timed library reduce is a verbatim copy of
//! `goldilocks.rs` checked against `Goldilocks` multiplication and `x % P`.
//! Public add, sub, mul, and square call the real impls. `fast_mul` is the
//! existing pow-core alternative and is one of the mul candidates.
//!
//! Timing is a dependent chain (one op per step, median of paired runs).
//! Correctness is canonical equality versus `x mod P` on the required edges,
//! extra stress edges, and 1000 random inputs, before any timed sample.

use pow_core::custom::fast_mul;
use qp_poseidon_core::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

const P: u64 = 0xFFFF_FFFF_0000_0001;
const NEG_ORDER: u64 = P.wrapping_neg();

const EDGES: [u64; 8] = [0, 1, P - 1, u64::MAX, P, P + 1, NEG_ORDER, NEG_ORDER - 1];

const SAMPLES: usize = 11;
const ITERS: u64 = 4_000_000;

#[inline(always)]
fn raw(g: Goldilocks) -> u64 {
    unsafe { core::mem::transmute::<Goldilocks, u64>(g) }
}

#[inline(always)]
fn canon(x: u64) -> u64 {
    if x >= P { x - P } else { x }
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// --- library (public ops + verbatim reduce128) --------------------------------

#[inline(always)]
fn lib_add(x: u64, y: u64) -> u64 {
    raw(Goldilocks::new(x) + Goldilocks::new(y))
}

#[inline(always)]
fn lib_sub(x: u64, y: u64) -> u64 {
    raw(Goldilocks::new(x) - Goldilocks::new(y))
}

#[inline(always)]
fn lib_mul(x: u64, y: u64) -> u64 {
    raw(Goldilocks::new(x) * Goldilocks::new(y))
}

#[inline(always)]
fn lib_square(x: u64) -> u64 {
    raw(Goldilocks::new(x).square())
}

#[inline(always)]
fn lib_pipe(a: u64, b: u64) -> u64 {
    lib_square(lib_mul(lib_add(a, b), b))
}

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

/// Verbatim `reduce128` + non-x86_64 `add_no_canonicalize_trashing_input`
/// from qp-poseidon-core 3.1.0 `goldilocks.rs`.
#[inline(always)]
fn lib_reduce(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & NEG_ORDER;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint();
        t0 -= NEG_ORDER;
    }
    let t1 = x_hi_lo * NEG_ORDER;
    let (res_wrapped, carry) = t0.overflowing_add(t1);
    res_wrapped + NEG_ORDER * u64::from(carry)
}

// --- fresh alternatives -------------------------------------------------------

#[inline(always)]
fn add_plain(x: u64, y: u64) -> u64 {
    let (sum, over) = x.overflowing_add(y);
    let (mut sum, over) = sum.overflowing_add(u64::from(over) * NEG_ORDER);
    if over {
        sum += NEG_ORDER;
    }
    sum
}

#[inline(always)]
fn add_branchless(x: u64, y: u64) -> u64 {
    let (sum, over) = x.overflowing_add(y);
    let (sum, over) = sum.overflowing_add(u64::from(over).wrapping_mul(NEG_ORDER));
    sum.wrapping_add(u64::from(over).wrapping_mul(NEG_ORDER))
}

#[inline(always)]
fn add_csel(x: u64, y: u64) -> u64 {
    let (sum, over) = x.overflowing_add(y);
    let adj = if over { NEG_ORDER } else { 0 };
    let (sum, over) = sum.overflowing_add(adj);
    let adj = if over { NEG_ORDER } else { 0 };
    sum.wrapping_add(adj)
}

#[cold]
#[inline(never)]
fn add_cold(sum: u64) -> u64 {
    sum.wrapping_add(NEG_ORDER)
}

#[inline(always)]
fn add_outline(x: u64, y: u64) -> u64 {
    let (sum, over) = x.overflowing_add(y);
    let (sum, over) = sum.overflowing_add(u64::from(over).wrapping_mul(NEG_ORDER));
    if over { add_cold(sum) } else { sum }
}

#[inline(always)]
fn sub_plain(x: u64, y: u64) -> u64 {
    let (diff, under) = x.overflowing_sub(y);
    let (mut diff, under) = diff.overflowing_sub(u64::from(under) * NEG_ORDER);
    if under {
        diff -= NEG_ORDER;
    }
    diff
}

#[inline(always)]
fn sub_branchless(x: u64, y: u64) -> u64 {
    let (diff, under) = x.overflowing_sub(y);
    let (diff, under) = diff.overflowing_sub(u64::from(under).wrapping_mul(NEG_ORDER));
    diff.wrapping_sub(u64::from(under).wrapping_mul(NEG_ORDER))
}

#[inline(always)]
fn sub_csel(x: u64, y: u64) -> u64 {
    let (diff, under) = x.overflowing_sub(y);
    let adj = if under { NEG_ORDER } else { 0 };
    let (diff, under) = diff.overflowing_sub(adj);
    let adj = if under { NEG_ORDER } else { 0 };
    diff.wrapping_sub(adj)
}

#[inline(always)]
fn limb_mul(hi_lo: u64) -> u64 {
    hi_lo.wrapping_mul(NEG_ORDER)
}

#[inline(always)]
fn limb_shift(hi_lo: u64) -> u64 {
    hi_lo.wrapping_shl(32).wrapping_sub(hi_lo)
}

#[inline(always)]
fn t0_of(lo: u64, hi_hi: u64) -> u64 {
    let (mut t0, borrow) = lo.overflowing_sub(hi_hi);
    if borrow {
        t0 = t0.wrapping_sub(NEG_ORDER);
    }
    t0
}

#[inline(always)]
fn fold_mul(t0: u64, t1: u64) -> u64 {
    let (sum, carry) = t0.overflowing_add(t1);
    sum.wrapping_add(NEG_ORDER * u64::from(carry))
}

#[inline(always)]
fn fold_csel(t0: u64, t1: u64) -> u64 {
    let (sum, carry) = t0.overflowing_add(t1);
    let adj = if carry { NEG_ORDER } else { 0 };
    sum.wrapping_add(adj)
}

#[inline(always)]
fn fold_branchless(t0: u64, t1: u64) -> u64 {
    let (sum, carry) = t0.overflowing_add(t1);
    let (sum, carry2) = sum.overflowing_add(u64::from(carry).wrapping_mul(NEG_ORDER));
    sum.wrapping_add(u64::from(carry2).wrapping_mul(NEG_ORDER))
}

#[inline(always)]
fn reduce_style(x: u128, shift: bool, fold: u8) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let t0 = t0_of(lo, hi >> 32);
    let hi_lo = hi & NEG_ORDER;
    let t1 = if shift { limb_shift(hi_lo) } else { limb_mul(hi_lo) };
    match fold {
        0 => fold_mul(t0, t1),
        1 => fold_csel(t0, t1),
        _ => fold_branchless(t0, t1),
    }
}

// Const-generic wrappers so each candidate monomorphizes into a distinct chain.
#[inline(always)]
fn reduce_plain(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let t0 = t0_of(lo, hi >> 32);
    fold_mul(t0, limb_mul(hi & NEG_ORDER))
}

#[inline(always)]
fn reduce_csel(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let t0 = t0_of(lo, hi >> 32);
    fold_csel(t0, limb_mul(hi & NEG_ORDER))
}

#[inline(always)]
fn reduce_shift(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let t0 = t0_of(lo, hi >> 32);
    fold_csel(t0, limb_shift(hi & NEG_ORDER))
}

#[inline(always)]
fn reduce_shift_branchless(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let hi_hi = hi >> 32;
    let hi_lo = hi & NEG_ORDER;
    let (t0, borrow) = lo.overflowing_sub(hi_hi);
    let t0 = t0.wrapping_sub(u64::from(borrow).wrapping_mul(NEG_ORDER));
    let t1 = limb_shift(hi_lo);
    fold_branchless(t0, t1)
}

#[inline(always)]
fn mul_plain(a: u64, b: u64) -> u64 {
    reduce_plain((a as u128) * (b as u128))
}

#[inline(always)]
fn mul_csel(a: u64, b: u64) -> u64 {
    reduce_csel((a as u128) * (b as u128))
}

#[inline(always)]
fn mul_shift(a: u64, b: u64) -> u64 {
    reduce_shift((a as u128) * (b as u128))
}

#[inline(always)]
fn mul_shift_branchless(a: u64, b: u64) -> u64 {
    reduce_shift_branchless((a as u128) * (b as u128))
}

#[inline(always)]
fn square_plain(a: u64) -> u64 {
    mul_plain(a, a)
}

#[inline(always)]
fn square_csel(a: u64) -> u64 {
    mul_csel(a, a)
}

#[inline(always)]
fn square_shift(a: u64) -> u64 {
    mul_shift(a, a)
}

#[inline(always)]
fn square_fast(a: u64) -> u64 {
    fast_mul(a, a)
}

#[inline(always)]
fn square_shift_branchless(a: u64) -> u64 {
    mul_shift_branchless(a, a)
}

#[inline(always)]
fn group_plain(a: u64, b: u64) -> u64 {
    square_plain(mul_plain(add_plain(a, b), b))
}

#[inline(always)]
fn group_csel(a: u64, b: u64) -> u64 {
    square_csel(mul_csel(add_csel(a, b), b))
}

#[inline(always)]
fn group_shift(a: u64, b: u64) -> u64 {
    square_shift(mul_shift(add_csel(a, b), b))
}

#[inline(always)]
fn group_fast(a: u64, b: u64) -> u64 {
    let s = add_plain(a, b);
    let m = fast_mul(s, b);
    fast_mul(m, m)
}

#[inline(always)]
fn group_branchless(a: u64, b: u64) -> u64 {
    square_shift_branchless(mul_shift_branchless(add_branchless(a, b), b))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn add_asm(x: u64, y: u64) -> u64 {
    let sum: u64;
    let tmp: u64;
    let out: u64;
    unsafe {
        core::arch::asm!(
            "adds {sum}, {x}, {y}",
            "csel {tmp}, {neg}, xzr, cs",
            "adds {sum}, {sum}, {tmp}",
            "csel {tmp}, {neg}, xzr, cs",
            "add {out}, {sum}, {tmp}",
            x = in(reg) x,
            y = in(reg) y,
            neg = in(reg) NEG_ORDER,
            sum = out(reg) sum,
            tmp = out(reg) tmp,
            out = out(reg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sub_asm(x: u64, y: u64) -> u64 {
    let diff: u64;
    let tmp: u64;
    let out: u64;
    unsafe {
        core::arch::asm!(
            "subs {diff}, {x}, {y}",
            "csel {tmp}, {neg}, xzr, cc",
            "subs {diff}, {diff}, {tmp}",
            "csel {tmp}, {neg}, xzr, cc",
            "sub {out}, {diff}, {tmp}",
            x = in(reg) x,
            y = in(reg) y,
            neg = in(reg) NEG_ORDER,
            diff = out(reg) diff,
            tmp = out(reg) tmp,
            out = out(reg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mul_asm_shift(a: u64, b: u64) -> u64 {
    let lo: u64;
    let hi: u64;
    let hh: u64;
    let hl: u64;
    let t1: u64;
    let tmp: u64;
    let out: u64;
    unsafe {
        core::arch::asm!(
            "mul {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",
            "lsr {hh}, {hi}, #32",
            "and {hl}, {hi}, {neg}",
            "subs {lo}, {lo}, {hh}",
            "csel {tmp}, {neg}, xzr, cc",
            "sub {lo}, {lo}, {tmp}",
            "lsl {t1}, {hl}, #32",
            "sub {t1}, {t1}, {hl}",
            "adds {lo}, {lo}, {t1}",
            "csel {tmp}, {neg}, xzr, cs",
            "add {out}, {lo}, {tmp}",
            a = in(reg) a,
            b = in(reg) b,
            neg = in(reg) NEG_ORDER,
            lo = out(reg) lo,
            hi = out(reg) hi,
            hh = out(reg) hh,
            hl = out(reg) hl,
            t1 = out(reg) t1,
            tmp = out(reg) tmp,
            out = out(reg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mul_asm_mul(a: u64, b: u64) -> u64 {
    let lo: u64;
    let hi: u64;
    let hh: u64;
    let hl: u64;
    let t1: u64;
    let tmp: u64;
    let out: u64;
    unsafe {
        core::arch::asm!(
            "mul {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",
            "lsr {hh}, {hi}, #32",
            "and {hl}, {hi}, {neg}",
            "subs {lo}, {lo}, {hh}",
            "csel {tmp}, {neg}, xzr, cc",
            "sub {lo}, {lo}, {tmp}",
            "mul {t1}, {hl}, {neg}",
            "adds {lo}, {lo}, {t1}",
            "csel {tmp}, {neg}, xzr, cs",
            "add {out}, {lo}, {tmp}",
            a = in(reg) a,
            b = in(reg) b,
            neg = in(reg) NEG_ORDER,
            lo = out(reg) lo,
            hi = out(reg) hi,
            hh = out(reg) hh,
            hl = out(reg) hl,
            t1 = out(reg) t1,
            tmp = out(reg) tmp,
            out = out(reg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn square_asm(a: u64) -> u64 {
    mul_asm_shift(a, a)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn reduce_asm_shift(x: u128) -> u64 {
    mul_reduce_lo_hi_shift(x as u64, (x >> 64) as u64)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mul_reduce_lo_hi_shift(lo_in: u64, hi_in: u64) -> u64 {
    let hh: u64;
    let hl: u64;
    let t0: u64;
    let t1: u64;
    let tmp: u64;
    let out: u64;
    unsafe {
        core::arch::asm!(
            "lsr {hh}, {hi}, #32",
            "and {hl}, {hi}, {neg}",
            "subs {t0}, {lo}, {hh}",
            "csel {tmp}, {neg}, xzr, cc",
            "sub {t0}, {t0}, {tmp}",
            "lsl {t1}, {hl}, #32",
            "sub {t1}, {t1}, {hl}",
            "adds {t0}, {t0}, {t1}",
            "csel {tmp}, {neg}, xzr, cs",
            "add {out}, {t0}, {tmp}",
            lo = in(reg) lo_in,
            hi = in(reg) hi_in,
            neg = in(reg) NEG_ORDER,
            hh = out(reg) hh,
            hl = out(reg) hl,
            t0 = out(reg) t0,
            t1 = out(reg) t1,
            tmp = out(reg) tmp,
            out = out(reg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn group_asm(a: u64, b: u64) -> u64 {
    let s = add_asm(a, b);
    let m = mul_asm_shift(s, b);
    mul_asm_shift(m, m)
}

fn ref_mod(x: u128) -> u64 {
    (x % (P as u128)) as u64
}

fn ref_add(a: u64, b: u64) -> u64 {
    ref_mod(u128::from(a) + u128::from(b))
}

fn ref_sub(a: u64, b: u64) -> u64 {
    let p = P as u128;
    let aa = u128::from(a) % p;
    let bb = u128::from(b) % p;
    ((aa + p - bb) % p) as u64
}

fn ref_mul(a: u64, b: u64) -> u64 {
    ref_mod(u128::from(a) * u128::from(b))
}

fn assert_canon(label: &str, got: u64, expect: u64) {
    assert_eq!(
        canon(got),
        expect,
        "{label}: got {got:#x} canon {:#x} expect {expect:#x}",
        canon(got)
    );
}

fn prove_all() {
    assert_eq!(NEG_ORDER, 0xFFFF_FFFF);
    assert_eq!(core::mem::size_of::<Goldilocks>(), 8);
    assert_eq!(Goldilocks::new(P).as_canonical_u64(), 0, "P must be 0 in-field");

    // Published vector from the 3.1.0 unit tests.
    let a = 13_835_875_475_997_267_463u64;
    let b = 13_593_300_247_443_167_546u64;
    assert_canon("vec add", lib_add(a, b), 8_982_431_654_025_850_688);
    assert_canon("vec sub", lib_sub(a, b), 242_575_228_554_099_917);
    assert_canon("vec mul", lib_mul(a, b), 16_746_386_726_560_462_281);

    for &x in &EDGES {
        for &y in &EDGES {
            check_binary(x, y);
        }
        check_unary(x);
    }

    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    for _ in 0..1000 {
        let x = xorshift(&mut rng);
        let y = xorshift(&mut rng);
        check_binary(x, y);
        check_unary(x);
        let hi = xorshift(&mut rng);
        let lo = xorshift(&mut rng);
        check_reduce((u128::from(hi) << 64) | u128::from(lo));
    }

    for &x in &EDGES {
        check_reduce(u128::from(x));
        check_reduce(u128::from(x) << 64);
        check_reduce((u128::from(x) << 64) | u128::from(x));
        for &y in &EDGES {
            check_reduce(u128::from(x) * u128::from(y));
        }
    }
    check_reduce(u128::MAX);

    // The copied reducer matches the public multiplier on products.
    let mut rng = 0x0FED_CBA9_8765_4321u64;
    for _ in 0..1000 {
        let x = xorshift(&mut rng);
        let y = xorshift(&mut rng);
        let prod = u128::from(x) * u128::from(y);
        assert_eq!(canon(lib_reduce(prod)), canon(lib_mul(x, y)), "reduce copy vs mul");
    }
}

fn check_binary(x: u64, y: u64) {
    let ea = ref_add(x, y);
    let es = ref_sub(x, y);
    let em = ref_mul(x, y);
    let eg = ref_mul(ref_add(x, y), y);
    let eg = ref_mul(eg, eg);

    assert_canon("lib add", lib_add(x, y), ea);
    assert_canon("add plain", add_plain(x, y), ea);
    assert_canon("add branchless", add_branchless(x, y), ea);
    assert_canon("add csel", add_csel(x, y), ea);
    assert_canon("add outline", add_outline(x, y), ea);

    assert_canon("lib sub", lib_sub(x, y), es);
    assert_canon("sub plain", sub_plain(x, y), es);
    assert_canon("sub branchless", sub_branchless(x, y), es);
    assert_canon("sub csel", sub_csel(x, y), es);

    assert_canon("lib mul", lib_mul(x, y), em);
    assert_canon("mul plain", mul_plain(x, y), em);
    assert_canon("mul csel", mul_csel(x, y), em);
    assert_canon("mul shift", mul_shift(x, y), em);
    assert_canon("mul shift branchless", mul_shift_branchless(x, y), em);
    assert_canon("fast_mul", fast_mul(x, y), em);

    assert_canon("lib pipe", lib_pipe(x, y), eg);
    assert_canon("group plain", group_plain(x, y), eg);
    assert_canon("group csel", group_csel(x, y), eg);
    assert_canon("group shift", group_shift(x, y), eg);
    assert_canon("group fast", group_fast(x, y), eg);
    assert_canon("group branchless", group_branchless(x, y), eg);

    #[cfg(target_arch = "aarch64")]
    {
        assert_canon("add asm", add_asm(x, y), ea);
        assert_canon("sub asm", sub_asm(x, y), es);
        assert_canon("mul asm shift", mul_asm_shift(x, y), em);
        assert_canon("mul asm mul", mul_asm_mul(x, y), em);
        assert_canon("group asm", group_asm(x, y), eg);
    }
}

fn check_unary(x: u64) {
    let es = ref_mul(x, x);
    assert_canon("lib square", lib_square(x), es);
    assert_canon("square plain", square_plain(x), es);
    assert_canon("square csel", square_csel(x), es);
    assert_canon("square shift", square_shift(x), es);
    assert_canon("square fast", square_fast(x), es);
    assert_canon("square shift branchless", square_shift_branchless(x), es);
    #[cfg(target_arch = "aarch64")]
    assert_canon("square asm", square_asm(x), es);
}

fn check_reduce(x: u128) {
    let expect = ref_mod(x);
    assert_canon("lib reduce", lib_reduce(x), expect);
    assert_canon("reduce plain", reduce_plain(x), expect);
    assert_canon("reduce csel", reduce_csel(x), expect);
    assert_canon("reduce shift", reduce_shift(x), expect);
    assert_canon("reduce shift branchless", reduce_shift_branchless(x), expect);
    let _ = reduce_style(x, false, 0);
    #[cfg(target_arch = "aarch64")]
    assert_canon("reduce asm shift", reduce_asm_shift(x), expect);
}

struct Stats {
    old_ns: f64,
    new_ns: f64,
    ratio: f64,
    min_ratio: f64,
    above: usize,
    n: usize,
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[inline(never)]
fn run_bin(mut op: impl FnMut(u64, u64) -> u64, iters: u64) -> (f64, u64) {
    let y0 = black_box(0xC002_E770_975B_1607u64);
    let y1 = black_box(0xBCA5_1A8D_FE14_593Au64);
    let mut x = black_box(0x1234_5678_9ABC_DEF0u64);
    let steps = iters / 2;
    let start = Instant::now();
    for _ in 0..steps {
        // Opaque inputs so an invariant operand cannot be hoisted out of the chain.
        x = op(black_box(x), black_box(y0));
        x = op(black_box(x), black_box(y1));
    }
    let ns = start.elapsed().as_nanos() as f64 / (steps * 2) as f64;
    (ns, black_box(x))
}

#[inline(never)]
fn run_unary(mut op: impl FnMut(u64) -> u64, iters: u64) -> (f64, u64) {
    let mut x = black_box(0xC002_E770_975B_1607u64);
    let start = Instant::now();
    for _ in 0..iters {
        x = op(black_box(x));
    }
    let ns = start.elapsed().as_nanos() as f64 / iters as f64;
    (ns, black_box(x))
}

#[inline(never)]
fn run_red(mut op: impl FnMut(u128) -> u64, iters: u64) -> (f64, u64) {
    let hi = black_box(0x1234_5678_9ABC_DEF0u64);
    let mut lo = black_box(0x0FED_CBA9_8765_4321u64);
    let start = Instant::now();
    for _ in 0..iters {
        let x = black_box((u128::from(hi) << 64) | u128::from(lo));
        lo = op(x);
    }
    let ns = start.elapsed().as_nanos() as f64 / iters as f64;
    (ns, black_box(lo))
}

fn finish(mut olds: Vec<f64>, mut news: Vec<f64>, ratios: &[f64]) -> Stats {
    let n = ratios.len();
    let min_ratio = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let above = ratios.iter().filter(|r| **r > 1.03).count();
    let old_ns = median(&mut olds);
    let new_ns = median(&mut news);
    assert!(old_ns > 0.12, "old chain collapsed: {old_ns}");
    assert!(new_ns > 0.12, "new chain collapsed: {new_ns}");
    Stats {
        old_ns,
        new_ns,
        ratio: old_ns / new_ns,
        min_ratio,
        above,
        n,
    }
}

fn measure_bin(mut old: impl FnMut(u64, u64) -> u64 + Copy, mut new: impl FnMut(u64, u64) -> u64 + Copy) -> Stats {
    let _ = run_bin(&mut old, 200_000);
    let _ = run_bin(&mut new, 200_000);
    let mut olds = Vec::with_capacity(SAMPLES);
    let mut news = Vec::with_capacity(SAMPLES);
    let mut ratios = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let ((o, ao), (n, an)) = if i % 2 == 0 {
            (run_bin(&mut old, ITERS), run_bin(&mut new, ITERS))
        } else {
            let n = run_bin(&mut new, ITERS);
            let o = run_bin(&mut old, ITERS);
            (o, n)
        };
        assert_eq!(canon(ao), canon(an), "binary chain diverged");
        ratios.push(o / n);
        olds.push(o);
        news.push(n);
    }
    finish(olds, news, &ratios)
}

fn measure_unary(mut old: impl FnMut(u64) -> u64 + Copy, mut new: impl FnMut(u64) -> u64 + Copy) -> Stats {
    let _ = run_unary(&mut old, 200_000);
    let _ = run_unary(&mut new, 200_000);
    let mut olds = Vec::with_capacity(SAMPLES);
    let mut news = Vec::with_capacity(SAMPLES);
    let mut ratios = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let ((o, ao), (n, an)) = if i % 2 == 0 {
            (run_unary(&mut old, ITERS), run_unary(&mut new, ITERS))
        } else {
            let n = run_unary(&mut new, ITERS);
            let o = run_unary(&mut old, ITERS);
            (o, n)
        };
        assert_eq!(canon(ao), canon(an), "unary chain diverged");
        ratios.push(o / n);
        olds.push(o);
        news.push(n);
    }
    finish(olds, news, &ratios)
}

fn measure_red(mut old: impl FnMut(u128) -> u64 + Copy, mut new: impl FnMut(u128) -> u64 + Copy) -> Stats {
    let _ = run_red(&mut old, 200_000);
    let _ = run_red(&mut new, 200_000);
    let mut olds = Vec::with_capacity(SAMPLES);
    let mut news = Vec::with_capacity(SAMPLES);
    let mut ratios = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let ((o, ao), (n, an)) = if i % 2 == 0 {
            (run_red(&mut old, ITERS), run_red(&mut new, ITERS))
        } else {
            let n = run_red(&mut new, ITERS);
            let o = run_red(&mut old, ITERS);
            (o, n)
        };
        assert_eq!(canon(ao), canon(an), "reduce chain diverged");
        ratios.push(o / n);
        olds.push(o);
        news.push(n);
    }
    finish(olds, news, &ratios)
}

fn judge(st: &Stats) -> &'static str {
    if st.ratio > 1.03 && st.min_ratio > 1.03 {
        "Adopt"
    } else {
        "Reject"
    }
}

fn consider(item: &str, name: &str, st: &Stats, best: &mut Option<(&'static str, Stats)>) {
    println!(
        "CAND {item} {name} old {:.4} new {:.4} ratio {:.4} min {:.4} above {}/{}",
        st.old_ns, st.new_ns, st.ratio, st.min_ratio, st.above, st.n
    );
    let replace = match best {
        None => true,
        Some((_, prev)) => st.ratio > prev.ratio,
    };
    if replace {
        *best = Some((leak_name(name), st.clone_stats()));
    }
}

fn leak_name(name: &str) -> &'static str {
    match name {
        "plain" => "plain",
        "csel" => "csel",
        "branchless" => "branchless",
        "outline" => "outline",
        "asm" => "asm",
        "shift" => "shift",
        "shift-branchless" => "shift-branchless",
        "fast_mul" => "fast_mul",
        "asm-shift" => "asm-shift",
        "asm-mul" => "asm-mul",
        "fast" => "fast",
        _ => "other",
    }
}

trait CloneStats {
    fn clone_stats(&self) -> Self;
}

impl CloneStats for Stats {
    fn clone_stats(&self) -> Self {
        Stats {
            old_ns: self.old_ns,
            new_ns: self.new_ns,
            ratio: self.ratio,
            min_ratio: self.min_ratio,
            above: self.above,
            n: self.n,
        }
    }
}

fn emit(item: &str, best: Option<(&str, Stats)>) {
    let (name, st) = best.expect("at least one alternative");
    let verdict = judge(&st);
    println!(
        "ROW | {item} | {old:.3} | {new:.3} | {ratio:.3} | {verdict} | alt {name}",
        old = st.old_ns,
        new = st.new_ns,
        ratio = st.ratio,
    );
}

#[test]
fn r2_01_field() {
    prove_all();
    println!("ALL_ASSERTS_PASSED");

    let mut best = None;
    consider("add", "plain", &measure_bin(lib_add, add_plain), &mut best);
    consider("add", "csel", &measure_bin(lib_add, add_csel), &mut best);
    consider("add", "branchless", &measure_bin(lib_add, add_branchless), &mut best);
    consider("add", "outline", &measure_bin(lib_add, add_outline), &mut best);
    #[cfg(target_arch = "aarch64")]
    consider("add", "asm", &measure_bin(lib_add, add_asm), &mut best);
    let add_row = best;

    let mut best = None;
    consider("sub", "plain", &measure_bin(lib_sub, sub_plain), &mut best);
    consider("sub", "csel", &measure_bin(lib_sub, sub_csel), &mut best);
    consider("sub", "branchless", &measure_bin(lib_sub, sub_branchless), &mut best);
    #[cfg(target_arch = "aarch64")]
    consider("sub", "asm", &measure_bin(lib_sub, sub_asm), &mut best);
    let sub_row = best;

    let mut best = None;
    consider("mul", "plain", &measure_bin(lib_mul, mul_plain), &mut best);
    consider("mul", "csel", &measure_bin(lib_mul, mul_csel), &mut best);
    consider("mul", "shift", &measure_bin(lib_mul, mul_shift), &mut best);
    consider(
        "mul",
        "shift-branchless",
        &measure_bin(lib_mul, mul_shift_branchless),
        &mut best,
    );
    consider("mul", "fast_mul", &measure_bin(lib_mul, fast_mul), &mut best);
    #[cfg(target_arch = "aarch64")]
    {
        consider("mul", "asm-shift", &measure_bin(lib_mul, mul_asm_shift), &mut best);
        consider("mul", "asm-mul", &measure_bin(lib_mul, mul_asm_mul), &mut best);
    }
    let mul_row = best;

    let mut best = None;
    consider("square", "plain", &measure_unary(lib_square, square_plain), &mut best);
    consider("square", "csel", &measure_unary(lib_square, square_csel), &mut best);
    consider("square", "shift", &measure_unary(lib_square, square_shift), &mut best);
    consider("square", "fast_mul", &measure_unary(lib_square, square_fast), &mut best);
    consider(
        "square",
        "shift-branchless",
        &measure_unary(lib_square, square_shift_branchless),
        &mut best,
    );
    #[cfg(target_arch = "aarch64")]
    consider("square", "asm-shift", &measure_unary(lib_square, square_asm), &mut best);
    let square_row = best;

    let mut best = None;
    consider("reduce128", "plain", &measure_red(lib_reduce, reduce_plain), &mut best);
    consider("reduce128", "csel", &measure_red(lib_reduce, reduce_csel), &mut best);
    consider("reduce128", "shift", &measure_red(lib_reduce, reduce_shift), &mut best);
    consider(
        "reduce128",
        "shift-branchless",
        &measure_red(lib_reduce, reduce_shift_branchless),
        &mut best,
    );
    #[cfg(target_arch = "aarch64")]
    consider(
        "reduce128",
        "asm-shift",
        &measure_red(lib_reduce, reduce_asm_shift),
        &mut best,
    );
    let reduce_row = best;

    let mut best = None;
    consider("add+mul+square", "plain", &measure_bin(lib_pipe, group_plain), &mut best);
    consider("add+mul+square", "csel", &measure_bin(lib_pipe, group_csel), &mut best);
    consider("add+mul+square", "shift", &measure_bin(lib_pipe, group_shift), &mut best);
    consider("add+mul+square", "fast", &measure_bin(lib_pipe, group_fast), &mut best);
    consider(
        "add+mul+square",
        "branchless",
        &measure_bin(lib_pipe, group_branchless),
        &mut best,
    );
    #[cfg(target_arch = "aarch64")]
    consider("add+mul+square", "asm", &measure_bin(lib_pipe, group_asm), &mut best);
    let group_row = best;

    println!("TABLE");
    emit("add", add_row);
    emit("sub", sub_row);
    emit("mul", mul_row);
    emit("square", square_row);
    emit("reduce128", reduce_row);
    emit("add+mul+square", group_row);

    let _ = std::io::Write::flush(&mut std::io::stdout());
}
