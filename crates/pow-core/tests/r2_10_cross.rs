//! Re-check the CPU midstate hash path and the CUDA host models together.
//!
//! Group A times the two `custom::permute_mut` calls that `mining_midstate` feeds,
//! against `qpow_math::get_nonce_hash` and `hash_from_nonce`, on `NONCE_HASH_KVS`.
//!
//! Group B is a host model of `field.cuh` / `poseidon.cuh` / `nonce_line.cuh` and of
//! `mining_main` / `mining_main_nonce_line` in `mining_entry.cu`. It does not compile
//! or edit the device sources. Field-add counts are `gf64_add` / `GF64_ADD_NONCE_STEP`
//! calls. M4 counts are `ext_layer64_chunk4` and `ext_layer64_chunk4_wide` calls.
//! Host prestate and `pk[14]` are inputs, not part of the per-nonce totals.

use pow_core::custom;
use pow_core::{hash_from_nonce, mining_midstate, JobContext, NONCE_HASH_KVS, SPONGE_WIDTH};
use primitive_types::U512;
use qp_poseidon_core::{Goldilocks, Poseidon2};
use std::hint::black_box;
use std::time::Instant;

const P64: u64 = 0xFFFF_FFFF_0000_0001;
const P128: u128 = 0xFFFF_FFFF_0000_0001;
const EPS: u64 = 0xFFFF_FFFF;
const T_SPAN: u32 = 1 << 26;
const TRIALS: usize = 5;
const REPS: u64 = 8_000;

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn decode64(s: &str) -> [u8; 64] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn canon(state: &[Goldilocks; SPONGE_WIDTH]) -> [u64; SPONGE_WIDTH] {
    state.map(|g| g.as_canonical_u64())
}

fn absorb(state: &mut [Goldilocks; SPONGE_WIDTH], bytes: &[u8]) {
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        state[i] += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
}

/// Same byte layout as `pow_core`'s midstate-resume test: three library permutes
/// after the custom midstate.
fn resume_full_hash(header: [u8; 32], nonce_be: [u8; 64]) -> U512 {
    let mut state =
        mining_midstate(header, nonce_be[..32].try_into().unwrap()).map(Goldilocks::from_u64);
    let poseidon = Poseidon2::new();
    absorb(&mut state, &nonce_be[32..]);
    poseidon.permute_mut(&mut state);
    state[0] += Goldilocks::ONE;
    state[1] += Goldilocks::ONE;
    poseidon.permute_mut(&mut state);
    let mut hash = [0u8; 64];
    for i in 0..4 {
        hash[i * 8..(i + 1) * 8].copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
    }
    poseidon.permute_mut(&mut state);
    for i in 0..4 {
        hash[32 + i * 8..32 + (i + 1) * 8]
            .copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
    }
    U512::from_big_endian(&hash)
}

struct KvCase {
    header: [u8; 32],
    high: [u8; 32],
    nonce_be: [u8; 64],
    nonce: U512,
    ctx: JobContext,
    before_first: [Goldilocks; SPONGE_WIDTH],
    before_second: [Goldilocks; SPONGE_WIDTH],
}

fn kv_cases() -> Vec<KvCase> {
    NONCE_HASH_KVS
        .iter()
        .map(|kv| {
            let header = decode32(kv.header);
            let nonce_be = decode64(kv.nonce);
            let high: [u8; 32] = nonce_be[..32].try_into().unwrap();
            let mut before_first = [Goldilocks::ZERO; SPONGE_WIDTH];
            absorb(&mut before_first, &header);
            let mut mid = before_first;
            custom::permute_mut(&mut mid);
            absorb(&mut mid, &high);
            KvCase {
                header,
                high,
                nonce_be,
                nonce: U512::from_big_endian(&nonce_be),
                ctx: JobContext::new(header, U512::one()),
                before_first,
                before_second: mid,
            }
        })
        .collect()
}

fn assert_golden_and_custom_permutes(cases: &[KvCase]) {
    assert_eq!(cases.len(), NONCE_HASH_KVS.len());
    assert_eq!(cases.len(), 5, "golden file currently has five vectors");
    let poseidon = Poseidon2::new();
    for (i, (case, kv)) in cases.iter().zip(NONCE_HASH_KVS.iter()).enumerate() {
        let mut lib = case.before_first;
        let mut fast = case.before_first;
        poseidon.permute_mut(&mut lib);
        custom::permute_mut(&mut fast);
        assert_eq!(
            canon(&lib),
            canon(&fast),
            "kv {i}: first custom permute != library"
        );
        absorb(&mut lib, &case.high);
        assert_eq!(
            canon(&lib),
            canon(&case.before_second),
            "kv {i}: second permute input drifted"
        );
        let mut lib2 = case.before_second;
        let mut fast2 = case.before_second;
        poseidon.permute_mut(&mut lib2);
        custom::permute_mut(&mut fast2);
        assert_eq!(
            canon(&lib2),
            canon(&fast2),
            "kv {i}: second custom permute != library"
        );

        let mid = mining_midstate(case.header, case.high);
        assert_eq!(
            mid,
            canon(&fast2),
            "kv {i}: mining_midstate != its two permutes"
        );
        assert_eq!(mid, kv.mid, "kv {i}: mining_midstate != stored mid");

        let direct = qpow_math::get_nonce_hash(case.header, case.nonce_be);
        let via_ctx = hash_from_nonce(&case.ctx, case.nonce);
        let resumed = resume_full_hash(case.header, case.nonce_be);
        let want = decode64(kv.hash);
        assert_eq!(
            direct.to_big_endian(),
            want,
            "kv {i}: get_nonce_hash != stored hash"
        );
        assert_eq!(
            via_ctx.to_big_endian(),
            want,
            "kv {i}: hash_from_nonce != stored hash"
        );
        assert_eq!(resumed, direct, "kv {i}: midstate resume != get_nonce_hash");
        assert_eq!(via_ctx, direct, "kv {i}: hash_from_nonce != get_nonce_hash");
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

struct Times {
    permute_pair_ns: f64,
    midstate_ns: f64,
    get_nonce_hash_ns: f64,
    hash_from_nonce_ns: f64,
}

fn bench_kvs(cases: &[KvCase]) -> Times {
    let n = cases.len() as u64;
    let mut sink = 0u64;
    for case in cases {
        let mut a = case.before_first;
        custom::permute_mut(&mut a);
        sink ^= a[0].as_canonical_u64();
        sink ^= mining_midstate(case.header, case.high)[0];
        sink ^= qpow_math::get_nonce_hash(case.header, case.nonce_be).low_u64();
        sink ^= hash_from_nonce(&case.ctx, case.nonce).low_u64();
    }
    black_box(sink);

    let mut pair = [0f64; TRIALS];
    let mut mid = [0f64; TRIALS];
    let mut get = [0f64; TRIALS];
    let mut hash = [0f64; TRIALS];
    let calls = REPS * n;

    for trial in 0..TRIALS {
        let t0 = Instant::now();
        for _ in 0..REPS {
            for case in cases {
                let mut a = case.before_first;
                custom::permute_mut(black_box(&mut a));
                sink ^= black_box(a[0].as_canonical_u64());
                let mut b = case.before_second;
                custom::permute_mut(black_box(&mut b));
                sink ^= black_box(b[0].as_canonical_u64());
            }
        }
        pair[trial] = t0.elapsed().as_nanos() as f64 / calls as f64;

        let t1 = Instant::now();
        for _ in 0..REPS {
            for case in cases {
                let got = mining_midstate(black_box(case.header), black_box(case.high));
                sink ^= black_box(got[0]);
            }
        }
        mid[trial] = t1.elapsed().as_nanos() as f64 / calls as f64;

        let t2 = Instant::now();
        for _ in 0..REPS {
            for case in cases {
                let got =
                    qpow_math::get_nonce_hash(black_box(case.header), black_box(case.nonce_be));
                sink ^= black_box(got.low_u64());
            }
        }
        get[trial] = t2.elapsed().as_nanos() as f64 / calls as f64;

        let t3 = Instant::now();
        for _ in 0..REPS {
            for case in cases {
                let got = hash_from_nonce(&case.ctx, black_box(case.nonce));
                sink ^= black_box(got.low_u64());
            }
        }
        hash[trial] = t3.elapsed().as_nanos() as f64 / calls as f64;
    }
    black_box(sink);
    Times {
        permute_pair_ns: median(&mut pair),
        midstate_ns: median(&mut mid),
        get_nonce_hash_ns: median(&mut get),
        hash_from_nonce_ns: median(&mut hash),
    }
}

// --- CUDA host model: gf64_add and the 96-bit M4 --------------------------------

#[inline(always)]
fn gf64_add(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    if !carry {
        sum
    } else {
        sum.wrapping_add(EPS)
    }
}

/// `gf64_add_lazy`: the EPS fold's carry into the high half is dropped.
#[inline(always)]
fn gf64_add_lazy(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    if !carry {
        return sum;
    }
    let low = (sum as u32).wrapping_add(EPS as u32);
    let high = (sum >> 32) as u32;
    ((high as u64) << 32) | u64::from(low)
}

#[inline(always)]
fn u_at(pk0: u64, t: u64) -> u64 {
    let d = 35 * t;
    gf64_add(pk0, P64.wrapping_sub(d))
}

#[inline(always)]
fn v_at(pk1: u64, t: u64) -> u64 {
    gf64_add(pk1, 35 * t)
}

#[inline(always)]
fn step_uv(u: u64, v: u64) -> (u64, u64) {
    (gf64_add(u, P64.wrapping_sub(35)), gf64_add(v, 35))
}

#[inline(never)]
fn walk_uv(pk0: u64, pk1: u64, base: u32, len: u32) -> u64 {
    assert!(u64::from(base) + u64::from(len) <= u64::from(u32::MAX));
    let mut u = u_at(pk0, u64::from(base));
    let mut v = v_at(pk1, u64::from(base));
    let mut acc = u ^ v;
    for step in 0..len {
        let t = u64::from(base) + u64::from(step);
        let expect_u = u_at(pk0, t);
        let expect_v = v_at(pk1, t);
        if u != expect_u || v != expect_v {
            panic!("uv drift pk0={pk0:#x} pk1={pk1:#x} t={t} u={u:#x}/{expect_u:#x} v={v:#x}/{expect_v:#x}");
        }
        (u, v) = step_uv(u, v);
        acc ^= u.wrapping_add(v);
    }
    let end = u64::from(base) + u64::from(len);
    if u != u_at(pk0, end) || v != v_at(pk1, end) {
        panic!("uv drift after last step at t={end}");
    }
    acc
}

fn assert_uv_step_matches_rebuild() {
    let full = [
        (0u64, 0u64),
        (P64 - 1, 1),
        (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
    ];
    let mut acc = 0u64;
    for (pk0, pk1) in full {
        acc ^= walk_uv(pk0, pk1, 0, T_SPAN);
    }

    let mut windows = vec![
        (0u64, u64::MAX),
        (u64::MAX, 0),
        (P64, P64),
        (0xFFFF_FFFF, 0xFFFF_FFFF_0000_0000),
        (0x8000_0000_0000_0000, 0x0000_0001_0000_0000),
    ];
    let mut seed = 0xA5A5_1234_F00D_u64;
    for _ in 0..8 {
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let pk0 = seed;
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        windows.push((pk0, seed));
    }
    // 35*t crosses 2^32 at t = 122_713_351 (35 * 122_713_351 = 2^32 - 11).
    let starts = [
        0u32,
        1 << 20,
        (1 << 26) - 4_096,
        122_713_251,
        122_713_351,
        u32::MAX - 4_095,
    ];
    for (pk0, pk1) in windows {
        for start in starts {
            let len = (u32::MAX - start).min(4_096);
            acc ^= walk_uv(pk0, pk1, start, len);
        }
    }

    // One step past u32::MAX stays on the u64 recurrence and leaves the wrapped formula.
    let pk0 = 0x1111_2222_3333_4444u64;
    let pk1 = 0x5555_6666_7777_8888u64;
    let (u, v) = step_uv(
        u_at(pk0, u64::from(u32::MAX)),
        v_at(pk1, u64::from(u32::MAX)),
    );
    assert_eq!(u, u_at(pk0, u64::from(u32::MAX) + 1));
    assert_eq!(v, v_at(pk1, u64::from(u32::MAX) + 1));
    assert_ne!(u, u_at(pk0, 0));
    assert_ne!(v, v_at(pk1, 0));

    // Lazy fold is not what `gf64_nonce_line_uv_step` calls. It already disagrees at t=1.
    let mut lu = u_at(0, 0);
    let mut lv = v_at(0, 0);
    let mut lazy_drift = false;
    for t in 0..64u64 {
        if lu != u_at(0, t) || lv != v_at(0, t) {
            lazy_drift = true;
            break;
        }
        lu = gf64_add_lazy(lu, P64.wrapping_sub(35));
        lv = gf64_add_lazy(lv, 35);
    }
    assert!(lazy_drift, "lazy ±35 matched 35*t for the first 64 steps");
    assert_eq!(u_at(0, 1), step_uv(u_at(0, 0), v_at(0, 0)).0);
    black_box(acc);
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Wide {
    l0: u32,
    l1: u32,
    h: u32,
}

fn wide_from(x: u64) -> Wide {
    Wide {
        l0: x as u32,
        l1: (x >> 32) as u32,
        h: 0,
    }
}

fn wide_add(w: &mut Wide, x: u64) {
    let (l0, c0) = w.l0.overflowing_add(x as u32);
    let (l1, c1) = w.l1.overflowing_add((x >> 32) as u32);
    let (l1, c1b) = l1.overflowing_add(u32::from(c0));
    w.l0 = l0;
    w.l1 = l1;
    w.h = w.h.wrapping_add(u32::from(c1)).wrapping_add(u32::from(c1b));
}

fn wide_add_wide(w: &mut Wide, x: Wide) {
    let (l0, c0) = w.l0.overflowing_add(x.l0);
    let (l1, c1) = w.l1.overflowing_add(x.l1);
    let (l1, c1b) = l1.overflowing_add(u32::from(c0));
    w.l0 = l0;
    w.l1 = l1;
    w.h =
        w.h.wrapping_add(x.h)
            .wrapping_add(u32::from(c1))
            .wrapping_add(u32::from(c1b));
}

fn as_u128(w: Wide) -> u128 {
    u128::from(w.l0) | (u128::from(w.l1) << 32) | (u128::from(w.h) << 64)
}

fn chunk4(x0: u64, x1: u64, x2: u64, x3: u64, m4: &mut u64) -> [Wide; 4] {
    *m4 += 1;
    let mut t01 = wide_from(x0);
    wide_add(&mut t01, x1);
    let mut t23 = wide_from(x2);
    wide_add(&mut t23, x3);
    let mut t0123 = t01;
    wide_add_wide(&mut t0123, t23);
    let mut t01123 = t0123;
    wide_add(&mut t01123, x1);
    let mut t01233 = t0123;
    wide_add(&mut t01233, x3);
    let mut y3 = t01233;
    wide_add(&mut y3, x0);
    wide_add(&mut y3, x0);
    let mut y1 = t01123;
    wide_add(&mut y1, x2);
    wide_add(&mut y1, x2);
    let mut y0 = t01123;
    wide_add_wide(&mut y0, t01);
    let mut y2 = t01233;
    wide_add_wide(&mut y2, t23);
    [y0, y1, y2, y3]
}

fn chunk4_wide(x0: Wide, x1: Wide, x2: Wide, x3: Wide, m4: &mut u64) -> [Wide; 4] {
    *m4 += 1;
    let mut t01 = x0;
    wide_add_wide(&mut t01, x1);
    let mut t23 = x2;
    wide_add_wide(&mut t23, x3);
    let mut t0123 = t01;
    wide_add_wide(&mut t0123, t23);
    let mut t01123 = t0123;
    wide_add_wide(&mut t01123, x1);
    let mut t01233 = t0123;
    wide_add_wide(&mut t01233, x3);
    let mut y3 = t01233;
    wide_add_wide(&mut y3, x0);
    wide_add_wide(&mut y3, x0);
    let mut y1 = t01123;
    wide_add_wide(&mut y1, x2);
    wide_add_wide(&mut y1, x2);
    let mut y0 = t01123;
    wide_add_wide(&mut y0, t01);
    let mut y2 = t01233;
    wide_add_wide(&mut y2, t23);
    [y0, y1, y2, y3]
}

fn col_sum_three_chunk(state: &[u64; 12], m4: &mut u64) -> [Wide; 4] {
    let mut col = [Wide::default(); 4];
    for chunk in 0..3 {
        let o = chunk * 4;
        let y = chunk4(state[o], state[o + 1], state[o + 2], state[o + 3], m4);
        for lane in 0..4 {
            wide_add_wide(&mut col[lane], y[lane]);
        }
    }
    col
}

fn col_sum_one_wide(state: &[u64; 12], m4: &mut u64) -> [Wide; 4] {
    let mut s = [Wide::default(); 4];
    for lane in 0..4 {
        s[lane] = wide_from(state[lane]);
        wide_add(&mut s[lane], state[lane + 4]);
        wide_add(&mut s[lane], state[lane + 8]);
    }
    chunk4_wide(s[0], s[1], s[2], s[3], m4)
}

fn field_mat4(x0: Goldilocks, x1: Goldilocks, x2: Goldilocks, x3: Goldilocks) -> [Goldilocks; 4] {
    let t01 = x0 + x1;
    let t23 = x2 + x3;
    let t0123 = t01 + t23;
    let t01123 = t0123 + x1;
    let t01233 = t0123 + x3;
    [
        t01123 + t01,
        t01123 + x2.double(),
        t01233 + t23,
        t01233 + x0.double(),
    ]
}

fn assert_column_sums_match() {
    let mut states = vec![
        [0u64; 12],
        [1u64; 12],
        [u64::MAX; 12],
        [P64 - 1; 12],
        [P64; 12],
        [0xFFFF_FFFF; 12],
        [0xFFFF_FFFF_0000_0000; 12],
        [0x8000_0000_0000_0000; 12],
    ];
    for lane in 0..12 {
        let mut state = [0u64; 12];
        state[lane] = u64::MAX;
        states.push(state);
    }
    let mut seed = 0xC0FFEE_u64;
    for _ in 0..256 {
        let mut state = [0u64; 12];
        for lane in &mut state {
            seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *lane = seed;
        }
        states.push(state);
    }

    let mut old_m4 = 0u64;
    let mut new_m4 = 0u64;
    for (n, state) in states.iter().enumerate() {
        let old = col_sum_three_chunk(state, &mut old_m4);
        let new = col_sum_one_wide(state, &mut new_m4);
        for lane in 0..4 {
            assert_eq!(old[lane], new[lane], "lane {lane} state {n}");
            assert_eq!(as_u128(old[lane]), as_u128(new[lane]));
            assert!(as_u128(old[lane]) < (1u128 << 96));
        }

        let felts = state.map(Goldilocks::from_u64);
        let mut summed = [Goldilocks::ZERO; 4];
        for chunk in 0..3 {
            let o = chunk * 4;
            let y = field_mat4(felts[o], felts[o + 1], felts[o + 2], felts[o + 3]);
            for lane in 0..4 {
                summed[lane] += y[lane];
            }
        }
        let mut lanes = [Goldilocks::ZERO; 4];
        for lane in 0..4 {
            lanes[lane] = felts[lane] + felts[lane + 4] + felts[lane + 8];
        }
        let folded = field_mat4(lanes[0], lanes[1], lanes[2], lanes[3]);
        for lane in 0..4 {
            let wide_mod = (as_u128(new[lane]) % P128) as u64;
            assert_eq!(
                wide_mod,
                summed[lane].as_canonical_u64(),
                "mod p sum state {n}"
            );
            assert_eq!(
                wide_mod,
                folded[lane].as_canonical_u64(),
                "mod p fold state {n}"
            );
        }
    }
    let n = states.len() as u64;
    assert_eq!(old_m4, 3 * n, "three-chunk column sum calls M4 three times");
    assert_eq!(new_m4, n, "one-wide column sum calls M4 once");
    println!("R2_10_COLSUM states {n} old_m4 {old_m4} new_m4 {new_m4}");
}

// --- Call counts for one thread walking N nonces --------------------------------

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Ops {
    gf64_add: u64,
    m4: u64,
    imul: u64,
}

fn ext_layer(ops: &mut Ops) {
    ops.m4 += 1; // ext_layer64_col_sum → ext_layer64_chunk4_wide
    ops.m4 += 3; // three ext_layer64_reduce_chunk4 → ext_layer64_chunk4
}

fn permute_after_initial(ops: &mut Ops) {
    for _ in 0..4 {
        ext_layer(ops); // RC_INITIAL[1..4], sboxes omitted
    }
    // 22 internal rounds go through int_round_p: no gf64_add, no M4.
    for _ in 0..12 {
        ops.gf64_add += 1; // RC_TERMINAL[0]
    }
    for _ in 0..4 {
        ext_layer(ops); // RC_TERMINAL[1..4]
    }
}

fn permute64(ops: &mut Ops) {
    ext_layer(ops); // RC_INITIAL[0]
    permute_after_initial(ops);
}

fn permute_twice_after_initial(ops: &mut Ops) {
    permute_after_initial(ops);
    ops.gf64_add += 2; // capacity +1,+1
    ext_layer(ops);
    permute_after_initial(ops);
}

fn flat_nonce(ops: &mut Ops) {
    ops.gf64_add += 12; // GF64_ADD_NONCE_STEP on the twelve lanes
    permute_twice_after_initial(ops);
}

fn mix_cblk(ops: &mut Ops) {
    ops.gf64_add += 6; // twice, thrice, four lane adds
}

fn round1_state(ops: &mut Ops) {
    ops.gf64_add += 3; // c2, c0, c1
    mix_cblk(ops);
    mix_cblk(ops);
    mix_cblk(ops);
}

fn permute_from_round1(ops: &mut Ops) {
    ext_layer(ops); // RC_TERMINAL[4], the zero round-constant row
    ext_layer(ops); // nonce_line_external_round RC_INITIAL[2]
    ext_layer(ops); // RC_INITIAL[3]
    ops.gf64_add += 1; // RC_INTERNAL[0] before the first internal S-box
    for _ in 0..12 {
        ops.gf64_add += 1;
    }
    for _ in 0..4 {
        ext_layer(ops);
    }
}

fn nonce_line_hash(ops: &mut Ops) {
    round1_state(ops);
    permute_from_round1(ops);
    ops.gf64_add += 2;
    permute64(ops);
}

/// One thread, `t_per_thread = n`, one k. `uv_at` once, then each nonce hashes and steps.
fn nonce_line_thread(n: u64) -> Ops {
    let mut ops = Ops::default();
    ops.imul += 1;
    ops.gf64_add += 2; // gf64_nonce_line_uv_at
    for _ in 0..n {
        nonce_line_hash(&mut ops);
        ops.gf64_add += 2; // gf64_nonce_line_uv_step, including after the last nonce
    }
    ops
}

fn flat_thread(n: u64) -> Ops {
    let mut ops = Ops::default();
    for _ in 0..n {
        flat_nonce(&mut ops);
    }
    ops
}

/// Old nonce-line u,v: one `35*t` multiply and two field adds per nonce.
fn uv_rebuild_ops(n: u64) -> Ops {
    Ops {
        gf64_add: 2 * n,
        imul: n,
        m4: 0,
    }
}

fn assert_op_counts() {
    let flat1 = flat_thread(1);
    let line1 = nonce_line_thread(1);
    let flat1000 = flat_thread(1000);
    let line1000 = nonce_line_thread(1000);
    assert_eq!(flat1.gf64_add, 38);
    assert_eq!(flat1.m4, 68);
    assert_eq!(line1.gf64_add, 52);
    assert_eq!(line1.m4, 64);
    assert_eq!(line1.imul, 1);
    assert_eq!(flat1000.gf64_add, 38 * 1000);
    assert_eq!(flat1000.m4, 68 * 1000);
    assert_eq!(line1000.gf64_add, 2 + 50 * 1000);
    assert_eq!(line1000.m4, 64 * 1000);
    assert_eq!(line1000.imul, 1);
    assert_eq!(flat1000.gf64_add, flat1.gf64_add * 1000);
    assert_eq!(flat1000.m4, flat1.m4 * 1000);

    let old_uv = uv_rebuild_ops(1000);
    let new_uv_adds = 2 + 2 * 1000;
    assert_eq!(old_uv.imul, 1000);
    assert_eq!(old_uv.gf64_add, 2000);
    assert_eq!(new_uv_adds, 2002);

    println!("R2_10_OPS flat_n1 adds {} m4 {}", flat1.gf64_add, flat1.m4);
    println!(
        "R2_10_OPS line_n1 adds {} m4 {} imul {}",
        line1.gf64_add, line1.m4, line1.imul
    );
    println!(
        "R2_10_OPS flat_n1000 adds {} m4 {}",
        flat1000.gf64_add, flat1000.m4
    );
    println!(
        "R2_10_OPS line_n1000 adds {} m4 {} imul {}",
        line1000.gf64_add, line1000.m4, line1000.imul
    );
    println!(
        "R2_10_OPS uv_rebuild_n1000 adds {} imul {}",
        old_uv.gf64_add, old_uv.imul
    );
    println!("R2_10_OPS uv_step_n1000 adds {new_uv_adds} imul 1");
}

#[test]
fn r2_10_cross() {
    let cases = kv_cases();
    assert_golden_and_custom_permutes(&cases);
    println!("R2_10_GOLDEN pass vectors {}", cases.len());

    assert_uv_step_matches_rebuild();
    println!("R2_10_UV pass full_span {T_SPAN} pk_pairs 3");

    assert_column_sums_match();
    assert_op_counts();

    let times = bench_kvs(&cases);
    println!("R2_10_TIME permute_pair_ns {:.1}", times.permute_pair_ns);
    println!("R2_10_TIME mining_midstate_ns {:.1}", times.midstate_ns);
    println!(
        "R2_10_TIME get_nonce_hash_ns {:.1}",
        times.get_nonce_hash_ns
    );
    println!(
        "R2_10_TIME hash_from_nonce_ns {:.1}",
        times.hash_from_nonce_ns
    );
    println!(
        "R2_10_TIME ratio_get {:.4} ratio_hash {:.4} ratio_mid {:.4}",
        times.get_nonce_hash_ns / times.permute_pair_ns,
        times.hash_from_nonce_ns / times.permute_pair_ns,
        times.get_nonce_hash_ns / times.midstate_ns
    );
}
