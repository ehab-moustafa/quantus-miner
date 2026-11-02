//! Bit-exact host models of `engine-cuda` `field.cuh` + `poseidon.cuh` GF64 ops.
//! Used to oracle nonce-line GPU filter limbs without a device.

use crate::nonce_line::gf64_add_u64;

pub const P64: u64 = 0xffff_ffff_0000_0001;
pub const EPS64: u64 = 0xffff_ffff;
const EPS32: u32 = 0xffff_ffff;

#[inline]
pub fn gf64_add(a: u64, b: u64) -> u64 {
    gf64_add_u64(a, b)
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Wide {
    pub l0: u32,
    pub l1: u32,
    pub h: u32,
}

#[inline]
pub fn wide_from(x: u64) -> Wide {
    Wide {
        l0: x as u32,
        l1: (x >> 32) as u32,
        h: 0,
    }
}

#[inline]
pub fn wide_add(w: &mut Wide, x: u64) {
    let x0 = x as u32;
    let x1 = (x >> 32) as u32;
    let (l0, c0) = w.l0.overflowing_add(x0);
    let (l1, c1) = w.l1.overflowing_add(x1);
    let (l1, c1b) = l1.overflowing_add(u32::from(c0));
    w.l0 = l0;
    w.l1 = l1;
    w.h = w.h.wrapping_add(u32::from(c1)).wrapping_add(u32::from(c1b));
}

#[inline]
pub fn wide_add_wide(w: &mut Wide, x: Wide) {
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

#[inline]
pub fn reduce128_from_u128(value: u128) -> u64 {
    let (r0, r1, r2, r3) = (
        value as u32,
        (value >> 32) as u32,
        (value >> 64) as u32,
        (value >> 96) as u32,
    );
    reduce128(r0, r1, r2, r3)
}

#[inline]
pub fn reduce128(r0: u32, r1: u32, r2: u32, r3: u32) -> u64 {
    let low = ((r1 as u64) << 32) | r0 as u64;
    let (folded, carry) = low.overflowing_add(r2 as u64 * EPS64);
    let (subtrahend, _) = r3.overflowing_add(carry as u32);
    let (hi, _) = ((folded >> 32) as u32).overflowing_add(carry as u32);
    let shifted = ((hi as u64) << 32) | (folded as u32 as u64);
    let (result, _) = shifted.overflowing_sub(subtrahend as u64);
    result
}

#[inline]
pub fn wide_reduce(w: Wide) -> u64 {
    reduce128(w.l0, w.l1, w.h, 0)
}

#[inline]
pub fn mul64wide(a: u64, b: u64) -> (u32, u32, u32, u32) {
    let lo = a.wrapping_mul(b);
    let hi = ((a as u128 * b as u128) >> 64) as u64;
    (lo as u32, (lo >> 32) as u32, hi as u32, (hi >> 32) as u32)
}

#[inline]
pub fn add128_wide(r0: &mut u32, r1: &mut u32, r2: &mut u32, r3: &mut u32, w: Wide) {
    let (l0, c0) = r0.overflowing_add(w.l0);
    let (l1, c1) = r1.overflowing_add(w.l1);
    let (l2, c2) = r2.overflowing_add(w.h);
    let (l1, c1b) = l1.overflowing_add(u32::from(c0));
    *r0 = l0;
    *r1 = l1;
    *r2 = l2;
    *r3 = r3
        .wrapping_add(u32::from(c1))
        .wrapping_add(u32::from(c1b))
        .wrapping_add(u32::from(c2));
}

#[inline]
pub fn mul128_add_wide(a: u64, b: u64, w: Wide) -> (u32, u32, u32, u32) {
    let (mut r0, mut r1, mut r2, mut r3) = mul64wide(a, b);
    add128_wide(&mut r0, &mut r1, &mut r2, &mut r3, w);
    (r0, r1, r2, r3)
}

#[inline]
pub fn mul_diag_add(a: u64, diag: u64, w: Wide) -> u64 {
    let (r0, r1, r2, r3) = mul128_add_wide(a, diag, w);
    reduce128(r0, r1, r2, r3)
}

#[inline]
pub fn gf64_mul(a: u64, b: u64) -> u64 {
    reduce128_from_u128(a as u128 * b as u128)
}

#[inline]
pub fn gf64_sqr(a: u64) -> u64 {
    gf64_mul(a, a)
}

#[inline]
pub fn gf64_sbox(x: u64) -> u64 {
    let x2 = gf64_sqr(x);
    let x3 = gf64_mul(x2, x);
    let x4 = gf64_sqr(x2);
    gf64_mul(x4, x3)
}

#[inline]
pub fn gf64_sbox_pair(u: u64, v: u64) -> (u64, u64) {
    let u2 = gf64_sqr(u);
    let v2 = gf64_sqr(v);
    let u3 = gf64_mul(u2, u);
    let v3 = gf64_mul(v2, v);
    let u4 = gf64_sqr(u2);
    let v4 = gf64_sqr(v2);
    (gf64_mul(u4, u3), gf64_mul(v4, v3))
}

#[inline]
pub fn gf64_canon(a: u64) -> u64 {
    a - if a >= P64 { P64 } else { 0 }
}

#[inline]
pub fn gf64_sbox_after_sqr(x: u64, x2: u64) -> u64 {
    let x4 = gf64_sqr(x2);
    let x3 = gf64_mul(x2, x);
    gf64_mul(x4, x3)
}

#[inline]
pub fn bswap32(v: u32) -> u32 {
    v.swap_bytes()
}

pub const RC_INTERNAL: [u64; 23] = [
    0x97f7798a784ad863,
    0xd1d2bf082f60d4f0,
    0x69a377a79f9ad206,
    0xa9d06906a3858e24,
    0x295275001eede5b5,
    0x5874e441117bd746,
    0x8a084bbba8ed86cc,
    0x3defd7645cde6425,
    0x3998cfe6871cc137,
    0x3e52ef8bca48314a,
    0x964a209f85dc9ecc,
    0x3fcc9ee82cc4577e,
    0x8e79b4a5d0096d6d,
    0x8492362ad2392556,
    0xee72f470262574d6,
    0x1e0e18496da2444a,
    0x0f3a74bf215eaac6,
    0x1b061b76a1c0ded3,
    0x192c42d86803d7a6,
    0xf6d49ff997ae0260,
    0x3ec372e7a0fa3786,
    0x5538cdf4f23445d3,
    0,
];

pub const RC_INITIAL: [[u64; 12]; 5] = [
    [
        0xc002e770975b1607,
        0xbca51a8dfe14593a,
        0x72938dfbe774f7f9,
        0xe4f2fe29e03234ac,
        0xd5e0ba2f541b6449,
        0xec33b868f3cc46c1,
        0x486dcb55419d475a,
        0x6c1cb2a358cc24f1,
        0xe3f30d509a1436bb,
        0xd9a64f068dca7c29,
        0xe59b3f57aabba1ae,
        0x2a3dd4505b478fdc,
    ],
    [
        0xada1f8dc7676ed25,
        0x2711aa8b5509d516,
        0x4ae6acd0c9c92897,
        0x56eb3d6b5256d67a,
        0x1f7a9d55923bf51e,
        0x3600427d397a7f68,
        0xe5076df75b72c3d0,
        0xfcd59aa12c6090ad,
        0xcd895e8c68b57a9e,
        0x41df7ef9d730ae3e,
        0xee3e2b889abe977d,
        0xd29bb7edbeb9c405,
    ],
    [
        0x7d5c08eef608e382,
        0x89ae889caaf0802c,
        0xb35a8e976d2af617,
        0xdb14234eafaf5173,
        0x78f04462d48b1c98,
        0x265293b0e47ce88a,
        0x999a649b69b9d32f,
        0x64b0a186698e01d3,
        0xee0b22d0dfae8bb8,
        0x4fd53e50ca04a7ee,
        0x5762bfe181f25047,
        0xf51593e2beb5e3bd,
    ],
    [
        0x1e5e2b5760e32477,
        0x622462a1f9aaaeed,
        0xaa284b3ecdb222ae,
        0x63c8e72f542bf3fc,
        0x3ba588cacb43b5e0,
        0x23eda6f3c99150dd,
        0xaad3bea4baac9a5a,
        0xe9da8d699b94184a,
        0xcdb13f4cd93e024c,
        0x902cbd0956f655e3,
        0x5b4e40ffc759532f,
        0xde795c20a2357af7,
    ],
    [0x97f7798a784ad863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

pub const RC_TERMINAL: [[u64; 12]; 5] = [
    [
        0x7b72c539e0ea4c6e,
        0x144573dae2ce9976,
        0x802028b68f35fc88,
        0x6d36c5022c4fe7c2,
        0xa205d0ffa9b9def3,
        0xf6e7e38b1ea6ba2f,
        0x34f7909ae5258d64,
        0xb0464d9d77b97fca,
        0x64ddb9d5de7e00a6,
        0x0ed0d75c27975d97,
        0x1cbb36f11127338b,
        0x6673e505cfd0b6ba,
    ],
    [
        0x605f902830872e01,
        0x3fd5eb927e95fe4f,
        0xe81025b5a24c69cd,
        0xf7d0ce75de23f74e,
        0xf39942b6a8585089,
        0x6d808a08f7b71df6,
        0xf8806b6588f49a8b,
        0x57df2d8c2a32107a,
        0x16e7c2074d654a2d,
        0x213de241fcf33835,
        0xb0f2b8905a0976f6,
        0xd8e3cf2bbd355417,
    ],
    [
        0xe498691679d9330f,
        0x763b45d2a3821b28,
        0x0908bf65eb0a1f0d,
        0x7691eb2d194b24f4,
        0x0e43551233ae13b2,
        0x93c393dbfc2fe76f,
        0x98f607485d48cdea,
        0xe3d95f30309819c0,
        0x1ef581a93eaf6acf,
        0x0b24c1b7a030fca4,
        0x624370be5670b327,
        0x5f1e28615a11e486,
    ],
    [
        0xfe04051f909e042b,
        0x7257e5b147fd3803,
        0xe6ae134bb82f2e78,
        0x5711fd5cf4784511,
        0xf83a42660c08c0bc,
        0x2cd8c96d9a3ce855,
        0x7d2ffb1bb0e17271,
        0x85ae1528caea3811,
        0x52a345d5c7adb0b8,
        0x504c4c51f3faee94,
        0xbce34a649cfccaf9,
        0xe0a3389266fb6dc9,
    ],
    [0; 12],
];

pub const MDS_DIAG: [u64; 12] = [
    0xc3b6c08e23ba9300,
    0xd84b5de94a324fb6,
    0x0d0c371c5b35b84f,
    0x7964f570e7188037,
    0x5daf18bbd996604b,
    0x6743bc47b9595257,
    0x5528b9362c59bb70,
    0xac45e25b7127b68b,
    0xa2077d7dfbb606b5,
    0xf3faac6faee378ae,
    0x0c6388b51545e883,
    0xd27dbb6944917b60,
];

#[inline]
fn ext_layer64_chunk4(x0: u64, x1: u64, x2: u64, x3: u64) -> [Wide; 4] {
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

#[inline]
fn ext_layer64_lane_reduce(mut w: Wide, col_sum: Wide, rc: u64) -> u64 {
    wide_add_wide(&mut w, col_sum);
    wide_add(&mut w, rc);
    wide_reduce(w)
}

#[inline]
fn ext_layer64_col_sum_twopass(state: &[u64; 12]) -> [Wide; 4] {
    let mut col = [Wide::default(); 4];
    for chunk in 0..3 {
        let o = chunk * 4;
        let y = ext_layer64_chunk4(state[o], state[o + 1], state[o + 2], state[o + 3]);
        for k in 0..4 {
            wide_add_wide(&mut col[k], y[k]);
        }
    }
    col
}

#[inline]
pub fn ext_layer64(state: &mut [u64; 12], rc12: &[u64; 12]) {
    let col_sum = ext_layer64_col_sum_twopass(state);
    for chunk in 0..3 {
        let o = chunk * 4;
        let y = ext_layer64_chunk4(state[o], state[o + 1], state[o + 2], state[o + 3]);
        state[o] = ext_layer64_lane_reduce(y[0], col_sum[0], rc12[o]);
        state[o + 1] = ext_layer64_lane_reduce(y[1], col_sum[1], rc12[o + 1]);
        state[o + 2] = ext_layer64_lane_reduce(y[2], col_sum[2], rc12[o + 2]);
        state[o + 3] = ext_layer64_lane_reduce(y[3], col_sum[3], rc12[o + 3]);
    }
}

#[inline]
fn int_row_sum(state: &[u64; 12], x: u64) -> Wide {
    let mut s = wide_from(state[1]);
    for i in 2..12 {
        wide_add(&mut s, state[i]);
    }
    wide_add(&mut s, x);
    s
}

#[inline]
pub fn int_round_head(state: &[u64; 12], x: u64, rc0: u64) -> u64 {
    let mut s0 = int_row_sum(state, x);
    wide_add(&mut s0, rc0);
    mul_diag_add(x, MDS_DIAG[0], s0)
}

#[inline]
pub fn int_round_tail(state: &mut [u64; 12], x: u64) {
    let s = int_row_sum(state, x);
    for i in 1..12 {
        state[i] = mul_diag_add(state[i], MDS_DIAG[i], s);
    }
}

#[inline]
pub fn int_round_p(state: &mut [u64; 12], x: u64, rc0: u64) -> u64 {
    let out0 = int_round_head(state, x, rc0);
    int_round_tail(state, x);
    out0
}

#[inline]
fn external_round_sboxes(state: &mut [u64; 12]) {
    for i in 0..12 {
        state[i] = gf64_sbox(state[i]);
    }
}

#[inline]
fn external_round(state: &mut [u64; 12], rc12: &[u64; 12]) {
    external_round_sboxes(state);
    ext_layer64(state, rc12);
}

#[inline]
pub fn terminal_external_rounds(state: &mut [u64; 12]) {
    for r in 1..=4 {
        external_round(state, &RC_TERMINAL[r]);
    }
}

#[inline]
pub fn run_internal_21_plus_finish(state: &mut [u64; 12], mut x: u64, pipelined: bool) {
    for r in 0..21 {
        if pipelined {
            let out0 = int_round_head(state, x, RC_INTERNAL[r + 1]);
            let x2 = gf64_sqr(out0);
            int_round_tail(state, x);
            x = gf64_sbox_after_sqr(out0, x2);
        } else {
            x = gf64_sbox(int_round_p(state, x, RC_INTERNAL[r + 1]));
        }
    }
    state[0] = int_round_p(state, x, 0);
}

#[inline]
pub fn permute64_after_initial(state: &mut [u64; 12], pipelined: bool) {
    for r in 0..4 {
        external_round_sboxes(state);
        ext_layer64(state, &RC_INITIAL[r + 1]);
    }
    let x = gf64_sbox(state[0]);
    run_internal_21_plus_finish(state, x, pipelined);
    for i in 0..12 {
        state[i] = gf64_add(state[i], RC_TERMINAL[0][i]);
    }
    terminal_external_rounds(state);
}

use crate::custom;
use crate::SPONGE_WIDTH;
use qp_poseidon_core::Goldilocks;

#[inline]
pub fn permute64(state: &mut [u64; 12], pipelined: bool) {
    if pipelined {
        ext_layer64(state, &RC_INITIAL[0]);
        permute64_after_initial(state, true);
        return;
    }
    let mut g: [Goldilocks; SPONGE_WIDTH] = state.map(|x| Goldilocks::from_u64(gf64_canon(x)));
    custom::permute_mut(&mut g);
    for i in 0..SPONGE_WIDTH {
        state[i] = g[i].as_canonical_u64();
    }
}

#[inline]
pub fn pack_state_first_be8(state: &[u64; 12]) -> [u32; 8] {
    let mut first = [0u32; 8];
    for i in 0..4 {
        let c = gf64_canon(state[i]);
        first[2 * i] = (c & EPS64) as u32;
        first[2 * i + 1] = (c >> 32) as u32;
    }
    std::array::from_fn(|i| bswap32(first[i]))
}
