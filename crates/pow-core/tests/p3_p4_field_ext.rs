//! Host models of the CUDA field step and the external-layer column sum.
//! These match `gf64_add` / `Wide` in `engine-cuda/src/kernels/{field,poseidon}.cuh`.

const P64: u64 = 0xFFFF_FFFF_0000_0001;
const EPS: u64 = 0xFFFF_FFFF;

fn gf64_add(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    if !carry {
        sum
    } else {
        sum.wrapping_add(EPS)
    }
}

fn u_rebuild(pk0: u64, t: u32) -> u64 {
    let d = 35u64.wrapping_mul(t as u64);
    gf64_add(pk0, P64.wrapping_sub(d))
}

fn v_rebuild(pk1: u64, t: u32) -> u64 {
    let d = 35u64.wrapping_mul(t as u64);
    gf64_add(pk1, d)
}

fn step_uv(u: u64, v: u64) -> (u64, u64) {
    (gf64_add(u, P64.wrapping_sub(35)), gf64_add(v, 35))
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
    let x0 = x as u32;
    let x1 = (x >> 32) as u32;
    let (l0, c0) = w.l0.overflowing_add(x0);
    let (l1, c1) = w.l1.overflowing_add(x1);
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
    w.l0 as u128 | ((w.l1 as u128) << 32) | ((w.h as u128) << 64)
}

fn chunk4(x0: u64, x1: u64, x2: u64, x3: u64) -> [Wide; 4] {
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

fn chunk4_wide(x0: Wide, x1: Wide, x2: Wide, x3: Wide) -> [Wide; 4] {
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

fn col_sum_two_pass(state: &[u64; 12]) -> [Wide; 4] {
    let mut col = [Wide::default(); 4];
    for chunk in 0..3 {
        let o = chunk * 4;
        let y = chunk4(state[o], state[o + 1], state[o + 2], state[o + 3]);
        for k in 0..4 {
            wide_add_wide(&mut col[k], y[k]);
        }
    }
    col
}

fn col_sum_one_mat4(state: &[u64; 12]) -> [Wide; 4] {
    let mut s = [Wide::default(); 4];
    for k in 0..4 {
        s[k] = wide_from(state[k]);
        wide_add(&mut s[k], state[k + 4]);
        wide_add(&mut s[k], state[k + 8]);
    }
    chunk4_wide(s[0], s[1], s[2], s[3])
}

fn mix_loop(pk: &[u64], c0: u64, c1: u64, c2: u64) -> [u64; 12] {
    let m0 = [2u32, 1, 1, 3];
    let mut s = [0u64; 12];
    for blk in 0..3 {
        let cblk = if blk == 0 {
            c0
        } else if blk == 1 {
            c1
        } else {
            c2
        };
        for j in 0..4 {
            let mut v = cblk;
            for _ in 1..m0[j] {
                v = gf64_add(v, cblk);
            }
            s[4 * blk + j] = gf64_add(pk[4 * blk + j], v);
        }
    }
    s
}

fn mix_unrolled(pk: &[u64], c0: u64, c1: u64, c2: u64) -> [u64; 12] {
    let mut s = [0u64; 12];
    for (blk, cblk) in [c0, c1, c2].into_iter().enumerate() {
        let twice = gf64_add(cblk, cblk);
        let thrice = gf64_add(twice, cblk);
        let o = 4 * blk;
        s[o] = gf64_add(pk[o], twice);
        s[o + 1] = gf64_add(pk[o + 1], cblk);
        s[o + 2] = gf64_add(pk[o + 2], cblk);
        s[o + 3] = gf64_add(pk[o + 3], thrice);
    }
    s
}

#[test]
fn nonce_line_uv_step_matches_rebuild_across_windows() {
    let mut seed = 0xA5A5_1234_F00D_u64;
    let mut next = || {
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        seed
    };
    let windows = [
        (0u32, 4_096u32),
        (1 << 20, 2_048),
        ((1 << 26) - 3_000, 4_000),
        (u32::MAX - 500, 400),
    ];
    for (base, len) in windows {
        let pk0 = next();
        let pk1 = next();
        let mut u = u_rebuild(pk0, base);
        let mut v = v_rebuild(pk1, base);
        for step in 0..len {
            let t = base.wrapping_add(step);
            assert_eq!(u, u_rebuild(pk0, t), "u at {t}");
            assert_eq!(v, v_rebuild(pk1, t), "v at {t}");
            (u, v) = step_uv(u, v);
        }
    }
}

#[test]
fn ext_layer_one_mat4_column_sum_matches_two_pass() {
    let mut seed = 0xC0FFEE_u64;
    for n in 0..256 {
        let mut state = [0u64; 12];
        for lane in &mut state {
            seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *lane = seed;
        }
        if n == 0 {
            state = [u64::MAX; 12];
        }
        let a = col_sum_two_pass(&state);
        let b = col_sum_one_mat4(&state);
        for k in 0..4 {
            assert_eq!(a[k], b[k], "lane {k} state {n}");
            assert!(as_u128(a[k]) >> 96 == 0);
        }
    }
}

#[test]
fn nonce_line_unrolled_mix_matches_loop() {
    let mut seed = 7u64;
    for _ in 0..128 {
        let pk: Vec<u64> = (0..12)
            .map(|_| {
                seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                seed
            })
            .collect();
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let c0 = seed;
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let c1 = seed;
        seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let c2 = seed;
        assert_eq!(mix_loop(&pk, c0, c1, c2), mix_unrolled(&pk, c0, c1, c2));
    }
}
