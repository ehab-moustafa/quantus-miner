//! Host oracles for CUDA nonce-line high-half filter limbs (`nonce_line.cuh`).

use crate::cuda_gf64::{
    bswap32, ext_layer64, gf64_add, gf64_canon, gf64_sbox, gf64_sbox_pair, pack_state_first_be8,
    permute64, run_internal_21_plus_finish, terminal_external_rounds, wide_add, wide_from,
    wide_reduce, EPS64, RC_INITIAL, RC_INTERNAL, RC_TERMINAL,
};
use crate::mining_midstate;
use crate::nonce_line::{counter_with_t, map_counter, uv_at, K_PARAMS};
use primitive_types::U512;

/// Internal-round schedule in `nonce_line_permute_from_round1` / `permute64_after_initial`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalSchedule {
    Scalar,
    Pipelined,
}

#[inline]
fn mix_cblk(dst: &mut [u64; 12], off: usize, pk: &[u64; K_PARAMS], pk_off: usize, cblk: u64) {
    let twice = gf64_add(cblk, cblk);
    let thrice = gf64_add(twice, cblk);
    dst[off] = gf64_add(pk[pk_off], twice);
    dst[off + 1] = gf64_add(pk[pk_off + 1], cblk);
    dst[off + 2] = gf64_add(pk[pk_off + 2], cblk);
    dst[off + 3] = gf64_add(pk[pk_off + 3], thrice);
}

#[inline]
pub fn round1_state_from_uv(pk: &[u64; K_PARAMS], u: u64, v: u64) -> [u64; 12] {
    let (f, g) = gf64_sbox_pair(u, v);
    let c2 = gf64_add(f, g);
    let c0 = gf64_add(f, c2);
    let c1 = gf64_add(g, c2);
    let mut s = [0u64; 12];
    mix_cblk(&mut s, 0, pk, 2, c0);
    mix_cblk(&mut s, 4, pk, 6, c1);
    mix_cblk(&mut s, 8, pk, 10, c2);
    s
}

#[inline]
fn permute_from_round1(state: &mut [u64; 12], pipelined: bool) {
    for i in 0..12 {
        state[i] = gf64_sbox(state[i]);
    }
    ext_layer64(state, &RC_TERMINAL[4]);
    for rc in [&RC_INITIAL[2], &RC_INITIAL[3]] {
        for i in 0..12 {
            state[i] = gf64_sbox(state[i]);
        }
        ext_layer64(state, rc);
    }
    let x = gf64_sbox(gf64_add(state[0], RC_INTERNAL[0]));
    run_internal_21_plus_finish(state, x, pipelined);
    for i in 0..12 {
        state[i] = gf64_add(state[i], RC_TERMINAL[0][i]);
    }
    terminal_external_rounds(state);
}

/// GF64 `hash_from_midstate` transcription (`poseidon.cuh`): absorb with `bswap32(le[7-i])`, two `permute64`.
pub fn legacy_hash_high_half_first_be_gf64_midstate(
    header: [u8; 32],
    counter: U512,
    t: u32,
    schedule: InternalSchedule,
) -> [u32; 8] {
    use crate::u512_to_le_u32s;
    let pipelined = schedule == InternalSchedule::Pipelined;
    let ctr_be = counter.to_big_endian();
    let mapped_be = map_counter(&counter_with_t(&ctr_be, t));
    let mut st = mining_midstate(header, ctr_be[..32].try_into().unwrap());
    let le = u512_to_le_u32s(U512::from_big_endian(&mapped_be));
    for i in 0..8 {
        st[i] = gf64_add(st[i], u64::from(bswap32(le[7 - i])));
    }
    permute64(&mut st, pipelined);
    st[0] = gf64_add(st[0], 1);
    st[1] = gf64_add(st[1], 1);
    permute64(&mut st, pipelined);
    pack_state_first_be8(&st)
}

/// Flat reference using `custom::permute_mut` (exact PoW field) for oracle cross-checks.
pub fn legacy_hash_high_half_midstate_goldilocks(
    header: [u8; 32],
    counter: U512,
    t: u32,
) -> [u32; 8] {
    use crate::custom;
    use crate::SPONGE_WIDTH;
    use qp_poseidon_core::Goldilocks;
    let ctr_be = counter.to_big_endian();
    let mapped_be = map_counter(&counter_with_t(&ctr_be, t));
    let mid = mining_midstate(header, ctr_be[..32].try_into().unwrap());
    let mut st: [Goldilocks; SPONGE_WIDTH] = mid.map(Goldilocks::from_u64);
    for (felt, chunk) in st.iter_mut().zip(mapped_be[32..].chunks_exact(4)) {
        *felt += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
    custom::permute_mut(&mut st);
    st[0] += Goldilocks::ONE;
    st[1] += Goldilocks::ONE;
    custom::permute_mut(&mut st);
    let mut first = [0u32; 8];
    for i in 0..4 {
        let c = st[i].as_canonical_u64();
        first[2 * i] = (c & EPS64) as u32;
        first[2 * i + 1] = (c >> 32) as u32;
    }
    std::array::from_fn(|i| bswap32(first[i]))
}

/// Legacy high-half pack: CUDA `round1`/`permute_from_round1` is algebraically
/// `absorb(low)` + PoW permute; host uses the Goldilocks reference for that fused
/// prefix, then GF64 `+1/+1` + `permute64` (scalar delegates to `custom::permute_mut`).
pub fn legacy_hash_high_half_first_be_from_uv(
    header: [u8; 32],
    counter: U512,
    t: u32,
    pk: &[u64; K_PARAMS],
    u: u64,
    v: u64,
    schedule: InternalSchedule,
) -> [u32; 8] {
    let _ = (pk, u, v);
    let pipelined = schedule == InternalSchedule::Pipelined;
    let mut st = goldilocks_state_after_third_permute(header, counter, t);
    st[0] = gf64_add(st[0], 1);
    st[1] = gf64_add(st[1], 1);
    permute64(&mut st, pipelined);
    pack_state_first_be8(&st)
}

/// Same round1 / `permute_from_round1` as CUDA, but final two permutes via exact PoW `custom::permute_mut`.
pub fn legacy_hash_high_half_first_be_goldilocks_tail(pk: &[u64; K_PARAMS], t: u32) -> [u32; 8] {
    use crate::custom;
    use crate::SPONGE_WIDTH;
    use qp_poseidon_core::Goldilocks;
    let (u, v) = uv_at(pk[0], pk[1], t);
    let mut st = round1_state_from_uv(pk, u, v);
    permute_from_round1(&mut st, false);
    st[0] = gf64_add(st[0], 1);
    st[1] = gf64_add(st[1], 1);
    let mut g: [Goldilocks; SPONGE_WIDTH] = st.map(|x| Goldilocks::from_u64(gf64_canon(x)));
    custom::permute_mut(&mut g);
    g[0] += Goldilocks::ONE;
    g[1] += Goldilocks::ONE;
    custom::permute_mut(&mut g);
    let mut first = [0u32; 8];
    for i in 0..4 {
        let c = g[i].as_canonical_u64();
        first[2 * i] = (c & EPS64) as u32;
        first[2 * i + 1] = (c >> 32) as u32;
    }
    std::array::from_fn(|i| bswap32(first[i]))
}

pub fn state_after_round1_from_uv(pk: &[u64; K_PARAMS], u: u64, v: u64) -> [u64; 12] {
    round1_state_from_uv(pk, u, v).map(gf64_canon)
}

pub fn goldilocks_state_after_absorb_low(header: [u8; 32], counter: U512, t: u32) -> [u64; 12] {
    use crate::SPONGE_WIDTH;
    use qp_poseidon_core::Goldilocks;
    let ctr_be = counter.to_big_endian();
    let mapped_be = map_counter(&counter_with_t(&ctr_be, t));
    let mid = mining_midstate(header, ctr_be[..32].try_into().unwrap());
    let mut st: [Goldilocks; SPONGE_WIDTH] = mid.map(Goldilocks::from_u64);
    for (felt, chunk) in st.iter_mut().zip(mapped_be[32..].chunks_exact(4)) {
        *felt += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
    st.map(|g| g.as_canonical_u64())
}

pub fn state_after_permute_from_round1(pk: &[u64; K_PARAMS], t: u32) -> [u64; 12] {
    let (u, v) = uv_at(pk[0], pk[1], t);
    let mut st = round1_state_from_uv(pk, u, v);
    permute_from_round1(&mut st, false);
    st.map(gf64_canon)
}

pub fn goldilocks_state_after_third_permute(header: [u8; 32], counter: U512, t: u32) -> [u64; 12] {
    use crate::custom;
    use crate::SPONGE_WIDTH;
    use qp_poseidon_core::Goldilocks;
    let ctr_be = counter.to_big_endian();
    let mapped_be = map_counter(&counter_with_t(&ctr_be, t));
    let mid = mining_midstate(header, ctr_be[..32].try_into().unwrap());
    let mut st: [Goldilocks; SPONGE_WIDTH] = mid.map(Goldilocks::from_u64);
    for (felt, chunk) in st.iter_mut().zip(mapped_be[32..].chunks_exact(4)) {
        *felt += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
    custom::permute_mut(&mut st);
    st.map(|g| g.as_canonical_u64())
}

pub fn legacy_hash_high_half_first_be_after_capacity_plus_one(
    header: [u8; 32],
    counter: U512,
    t: u32,
    schedule: InternalSchedule,
) -> [u32; 8] {
    use crate::SPONGE_WIDTH;
    use qp_poseidon_core::Goldilocks;
    let pipelined = schedule == InternalSchedule::Pipelined;
    let g = goldilocks_state_after_third_permute(header, counter, t);
    let mut st: [Goldilocks; SPONGE_WIDTH] = g.map(Goldilocks::from_u64);
    st[0] += Goldilocks::ONE;
    st[1] += Goldilocks::ONE;
    let mut gf64: [u64; 12] = st.map(|x| x.as_canonical_u64());
    permute64(&mut gf64, pipelined);
    pack_state_first_be8(&gf64)
}

pub fn legacy_hash_high_half_first_be_goldilocks_only(
    header: [u8; 32],
    counter: U512,
    t: u32,
) -> [u32; 8] {
    legacy_hash_high_half_midstate_goldilocks(header, counter, t)
}

pub fn legacy_hash_high_half_first_be_for_segment(
    header: [u8; 32],
    counter: U512,
    t: u32,
    schedule: InternalSchedule,
) -> [u32; 8] {
    let pk = crate::nonce_line::segment_launch_params(header, counter);
    let (u, v) = uv_at(pk[0], pk[1], t);
    legacy_hash_high_half_first_be_from_uv(header, counter, t, &pk, u, v, schedule)
}

pub fn legacy_hash_high_half_first_be(
    header: [u8; 32],
    counter: U512,
    pk: &[u64; K_PARAMS],
    t: u32,
    schedule: InternalSchedule,
) -> [u32; 8] {
    let _ = pk;
    legacy_hash_high_half_first_be_for_segment(header, counter, t, schedule)
}

fn cpp_out0_c_post_row(g: i32) -> [u64; 12] {
    let mut row = [0u64; 12];
    let gg = g & 7;
    if gg <= 2 {
        row.copy_from_slice(&RC_INITIAL[(gg + 1) as usize]);
    } else if gg == 3 {
        row[0] = RC_INTERNAL[0];
    } else if gg <= 6 {
        row.copy_from_slice(&RC_TERMINAL[(gg - 3) as usize]);
    } else {
        row[0] = 1;
        row[1] = 1;
    }
    row
}

fn cpp_out0_internal22(s: &mut [u64; 12]) {
    for r in 0..22 {
        let x = gf64_sbox(s[0]);
        let _ = crate::cuda_gf64::int_round_p(s, x, RC_INTERNAL[r + 1]);
    }
    for i in 0..12 {
        s[i] = gf64_add(s[i], RC_TERMINAL[0][i]);
    }
}

fn cpp_out0_g_step(s: &mut [u64; 12], g: i32, final_g: bool) {
    for i in 0..12 {
        s[i] = gf64_sbox(s[i]);
    }
    if final_g {
        return;
    }
    let post = cpp_out0_c_post_row(g);
    ext_layer64(s, &post);
    if g == 7 {
        ext_layer64(s, &RC_INITIAL[0]);
    }
    if (g & 7) == 3 {
        cpp_out0_internal22(s);
    }
}

fn cpp_out0_second_permute_loops(s: &mut [u64; 12]) {
    for g in 1..=15 {
        cpp_out0_g_step(s, g, g == 15);
    }
}

fn cpp_out0_ext_row0_canon(s: &[u64; 12]) -> u64 {
    let mut acc = wide_from(s[0]);
    wide_add(&mut acc, s[0]);
    for k in 0..3 {
        let rep_m = if k == 0 { 2 } else { 1 };
        for rep in 0..rep_m {
            if !(k == 0 && rep == 0) {
                wide_add(&mut acc, s[4 * k]);
                wide_add(&mut acc, s[4 * k]);
            }
            wide_add(&mut acc, s[4 * k + 1]);
            wide_add(&mut acc, s[4 * k + 1]);
            wide_add(&mut acc, s[4 * k + 1]);
            wide_add(&mut acc, s[4 * k + 2]);
            wide_add(&mut acc, s[4 * k + 3]);
        }
    }
    gf64_canon(wide_reduce(acc))
}

/// CPP_OUT0 filter path (`MINER_CUDA_CPP_OUT0`): two nonzero BE limbs, rest zero.
/// Host oracle: filter limbs match the legacy high-half prefix (`get_nonce_hash` binding).
pub fn cpp_out0_hash_high_half_first_be_from_uv(
    header: [u8; 32],
    counter: U512,
    t: u32,
    pk: &[u64; K_PARAMS],
    u: u64,
    v: u64,
) -> [u32; 8] {
    let leg = legacy_hash_high_half_first_be_from_uv(
        header,
        counter,
        t,
        pk,
        u,
        v,
        InternalSchedule::Scalar,
    );
    let mut first_be = [0u32; 8];
    first_be[0] = leg[0];
    first_be[1] = leg[1];
    first_be
}

/// Matches `target_cmp.cuh` / engine-cuda `cpu_hash_high_half_cmp_be` (0=equal, 1=reject, 2=candidate).
pub fn hash_high_half_cmp_be(first_be: &[u32; 8], tgt_hi: &[u32; 8], cmp_limbs: usize) -> u32 {
    let cmp_limbs = cmp_limbs.min(8);
    for i in 0..cmp_limbs {
        let fi = first_be[i];
        let ti = tgt_hi[7 - i];
        if fi > ti {
            return 2;
        }
        if fi < ti {
            return 1;
        }
    }
    0
}
