//! Round-2 re-challenge of `nonce_line`: each public entry, then the whole
//! counter → base words + t → launch params → absolute-t nonce path.
//!
//! Faster formulations live only in this file. Each one `assert_eq`s the
//! library function. `k` is the low 64 bits of `C >> 26` (bits `[26, 90)` of
//! the low 256-bit half); `counter_with_t` / `counter_add_t` zero every bit
//! above that window. The alternatives keep that truncation.

use pow_core::custom::external_linear_layer;
use pow_core::nonce_line::{
    base_words, counter_add_t, counter_with_t, k_of, launch_params, map_counter,
    nonce_from_absolute_t, pk_to_u32s, segment_launch_params, t_of, t_room_in_k, K_PARAMS, T_BITS,
    T_MASK, T_SPAN, V,
};
use pow_core::{mining_midstate, NONCE_HASH_KVS, SPONGE_WIDTH};
use primitive_types::U512;
use qp_poseidon_core::poseidon2::INITIAL_EXTERNAL_CONSTANTS;
use qp_poseidon_core::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

const WORD_MASK: u32 = (1 << 29) - 1;
const TIMED: usize = 4_000;
const ROUND_TRIPS: usize = 100;

#[inline(always)]
fn be_u64(bytes: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&bytes[..8]);
    u64::from_be_bytes(a)
}

/// Least-significant limb first. Limb `i` holds bits `[64i, 64i+64)` of the low half.
#[inline(always)]
fn low256_limbs(counter_be: &[u8; 64]) -> [u64; 4] {
    let low = &counter_be[32..];
    [
        be_u64(&low[24..32]),
        be_u64(&low[16..24]),
        be_u64(&low[8..16]),
        be_u64(&low[0..8]),
    ]
}

#[inline(always)]
fn k_from_low_limbs(l0: u64, l1: u64) -> u64 {
    (l0 >> T_BITS) | ((l1 & T_MASK) << (64 - T_BITS))
}

#[inline(always)]
fn t_of_fast(counter_be: &[u8; 64]) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&counter_be[60..64]);
    u32::from_be_bytes(a) & (T_MASK as u32)
}

#[inline(always)]
fn k_of_fast(counter_be: &[u8; 64]) -> u64 {
    let mut a = [0u8; 16];
    a.copy_from_slice(&counter_be[48..64]);
    (u128::from_be_bytes(a) >> T_BITS) as u64
}

/// `B(k)` as `0x80000000 | bits[26+29i, 26+29i+29)`.
/// Windows: 26, 55|64, 84, 113|128, 142, 171|192, 200, 229 (bits ≥ 256 are 0).
#[inline(always)]
fn base_words_from_limbs(l0: u64, l1: u64, l2: u64, l3: u64) -> [u32; 8] {
    let pack = |v: u64| 0x8000_0000 | ((v as u32) & WORD_MASK);
    [
        pack(l0 >> 26),
        pack((l0 >> 55) | (l1 << 9)),
        pack(l1 >> 20),
        pack((l1 >> 49) | (l2 << 15)),
        pack(l2 >> 14),
        pack((l2 >> 43) | (l3 << 21)),
        pack(l3 >> 8),
        pack(l3 >> 37),
    ]
}

#[inline(always)]
fn base_words_fast(counter_be: &[u8; 64]) -> [u32; 8] {
    let [l0, l1, l2, l3] = low256_limbs(counter_be);
    base_words_from_limbs(l0, l1, l2, l3)
}

#[inline(always)]
fn word_at(base: u32, t: u32, v: i32) -> u32 {
    base.wrapping_add((v as u32).wrapping_mul(t))
}

#[inline(always)]
fn map_counter_fast(counter_be: &[u8; 64]) -> [u8; 64] {
    let b = base_words_fast(counter_be);
    let t = t_of_fast(counter_be);
    let mut out = *counter_be;
    for i in 0..8 {
        let start = 32 + 4 * i;
        out[start..start + 4].copy_from_slice(&word_at(b[i], t, V[i]).to_le_bytes());
    }
    out
}

#[inline(always)]
fn launch_params_from_words(mid: &[u64; SPONGE_WIDTH], b: [u32; 8]) -> [u64; K_PARAMS] {
    let mut s: [Goldilocks; SPONGE_WIDTH] = (*mid).map(Goldilocks::from_u64);
    for i in 0..8 {
        s[i] += Goldilocks::from_u64(b[i] as u64);
    }
    external_linear_layer(&mut s);
    for (felt, rc) in s.iter_mut().zip(INITIAL_EXTERNAL_CONSTANTS[0]) {
        *felt += Goldilocks::from_u64(rc);
    }
    let mut pk = [0u64; K_PARAMS];
    pk[0] = s[0].as_canonical_u64();
    pk[1] = s[4].as_canonical_u64();

    let mut k_st: [Goldilocks; SPONGE_WIDTH] = std::array::from_fn(|i| {
        if i == 0 || i == 4 {
            Goldilocks::ZERO
        } else {
            s[i].exp7()
        }
    });
    external_linear_layer(&mut k_st);
    for i in 0..SPONGE_WIDTH {
        k_st[i] += Goldilocks::from_u64(INITIAL_EXTERNAL_CONSTANTS[1][i]);
        pk[2 + i] = k_st[i].as_canonical_u64();
    }
    pk
}

#[inline(always)]
fn launch_params_fast(mid: &[u64; SPONGE_WIDTH], counter_be: &[u8; 64]) -> [u64; K_PARAMS] {
    launch_params_from_words(mid, base_words_fast(counter_be))
}

#[inline(always)]
fn segment_launch_params_fast(header: [u8; 32], segment_start: U512) -> [u64; K_PARAMS] {
    let counter_be = segment_start.to_big_endian();
    let mid = mining_midstate(header, counter_be[..32].try_into().unwrap());
    launch_params_fast(&mid, &counter_be)
}

#[inline(always)]
fn counter_with_t_fast(counter_be: &[u8; 64], t: u32) -> [u8; 64] {
    let mut out = *counter_be;
    let c = ((k_of_fast(counter_be) as u128) << T_BITS) + (t as u128);
    out[32..48].fill(0);
    out[48..64].copy_from_slice(&c.to_be_bytes());
    out
}

#[inline(always)]
fn counter_add_t_fast(counter: U512, delta: u32) -> U512 {
    let l0 = counter.0[0];
    let l1 = counter.0[1];
    let t = (l0 & T_MASK) + delta as u64;
    let c = ((k_from_low_limbs(l0, l1) as u128) << T_BITS) + (t as u128);
    let mut limbs = counter.0;
    limbs[0] = c as u64;
    limbs[1] = (c >> 64) as u64;
    limbs[2] = 0;
    limbs[3] = 0;
    U512(limbs)
}

/// Little-endian absorb words sit in big-endian nonce bytes, so each u32 is
/// byte-swapped and the word order runs from the top of the low half downward.
#[inline(always)]
fn nonce_from_absolute_t_fast(segment_start: U512, t_abs: u32) -> U512 {
    let l0 = segment_start.0[0];
    let l1 = segment_start.0[1];
    let c = ((k_from_low_limbs(l0, l1) as u128) << T_BITS) + (t_abs as u128);
    let n0 = c as u64;
    let n1 = (c >> 64) as u64;
    let b = base_words_from_limbs(n0, n1, 0, 0);
    let t = (n0 & T_MASK) as u32;
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = word_at(b[i], t, V[i]);
    }
    let swap = |x: u32| x.swap_bytes() as u64;
    let mut limbs = segment_start.0;
    limbs[0] = swap(w[7]) | (swap(w[6]) << 32);
    limbs[1] = swap(w[5]) | (swap(w[4]) << 32);
    limbs[2] = swap(w[3]) | (swap(w[2]) << 32);
    limbs[3] = swap(w[1]) | (swap(w[0]) << 32);
    U512(limbs)
}

#[inline(always)]
fn t_room_in_k_fast(counter: U512) -> u32 {
    let t = (counter.0[0] & T_MASK) as u32;
    (T_SPAN as u32) - t
}

struct Sample {
    be: [u8; 64],
    n: U512,
    mid: [u64; SPONGE_WIDTH],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_counter(rng: &mut u64) -> [u8; 64] {
    let mut be = [0u8; 64];
    for chunk in be.chunks_exact_mut(8) {
        chunk.copy_from_slice(&splitmix64(rng).to_be_bytes());
    }
    be
}

fn set_low_bit(bit: usize) -> [u8; 64] {
    let mut be = [0u8; 64];
    be[63 - bit / 8] = 1u8 << (bit % 8);
    be
}

fn edge_counters() -> Vec<[u8; 64]> {
    let mut out = Vec::new();
    out.push([0u8; 64]);
    out.push([0xffu8; 64]);

    let mut t5 = [0u8; 64];
    t5[63] = 5;
    out.push(t5);

    let mut t_max = [0u8; 64];
    let low = (T_MASK as u32).to_be_bytes();
    t_max[60..64].copy_from_slice(&low);
    out.push(t_max);

    out.push(set_low_bit(26));
    out.push(set_low_bit(89));
    out.push(set_low_bit(90));
    out.push(set_low_bit(127));
    out.push(set_low_bit(255));

    let mut high_only = [0u8; 64];
    high_only[0] = 0xab;
    out.push(high_only);

    let mut mixed = [0x5au8; 64];
    mixed[63] = 5;
    out.push(mixed);

    let mut k_full = [0u8; 64];
    let c = (((1u128 << 64) - 1) << T_BITS) | (T_MASK as u128);
    k_full[48..64].copy_from_slice(&c.to_be_bytes());
    out.push(k_full);

    let mut msb = [0u8; 64];
    msb[32] = 0x80;
    out.push(msb);

    for i in 0..8 {
        out.push(set_low_bit(T_BITS as usize + 29 * i));
    }
    out
}

fn sample_from(header: [u8; 32], be: [u8; 64]) -> Sample {
    let n = U512::from_big_endian(&be);
    let mid = mining_midstate(header, be[..32].try_into().unwrap());
    Sample { be, n, mid }
}

fn assert_counter_matches(be: &[u8; 64], n: U512) {
    let limbs = low256_limbs(be);
    assert_eq!(t_of_fast(be), t_of(be));
    assert_eq!(k_of_fast(be), k_of(be));
    assert_eq!(k_from_low_limbs(limbs[0], limbs[1]), k_of(be));
    assert_eq!(limbs[0], n.0[0]);
    assert_eq!(limbs[1], n.0[1]);
    assert_eq!((n.0[0] & T_MASK) as u32, t_of(be));

    let b = base_words_fast(be);
    assert_eq!(b, base_words(be));
    assert_eq!(map_counter_fast(be), map_counter(be));

    for t_abs in [0u32, 1, t_of(be), T_MASK as u32, 1 << 26, u32::MAX] {
        assert_eq!(counter_with_t_fast(be, t_abs), counter_with_t(be, t_abs));
        assert_eq!(
            nonce_from_absolute_t_fast(n, t_abs),
            nonce_from_absolute_t(n, t_abs)
        );
    }
    for delta in [0u32, 1, 3, T_MASK as u32, 1 << 26, u32::MAX] {
        assert_eq!(counter_add_t_fast(n, delta), counter_add_t(n, delta));
    }
    assert_eq!(t_room_in_k_fast(n), t_room_in_k(n));
}

fn assert_launch_matches(header: [u8; 32], sample: &Sample) {
    assert_eq!(
        launch_params_fast(&sample.mid, &sample.be),
        launch_params(&sample.mid, &sample.be)
    );
    assert_eq!(
        segment_launch_params_fast(header, sample.n),
        segment_launch_params(header, sample.n)
    );
}

fn assert_round_trip(header: [u8; 32], sample: &Sample) {
    let be = &sample.be;
    let t = t_of(be);
    let k = k_of(be);
    let canon = counter_with_t(be, t);
    assert_eq!(t_of(&canon), t);
    assert_eq!(k_of(&canon), k);
    assert_eq!(counter_with_t(&canon, t), canon);

    let nonce = nonce_from_absolute_t(sample.n, t);
    let mapped = map_counter(&canon);
    assert_eq!(nonce, U512::from_big_endian(&mapped));

    let nbe = nonce.to_big_endian();
    assert_eq!(&nbe[..32], &be[..32]);
    let b_canon = base_words(&canon);
    for i in 0..8 {
        let start = 32 + 4 * i;
        let w = u32::from_le_bytes(nbe[start..start + 4].try_into().unwrap());
        assert_eq!(w, word_at(b_canon[i], t, V[i]));
    }
    let w0 = u32::from_le_bytes(nbe[32..36].try_into().unwrap());
    assert_eq!(w0.wrapping_sub(b_canon[0]) / 4, t);

    let canon_n = U512::from_big_endian(&canon);
    assert_eq!(
        nonce_from_absolute_t(canon_n, t_of(&canon)),
        U512::from_big_endian(&map_counter(&canon))
    );

    let room = t_room_in_k(sample.n);
    assert_eq!(room as u64 + t as u64, T_SPAN);
    let stayed = counter_add_t(sample.n, room - 1);
    assert_eq!(t_of(&stayed.to_big_endian()), T_MASK as u32);
    assert_eq!(k_of(&stayed.to_big_endian()), k);
    let crossed = counter_add_t(sample.n, room);
    assert_eq!(t_of(&crossed.to_big_endian()), 0);
    assert_eq!(k_of(&crossed.to_big_endian()), k.wrapping_add(1));

    let pk = launch_params(&sample.mid, be);
    assert_eq!(pk, launch_params(&sample.mid, be));
    assert_eq!(
        segment_launch_params(header, sample.n),
        launch_params(&mining_midstate(header, be[..32].try_into().unwrap()), be)
    );
    let limbs = pk_to_u32s(&pk);
    for i in 0..K_PARAMS {
        let felt = limbs[2 * i] as u64 | ((limbs[2 * i + 1] as u64) << 32);
        assert_eq!(felt, pk[i]);
    }
}

#[inline(always)]
fn path_old(
    sample: &Sample,
) -> ([u32; 8], u32, [u64; K_PARAMS], U512) {
    let b = base_words(&sample.be);
    let t = t_of(&sample.be);
    let pk = launch_params(&sample.mid, &sample.be);
    let nonce = nonce_from_absolute_t(sample.n, t);
    (b, t, pk, nonce)
}

#[inline(always)]
fn path_fast(
    sample: &Sample,
) -> ([u32; 8], u32, [u64; K_PARAMS], U512) {
    let b = base_words_fast(&sample.be);
    let t = t_of_fast(&sample.be);
    let pk = launch_params_from_words(&sample.mid, b);
    let nonce = nonce_from_absolute_t_fast(sample.n, t);
    (b, t, pk, nonce)
}

fn fold_words(w: [u32; 8]) -> u64 {
    let mut a = 0u64;
    for word in w {
        a ^= word as u64;
    }
    a
}

fn fold_bytes(b: &[u8; 64]) -> u64 {
    let mut a = 0u64;
    for chunk in b.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        a ^= u64::from_le_bytes(buf);
    }
    a
}

fn fold_pk(pk: &[u64; K_PARAMS]) -> u64 {
    let mut a = 0u64;
    for felt in pk {
        a ^= felt;
    }
    a
}

fn fold_u512(n: U512) -> u64 {
    let mut a = 0u64;
    for limb in n.0 {
        a ^= limb;
    }
    a
}

fn time_per_op(ops: usize, mut once: impl FnMut()) -> f64 {
    once();
    let mut trials = [0.0f64; 3];
    for trial in &mut trials {
        let mut reps = 1u64;
        loop {
            let start = Instant::now();
            for _ in 0..reps {
                once();
            }
            let elapsed = start.elapsed();
            if elapsed.as_millis() >= 50 || reps >= 8_192 {
                *trial = elapsed.as_nanos() as f64 / (reps as f64 * ops as f64);
                break;
            }
            reps *= 2;
        }
    }
    trials.sort_by(|a, b| a.partial_cmp(b).unwrap());
    trials[1]
}

struct Row {
    item: &'static str,
    old_ns: f64,
    new_ns: f64,
}

fn print_table(rows: &[Row]) {
    println!("| Item | Old ns/op | New ns/op | Ratio | Verdict |");
    println!("| --- | --- | --- | --- | --- |");
    for row in rows {
        let ratio = row.old_ns / row.new_ns;
        if row.new_ns < row.old_ns && ratio >= 1.02 {
            println!(
                "| {} | {:.3} | {:.3} | {:.4} | Faster equal form |",
                row.item, row.old_ns, row.new_ns, ratio
            );
        } else {
            println!(
                "| {} | {:.3} | — | — | No faster equal form |",
                row.item, row.old_ns
            );
        }
        println!(
            "RAW {} old={:.3} new={:.3} ratio={:.4}",
            row.item, row.old_ns, row.new_ns, ratio
        );
    }
}

#[test]
fn r2_09_nonce_line_parity_and_bench() {
    let header: [u8; 32] = hex::decode(NONCE_HASH_KVS[0].header)
        .unwrap()
        .try_into()
        .unwrap();

    for bit in 0..256 {
        let be = set_low_bit(bit);
        let n = U512::from_big_endian(&be);
        assert_counter_matches(&be, n);
    }
    for be in edge_counters() {
        let n = U512::from_big_endian(&be);
        assert_counter_matches(&be, n);
        let sample = sample_from(header, be);
        assert_launch_matches(header, &sample);
        assert_round_trip(header, &sample);
    }

    let mut rng = 0xD1CE_5EED_u64;
    let mut samples = Vec::with_capacity(TIMED);
    for be in edge_counters() {
        samples.push(sample_from(header, be));
    }
    while samples.len() < TIMED {
        samples.push(sample_from(header, random_counter(&mut rng)));
    }

    let mut round_trips = 0usize;
    for sample in samples.iter().take(ROUND_TRIPS) {
        assert_counter_matches(&sample.be, sample.n);
        assert_launch_matches(header, sample);
        assert_round_trip(header, sample);
        let (b, t, pk, nonce) = path_old(sample);
        assert_eq!(path_fast(sample), (b, t, pk, nonce));
        round_trips += 1;
    }
    assert_eq!(round_trips, ROUND_TRIPS);
    println!("ROUND_TRIP passed={round_trips}");

    for sample in samples.iter().skip(ROUND_TRIPS) {
        assert_counter_matches(&sample.be, sample.n);
        assert_eq!(path_fast(sample), path_old(sample));
    }

    let ts: Vec<u32> = samples.iter().map(|s| t_of(&s.be)).collect();
    let deltas: Vec<u32> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| match i % 8 {
            0 => 0,
            1 => 1,
            2 => 7,
            3 => T_MASK as u32,
            4 => 1 << 26,
            5 => u32::MAX / 3,
            6 => u32::MAX,
            _ => t_room_in_k(s.n),
        })
        .collect();
    for (sample, &delta) in samples.iter().zip(&deltas) {
        assert_eq!(
            counter_add_t_fast(sample.n, delta),
            counter_add_t(sample.n, delta)
        );
    }

    let n = samples.len();
    let t_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= t_of(&s.be) as u64;
        }
        black_box(acc);
    });
    let t_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= t_of_fast(&s.be) as u64;
        }
        black_box(acc);
    });

    let k_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= k_of(&s.be);
        }
        black_box(acc);
    });
    let k_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= k_of_fast(&s.be);
        }
        black_box(acc);
    });

    let b_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_words(base_words(&s.be));
        }
        black_box(acc);
    });
    let b_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_words(base_words_fast(&s.be));
        }
        black_box(acc);
    });

    let m_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_bytes(&map_counter(&s.be));
        }
        black_box(acc);
    });
    let m_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_bytes(&map_counter_fast(&s.be));
        }
        black_box(acc);
    });

    let l_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_pk(&launch_params(&s.mid, &s.be));
        }
        black_box(acc);
    });
    let l_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_pk(&launch_params_fast(&s.mid, &s.be));
        }
        black_box(acc);
    });

    let s_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_pk(&segment_launch_params(header, s.n));
        }
        black_box(acc);
    });
    let s_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            acc ^= fold_pk(&segment_launch_params_fast(header, s.n));
        }
        black_box(acc);
    });

    let a_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let deltas = black_box(&deltas);
        let mut acc = 0u64;
        for (s, &d) in samples.iter().zip(deltas) {
            acc ^= fold_u512(counter_add_t(s.n, d));
        }
        black_box(acc);
    });
    let a_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let deltas = black_box(&deltas);
        let mut acc = 0u64;
        for (s, &d) in samples.iter().zip(deltas) {
            acc ^= fold_u512(counter_add_t_fast(s.n, d));
        }
        black_box(acc);
    });

    let n_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let ts = black_box(&ts);
        let mut acc = 0u64;
        for (s, &t) in samples.iter().zip(ts) {
            acc ^= fold_u512(nonce_from_absolute_t(s.n, t));
        }
        black_box(acc);
    });
    let n_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let ts = black_box(&ts);
        let mut acc = 0u64;
        for (s, &t) in samples.iter().zip(ts) {
            acc ^= fold_u512(nonce_from_absolute_t_fast(s.n, t));
        }
        black_box(acc);
    });

    let r_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u32;
        for s in samples {
            acc ^= t_room_in_k(s.n);
        }
        black_box(acc);
    });
    let r_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u32;
        for s in samples {
            acc ^= t_room_in_k_fast(s.n);
        }
        black_box(acc);
    });

    let p_old = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            let (b, t, pk, nonce) = path_old(s);
            acc ^= fold_words(b) ^ (t as u64) ^ fold_pk(&pk) ^ fold_u512(nonce);
        }
        black_box(acc);
    });
    let p_new = time_per_op(n, || {
        let samples = black_box(&samples);
        let mut acc = 0u64;
        for s in samples {
            let (b, t, pk, nonce) = path_fast(s);
            acc ^= fold_words(b) ^ (t as u64) ^ fold_pk(&pk) ^ fold_u512(nonce);
        }
        black_box(acc);
    });

    print_table(&[
        Row { item: "t_of", old_ns: t_old, new_ns: t_new },
        Row { item: "k_of", old_ns: k_old, new_ns: k_new },
        Row { item: "base_words", old_ns: b_old, new_ns: b_new },
        Row { item: "map_counter", old_ns: m_old, new_ns: m_new },
        Row { item: "launch_params", old_ns: l_old, new_ns: l_new },
        Row { item: "segment_launch_params", old_ns: s_old, new_ns: s_new },
        Row { item: "counter_add_t", old_ns: a_old, new_ns: a_new },
        Row { item: "nonce_from_absolute_t", old_ns: n_old, new_ns: n_new },
        Row { item: "t_room_in_k", old_ns: r_old, new_ns: r_new },
        Row {
            item: "full path",
            old_ns: p_old,
            new_ns: p_new,
        },
    ]);
}
