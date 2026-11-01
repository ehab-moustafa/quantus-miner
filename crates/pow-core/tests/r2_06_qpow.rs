//! Fresh re-challenge of the qpow-math nonce hash.
//!
//! Library path (`qpow-math` `get_nonce_hash` → `hash_squeeze_twice`):
//! 96-byte header‖nonce, injective 4-byte felts (24 limbs + terminator),
//! sponge pad `1`, five Poseidon2 permutations, 64-byte double squeeze, `U512`.
//!
//! Alternatives live only in this test. They do not change library semantics.

use pow_core::{hash_from_nonce, JobContext, NONCE_HASH_KVS, SPONGE_WIDTH};
use primitive_types::U512;
use qp_poseidon_core::serialization::bytes_to_felts_iter;
use qp_poseidon_core::{hash_squeeze_twice, Goldilocks, Poseidon2};
use qpow_math::{achieved_difficulty_from_hash, get_nonce_hash, is_valid_nonce, mine_range};
use std::hint::black_box;
use std::time::Instant;

const HASH_ITERS: usize = 2_000;
const ROUNDS: usize = 5;
const BYTE_REPEATS: usize = 80;
const CMP_PASSES: usize = 8_000;
const ACH_REPEATS: usize = 128;
const MINE_STEPS: u64 = 1_500;

type State = [Goldilocks; SPONGE_WIDTH];
type MidCache = Option<([u8; 32], State)>;

struct Input {
    nonce_u: U512,
    nonce_be: [u8; 64],
    preimage: [u8; 96],
}

#[inline(always)]
fn absorb8(state: &mut State, bytes: &[u8]) {
    debug_assert_eq!(bytes.len(), 32);
    for i in 0..4 {
        let v = u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        let o = i * 2;
        state[o] += Goldilocks::new(v as u32 as u64);
        state[o + 1] += Goldilocks::new(v >> 32);
    }
}

#[inline(always)]
fn write_digest(state: &State, out: &mut [u8]) {
    for (i, chunk) in out.chunks_exact_mut(8).take(4).enumerate() {
        chunk.copy_from_slice(&state[i].as_canonical_u64().to_le_bytes());
    }
}

/// Fixed 96-byte PoW sponge. `permute` is either `custom::permute_mut` or `Poseidon2::permute_mut`.
#[inline(always)]
fn squeeze_fused(
    header: &[u8; 32],
    nonce: &[u8; 64],
    mut permute: impl FnMut(&mut State),
) -> [u8; 64] {
    let mut state = [Goldilocks::ZERO; SPONGE_WIDTH];
    absorb8(&mut state, header);
    permute(&mut state);
    absorb8(&mut state, &nonce[..32]);
    permute(&mut state);
    absorb8(&mut state, &nonce[32..]);
    permute(&mut state);
    // Terminator felt from `bytes_to_felts_iter`, then the sponge padding 1.
    state[0] += Goldilocks::ONE;
    state[1] += Goldilocks::ONE;
    permute(&mut state);
    let mut out = [0u8; 64];
    write_digest(&state, &mut out[..32]);
    permute(&mut state);
    write_digest(&state, &mut out[32..]);
    out
}

#[inline(always)]
fn squeeze_custom(header: &[u8; 32], nonce: &[u8; 64]) -> [u8; 64] {
    squeeze_fused(header, nonce, |state| pow_core::custom::permute_mut(state))
}

#[inline(always)]
fn squeeze_lib(poseidon: &Poseidon2, header: &[u8; 32], nonce: &[u8; 64]) -> [u8; 64] {
    squeeze_fused(header, nonce, |state| poseidon.permute_mut(state))
}

fn alt_get_nonce_hash(header: [u8; 32], nonce: [u8; 64]) -> U512 {
    U512::from_big_endian(&squeeze_custom(&header, &nonce))
}

#[inline(always)]
fn permute<const CUSTOM: bool>(poseidon: &Poseidon2, state: &mut State) {
    if CUSTOM {
        pow_core::custom::permute_mut(state);
    } else {
        poseidon.permute_mut(state);
    }
}

fn mid_state<const CUSTOM: bool>(
    poseidon: &Poseidon2,
    header: &[u8; 32],
    high: &[u8; 32],
    cache: &mut MidCache,
) -> State {
    if let Some((cached_high, state)) = cache {
        if cached_high == high {
            return *state;
        }
    }
    let mut state = [Goldilocks::ZERO; SPONGE_WIDTH];
    absorb8(&mut state, header);
    permute::<CUSTOM>(poseidon, &mut state);
    absorb8(&mut state, high);
    permute::<CUSTOM>(poseidon, &mut state);
    *cache = Some((*high, state));
    state
}

/// Full 64-byte hash resuming after header + nonce-high. Always squeezes twice.
fn hash_midstate<const CUSTOM: bool>(
    poseidon: &Poseidon2,
    header: &[u8; 32],
    nonce: &[u8; 64],
    cache: &mut MidCache,
) -> [u8; 64] {
    let high: [u8; 32] = nonce[..32].try_into().unwrap();
    let mut state = mid_state::<CUSTOM>(poseidon, header, &high, cache);
    absorb8(&mut state, &nonce[32..]);
    permute::<CUSTOM>(poseidon, &mut state);
    state[0] += Goldilocks::ONE;
    state[1] += Goldilocks::ONE;
    permute::<CUSTOM>(poseidon, &mut state);
    let mut out = [0u8; 64];
    write_digest(&state, &mut out[..32]);
    permute::<CUSTOM>(poseidon, &mut state);
    write_digest(&state, &mut out[32..]);
    out
}

/// Mining group: absorb, permute, squeeze, compare. Skips the second squeeze on a high-half reject.
fn meets_target<const CUSTOM: bool>(
    poseidon: &Poseidon2,
    header: &[u8; 32],
    nonce: &[u8; 64],
    target_be: &[u8; 64],
    cache: &mut MidCache,
) -> bool {
    let high: [u8; 32] = nonce[..32].try_into().unwrap();
    let mut state = mid_state::<CUSTOM>(poseidon, header, &high, cache);
    absorb8(&mut state, &nonce[32..]);
    permute::<CUSTOM>(poseidon, &mut state);
    state[0] += Goldilocks::ONE;
    state[1] += Goldilocks::ONE;
    permute::<CUSTOM>(poseidon, &mut state);
    let mut hi = [0u8; 32];
    write_digest(&state, &mut hi);
    if hi.as_slice() > &target_be[..32] {
        return false;
    }
    if hi.as_slice() < &target_be[..32] {
        return true;
    }
    permute::<CUSTOM>(poseidon, &mut state);
    let mut full = [0u8; 64];
    full[..32].copy_from_slice(&hi);
    write_digest(&state, &mut full[32..]);
    full.as_slice() < target_be.as_slice()
}

fn alt_is_valid(header: [u8; 32], nonce: [u8; 64], difficulty: U512) -> (bool, U512) {
    if difficulty.is_zero() {
        return (false, U512::zero());
    }
    let hash = alt_get_nonce_hash(header, nonce);
    (hash < U512::MAX / difficulty, hash)
}

fn alt_achieved(hash: U512) -> U512 {
    if hash.is_zero() {
        U512::MAX
    } else {
        U512::MAX / hash
    }
}

fn limb_lt(hash: U512, target: U512) -> bool {
    let hash = hash.0;
    let target = target.0;
    let mut i = 8;
    while i > 0 {
        i -= 1;
        if hash[i] != target[i] {
            return hash[i] < target[i];
        }
    }
    false
}

/// 24 little-endian u32 limbs plus the empty-remainder terminator word `1`.
fn felts96(input: &[u8; 96]) -> [u64; 25] {
    let mut out = [0u64; 25];
    for i in 0..12 {
        let v = u64::from_le_bytes(input[i * 8..i * 8 + 8].try_into().unwrap());
        out[i * 2] = v as u32 as u64;
        out[i * 2 + 1] = v >> 32;
    }
    out[24] = 1;
    out
}

fn felts96_lib(input: &[u8; 96]) -> [u64; 25] {
    let mut out = [0u64; 25];
    for (slot, felt) in out.iter_mut().zip(bytes_to_felts_iter(input)) {
        *slot = felt.as_canonical_u64();
    }
    out
}

fn alt_mine_range<const CUSTOM: bool>(
    header: [u8; 32],
    start_nonce: [u8; 64],
    steps: u64,
    difficulty: U512,
) -> Option<([u8; 64], U512)> {
    if difficulty.is_zero() {
        return None;
    }
    let target_be = (U512::MAX / difficulty).to_big_endian();
    let poseidon = Poseidon2::new();
    let mut cache = None;
    let mut nonce = U512::from_big_endian(&start_nonce);
    for _ in 0..steps {
        let nonce_be = nonce.to_big_endian();
        if meets_target::<CUSTOM>(&poseidon, &header, &nonce_be, &target_be, &mut cache) {
            let hash = U512::from_big_endian(&hash_midstate::<CUSTOM>(
                &poseidon, &header, &nonce_be, &mut cache,
            ));
            return Some((nonce_be, hash));
        }
        nonce = nonce.saturating_add(U512::one());
    }
    None
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn decode64(s: &str) -> [u8; 64] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn rand_bytes(rng: &mut u64, dest: &mut [u8]) {
    for chunk in dest.chunks_mut(8) {
        let bytes = splitmix(rng).to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
}

fn rand_nonzero_u512(rng: &mut u64) -> U512 {
    loop {
        let limbs = std::array::from_fn(|_| splitmix(rng));
        let n = U512(limbs);
        if !n.is_zero() {
            return n;
        }
    }
}

fn assert_hash_eq(header: [u8; 32], nonce: [u8; 64]) {
    let lib = get_nonce_hash(header, nonce);
    let custom = alt_get_nonce_hash(header, nonce);
    assert_eq!(custom, lib, "custom fused squeeze != get_nonce_hash");

    let poseidon = Poseidon2::new();
    let lib_perm = U512::from_big_endian(&squeeze_lib(&poseidon, &header, &nonce));
    assert_eq!(
        lib_perm, lib,
        "hoisted library permute squeeze != get_nonce_hash"
    );

    let mut preimage = [0u8; 96];
    preimage[..32].copy_from_slice(&header);
    preimage[32..].copy_from_slice(&nonce);
    assert_eq!(
        hash_squeeze_twice(&preimage),
        lib.to_big_endian(),
        "hash_squeeze_twice != get_nonce_hash bytes"
    );
    assert_eq!(felts96(&preimage), felts96_lib(&preimage));

    let mut cache = None;
    let mid = U512::from_big_endian(&hash_midstate::<false>(
        &poseidon, &header, &nonce, &mut cache,
    ));
    assert_eq!(mid, lib, "midstate resume != get_nonce_hash");
    let mid_again = U512::from_big_endian(&hash_midstate::<false>(
        &poseidon, &header, &nonce, &mut cache,
    ));
    assert_eq!(mid_again, lib, "warm midstate != get_nonce_hash");
    let mut custom_cache = None;
    let custom_mid = U512::from_big_endian(&hash_midstate::<true>(
        &poseidon,
        &header,
        &nonce,
        &mut custom_cache,
    ));
    assert_eq!(custom_mid, lib, "custom midstate != get_nonce_hash");
}

fn assert_meets(header: [u8; 32], nonce: [u8; 64], target: U512) {
    let hash = get_nonce_hash(header, nonce);
    let poseidon = Poseidon2::new();
    let target_be = target.to_big_endian();
    let mut cache = None;
    let got = meets_target::<false>(&poseidon, &header, &nonce, &target_be, &mut cache);
    assert_eq!(got, hash < target, "meets_target diverged");
    let mut custom_cache = None;
    let custom_got =
        meets_target::<true>(&poseidon, &header, &nonce, &target_be, &mut custom_cache);
    assert_eq!(custom_got, hash < target, "custom meets_target diverged");
    assert_eq!(limb_lt(hash, target), hash < target);
}

fn assert_parity() {
    for (i, kv) in NONCE_HASH_KVS.iter().enumerate() {
        let header = decode32(kv.header);
        let nonce = decode64(kv.nonce);
        assert_hash_eq(header, nonce);
        assert_eq!(
            get_nonce_hash(header, nonce).to_big_endian(),
            decode64(kv.hash),
            "golden kv {i}"
        );
        let hash = U512::from_big_endian(&decode64(kv.hash));
        for difficulty in [
            U512::one(),
            U512::from(256u64),
            U512::from(1u64) << 20,
            U512::MAX,
        ] {
            assert_eq!(
                alt_is_valid(header, nonce, difficulty),
                is_valid_nonce(header, nonce, difficulty),
                "is_valid kv {i}"
            );
            assert_meets(header, nonce, U512::MAX / difficulty);
        }
        for target in [
            U512::zero(),
            U512::one(),
            hash,
            hash.saturating_sub(U512::one()),
            hash.saturating_add(U512::one()),
            U512::MAX,
        ] {
            assert_meets(header, nonce, target);
        }
        assert_eq!(alt_achieved(hash), achieved_difficulty_from_hash(hash));
    }
    assert_eq!(
        alt_achieved(U512::zero()),
        achieved_difficulty_from_hash(U512::zero())
    );
    assert_eq!(
        alt_is_valid([0; 32], [0; 64], U512::zero()),
        is_valid_nonce([0; 32], [0; 64], U512::zero())
    );

    let mut rng = 0xA206_D1CE_u64;
    for _ in 0..20 {
        let mut header = [0u8; 32];
        let mut nonce = [0u8; 64];
        rand_bytes(&mut rng, &mut header);
        rand_bytes(&mut rng, &mut nonce);
        assert_hash_eq(header, nonce);
        let difficulty = rand_nonzero_u512(&mut rng);
        assert_eq!(
            alt_is_valid(header, nonce, difficulty),
            is_valid_nonce(header, nonce, difficulty)
        );
        let hash = get_nonce_hash(header, nonce);
        assert_meets(header, nonce, U512::MAX / difficulty);
        assert_meets(header, nonce, hash);
        assert_eq!(alt_achieved(hash), achieved_difficulty_from_hash(hash));
        let ctx = JobContext::new(header, U512::one());
        assert_eq!(hash_from_nonce(&ctx, U512::from_big_endian(&nonce)), hash);
    }

    let mut nonce = (U512::one() << 256) - U512::from(10u64);
    let header = [0x5au8; 32];
    let poseidon = Poseidon2::new();
    let mut cache = None;
    for _ in 0..20 {
        let nonce_be = nonce.to_big_endian();
        let lib = get_nonce_hash(header, nonce_be);
        let got = U512::from_big_endian(&hash_midstate::<false>(
            &poseidon, &header, &nonce_be, &mut cache,
        ));
        assert_eq!(got, lib, "midstate across the 2^256 boundary");
        assert_meets(header, nonce_be, lib);
        assert_meets(header, nonce_be, lib.saturating_add(U512::one()));
        nonce = nonce.saturating_add(U512::one());
    }

    let header = [0u8; 32];
    let start = [0u8; 64];
    assert_eq!(
        alt_mine_range::<true>(header, start, 0, U512::one()),
        mine_range(header, start, 0, U512::one())
    );
    assert_eq!(
        alt_mine_range::<false>(header, start, 4, U512::zero()),
        mine_range(header, start, 4, U512::zero())
    );
    assert_eq!(
        alt_mine_range::<true>(header, start, 3, U512::one()),
        mine_range(header, start, 3, U512::one())
    );
    assert_eq!(
        alt_mine_range::<false>(header, start, 8, U512::MAX),
        mine_range(header, start, 8, U512::MAX)
    );
    for _ in 0..4 {
        let mut header = [0u8; 32];
        let mut start = [0u8; 64];
        rand_bytes(&mut rng, &mut header);
        rand_bytes(&mut rng, &mut start);
        let difficulty = rand_nonzero_u512(&mut rng);
        assert_eq!(
            alt_mine_range::<true>(header, start, 6, difficulty),
            mine_range(header, start, 6, difficulty)
        );
    }
}

fn bench_inputs() -> ([u8; 32], Vec<Input>) {
    let header = [0x42u8; 32];
    let mut nonce = (U512::from(0x9e37_79b9_u64) << 320) + U512::from(0x1000_0000_u64);
    let mut out = Vec::with_capacity(HASH_ITERS);
    for _ in 0..HASH_ITERS {
        let nonce_be = nonce.to_big_endian();
        let mut preimage = [0u8; 96];
        preimage[..32].copy_from_slice(&header);
        preimage[32..].copy_from_slice(&nonce_be);
        out.push(Input {
            nonce_u: nonce,
            nonce_be,
            preimage,
        });
        nonce = nonce.saturating_add(U512::one());
    }
    (header, out)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn report(item: &str, old_total_ns: &[f64], new_total_ns: &[f64], iters: f64) {
    let old_ns: Vec<f64> = old_total_ns.iter().map(|t| t / iters).collect();
    let new_ns: Vec<f64> = new_total_ns.iter().map(|t| t / iters).collect();
    let ratios: Vec<f64> = old_ns.iter().zip(&new_ns).map(|(o, n)| o / n).collect();
    let old_med = median(&old_ns);
    let new_med = median(&new_ns);
    let ratio = old_med / new_med;
    let rmin = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let rmax = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("ROW\t{item}\t{old_med:.3}\t{new_med:.3}\t{ratio:.4}\t{rmin:.4}\t{rmax:.4}");
}

fn time_pair(
    mut old_run: impl FnMut() -> u64,
    mut new_run: impl FnMut() -> u64,
) -> ([f64; ROUNDS], [f64; ROUNDS]) {
    black_box(old_run());
    black_box(new_run());
    let mut old_ns = [0.0; ROUNDS];
    let mut new_ns = [0.0; ROUNDS];
    for i in 0..ROUNDS {
        if i % 2 == 0 {
            let start = Instant::now();
            black_box(old_run());
            old_ns[i] = start.elapsed().as_nanos() as f64;
            let start = Instant::now();
            black_box(new_run());
            new_ns[i] = start.elapsed().as_nanos() as f64;
        } else {
            let start = Instant::now();
            black_box(new_run());
            new_ns[i] = start.elapsed().as_nanos() as f64;
            let start = Instant::now();
            black_box(old_run());
            old_ns[i] = start.elapsed().as_nanos() as f64;
        }
    }
    (old_ns, new_ns)
}

#[test]
fn r2_06_qpow_rechallenge() {
    assert_parity();
    println!("ASSERTS passed");

    let (header, inputs) = bench_inputs();
    let difficulty = U512::from(1u64) << 40;
    let ctx = JobContext::new(header, difficulty);
    let target = ctx.target;
    let target_be = target.to_big_endian();
    let mut reject_samples = 0;
    for input in inputs.iter().take(32) {
        if get_nonce_hash(header, input.nonce_be) >= target {
            reject_samples += 1;
        }
    }
    assert!(
        reject_samples >= 30,
        "group bench must be reject-heavy, got {reject_samples}/32"
    );
    println!("REJECT_FRAC_SAMPLE {reject_samples}/32 difficulty_bits=40");

    let poseidon = Poseidon2::new();

    let byte_ops = (HASH_ITERS * BYTE_REPEATS) as f64;
    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for _ in 0..BYTE_REPEATS {
                for input in &inputs {
                    let felt = felts96_lib(black_box(&input.preimage));
                    acc ^= felt[0] ^ felt[23] ^ felt[24];
                }
            }
            acc
        },
        || {
            let mut acc = 0u64;
            for _ in 0..BYTE_REPEATS {
                for input in &inputs {
                    let felt = felts96(black_box(&input.preimage));
                    acc ^= felt[0] ^ felt[23] ^ felt[24];
                }
            }
            acc
        },
    );
    report("bytes_to_felts_iter 96B", &old_ns, &new_ns, byte_ops);

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = hash_squeeze_twice(black_box(&input.preimage));
                acc ^= u64::from_le_bytes(hash[..8].try_into().unwrap());
            }
            acc
        },
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = squeeze_custom(black_box(&header), black_box(&input.nonce_be));
                acc ^= u64::from_le_bytes(hash[..8].try_into().unwrap());
            }
            acc
        },
    );
    report(
        "hash_squeeze_twice vs fused custom permute",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = hash_squeeze_twice(black_box(&input.preimage));
                acc ^= u64::from_le_bytes(hash[..8].try_into().unwrap());
            }
            acc
        },
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = squeeze_lib(&poseidon, black_box(&header), black_box(&input.nonce_be));
                acc ^= u64::from_le_bytes(hash[..8].try_into().unwrap());
            }
            acc
        },
    );
    report(
        "hash_squeeze_twice vs fused hoisted Poseidon2",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = get_nonce_hash(black_box(header), black_box(input.nonce_be));
                acc ^= hash.0[0];
            }
            acc
        },
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = alt_get_nonce_hash(black_box(header), black_box(input.nonce_be));
                acc ^= hash.0[0];
            }
            acc
        },
    );
    report(
        "get_nonce_hash single-shot",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = get_nonce_hash(black_box(header), black_box(input.nonce_be));
                acc ^= hash.0[0];
            }
            acc
        },
        || {
            let mut cache = None;
            let mut acc = 0u64;
            for input in &inputs {
                let hash = U512::from_big_endian(&hash_midstate::<false>(
                    &poseidon,
                    black_box(&header),
                    black_box(&input.nonce_be),
                    &mut cache,
                ));
                acc ^= hash.0[0];
            }
            acc
        },
    );
    report(
        "get_nonce_hash sequential midstate full hash lib permute",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let hash = get_nonce_hash(black_box(header), black_box(input.nonce_be));
                acc ^= hash.0[0];
            }
            acc
        },
        || {
            let mut cache = None;
            let mut acc = 0u64;
            for input in &inputs {
                let hash = U512::from_big_endian(&hash_midstate::<true>(
                    &poseidon,
                    black_box(&header),
                    black_box(&input.nonce_be),
                    &mut cache,
                ));
                acc ^= hash.0[0];
            }
            acc
        },
    );
    report(
        "get_nonce_hash sequential midstate full hash custom permute",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let (valid, hash) = is_valid_nonce(
                    black_box(header),
                    black_box(input.nonce_be),
                    black_box(difficulty),
                );
                acc ^= hash.0[0] ^ u64::from(valid);
            }
            acc
        },
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let (valid, hash) = alt_is_valid(
                    black_box(header),
                    black_box(input.nonce_be),
                    black_box(difficulty),
                );
                acc ^= hash.0[0] ^ u64::from(valid);
            }
            acc
        },
    );
    report("is_valid_nonce", &old_ns, &new_ns, HASH_ITERS as f64);

    let hashes: Vec<U512> = inputs
        .iter()
        .take(256)
        .map(|i| get_nonce_hash(header, i.nonce_be))
        .collect();
    let cmp_target = hashes[0];
    for hash in &hashes {
        assert_eq!(limb_lt(*hash, cmp_target), *hash < cmp_target);
    }
    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for _ in 0..CMP_PASSES {
                for hash in &hashes {
                    acc ^= u64::from(black_box(*hash) < black_box(cmp_target));
                }
            }
            acc
        },
        || {
            let mut acc = 0u64;
            for _ in 0..CMP_PASSES {
                for hash in &hashes {
                    acc ^= u64::from(limb_lt(black_box(*hash), black_box(cmp_target)));
                }
            }
            acc
        },
    );
    report(
        "hash < target limb walk",
        &old_ns,
        &new_ns,
        (CMP_PASSES * hashes.len()) as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for hash in &hashes {
                for _ in 0..ACH_REPEATS {
                    acc ^= alt_touch(achieved_difficulty_from_hash(black_box(*hash)));
                }
            }
            acc
        },
        || {
            let mut acc = 0u64;
            for hash in &hashes {
                for _ in 0..ACH_REPEATS {
                    acc ^= alt_touch(alt_achieved(black_box(*hash)));
                }
            }
            acc
        },
    );
    report(
        "achieved_difficulty_from_hash",
        &old_ns,
        &new_ns,
        (hashes.len() * ACH_REPEATS) as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let valid = get_nonce_hash(black_box(header), black_box(input.nonce_be)) < target;
                acc ^= u64::from(valid);
            }
            acc
        },
        || {
            let mut cache = None;
            let mut acc = 0u64;
            for input in &inputs {
                let valid = meets_target::<false>(
                    &poseidon,
                    black_box(&header),
                    black_box(&input.nonce_be),
                    &target_be,
                    &mut cache,
                );
                acc ^= u64::from(valid);
            }
            acc
        },
    );
    report(
        "group vs get_nonce_hash lib permute",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let valid = get_nonce_hash(black_box(header), black_box(input.nonce_be)) < target;
                acc ^= u64::from(valid);
            }
            acc
        },
        || {
            let mut cache = None;
            let mut acc = 0u64;
            for input in &inputs {
                let valid = meets_target::<true>(
                    &poseidon,
                    black_box(&header),
                    black_box(&input.nonce_be),
                    &target_be,
                    &mut cache,
                );
                acc ^= u64::from(valid);
            }
            acc
        },
    );
    report(
        "group vs get_nonce_hash custom permute",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let (old_ns, new_ns) = time_pair(
        || {
            let mut acc = 0u64;
            for input in &inputs {
                let valid = hash_from_nonce(black_box(&ctx), black_box(input.nonce_u)) < target;
                acc ^= u64::from(valid);
            }
            acc
        },
        || {
            let mut cache = None;
            let mut acc = 0u64;
            for input in &inputs {
                let nonce_be = input.nonce_u.to_big_endian();
                let valid = meets_target::<true>(
                    &poseidon,
                    black_box(&header),
                    black_box(&nonce_be),
                    &target_be,
                    &mut cache,
                );
                acc ^= u64::from(valid);
            }
            acc
        },
    );
    report(
        "group vs hash_from_nonce custom permute",
        &old_ns,
        &new_ns,
        HASH_ITERS as f64,
    );

    let start = inputs[0].nonce_be;
    let hard = U512::MAX;
    assert_eq!(
        alt_mine_range::<true>(header, start, MINE_STEPS, hard),
        mine_range(header, start, MINE_STEPS, hard)
    );
    let (old_ns, new_ns) = time_pair(
        || {
            let found = mine_range(
                black_box(header),
                black_box(start),
                MINE_STEPS,
                black_box(hard),
            );
            u64::from(found.is_some())
        },
        || {
            let found = alt_mine_range::<true>(
                black_box(header),
                black_box(start),
                MINE_STEPS,
                black_box(hard),
            );
            u64::from(found.is_some())
        },
    );
    report(
        "qpow-math hash file mine_range custom permute",
        &old_ns,
        &new_ns,
        MINE_STEPS as f64,
    );
}

fn alt_touch(n: U512) -> u64 {
    n.0[0] ^ n.0[7]
}
