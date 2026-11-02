//! Re-time `lib.rs` mining entry points against `hash_from_nonce`.
//!
//! `hash_from_nonce` is `qpow_math::get_nonce_hash`. `mining_midstate` is the
//! first two permutations. `mining_prestate_low64` is that midstate plus the
//! nonce-invariant prefix of the third permutation (low 64 nonce bits left out).

use pow_core::{
    hash_from_nonce, mining_midstate, mining_midstate_u32s, mining_prestate_low64,
    mining_prestate_low64_u32s, JobContext, NONCE_HASH_KVS, SPONGE_WIDTH,
};
use primitive_types::U512;
use qp_poseidon_core::poseidon2::INITIAL_EXTERNAL_CONSTANTS;
use qp_poseidon_core::{Goldilocks, Poseidon2};
use std::hint::black_box;
use std::time::Instant;

const TRIALS: usize = 5;
const FIRST_REPS: u64 = 3_000;
const RANDOM_REPS: u64 = 120;
const RANDOM_HEADERS: usize = 32;

struct Case {
    header: [u8; 32],
    high: [u8; 32],
    nonce_be: [u8; 64],
    nonce: U512,
    ctx: JobContext,
}

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn decode64(s: &str) -> [u8; 64] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fill_bytes(seed: &mut u64, out: &mut [u8]) {
    for chunk in out.chunks_exact_mut(8) {
        chunk.copy_from_slice(&splitmix(seed).to_le_bytes());
    }
}

fn case_from_parts(header: [u8; 32], nonce_be: [u8; 64]) -> Case {
    let high = nonce_be[..32].try_into().unwrap();
    let nonce = U512::from_big_endian(&nonce_be);
    let ctx = JobContext::new(header, U512::one());
    Case {
        header,
        high,
        nonce_be,
        nonce,
        ctx,
    }
}

fn first_kv_case() -> Case {
    let kv = &NONCE_HASH_KVS[0];
    case_from_parts(decode32(kv.header), decode64(kv.nonce))
}

fn random_cases() -> Vec<Case> {
    let mut seed = 0xA208_11BF_11E0_u64;
    let mut cases = Vec::with_capacity(RANDOM_HEADERS);
    for _ in 0..RANDOM_HEADERS {
        let mut header = [0u8; 32];
        let mut nonce_be = [0u8; 64];
        fill_bytes(&mut seed, &mut header);
        fill_bytes(&mut seed, &mut nonce_be);
        cases.push(case_from_parts(header, nonce_be));
    }
    cases
}

fn absorb_u32s(state: &mut [Goldilocks; SPONGE_WIDTH], bytes: &[u8]) {
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        state[i] += Goldilocks::from_u64(u32::from_le_bytes(chunk.try_into().unwrap()) as u64);
    }
}

/// qp-poseidon-core 3.1.0 external linear layer (chunk loop), not `custom.rs`.
fn library_external_linear_layer(state: &mut [Goldilocks; SPONGE_WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        let x: &mut [Goldilocks; 4] = chunk.try_into().unwrap();
        let t01 = x[0] + x[1];
        let t23 = x[2] + x[3];
        let t0123 = t01 + t23;
        let t01123 = t0123 + x[1];
        let t01233 = t0123 + x[3];
        x[3] = t01233 + x[0].double();
        x[1] = t01123 + x[2].double();
        x[0] = t01123 + t01;
        x[2] = t01233 + t23;
    }
    let sums: [Goldilocks; 4] =
        core::array::from_fn(|k| (0..SPONGE_WIDTH).step_by(4).map(|j| state[j + k]).sum());
    for (i, elem) in state.iter_mut().enumerate() {
        *elem += sums[i % 4];
    }
}

fn library_midstate(poseidon: &Poseidon2, header: [u8; 32], high: [u8; 32]) -> [u64; SPONGE_WIDTH] {
    let mut state = [Goldilocks::ZERO; SPONGE_WIDTH];
    absorb_u32s(&mut state, &header);
    poseidon.permute_mut(&mut state);
    absorb_u32s(&mut state, &high);
    poseidon.permute_mut(&mut state);
    state.map(|g| g.as_canonical_u64())
}

fn library_prestate(
    poseidon: &Poseidon2,
    header: [u8; 32],
    nonce_be: [u8; 64],
) -> [u64; SPONGE_WIDTH] {
    let high: [u8; 32] = nonce_be[..32].try_into().unwrap();
    let mut state = library_midstate(poseidon, header, high).map(Goldilocks::from_u64);
    absorb_u32s(&mut state, &nonce_be[32..56]);
    library_external_linear_layer(&mut state);
    for (felt, constant) in state.iter_mut().zip(INITIAL_EXTERNAL_CONSTANTS[0]) {
        *felt += Goldilocks::from_u64(constant);
    }
    state.map(|g| g.as_canonical_u64())
}

fn pack24(felts: &[u64; SPONGE_WIDTH]) -> [u32; 24] {
    let mut out = [0u32; 24];
    for (i, felt) in felts.iter().enumerate() {
        out[2 * i] = *felt as u32;
        out[2 * i + 1] = (*felt >> 32) as u32;
    }
    out
}

fn assert_case_matches_library(poseidon: &Poseidon2, case: &Case, label: &str) {
    let mid = mining_midstate(case.header, case.high);
    let lib_mid = library_midstate(poseidon, case.header, case.high);
    assert_eq!(
        mid, lib_mid,
        "{label}: mining_midstate != library two-permute"
    );

    let pre = mining_prestate_low64(case.header, case.nonce_be);
    let lib_pre = library_prestate(poseidon, case.header, case.nonce_be);
    assert_eq!(
        pre, lib_pre,
        "{label}: mining_prestate_low64 != library mid + linear + RC0"
    );
    assert_eq!(
        mining_midstate_u32s(case.header, case.high),
        pack24(&mid),
        "{label}: midstate u32 pack"
    );
    assert_eq!(
        mining_prestate_low64_u32s(case.header, case.nonce_be),
        pack24(&pre),
        "{label}: prestate u32 pack"
    );

    let hash = hash_from_nonce(&case.ctx, case.nonce);
    let direct = qpow_math::get_nonce_hash(case.header, case.nonce_be);
    assert_eq!(hash, direct, "{label}: hash_from_nonce != get_nonce_hash");

    let mut low_only = case.nonce_be;
    low_only[63] ^= 0x5a;
    assert_eq!(
        mining_midstate(case.header, case.high),
        mid,
        "{label}: low-half flip changed midstate"
    );
    assert_eq!(
        mining_prestate_low64(case.header, low_only),
        pre,
        "{label}: low-64 flip changed prestate"
    );
    let flipped = U512::from_big_endian(&low_only);
    assert_ne!(
        hash_from_nonce(&case.ctx, flipped),
        hash,
        "{label}: low-64 flip left the full hash unchanged"
    );

    let mut upper_low = case.nonce_be;
    upper_low[32] ^= 0xa5;
    assert_eq!(
        mining_midstate(case.header, upper_low[..32].try_into().unwrap()),
        mid,
        "{label}: nonce[32] flip changed midstate"
    );
    assert_ne!(
        mining_prestate_low64(case.header, upper_low),
        pre,
        "{label}: nonce[32] flip left prestate unchanged"
    );
}

fn assert_goldens(poseidon: &Poseidon2) {
    assert!(
        !NONCE_HASH_KVS.is_empty(),
        "NONCE_HASH_KVS must contain the first vector"
    );
    for (i, kv) in NONCE_HASH_KVS.iter().enumerate() {
        let header = decode32(kv.header);
        let nonce_be = decode64(kv.nonce);
        let case = case_from_parts(header, nonce_be);
        let mid = mining_midstate(header, case.high);
        assert_eq!(mid, kv.mid, "kv {i}: mining_midstate != stored mid");
        let hash = hash_from_nonce(&case.ctx, case.nonce);
        assert_eq!(
            hash.to_big_endian(),
            decode64(kv.hash),
            "kv {i}: hash_from_nonce != stored hash"
        );
        assert_case_matches_library(poseidon, &case, &format!("kv {i}"));
    }
}

fn median_ns(mut samples: [f64; TRIALS]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[TRIALS / 2]
}

fn bench_first(case: &Case) -> (f64, f64, f64, f64) {
    let mut sink = 0u64;
    for _ in 0..64 {
        sink ^= mining_midstate(black_box(case.header), black_box(case.high))[0];
        sink ^= mining_prestate_low64(black_box(case.header), black_box(case.nonce_be))[0];
        sink ^= hash_from_nonce(&case.ctx, black_box(case.nonce)).low_u64();
    }
    black_box(sink);

    let mut mid_ns = [0f64; TRIALS];
    let mut pre_ns = [0f64; TRIALS];
    let mut hash_ns = [0f64; TRIALS];
    let mut group_ns = [0f64; TRIALS];

    for trial in 0..TRIALS {
        let t0 = Instant::now();
        for _ in 0..FIRST_REPS {
            let mid = mining_midstate(black_box(case.header), black_box(case.high));
            sink ^= black_box(mid[0]);
        }
        mid_ns[trial] = t0.elapsed().as_nanos() as f64 / FIRST_REPS as f64;

        let t1 = Instant::now();
        for _ in 0..FIRST_REPS {
            let pre = mining_prestate_low64(black_box(case.header), black_box(case.nonce_be));
            sink ^= black_box(pre[0]);
        }
        pre_ns[trial] = t1.elapsed().as_nanos() as f64 / FIRST_REPS as f64;

        let t2 = Instant::now();
        for _ in 0..FIRST_REPS {
            let hash = hash_from_nonce(&case.ctx, black_box(case.nonce));
            sink ^= black_box(hash.low_u64());
        }
        hash_ns[trial] = t2.elapsed().as_nanos() as f64 / FIRST_REPS as f64;

        let t3 = Instant::now();
        for _ in 0..FIRST_REPS {
            let mid = mining_midstate(black_box(case.header), black_box(case.high));
            let pre = mining_prestate_low64(black_box(case.header), black_box(case.nonce_be));
            sink ^= black_box(mid[0] ^ pre[0]);
        }
        group_ns[trial] = t3.elapsed().as_nanos() as f64 / FIRST_REPS as f64;
    }
    black_box(sink);
    (
        median_ns(hash_ns),
        median_ns(mid_ns),
        median_ns(pre_ns),
        median_ns(group_ns),
    )
}

fn bench_random(cases: &[Case]) -> (f64, f64, f64, f64) {
    assert_eq!(cases.len(), RANDOM_HEADERS);
    let calls = RANDOM_REPS * cases.len() as u64;
    let mut sink = 0u64;
    for case in cases {
        sink ^= mining_midstate(black_box(case.header), black_box(case.high))[0];
        sink ^= mining_prestate_low64(black_box(case.header), black_box(case.nonce_be))[0];
        sink ^= hash_from_nonce(&case.ctx, black_box(case.nonce)).low_u64();
    }
    black_box(sink);

    let mut mid_ns = [0f64; TRIALS];
    let mut pre_ns = [0f64; TRIALS];
    let mut hash_ns = [0f64; TRIALS];
    let mut group_ns = [0f64; TRIALS];

    for trial in 0..TRIALS {
        let t0 = Instant::now();
        for _ in 0..RANDOM_REPS {
            for case in cases {
                let mid = mining_midstate(black_box(case.header), black_box(case.high));
                sink ^= black_box(mid[0]);
            }
        }
        mid_ns[trial] = t0.elapsed().as_nanos() as f64 / calls as f64;

        let t1 = Instant::now();
        for _ in 0..RANDOM_REPS {
            for case in cases {
                let pre = mining_prestate_low64(black_box(case.header), black_box(case.nonce_be));
                sink ^= black_box(pre[0]);
            }
        }
        pre_ns[trial] = t1.elapsed().as_nanos() as f64 / calls as f64;

        let t2 = Instant::now();
        for _ in 0..RANDOM_REPS {
            for case in cases {
                let hash = hash_from_nonce(&case.ctx, black_box(case.nonce));
                sink ^= black_box(hash.low_u64());
            }
        }
        hash_ns[trial] = t2.elapsed().as_nanos() as f64 / calls as f64;

        let t3 = Instant::now();
        for _ in 0..RANDOM_REPS {
            for case in cases {
                let mid = mining_midstate(black_box(case.header), black_box(case.high));
                let pre = mining_prestate_low64(black_box(case.header), black_box(case.nonce_be));
                sink ^= black_box(mid[0] ^ pre[0]);
            }
        }
        group_ns[trial] = t3.elapsed().as_nanos() as f64 / calls as f64;
    }
    black_box(sink);
    (
        median_ns(hash_ns),
        median_ns(mid_ns),
        median_ns(pre_ns),
        median_ns(group_ns),
    )
}

fn emit(dataset: &str, hash_ns: f64, mid_ns: f64, pre_ns: f64, group_ns: f64) {
    println!("R2_08 {dataset} hash_from_nonce {hash_ns:.3}");
    println!("R2_08 {dataset} mining_midstate {mid_ns:.3}");
    println!("R2_08 {dataset} mining_prestate_low64 {pre_ns:.3}");
    println!("R2_08 {dataset} midstate_plus_prestate {group_ns:.3}");
    println!(
        "R2_08 {dataset} ratios mid {:.4} pre {:.4} group {:.4}",
        hash_ns / mid_ns,
        hash_ns / pre_ns,
        hash_ns / group_ns
    );
}

#[test]
fn r2_08_lib_file_mining_functions() {
    let poseidon = Poseidon2::new();
    assert_goldens(&poseidon);
    let random = random_cases();
    assert_eq!(random.len(), RANDOM_HEADERS);
    for (i, case) in random.iter().enumerate() {
        assert_ne!(
            case.header, [0u8; 32],
            "random header {i} collapsed to the zero header"
        );
        assert_case_matches_library(&poseidon, case, &format!("random {i}"));
    }
    println!("R2_08_GOLDEN pass");
    println!("R2_08_RANDOM pass");

    let first = first_kv_case();
    let (h0, m0, p0, g0) = bench_first(&first);
    emit("first_kv", h0, m0, p0, g0);
    let (h1, m1, p1, g1) = bench_random(&random);
    emit("random32", h1, m1, p1, g1);
}
