//! Phase 1 agent 09: `hash_squeeze_twice` (qp-poseidon-core 3.1.0) vs custom sponge.
//!
//! Mining reference hash is `qpow_math::get_nonce_hash` → `hash_squeeze_twice` on a
//! 96-byte header‖nonce preimage (64-byte BE hash output). Production `MiningHasher`
//! can skip the second squeeze when the high 32 bytes already exceed the target; the
//! first 32 bytes of that path match the library's first squeeze on NONCE_HASH fixtures.

use pow_core::NONCE_HASH_KVS;
use qp_poseidon_core::{
    hash_squeeze_twice,
    poseidon2::{Poseidon2, POSEIDON2_OUTPUT, SPONGE_RATE, SPONGE_WIDTH},
    serialization::{bytes_to_felts_iter, digest_to_bytes},
    Goldilocks,
};
use std::hint::black_box;
use std::time::Instant;

struct Sponge {
    poseidon2: Poseidon2,
    state: [Goldilocks; SPONGE_WIDTH],
    buf: [Goldilocks; SPONGE_RATE],
    buf_len: usize,
}

impl Sponge {
    fn new() -> Self {
        Self {
            poseidon2: Poseidon2::new(),
            state: [Goldilocks::ZERO; SPONGE_WIDTH],
            buf: [Goldilocks::ZERO; SPONGE_RATE],
            buf_len: 0,
        }
    }

    #[inline]
    fn push_to_buf(&mut self, x: Goldilocks) {
        self.buf[self.buf_len] = x;
        self.buf_len += 1;
        if self.buf_len == SPONGE_RATE {
            self.absorb_full_block();
        }
    }

    #[inline]
    fn absorb_full_block(&mut self) {
        for i in 0..SPONGE_RATE {
            self.state[i] += self.buf[i];
        }
        self.poseidon2.permute_mut(&mut self.state);
        self.buf = [Goldilocks::ZERO; SPONGE_RATE];
        self.buf_len = 0;
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        for felt in bytes_to_felts_iter(bytes) {
            self.push_to_buf(felt);
        }
    }

    fn finalize_state(&mut self) {
        self.push_to_buf(Goldilocks::ONE);
        while self.buf_len != 0 {
            self.push_to_buf(Goldilocks::ZERO);
        }
    }

    fn digest(&self) -> &[Goldilocks; POSEIDON2_OUTPUT] {
        self.state[..POSEIDON2_OUTPUT]
            .try_into()
            .expect("POSEIDON2_OUTPUT <= SPONGE_WIDTH")
    }

    /// Library-equivalent double squeeze (64-byte output).
    fn finalize_squeeze_twice(&mut self) -> [u8; 64] {
        self.finalize_state();
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&digest_to_bytes(self.digest()));
        self.poseidon2.permute_mut(&mut self.state);
        out[32..].copy_from_slice(&digest_to_bytes(self.digest()));
        out
    }

    /// Bytes mining reads first for BE target compare (high half of the hash).
    fn finalize_squeeze_high(&mut self) -> [u8; 32] {
        self.finalize_state();
        digest_to_bytes(self.digest())
    }
}

#[inline(never)]
fn old_squeeze(preimage: &[u8; 96]) -> [u8; 64] {
    hash_squeeze_twice(preimage)
}

#[inline(never)]
fn new_squeeze_full(preimage: &[u8; 96]) -> [u8; 64] {
    let mut st = Sponge::new();
    st.append_bytes(preimage);
    st.finalize_squeeze_twice()
}

#[inline(never)]
fn new_squeeze_high_only(preimage: &[u8; 96]) -> [u8; 32] {
    let mut st = Sponge::new();
    st.append_bytes(preimage);
    st.finalize_squeeze_high()
}

fn mining_preimage(header: &[u8; 32], nonce: &[u8; 64]) -> [u8; 96] {
    let mut input = [0u8; 96];
    input[..32].copy_from_slice(header);
    input[32..].copy_from_slice(nonce);
    input
}

fn decode32(s: &str) -> [u8; 32] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn decode64(s: &str) -> [u8; 64] {
    hex::decode(s).unwrap().try_into().unwrap()
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn p1_09_squeeze_parity_and_timing() {
    // Mining reference path is hash_squeeze_twice (not a separate hash).
    for (i, kv) in NONCE_HASH_KVS.iter().enumerate() {
        let header = decode32(kv.header);
        let nonce = decode64(kv.nonce);
        let preimage = mining_preimage(&header, &nonce);
        let want = decode64(kv.hash);

        let lib = old_squeeze(&preimage);
        assert_eq!(lib, want, "kv {i}: library hash vs fixture");

        let custom = new_squeeze_full(&preimage);
        assert_eq!(custom, lib, "kv {i}: custom full squeeze");

        let high = new_squeeze_high_only(&preimage);
        assert_eq!(
            high,
            lib[..32],
            "kv {i}: high-half fast path must match first squeeze"
        );
    }

    let mut rng = 0x5109_5EED_CAFE_BABEu64;
    for trial in 0..100 {
        let mut preimage = [0u8; 96];
        for chunk in preimage.chunks_mut(8) {
            chunk.copy_from_slice(&splitmix64(&mut rng).to_le_bytes());
        }
        let lib = old_squeeze(&preimage);
        let custom = new_squeeze_full(&preimage);
        assert_eq!(lib, custom, "random preimage trial {trial}");
        assert_eq!(new_squeeze_high_only(&preimage), lib[..32]);
    }

    let bench_preimage = mining_preimage(
        &decode32(NONCE_HASH_KVS[0].header),
        &decode64(NONCE_HASH_KVS[0].nonce),
    );

    const ITERS: u32 = 20_000;
    let mut acc = 0u64;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let h = old_squeeze(black_box(&bench_preimage));
        acc ^= u64::from_le_bytes(h[..8].try_into().unwrap());
    }
    let old_ns = t0.elapsed().as_nanos();

    let t1 = Instant::now();
    for _ in 0..ITERS {
        let h = new_squeeze_full(black_box(&bench_preimage));
        acc ^= u64::from_le_bytes(h[..8].try_into().unwrap());
    }
    let new_ns = t1.elapsed().as_nanos();

    black_box(acc);
    let ratio = old_ns as f64 / new_ns as f64;
    println!("OLD_NS {old_ns} NEW_NS {new_ns} RATIO {ratio:.4}");
}
