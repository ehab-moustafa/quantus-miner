//! Phase 1 agent 12: `bytes_to_felts` / 32-byte header absorb packing vs qp-poseidon-core 3.1.0
//! `serialization.rs` (4-byte limbs + terminator; mining uses 8× u32 LE without terminator).

use qp_poseidon_core::serialization::{bytes_to_felts, bytes_to_u64s};
use qp_poseidon_core::Goldilocks;
use std::hint::black_box;
use std::time::Instant;

const BYTES_PER_FELT: usize = 4;

// --- Copied from qp-poseidon-core 3.1.0 `serialization.rs` (baseline) ---

struct BytesToU64sIterOld<'a> {
    input: &'a [u8],
    pos: usize,
    emitted_terminator: bool,
}

impl Iterator for BytesToU64sIterOld<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + BYTES_PER_FELT <= self.input.len() {
            let chunk = &self.input[self.pos..self.pos + BYTES_PER_FELT];
            self.pos += BYTES_PER_FELT;
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            return Some(u32::from_le_bytes(bytes) as u64);
        }

        if self.emitted_terminator {
            return None;
        }

        self.emitted_terminator = true;
        let remainder = &self.input[self.pos..];
        let mut last = [0u8; BYTES_PER_FELT];
        last[..remainder.len()].copy_from_slice(remainder);
        last[remainder.len()] = 1u8;
        Some(u32::from_le_bytes(last) as u64)
    }
}

#[inline(never)]
fn bytes_to_u64s_old(input: &[u8]) -> Vec<u64> {
    BytesToU64sIterOld {
        input,
        pos: 0,
        emitted_terminator: false,
    }
    .collect()
}

#[inline(never)]
fn bytes_to_felts_old(input: &[u8]) -> Vec<Goldilocks> {
    bytes_to_u64s_old(input)
        .into_iter()
        .map(Goldilocks::from_u64)
        .collect()
}

// --- Mining header absorb: 8 little-endian u32 limbs, no terminator ---

#[inline(never)]
fn header_absorb_u64s_old(header: &[u8; 32]) -> [u64; 8] {
    let mut out = [0u64; 8];
    for (i, chunk) in header.chunks_exact(BYTES_PER_FELT).enumerate() {
        out[i] = u32::from_le_bytes(chunk.try_into().unwrap()) as u64;
    }
    out
}

#[inline(always)]
fn read_u64_le_unaligned(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() >= 8);
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(buf)
}

#[inline(never)]
fn bytes_to_u64s_fast(input: &[u8]) -> Vec<u64> {
    let cap = (input.len() + 1).div_ceil(BYTES_PER_FELT);
    let mut out = Vec::with_capacity(cap);
    let len = input.len();
    let mut pos = 0;

    while pos + 8 <= len {
        let v = read_u64_le_unaligned(&input[pos..]);
        out.push(v as u32 as u64);
        out.push(v >> 32);
        pos += 8;
    }

    if pos + BYTES_PER_FELT <= len {
        let chunk = &input[pos..pos + BYTES_PER_FELT];
        pos += BYTES_PER_FELT;
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64);
    }

    if pos < len {
        let remainder = &input[pos..];
        let mut last = [0u8; BYTES_PER_FELT];
        last[..remainder.len()].copy_from_slice(remainder);
        last[remainder.len()] = 1;
        out.push(u32::from_le_bytes(last) as u64);
    } else {
        out.push(1);
    }

    out
}

#[inline(never)]
fn bytes_to_u64s_32_fast(input: &[u8; 32]) -> [u64; 9] {
    let mut out = [0u64; 9];
    for i in 0..4 {
        let v = read_u64_le_unaligned(&input[i * 8..]);
        out[i * 2] = v as u32 as u64;
        out[i * 2 + 1] = v >> 32;
    }
    out[8] = 1;
    out
}

#[inline(never)]
fn bytes_to_u64s_64_fast(input: &[u8; 64]) -> [u64; 17] {
    let mut out = [0u64; 17];
    for i in 0..8 {
        let v = read_u64_le_unaligned(&input[i * 8..]);
        out[i * 2] = v as u32 as u64;
        out[i * 2 + 1] = v >> 32;
    }
    out[16] = 1;
    out
}

#[inline(never)]
fn header_absorb_u64s_fast(header: &[u8; 32]) -> [u64; 8] {
    let mut out = [0u64; 8];
    for i in 0..4 {
        let v = read_u64_le_unaligned(&header[i * 8..]);
        out[i * 2] = v as u32 as u64;
        out[i * 2 + 1] = v >> 32;
    }
    out
}

fn assert_u64s_match_lib(input: &[u8]) {
    let lib = bytes_to_u64s(input);
    let old = bytes_to_u64s_old(input);
    let fast = bytes_to_u64s_fast(input);
    assert_eq!(lib, old, "old iter diverged from library (len {})", input.len());
    assert_eq!(lib, fast, "fast diverged from library (len {})", input.len());

    let lib_f = bytes_to_felts(input);
    let old_f = bytes_to_felts_old(input);
    assert_eq!(lib_f, old_f);
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_bytes32(rng: &mut u64) -> [u8; 32] {
    let mut b = [0u8; 32];
    for w in b.chunks_exact_mut(8) {
        w.copy_from_slice(&splitmix64(rng).to_le_bytes());
    }
    b
}

fn random_bytes64(rng: &mut u64) -> [u8; 64] {
    let mut b = [0u8; 64];
    for w in b.chunks_exact_mut(8) {
        w.copy_from_slice(&splitmix64(rng).to_le_bytes());
    }
    b
}

#[test]
fn p1_12_bytes_parity_and_bench() {
    for len in [0usize, 1, 3, 4, 5, 7, 8, 31, 33, 63, 65] {
        let input = vec![0xA5u8; len];
        assert_u64s_match_lib(&input);
    }

    let mut rng = 0xC0FF_EE00_1234_5678u64;
    for _ in 0..10_000 {
        let b32 = random_bytes32(&mut rng);
        assert_u64s_match_lib(&b32);
        assert_eq!(
            bytes_to_u64s_32_fast(&b32).as_slice(),
            bytes_to_u64s(&b32)
        );

        let header_old = header_absorb_u64s_old(&b32);
        let header_fast = header_absorb_u64s_fast(&b32);
        assert_eq!(header_old, header_fast);

        let b64 = random_bytes64(&mut rng);
        assert_u64s_match_lib(&b64);
        assert_eq!(
            bytes_to_u64s_64_fast(&b64).as_slice(),
            bytes_to_u64s(&b64)
        );
    }

    const ITERS: u32 = 200_000;
    let mut header = [0x2Au8; 32];
    let mut acc_old = 0u64;
    let t0 = Instant::now();
    for i in 0..ITERS {
        header[0] = i as u8;
        header[1] = (i >> 8) as u8;
        acc_old ^= black_box(bytes_to_u64s_old(black_box(&header))[8]);
    }
    let old_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    let mut acc_new = 0u64;
    let t1 = Instant::now();
    for i in 0..ITERS {
        header[0] = i as u8;
        header[1] = (i >> 8) as u8;
        acc_new ^= black_box(bytes_to_u64s_32_fast(black_box(&header))[8]);
    }
    let new_ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;

    assert_eq!(acc_old, acc_new);
    let ratio = old_ns / new_ns;
    println!("OLD_NS {old_ns:.3} NEW_NS {new_ns:.3} RATIO {ratio:.4}");
    black_box(acc_new);
}
