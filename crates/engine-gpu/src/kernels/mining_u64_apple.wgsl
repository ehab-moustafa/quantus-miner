// Native-u64 Poseidon2 mining kernel. Apple Metal path only.
// Requires wgpu Features::SHADER_INT64. Bit-exact with mining.wgsl / pow_core.

@group(0) @binding(0) var<storage, read_write> results: array<atomic<u32>>;
// Sponge state after absorbing header + high nonce half (12 felts as LE u32 pairs),
// precomputed on the host per batch. See pow_core::mining_midstate.
// Pack u32 words into vec4s to preserve the byte layout with uniform alignment.
@group(0) @binding(1) var<uniform> midstate: array<vec4<u32>, 6>;
@group(0) @binding(2) var<uniform> start_nonce: array<vec4<u32>, 4>;
@group(0) @binding(3) var<uniform> difficulty_target: array<vec4<u32>, 4>;
@group(0) @binding(4) var<uniform> dispatch_config: vec4<u32>;

const P64: u64 = 0xFFFFFFFF00000001lu;
// EPS64 = 2^32 - 1 = 2^64 mod P
const EPS64: u64 = 0xFFFFFFFFlu;

// Padded with a trailing zero so the internal layer can add the next round's
// constant unconditionally.
const RC_INTERNAL: array<u64, 23> = array<u64, 23>(
    0x97f7798a784ad863lu, 0xd1d2bf082f60d4f0lu, 0x69a377a79f9ad206lu, 0xa9d06906a3858e24lu, 0x295275001eede5b5lu, 0x5874e441117bd746lu, 0x8a084bbba8ed86cclu, 0x3defd7645cde6425lu, 0x3998cfe6871cc137lu, 0x3e52ef8bca48314alu, 0x964a209f85dc9ecclu, 0x3fcc9ee82cc4577elu, 0x8e79b4a5d0096d6dlu, 0x8492362ad2392556lu, 0xee72f470262574d6lu, 0x1e0e18496da2444alu, 0x0f3a74bf215eaac6lu, 0x1b061b76a1c0ded3lu, 0x192c42d86803d7a6lu, 0xf6d49ff997ae0260lu, 0x3ec372e7a0fa3786lu, 0x5538cdf4f23445d3lu, 0lu
);
const RC_INITIAL: array<array<u64, 12>, 4> = array<array<u64, 12>, 4>(
    array<u64, 12>(0xc002e770975b1607lu, 0xbca51a8dfe14593alu, 0x72938dfbe774f7f9lu, 0xe4f2fe29e03234aclu, 0xd5e0ba2f541b6449lu, 0xec33b868f3cc46c1lu, 0x486dcb55419d475alu, 0x6c1cb2a358cc24f1lu, 0xe3f30d509a1436bblu, 0xd9a64f068dca7c29lu, 0xe59b3f57aabba1aelu, 0x2a3dd4505b478fdclu),
    array<u64, 12>(0xada1f8dc7676ed25lu, 0x2711aa8b5509d516lu, 0x4ae6acd0c9c92897lu, 0x56eb3d6b5256d67alu, 0x1f7a9d55923bf51elu, 0x3600427d397a7f68lu, 0xe5076df75b72c3d0lu, 0xfcd59aa12c6090adlu, 0xcd895e8c68b57a9elu, 0x41df7ef9d730ae3elu, 0xee3e2b889abe977dlu, 0xd29bb7edbeb9c405lu),
    array<u64, 12>(0x7d5c08eef608e382lu, 0x89ae889caaf0802clu, 0xb35a8e976d2af617lu, 0xdb14234eafaf5173lu, 0x78f04462d48b1c98lu, 0x265293b0e47ce88alu, 0x999a649b69b9d32flu, 0x64b0a186698e01d3lu, 0xee0b22d0dfae8bb8lu, 0x4fd53e50ca04a7eelu, 0x5762bfe181f25047lu, 0xf51593e2beb5e3bdlu),
    array<u64, 12>(0x1e5e2b5760e32477lu, 0x622462a1f9aaaeedlu, 0xaa284b3ecdb222aelu, 0x63c8e72f542bf3fclu, 0x3ba588cacb43b5e0lu, 0x23eda6f3c99150ddlu, 0xaad3bea4baac9a5alu, 0xe9da8d699b94184alu, 0xcdb13f4cd93e024clu, 0x902cbd0956f655e3lu, 0x5b4e40ffc759532flu, 0xde795c20a2357af7lu)
);
const RC_TERMINAL: array<array<u64, 12>, 4> = array<array<u64, 12>, 4>(
    array<u64, 12>(0x7b72c539e0ea4c6elu, 0x144573dae2ce9976lu, 0x802028b68f35fc88lu, 0x6d36c5022c4fe7c2lu, 0xa205d0ffa9b9def3lu, 0xf6e7e38b1ea6ba2flu, 0x34f7909ae5258d64lu, 0xb0464d9d77b97fcalu, 0x64ddb9d5de7e00a6lu, 0x0ed0d75c27975d97lu, 0x1cbb36f11127338blu, 0x6673e505cfd0b6balu),
    array<u64, 12>(0x605f902830872e01lu, 0x3fd5eb927e95fe4flu, 0xe81025b5a24c69cdlu, 0xf7d0ce75de23f74elu, 0xf39942b6a8585089lu, 0x6d808a08f7b71df6lu, 0xf8806b6588f49a8blu, 0x57df2d8c2a32107alu, 0x16e7c2074d654a2dlu, 0x213de241fcf33835lu, 0xb0f2b8905a0976f6lu, 0xd8e3cf2bbd355417lu),
    array<u64, 12>(0xe498691679d9330flu, 0x763b45d2a3821b28lu, 0x0908bf65eb0a1f0dlu, 0x7691eb2d194b24f4lu, 0x0e43551233ae13b2lu, 0x93c393dbfc2fe76flu, 0x98f607485d48cdealu, 0xe3d95f30309819c0lu, 0x1ef581a93eaf6acflu, 0x0b24c1b7a030fca4lu, 0x624370be5670b327lu, 0x5f1e28615a11e486lu),
    array<u64, 12>(0xfe04051f909e042blu, 0x7257e5b147fd3803lu, 0xe6ae134bb82f2e78lu, 0x5711fd5cf4784511lu, 0xf83a42660c08c0bclu, 0x2cd8c96d9a3ce855lu, 0x7d2ffb1bb0e17271lu, 0x85ae1528caea3811lu, 0x52a345d5c7adb0b8lu, 0x504c4c51f3faee94lu, 0xbce34a649cfccaf9lu, 0xe0a3389266fb6dc9lu)
);
const MDS_DIAG: array<u64, 12> = array<u64, 12>(
    0xc3b6c08e23ba9300lu, 0xd84b5de94a324fb6lu, 0x0d0c371c5b35b84flu, 0x7964f570e7188037lu, 0x5daf18bbd996604blu, 0x6743bc47b9595257lu, 0x5528b9362c59bb70lu, 0xac45e25b7127b68blu, 0xa2077d7dfbb606b5lu, 0xf3faac6faee378aelu, 0x0c6388b51545e883lu, 0xd27dbb6944917b60lu
);

// a + b mod P in lazy form. Wrapping carries fold back via 2^64 ≡ EPS64 (mod P).
fn gf64_add(a: u64, b: u64) -> u64 {
    let s0 = a + b;
    let c1 = s0 < a;
    let s1 = s0 + select(0lu, EPS64, c1);
    let c2 = c1 && (s1 < s0);
    return s1 + select(0lu, EPS64, c2);
}

// Sum accumulator: value = lo + carries * 2^64, folded once via 2^64 ≡ EPS64 (mod P).
struct Acc {
    lo: u64,
    carries: u32,
}

fn acc_add(a: Acc, b: u64) -> Acc {
    let s = a.lo + b;
    return Acc(s, a.carries + select(0u, 1u, s < a.lo));
}

fn acc_add2(a: Acc, b: Acc) -> Acc {
    let s = a.lo + b.lo;
    return Acc(s, a.carries + b.carries + select(0u, 1u, s < a.lo));
}

fn acc_fold(a: Acc) -> u64 {
    // Add carries * (2^32 - 1) in radix 2^32. The low subtraction
    // borrows at most one; delta is nonnegative, even when carries=0.
    let low = u32(a.lo) - a.carries;
    let delta = a.carries - select(0u, 1u, u32(a.lo) < a.carries);
    let high0 = u32(a.lo >> 32u);
    let high = high0 + delta;
    let result = (u64(high) << 32u) | u64(low);
    return result + select(0lu, EPS64, high < high0);
}

struct U128 {
    lo: u64,
    hi: u64,
}

// Reduce a 128-bit value (lo + hi*2^64) mod P using
// 2^64 ≡ EPS64 and 2^96 ≡ -1 (mod P).
fn gf64_reduce(v: U128) -> u64 {
    // With B=2^32, reduce to (w0-w2-w3) + (w1+w2)*B.
    // Track the low limb's two possible borrows in u32 instead of
    // carrying a signed 64-bit intermediate through the reduction.
    let w0 = u32(v.lo);
    let w1 = u32(v.lo >> 32u);
    let w2 = u32(v.hi);
    let w3 = u32(v.hi >> 32u);
    let low0 = w0 - w2;
    let low = low0 - w3;
    let borrow = select(0u, 1u, w0 < w2) + select(0u, 1u, low0 < w3);
    let high0 = w1 + w2;
    let high = high0 - borrow;
    // The mathematical high limb lies in [-1, 2B-2], so carry minus
    // borrow is -1, 0 or 1. A positive correction cannot overflow;
    // a negative correction has high=B-1 and cannot underflow.
    let correction = i64(select(0i, 1i, high0 < w1) - select(0i, 1i, high0 < borrow));
    let bits = bitcast<u64>(correction);
    return ((u64(high) << 32u) | u64(low)) + ((bits << 32u) - bits);
}

fn mul_wide(a: u64, b: u64) -> U128 {
    let a_lo = a & EPS64;
    let a_hi = a >> 32u;
    let b_lo = b & EPS64;
    let b_hi = b >> 32u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    // Accumulate the middle limb in 32 bits; keep both carry bits explicitly.
    let mid0 = u32(ll >> 32u) + u32(lh);
    let c0 = select(0u, 1u, mid0 < u32(lh));
    let mid = mid0 + u32(hl);
    let c = c0 + select(0u, 1u, mid < mid0);
    return U128((u64(mid) << 32u) | u64(u32(ll)), hh + (lh >> 32u) + (hl >> 32u) + u64(c));
}

// (a*b + addend) mod P for b <= 2^64 - 2^32 (all MDS_DIAG entries): the addend's
// value and carries are folded into the 128-bit product before reduction.
fn gf64_mul_add(a: u64, b: u64, addend: Acc) -> u64 {
    let v = mul_wide(a, b);
    let lo = v.lo + addend.lo;
    return gf64_reduce(U128(lo, v.hi + u64(addend.carries) + select(0lu, 1lu, lo < v.lo)));
}

fn gf64_mul(a: u64, b: u64) -> u64 {
    return gf64_reduce(mul_wide(a, b));
}

fn gf64_sqr(a: u64) -> u64 {
    let a_lo = a & EPS64;
    let a_hi = a >> 32u;
    let ll = a_lo * a_lo;
    let lh = a_lo * a_hi;
    let hh = a_hi * a_hi;
    // Accumulate the middle limb in 32 bits; keep both carry bits explicitly.
    let mid0 = u32(ll >> 32u) + u32(lh);
    let c0 = select(0u, 1u, mid0 < u32(lh));
    let mid = mid0 + u32(lh);
    let c = c0 + select(0u, 1u, mid < mid0);
    return gf64_reduce(U128((u64(mid) << 32u) | u64(u32(ll)), hh + ((lh >> 32u) << 1u) + u64(c)));
}

fn gf64_sbox(x: u64) -> u64 {
    let x2 = gf64_sqr(x);
    let x4 = gf64_sqr(x2);
    let x6 = gf64_mul(x4, x2);
    return gf64_mul(x6, x);
}

fn gf64_canon(a: u64) -> u64 {
    return a - select(0lu, P64, a >= P64);
}

// 4x4 MDS circ(2, 3, 1, 1) on one chunk, results left unreduced.
fn mds4(x0: u64, x1: u64, x2: u64, x3: u64) -> array<Acc, 4> {
    let t01 = acc_add(Acc(x0, 0u), x1);
    let t23 = acc_add(Acc(x2, 0u), x3);
    let t0123 = acc_add2(t01, t23);
    let t01123 = acc_add(t0123, x1);
    let t01233 = acc_add(t0123, x3);
    return array<Acc, 4>(
        acc_add2(t01123, t01),
        acc_add2(t01123, Acc(x2 << 1u, u32(x2 >> 63u))),
        acc_add2(t01233, t23),
        acc_add2(t01233, Acc(x0 << 1u, u32(x0 >> 63u)))
    );
}

// External linear layer: 4x4 MDS on each chunk, then circulant sums, plus the
// next round's constants. Additions are accumulated unreduced (at most 27
// carries) and folded once per output.
fn ext_layer64(state: ptr<function, array<u64, 12>>, rc_index: u32) {
    var y: array<Acc, 12>;
    for (var chunk = 0u; chunk < 3u; chunk++) {
        let o = chunk * 4u;
        let m = mds4((*state)[o], (*state)[o + 1u], (*state)[o + 2u], (*state)[o + 3u]);
        y[o] = m[0];
        y[o + 1u] = m[1];
        y[o + 2u] = m[2];
        y[o + 3u] = m[3];
    }
    for (var k = 0u; k < 4u; k++) {
        let s = acc_add2(acc_add2(y[k], y[k + 4u]), y[k + 8u]);
        (*state)[k] = acc_fold(acc_add(acc_add2(y[k], s), RC_EXT[rc_index][k]));
        (*state)[k + 4u] = acc_fold(acc_add(acc_add2(y[k + 4u], s), RC_EXT[rc_index][k + 4u]));
        (*state)[k + 8u] = acc_fold(acc_add(acc_add2(y[k + 8u], s), RC_EXT[rc_index][k + 8u]));
    }
}

// Internal linear layer: diagonal matrix plus full sum, plus the next round's
// constant on element 0 (the S-box output, which is summed last).
fn int_layer64(state: ptr<function, array<u64, 12>>, rc0: u64) {
    let s12 = acc_add(Acc((*state)[1], 0u), (*state)[2]);
    let s34 = acc_add(Acc((*state)[3], 0u), (*state)[4]);
    let s56 = acc_add(Acc((*state)[5], 0u), (*state)[6]);
    let s78 = acc_add(Acc((*state)[7], 0u), (*state)[8]);
    let s910 = acc_add(Acc((*state)[9], 0u), (*state)[10]);
    let s1234 = acc_add2(s12, s34);
    let s5678 = acc_add2(s56, s78);
    let s91011 = acc_add(s910, (*state)[11]);
    let sum = acc_add(acc_add2(acc_add2(s1234, s5678), s91011), (*state)[0]);
    (*state)[0] = gf64_mul_add((*state)[0], MDS_DIAG[0], acc_add(sum, rc0));
    (*state)[1] = gf64_mul_add((*state)[1], MDS_DIAG[1], sum);
    (*state)[2] = gf64_mul_add((*state)[2], MDS_DIAG[2], sum);
    (*state)[3] = gf64_mul_add((*state)[3], MDS_DIAG[3], sum);
    (*state)[4] = gf64_mul_add((*state)[4], MDS_DIAG[4], sum);
    (*state)[5] = gf64_mul_add((*state)[5], MDS_DIAG[5], sum);
    (*state)[6] = gf64_mul_add((*state)[6], MDS_DIAG[6], sum);
    (*state)[7] = gf64_mul_add((*state)[7], MDS_DIAG[7], sum);
    (*state)[8] = gf64_mul_add((*state)[8], MDS_DIAG[8], sum);
    (*state)[9] = gf64_mul_add((*state)[9], MDS_DIAG[9], sum);
    (*state)[10] = gf64_mul_add((*state)[10], MDS_DIAG[10], sum);
    (*state)[11] = gf64_mul_add((*state)[11], MDS_DIAG[11], sum);
}

fn sbox_lanes(state: ptr<function, array<u64, 12>>, lanes: u32) {
    for (var i = 0u; i < lanes; i++) {
        (*state)[i] = gf64_sbox((*state)[i]);
    }
}

fn add_rc(state: ptr<function, array<u64, 12>>, rc: array<u64, 12>) {
    for (var i = 0u; i < 12u; i++) {
        (*state)[i] = gf64_add((*state)[i], rc[i]);
    }
}

const RC_ZERO: array<u64, 12> = array<u64, 12>(0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu);
// Constants added by each external layer, which feeds the S-box that follows it:
// entries 0..3 precede the initial rounds, entry 4 carries the first internal
// round's constant on element 0, entries 5..7 precede terminal rounds 1..3.
const RC_EXT: array<array<u64, 12>, 9> = array<array<u64, 12>, 9>(
    RC_INITIAL[0], RC_INITIAL[1], RC_INITIAL[2], RC_INITIAL[3],
    array<u64, 12>(RC_INTERNAL[0], 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu, 0lu),
    RC_TERMINAL[1], RC_TERMINAL[2], RC_TERMINAL[3], RC_ZERO
);

// One loop drives all 30 rounds so each layer is emitted once: step 0 is the
// initial external layer, steps 1..4 and 27..30 are external rounds, steps
// 5..26 are internal rounds. Round constants are added by the linear layer
// preceding each S-box (the last internal layer is followed by the first
// terminal round's constants).
fn permute64(state: ptr<function, array<u64, 12>>) {
    for (var k = 0u; k < 31u; k++) {
        let is_ext = k < 5u || k > 26u;
        if (k > 0u) {
            sbox_lanes(state, select(1u, 12u, is_ext));
        }
        if (is_ext) {
            ext_layer64(state, select(k, k - 22u, k > 26u));
        } else {
            int_layer64(state, RC_INTERNAL[k - 4u]);
            if (k == 26u) {
                add_rc(state, RC_TERMINAL[0]);
            }
        }
    }
}

fn bswap32(v: u32) -> u32 {
    return ((v & 0xFFu) << 24u) | ((v & 0xFF00u) << 8u) | ((v >> 8u) & 0xFF00u) | (v >> 24u);
}

@compute @workgroup_size(256)
fn mining_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let total_threads = dispatch_config[0];
    let nonces_per_thread = dispatch_config[1];
    let total_nonces = dispatch_config[2];
    if (thread_id >= total_threads) {
        return;
    }
    let base_index = thread_id * nonces_per_thread;

    // Hoist uniform storage reads out of the nonce loop
    var mid: array<u64, 12>;
    for (var i = 0u; i < 12u; i++) {
        mid[i] = (u64(midstate[(2u * i + 1u) / 4u][(2u * i + 1u) % 4u]) << 32u) | u64(midstate[(2u * i) / 4u][(2u * i) % 4u]);
    }
    var tgt: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        tgt[i] = difficulty_target[(i) / 4u][(i) % 4u];
    }
    var nonce_base: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        nonce_base[i] = start_nonce[(i) / 4u][(i) % 4u];
    }

    for (var j = 0u; j < nonces_per_thread; j = j + 1u) {
        let logical_index = base_index + j;
        if (logical_index >= total_nonces) {
            break;
        }
        if (j > 0u && dispatch_config[3] == 0u && atomicLoad(&results[0]) != 0u) {
            return;
        }

        // The host guarantees a batch never carries into the high nonce half
        // (limbs 8..15), so only the low 256 bits are incremented here.
        var current_nonce: array<u32, 16>;
        let val0 = nonce_base[0];
        let sum0 = val0 + logical_index;
        current_nonce[0] = sum0;
        var carry = select(0u, 1u, sum0 < val0);
        for (var i = 1u; i < 8u; i++) {
            let val = nonce_base[i];
            let sum = val + carry;
            current_nonce[i] = sum;
            carry = select(0u, 1u, sum < val);
        }
        for (var i = 8u; i < 16u; i++) {
            current_nonce[i] = nonce_base[i];
        }

        // Resume the sponge from the precomputed midstate: absorb the low
        // nonce half, pad, squeeze twice (3 permutations instead of 5).
        var st = mid;
        for (var i = 0u; i < 8u; i++) {
            st[i] = gf64_add(st[i], u64(bswap32(current_nonce[7u - i])));
        }
        // The overwhelmingly common rejection needs only the first hash word.
        // Delay materializing the full result until that comparison passes.
        for (var phase = 0u; phase < 2u; phase++) {
            permute64(&st);
            if (phase == 0u) {
                st[0] = gf64_add(st[0], 1lu);
                st[1] = gf64_add(st[1], 1lu);
            }
        }
        // Host uploads compare-ready target (u32 byte-swapped per word); match without bswap here.
        let first = u32(gf64_canon(st[0]) & EPS64);
        if (first > tgt[15]) {
            continue;
        }
        var hash_le: array<u32, 16>;
        var cmp = 0u;
        for (var i = 0u; i < 4u; i++) {
            let c = gf64_canon(st[i]);
            hash_le[15u - 2u * i] = bswap32(u32(c & EPS64));
            hash_le[14u - 2u * i] = bswap32(u32(c >> 32u));
        }
        for (var i = 0u; i < 8u; i++) {
            let h = hash_le[15u - i];
            let t = tgt[15u - i];
            if (h != t) {
                cmp = select(2u, 1u, h > t);
                break;
            }
        }
        if (cmp == 1u) {
            continue;
        }
        permute64(&st);
        for (var i = 0u; i < 4u; i++) {
            let c = gf64_canon(st[i]);
            hash_le[7u - 2u * i] = bswap32(u32(c & EPS64));
            hash_le[6u - 2u * i] = bswap32(u32(c >> 32u));
        }
        var below = cmp == 2u;
        if (!below) {
            for (var i = 0u; i < 8u; i++) {
                let h = hash_le[7u - i];
                let t = tgt[7u - i];
                if (h != t) {
                    below = h < t;
                    break;
                }
            }
        }

        if (below) {
            if (atomicExchange(&results[0], 1u) == 0u) {
                for (var i = 0u; i < 16u; i++) {
                    atomicStore(&results[1u + i], current_nonce[i]);
                    atomicStore(&results[17u + i], hash_le[i]);
                }
            }
            return;
        }
    }
}
