typedef int i32;
typedef unsigned int u32;
typedef unsigned long long u64;

const u64 P64 = 0xFFFFFFFF00000001ULL;
const u64 EPS64 = 0xFFFFFFFFULL;
const u32 EPS32 = 0xFFFFFFFFu;
// u <- u + (p - 35), v <- v + 35 on the nonce line (exact gf64_add, not lazy).
const u64 P64_MINUS_35 = 0xFFFFFFFEFFFFFFDCULL;
const u64 GF64_35 = 35ULL;

// Sum in [0, 2^64) with a single 2^64 -> EPS fold. The fold itself can only
// wrap when both inputs are within 2^32 of 2^64, which reducer outputs never are
// in practice; that case is left uncorrected (see reduce128).
__device__ __forceinline__ u64 gf64_add(u64 a, u64 b) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
    u32 o0, o1;
    asm("{\n\t"
        ".reg .u32 c;\n\t"
        "add.cc.u32 %0, %2, %4;\n\t"
        "addc.cc.u32 %1, %3, %5;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.cc.u32 %0, c, %6, %0;\n\t"
        "madc.hi.u32 %1, c, %6, %1;\n\t"
        "}"
        : "=&r"(o0), "=&r"(o1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(EPS32));
    return ((u64)o1 << 32) | (u64)o0;
}

// Quantus-Network/quantus-miner#110 item 3: omit madc.hi on the EPS fold (~1 in 2^32
// wrong mod p on nonce-step adds; host CPU verify rejects bad hits). NVRTC:
// MINER_CUDA_LAZY_CARRY=1.
__device__ __forceinline__ u64 gf64_add_lazy(u64 a, u64 b) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
    u32 o0, o1;
    asm("{\n\t"
        ".reg .u32 c;\n\t"
        "add.cc.u32 %0, %2, %4;\n\t"
        "addc.cc.u32 %1, %3, %5;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.cc.u32 %0, c, %6, %0;\n\t"
        "}"
        : "=&r"(o0), "=&r"(o1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(EPS32));
    return ((u64)o1 << 32) | (u64)o0;
}

#if defined(MINER_CUDA_LAZY_CARRY) && MINER_CUDA_LAZY_CARRY
#define GF64_ADD_NONCE_STEP gf64_add_lazy
#else
#define GF64_ADD_NONCE_STEP gf64_add
#endif

// Pre-S-box nonce-line inputs: u(t) = pk0 + (p - 35*t), v(t) = pk1 + 35*t.
// Stepping uses gf64_add, not the lazy nonce add: a long chain of lazy folds
// drifts from the single 35*t add. Host model: pow-core tests/p3_p4_field_ext.rs.
__device__ __forceinline__ void gf64_nonce_line_uv_at(u64 pk0, u64 pk1, u32 t, u64 &u, u64 &v) {
    const u64 d = 35ULL * (u64)t;
    u = gf64_add(pk0, P64 - d);
    v = gf64_add(pk1, d);
}

__device__ __forceinline__ void gf64_nonce_line_uv_step(u64 &u, u64 &v) {
    u = gf64_add(u, P64_MINUS_35);
    v = gf64_add(v, GF64_35);
}

__device__ __forceinline__ void mul64wide(u64 a, u64 b, u32 &r0, u32 &r1,
                                          u32 &r2, u32 &r3) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    r0 = (u32)lo;
    r1 = (u32)(lo >> 32);
    r2 = (u32)hi;
    r3 = (u32)(hi >> 32);
}

// 128 -> 64 bit fold using 2^64 = EPS and 2^96 = -1 (mod p):
//   (r1:r0) + r2*EPS with carry c, then + c*2^32 - (r3 + c).
// The final borrow and the two 32-bit wraps are deliberately not corrected;
// each occurs with probability about 2^-33 per multiply on reducer-distributed
// inputs and shifts the result by +-EPS (mod p). A hash built on a slipped
// value is simply wrong for that nonce, and the host re-verifies every
// candidate on the CPU. This is the same trade as a lazy Montgomery reduction:
// bit-exact results are not required for mining, only for verification.
__device__ __forceinline__ u64 reduce128(u32 r0, u32 r1, u32 r2, u32 r3) {
    u32 o0, o1;
    asm("{\n\t"
        ".reg .u32 c;\n\t"
        "mad.lo.cc.u32 %0, %4, %6, %2;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %3;\n\t"
        "addc.u32 c, %5, 0;\n\t"
        "addc.u32 %1, %1, 0;\n\t"
        "sub.cc.u32 %0, %0, c;\n\t"
        "subc.u32 %1, %1, 0;\n\t"
        "}"
        : "=&r"(o0), "=&r"(o1)
        : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(EPS32));
    return ((u64)o1 << 32) | (u64)o0;
}

// CPPminer CP_QPOW_WRED_FAST-style fold: omit the high-lane carry bump after EPS
// madc (filter path only; host CPU re-verifies hits). NVRTC: MINER_CUDA_OUT0_LAZY_W=1.
__device__ __forceinline__ u64 reduce128_wred_fast(u32 r0, u32 r1, u32 r2, u32 r3) {
    u32 o0, o1;
    asm("{\n\t"
        ".reg .u32 c;\n\t"
        "mad.lo.cc.u32 %0, %4, %6, %2;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %3;\n\t"
        "addc.u32 c, %5, 0;\n\t"
        "sub.cc.u32 %0, %0, c;\n\t"
        "subc.u32 %1, %1, 0;\n\t"
        "}"
        : "=&r"(o0), "=&r"(o1)
        : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(EPS32));
    return ((u64)o1 << 32) | (u64)o0;
}

#if defined(MINER_CUDA_OUT0_LAZY_W) && MINER_CUDA_OUT0_LAZY_W
#define CPP_OUT0_REDUCE128 reduce128_wred_fast
#define CPP_OUT0_GF64_ADD gf64_add_lazy
#else
#define CPP_OUT0_REDUCE128 reduce128
#define CPP_OUT0_GF64_ADD gf64_add
#endif

__device__ __forceinline__ u64 gf64_mul(u64 a, u64 b) {
    u32 r0, r1, r2, r3;
    mul64wide(a, b, r0, r1, r2, r3);
    return reduce128(r0, r1, r2, r3);
}

// Three partial products instead of four: (a1:a0)^2 = a0^2 + 2*a0*a1*2^32 + a1^2*2^64.
__device__ __forceinline__ u64 gf64_sqr(u64 a) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32);
    u64 ll = (u64)a0 * a0;
    u64 lh = (u64)a0 * a1;
    u64 hh = (u64)a1 * a1;
    u64 mid = lh << 1;
    u32 mid_top = (u32)(lh >> 63);
    u32 r0 = (u32)ll, r1, r2, r3;
    asm("{\n\t"
        "add.cc.u32 %0, %3, %4;\n\t"
        "addc.cc.u32 %1, %5, %6;\n\t"
        "addc.u32 %2, %7, %8;\n\t"
        "}"
        : "=&r"(r1), "=&r"(r2), "=&r"(r3)
        : "r"((u32)(ll >> 32)), "r"((u32)mid), "r"((u32)hh),
          "r"((u32)(mid >> 32)), "r"((u32)(hh >> 32)), "r"(mid_top));
    return reduce128(r0, r1, r2, r3);
}

// x^7 with a depth-3 chain: x3 and x4 come from x2 in parallel.
__device__ __forceinline__ u64 gf64_sbox(u64 x) {
    u64 x2 = gf64_sqr(x);
    u64 x3 = gf64_mul(x2, x);
    u64 x4 = gf64_sqr(x2);
    return gf64_mul(x4, x3);
}

// Round-1 f=S(u), g=S(v): interleave squares/muls so u and v chains overlap (NCU).
__device__ __forceinline__ void gf64_sbox_pair(u64 u, u64 v, u64 &fu, u64 &gv) {
    u64 u2 = gf64_sqr(u);
    u64 v2 = gf64_sqr(v);
    u64 u3 = gf64_mul(u2, u);
    u64 v3 = gf64_mul(v2, v);
    u64 u4 = gf64_sqr(u2);
    u64 v4 = gf64_sqr(v2);
    fu = gf64_mul(u4, u3);
    gv = gf64_mul(v4, v3);
}

__device__ __forceinline__ u64 gf64_canon(u64 a) {
    return a - ((a >= P64) ? P64 : 0ULL);
}

struct Wide {
    u32 l0;
    u32 l1;
    u32 h;
};

__device__ __forceinline__ Wide wide_from(u64 x) {
    Wide w;
    w.l0 = (u32)x;
    w.l1 = (u32)(x >> 32);
    w.h = 0;
    return w;
}

__device__ __forceinline__ void wide_add(Wide &w, u64 x) {
    u32 x0 = (u32)x, x1 = (u32)(x >> 32);
    asm("{\n\t"
        "add.cc.u32 %0, %0, %3;\n\t"
        "addc.cc.u32 %1, %1, %4;\n\t"
        "addc.u32 %2, %2, 0;\n\t"
        "}"
        : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
        : "r"(x0), "r"(x1));
}

__device__ __forceinline__ void wide_add_wide(Wide &w, const Wide &x) {
    asm("{\n\t"
        "add.cc.u32 %0, %0, %3;\n\t"
        "addc.cc.u32 %1, %1, %4;\n\t"
        "addc.u32 %2, %2, %5;\n\t"
        "}"
        : "+r"(w.l0), "+r"(w.l1), "+r"(w.h)
        : "r"(x.l0), "r"(x.l1), "r"(x.h));
}

// A 96-bit lane is a 128-bit value with a zero top word.
__device__ __forceinline__ u64 wide_reduce(const Wide &w) {
    return reduce128(w.l0, w.l1, w.h, 0u);
}

__device__ __forceinline__ void add128_wide(u32 &r0, u32 &r1, u32 &r2,
                                            u32 &r3, const Wide &w) {
    asm("{\n\t"
        "add.cc.u32 %0, %0, %4;\n\t"
        "addc.cc.u32 %1, %1, %5;\n\t"
        "addc.cc.u32 %2, %2, %6;\n\t"
        "addc.u32 %3, %3, 0;\n\t"
        "}"
        : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3)
        : "r"(w.l0), "r"(w.l1), "r"(w.h));
}

// a * b + w as a 128-bit value. The three words of the 96-bit addend ride in
// the 64-bit accumulators of the partial products, so the add costs nothing.
// Requires b < 2^64 - 2^59 (true for MDS_DIAG) and w.h small, so every partial
// sum and the total fit.
__device__ __forceinline__ void mul128_add_wide(u64 a, u64 b, const Wide &w,
                                                u32 &r0, u32 &r1, u32 &r2,
                                                u32 &r3) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
    u64 w0 = w.l0, w1 = w.l1, w2 = w.h;
    asm("{\n\t"
        ".reg .b64 p0, m, m2, p3;\n\t"
        ".reg .b32 m0, m1, p0h, p3l, p3h, cw;\n\t"
        "mad.wide.u32 p0, %4, %6, %8;\n\t"
        "mad.wide.u32 m, %5, %6, %9;\n\t"
        "mul.wide.u32 m2, %4, %7;\n\t"
        "mad.wide.u32 p3, %5, %7, %10;\n\t"
        "mov.b64 {%0, p0h}, p0;\n\t"
        "mov.b64 {p3l, p3h}, p3;\n\t"
        "add.cc.u64 m, m, m2;\n\t"
        "addc.u32 cw, p3h, 0;\n\t"
        "mov.b64 {m0, m1}, m;\n\t"
        "add.cc.u32 %1, p0h, m0;\n\t"
        "addc.cc.u32 %2, p3l, m1;\n\t"
        "addc.u32 %3, cw, 0;\n\t"
        "}"
        : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "l"(w0), "l"(w1), "l"(w2));
}
