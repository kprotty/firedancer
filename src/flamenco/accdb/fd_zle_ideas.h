// zle.h -- zero-run extraction over AVX-512 byte masks.
//
// Two primitives, both operating on 256-byte blocks turned into a 256-bit
// "this byte is zero" bitmap by four vptestnmb:
//
//   first_zero_run(p)  -- the one-shot probe: first zero byte + its run length
//   encode_dense/sparse -- the streaming RLE: the whole (L,Z) length sequence
//
// Requires: AVX512F/BW, BMI1/2, POPCNT. encode_dense also needs AVX512_VBMI2.
// Build GCC with -fno-tree-slp-vectorize (see Makefile / README).
#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

namespace zle {

typedef unsigned long long ull_;

// 256 bytes -> 4x64-bit bitmap, bit i set <=> byte i == 0.
static inline void zero_masks(const uint8_t* p, uint64_t m[4]) {
    const __m512i* v = (const __m512i*)p;
    m[0] = _mm512_testn_epi8_mask(_mm512_loadu_si512(v+0), _mm512_loadu_si512(v+0));
    m[1] = _mm512_testn_epi8_mask(_mm512_loadu_si512(v+1), _mm512_loadu_si512(v+1));
    m[2] = _mm512_testn_epi8_mask(_mm512_loadu_si512(v+2), _mm512_loadu_si512(v+2));
    m[3] = _mm512_testn_epi8_mask(_mm512_loadu_si512(v+3), _mm512_loadu_si512(v+3));
}

// ---------------------------------------------------------------------------
// first_zero_run: lowest set bit of a 256-bit bitmap, twice.
// ---------------------------------------------------------------------------
// One `test` feeds both cmovs. Written as asm because GCC (all versions through
// 16.2) flattens the equivalent C ternary tree into a linear branch chain, which
// mispredicts; see README. clang gets it right from plain C, but this costs it
// nothing and keeps one source for both.
static inline void sel_(uint64_t& v, unsigned& o, uint64_t b, unsigned ob) {
    __asm__("testq %[v],%[v]\n\t"
            "cmovzq %[b],%[v]\n\t"
            "cmovzl %[ob],%[o]"
            : [v]"+r"(v), [o]"+r"(o) : [b]"r"(b), [ob]"r"(ob) : "cc");
}

// Index of the lowest set bit, or 256 if the bitmap is empty.
// The empty case needs no guard: it falls out as 192 + tzcnt(0) == 192 + 64.
static inline unsigned tz256(const uint64_t x[4]) {
    uint64_t a = x[0]; unsigned oa = 0;   sel_(a, oa, x[1], 64);
    uint64_t b = x[2]; unsigned ob = 128; sel_(b, ob, x[3], 192);
    sel_(a, oa, b, ob);
    return oa + (unsigned)_tzcnt_u64(a);
}

struct Run {
    unsigned start;   // 256 => no zero byte in this block
    unsigned len;     // start+len == 256 => the run continues into the next block
};

static inline Run first_zero_run(const uint8_t* p) {
    uint64_t x[4], y[4];
    zero_masks(p, x);
    unsigned s = tz256(x);

    // y = x - 1. The borrow crossing word boundaries is exactly the
    // "skip the leading empty words" logic -- it cannot be done per-word.
    unsigned char k;
    k = _subborrow_u64(0, x[0], 1, (ull_*)&y[0]);
    k = _subborrow_u64(k, x[1], 0, (ull_*)&y[1]);
    k = _subborrow_u64(k, x[2], 0, (ull_*)&y[2]);
    k = _subborrow_u64(k, x[3], 0, (ull_*)&y[3]);

    // y = ~(x | (x-1)). Bits [0,end) are clear and bit `end` is set, so the
    // lowest set bit IS end -- no +1 carry chain needed. Both saturating cases
    // (no zero byte / run reaches byte 255) give y == 0, hence tz256 == 256.
    y[0] = ~(x[0]|y[0]); y[1] = ~(x[1]|y[1]);
    y[2] = ~(x[2]|y[2]); y[3] = ~(x[3]|y[3]);

    Run r; r.start = s; r.len = tz256(y) - s;
    return r;
}

// ---------------------------------------------------------------------------
// Streaming RLE.
// ---------------------------------------------------------------------------
// edges = m ^ ((m<<1) | carry): every set bit marks a run boundary, so the run
// lengths are the gaps between consecutive set bits. One bitmap, one scan --
// rather than alternating two masked searches per run.
static inline void edges4(const uint64_t m[4], uint64_t e[4], uint64_t& carry) {
    for (int w = 0; w < 4; ++w) { e[w] = m[w] ^ ((m[w] << 1) | carry); carry = m[w] >> 63; }
}

alignas(64) static const uint8_t IDXB_[256] = {
#define ZLE_R16_(a) a+0,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9,a+10,a+11,a+12,a+13,a+14,a+15
    ZLE_R16_(0),  ZLE_R16_(16), ZLE_R16_(32), ZLE_R16_(48),
    ZLE_R16_(64), ZLE_R16_(80), ZLE_R16_(96), ZLE_R16_(112),
    ZLE_R16_(128),ZLE_R16_(144),ZLE_R16_(160),ZLE_R16_(176),
    ZLE_R16_(192),ZLE_R16_(208),ZLE_R16_(224),ZLE_R16_(240)
#undef ZLE_R16_
};

// Both encoders write the same thing: alternating run lengths starting with a
// literal run (a leading zero-length literal if byte 0 is already zero).
// `out` needs nblocks*256 + 64 slots of slack (encode_dense overwrites past the
// end of each block's output before committing k). Lengths are uint16, so a
// single run must stay under 65536 bytes -- swap in a varint sink if not.

// Dense: >~1 run per 256B block. Fixed cost per block, ~free per run.
static inline size_t encode_dense(const uint8_t* p, size_t nblocks, uint16_t* out) {
    uint64_t carry = 0, start = 0, base = 0; size_t k = 0;
    alignas(64) uint8_t pos[320];
    const __m512i I0 = _mm512_load_si512((const __m512i*)(IDXB_+0));
    const __m512i I1 = _mm512_load_si512((const __m512i*)(IDXB_+64));
    const __m512i I2 = _mm512_load_si512((const __m512i*)(IDXB_+128));
    const __m512i I3 = _mm512_load_si512((const __m512i*)(IDXB_+192));
    for (size_t bi = 0; bi < nblocks; ++bi, p += 256, base += 256) {
        uint64_t m[4], e[4]; zero_masks(p, m); edges4(m, e, carry);
        uint8_t* o = pos;                        // all boundary offsets, 64 at a time
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[0],I0); o += _mm_popcnt_u64(e[0]);
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[1],I1); o += _mm_popcnt_u64(e[1]);
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[2],I2); o += _mm_popcnt_u64(e[2]);
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[3],I3); o += _mm_popcnt_u64(e[3]);
        size_t n = o - pos;
        if (!n) continue;
        out[k++] = (uint16_t)(base + pos[0] - start);   // only this one can exceed 255
        for (size_t i = 0; i + 1 < n; i += 32) {        // the rest are adjacent diffs
            __m256i a = _mm256_loadu_si256((const __m256i*)(pos+i+1));
            __m256i b = _mm256_loadu_si256((const __m256i*)(pos+i));
            _mm512_storeu_si512((__m512i*)(out+k+i),
                                _mm512_cvtepu8_epi16(_mm256_sub_epi8(a,b)));
        }
        k += n - 1; start = base + pos[n-1];
    }
    out[k++] = (uint16_t)(base - start);
    return k;
}

// Sparse: <~1 run per 256B block. Work proportional to run count.
static inline size_t encode_sparse(const uint8_t* p, size_t nblocks, uint16_t* out) {
    uint64_t carry = 0, start = 0, base = 0; size_t k = 0;
    for (size_t bi = 0; bi < nblocks; ++bi, p += 256, base += 256) {
        uint64_t m[4], e[4]; zero_masks(p, m); edges4(m, e, carry);
        for (int w = 0; w < 4; ++w) {
            uint64_t ew = e[w];
            while (ew) {
                uint64_t g = base + w*64 + (unsigned)_tzcnt_u64(ew);
                out[k++] = (uint16_t)(g - start); start = g; ew = _blsr_u64(ew);
            }
        }
    }
    out[k++] = (uint16_t)(base - start);
    return k;
}

} // namespace zle



// zle_codec.h -- zero-run-length codec built on the zle.h bitmap primitives.
//
// Stream = a sequence of (L,Z) pairs:
//     token     u8 { min(L,15):u4 , min(Z,15):u4 }
//     lit_ext   varint(L-15)   if L >= 15
//     zero_ext  varint(Z-15)   if Z >= 15
//     literals  L bytes verbatim
// The stored Z field is biased: 0 means "no zero run at all", and a run of R
// bytes (always >= ZLE_MIN_ZERO, shorter ones stay in the literal stream) is
// stored as R - ZLE_MIN_ZERO + 1. So the nibble covers R = 2..15 directly and
// 15 escapes to the extension. This costs one value of direct range versus
// storing R - ZLE_MIN_ZERO, and buys a self-terminating stream: the last frame
// is an ordinary frame with Z=0, no special case. Token 0x00 (no literals, no
// zeros) is therefore invalid and is rejected.
//
// Requires AVX512F/BW/VBMI2 + BMI1/BMI2 + POPCNT. Note pdep/pext are microcoded
// and slow on Zen 1/2; the varint path wants Zen 3+ or Intel.
#pragma once
#include "zle.h"
#include <string.h>

#define ZLE_MIN_ZERO 2u
// Worst case is an incompressible (zero-free) input: 1 token + varint + n bytes.
// The +64 covers varint_put's fixed 8-byte store, which may write past the
// returned length. Literal/zero emission is exact (see copy/zero below).
#define ZLE_COMPRESS_BOUND(n) ((n) + ((n)/255) + 64)

namespace zle {

static const uint64_t DATA_MASK = 0x7F7F7F7F7F7F7F7FULL;
static const uint64_t CONT_MASK = 0x8080808080808080ULL;

// LEB128 in ~6 instructions. Always stores 8 bytes; advances by the real length.
// Valid for x < 2^56 (pdep only spreads 56 payload bits).
static inline uint8_t* varint_put(uint8_t* out, uint64_t x) {
    unsigned bits = 64u - (unsigned)_lzcnt_u64(x | 1);
    unsigned n    = (bits + 6u) / 7u;
    uint64_t w    = _pdep_u64(x, DATA_MASK);
    w |= _bzhi_u64(CONT_MASK, 8u * (n - 1u));
    memcpy(out, &w, 8);
    return out + n;
}

// Inverse: pext gathers the payload nibbles, tzcnt locates the terminator.
#if ZLE_VARINT_MASKED
// Masked-load variant: correct, and it removes the fallback path entirely --
// AVX-512 suppresses faults on masked-off lanes, and zero padding reads as a
// terminator so a truncated varint stops at the buffer end. But it is MEASURED
// 13-20% SLOWER (see README) and is kept only for reference. Unlike copy(),
// which wants its data in a vector register anyway, tzcnt/pext are GPR
// instructions -- so a masked load forces a kmov GPR->k plus a vmovq vec->GPR
// round trip through the vector domain, replacing one plain 8-byte mov. The
// bounds branch it removes was free: it is false exactly once per stream.
static inline uint64_t varint_get(const uint8_t** ipp, const uint8_t* iend) {
    const uint8_t* ip = *ipp;
    size_t avail = (size_t)(iend - ip);
    unsigned take = avail < 8 ? (unsigned)avail : 8u;
    __mmask16 k = (__mmask16)_bzhi_u32(0xFFFFu, take);
    uint64_t w = (uint64_t)_mm_cvtsi128_si64(_mm_maskz_loadu_epi8(k, (const __m128i*)ip));
    unsigned n = ((unsigned)_tzcnt_u64(~w & CONT_MASK) >> 3) + 1u;
    if (n > take) n = take;                  // truncated: never advance past iend
    *ipp = ip + n;
    return _bzhi_u64(_pext_u64(w, DATA_MASK), 7u * n);
}
#else
static inline uint64_t varint_get(const uint8_t** ipp, const uint8_t* iend) {
    const uint8_t* ip = *ipp;
    if ((size_t)(iend - ip) >= 8) {
        uint64_t w; memcpy(&w, ip, 8);
        unsigned n = ((unsigned)_tzcnt_u64(~w & CONT_MASK) >> 3) + 1u;
        *ipp = ip + n;
        return _bzhi_u64(_pext_u64(w, DATA_MASK), 7u * n);
    }
    uint64_t v = 0; unsigned s = 0;                      // safe tail
    while (ip < iend) { uint8_t b = *ip++; v |= (uint64_t)(b & 0x7F) << s;
                        if (!(b & 0x80)) break; s += 7; }
    *ipp = ip; return v;
}
#endif

// Exact copy/fill: no overshoot, so no slack is required on either side and
// the length guards disappear. _mm512_maskz_loadu_epi8 does not fault on
// masked-off lanes, so the n<=64 case is a single branchless load/store pair
// even when the source ends mid-register. Large runs go to libc, which has
// rep-movsb/ERMS paths a 64-byte loop cannot beat.
// Upper bound of the straight-line masked copy/zero path. Capped at 256: the
// path is four unrolled words wide, so a larger value would silently truncate.
#ifndef ZLE_COPY_MAX
#define ZLE_COPY_MAX 256u
#endif
#ifndef ZLE_VARINT_MASKED
#define ZLE_VARINT_MASKED 0
#endif
#ifndef ZLE_LIT_SKIP
#define ZLE_LIT_SKIP 1
#endif
#ifndef ZLE_DEC_BRANCHLESS
#define ZLE_DEC_BRANCHLESS 0
#endif
#ifndef ZLE_TAIL_MASKED
#define ZLE_TAIL_MASKED 1
#endif
// One compress_small for every n in 1..256 instead of four NW-templated ones.
// MEASURED AND REJECTED: -9% on an interleaved-size stream, -39% to -49% on the
// smallest shapes (20 B, 29 B, 32 B, 64 B). The merge was meant to buy back the
// dispatch chain's mispredicts, but shuffling the sizes costs the templated
// version nothing at all (75.2 ns/acct shuffled vs 75.1 grouped), so there was
// no misprediction to recover -- only four unconditional words of load, filter,
// edge and walk work to pay on a 20-byte input. Kept building; see README.
// Which ZLE_MIN_ZERO strategy the <=256 path uses, independent of the general
// path's. Both emit identical bytes; see README.
#ifndef ZLE_SMALL_FILTER
#define ZLE_SMALL_FILTER 0
#endif
#ifndef ZLE_SMALL_WALK
#define ZLE_SMALL_WALK 0
#endif
#ifndef ZLE_SMALL_VARINT
#define ZLE_SMALL_VARINT 1
#endif
#ifndef ZLE_SMALL_MERGED
#define ZLE_SMALL_MERGED 0
#endif
#ifndef ZLE_MERGED_VARINT
#define ZLE_MERGED_VARINT 1
#endif
#ifndef ZLE_VARINT_FUSED
#define ZLE_VARINT_FUSED 0
#endif

// The 65..256 case is written STRAIGHT-LINE, not as a 64-byte loop, because GCC
// recognises such a loop as the memcpy idiom and emits `rep movsq` -- ~40 cycles
// of startup that swamped every literal run in this range (see README). The
// per-word limits come from one unsigned byte compare against IDXB_, the same
// trick the <=256 mask build uses: byte saturation clamps each word for free,
// so words wholly past n get an empty mask with no subtract and no branch.
static_assert(ZLE_COPY_MAX <= 256u, "ZLE_COPY_MAX > 256 needs more unrolled words in copy/zero");

static inline void copy(uint8_t* d, const uint8_t* s, size_t n) {
    if (n <= 64) {
        __mmask64 k = (__mmask64)_bzhi_u64(~0ULL, (unsigned)n);
        _mm512_mask_storeu_epi8(d, k, _mm512_maskz_loadu_epi8(k, s));
        return;
    }
    if (n <= ZLE_COPY_MAX) {
        const __m512i lim = _mm512_set1_epi8((char)(unsigned)(n - 1u));
        _mm512_storeu_si512((__m512i*)d, _mm512_loadu_si512((const __m512i*)s));
#define ZLE_CPW_(w) {                                                          \
            __mmask64 k_ = _mm512_cmple_epu8_mask(                             \
                _mm512_load_si512((const __m512i*)(IDXB_ + 64*(w))), lim);     \
            _mm512_mask_storeu_epi8(d + 64*(w), k_,                            \
                _mm512_maskz_loadu_epi8(k_, s + 64*(w))); }
        ZLE_CPW_(1) ZLE_CPW_(2) ZLE_CPW_(3)
#undef ZLE_CPW_
        return;
    }
    memcpy(d, s, n);
}

static inline void zero(uint8_t* d, size_t n) {
    const __m512i z = _mm512_setzero_si512();
    if (n <= 64) {
        _mm512_mask_storeu_epi8(d, (__mmask64)_bzhi_u64(~0ULL, (unsigned)n), z);
        return;
    }
    if (n <= ZLE_COPY_MAX) {            // straight-line for the same reason as copy
        const __m512i lim = _mm512_set1_epi8((char)(unsigned)(n - 1u));
        _mm512_storeu_si512((__m512i*)d, z);
#define ZLE_ZW_(w) _mm512_mask_storeu_epi8(d + 64*(w),                          \
            _mm512_cmple_epu8_mask(                                             \
                _mm512_load_si512((const __m512i*)(IDXB_ + 64*(w))), lim), z);
        ZLE_ZW_(1) ZLE_ZW_(2) ZLE_ZW_(3)
#undef ZLE_ZW_
        return;
    }
    memset(d, 0, n);
}

// Emit one (L,Z) sequence. Pass Z==0 for a trailing literals-only sequence.
//
// The two extension fields are produced BRANCHLESSLY and stored as a single
// 16-byte write. `L >= 15` and `Z >= 15` are genuinely unpredictable for most
// data (~30-40% taken at lit12/zero20), so a mispredict costs far more than the
// shift that replaces it -- the reverse of the decoder's varint branch, which
// is perfectly predicted and where the same trick loses.
//
// This works without any masking because pdep leaves the bytes ABOVE a varint's
// length zeroed: bits of x past 7*n are zero, and bzhi only lights continuation
// bits in bytes 0..n-2. So wl is exactly the varint in its low nl bytes, and wz
// can simply be shifted up by nl and OR-ed in.
// Zrun is the true run length, or 0 for "no zero run" (the last frame).
static inline uint8_t* emit_seq(uint8_t* op, const uint8_t* lit, size_t L, size_t Zrun) {
    size_t   Z  = Zrun ? Zrun - ZLE_MIN_ZERO + 1u : 0u;   // biased, 0 == none
    unsigned lt = L < 15 ? (unsigned)L : 15u;
    unsigned zt = Z < 15 ? (unsigned)Z : 15u;
    *op++ = (uint8_t)((lt << 4) | zt);
#if ZLE_MERGED_VARINT
    {
    uint64_t ml = (uint64_t)0 - (uint64_t)(L >= 15);
    uint64_t mz = (uint64_t)0 - (uint64_t)(Z >= 15);
    uint64_t xl = (uint64_t)(L - 15) & ml, xz = (uint64_t)(Z - 15) & mz;
    unsigned nl = ((64u - (unsigned)_lzcnt_u64(xl | 1)) + 6u) / 7u & (unsigned)ml;
    unsigned nz = ((64u - (unsigned)_lzcnt_u64(xz | 1)) + 6u) / 7u & (unsigned)mz;
    uint64_t wl = (_pdep_u64(xl, DATA_MASK) | _bzhi_u64(CONT_MASK, 8u*(nl - 1u))) & ml;
    uint64_t wz = (_pdep_u64(xz, DATA_MASK) | _bzhi_u64(CONT_MASK, 8u*(nz - 1u))) & mz;
    __uint128_t comb = (__uint128_t)wl | ((__uint128_t)wz << (8u * nl));
    memcpy(op, &comb, 16);
    op += nl + nz;
    }
#else
    if (L >= 15) op = varint_put(op, L - 15);
    if (Z >= 15) op = varint_put(op, Z - 15);
#endif
    copy(op, lit, L);
    return op + L;
}

// Zero-mask for a final partial block, straight from src with no scratch
// buffer. Merging into a register of 0xFF makes the out-of-range lanes read as
// NON-zero, i.e. bit clear -- which is exactly the padding semantics we want,
// and masked lanes never fault.
static inline void zero_masks_tail(const uint8_t* p, size_t n, uint64_t m[4]) {
    const __m512i ones = _mm512_set1_epi8((char)0xFF);
    for (unsigned w = 0; w < 4; ++w) {
        size_t off = 64u * w;
        size_t rem = off < n ? n - off : 0;
        __mmask64 k = (__mmask64)_bzhi_u64(~0ULL, rem > 64 ? 64u : (unsigned)rem);
        __m512i v = _mm512_mask_loadu_epi8(ones, k, p + off);
        m[w] = _mm512_testn_epi8_mask(v, v);
    }
}

// True iff all 256 bytes are zero. ~8 uops, versus ~25 to build the masks and
// run the edge/compress pass -- which is what lets a long zero run be skipped
// at memory speed instead of at mask-building speed.
static inline int all_zero_256(const uint8_t* p) {
    const __m512i* v = (const __m512i*)p;
    __m512i o = _mm512_or_si512(
                  _mm512_or_si512(_mm512_loadu_si512(v+0), _mm512_loadu_si512(v+1)),
                  _mm512_or_si512(_mm512_loadu_si512(v+2), _mm512_loadu_si512(v+3)));
    return !_mm512_test_epi8_mask(o, o);
}

// True iff no two ADJACENT zero bytes occur in p[0..256) -- i.e. the span holds
// no zero run long enough to earn a token. or(p[i],p[i+1]) is zero exactly at a
// candidate, and min_epu8 folds the four vectors so one testn answers for all
// 256 bytes. Reads p[256], so callers must have that byte.
static inline int no_zero_pair_256(const uint8_t* p) {
    const __m512i a0 = _mm512_or_si512(_mm512_loadu_si512((const __m512i*)(p      )),
                                       _mm512_loadu_si512((const __m512i*)(p +  1)));
    const __m512i a1 = _mm512_or_si512(_mm512_loadu_si512((const __m512i*)(p +  64)),
                                       _mm512_loadu_si512((const __m512i*)(p +  65)));
    const __m512i a2 = _mm512_or_si512(_mm512_loadu_si512((const __m512i*)(p + 128)),
                                       _mm512_loadu_si512((const __m512i*)(p + 129)));
    const __m512i a3 = _mm512_or_si512(_mm512_loadu_si512((const __m512i*)(p + 192)),
                                       _mm512_loadu_si512((const __m512i*)(p + 193)));
    const __m512i t  = _mm512_min_epu8(_mm512_min_epu8(a0, a1), _mm512_min_epu8(a2, a3));
    return !_mm512_testn_epi8_mask(t, t);
}

// ---------------------------------------------------------------------------
// compress
// ---------------------------------------------------------------------------
// dst must hold ZLE_COMPRESS_BOUND(src_len); no bounds are checked.
//
// FILTER=true  strips zero runs shorter than ZLE_MIN_ZERO from the bitmap
//              before the edge pass, so they never become boundaries.
// FILTER=false leaves them in and drops them in the walk instead (the "merge"
//              strategy: a too-short run simply doesn't move lit_start, so the
//              literal run absorbs it). No buffer needed -- it is one compare.
// Both produce byte-identical output; see README for which is faster when.
// The <=256 path's own emit. Every length here is bounded by 256, so L-15 and
// Z-15 are at most 241 and each extension is ONE OR TWO varint bytes -- never
// more. That retires the general encoder's whole apparatus: no pdep, no lzcnt,
// no divide-by-7 length calculation, no 128-bit variable shift, no 16-byte
// store. What is left is a pair of shifts and one 4-byte store into slack the
// bound already guarantees. It also moves the work off port 1, which pdep,
// lzcnt and the walk's tzcnt all contend for.
static inline uint8_t* emit_seq_small(uint8_t* op, const uint8_t* lit,
                                      size_t L, size_t Zrun) {
    size_t   Z  = Zrun ? Zrun - ZLE_MIN_ZERO + 1u : 0u;
    unsigned lt = L < 15 ? (unsigned)L : 15u;
    unsigned zt = Z < 15 ? (unsigned)Z : 15u;
    *op++ = (uint8_t)((lt << 4) | zt);
    uint32_t nl = (uint32_t)(L >= 15),  nz = (uint32_t)(Z >= 15);
    uint32_t xl = (uint32_t)(L - 15) & (0u - nl);
    uint32_t xz = (uint32_t)(Z - 15) & (0u - nz);
    uint32_t bl = (uint32_t)(xl >= 128), bz = (uint32_t)(xz >= 128);
    uint32_t el = (xl & 0x7Fu) | (bl << 7) | ((xl >> 7) << 8);   // 1-2 bytes
    uint32_t ez = (xz & 0x7Fu) | (bz << 7) | ((xz >> 7) << 8);
    nl += bl; nz += bz;
    uint32_t v = el | (ez << (8u * nl));
    memcpy(op, &v, 4);
    op += nl + nz;
    copy(op, lit, L);
    return op + L;
}
#if !ZLE_SMALL_VARINT
#define emit_seq_small emit_seq
#endif

// n <= 256: one block, and the vpcompressb round trip through a stack buffer
// is pure loss -- the store-to-load forward stall costs more than the handful
// of boundaries it materializes. Walk the edge words directly with tzcnt/blsr.
// This is the dominant shape in practice (most accounts are < 256 B).
//
#if ZLE_SMALL_MERGED
// One path for every n in 1..256, no dispatch chain. The four in-range masks
// come from a single unsigned byte compare against IDXB_ (which holds 0..255
// across the four vectors): lane 64w+i survives iff 64w+i <= n-1. Byte
// saturation does the clamping for free -- at n=20 every lane of I1/I2/I3
// exceeds 19, so k1..k3 fall out empty with no per-word subtract, no clamp, no
// branch and no GPR->k move. That is why this beats building the masks in
// GPRs: `bzhi(~0, n - 64w)` saturates the wrong way (n-64w wraps negative and
// bzhi reads only bits[7:0], yielding all-ones instead of none), so the GPR
// form needs a clamp per word; the compare form needs nothing.
//
// The loads MERGE INTO ALL-ONES rather than zeroing. maskz would leave the
// out-of-range lanes at 0, vptestnmb would call them zero bytes, and every
// short input would grow a phantom trailing zero run. Masked lanes suppress
// faults either way, so a 20-byte source is still only read for 20 bytes.
//
// Nothing about the (L,Z) walk changes. Past n the bitmap is all zeros, so:
//   - the FILTER neighbour pass cannot resurrect a bit there;
//   - edges4 closes a run that ends at n-1 by emitting its closing boundary at
//     exactly n, which is the right length, and lit_p then lands on src+n so
//     the tail's `if (!L) return` fires;
//   - only a run touching byte 255 stays open, and that is the existing
//     in_zero tail case.
template <bool FILTER>
static inline size_t compress_small(uint8_t* op, const uint8_t* src, size_t n) {
    uint8_t* const op0 = op;
    const __m512i lim  = _mm512_set1_epi8((char)(unsigned)(n - 1u));   // n >= 1
    const __m512i ones = _mm512_set1_epi8((char)0xFF);
    uint64_t m[4], f[4], e[4];
#define ZLE_SMW_(w) {                                                          \
        __mmask64 k_ = _mm512_cmple_epu8_mask(                                 \
            _mm512_load_si512((const __m512i*)(IDXB_ + 64*(w))), lim);         \
        __m512i v_ = _mm512_mask_loadu_epi8(ones, k_, src + 64*(w));           \
        m[w] = _mm512_testn_epi8_mask(v_, v_); }
    ZLE_SMW_(0) ZLE_SMW_(1) ZLE_SMW_(2) ZLE_SMW_(3)
#undef ZLE_SMW_

    if (FILTER) {
        uint64_t pm = 0;
        for (unsigned w = 0; w < 4; ++w) {
            uint64_t nb = (w < 3) ? (m[w+1] & 1u) : 0u;
            f[w] = m[w] & (((m[w] << 1) | pm) | ((m[w] >> 1) | (nb << 63)));
            pm = m[w] >> 63;
        }
    } else { f[0]=m[0]; f[1]=m[1]; f[2]=m[2]; f[3]=m[3]; }

    uint64_t fc = 0;
    for (unsigned w = 0; w < 4; ++w) { e[w] = f[w] ^ ((f[w] << 1) | fc); fc = f[w] >> 63; }

    const uint8_t* lit_p = src; const uint8_t* zero_p = src; int in_zero = 0;
    for (unsigned w = 0; w < 4; ++w) {
        uint64_t ew = e[w];
        while (ew) {
            const uint8_t* g = src + (size_t)w*64 + (unsigned)_tzcnt_u64(ew);
            ew = _blsr_u64(ew);
            if (!in_zero) { zero_p = g; in_zero = 1; continue; }
            in_zero = 0;
            size_t zact = (size_t)(g - zero_p);
            if (!FILTER && zact < ZLE_MIN_ZERO) continue;
            op = emit_seq_small(op, lit_p, (size_t)(zero_p - lit_p), zact);
            lit_p = g;
        }
    }
    const uint8_t* send = src + n;
    size_t L, Zrun;
    if (in_zero && (size_t)(send - zero_p) >= ZLE_MIN_ZERO) {
        L = (size_t)(zero_p - lit_p); Zrun = (size_t)(send - zero_p);
    } else {
        L = (size_t)(send - lit_p); Zrun = 0;
        if (!L) return (size_t)(op - op0);
    }
    return (size_t)(emit_seq_small(op, lit_p, L, Zrun) - op0);
}
#else
// NW (the live mask word count) is a template parameter, so every loop here
// unrolls to straight-line code with the masks in registers and no dead work
// on words past the end. A runtime bound would cost a branch per word; four
// unconditional words would cost three dead masked loads on a 40-byte input.
template <bool FILTER, unsigned NW>
static inline size_t compress_small(uint8_t* op, const uint8_t* src, size_t n) {
    uint8_t* const op0 = op;
    uint64_t m[NW], f[NW], e[NW];
    // With NW fixed, n > 64*(NW-1), so every word but the last is FULLY in
    // range: plain loads, no mask arithmetic at all. Only the final word needs
    // a mask, and its length is 1..64 so bzhi's saturation covers 64 exactly.
    for (unsigned w = 0; w + 1 < NW; ++w) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(src + 64u * w));
        m[w] = _mm512_testn_epi8_mask(v, v);
    }
    {
        const unsigned w = NW - 1u;
        __mmask64 k = (__mmask64)_bzhi_u64(~0ULL, (unsigned)(n - 64u * w));
        __m512i v = _mm512_mask_loadu_epi8(_mm512_set1_epi8((char)0xFF), k, src + 64u * w);
        m[w] = _mm512_testn_epi8_mask(v, v);
    }
    if (FILTER) {
        uint64_t pm = 0;
        for (unsigned w = 0; w < NW; ++w) {
            uint64_t nb = (w + 1 < NW) ? (m[w+1] & 1u) : 0u;
            f[w] = m[w] & (((m[w] << 1) | pm) | ((m[w] >> 1) | (nb << 63)));
            pm = m[w] >> 63;
        }
    } else { for (unsigned w = 0; w < NW; ++w) f[w] = m[w]; }

    uint64_t fc = 0;
    for (unsigned w = 0; w < NW; ++w) { e[w] = f[w] ^ ((f[w] << 1) | fc); fc = f[w] >> 63; }

    const uint8_t* lit_p = src; const uint8_t* zero_p = src; int in_zero = 0;
    for (unsigned w = 0; w < NW; ++w) {
        uint64_t ew = e[w];
        while (ew) {
            const uint8_t* g = src + (size_t)w*64 + (unsigned)_tzcnt_u64(ew);
            ew = _blsr_u64(ew);
            if (!in_zero) { zero_p = g; in_zero = 1; continue; }
            in_zero = 0;
            size_t zact = (size_t)(g - zero_p);
            if (!FILTER && zact < ZLE_MIN_ZERO) continue;
            op = emit_seq_small(op, lit_p, (size_t)(zero_p - lit_p), zact);
            lit_p = g;
        }
    }
    const uint8_t* send = src + n;
    size_t L, Zrun;
    if (in_zero && (size_t)(send - zero_p) >= ZLE_MIN_ZERO) {
        L = (size_t)(zero_p - lit_p); Zrun = (size_t)(send - zero_p);
    } else {
        L = (size_t)(send - lit_p); Zrun = 0;
        if (!L) return (size_t)(op - op0);
    }
    return (size_t)(emit_seq_small(op, lit_p, L, Zrun) - op0);
}
#endif  // ZLE_SMALL_MERGED

#if ZLE_SMALL_WALK
// fd_zle's walk shape: derive the run LENGTH directly instead of building a
// boundary bitmap. tzcnt(m) is the run start, tzcnt(~(m>>b)) is its length, so
// one iteration handles a whole run -- no edges pass, no open/close state
// machine, no per-edge branch, and the MIN_ZERO test is free.
template <bool FILTER, unsigned NW>
static inline size_t compress_small_walk(uint8_t* op, const uint8_t* src, size_t n) {
    uint8_t* const op0 = op;
    uint64_t m[NW];
    for (unsigned w = 0; w + 1 < NW; ++w) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(src + 64u * w));
        m[w] = _mm512_testn_epi8_mask(v, v);
    }
    {
        const unsigned w = NW - 1u;
        __mmask64 k = (__mmask64)_bzhi_u64(~0ULL, (unsigned)(n - 64u * w));
        __m512i v = _mm512_mask_loadu_epi8(_mm512_set1_epi8((char)0xFF), k, src + 64u * w);
        m[w] = _mm512_testn_epi8_mask(v, v);
    }
    size_t fs = 0;
    for (unsigned w = 0; w < NW; ++w) {
        uint64_t mm = m[w];
        while (mm) {
            unsigned b  = (unsigned)_tzcnt_u64(mm);
            size_t   zs = (size_t)w * 64u + b;
            uint64_t hi = mm >> b;
            size_t   len;
            if (~hi) { len = (size_t)_tzcnt_u64(~hi); mm &= ~(((1ULL << len) - 1ULL) << b); }
            else     { len = 64u - b; mm = 0; }
            if (b + len == 64u) {                       // run reached the word end
                for (unsigned x = w + 1u; x < NW; ++x) {
                    uint64_t t = m[x];
                    if (~t) { unsigned e = (unsigned)_tzcnt_u64(~t);
                              len += e; m[x] = t & ~((1ULL << e) - 1ULL); break; }
                    len += 64u; m[x] = 0;
                }
            }
            if (len >= ZLE_MIN_ZERO) {
                op = emit_seq_small(op, src + fs, zs - fs, len);
                fs = zs + len;
            }
        }
    }
    if (fs < n) op = emit_seq_small(op, src + fs, n - fs, 0);
    return (size_t)(op - op0);
}
#define compress_small compress_small_walk
#endif  // ZLE_SMALL_WALK

template <bool FILTER>
static inline size_t compress_impl(void* __restrict dstv,
                                   const void* __restrict srcv, size_t src_len) {
    uint8_t* op = (uint8_t*)dstv;
    const uint8_t* src = (const uint8_t*)srcv;
    if (!src_len) return 0;
    constexpr bool SF = (bool)ZLE_SMALL_FILTER;
#if ZLE_SMALL_MERGED
    if (src_len <= 256) return compress_small<SF>((uint8_t*)dstv, src, src_len);
#else
    if (src_len <=  64) return compress_small<SF,1>((uint8_t*)dstv, src, src_len);
    if (src_len <= 128) return compress_small<SF,2>((uint8_t*)dstv, src, src_len);
    if (src_len <= 192) return compress_small<SF,3>((uint8_t*)dstv, src, src_len);
    if (src_len <= 256) return compress_small<SF,4>((uint8_t*)dstv, src, src_len);
#endif

    alignas(64) uint8_t pos[320];
    const __m512i I0 = _mm512_load_si512((const __m512i*)(IDXB_+0));
    const __m512i I1 = _mm512_load_si512((const __m512i*)(IDXB_+64));
    const __m512i I2 = _mm512_load_si512((const __m512i*)(IDXB_+128));
    const __m512i I3 = _mm512_load_si512((const __m512i*)(IDXB_+192));

    const uint8_t* lit_p  = src;      // start of the open literal run
    const uint8_t* zero_p = src;      // start of the open zero run
    int    in_zero = 0;
    uint64_t mcarry = 0, fcarry = 0;   // last raw bit / last filtered bit

    const size_t nblk = (src_len + 255) / 256;
    for (size_t b = 0; b < nblk; ++b) {
        const size_t base = b * 256;
        uint64_t m[4];
        if (base + 256 <= src_len) zero_masks(src + base, m);
#if ZLE_TAIL_MASKED
        else zero_masks_tail(src + base, src_len - base, m);
#else
        else { alignas(64) uint8_t tb[256]; memset(tb, 0xFF, 256);
               memcpy(tb, src + base, src_len - base); zero_masks(tb, m); }
#endif

        uint64_t f[4];
        if (FILTER) {
            // Keep only bytes belonging to a zero run of length >= 2: a set bit
            // survives iff its left or right neighbour is also set. Singletons
            // drop out here and never generate a boundary.
            uint64_t pm = mcarry;
            const uint64_t nxt = (base + 256 < src_len) ? (uint64_t)(src[base+256] == 0) : 0;
            for (int w = 0; w < 4; ++w) {
                uint64_t nb = (w < 3) ? (m[w+1] & 1u) : nxt;
                f[w] = m[w] & (((m[w] << 1) | pm) | ((m[w] >> 1) | (nb << 63)));
                pm = m[w] >> 63;
            }
            mcarry = pm;
        } else { f[0]=m[0]; f[1]=m[1]; f[2]=m[2]; f[3]=m[3]; }

        // Two symmetric fast-forwards, each gated on this block having produced
        // no boundaries, so ordinary input pays only 3 ANDs / 3 ORs.
        //
        // Inside an open zero run: a wholly zero block yields no boundaries.
        if (fcarry && (f[0] & f[1] & f[2] & f[3]) == ~0ULL) {
            size_t nb = b + 1;
            while (nb < nblk && (nb + 1) * 256 <= src_len && all_zero_256(src + nb * 256)) ++nb;
            mcarry = 1; fcarry = 1; b = nb - 1;   // the for-loop ++b lands on nb
            continue;
        }
        // Inside a literal run: a block with no adjacent-zero pair likewise
        // yields none. This is the incompressible-span case.
        if (ZLE_LIT_SKIP && !fcarry && (f[0] | f[1] | f[2] | f[3]) == 0) {
            size_t nb = b + 1;
            while (nb < nblk && (nb + 1) * 256 + 1 <= src_len && no_zero_pair_256(src + nb * 256)) ++nb;
            mcarry = (uint64_t)(src[nb * 256 - 1] == 0); fcarry = 0; b = nb - 1;
            continue;
        }

        uint64_t e[4]; edges4(f, e, fcarry);      // run boundaries, all at once
        uint8_t* o = pos;
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[0],I0); o += _mm_popcnt_u64(e[0]);
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[1],I1); o += _mm_popcnt_u64(e[1]);
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[2],I2); o += _mm_popcnt_u64(e[2]);
        _mm512_mask_compressstoreu_epi8(o,(__mmask64)e[3],I3); o += _mm_popcnt_u64(e[3]);

        const uint8_t* blk = src + base;
        for (size_t i = 0, n = (size_t)(o - pos); i < n; ++i) {
            const uint8_t* g = blk + pos[i];
            if (!in_zero) { zero_p = g; in_zero = 1; }
            else {
                in_zero = 0;
                size_t zact = (size_t)(g - zero_p);
                // Too short to be worth a token: leave lit_p alone and the
                // literal run swallows it. Always false when FILTER.
                if (!FILTER && zact < ZLE_MIN_ZERO) continue;
                op = emit_seq(op, lit_p, (size_t)(zero_p - lit_p), zact);
                lit_p = g;
            }
        }
    }

    // Flush. A trailing zero run long enough to earn a token becomes a normal
    // sequence; otherwise everything left is literals and Z==0 marks the
    // literals-only terminator (see decompress).
    const uint8_t* send = src + src_len;
    size_t L, Zrun;
    if (in_zero && (size_t)(send - zero_p) >= ZLE_MIN_ZERO) {
        L = (size_t)(zero_p - lit_p); Zrun = (size_t)(send - zero_p);
    } else {
        L = (size_t)(send - lit_p); Zrun = 0;
        if (!L) return (size_t)(op - (uint8_t*)dstv);        // ended exactly on a run
    }
    op = emit_seq(op, lit_p, L, Zrun);
    return (size_t)(op - (uint8_t*)dstv);
}

static inline size_t compress(void* __restrict d, const void* __restrict s, size_t n) {
    return compress_impl<true>(d, s, n);
}
static inline size_t compress_merge(void* __restrict d, const void* __restrict s, size_t n) {
    return compress_impl<false>(d, s, n);
}

// Decode one varint already sitting in a register; returns its byte length.
static inline unsigned varint_dec(uint64_t x, uint64_t* out) {
    unsigned n = ((unsigned)_tzcnt_u64(~x & CONT_MASK) >> 3) + 1u;
    *out = _bzhi_u64(_pext_u64(x, DATA_MASK), 7u * n);
    return n;
}

// ---------------------------------------------------------------------------
// decompress -- returns bytes written, or 0 if the stream is malformed.
// ---------------------------------------------------------------------------
static inline size_t decompress(void* __restrict dstv, size_t dst_len,
                                const void* __restrict srcv, size_t src_len) {
    uint8_t* const dst = (uint8_t*)dstv;
    uint8_t* op = dst; uint8_t* const oend = dst + dst_len;
    const uint8_t* ip = (const uint8_t*)srcv;
    const uint8_t* const iend = ip + src_len;

    while (ip < iend) {
        unsigned tok = *ip++;
        if (!tok) return 0;                       // 0x00 is not a legal token
        size_t L = tok >> 4, Z = tok & 15u;

        if (ZLE_DEC_BRANCHLESS && __builtin_expect((size_t)(iend - ip) >= 16, 1)) {
            // Both extension fields are adjacent and at most 8 bytes each, so
            // one 16-byte window serves both. Decoding them BRANCHLESSLY is
            // what matters: `L==15` / `Z==15` are as unpredictable here as in
            // the encoder, where removing them was worth 40%.
            __uint128_t ww; memcpy(&ww, ip, 16);
            uint64_t w1 = (uint64_t)ww;
            uint64_t mL = (uint64_t)0 - (uint64_t)(L == 15);
            unsigned n1 = ((((unsigned)_tzcnt_u64(~w1 & CONT_MASK) >> 3) + 1u)) & (unsigned)mL;
            L += _bzhi_u64(_pext_u64(w1, DATA_MASK), 7u * n1);
            uint64_t w2 = (uint64_t)(ww >> (8u * n1));
            uint64_t mZ = (uint64_t)0 - (uint64_t)(Z == 15);
            unsigned n2 = ((((unsigned)_tzcnt_u64(~w2 & CONT_MASK) >> 3) + 1u)) & (unsigned)mZ;
            Z += _bzhi_u64(_pext_u64(w2, DATA_MASK), 7u * n2);
            ip += n1 + n2;
        } else {
            if (L == 15) L += varint_get(&ip, iend);
            if (Z == 15) Z += varint_get(&ip, iend);
        }
        // unbias: 0 stays 0 (no run), k>=1 means a run of k - 1 + ZLE_MIN_ZERO
        Z += (ZLE_MIN_ZERO - 1u) & (size_t)(0u - (size_t)(Z != 0));

        // One guard for the whole frame. copy/zero are exact and mask-driven,
        // so L==0 and Z==0 need no tests of their own.
        if (__builtin_expect((size_t)(iend - ip) < L
                          || (size_t)(oend - op) < L + Z, 0)) return 0;
        copy(op, ip, L); op += L; ip += L;
        zero(op, Z);     op += Z;
    }
    return (size_t)(op - dst);
}

} // namespace zle
