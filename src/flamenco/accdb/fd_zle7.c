#include "fd_zle7.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format (from fd_zle_pseudocode2, transcribed here since that
   file is pseudocode, not the spec of record):

     [ header:u8, payload... ]
     [ type:u2{raw=0, fill=1, last=2, more=3}, ext:u6 ]:header

     [ L:u8, Z:u8 ]:token
       if L == 0xFF: L += read_varint(payload)
       if Z == 0xFF: Z += read_varint(payload)

     type.raw:  payload is uncompressed bytes, copy all out; the real
                payload starts at header.ext bytes into the payload
                (data_sz bytes are copied from comp+1+ext).

     type.fill: write out n zero bytes.
                n = header.ext; if n==63: n += read_varint(payload); n += 1

     type.last: payload is a single frame.
                literals written forward from payload[2..]
                tokens written backwards from payload[-1]
                payload[0..2) holds the first token's (L, Z) header

     type.more: [size_ext:u16, literals:[size]u8, tokens:[...]u8]
                size:u22 = (size_ext<<6)|header.ext, tokens follow the
                literal plane, written forwards.

   All four frame types are now implemented: data_sz<=64 uses
   zle7_compress_small (type.raw/fill/last, unchanged from the original
   prototype), data_sz>64 uses zle7_compress_more (type.more, this
   session's addition) which falls back to a type.raw frame whenever
   the more-compressed form doesn't actually win (see zle7_finish_more).

   type.more's header is a *fixed* 3 bytes (the type/ext byte plus the
   u16 size_ext) -- unlike fd_zle6's variable-width c0/0xFF header, the
   literal plane's final address is known before the scan even starts,
   so there is no zle6-style "shift the literal plane left/right once
   the header width is known" step here at all.  Only the (much
   smaller) token stream ever moves: it is written provisionally past
   the literal plane's worst-case extent (offset+n, since the literal
   plane can never be longer than the input) and copied down to abut
   the literal plane's real end once Lsz is known -- see
   zle7_finish_more.  The 22-bit size field caps the literal plane at
   2^22-1 (~4 MiB); an input whose literal plane would not fit (only
   reachable with a multi-MiB incompressible span) falls back to
   type.raw instead, same as the "doesn't win" case.

   type.more's tokens reuse the exact (L, Z) field format documented
   above (0xFF escapes to a varint, added back to the 0xFF base -- the
   file's original delta convention, not fd_zle6's fixed-width u32 or
   its +2 Z bias) and the same terminal-lone-L rule zle7_compress_small
   already established for type.last: the decoder knows data_sz, so the
   token that completes the output may omit its Z field with no wire
   marker.  The scan itself (zero mask, run-start candidate mask, the
   carry-propagate run-length resolution, and the 256-byte skip-ahead in
   both directions) is fd_zle5/6's, unchanged -- see ZLE4_DESIGN.md.

   One correctness point NOT visible in the pseudocode's comments, only
   in what makes the backward token walk decodable: the (L, Z) pair
   for each run must be laid out backwards as (Z, L) -- Z at the lower
   address of the pair, L at the higher one -- not (L, Z). A decoder
   walking backward from payload[-1] hits the higher-address field of
   each pair first, and it needs that first hit to be an L (to decide,
   fd_zle6-style, "did this L field alone complete the output -- if so
   stop, this run's Z was actually the *previous* real run's trailing
   last-literal-byte-in-disguise" -- MIN_Z==2 makes that unambiguous:
   a real run's L can never by itself bring output to data_sz, only
   the always-present trailing lone-L field can). Writing (L, Z) in
   that address order instead (what a first reading of the pseudocode
   suggests) makes the terminal case undecodable without also tracking
   where the literal-plane cursor and the backward cursor meet. See the
   trace in zle7_compress_small below; the header extraction swaps
   back to (L, Z) order since that's the documented header layout. */

#define ZLE7_TYPE_RAW  (0UL)
#define ZLE7_TYPE_FILL (1UL)
#define ZLE7_TYPE_LAST (2UL)
#define ZLE7_TYPE_MORE (3UL)

#define ZLE7_TOK_ESC (0xFFUL) /* token field escape marker -- unreachable from
                                  the <=64B encoder (L,Z<=64<0xFF always;
                                  fd_zle7_decompress's type.last case treats
                                  seeing it there as corrupt), but real and
                                  exercised by zle7_compress_more's type.more
                                  tokens below, where L/Z routinely exceed
                                  0xFF. */

#define ZLE7_MIN_Z      (2UL)        /* shortest zero run zle7_compress_more will frame */
#define ZLE7_MORE_HDR   (3UL)        /* type.more header: 1 type/ext byte + u16 size_ext, always */
#define ZLE7_SIZE_BITS  (22UL)       /* size_ext(16) + header.ext(6) */
#define ZLE7_SIZE_MAX   ( (1UL<<ZLE7_SIZE_BITS)-1UL ) /* largest literal-plane size a type.more header can hold */

FD_STATIC_ASSERT( ZLE7_MIN_Z==2UL, unit );

/* ==================================================================
   Varint (LEB128).  Originally used only by type.fill's escape
   extension (n==63 when len==64, the one escape path the <=64B encoder
   can reach); zle7_compress_more's type.more tokens below are the
   second, much hotter, user.  Same shape as fd_zle.c's
   fd_zle_vput/fd_zle_vget. */

static inline int
zle7_vget( uchar const * s,
           ulong         slen,
           ulong *       pi,
           ulong *       out ) {
  ulong v  = 0UL;
  ulong i  = *pi;
  int   sh = 0;
  for(;;) {
    if( FD_UNLIKELY( (i>=slen) | (sh>42) ) ) return 0;
    uchar b = s[i++];
    v |= (ulong)( b & 0x7f )<<sh;
    if( !(b & 0x80) ) break;
    sh += 7;
  }
  *pi  = i;
  *out = v;
  return 1;
}

static inline ulong
zle7_vput( uchar * p,
           ulong   v ) {
  ulong k = 0UL;
  do {
    p[k] = (uchar)( v & 0x7fUL );
    v >>= 7;
    if( v ) p[k] = (uchar)( p[k] | 0x80U );
    k++;
  } while( v );
  return k;
}

/* zle7_tok_get reads one token field (an L or a Z) at *pi: a direct
   byte, or ZLE7_TOK_ESC followed by a varint added back to the
   ZLE7_TOK_ESC base -- the `if L==0xFF: L += read_varint(...)`
   convention documented at the top of this file, shared by both
   fields.  Returns 0 (leaving pi/out untouched) on truncated or
   overflowing input. */

static inline int
zle7_tok_get( uchar const * s,
              ulong         slen,
              ulong *       pi,
              ulong *       out ) {
  ulong i = *pi;
  if( FD_UNLIKELY( i>=slen ) ) return 0;
  ulong v = (ulong)s[i++];
  if( FD_UNLIKELY( v==ZLE7_TOK_ESC ) ) {
    ulong ext;
    if( FD_UNLIKELY( !zle7_vget( s, slen, &i, &ext ) ) ) return 0;
    v = ZLE7_TOK_ESC+ext;
  }
  *pi  = i;
  *out = v;
  return 1;
}

/* zle7_emit_l / zle7_emit write one token's L field, or a full (L, Z)
   token, at cp -- the encode-side counterpart of zle7_tok_get.  Pure
   scalar bit-twiddling, no AVX-512 needed, so (like fd_zle6's
   zle6_emit_l/zle6_emit) these live outside the FD_HAS_AVX512 gate
   below even though their only caller, zle7_compress_more, is inside
   it. */

static inline uchar *
zle7_emit_l( uchar * cp,
             ulong   L ) {
  if( FD_LIKELY( L<ZLE7_TOK_ESC ) ) { cp[0] = (uchar)L; return cp+1UL; }
  cp[0] = (uchar)ZLE7_TOK_ESC;
  return cp+1UL+zle7_vput( cp+1UL, L-ZLE7_TOK_ESC );
}

static inline uchar *
zle7_emit( uchar * cp,
           ulong   L,
           ulong   Z ) {
  cp = zle7_emit_l( cp, L );
  if( FD_LIKELY( Z<ZLE7_TOK_ESC ) ) { cp[0] = (uchar)Z; return cp+1UL; }
  cp[0] = (uchar)ZLE7_TOK_ESC;
  return cp+1UL+zle7_vput( cp+1UL, Z-ZLE7_TOK_ESC );
}

/* zle7_raw writes the type.raw fallback frame: header byte (ext=0)
   followed by a verbatim copy of the input.  comp_sz = n+1. */

static inline ulong
zle7_raw( uchar *       FD_RESTRICT dst,
          uchar const * FD_RESTRICT src,
          ulong                     n ) {
  dst[0] = (uchar)ZLE7_TYPE_RAW;
  fd_memcpy( dst+1UL, src, n );
  return n+1UL;
}

#if FD_HAS_AVX512

/* zle7_copy_le64 -- masked-store copy of sz<=64 bytes.  Every call site
   in zle7_compress_small has sz<=64 (the largest possible is the
   backward metadata region, itself bounded by len<=64 -- see the
   comment on FD_ZLE7_COMPRESS_BOUND's worst case), so unlike
   fd_zle5/6's zle_copy this doesn't need the >64 / >256 tiers at all. */

static inline void
zle7_copy_le64( uchar *       FD_RESTRICT d,
                 uchar const * FD_RESTRICT s,
                 ulong                     sz ) {
  __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
  _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
}

/* zle7_compress_small implements fd_zle_pseudocode2's
   zle_compress_small for 1<=len<=64, with three fixes over the literal
   pseudocode (found by trace/inspection, not yet by fuzzing -- see
   test_zle7.c):

   1. `d` is masked to bzhi(d,len-1) right after being computed.  The
      top-level `v` is a masked load that zero-fills every lane past
      len, so the zero-byte mask `m` (and therefore the adjacency mask
      `d`) sees phantom zero bytes past len.  Unmasked, this both
      manufactures a candidate run out of a lone real trailing zero
      byte (corrupting output -- the token would claim more zero bytes
      than the input actually has) and hides the all-zero fast path
      for any len<64 (the real all-zero run merges with the padding,
      so `d` never matches the exact bzhi(~0,len-1) pattern being
      compared against).

   2. the loop continuation test is `d!=0`, not `d==0` -- literally
      inverted, it either re-enters the loop with a stale d==0 (garbage
      second token) or exits after the first run and drops every run
      after it.

   3. the trailing literal segment (length `L` after the last run) is
      actually copied to the literal plane before the header/token
      metadata is written -- the pseudocode computed the count and
      recorded it in the token trailer but never copied the bytes,
      leaving the decoder to read uninitialized/wrong data for that
      span.

   4. (this file only, not something the user's edits needed to touch)
      each run's (L,Z) pair is written to the backward scratch as
      (Z,L) -- Z at the lower address, L at the higher -- not (L,Z).
      See the wire-format comment above for why. */

static ulong
zle7_compress_small( uchar *       FD_RESTRICT dst,
                      uchar const * FD_RESTRICT src,
                      ulong                     len ) {
  ulong     k  = _bzhi_u64( ~0UL, (uint)len );
  __mmask64 km = _cvtu64_mask64( k );
  __m512i   v  = _mm512_maskz_loadu_epi8( km, src );
  ulong     m  = (ulong)_mm512_cmpeq_epi8_mask( v, _mm512_setzero_si512() );
  ulong     d  = m & ( m>>1 );
  d = _bzhi_u64( d, (uint)(len-1UL) ); /* fix 1: strip padding-only candidates */

  if( FD_LIKELY( d==0UL ) ) { /* raw: no run of >=2 zero bytes anywhere */
    dst[0] = (uchar)ZLE7_TYPE_RAW;
    _mm512_mask_storeu_epi8( (void *)( dst+1 ), km, v );
    return len+1UL;
  }

  if( FD_UNLIKELY( d==_bzhi_u64( ~0UL, (uint)(len-1UL) ) ) ) { /* fill: all zero */
    ulong n = len-1UL; /* <=63, guaranteed by len<=64 */
    dst[0] = (uchar)( ZLE7_TYPE_FILL | (n<<2) );
    if( FD_UNLIKELY( n==63UL ) ) { dst[1] = 0U; return 2UL; } /* varint(0) */
    return 1UL;
  }

  uchar   tok[ 128 ];
  ulong   tpos = 128UL;
  ulong   base = 0UL;
  uchar * out  = dst+3;

  for(;;) {
    ulong start = (ulong)__builtin_ctzll( d );
    ulong lo    = (ulong)_blsi_u64( d );
    d += lo;
    ulong end = (ulong)__builtin_ctzll( d ) + 1UL;
    d = (ulong)_blsr_u64( d );

    ulong n = start<len ? start : len;
    ulong L = n-base;
    zle7_copy_le64( out, src+base, L ); out += L;

    base = end;
    tpos -= 2UL;
    tok[ tpos   ] = (uchar)( end-n ); /* Z -- lower address (fix 4) */
    tok[ tpos+1 ] = (uchar)L;         /* L -- higher address (fix 4) */
    if( FD_LIKELY( d!=0UL ) ) continue; /* fix 2 */
    break;
  }

  ulong L = len>=base ? len-base : 0UL;
  zle7_copy_le64( out, src+base, L ); out += L; /* fix 3 */

  tpos -= 1UL;
  tok[ tpos ] = (uchar)L; /* trailing lone L, no matching Z */

  dst[0] = (uchar)ZLE7_TYPE_LAST;
  dst[1] = tok[ 127 ]; /* L0 -- higher address of the first pair */
  dst[2] = tok[ 126 ]; /* Z0 -- lower address of the first pair */

  ulong n2 = 126UL-tpos;
  zle7_copy_le64( out, tok+tpos, n2 ); out += n2;
  return (ulong)( out-dst );
}

/* ==================================================================
   zle7_compress_more -- the type.more, data_sz>64 path.  Same scan
   primitives as fd_zle5/6 (see their comments for the derivation);
   only the emit side changed, to fd_zle_pseudocode2's independently
   delta-escaping (L, Z) token instead of fd_zle6's fixed-u32 one, atop
   a fixed-width 3-byte header instead of fd_zle6's variable one. */

/* zle7_copy -- general copy, sz<=n.  Unlike zle7_copy_le64 (the small
   encoder's only copy, always <=64B), zle7_compress_more's literal
   runs and its final token-stream slide can both be arbitrarily large,
   so this needs a >64B tier too.

   The (64,256] tier is 4 unconditional loads+stores at byte offsets
   {0, mid_a, mid_b, tail}, no branches on sz beyond the one that
   selects this tier.  tail=sz-64 is always in range (sz>64 here);
   mid_a/mid_b clamp to tail, so for sz<=128 (only one interior word
   actually needed) they degenerate to a harmless repeat of the tail
   store, and for sz<=192 mid_b does too -- every offset satisfies
   off<=sz-64 by construction, so this never touches a byte at or past
   d+sz.  (fd_zle_pseudocode2's own zle_copy/zle_zero sketch this same
   4-store shape but with an offset expression -- (len/2)*64 -- that is
   dimensionally wrong for len in byte units, i.e. buggy as literally
   written; this derives the same intent correctly instead of
   transcribing it.) */

static inline void
zle7_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  if( FD_LIKELY( sz<=64UL ) ) {
    __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
    _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
    return;
  }
  if( FD_UNLIKELY( sz>256UL ) ) { fd_memcpy( d, s, sz ); return; }

  ulong tail  = sz-64UL;
  ulong mid_a = fd_ulong_min(  64UL, tail );
  ulong mid_b = fd_ulong_min( 128UL, tail );
  wwb_stu( d,        wwb_ldu( s        ) );
  wwb_stu( d+mid_a,  wwb_ldu( s+mid_a  ) );
  wwb_stu( d+mid_b,  wwb_ldu( s+mid_b  ) );
  wwb_stu( d+tail,   wwb_ldu( s+tail   ) );
}

/* zle7_zmask returns the zero-byte mask of the 64-byte word at s+p,
   for p<n (p need not be 64-aligned, but every call site here passes
   an absolute byte position that is).  Bits at positions >=n read as
   non-zero, so runs terminate at n. */

static inline ulong
zle7_zmask( uchar const * s,
            ulong         p,
            ulong         n ) {
  ulong m = wwb_eq( wwb_ldu( s+p ), wwb_zero() );
  if( FD_LIKELY( p+64UL<=n ) ) return m;
  if( p>=n                  ) return 0UL;
  return m & ( ~0UL>>( 64UL-( n-p ) ) );
}

/* zle7_cand marks run-start candidates (byte p+i and p+i+1 both zero)
   in the word at s+p, same derivation as fd_zle5/6's. */

static inline ulong
zle7_cand( uchar const * s,
           ulong         p,
           ulong         n ) {
  if( FD_LIKELY( p+64UL<n ) ) {
    wwb_t v = wwb_or( wwb_ldu( s+p ), wwb_ldu( s+p+1UL ) );
    return (ulong)_mm512_testn_epi8_mask( v, v );
  }
  ulong m = zle7_zmask( s, p, n );
  return m & ( m>>1 );
}

/* zle7_spill extends a zero run past the end of the word at *pp,
   returning the run's total length and leaving *pp / *pm at the word
   where it ended (or *pp>=n, *pm=0 if it ran off the end of the
   input).  *pp is an absolute byte position throughout -- no separate
   word-count bound needed, "*pp>=n" is the only check, in byte units
   directly instead of via a precomputed `words`. */

static inline ulong
zle7_spill( uchar const * s,
            ulong         n,
            ulong *       pp,
            ulong *       pm,
            ulong         len ) {
  ulong p = *pp+64UL;
  for(;;) {
    if( p>=n ) { *pp = p; *pm = 0UL; return len; }

    ulong t = zle7_zmask( s, p, n );
    if( FD_UNLIKELY( t!=~0UL ) ) {
      ulong ext = (ulong)__builtin_ctzll( ~t );
      *pp = p;
      *pm = t & ~( ( 1UL<<ext )-1UL );
      return len+ext;
    }

    p += 64UL; len += 64UL;
    while( p+256UL<=n ) {
      uchar const * q = s+p;
      wwb_t v = wwb_or( wwb_or( wwb_ldu( q       ), wwb_ldu( q+ 64UL ) ),
                        wwb_or( wwb_ldu( q+128UL ), wwb_ldu( q+192UL ) ) );
      if( FD_UNLIKELY( _mm512_test_epi8_mask( v, v ) ) ) break;
      p += 256UL; len += 256UL;
    }
  }
}

/* zle7_finish_more lays out the type.more header around a literal
   plane of Lsz bytes already at dst+ZLE7_MORE_HDR and a token stream
   of C bytes at ctrl (zle7_compress_more's provisional placement past
   the literal plane's worst-case extent, dst+ZLE7_MORE_HDR+n).  Unlike
   zle6_finish, the header here is always exactly ZLE7_MORE_HDR bytes --
   there is no header-width branch and therefore no literal-plane
   shift -- so this only ever moves the (much smaller) token stream.
   Every call site into this function got here through a bailout check
   that keeps Lsz+C a safety margin below n at every step (see
   zle7_compress_more), which is what makes the copy below provably
   non-overlapping: dst+ZLE7_MORE_HDR+Lsz+C < dst+ZLE7_MORE_HDR+n == ctrl. */

static inline ulong
zle7_finish_more( uchar *       FD_RESTRICT dst,
                   uchar const * FD_RESTRICT src,
                   ulong                     n,
                   ulong                     Lsz,
                   uchar const * FD_RESTRICT ctrl,
                   ulong                     C ) {
  if( FD_UNLIKELY( Lsz>ZLE7_SIZE_MAX || ZLE7_MORE_HDR+Lsz+C>=n+1UL ) ) return zle7_raw( dst, src, n );
  zle7_copy( dst+ZLE7_MORE_HDR+Lsz, ctrl, C );
  dst[0] = (uchar)( ZLE7_TYPE_MORE | ( ( Lsz & 0x3FUL )<<2 ) );
  dst[1] = (uchar)( Lsz>>6  );
  dst[2] = (uchar)( Lsz>>14 );
  return ZLE7_MORE_HDR+Lsz+C;
}

/* zle7_compress_more handles n>64.  Layout: literal plane grows up
   from dst+ZLE7_MORE_HDR (its final position -- it never moves), the
   token stream grows up provisionally from dst+ZLE7_MORE_HDR+n (past
   anywhere the literal plane could reach) and is slid down to abut the
   literal plane once its final size is known.  Margins (13 mid-scan /
   15 against cpx) size for this token's larger worst case than
   fd_zle6's: two independently-escaping fields, each up to 1 base byte
   + a 5-byte varint (covers deltas up to 35 bits, comfortably above
   FD_ZLE7_MAX_SZ), i.e. 12 bytes worst case vs fd_zle6's fixed 10.

   Unlike fd_zle5/6, the scan below tracks one absolute byte cursor
   `p` throughout instead of a word index `w` plus a separately
   precomputed `words` bound: every helper above already takes/returns
   an absolute position, so there is no `w<<6` to redo at each use site
   (fd_zle5/6 do this at least half a dozen times per scan), and the
   end-of-input check is just `p>=n` instead of needing `words =
   (n+63)>>6` carried around as its own variable.

   The 256B skip-ahead also stops paying for its own redundancy: once
   `a0..a3` (the four words' adjacent-OR values) are already loaded
   into registers to answer "is there a candidate anywhere in this
   256B", finding *which* of the four holds it is four more
   `_mm512_testn_epi8_mask` calls on those same registers -- fd_zle5/6
   instead just `break` and let the caller's per-word `zmask`/`cand`
   loop re-discover the answer word by word, reloading and re-ORing
   bytes already sitting in a0..a3 (up to 3 of the 4 words redone, and
   whichever one first reports empty also silently re-triggers the
   *next* 256B group scan, which itself re-touches the tail of this
   one). */

static ulong __attribute__((noinline))
zle7_compress_more( uchar *       FD_RESTRICT dst,
                     uchar const * FD_RESTRICT src,
                     ulong                     n ) {
  uchar * lp0 = dst+ZLE7_MORE_HDR;
  uchar * lp  = lp0;
  uchar * cp0 = dst+ZLE7_MORE_HDR+n; /* provisional: literal plane can never exceed n bytes */
  uchar * cp  = cp0;
  uchar * cpx = dst+FD_ZLE7_COMPRESS_BOUND( n )-16UL; /* one zle7_emit writes <=12 bytes */

  ulong   fs  = 0UL;
  ulong   p   = 0UL;
  ulong   m   = 0UL;
  ulong   d   = zle7_cand( src, 0UL, n );

  for(;;) {

    if( !d ) {
      p += 64UL;
      for(;;) {
        if( FD_UNLIKELY( p>=n ) ) goto trailing;
        d = zle7_cand( src, p, n );
        if( d ) break;
        p += 64UL;

        while( p+257UL<=n ) {
          uchar const * q  = src+p;
          wwb_t a0 = wwb_or( wwb_ldu( q       ), wwb_ldu( q+  1UL ) );
          wwb_t a1 = wwb_or( wwb_ldu( q+ 64UL ), wwb_ldu( q+ 65UL ) );
          wwb_t a2 = wwb_or( wwb_ldu( q+128UL ), wwb_ldu( q+129UL ) );
          wwb_t a3 = wwb_or( wwb_ldu( q+192UL ), wwb_ldu( q+193UL ) );
          wwb_t t  = _mm512_min_epu8( _mm512_min_epu8( a0, a1 ), _mm512_min_epu8( a2, a3 ) );
          if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) {
            /* a candidate exists somewhere in a0..a3 (that's what `t`
               just proved) -- find which word directly from the
               registers already in hand, instead of breaking out to
               let the per-word loop above reload and re-OR them. */
            d = (ulong)_mm512_testn_epi8_mask( a0, a0 ); if( d ) goto search_done;
            p += 64UL; d = (ulong)_mm512_testn_epi8_mask( a1, a1 ); if( d ) goto search_done;
            p += 64UL; d = (ulong)_mm512_testn_epi8_mask( a2, a2 ); if( d ) goto search_done;
            p += 64UL; d = (ulong)_mm512_testn_epi8_mask( a3, a3 );
            goto search_done; /* must be nonzero: a0..a2 already ruled out, t proved one of the four has it */
          }
          p += 256UL;
        }
      }
    }
    search_done:;

    /* see the ze/len derivation added to fd_zle_big -- ze is the run's
       absolute end position, computed directly instead of via a len
       that would just get added back to zs (their common `b`/`p`
       terms cancel). */
    ulong lo = (ulong)_blsi_u64( d );
    ulong b  = (ulong)__builtin_ctzll( d );
    ulong zs = p+b;
    ulong x  = d + lo;
    ulong len, ze;
    if( FD_LIKELY( x ) ) {
      ze  = p+(ulong)__builtin_ctzll( x )+1UL;
      len = ze-zs;
      d   = (ulong)_blsr_u64( x );
    } else {
      len = zle7_spill( src, n, &p, &m, 64UL-b );
      ze  = zs+len;
      ulong nn = ( p+64UL<n ) ? ( zle7_zmask( src, p+64UL, n ) & 1UL ) : 0UL;
      d = m & ( ( m>>1 ) | ( nn<<63 ) );
    }

    if( FD_LIKELY( len>=ZLE7_MIN_Z ) ) {
      ulong L = zs-fs;
      if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+13UL>=n || cp+15UL>=cpx ) )
        return zle7_raw( dst, src, n );
      zle7_copy( lp, src+fs, L ); lp += L;
      cp = zle7_emit( cp, L, len );
      fs = ze;
    }
  }

trailing:
  if( fs<n ) {
    ulong L = n-fs;
    if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+13UL>=n || cp+15UL>=cpx ) )
      return zle7_raw( dst, src, n );
    zle7_copy( lp, src+fs, L ); lp += L;
    cp = zle7_emit_l( cp, L );
  }

  return zle7_finish_more( dst, src, n, (ulong)( lp-lp0 ), cp0, (ulong)( cp-cp0 ) );
}

#else /* !FD_HAS_AVX512 -- Firedancer requires AVX-512; this stub only
         exists so the translation unit still builds elsewhere. */

static ulong
zle7_compress_small( uchar *       FD_RESTRICT dst,
                      uchar const * FD_RESTRICT src,
                      ulong                     len ) {
  dst[0] = (uchar)ZLE7_TYPE_RAW;
  fd_memcpy( dst+1, src, len );
  return len+1UL;
}

static ulong
zle7_compress_more( uchar *       FD_RESTRICT dst,
                     uchar const * FD_RESTRICT src,
                     ulong                     n ) {
  return zle7_raw( dst, src, n );
}

#endif /* FD_HAS_AVX512 */

ulong
fd_zle7_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz      ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<=64UL ) ) return zle7_compress_small( dst, src, data_sz );
  return                                    zle7_compress_more ( dst, src, data_sz );
}

/* ==================================================================
   Decoder -- plain scalar, not on the performance-critical path this
   prototype is actually about yet. */

long
fd_zle7_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       d = (uchar *)data;
  uchar const * s = (uchar const *)comp;

  if( FD_UNLIKELY( comp_sz<1UL ) ) return data_sz==0UL ? 0L : FD_ZLE7_ERR_CORRUPT;

  uchar b0   = s[0];
  ulong type = (ulong)b0 & 0x3UL;
  ulong ext  = (ulong)b0 >> 2;

  switch( type ) {

  case ZLE7_TYPE_RAW: {
    if( FD_UNLIKELY( 1UL+ext>comp_sz ) ) return FD_ZLE7_ERR_CORRUPT;
    ulong n = comp_sz-1UL-ext;
    if( FD_UNLIKELY( n!=data_sz ) ) return FD_ZLE7_ERR_CORRUPT;
    fd_memcpy( d, s+1UL+ext, n );
    return (long)n;
  }

  case ZLE7_TYPE_FILL: {
    ulong n   = ext;
    ulong hdr = 1UL;
    if( n==63UL ) {
      ulong i = 1UL, ex;
      if( FD_UNLIKELY( !zle7_vget( s, comp_sz, &i, &ex ) ) ) return FD_ZLE7_ERR_CORRUPT;
      n   = 63UL+ex;
      hdr = i;
    }
    n += 1UL;
    if( FD_UNLIKELY( hdr!=comp_sz || n!=data_sz ) ) return FD_ZLE7_ERR_CORRUPT;
    fd_memset( d, 0, n );
    return (long)n;
  }

  case ZLE7_TYPE_LAST: {
    if( FD_UNLIKELY( comp_sz<3UL ) ) return FD_ZLE7_ERR_CORRUPT;
    if( FD_UNLIKELY( s[1]==(uchar)ZLE7_TOK_ESC || s[2]==(uchar)ZLE7_TOK_ESC ) )
      return FD_ZLE7_ERR_CORRUPT; /* escape unreachable from this encoder -- see header comment */

    uchar const * lp    = s+3UL;
    uchar *       o     = d;
    uchar *       oend  = d+data_sz;
    ulong         bp    = comp_sz;     /* backward cursor: next control byte is s[bp-1] */
    ulong         avail = comp_sz-3UL; /* bytes between lp and bp claimed by neither
                                           cursor yet -- shared budget: every literal
                                           byte lp takes and every control byte bp
                                           takes comes out of this same pool, so one
                                           counter (kept as bp-(lp-s), maintained
                                           incrementally instead of recomputed from
                                           the two pointers) is all either side needs
                                           to check against. */

    ulong L0 = s[1], Z0 = s[2];
    if( FD_UNLIKELY( L0>avail || L0>(ulong)(oend-o) ) ) return FD_ZLE7_ERR_CORRUPT;
    fd_memcpy( o, lp, L0 ); o += L0; lp += L0; avail -= L0;
    if( FD_UNLIKELY( Z0>(ulong)(oend-o) ) ) return FD_ZLE7_ERR_CORRUPT;
    fd_memset( o, 0, Z0 ); o += Z0;

    for(;;) {
      if( FD_UNLIKELY( !avail ) ) return FD_ZLE7_ERR_CORRUPT;
      bp -= 1UL; avail -= 1UL;
      ulong L = s[bp];
      if( FD_UNLIKELY( L==ZLE7_TOK_ESC || L>avail || L>(ulong)(oend-o) ) ) return FD_ZLE7_ERR_CORRUPT;
      fd_memcpy( o, lp, L ); o += L; lp += L; avail -= L;
      if( o==oend ) break; /* terminal lone-L field -- see wire format comment */

      if( FD_UNLIKELY( !avail ) ) return FD_ZLE7_ERR_CORRUPT;
      bp -= 1UL; avail -= 1UL;
      ulong Z = s[bp];
      if( FD_UNLIKELY( Z==ZLE7_TOK_ESC || Z>(ulong)(oend-o) ) ) return FD_ZLE7_ERR_CORRUPT;
      fd_memset( o, 0, Z ); o += Z;
    }

    /* full consumption: the gap between the two cursors must be
       exactly zero -- no unconsumed control bytes, no literal bytes
       the token stream never claimed. */
    if( FD_UNLIKELY( avail!=0UL ) ) return FD_ZLE7_ERR_CORRUPT;
    return (long)data_sz;
  }

  case ZLE7_TYPE_MORE: {
    if( FD_UNLIKELY( comp_sz<ZLE7_MORE_HDR ) ) return FD_ZLE7_ERR_CORRUPT;
    ulong size_ext = (ulong)s[1] | ( (ulong)s[2]<<8 );
    ulong Lsz      = ( size_ext<<6 ) | ext;
    if( FD_UNLIKELY( ZLE7_MORE_HDR+Lsz>comp_sz ) ) return FD_ZLE7_ERR_CORRUPT;

    uchar const * lp    = s+ZLE7_MORE_HDR;
    uchar const * lpend = lp+Lsz;
    uchar *       o     = d;
    uchar *       oend  = d+data_sz;
    ulong         i     = (ulong)( lpend-s ); /* token-stream cursor, byte offset into s */

    while( o<oend ) {
      ulong ll;
      if( FD_UNLIKELY( !zle7_tok_get( s, comp_sz, &i, &ll ) ) ) return FD_ZLE7_ERR_CORRUPT;
      if( FD_UNLIKELY( ll>(ulong)(lpend-lp) || ll>(ulong)(oend-o) ) ) return FD_ZLE7_ERR_CORRUPT;
      fd_memcpy( o, lp, ll ); o += ll; lp += ll;
      if( o==oend ) break; /* terminal token: no Z field -- same rule as type.last */

      ulong zl;
      if( FD_UNLIKELY( !zle7_tok_get( s, comp_sz, &i, &zl ) ) ) return FD_ZLE7_ERR_CORRUPT;
      if( FD_UNLIKELY( zl>(ulong)(oend-o) ) ) return FD_ZLE7_ERR_CORRUPT;
      fd_memset( o, 0, zl ); o += zl;
    }

    /* every literal byte consumed, every output byte produced, and no
       unconsumed trailing garbage in the token stream */
    if( FD_UNLIKELY( (lp!=lpend) | (i!=comp_sz) ) ) return FD_ZLE7_ERR_CORRUPT;
    return (long)data_sz;
  }

  default: return FD_ZLE7_ERR_CORRUPT; /* unreachable: type is 2 bits, all 4 values handled above */
  }
}

char const *
fd_zle7_strerror( long res ) {
  switch( res ) {
  case FD_ZLE7_ERR_SPACE:   return "out of buffer space";
  case FD_ZLE7_ERR_CORRUPT: return "malformed compressed data";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
