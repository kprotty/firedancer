#include "fd_zle6.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format
   ===========

   This is `fd_zle_pseudocode`'s "sequence of frames" idea, realised as
   a single frame per fd_zle6_compress() call (fd_zle5's planar layout
   already is one frame per call, so this keeps that and only changes
   what a *token* looks like), with a header borrowed from fd_zle5's
   own mode+c0 byte instead of a flat u32:

     [0]              u8   b0        bit 7: 0 = zle, 1 = stored
                                      bits 6..0: c0 (see below)

     b0 bit 7 == 1  ->  stored (the incompressible / tiny fallback):
       [1 .. 1+n)              raw bytes                 comp_sz = n + 1

     b0 bit 7 == 0  ->  zle:
       c0 = b0 & 0x7F
       c0 != 0x7F:  hdr = 1,  L = c0                      (L in 0..126)
       c0 == 0x7F:  hdr = 4,  L = u24LE at [1 .. 4)        (L in 0..2^24-1)
       [hdr .. hdr+L)          literal plane   (every literal byte, packed)
       [hdr+L .. comp_sz)      token stream    (see below)

   The point of folding the stored/zle flag into the same byte as the
   size, rather than a separate mode byte (fd_zle5) or a flat u32
   (this file's first cut), is that the *overwhelmingly common* case --
   a literal plane under 127 bytes, i.e. every account this codec
   actually helps with, `spl-token account`'s ~615M/1.08B included --
   costs a 1-byte header, cheaper than fd_zle5's 2 and this file's
   original always-4.  The worst case (L >= 127) still costs 4, same as
   before.  See "fd_zle6: independently-escaping tokens" and its
   follow-up in ZLE_TECHNIQUES.md for the numbers this was a response
   to: fd_zle6's first cut had a ratio *worse* than both fd_zle and
   fd_zle4/5 because the flat 4-byte header was disproportionately
   expensive on exactly the small-account band that dominates the
   workload.

   L is capped at 2^24-1 (~16.7 MiB) by the 3-byte extension -- comfortably
   above any real Solana account (the protocol's own MAX_PERMITTED_DATA_LENGTH
   is 10 MiB) and above every literal plane this codec has ever produced in
   `bench_zle6`'s multi-MiB `tail`/`uniform` shapes, but a real, lower ceiling
   than fd_zle5's ~4 GiB.  A frame whose literal plane would not fit falls
   back to a stored frame -- correct, just not optimally compressed, and only
   reachable with an adversarially incompressible multi-megabyte input.

   Token stream
   ------------
   A sequence of (L, Z) tokens, each describing one (literal run, zero
   run) pair in input order, consumed until the decoder has produced
   data_sz output bytes -- there is no explicit end marker or count,
   `data_sz` (passed out of band, exactly as fd_zle/fd_zle5) is what
   tells the decoder when to stop, same as it already tells it how much
   to decode.

     L  field:  u8 ll            ll < 0xFF
                or 0xFF, u32LE   ll >= 0xFF
     [ if this token's L brought total output to data_sz: stop here,
       no Z field -- this is the last token in the stream ]
     Z  field:  u8 zb            zl = zb + 2       zb < 0xFF
                or 0xFF, u32LE   zl = ext + 2

   L and Z escape *independently* -- this is the actual change from
   fd_zle5's control frame, which escapes both fields together
   (`0xFF, u32 ll, u32 zl`) as soon as either one is >= 0xFF.  A run
   with one huge field and one small one (plausible: a long literal
   span followed by a short zero run, or vice versa) costs 6 bytes here
   (1 + 4 + 1) instead of fd_zle5's 9; the cost is a 10-byte worst case
   (both fields escape) instead of fd_zle5's 9, and a second
   predicted-not-taken branch per token.

   Z's non-escaped byte and its escape extension both carry a uniform
   +2 bias (`fd_zle_pseudocode`'s "Z += 2, the minimum zero-run match"):
   a run shorter than ZLE6_MIN_Z==2 is never framed (it would cost as
   many control bytes as it saves), so the wire value 0 would otherwise
   go unused -- spending it on "actual length 2" instead means a
   non-escaped Z byte covers 2..256 rather than 0..254.

   This is also what removes the terminal-frame ambiguity the +2 bias
   would otherwise create: a token can never *represent* Z==0 (0 always
   means "actual 2"), so a trailing literal run with no zero run after
   it (end of input, or the whole input never framed a run) is encoded
   as an L field with **no Z field at all**.  The decoder already knows
   when it has produced data_sz bytes, so it can tell -- right after
   copying that L field's literal bytes -- whether a Z field exists
   without any wire marker: if output is complete, stop; otherwise a
   run must follow (there is more output to produce and no more
   literal budget to produce it from), so read one.

   Two other deliberate deviations from `fd_zle_pseudocode`'s literal
   text, both forced by the same kind of gap (the pseudocode has "..."
   where the real body would go, so these corners were never pinned
   down there):

   - The pseudocode's frame header is "size:u32 ... if size=0, all
     literals are copied to dst and stream ends" -- i.e. size==0 double
     as the stored-fallback sentinel.  But size==0 is also the *only*
     legitimate encoding of a frame whose entire input is one zero run
     (an all-zero input has L==0), so that sentinel collides with a
     real, benchmarked case (bench_zle's "all-zero N" shapes) and would
     force them through the incompressible-sized stored path.  The
     stored flag is a dedicated bit instead -- L==0 decodes normally in
     either header width.

   - Ties into the point above: with the +2 bias applying to *every*
     Z, per the pseudocode text, there is no token encoding for "zero
     zero-length", so a frame that ends on a literal run cannot use a
     (L, Z=0) token the way fd_zle5's (ll, zl=0) trailing frame does.
     Solved by the terminal-token rule described above instead: the
     last token in a stream may omit Z entirely, which needs no wire
     bit either -- the decoder infers it from data_sz.

   Everything else -- the zero mask, the run-start candidate mask and
   its carry-propagate resolution, the 256-byte skip-aheads in both
   directions, the tiny/small/big size dispatch, and the
   literal-plane-then-token-stream physical layout (fd_zle6_big's token
   stream grows provisionally past the literal plane's worst-case
   extent and is slid down to abut it once its final size is known) --
   is fd_zle5's, unchanged.  See ZLE4_DESIGN.md for the full rationale,
   and zle6_finish below for how the *header* -- unlike fd_zle5's, which
   this originally copied verbatim -- gets to 1 byte in the common case
   without paying for it in the encode hot path. */

#define ZLE6_STORED_BIT (0x80U)      /* b0 bit 7: stored frame follows */
#define ZLE6_C0_ESC     (0x7FUL)    /* b0 bits 6..0 == 0x7F: L escapes to 3 more bytes */
#define ZLE6_HDR_MIN    (1UL)       /* header size when c0 != 0x7F */
#define ZLE6_HDR_MAX    (4UL)       /* header size when c0 == 0x7F */
#define ZLE6_L_MAX      (0xFFFFFFUL) /* largest L the 3-byte extension can hold (2^24-1) */
#define ZLE6_MIN_Z      (2UL)       /* shortest zero run the encoder will frame */

FD_STATIC_ASSERT( ZLE6_MIN_Z==2UL, unit );

/* zle6_emit_l writes just an L field (a terminal token, no zero run
   after it) at cp and returns the advanced cursor. */

static inline uchar *
zle6_emit_l( uchar * cp,
             ulong   ll ) {
  if( FD_LIKELY( ll<0xFFUL ) ) { cp[0] = (uchar)ll; return cp+1UL; }
  cp[0] = 0xFF;
  cp[1] = (uchar)( ll       ); cp[2] = (uchar)( ll>> 8 );
  cp[3] = (uchar)( ll>>16   ); cp[4] = (uchar)( ll>>24 );
  return cp+5UL;
}

/* zle6_emit writes a full (L, Z) token -- zl must be >= ZLE6_MIN_Z --
   at cp and returns the advanced cursor.  L and Z escape independently
   (see the wire format comment above); Z carries the +2 bias both in
   its direct byte and its escape extension. */

static inline uchar *
zle6_emit( uchar * cp,
           ulong   ll,
           ulong   zl ) {
  cp = zle6_emit_l( cp, ll );
  ulong zb = zl-2UL;
  if( FD_LIKELY( zb<0xFFUL ) ) { cp[0] = (uchar)zb; return cp+1UL; }
  cp[0] = 0xFF;
  cp[1] = (uchar)( zb       ); cp[2] = (uchar)( zb>> 8 );
  cp[3] = (uchar)( zb>>16   ); cp[4] = (uchar)( zb>>24 );
  return cp+5UL;
}

/* fd_zle6_stored writes the raw fallback frame: the 1-byte stored
   header followed by a verbatim copy of the input.  comp_sz = n + 1. */

static inline ulong
fd_zle6_stored( uchar *       FD_RESTRICT dst,
                uchar const * FD_RESTRICT src,
                ulong                     n ) {
  dst[0] = (uchar)ZLE6_STORED_BIT;
  fd_memcpy( dst+1UL, src, n );
  return n+1UL;
}

#if FD_HAS_AVX512

/* ==================================================================
   Scan primitives -- identical to fd_zle5's; see fd_zle5.c for the
   derivation comments.
   ================================================================== */

static inline ulong
zle6_zmask( uchar const * s,
            ulong         w,
            ulong         n ) {
  ulong off = w<<6;
  ulong m   = wwb_eq( wwb_ldu( s+off ), wwb_zero() );
  if( FD_LIKELY( off+64UL<=n ) ) return m;
  if( off>=n                  ) return 0UL;
  return m & ( ~0UL>>( 64UL-( n-off ) ) );
}

static inline ulong
zle6_cand( uchar const * s,
           ulong         w,
           ulong         n ) {
  ulong off = w<<6;
  if( FD_LIKELY( off+64UL<n ) ) {
    wwb_t p = wwb_or( wwb_ldu( s+off ), wwb_ldu( s+off+1UL ) );
    return (ulong)_mm512_testn_epi8_mask( p, p );
  }
  ulong m = zle6_zmask( s, w, n );
  return m & ( m>>1 );
}

/* zle6_copy / zle6_zero -- straight-line, no loop, same memcpy-fold
   avoidance as fd_zle5_copy / fd_zle5_zero (see fd_zle5.c). */

static inline void
zle6_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  if( FD_LIKELY( sz<=64UL ) ) {
    __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
    _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
    return;
  }
  if( FD_UNLIKELY( sz>256UL ) ) { fd_memcpy( d, s, sz ); return; }

  wwb_stu( d, wwb_ldu( s ) );
  if( sz>128UL ) {
    wwb_stu( d+64UL, wwb_ldu( s+64UL ) );
    if( sz>192UL ) wwb_stu( d+128UL, wwb_ldu( s+128UL ) );
  }
  wwb_stu( d+sz-64UL, wwb_ldu( s+sz-64UL ) );
}

static inline void
zle6_zero( uchar * d,
           ulong   sz ) {
  wwb_t z = wwb_zero();

  if( FD_LIKELY( sz<=64UL ) ) {
    _mm512_mask_storeu_epi8( (void *)d, _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ), z );
    return;
  }
  if( FD_LIKELY( sz<=256UL ) ) {
    wwb_stu( d, z );
    if( sz>128UL ) {
      wwb_stu( d+64UL, z );
      if( sz>192UL ) wwb_stu( d+128UL, z );
    }
    wwb_stu( d+sz-64UL, z );
    return;
  }
  if( FD_LIKELY( sz<512UL ) ) {
    ulong k = 0UL;
    do { wwb_stu( d+k, z ); k += 64UL; } while( k+64UL<=sz );
    wwb_stu( d+sz-64UL, z );
    return;
  }
  fd_memset( d, 0, sz );
}

static inline ulong
zle6_spill( uchar const * s,
            ulong         n,
            ulong         words,
            ulong *       pw,
            ulong *       pm,
            ulong         len ) {
  ulong w = *pw+1UL;
  for(;;) {
    if( w>=words ) { *pw = w; *pm = 0UL; return len; }

    ulong t = zle6_zmask( s, w, n );
    if( FD_UNLIKELY( t!=~0UL ) ) {
      ulong ext = (ulong)__builtin_ctzll( ~t );
      *pw = w;
      *pm = t & ~( ( 1UL<<ext )-1UL );
      return len+ext;
    }

    w++; len += 64UL;
    while( ( w<<6 )+256UL<=n ) {
      uchar const * p = s+( w<<6 );
      wwb_t v = wwb_or( wwb_or( wwb_ldu( p       ), wwb_ldu( p+ 64UL ) ),
                        wwb_or( wwb_ldu( p+128UL ), wwb_ldu( p+192UL ) ) );
      if( FD_UNLIKELY( _mm512_test_epi8_mask( v, v ) ) ) break;
      w += 4UL; len += 256UL;
    }
  }
}

/* zle6_finish lays out the smallest header this format allows around a
   literal plane of Lsz bytes already sitting at dst+ZLE6_HDR_MIN (every
   tier writes literal bytes there during the scan, optimistically
   assuming the common 1-byte header -- see the tier functions below)
   and a control stream of C bytes at ctrl (the small tiers' stack
   scratch, or fd_zle6_big's placeholder past the literal plane's
   worst-case extent).  Returns the final comp_sz, or falls back to a
   stored frame.

   The optimistic placement is deliberate, not incidental: a first cut
   of this function always wrote the literal plane at the *worst-case*
   dst+ZLE6_HDR_MAX and shifted it left by 3 whenever the common (< 0x7F)
   case applied -- i.e. it paid a shift on the branch that fires for
   almost every small account (spl-token's ~615M/1.08B included) to
   avoid ever shifting the rare one.  That measured 15.7-15.9 GB/s mix
   encode, an 11% loss from before this header change, regardless of
   whether the shift itself was a libc memmove or the hand-rolled
   zle6_shift3 below -- the extra pass over the whole (literal,
   control) block was the cost, not the call.  Placing at dst+ZLE6_HDR_MIN
   up front instead makes the *common* case free (no shift at all) and
   only the escape case (Lsz >= 0x7F) pays a shift, and for the tiny
   tier (n < 64, so Lsz < 64 always) that escape case is simply
   unreachable -- the tier that dominates account count never shifts. */

/* zle6_shift3r right-shifts sz bytes at d into d+3, in place -- used
   only on the rare (Lsz >= 0x7F) path, to open up the extra 3 header
   bytes the escape needs.  Processed high-to-low: writing chunk w's
   bytes to [d+w+3, d+w+67) happens strictly before any not-yet-read
   byte at address < d+w+3 is touched (every already-written region
   sits at or above the next chunk's read range plus 3), so this is
   correct for exactly this fixed 3-byte shift without memmove's
   general-overlap machinery, and stays inlined. */

static inline void
zle6_shift3r( uchar * d,
              ulong   sz ) {
  ulong off = sz;
  while( off>=64UL ) { off -= 64UL; wwb_stu( d+off+3UL, wwb_ldu( d+off ) ); }
  if( FD_LIKELY( off ) ) {
    __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)off ) );
    _mm512_mask_storeu_epi8( (void *)( d+3UL ), mm, _mm512_maskz_loadu_epi8( mm, d ) );
  }
}

static inline ulong
zle6_finish( uchar *       FD_RESTRICT dst,
             uchar const * FD_RESTRICT src,
             ulong                     n,
             ulong                     Lsz,
             uchar const * FD_RESTRICT ctrl,
             ulong                     C ) {
  if( FD_LIKELY( Lsz<ZLE6_C0_ESC ) ) {
    if( FD_UNLIKELY( ZLE6_HDR_MIN+Lsz+C>=n+1UL ) ) return fd_zle6_stored( dst, src, n );
    zle6_copy( dst+ZLE6_HDR_MIN+Lsz, ctrl, C ); /* literal plane already at dst+1..dst+1+Lsz */
    dst[0] = (uchar)Lsz;
    return ZLE6_HDR_MIN+Lsz+C;
  }

  if( FD_UNLIKELY( Lsz>ZLE6_L_MAX || ZLE6_HDR_MAX+Lsz+C>=n+1UL ) ) return fd_zle6_stored( dst, src, n );
  if( FD_LIKELY( Lsz ) ) zle6_shift3r( dst+ZLE6_HDR_MIN, Lsz ); /* dst+1..+Lsz -> dst+4..+Lsz */
  zle6_copy( dst+ZLE6_HDR_MAX+Lsz, ctrl, C );
  dst[0] = (uchar)ZLE6_C0_ESC;
  dst[1] = (uchar)( Lsz     ); dst[2] = (uchar)( Lsz>>8 ); dst[3] = (uchar)( Lsz>>16 );
  return ZLE6_HDR_MAX+Lsz+C;
}

/* ==================================================================
   Encoder
   ================================================================== */

#define ZLE6_EXT( mm ) do {                                            \
    ulong _e = ~(mm) ? (ulong)__builtin_ctzll( ~(mm) ) : 64UL;         \
    len  += _e;                                                        \
    (mm)  = _e<64UL ? ( (mm) & ~( ( 1UL<<_e )-1UL ) ) : 0UL;           \
    full  = _e==64UL;                                                  \
  } while(0)

/* ZLE6_WALK -- identical shape to fd_zle5's ZLE5_WALK, but appends a
   full (L, Z) token via zle6_emit instead of a fixed 2-byte frame. */

#define ZLE6_WALK( m, wb, EXT ) do {                                   \
    while( m ) {                                                       \
      ulong b   = (ulong)__builtin_ctzll( m );                         \
      ulong zs  = (wb)+b;                                              \
      ulong hi  = (m)>>b;                                              \
      ulong len;                                                       \
      ulong full FD_FN_UNUSED = 0UL;                                   \
      if( ~hi ) {                                                      \
        len = (ulong)__builtin_ctzll( ~hi );                          \
        (m) &= ~( ( ( 1UL<<len )-1UL )<<b );                           \
      } else {                                                         \
        len = 64UL-b;                                                  \
        (m) = 0UL;                                                     \
      }                                                                \
      if( b+len==64UL ) { EXT; }                                       \
      if( len>=ZLE6_MIN_Z ) {                                          \
        ulong _L = zs-fs;                                              \
        zle6_copy( lp, src+fs, _L ); lp += _L;                         \
        cp = zle6_emit( cp, _L, len );                                 \
        fs = zs+len;                                                   \
      }                                                                \
    }                                                                  \
  } while(0)

/* fd_zle6_tiny handles 1 <= n < 64, i.e. always exactly one mask word
   that never fills (bit 63 is never a run boundary that needs an EXT).
   The pseudocode's leanest tier: no word-count switch, no EXT chain --
   this is the size band ZLE4_DESIGN.md's tiny-account regression (see
   its 5.4) came from, so it gets the least overhead fd_zle6 can give
   it. */

static ulong
fd_zle6_tiny( uchar *       FD_RESTRICT dst,
              uchar const * FD_RESTRICT src,
              ulong                     n ) {
  ulong m0 = wwb_eq( wwb_ldu( src ), wwb_zero() ) & ( ~0UL>>( 64UL-n ) );

  uchar   ctrl[ 160 ];
  uchar * cp  = ctrl;
  uchar * lp0 = dst+ZLE6_HDR_MIN;
  uchar * lp  = lp0;
  ulong   fs  = 0UL;

  ZLE6_WALK( m0, 0UL, (void)0 );

  if( fs<n ) {
    ulong L = n-fs;
    zle6_copy( lp, src+fs, L ); lp += L;
    cp = zle6_emit_l( cp, L );
  }

  return zle6_finish( dst, src, n, (ulong)( lp-lp0 ), ctrl, (ulong)( cp-ctrl ) );
}

/* fd_zle6_small handles 64 <= n < 256, i.e. 1..4 mask words -- same
   straight-line switch(words) shape as fd_zle5_small. */

static ulong
fd_zle6_small( uchar *       FD_RESTRICT dst,
               uchar const * FD_RESTRICT src,
               ulong                     n ) {
  ulong words = ( n+63UL )>>6;

  ulong m0=wwb_eq( wwb_ldu( src ), wwb_zero() ), m1=0UL, m2=0UL, m3=0UL;
  if( words>1UL ) m1 = wwb_eq( wwb_ldu( src+ 64UL ), wwb_zero() );
  if( words>2UL ) m2 = wwb_eq( wwb_ldu( src+128UL ), wwb_zero() );
  if( words>3UL ) m3 = wwb_eq( wwb_ldu( src+192UL ), wwb_zero() );
  ulong tail  = n & 63UL; if( !tail ) tail = 64UL;
  ulong tmask = ~0UL>>( 64UL-tail );
  switch( words ) {
    case 1UL: m0 &= tmask; break;
    case 2UL: m1 &= tmask; break;
    case 3UL: m2 &= tmask; break;
    default:  m3 &= tmask; break;
  }

  uchar   ctrl[ 512 ];       /* control stream scratch */
  uchar * cp   = ctrl;
  uchar * lp0  = dst+ZLE6_HDR_MIN;
  uchar * lp   = lp0;
  ulong   fs   = 0UL;

  switch( words ) {
    case 1UL:
      ZLE6_WALK( m0, 0UL, (void)0 );
      break;
    case 2UL:
      ZLE6_WALK( m0,  0UL, ZLE6_EXT( m1 ) );
      ZLE6_WALK( m1, 64UL, (void)0 );
      break;
    case 3UL:
      ZLE6_WALK( m0,   0UL, { ZLE6_EXT( m1 ); if( full ) ZLE6_EXT( m2 ); } );
      ZLE6_WALK( m1,  64UL, ZLE6_EXT( m2 ) );
      ZLE6_WALK( m2, 128UL, (void)0 );
      break;
    default:
      ZLE6_WALK( m0,   0UL, { ZLE6_EXT( m1 ); if( full ) { ZLE6_EXT( m2 ); if( full ) ZLE6_EXT( m3 ); } } );
      ZLE6_WALK( m1,  64UL, { ZLE6_EXT( m2 ); if( full ) ZLE6_EXT( m3 ); } );
      ZLE6_WALK( m2, 128UL, ZLE6_EXT( m3 ) );
      ZLE6_WALK( m3, 192UL, (void)0 );
      break;
  }

  if( fs<n ) {
    ulong L = n-fs;
    zle6_copy( lp, src+fs, L ); lp += L;
    cp = zle6_emit_l( cp, L );
  }

  return zle6_finish( dst, src, n, (ulong)( lp-lp0 ), ctrl, (ulong)( cp-ctrl ) );
}

/* fd_zle6_big handles n >= 256.  Same layout/guard structure as
   fd_zle5_big: literal plane grows up from dst+4 (its final position --
   it never moves), the token stream grows up provisionally from
   dst+4+n (past anywhere the literal plane could reach) and is slid
   down to abut the literal plane once its final size is known.  The
   guard constants are bumped from fd_zle5's (+10/+12) to (+11/+13) for
   this format's slightly larger 10-byte worst-case token (both fields
   escaping independently, vs fd_zle5's single 9-byte joint escape). */

static ulong __attribute__((noinline))
fd_zle6_big( uchar *       FD_RESTRICT dst,
             uchar const * FD_RESTRICT src,
             ulong                     n ) {
  ulong words = ( n+63UL )>>6;

  uchar * lp0 = dst+ZLE6_HDR_MIN;
  uchar * lp  = lp0;
  uchar * cp0 = dst+ZLE6_HDR_MIN+n;
  uchar * cp  = cp0;
  uchar * cpx = dst+FD_ZLE6_COMPRESS_BOUND( n )-16UL; /* one zle6_emit writes <= 10 bytes */

  ulong   fs  = 0UL;
  ulong   w   = 0UL;
  ulong   m   = 0UL;
  ulong   d   = zle6_cand( src, 0UL, n );

  for(;;) {

    if( !d ) {
      w++;
      for(;;) {
        if( FD_UNLIKELY( w>=words ) ) goto trailing;
        d = zle6_cand( src, w, n );
        if( d ) break;
        w++;

        while( ( w<<6 )+257UL<=n ) {
          uchar const * p  = src+( w<<6 );
          wwb_t a0 = wwb_or( wwb_ldu( p       ), wwb_ldu( p+  1UL ) );
          wwb_t a1 = wwb_or( wwb_ldu( p+ 64UL ), wwb_ldu( p+ 65UL ) );
          wwb_t a2 = wwb_or( wwb_ldu( p+128UL ), wwb_ldu( p+129UL ) );
          wwb_t a3 = wwb_or( wwb_ldu( p+192UL ), wwb_ldu( p+193UL ) );
          wwb_t t  = _mm512_min_epu8( _mm512_min_epu8( a0, a1 ), _mm512_min_epu8( a2, a3 ) );
          if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) break;
          w += 4UL;
        }
      }
    }

    ulong lo = d & ( ~d+1UL );
    ulong b  = (ulong)__builtin_ctzll( d );
    ulong zs = ( w<<6 )+b;
    ulong x  = d + lo;
    ulong len;
    if( FD_LIKELY( x ) ) {
      len = (ulong)__builtin_popcountll( d ^ x );
      d   = x & ( x-1UL );
    } else {
      len = zle6_spill( src, n, words, &w, &m, 64UL-b );
      ulong nn = ( ( (w+1UL)<<6 )<n ) ? ( zle6_zmask( src, w+1UL, n ) & 1UL ) : 0UL;
      d = m & ( ( m>>1 ) | ( nn<<63 ) );
    }

    if( FD_LIKELY( len>=ZLE6_MIN_Z ) ) {
      ulong L = zs-fs;
      if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+11UL>=n || cp+13UL>=cpx ) )
        return fd_zle6_stored( dst, src, n );
      zle6_copy( lp, src+fs, L ); lp += L;
      cp = zle6_emit( cp, L, len );
      fs = zs+len;
    }
  }

trailing:
  if( fs<n ) {
    ulong L = n-fs;
    if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+11UL>=n || cp+13UL>=cpx ) )
      return fd_zle6_stored( dst, src, n );
    zle6_copy( lp, src+fs, L ); lp += L;
    cp = zle6_emit_l( cp, L );
  }

  return zle6_finish( dst, src, n, (ulong)( lp-lp0 ), cp0, (ulong)( cp-cp0 ) );
}

ulong
fd_zle6_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz       ) ) return 0UL;
  if( FD_LIKELY  ( data_sz< 64UL  ) ) return fd_zle6_tiny ( dst, src, data_sz );
  if( FD_LIKELY  ( data_sz<256UL  ) ) return fd_zle6_small( dst, src, data_sz );
  return                                     fd_zle6_big  ( dst, src, data_sz );
}

/* ==================================================================
   Decoder

   Driven by data_sz, not by an explicit control-stream length: the
   loop consumes tokens until it has produced data_sz output bytes.
   That's what lets the last token in the stream carry no Z field --
   see the wire format comment at the top of this file.
   ================================================================== */

long
fd_zle6_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       d = (uchar *)data;
  uchar const * s = (uchar const *)comp;

  if( FD_UNLIKELY( comp_sz<1UL ) ) return data_sz==0UL ? 0L : FD_ZLE6_ERR_CORRUPT;

  uchar b0 = s[0];
  if( FD_UNLIKELY( b0&ZLE6_STORED_BIT ) ) {
    ulong n = comp_sz-1UL;
    if( FD_UNLIKELY( n>data_sz ) ) return FD_ZLE6_ERR_SPACE;
    fd_memcpy( d, s+1UL, n );
    return (long)n;
  }

  ulong hdr, L;
  ulong c0 = (ulong)b0 & 0x7FUL;
  if( FD_LIKELY( c0!=ZLE6_C0_ESC ) ) { hdr = ZLE6_HDR_MIN; L = c0; }
  else {
    if( FD_UNLIKELY( comp_sz<ZLE6_HDR_MAX ) ) return FD_ZLE6_ERR_CORRUPT;
    L   = (ulong)s[1] | ((ulong)s[2]<<8) | ((ulong)s[3]<<16);
    hdr = ZLE6_HDR_MAX;
  }
  if( FD_UNLIKELY( hdr+L>comp_sz ) ) return FD_ZLE6_ERR_CORRUPT;

  uchar const * lp    = s+hdr;
  uchar const * lpend = lp+L;
  uchar const * cp    = lpend;
  uchar const * ce    = s+comp_sz;
  uchar *       o     = d;
  uchar *       oend  = d+data_sz;

  while( o<oend ) {
    if( FD_UNLIKELY( cp>=ce ) ) return FD_ZLE6_ERR_CORRUPT;
    ulong ll = *cp++;
    if( FD_UNLIKELY( ll==0xFFUL ) ) {
      if( FD_UNLIKELY( cp+4UL>ce ) ) return FD_ZLE6_ERR_CORRUPT;
      ll  = (ulong)cp[0] | ((ulong)cp[1]<<8) | ((ulong)cp[2]<<16) | ((ulong)cp[3]<<24);
      cp += 4UL;
    }
    if( FD_UNLIKELY( ll>(ulong)(lpend-lp) || ll>(ulong)(oend-o) ) ) return FD_ZLE6_ERR_CORRUPT;
    zle6_copy( o, lp, ll );
    o += ll; lp += ll;
    if( o==oend ) break; /* terminal token: no Z field -- see wire format comment */

    if( FD_UNLIKELY( cp>=ce ) ) return FD_ZLE6_ERR_CORRUPT;
    ulong zb = *cp++;
    ulong zl;
    if( FD_LIKELY( zb!=0xFFUL ) ) zl = zb+2UL;
    else {
      if( FD_UNLIKELY( cp+4UL>ce ) ) return FD_ZLE6_ERR_CORRUPT;
      ulong ext = (ulong)cp[0] | ((ulong)cp[1]<<8) | ((ulong)cp[2]<<16) | ((ulong)cp[3]<<24);
      cp += 4UL;
      zl  = ext+2UL;
    }
    if( FD_UNLIKELY( zl>(ulong)(oend-o) ) ) return FD_ZLE6_ERR_CORRUPT;
    zle6_zero( o, zl );
    o += zl;
  }

  /* every literal byte consumed, every output byte produced, and no
     unconsumed trailing garbage in the token stream */
  if( FD_UNLIKELY( (lp!=lpend) | (cp!=ce) ) ) return FD_ZLE6_ERR_CORRUPT;
  return (long)data_sz;
}

#else /* !FD_HAS_AVX512 --------------------------------------------------

   Firedancer requires AVX-512; this scalar path exists purely so the
   codec still builds and fuzzes on other hosts.  Compress always
   stores; decompress is a straightforward scalar walk of the same wire
   format. */

ulong
fd_zle6_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz ) ) return 0UL;
  return fd_zle6_stored( dst, src, data_sz );
}

long
fd_zle6_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       d = (uchar *)data;
  uchar const * s = (uchar const *)comp;

  if( FD_UNLIKELY( comp_sz<1UL ) ) return data_sz==0UL ? 0L : FD_ZLE6_ERR_CORRUPT;

  uchar b0 = s[0];
  if( b0&ZLE6_STORED_BIT ) {
    ulong n = comp_sz-1UL;
    if( FD_UNLIKELY( n>data_sz ) ) return FD_ZLE6_ERR_SPACE;
    fd_memcpy( d, s+1UL, n );
    return (long)n;
  }

  ulong hdr, L;
  ulong c0 = (ulong)b0 & 0x7FUL;
  if( c0!=ZLE6_C0_ESC ) { hdr = ZLE6_HDR_MIN; L = c0; }
  else {
    if( FD_UNLIKELY( comp_sz<ZLE6_HDR_MAX ) ) return FD_ZLE6_ERR_CORRUPT;
    L   = (ulong)s[1] | ((ulong)s[2]<<8) | ((ulong)s[3]<<16);
    hdr = ZLE6_HDR_MAX;
  }
  if( FD_UNLIKELY( hdr+L>comp_sz ) ) return FD_ZLE6_ERR_CORRUPT;

  uchar const * lp    = s+hdr;
  uchar const * lpend = lp+L;
  uchar const * cp    = lpend;
  uchar const * ce    = s+comp_sz;
  uchar *       o     = d;
  uchar *       oend  = d+data_sz;

  while( o<oend ) {
    if( FD_UNLIKELY( cp>=ce ) ) return FD_ZLE6_ERR_CORRUPT;
    ulong ll = *cp++;
    if( ll==0xFFUL ) {
      if( FD_UNLIKELY( cp+4UL>ce ) ) return FD_ZLE6_ERR_CORRUPT;
      ll  = (ulong)cp[0] | ((ulong)cp[1]<<8) | ((ulong)cp[2]<<16) | ((ulong)cp[3]<<24);
      cp += 4UL;
    }
    if( FD_UNLIKELY( ll>(ulong)(lpend-lp) || ll>(ulong)(oend-o) ) ) return FD_ZLE6_ERR_CORRUPT;
    fd_memcpy( o, lp, ll ); o += ll; lp += ll;
    if( o==oend ) break;

    if( FD_UNLIKELY( cp>=ce ) ) return FD_ZLE6_ERR_CORRUPT;
    ulong zb = *cp++;
    ulong zl;
    if( zb!=0xFFUL ) zl = zb+2UL;
    else {
      if( FD_UNLIKELY( cp+4UL>ce ) ) return FD_ZLE6_ERR_CORRUPT;
      ulong ext = (ulong)cp[0] | ((ulong)cp[1]<<8) | ((ulong)cp[2]<<16) | ((ulong)cp[3]<<24);
      cp += 4UL;
      zl  = ext+2UL;
    }
    if( FD_UNLIKELY( zl>(ulong)(oend-o) ) ) return FD_ZLE6_ERR_CORRUPT;
    fd_memset( o, 0, zl ); o += zl;
  }

  if( FD_UNLIKELY( (lp!=lpend) | (cp!=ce) ) ) return FD_ZLE6_ERR_CORRUPT;
  return (long)data_sz;
}

#endif /* FD_HAS_AVX512 */

char const *
fd_zle6_strerror( long res ) {
  switch( res ) {
  case FD_ZLE6_ERR_SPACE:   return "out of buffer space";
  case FD_ZLE6_ERR_CORRUPT: return "malformed compressed data";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
