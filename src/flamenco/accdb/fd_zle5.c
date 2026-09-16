#include "fd_zle5.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format
   ===========

     [0]              u8   mode        0 = stored, 1 = zle

     mode 0 (stored) -- the incompressible / tiny fallback:
       [1 .. 1+n)          raw bytes                    comp_sz = n + 1

     mode 1 (zle):
       [1]              u8   c0         control-plane size C, or 0xFF
       [2 .. 6)        u32LE C          present only when c0 == 0xFF
       [h .. h+L)           literal plane   (every literal byte, packed)
       [h+L .. comp_sz)     control plane   (C bytes)
                            h = 2 when c0 != 0xFF, else 6
                            L = comp_sz - h - C

   Control plane
   -------------
   A stream of frames, each describing one (literal run, zero run) pair
   in input order:

     2-byte frame:  (u8 ll, u8 zl)                    ll, zl in 0..254
     9-byte frame:  (0xFF, u32LE ll, u32LE zl)        either value >= 255

   One test of the lead byte (== 0xFF?) tells the two apart.  Long runs
   stay in a single 9-byte frame, so a multi-megabyte zero run decodes
   to one memset.  Invariants over all frames: sum(ll+zl) == n and
   sum(ll) == L.

   A zero run shorter than ZLE5_MIN_Z is not worth a frame -- it would
   cost as many control bytes as it saves -- so the encoder leaves those
   bytes in the literal plane and the decoder just copies them out.
   ZLE5_MIN_Z == 2 also bounds the frame count: every frame consumes at
   least 2 input bytes, so there are at most n/2 frames. */

#define ZLE5_MODE_STORED (0)
#define ZLE5_MODE_ZLE    (1)
#define ZLE5_HDR_MAX     (6UL)   /* header size when c0 == 0xFF */
#define ZLE5_MIN_Z       (2UL)   /* shortest zero run the encoder will frame */

FD_STATIC_ASSERT( ZLE5_MIN_Z==2UL, unit );

/* zle5_emit writes one control frame (literal length ll, zero length
   zl) at cp and returns the advanced cursor.  2 bytes when both lengths
   fit in a u8 with room for the 0xFF sentinel, else the 9-byte escape. */

static inline uchar *
zle5_emit( uchar * cp,
           ulong   ll,
           ulong   zl ) {
  if( FD_LIKELY( (ll|zl)<0xFFUL ) ) {
    cp[0] = (uchar)ll;
    cp[1] = (uchar)zl;
    return cp+2UL;
  }
  cp[0] = 0xFF;
  cp[1] = (uchar)( ll       ); cp[2] = (uchar)( ll>> 8 ); cp[3] = (uchar)( ll>>16 ); cp[4] = (uchar)( ll>>24 );
  cp[5] = (uchar)( zl       ); cp[6] = (uchar)( zl>> 8 ); cp[7] = (uchar)( zl>>16 ); cp[8] = (uchar)( zl>>24 );
  return cp+9UL;
}

/* fd_zle5_stored writes the raw fallback frame: a mode byte followed by
   a verbatim copy of the input.  comp_sz = n + 1. */

static inline ulong
fd_zle5_stored( uchar *       FD_RESTRICT dst,
                uchar const * FD_RESTRICT src,
                ulong                     n ) {
  dst[0] = ZLE5_MODE_STORED;
  fd_memcpy( dst+1UL, src, n );
  return n+1UL;
}

#if FD_HAS_AVX512

/* ==================================================================
   Scan primitives

   Everything in the encoder operates on a "zero mask", never on the
   bytes directly.  For a 64-byte word, bit i of the mask is set iff
   byte i of the word is zero (vpcmpeqb against zero).
   ================================================================== */

/* zle5_zmask returns the zero mask of word w of the n-byte input at s.
   Bits past the end of the input are forced to 0 so a partial final
   word does not report phantom zero runs.  A word entirely past the end
   returns 0. */

static inline ulong
zle5_zmask( uchar const * s,
            ulong         w,
            ulong         n ) {
  ulong off = w<<6;
  ulong m   = wwb_eq( wwb_ldu( s+off ), wwb_zero() );
  if( FD_LIKELY( off+64UL<=n ) ) return m;          /* whole word in range */
  if( off>=n                  ) return 0UL;         /* whole word past end  */
  return m & ( ~0UL>>( 64UL-( n-off ) ) );          /* mask the tail        */
}

/* zle5_cand returns the run-start candidate mask for word w: bit i is
   set iff bytes (w*64+i) and (w*64+i+1) are both zero -- i.e. offset i
   begins a zero run of length >= 2.

   Scalar form: d = m & (m>>1), with the low bit of the next word's mask
   shifted into bit 63 to carry a run across the word boundary.

   Vector form: m>>1 is just the zero mask of the bytes one further
   along, so or(ldu(s+off), ldu(s+off+1)) is zero at byte i iff both
   src[off+i] and src[off+i+1] are zero, and vptestnmb turns that
   directly into the 64-bit mask.  The +1 load also pulls the next
   word's low byte into lane 63, so the cross-word carry is free -- no
   separate neighbour mask, no scalar shift/or, and m itself never has
   to be materialised on the common path.

   The final word falls back to the scalar form: the +1 load would read
   one byte past the caller's guaranteed align_up(n,64) span. */

static inline ulong
zle5_cand( uchar const * s,
           ulong         w,
           ulong         n ) {
  ulong off = w<<6;
  if( FD_LIKELY( off+64UL<n ) ) {
    wwb_t p = wwb_or( wwb_ldu( s+off ), wwb_ldu( s+off+1UL ) );
    return (ulong)_mm512_testn_epi8_mask( p, p );
  }
  ulong m = zle5_zmask( s, w, n );
  return m & ( m>>1 );
}

/* ==================================================================
   Byte movers -- and the memcpy-fold trap

   zle5_copy / zle5_zero move a frame's literal bytes / write its zero
   run.  They run on every frame of both the encoder and the decoder.

   The natural implementation -- a 64-byte-chunk loop plus a masked tail
   -- is recognised by clang's loop-idiom pass and folded into a
   call to memcpy@PLT (or memset@PLT), *even on a 6-70 byte spl-token
   literal run*.  Inside an otherwise all-vector function that call
   costs a vzeroupper, a pile of caller-saved spills/reloads, and the
   call itself: ~15-20 cycles, roughly half of a small-account encode.

   So these are written straight-line, no loop: a handful of full
   64-byte stores plus one *overlapping* 64-byte store for the ragged
   tail, capped at 256 bytes.  Larger copies do call fd_memcpy -- rare,
   amortised, and clang would emit the call there anyway.  Every full
   64-byte load is either size-gated or aimed at the last 64 bytes of
   the run, so it stays inside the source and the decoder never
   over-reads the compressed blob.  (An asm barrier on a chunk loop's
   induction variable also defeats the fold with a smaller diff, but
   benched 2-4% slower: it forces the large copies off memcpy's ERMS
   path.)
   ================================================================== */

/* zle5_copy copies sz bytes from s to d.  s and d never overlap in
   practice: the per-frame calls go input->output, and the one
   slide-down (fd_zle5_big's control-plane placement) is guarded so the
   gap always exceeds C.

   For 64 < sz <= 256 the bulk is covered by 1-3 aligned-stride 64-byte
   copies and the ragged tail by a single *overlapping* 64-byte copy of
   the last 64 bytes -- no k-mask register, no bzhi, no masked load.
   The overlap reads [s+sz-64, s+sz), still inside the source run, so
   the decoder never over-reads the compressed blob.  Only the sz <= 64
   case, where a full 64-byte load could over-read, needs a mask. */

static inline void
zle5_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  if( FD_LIKELY( sz<=64UL ) ) {
    __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ); /* bzhi saturates at 64 */
    _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
    return;
  }
  if( FD_UNLIKELY( sz>256UL ) ) { fd_memcpy( d, s, sz ); return; }

  wwb_stu( d, wwb_ldu( s ) );                                 /* d[0..64)          */
  if( sz>128UL ) {
    wwb_stu( d+64UL, wwb_ldu( s+64UL ) );                     /* d[64..128)        */
    if( sz>192UL ) wwb_stu( d+128UL, wwb_ldu( s+128UL ) );    /* d[128..192)       */
  }
  wwb_stu( d+sz-64UL, wwb_ldu( s+sz-64UL ) );                 /* d[sz-64..sz), overlapping */
}

/* zle5_zero writes sz zero bytes at d.  Same shape as zle5_copy: for
   64 < sz <= 256, full 64-byte zero stores at fixed stride plus one
   overlapping store of the last 64 bytes -- no k-mask.  sz <= 64 keeps
   the mask because a full store would run past d+sz (a real OOB write,
   unlike the load in zle5_copy).  256 < sz < 512 stays a short counted
   loop; >= 512 is left to fd_memset. */

static inline void
zle5_zero( uchar * d,
           ulong   sz ) {
  wwb_t z = wwb_zero();

  if( FD_LIKELY( sz<=64UL ) ) {
    _mm512_mask_storeu_epi8( (void *)d, _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ), z );
    return;
  }
  if( FD_LIKELY( sz<=256UL ) ) {
    wwb_stu( d, z );                                          /* d[0..64)          */
    if( sz>128UL ) {
      wwb_stu( d+64UL, z );                                   /* d[64..128)        */
      if( sz>192UL ) wwb_stu( d+128UL, z );                   /* d[128..192)       */
    }
    wwb_stu( d+sz-64UL, z );                                  /* d[sz-64..sz), overlapping */
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

/* zle5_frame emits one decoded frame at o: ll literal bytes from lp,
   then zl zero bytes.  When the whole frame fits in one 64-byte vector
   -- the common case for account data -- it is a single masked store:
   the masked load reads ll literal bytes and zero-fills lanes [ll,64)
   for free, and the masked store writes exactly ll+zl bytes.  Literal
   copy and zero fill in one instruction, with no branch on ll vs zl.

   The tot > 64 path is deliberately left as out-of-line calls: inlining
   more masked stores here pessimises register allocation on the common
   single-store path (measured ~2 GB/s on the mix). */

static inline void
zle5_frame( uchar *       FD_RESTRICT o,
            uchar const * FD_RESTRICT lp,
            ulong                     ll,
            ulong                     zl ) {
  ulong tot = ll+zl;
  if( FD_LIKELY( tot<=64UL ) ) {
    __mmask64 ml = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)ll  ) );
    __mmask64 mt = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)tot ) );
    _mm512_mask_storeu_epi8( (void *)o, mt, _mm512_maskz_loadu_epi8( ml, lp ) );
    return;
  }
  zle5_copy( o,    lp, ll );
  zle5_zero( o+ll,     zl );
}

/* zle5_spill continues a zero run that ran off the end of word *pw
   (its carry-propagate resolution overflowed bit 63).  It advances
   word-at-a-time while words are all-zero, and 256 bytes at a time once
   it can prove a whole 256-byte block is zero with one OR + test.  On
   return *pw is the landing word, *pm its zero mask with the run's
   leading bits cleared, and the function result is the total run length.

   A 14 KB zero run (a Serum slab) costs ~57 of these OR/test sequences
   instead of 14000 byte comparisons. */

static inline ulong
zle5_spill( uchar const * s,
            ulong         n,
            ulong         words,
            ulong *       pw,
            ulong *       pm,
            ulong         len ) {
  ulong w = *pw+1UL;
  for(;;) {
    if( w>=words ) { *pw = w; *pm = 0UL; return len; } /* run to end of input */

    ulong t = zle5_zmask( s, w, n );
    if( FD_UNLIKELY( t!=~0UL ) ) {
      /* this word has a non-zero byte -- the run ends inside it */
      ulong ext = (ulong)__builtin_ctzll( ~t );        /* trailing zeros of the word */
      *pw = w;
      *pm = t & ~( ( 1UL<<ext )-1UL );                 /* clear the run's bits */
      return len+ext;
    }

    /* whole word is zero */
    w++; len += 64UL;

    /* fast-forward over all-zero 256-byte blocks */
    while( ( w<<6 )+256UL<=n ) {
      uchar const * p = s+( w<<6 );
      wwb_t v = wwb_or( wwb_or( wwb_ldu( p       ), wwb_ldu( p+ 64UL ) ),
                        wwb_or( wwb_ldu( p+128UL ), wwb_ldu( p+192UL ) ) );
      if( FD_UNLIKELY( _mm512_test_epi8_mask( v, v ) ) ) break; /* some non-zero byte */
      w += 4UL; len += 256UL;
    }
  }
}

/* ==================================================================
   Encoder
   ================================================================== */

/* ZLE5_EXT continues the current zero run (len) into the next mask word
   mm.  _e is the count of leading zero bytes of mm (== the number of
   run bytes contributed by this word).  If the whole word is zero
   (_e == 64) the run may continue into a further word, which the caller
   signals by inspecting `full`. */

#define ZLE5_EXT( mm ) do {                                            \
    ulong _e = ~(mm) ? (ulong)__builtin_ctzll( ~(mm) ) : 64UL;         \
    len  += _e;                                                        \
    (mm)  = _e<64UL ? ( (mm) & ~( ( 1UL<<_e )-1UL ) ) : 0UL;           \
    full  = _e==64UL;                                                  \
  } while(0)

/* ZLE5_WALK scans zero mask m (whose bit 0 is input offset wb) and, for
   every zero run of at least ZLE5_MIN_Z bytes, streams the preceding
   literal run into the literal plane lp and appends a 2-byte (ll,zl)
   frame to the control plane cp.

     fs   -- input offset just past the previous frame (start of the
             pending literal run)
     emax -- largest ll or zl seen so far; fd_zle5_small punts to a
             stored frame if this ever needs the 9-byte escape, which
             for n <= 256 is vanishingly rare
     EXT  -- statement that continues a run past bit 63 into the
             following mask word(s)

   Per run:
     b   = ctz(m)          -- run starts at bit b, offset wb+b
     hi  = m >> b          -- mask from the run's start
     len = ctz(~hi)        -- length of the leading run of ones in hi
           (or 64-b if hi is all ones, i.e. the run fills the word) */

#define ZLE5_WALK( m, wb, EXT ) do {                                   \
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
      if( len>=ZLE5_MIN_Z ) {                                          \
        ulong _L = zs-fs;                                              \
        zle5_copy( lp, src+fs, _L ); lp += _L;                         \
        cp[0] = (uchar)_L; cp[1] = (uchar)len; cp += 2UL;              \
        emax  = fd_ulong_max( emax, fd_ulong_max( _L, len ) );         \
        fs    = zs+len;                                                \
      }                                                                \
    }                                                                  \
  } while(0)

/* fd_zle5_small handles 1 <= n <= 256, i.e. 1..4 mask words.  This is
   the band that dominates the workload (spl-token alone is 165 bytes
   and ~57% of all accounts), so the whole thing is straight-line: the
   masks live in registers, a switch selects one of four fully unrolled
   bodies, and there is no per-word size check in the hot path.

   The control plane is built in a stack buffer and then placed right
   after the literal plane.  For n <= 256 it can hold at most ~85 frames
   (170 bytes) so C < 255 always: the header is always 2 bytes and the
   literal plane sits at the fixed offset dst+2. */

static ulong
fd_zle5_small( uchar *       FD_RESTRICT dst,
               uchar const * FD_RESTRICT src,
               ulong                     n ) {
  ulong words = ( n+63UL )>>6;

  /* Load the 1..4 zero masks.  Mask off bits past n in the final word. */
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

  uchar   ctrl[ 256 ];       /* control plane scratch */
  uchar * cp   = ctrl;
  uchar * lp0  = dst+2UL;    /* literal plane starts after the 2-byte header */
  uchar * lp   = lp0;
  ulong   fs   = 0UL;
  ulong   emax = 0UL;

  /* Walk each word.  The EXT argument spells out how a run that fills
     word i continues into words i+1, i+2, ...  `full` chains the check. */
  switch( words ) {
    case 1UL:
      ZLE5_WALK( m0, 0UL, (void)0 );
      break;
    case 2UL:
      ZLE5_WALK( m0,  0UL, ZLE5_EXT( m1 ) );
      ZLE5_WALK( m1, 64UL, (void)0 );
      break;
    case 3UL:
      ZLE5_WALK( m0,   0UL, { ZLE5_EXT( m1 ); if( full ) ZLE5_EXT( m2 ); } );
      ZLE5_WALK( m1,  64UL, ZLE5_EXT( m2 ) );
      ZLE5_WALK( m2, 128UL, (void)0 );
      break;
    default:
      ZLE5_WALK( m0,   0UL, { ZLE5_EXT( m1 ); if( full ) { ZLE5_EXT( m2 ); if( full ) ZLE5_EXT( m3 ); } } );
      ZLE5_WALK( m1,  64UL, { ZLE5_EXT( m2 ); if( full ) ZLE5_EXT( m3 ); } );
      ZLE5_WALK( m2, 128UL, ZLE5_EXT( m3 ) );
      ZLE5_WALK( m3, 192UL, (void)0 );
      break;
  }

  /* Trailing literal run (no zero run after it). */
  if( fs<n ) {
    ulong L = n-fs;
    zle5_copy( lp, src+fs, L ); lp += L;
    cp[0] = (uchar)L; cp[1] = 0; cp += 2UL;
    emax  = fd_ulong_max( emax, L );
  }

  ulong Lsz = (ulong)( lp-lp0 );
  ulong C   = (ulong)( cp-ctrl );

  /* Bail to stored if a run needs the 9-byte escape (fd_zle5_small
     never emits one), or if the "compressed" form is not smaller. */
  if( FD_UNLIKELY( (emax>=0xFFUL) | (2UL+Lsz+C>=n+1UL) ) ) return fd_zle5_stored( dst, src, n );

  zle5_copy( lp, ctrl, C );  /* place the control plane after the literal plane */
  dst[0] = ZLE5_MODE_ZLE;
  dst[1] = (uchar)C;
  return 2UL+Lsz+C;
}

/* fd_zle5_big handles n > 256.  noinline so its register pressure does
   not spill into the small fast path.

   Layout while encoding:
     - literal plane grows up from dst+6 (fixed 6-byte header)
     - control plane grows up from dst+6+n -- past anywhere the literal
       plane could reach -- then is slid down to abut the literal plane
       at the end.  The gap between the two is the caller's buffer slack
       (FD_ZLE5_COMPRESS_BOUND leaves n/2 + 32 spare).

   The scan uses the vector candidate mask (zle5_cand), the
   carry-propagate run resolver, zle5_spill for cross-word runs, and a
   256-byte literal skip-ahead.  There is no per-frame bound check: the
   two guards below are "stop if we already lost" and "stop if the
   buffer is nearly full", both rare and predicted-not-taken. */

static ulong __attribute__((noinline))
fd_zle5_big( uchar *       FD_RESTRICT dst,
             uchar const * FD_RESTRICT src,
             ulong                     n ) {
  ulong words = ( n+63UL )>>6;

  uchar * lp0 = dst+ZLE5_HDR_MAX;
  uchar * lp  = lp0;
  uchar * cp0 = dst+ZLE5_HDR_MAX+n;
  uchar * cp  = cp0;
  uchar * cpx = dst+FD_ZLE5_COMPRESS_BOUND( n )-16UL; /* one zle5_emit writes <= 9 bytes */

  ulong   fs  = 0UL;                    /* input offset just past the last frame     */
  ulong   w   = 0UL;                    /* current mask word                         */
  ulong   m   = 0UL;                    /* landing-word zero mask (spill path only)  */
  ulong   d   = zle5_cand( src, 0UL, n ); /* run-start candidate mask for word 0     */

  for(;;) {

    /* ---- no candidate in this word: find the next word that has one ---- */
    if( !d ) {
      w++;
      for(;;) {
        if( FD_UNLIKELY( w>=words ) ) goto trailing;
        d = zle5_cand( src, w, n );
        if( d ) break;
        w++;

        /* Literal skip-ahead: if no adjacent-zero *pair* exists anywhere
           in the next 256 bytes there is no run to find there, so skip
           the whole block.  a0..a3 are the same or(ldu(p),ldu(p+1))
           quantity as zle5_cand; a byte is 0 in ak iff that position
           begins a zero pair.  min of the four is 0 iff any block has a
           pair. */
        while( ( w<<6 )+257UL<=n ) {
          uchar const * p  = src+( w<<6 );
          wwb_t a0 = wwb_or( wwb_ldu( p       ), wwb_ldu( p+  1UL ) );
          wwb_t a1 = wwb_or( wwb_ldu( p+ 64UL ), wwb_ldu( p+ 65UL ) );
          wwb_t a2 = wwb_or( wwb_ldu( p+128UL ), wwb_ldu( p+129UL ) );
          wwb_t a3 = wwb_or( wwb_ldu( p+192UL ), wwb_ldu( p+193UL ) );
          wwb_t t  = _mm512_min_epu8( _mm512_min_epu8( a0, a1 ), _mm512_min_epu8( a2, a3 ) );
          if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) break; /* a pair exists */
          w += 4UL;
        }
      }
    }

    /* ---- resolve the run starting at the lowest candidate bit ----

       d holds the ones-span [p, q-1] of this run (bit i set means bytes
       i and i+1 are both zero), plus the candidate bits of any later
       runs in this word.  Adding lo = 1<<p carries through the span and
       lands a single set bit at q (one past the run's last byte).  Then
       d ^ x is exactly bits [p, q], so popcount(d ^ x) is the run
       length -- and it does not depend on b, so it stays off the ctz(d)
       critical path.  x & (x-1) clears bit q, leaving the later runs'
       candidates for the next iteration. */
    ulong lo = d & ( ~d+1UL );
    ulong b  = (ulong)__builtin_ctzll( d );
    ulong zs = ( w<<6 )+b;
    ulong x  = d + lo;
    ulong len;
    if( FD_LIKELY( x ) ) {
      len = (ulong)__builtin_popcountll( d ^ x );
      d   = x & ( x-1UL );
    } else {
      /* x == 0: the add carried out of bit 63.  The run reaches the end
         of the word -- continue it through the following words. */
      len = zle5_spill( src, n, words, &w, &m, 64UL-b );
      /* rebuild the candidate mask at the landing word from its zero
         mask m, carrying in bit 0 of the word after it */
      ulong nn = ( ( (w+1UL)<<6 )<n ) ? ( zle5_zmask( src, w+1UL, n ) & 1UL ) : 0UL;
      d = m & ( ( m>>1 ) | ( nn<<63 ) );
    }

    /* ---- emit the frame for this run ---- */
    if( FD_LIKELY( len>=ZLE5_MIN_Z ) ) {
      ulong L = zs-fs;
      if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+10UL>=n || cp+12UL>=cpx ) )
        return fd_zle5_stored( dst, src, n ); /* already lost, or buffer nearly full */
      zle5_copy( lp, src+fs, L ); lp += L;
      cp = zle5_emit( cp, L, len );
      fs = zs+len;
    }
  }

trailing:
  /* trailing literal run, if any */
  if( fs<n ) {
    ulong L = n-fs;
    if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+10UL>=n || cp+12UL>=cpx ) )
      return fd_zle5_stored( dst, src, n );
    zle5_copy( lp, src+fs, L ); lp += L;
    cp = zle5_emit( cp, L, 0UL );
  }

  /* finalise: "is it actually smaller?" check, then write the header
     and slide the control plane down to abut the literal plane */
  {
    ulong Lsz = (ulong)( lp-lp0 );
    ulong C   = (ulong)( cp-cp0 );
    if( FD_UNLIKELY( ZLE5_HDR_MAX+Lsz+C>=n+1UL ) ) return fd_zle5_stored( dst, src, n );

    dst[0] = ZLE5_MODE_ZLE;
    dst[1] = 0xFF;
    dst[2] = (uchar)( C       );
    dst[3] = (uchar)( C>> 8   );
    dst[4] = (uchar)( C>>16   );
    dst[5] = (uchar)( C>>24   );
    zle5_copy( lp, cp0, C ); /* dst < src, so this forward-overlapping slide is safe */
    return ZLE5_HDR_MAX+Lsz+C;
  }
}

ulong
fd_zle5_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz       ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<=256UL ) ) return fd_zle5_small( dst, src, data_sz );
  return                                     fd_zle5_big  ( dst, src, data_sz );
}

/* ==================================================================
   Decoder

   Single fused parse+emit pass over the control plane.  A two-pass
   design (validate everything, then emit with no per-frame branches)
   benched ~5% slower: the second parse cost more than the
   predicted-not-taken per-frame bounds branch it removed.
   ================================================================== */

long
fd_zle5_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       d = (uchar *)data;
  uchar const * s = (uchar const *)comp;

  if( FD_UNLIKELY( !comp_sz ) ) return data_sz==0UL ? 0L : FD_ZLE5_ERR_CORRUPT;

  ulong mode = s[0];

  /* stored: one checked memcpy */
  if( mode==ZLE5_MODE_STORED ) {
    ulong n = comp_sz-1UL;
    if( FD_UNLIKELY( n>data_sz ) ) return FD_ZLE5_ERR_SPACE;
    fd_memcpy( d, s+1UL, n );
    return (long)n;
  }
  if( FD_UNLIKELY( mode!=ZLE5_MODE_ZLE ) ) return FD_ZLE5_ERR_CORRUPT;
  if( FD_UNLIKELY( comp_sz<2UL         ) ) return FD_ZLE5_ERR_CORRUPT;

  /* parse the header: control-plane size C and header size hdr */
  ulong hdr = 2UL;
  ulong C   = s[1];
  if( FD_UNLIKELY( C==0xFFUL ) ) {
    if( FD_UNLIKELY( comp_sz<ZLE5_HDR_MAX ) ) return FD_ZLE5_ERR_CORRUPT;
    C   = (ulong)s[2] | ((ulong)s[3]<<8) | ((ulong)s[4]<<16) | ((ulong)s[5]<<24);
    hdr = ZLE5_HDR_MAX;
  }
  if( FD_UNLIKELY( hdr+C>comp_sz ) ) return FD_ZLE5_ERR_CORRUPT;

  /* derive the plane extents: [header][literal plane][control plane] */
  ulong lit_sz = comp_sz-hdr-C;

  uchar const * lp    = s+hdr;          /* literal plane cursor */
  uchar const * lpend = lp+lit_sz;
  uchar const * cp    = lpend;          /* control plane cursor */
  uchar const * ce    = s+comp_sz;
  uchar *       o     = d;              /* output cursor */
  uchar *       oend  = d+data_sz;

  while( cp<ce ) {
    ulong ll = cp[0];
    ulong zl;
    if( FD_LIKELY( ll!=0xFFUL ) ) {
      if( FD_UNLIKELY( cp+2UL>ce ) ) return FD_ZLE5_ERR_CORRUPT;
      zl  = cp[1];
      cp += 2UL;
    } else {
      if( FD_UNLIKELY( cp+9UL>ce ) ) return FD_ZLE5_ERR_CORRUPT;
      ll  = (ulong)cp[1] | ((ulong)cp[2]<<8) | ((ulong)cp[3]<<16) | ((ulong)cp[4]<<24);
      zl  = (ulong)cp[5] | ((ulong)cp[6]<<8) | ((ulong)cp[7]<<16) | ((ulong)cp[8]<<24);
      cp += 9UL;
    }
    /* the literal run must fit the literal plane and the whole frame
       must fit the output */
    if( FD_UNLIKELY( ll>(ulong)(lpend-lp) || ll+zl>(ulong)(oend-o) ) ) return FD_ZLE5_ERR_CORRUPT;
    zle5_frame( o, lp, ll, zl );
    o  += ll+zl;
    lp += ll;
  }
  /* every literal byte consumed and every output byte produced */
  if( FD_UNLIKELY( (o!=oend) | (lp!=lpend) ) ) return FD_ZLE5_ERR_CORRUPT;
  return (long)data_sz;
}

#else /* !FD_HAS_AVX512 --------------------------------------------------

   Firedancer requires AVX-512; this scalar path exists purely so the
   codec still builds and fuzzes on other hosts.  Compress always
   stores; decompress is a straightforward scalar walk of the same wire
   format.  */

ulong
fd_zle5_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz ) ) return 0UL;
  return fd_zle5_stored( dst, src, data_sz );
}

long
fd_zle5_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       d = (uchar *)data;
  uchar const * s = (uchar const *)comp;

  if( FD_UNLIKELY( !comp_sz ) ) return data_sz==0UL ? 0L : FD_ZLE5_ERR_CORRUPT;
  ulong mode = s[0];
  if( mode==ZLE5_MODE_STORED ) {
    ulong n = comp_sz-1UL;
    if( FD_UNLIKELY( n>data_sz ) ) return FD_ZLE5_ERR_SPACE;
    fd_memcpy( d, s+1UL, n );
    return (long)n;
  }
  if( FD_UNLIKELY( (mode!=ZLE5_MODE_ZLE) | (comp_sz<2UL) ) ) return FD_ZLE5_ERR_CORRUPT;

  ulong hdr = 2UL, C = s[1];
  if( FD_UNLIKELY( C==0xFFUL ) ) {
    if( FD_UNLIKELY( comp_sz<ZLE5_HDR_MAX ) ) return FD_ZLE5_ERR_CORRUPT;
    C   = (ulong)s[2] | ((ulong)s[3]<<8) | ((ulong)s[4]<<16) | ((ulong)s[5]<<24);
    hdr = ZLE5_HDR_MAX;
  }
  if( FD_UNLIKELY( hdr+C>comp_sz ) ) return FD_ZLE5_ERR_CORRUPT;

  ulong         lit_sz = comp_sz-hdr-C;
  uchar const * lp     = s+hdr;
  uchar const * lpend  = lp+lit_sz;
  uchar const * cp     = lpend;
  uchar const * ce     = s+comp_sz;
  uchar *       o      = d;
  uchar *       oend   = d+data_sz;

  while( cp<ce ) {
    ulong ll = cp[0], zl;
    if( FD_LIKELY( ll!=0xFFUL ) ) {
      if( FD_UNLIKELY( cp+2UL>ce ) ) return FD_ZLE5_ERR_CORRUPT;
      zl = cp[1]; cp += 2UL;
    } else {
      if( FD_UNLIKELY( cp+9UL>ce ) ) return FD_ZLE5_ERR_CORRUPT;
      ll = (ulong)cp[1] | ((ulong)cp[2]<<8) | ((ulong)cp[3]<<16) | ((ulong)cp[4]<<24);
      zl = (ulong)cp[5] | ((ulong)cp[6]<<8) | ((ulong)cp[7]<<16) | ((ulong)cp[8]<<24);
      cp += 9UL;
    }
    if( FD_UNLIKELY( ll>(ulong)(lpend-lp) || ll+zl>(ulong)(oend-o) ) ) return FD_ZLE5_ERR_CORRUPT;
    fd_memcpy( o, lp, ll ); o += ll; lp += ll;
    fd_memset( o, 0, zl );  o += zl;
  }
  if( FD_UNLIKELY( (o!=oend) | (lp!=lpend) ) ) return FD_ZLE5_ERR_CORRUPT;
  return (long)data_sz;
}

#endif

char const *
fd_zle5_strerror( long res ) {
  switch( res ) {
  case FD_ZLE5_ERR_SPACE:   return "out of buffer space";
  case FD_ZLE5_ERR_CORRUPT: return "malformed compressed data";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
