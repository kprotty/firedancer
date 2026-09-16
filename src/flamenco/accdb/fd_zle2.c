#include "fd_zle2.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format: identical to fd_zle.c, see the comment there.  The
   helpers below are duplicated (not shared) because fd_zle.c's are
   file-local. */

#define FD_ZLE2_MIN_Z (2UL)

/* FD_ZLE2_LIT_CAP bounds how many bytes of an in-flight literal run
   fd_zle2_compress keeps staged in its stack scratch buffer before
   forcing an interstitial split.  Must be a multiple of 64. */

#define FD_ZLE2_LIT_CAP (256UL)

static inline ulong
fd_zle2_vput( uchar * p,
              ulong   v ) {
  ulong k = 0UL;
  do {
    p[k] = (uchar)( v & 0x7fUL );
    v >>= 7;
    if( v ) p[k] = (uchar)( p[k] | 0x80 );
    k++;
  } while( v );
  return k;
}

static inline ulong
fd_zle2_vlen( ulong v ) {
  ulong k = 1UL;
  while( v>=128UL ) { v >>= 7; k++; }
  return k;
}

static inline ulong
fd_zle2_ext( ulong v ) {
  return v>=15UL ? fd_zle2_vlen( v-15UL ) : 0UL;
}

static inline ulong
fd_zle2_frame_sz( ulong L,
                   ulong Z ) {
  return 1UL + fd_zle2_ext( L ) + L + fd_zle2_ext( Z );
}

#if FD_HAS_AVX512

/* fd_zle2_emit writes one (L,Z) frame.  `lit` is L literal bytes to
   copy verbatim -- in fd_zle2_compress's general path this points
   into the scratch buffer, not into the original source, which is
   the entire point (see fd_zle2.h). */

static uchar *
fd_zle2_emit( uchar *                   o,
              uchar const * FD_RESTRICT lit,
              ulong                     L,
              ulong                     Z ) {
  ulong ln = L<15UL ? L : 15UL;
  ulong zn = Z<15UL ? Z : 15UL;
  *o++ = (uchar)( (ln<<4) | zn );
  if( L>=15UL ) o += fd_zle2_vput( o, L-15UL );
  ulong k = 0UL;
  for( ; k+64UL<=L; k+=64UL ) wwb_stu( o+k, wwb_ldu( lit+k ) );
  if( k<L ) {
    __mmask64 mm = _cvtu64_mask64( ~0UL>>( 64UL-(L-k) ) );
    _mm512_mask_storeu_epi8( (void *)( o+k ), mm, _mm512_maskz_loadu_epi8( mm, lit+k ) );
  }
  o += L;
  if( Z>=15UL ) o += fd_zle2_vput( o, Z-15UL );
  return o;
}

/* fd_zle2_zmask: zero-byte mask of word w (bit i set iff byte i of
   that word is zero), clipped so bits at/past n read as non-zero.
   Used only inside the zero-run extension loop below, where the raw
   bytes themselves are not needed (they belong to the run, not to
   any literal span). */

static inline ulong
fd_zle2_zmask( uchar const * s,
               ulong         w,
               ulong         n ) {
  ulong off = w<<6;
  ulong m   = wwb_eq( wwb_ldu( s+off ), wwb_zero() );
  if( FD_LIKELY( off+64UL<=n ) ) return m;
  return m & ( ~0UL>>( 64UL-( n-off ) ) );
}

ulong
fd_zle2_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz ) ) return 0UL;

  ulong esc_sz = 1UL + fd_zle2_ext( data_sz ) + data_sz;
  ulong lim    = (ulong)dst + esc_sz;
  ulong words  = ( data_sz+63UL )>>6;

  uchar * o  = dst;
  ulong   fs = 0UL;   /* source offset: start of the currently open literal run */

  /* `lit` mirrors src[ base<<6, (base<<6)+... ) for whatever words of
     the open literal run have been scanned so far.  Every word that
     the scan visits for literal purposes gets stored here off the
     same load used to test it for zero bytes -- src is read once per
     byte, full stop.  Words consumed by a zero run extension never
     touch `lit` at all: they aren't literal, so there is nothing to
     stage. */

  uchar   lit[ FD_ZLE2_LIT_CAP ] __attribute__((aligned(64)));
  ulong   base = 0UL;  /* word index corresponding to lit[0] */

  ulong w = ~0UL;   /* word index the pending mask `m` describes; wraps to 0 below */
  ulong m = 0UL;     /* zero-byte bits of word w not yet processed */

  for(;;) {

    if( !m ) {
      w++;
      if( w>=words ) break;

      /* Keep the open literal run's tracked span inside `lit`: split
         it into an interstitial Z=0 frame before it would grow past
         FD_ZLE2_LIT_CAP.  fs<=w<<6 always holds here -- any boundary
         inside an earlier word has already been resolved by the time
         the scan reaches w. */

      if( FD_UNLIKELY( ( w-base+1UL )<<6 > FD_ZLE2_LIT_CAP ) ) {
        ulong L = ( w<<6 )-fs;
        if( L ) {
          if( FD_UNLIKELY( (ulong)o+fd_zle2_frame_sz( L, 0UL )>=lim ) ) goto escape;
          o = fd_zle2_emit( o, lit+( fs-( base<<6 ) ), L, 0UL );
        }
        fs   = w<<6;
        base = w;
      }

      wwb_t v = wwb_ldu( src+( w<<6 ) );
      wwb_stu( lit+( ( w-base )<<6 ), v );
      m = wwb_eq( v, wwb_zero() );
      if( FD_UNLIKELY( ( w<<6 )+64UL>data_sz ) ) m &= ( ~0UL>>( 64UL-( data_sz-( w<<6 ) ) ) );
      continue;
    }

    ulong b  = (ulong)__builtin_ctzll( m );
    ulong zs = ( w<<6 )+b;
    ulong hi = m>>b;
    ulong len;
    if( FD_LIKELY( ~hi ) ) {
      len = (ulong)__builtin_ctzll( ~hi );
      m &= ~( ( ( 1UL<<len )-1UL )<<b );
    } else {
      len = 64UL-b;
      m   = 0UL;
    }

    /* `base` may get moved on to stage the word starting the NEXT
       literal run below, before this frame (whose literal bytes
       still live at the OLD base) is emitted -- so latch the base
       this frame's literal span was staged against first. */

    ulong emit_base = base;

    /* A word touched by the extension loop below may need staging
       into `lit` too (see the comment inside it for when and why),
       but never before this iteration's own emit below has read
       whatever the OLD base still has pending -- writing it early is
       exactly the bug this deferral exists to avoid. */

    wwb_t stage_v;
    int   have_stage = 0;

    if( b+len==64UL ) {

      /* Run reaches the end of word w: extend through however many
         more words are entirely zero.  These are pure zero-run
         words -- never staged into `lit` -- so this loop is the one
         place fd_zle2 still pays for a plain re-test (fd_zle2_zmask)
         of a word it already touched once via fd_zle_zmask-alike
         logic above; unlike fd_zle_big it does not fast-skip 256 B
         at a time through them, which is the throughput this design
         trades away for the simpler, branch-lighter main loop. */

      for(;;) {
        w++;
        if( FD_UNLIKELY( w>=words ) ) { m = 0UL; break; }
        ulong xm = fd_zle2_zmask( src, w, data_sz );
        if( xm==~0UL ) { len += 64UL; continue; }
        ulong e = (ulong)__builtin_ctzll( ~xm );
        len += e;
        m = xm & ~( ( 1UL<<e )-1UL );

        if( len>=FD_ZLE2_MIN_Z ) {

          /* This run is about to close a frame below, which consumes
             everything up to zs+len -- inside or past word w -- so
             nothing under the OLD base survives past that emit.  Safe
             to make word w the start of a fresh window, but only
             AFTER that emit has run; stage it via `have_stage`. */

          base       = w;
          stage_v    = wwb_ldu( src+( w<<6 ) );
          have_stage = 1;

        } else {

          /* Doesn't reach FD_ZLE2_MIN_Z: fs stays put below, so the
             OLD window is still needed and cannot be discarded.  Keep
             word w in the OLD base's coordinates instead of rebasing
             to it, splitting off an interstitial Z=0 frame first if
             it wouldn't fit -- the same rule the main scan uses. */

          if( FD_UNLIKELY( ( w-base+1UL )<<6 > FD_ZLE2_LIT_CAP ) ) {
            ulong L = ( w<<6 )-fs;
            if( L ) {
              if( FD_UNLIKELY( (ulong)o+fd_zle2_frame_sz( L, 0UL )>=lim ) ) goto escape;
              o = fd_zle2_emit( o, lit+( fs-( base<<6 ) ), L, 0UL );
            }
            fs   = w<<6;
            base = w;
          }
          wwb_stu( lit+( ( w-base )<<6 ), wwb_ldu( src+( w<<6 ) ) );
        }
        break;
      }
    }

    if( len>=FD_ZLE2_MIN_Z ) {
      ulong L = zs-fs;
      if( FD_UNLIKELY( (ulong)o+fd_zle2_frame_sz( L, len )>=lim ) ) goto escape;
      o  = fd_zle2_emit( o, lit+( fs-( emit_base<<6 ) ), L, len );
      fs = zs+len;
    }

    if( have_stage ) wwb_stu( lit, stage_v );
  }

  if( fs<data_sz ) {
    ulong L = data_sz-fs;
    if( FD_UNLIKELY( (ulong)o+fd_zle2_frame_sz( L, 0UL )>=lim ) ) goto escape;
    o = fd_zle2_emit( o, lit+( fs-( base<<6 ) ), L, 0UL );
  }
  return (ulong)( o-dst );

escape: {

  /* Rare / pathological: the greedy scan above cannot beat a single
     flat literal frame over the whole input.  Falls back to reading
     straight from src -- the scratch buffer optimization above only
     pays for itself on the common, non-escaping path. */

  uchar * oo = dst;
  ulong   ln = data_sz<15UL ? data_sz : 15UL;
  *oo++ = (uchar)( ln<<4 );
  if( data_sz>=15UL ) oo += fd_zle2_vput( oo, data_sz-15UL );
  ulong k = 0UL;
  for( ; k+64UL<=data_sz; k+=64UL ) wwb_stu( oo+k, wwb_ldu( src+k ) );
  if( k<data_sz ) {
    __mmask64 mm = _cvtu64_mask64( ~0UL>>( 64UL-( data_sz-k ) ) );
    _mm512_mask_storeu_epi8( (void *)( oo+k ), mm, _mm512_maskz_loadu_epi8( mm, src+k ) );
  }
  oo += data_sz;
  return (ulong)( oo-dst );
}
}

#else
#error "fd_zle2 is a prototype and currently requires FD_HAS_AVX512 (no scalar fallback yet)"
#endif /* FD_HAS_AVX512 */
