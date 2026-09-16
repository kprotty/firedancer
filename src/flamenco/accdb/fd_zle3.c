#include "fd_zle3.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format
   ===========

     [0]          u8    mode        0 = stored, 1 = zle
     mode 0 (stored):
       [1 .. 1+n]       raw bytes
     mode 1 (zle):
       [1]              u8  c0       control plane size C, or 0xFF
       [2 .. 6]         u32 LE C     only when c0 == 0xFF
       [h .. h+L)       literal plane  (every literal byte, packed)
       [h+L .. comp_sz) control plane  (C bytes; h = 2 or 6, L = comp_sz-h-C)

   fd_zle3 is a planar redesign of fd_zle: the run-length control stream
   is a contiguous block separate from the literal bytes, so the decoder
   walks it with no literal bytes in the way, and can bounds-check the
   whole thing implicitly (the plane extent is exactly C).  See the
   control-plane comment below for the frame layout.  Over all frames,
   sum(ll+zl) == n and sum(ll) == L.

   A zero run shorter than ZLE3_MIN_Z is not worth a frame (it costs as
   many control bytes as it saves) so the encoder leaves those bytes in
   the literal plane verbatim, exactly like fd_zle -- the decoder just
   copies whatever the literal plane holds.

   The encoder falls back to a stored frame whenever the compressed form
   is not smaller than the input, so it never needs fd_zle's per-frame
   "would this overflow the bound" test. */

#define ZLE3_MODE_STORED (0)
#define ZLE3_MODE_ZLE    (1)
#define ZLE3_HDR_MAX     (6UL)
#define ZLE3_MIN_Z       (2UL)

FD_STATIC_ASSERT( ZLE3_MIN_Z==2UL, unit );

/* ---------- control plane ----------

   The control plane is a stream of frames.  A frame is 2 bytes,
   (u8 ll, u8 zl), when both runs fit; otherwise 9 bytes,
   (0xFF, u32 LE ll, u32 LE zl).  One lead-byte test per frame covers
   both lengths.  Nearly every mainnet account frame is 2 bytes; a
   nibble-packed single-byte token was tried and lost (the dominant
   spl-token shape has 38-70 byte runs that always overflow the nibble).
   Long runs stay whole -- one memset on decode. */

static inline uchar *
zle3_emit( uchar * cp,
           ulong   ll,
           ulong   zl ) {
  if( FD_LIKELY( (ll|zl)<0xFFUL ) ) { cp[0] = (uchar)ll; cp[1] = (uchar)zl; return cp+2UL; }
  cp[0] = 0xFF;
  cp[1] = (uchar)( ll      ); cp[2] = (uchar)( ll>> 8 ); cp[3] = (uchar)( ll>>16 ); cp[4] = (uchar)( ll>>24 );
  cp[5] = (uchar)( zl      ); cp[6] = (uchar)( zl>> 8 ); cp[7] = (uchar)( zl>>16 ); cp[8] = (uchar)( zl>>24 );
  return cp+9UL;
}

static inline ulong
fd_zle3_stored( uchar *       FD_RESTRICT dst,
                uchar const * FD_RESTRICT src,
                ulong                     n ) {
  dst[0] = ZLE3_MODE_STORED;
  fd_memcpy( dst+1UL, src, n );
  return n+1UL;
}

#if FD_HAS_AVX512

/* ---------- scan primitives ---------- */

static inline ulong
zle3_zmask( uchar const * s,
            ulong         w,
            ulong         n ) {
  ulong off = w<<6;
  ulong m   = wwb_eq( wwb_ldu( s+off ), wwb_zero() );
  if( FD_LIKELY( off+64UL<=n ) ) return m;
  if( off>=n ) return 0UL;
  return m & ( ~0UL>>( 64UL-( n-off ) ) );
}

static inline void
zle3_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  if( FD_LIKELY( sz<=64UL ) ) { /* the overwhelmingly common case; no loop => clang cannot fold it to a memcpy call */
    __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ); /* bzhi saturates at 64 */
    _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
    return;
  }
  ulong k = 0UL;
  for( ; k+64UL<=sz; k+=64UL ) wwb_stu( d+k, wwb_ldu( s+k ) );
  if( k<sz ) {
    __mmask64 mm = _cvtu64_mask64( ~0UL>>( 64UL-( sz-k ) ) );
    _mm512_mask_storeu_epi8( (void *)( d+k ), mm, _mm512_maskz_loadu_epi8( mm, s+k ) );
  }
}

static inline void
zle3_zero( uchar * d,
           ulong   sz ) {
  wwb_t z = wwb_zero();
  if( FD_LIKELY( sz<=64UL ) ) {
    _mm512_mask_storeu_epi8( (void *)d, _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ), z );
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

/* zle3_frame emits one decoded frame: ll literal bytes from lp then zl
   zero bytes, at o.  When the whole frame fits in one 64-byte vector
   (the common case for account data) it is a single masked store -- the
   masked load zero-fills past ll, so the zeros come for free. */

static inline void
zle3_frame( uchar *       FD_RESTRICT o,
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
  zle3_copy( o,    lp, ll );
  zle3_zero( o+ll,     zl );
}

/* zle3_spill: extend a zero run past the end of word *pw, 256 bytes at
   a time.  Returns the total run length, leaves *pw at the landing word
   and *pm at its zero mask with the run's leading bits cleared.
   Mirrors fd_zle_spill. */

static inline ulong
zle3_spill( uchar const * s,
            ulong         n,
            ulong         words,
            ulong *       pw,
            ulong *       pm,
            ulong         len ) {
  ulong w = *pw+1UL;
  for(;;) {
    if( w>=words ) { *pw = w; *pm = 0UL; return len; }
    ulong t = zle3_zmask( s, w, n );
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

/* ---------- encoder ---------- */

/* ZLE3_WALK scans zero mask m (word base wb) and, for each zero run of
   at least ZLE3_MIN_Z bytes, streams the preceding literal run into the
   literal plane (lp) and writes a 2-byte (ll,zl) frame to the control
   plane (cp).  fs is the input offset of the last frame's end; emax
   tracks the largest run seen (fd_zle3_small punts to a stored frame if
   any run needs the >255 escape, which for n<=256 is vanishingly rare).
   EXT continues a run past bit 63 into the following mask words. */

#define ZLE3_EXT( mm ) do {                                            \
    ulong _e = ~(mm) ? (ulong)__builtin_ctzll( ~(mm) ) : 64UL;         \
    len  += _e;                                                        \
    (mm)  = _e<64UL ? ( (mm) & ~( ( 1UL<<_e )-1UL ) ) : 0UL;           \
    full  = _e==64UL;                                                  \
  } while(0)

#define ZLE3_WALK( m, wb, EXT ) do {                                   \
    while( m ) {                                                       \
      ulong b   = (ulong)__builtin_ctzll( m );                         \
      ulong zs  = (wb)+b;                                              \
      ulong hi  = (m)>>b;                                              \
      ulong len;                                                       \
      ulong full FD_FN_UNUSED = 0UL;                                   \
      if( ~hi ) {                                                      \
        len = (ulong)__builtin_ctzll( ~hi );                           \
        (m) &= ~( ( ( 1UL<<len )-1UL )<<b );                           \
      } else {                                                         \
        len = 64UL-b;                                                  \
        (m) = 0UL;                                                     \
      }                                                                \
      if( b+len==64UL ) { EXT; }                                       \
      if( len>=ZLE3_MIN_Z ) {                                          \
        ulong _L = zs-fs;                                              \
        zle3_copy( lp, src+fs, _L ); lp += _L;                         \
        cp[0] = (uchar)_L; cp[1] = (uchar)len; cp += 2UL;              \
        emax  = fd_ulong_max( emax, fd_ulong_max( _L, len ) );         \
        fs    = zs+len;                                                \
      }                                                                \
    }                                                                  \
  } while(0)

static ulong
fd_zle3_small( uchar *       FD_RESTRICT dst,
               uchar const * FD_RESTRICT src,
               ulong                     n ) {
  /* 1 <= n <= 256, i.e. 1..4 mask words.  Control plane can never reach
     255 bytes here (<=85 frames * 2), so the header is always 2 bytes
     and the literal plane sits at a fixed offset. */
  ulong words = ( n+63UL )>>6;

  ulong m0=wwb_eq( wwb_ldu( src ), wwb_zero() ), m1=0UL, m2=0UL, m3=0UL;
  if( words>1UL ) m1 = wwb_eq( wwb_ldu( src+ 64UL ), wwb_zero() );
  if( words>2UL ) m2 = wwb_eq( wwb_ldu( src+128UL ), wwb_zero() );
  if( words>3UL ) m3 = wwb_eq( wwb_ldu( src+192UL ), wwb_zero() );
  ulong tail = n & 63UL; if( !tail ) tail = 64UL;
  ulong tmask = ~0UL>>( 64UL-tail );
  switch( words ) {
    case 1UL: m0 &= tmask; break;
    case 2UL: m1 &= tmask; break;
    case 3UL: m2 &= tmask; break;
    default:  m3 &= tmask; break;
  }

  uchar   ctrl[ 256 ];
  uchar * cp   = ctrl;
  uchar * lp0  = dst+2UL;
  uchar * lp   = lp0;
  ulong   fs   = 0UL;
  ulong   emax = 0UL; /* max run length seen; a small input with a >=255 run is rare, punt to stored */
  switch( words ) {
    case 1UL:
      ZLE3_WALK( m0, 0UL, (void)0 );
      break;
    case 2UL:
      ZLE3_WALK( m0,  0UL, ZLE3_EXT( m1 ) );
      ZLE3_WALK( m1, 64UL, (void)0 );
      break;
    case 3UL:
      ZLE3_WALK( m0,   0UL, { ZLE3_EXT( m1 ); if( full ) ZLE3_EXT( m2 ); } );
      ZLE3_WALK( m1,  64UL, ZLE3_EXT( m2 ) );
      ZLE3_WALK( m2, 128UL, (void)0 );
      break;
    default:
      ZLE3_WALK( m0,   0UL, { ZLE3_EXT( m1 ); if( full ) { ZLE3_EXT( m2 ); if( full ) ZLE3_EXT( m3 ); } } );
      ZLE3_WALK( m1,  64UL, { ZLE3_EXT( m2 ); if( full ) ZLE3_EXT( m3 ); } );
      ZLE3_WALK( m2, 128UL, ZLE3_EXT( m3 ) );
      ZLE3_WALK( m3, 192UL, (void)0 );
      break;
  }
  if( fs<n ) {
    ulong L = n-fs;
    zle3_copy( lp, src+fs, L ); lp += L;
    cp[0] = (uchar)L; cp[1] = 0; cp += 2UL;
    emax  = fd_ulong_max( emax, L );
  }

  ulong Lsz = (ulong)( lp-lp0 );
  ulong C   = (ulong)( cp-ctrl );
  if( FD_UNLIKELY( (emax>=0xFFUL) | (2UL+Lsz+C>=n+1UL) ) ) return fd_zle3_stored( dst, src, n );

  zle3_copy( lp, ctrl, C ); /* C < 255 for n<=256; masked vector store, never a libc call */
  dst[0] = ZLE3_MODE_ZLE;
  dst[1] = (uchar)C;
  return 2UL+Lsz+C;
}

static ulong __attribute__((noinline))
fd_zle3_big( uchar *       FD_RESTRICT dst,
             uchar const * FD_RESTRICT src,
             ulong                     n ) {
  ulong words = ( n+63UL )>>6;

  /* literal plane grows up from dst+6 (a fixed 6-byte header); control
     plane grows up from dst+6+n and is slid down to abut the literal
     plane at the end.  The gap between them is the caller's buffer
     slack (FD_ZLE3_COMPRESS_BOUND leaves n/2+... spare). */
  uchar * lp0 = dst+ZLE3_HDR_MAX;
  uchar * lp  = lp0;
  uchar * cp0 = dst+ZLE3_HDR_MAX+n;
  uchar * cp  = cp0;
  uchar * cpx = dst+FD_ZLE3_COMPRESS_BOUND( n )-16UL; /* one zle3_emit writes <=10 bytes */

  ulong   fs  = 0UL;
  ulong   w   = 0UL;
  ulong   m   = zle3_zmask( src, 0UL, n );
  ulong   nzn = ( 64UL<n ) ? ( zle3_zmask( src, 1UL, n ) & 1UL ) : 0UL;
  ulong   d   = m & ( ( m>>1 ) | ( nzn<<63 ) );

  for(;;) {

    if( !d ) {
      w++;
      for(;;) {
        if( FD_UNLIKELY( w>=words ) ) goto trailing;
        m = zle3_zmask( src, w, n );
        ulong nn = ( ( (w+1UL)<<6 )<n ) ? ( zle3_zmask( src, w+1UL, n ) & 1UL ) : 0UL;
        d = m & ( ( m>>1 ) | ( nn<<63 ) );
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
      len = (ulong)__builtin_ctzll( x )-b+1UL;
      d   = x & ( x-1UL );
    } else {
      len = zle3_spill( src, n, words, &w, &m, 64UL-b );
      ulong nn = ( ( (w+1UL)<<6 )<n ) ? ( zle3_zmask( src, w+1UL, n ) & 1UL ) : 0UL;
      d   = m & ( ( m>>1 ) | ( nn<<63 ) );
    }
    if( FD_LIKELY( len>=ZLE3_MIN_Z ) ) {
      ulong L = zs-fs;
      if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+10UL>=n || cp+12UL>=cpx ) )
        return fd_zle3_stored( dst, src, n );
      zle3_copy( lp, src+fs, L ); lp += L;
      cp = zle3_emit( cp, L, len );
      fs = zs+len;
    }
  }

trailing:
  if( fs<n ) {
    ulong L = n-fs;
    if( FD_UNLIKELY( (ulong)(lp-lp0)+L+(ulong)(cp-cp0)+10UL>=n || cp+12UL>=cpx ) ) return fd_zle3_stored( dst, src, n );
    zle3_copy( lp, src+fs, L ); lp += L;
    cp = zle3_emit( cp, L, 0UL );
  }

  {
    ulong Lsz = (ulong)( lp-lp0 );
    ulong C   = (ulong)( cp-cp0 );
    if( FD_UNLIKELY( ZLE3_HDR_MAX+Lsz+C>=n+1UL ) ) return fd_zle3_stored( dst, src, n );

    dst[0] = ZLE3_MODE_ZLE;
    dst[1] = 0xFF;
    dst[2] = (uchar)( C      );
    dst[3] = (uchar)( C>> 8  );
    dst[4] = (uchar)( C>>16  );
    dst[5] = (uchar)( C>>24  );
    zle3_copy( lp, cp0, C ); /* slide control down; dst<src so forward copy is overlap safe */
    return ZLE3_HDR_MAX+Lsz+C;
  }
}

ulong
fd_zle3_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz       ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<=256UL ) ) return fd_zle3_small( dst, src, data_sz );
  return                                     fd_zle3_big  ( dst, src, data_sz );
}

/* ---------- decoder ---------- */

long
fd_zle3_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       d = (uchar *)data;
  uchar const * s = (uchar const *)comp;

  if( FD_UNLIKELY( !comp_sz ) ) return data_sz==0UL ? 0L : FD_ZLE3_ERR_CORRUPT;

  ulong mode = s[0];

  if( mode==ZLE3_MODE_STORED ) {
    ulong n = comp_sz-1UL;
    if( FD_UNLIKELY( n>data_sz ) ) return FD_ZLE3_ERR_SPACE;
    fd_memcpy( d, s+1UL, n );
    return (long)n;
  }
  if( FD_UNLIKELY( mode!=ZLE3_MODE_ZLE ) ) return FD_ZLE3_ERR_CORRUPT;
  if( FD_UNLIKELY( comp_sz<2UL         ) ) return FD_ZLE3_ERR_CORRUPT;

  ulong hdr = 2UL;
  ulong C   = s[1];
  if( FD_UNLIKELY( C==0xFFUL ) ) {
    if( FD_UNLIKELY( comp_sz<ZLE3_HDR_MAX ) ) return FD_ZLE3_ERR_CORRUPT;
    C   = (ulong)s[2] | ((ulong)s[3]<<8) | ((ulong)s[4]<<16) | ((ulong)s[5]<<24);
    hdr = ZLE3_HDR_MAX;
  }
  if( FD_UNLIKELY( hdr+C>comp_sz ) ) return FD_ZLE3_ERR_CORRUPT;

  /* layout: [header][literal plane][control plane]. */
  ulong lit_sz = comp_sz-hdr-C;

  uchar const * lp    = s+hdr;
  uchar const * lpend = lp+lit_sz;
  uchar const * cp    = lpend;
  uchar const * ce    = s+comp_sz;
  uchar *       o     = d;
  uchar *       oend  = d+data_sz;

  while( cp<ce ) {
    ulong ll = cp[0];
    ulong zl;
    if( FD_LIKELY( ll!=0xFFUL ) ) {
      if( FD_UNLIKELY( cp+2UL>ce ) ) return FD_ZLE3_ERR_CORRUPT;
      zl  = cp[1];
      cp += 2UL;
    } else {
      if( FD_UNLIKELY( cp+9UL>ce ) ) return FD_ZLE3_ERR_CORRUPT;
      ll  = (ulong)cp[1] | ((ulong)cp[2]<<8) | ((ulong)cp[3]<<16) | ((ulong)cp[4]<<24);
      zl  = (ulong)cp[5] | ((ulong)cp[6]<<8) | ((ulong)cp[7]<<16) | ((ulong)cp[8]<<24);
      cp += 9UL;
    }
    if( FD_UNLIKELY( ll>(ulong)(lpend-lp) || ll+zl>(ulong)(oend-o) ) ) return FD_ZLE3_ERR_CORRUPT;
    zle3_frame( o, lp, ll, zl );
    o  += ll+zl;
    lp += ll;
  }
  if( FD_UNLIKELY( (o!=oend) | (lp!=lpend) ) ) return FD_ZLE3_ERR_CORRUPT;
  return (long)data_sz;
}

#else /* !FD_HAS_AVX512 -- Firedancer requires AVX-512; this keeps the
         codec buildable and fuzzable on other hosts */

ulong
fd_zle3_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz ) ) return 0UL;
  return fd_zle3_stored( dst, src, data_sz );
}

long
fd_zle3_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       d = (uchar *)data;
  uchar const * s = (uchar const *)comp;

  if( FD_UNLIKELY( !comp_sz ) ) return data_sz==0UL ? 0L : FD_ZLE3_ERR_CORRUPT;
  ulong mode = s[0];
  if( mode==ZLE3_MODE_STORED ) {
    ulong n = comp_sz-1UL;
    if( FD_UNLIKELY( n>data_sz ) ) return FD_ZLE3_ERR_SPACE;
    fd_memcpy( d, s+1UL, n );
    return (long)n;
  }
  if( FD_UNLIKELY( (mode!=ZLE3_MODE_ZLE) | (comp_sz<2UL) ) ) return FD_ZLE3_ERR_CORRUPT;

  ulong hdr = 2UL, C = s[1];
  if( FD_UNLIKELY( C==0xFFUL ) ) {
    if( FD_UNLIKELY( comp_sz<ZLE3_HDR_MAX ) ) return FD_ZLE3_ERR_CORRUPT;
    C   = (ulong)s[2] | ((ulong)s[3]<<8) | ((ulong)s[4]<<16) | ((ulong)s[5]<<24);
    hdr = ZLE3_HDR_MAX;
  }
  if( FD_UNLIKELY( hdr+C>comp_sz ) ) return FD_ZLE3_ERR_CORRUPT;

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
      if( FD_UNLIKELY( cp+2UL>ce ) ) return FD_ZLE3_ERR_CORRUPT;
      zl = cp[1]; cp += 2UL;
    } else {
      if( FD_UNLIKELY( cp+9UL>ce ) ) return FD_ZLE3_ERR_CORRUPT;
      ll = (ulong)cp[1] | ((ulong)cp[2]<<8) | ((ulong)cp[3]<<16) | ((ulong)cp[4]<<24);
      zl = (ulong)cp[5] | ((ulong)cp[6]<<8) | ((ulong)cp[7]<<16) | ((ulong)cp[8]<<24);
      cp += 9UL;
    }
    if( FD_UNLIKELY( ll>(ulong)(lpend-lp) || ll+zl>(ulong)(oend-o) ) ) return FD_ZLE3_ERR_CORRUPT;
    fd_memcpy( o, lp, ll ); o += ll; lp += ll;
    fd_memset( o, 0, zl );  o += zl;
  }
  if( FD_UNLIKELY( (o!=oend) | (lp!=lpend) ) ) return FD_ZLE3_ERR_CORRUPT;
  return (long)data_sz;
}

#endif

char const *
fd_zle3_strerror( long res ) {
  switch( res ) {
  case FD_ZLE3_ERR_SPACE:   return "out of buffer space";
  case FD_ZLE3_ERR_CORRUPT: return "malformed compressed data";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
