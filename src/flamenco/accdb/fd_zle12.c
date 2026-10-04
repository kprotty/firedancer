#include "fd_zle12.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format: a stream of tokens, each producing L literal bytes then
   Z zero bytes.

     token     u8 min(L,15)<<4 | min(Z,15)
     L ext     varint( L-15 ), if L>=15
     Z ext     varint( Z-15 ), if Z>=15
     literals  L bytes verbatim

   varint is LEB128 (7 bits per byte, low group first, high bit set on
   all but the last byte).

   The encoder emits a token for every maximal zero run of at least 2
   bytes, so the output is canonical.  The last token covers the
   trailing literals with Z=0 and is omitted if the input ends with a
   zero run.

   A token saves at least one byte unless L needs a 2+ byte extension,
   which takes a literal run of at least 143 bytes.  The output never
   exceeds data_sz+data_sz/145+6 bytes.  The encoder may write up to 7
   scratch bytes past its output. */

#define FD_ZLE12_NIB       (15UL) /* nibble escape */
#define FD_ZLE12_VARINT_SZ (5UL)  /* max varint bytes of a valid field */

/* fd_zle12_get_vint_slow reads a varint at src.  Sets *n to its value and
   returns the cursor past it.  Returns NULL if it is truncated or too
   long. */

static uchar const *
fd_zle12_get_vint_slow( uchar const * src,
                        uchar const * src_end,
                        ulong *       n ) {
  ulong v = 0UL;
  for( ulong i=0UL; i<FD_ZLE12_VARINT_SZ; i++ ) {
    if( FD_UNLIKELY( src==src_end ) ) return NULL;
    ulong b = (ulong)*src++;
    v |= ( b & 0x7fUL )<<( 7UL*i );
    if( !( b & 0x80UL ) ) { *n = v; return src; }
  }
  return NULL;
}

/* fd_zle12_get_vint inlines the one byte case */

static inline __attribute__((always_inline)) uchar const *
fd_zle12_get_vint( uchar const * src,
                   uchar const * src_end,
                   ulong *       n ) {
  if( FD_LIKELY( ( src<src_end ) && !( *src & 0x80 ) ) ) { *n = (ulong)*src; return src+1UL; }
  return fd_zle12_get_vint_slow( src, src_end, n );
}

#if FD_HAS_AVX512

/* fd_zle12_put_vint writes varint v<2^32 at out.  Writes 8 bytes and
   returns the cursor past the varint. */

static inline __attribute__((always_inline)) uchar *
fd_zle12_put_vint( uchar * out,
                   ulong   v ) {
  ulong z  = (ulong)_lzcnt_u64( v );
  ulong n  = ( 640UL-9UL*z )>>6; /* ceil( bits/7 ), 1 for v==0 */
  ulong lo = _pdep_u64( v, 0x7f7f7f7f7f7f7f7fUL ) |
             _bzhi_u64( 0x8080808080808080UL, (uint)( 8UL*n-8UL ) );
  memcpy( out, &lo, 8UL );
  return out+n;
}

/* fd_zle12_put_vint_small writes varint v<256 at out.  Writes 2 bytes
   and returns the cursor past the varint.  Bit 7 of v is already the
   continuation bit. */

static inline __attribute__((always_inline)) uchar *
fd_zle12_put_vint_small( uchar * out,
                         ulong   v ) {
  ulong hi = v>>7;
  out[0] = (uchar)v;
  out[1] = (uchar)hi;
  return out+1UL+hi;
}

/* The copy and zero helpers below are unrolled on purpose: clang turns
   loops here into memcpy/memset calls. */

/* fd_zle12_copy64 copies sz<=64 bytes. */

static inline __attribute__((always_inline)) void
fd_zle12_copy64( uchar *       FD_RESTRICT d,
                 uchar const * FD_RESTRICT s,
                 ulong                     sz ) {
  __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
  _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
}

/* fd_zle12_copy copies sz bytes without touching bytes past d+sz or
   s+sz.  If narrow, sz<=256. */

static inline __attribute__((always_inline)) void
fd_zle12_copy( uchar *       FD_RESTRICT d,
               uchar const * FD_RESTRICT s,
               ulong                     sz,
               int                       narrow ) {
  if( FD_LIKELY( sz<=64UL ) ) {
    fd_zle12_copy64( d, s, sz );
    return;
  }

  /* over 256 and the optimized-memcpy records faster */
  if( !narrow && FD_UNLIKELY( sz>256UL ) ) {
    fd_memcpy( d, s, sz );
    return;
  }

  /* 65..256 bytes: four overlapping words */
  ulong tail  = sz-64UL;
  ulong mid_a = fd_ulong_min(  64UL, tail );
  ulong mid_b = fd_ulong_min( 128UL, tail );
  wwb_stu( d,       wwb_ldu( s       ) );
  wwb_stu( d+tail,  wwb_ldu( s+tail  ) );
  wwb_stu( d+mid_a, wwb_ldu( s+mid_a ) );
  wwb_stu( d+mid_b, wwb_ldu( s+mid_b ) );
}

/* fd_zle12_zero zeroes sz bytes without touching bytes past d+sz. */

static inline __attribute__((always_inline)) void
fd_zle12_zero( uchar * d,
               ulong   sz ) {
  wwb_t z = wwb_zero();
  if( FD_LIKELY( sz<=64UL ) ) {
    _mm512_mask_storeu_epi8( (void *)d, _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ), z );
    return;
  }

  if( FD_LIKELY( sz<=256UL ) ) {
    wwb_stu( d, z );
    if( FD_LIKELY( sz<=128UL ) ) { wwb_stu( d+sz-64UL, z ); return; }
    wwb_stu( d+64UL, z );
    if( sz<=192UL ) { wwb_stu( d+sz-64UL, z ); return; }
    wwb_stu( d+128UL, z );
    wwb_stu( d+sz-64UL, z );
    return;
  }

  if( FD_UNLIKELY( sz>=512UL ) ) { fd_memset( d, 0, sz ); return; }
  ulong p = 0UL;
  do {
    wwb_stu( d+p, z );
    p += 64UL;
  } while( p+64UL<sz );
  wwb_stu( d+sz-64UL, z );
}

/* fd_zle12_emit writes the token for literals [lp,zp) and zeros
   [zp,ze).  Returns the cursor past it.  If narrow, the input is at
   most 256 bytes. */

static inline __attribute__((always_inline)) uchar *
fd_zle12_emit( uchar *       out,
               uchar const * lp,
               uchar const * zp,
               uchar const * ze,
               int           narrow ) {
  ulong L = (ulong)( zp-lp );
  ulong Z = (ulong)( ze-zp );
  *out++ = (uchar)( ( fd_ulong_min( L, FD_ZLE12_NIB )<<4 ) | fd_ulong_min( Z, FD_ZLE12_NIB ) );
  if( narrow ) {
    if( L>=FD_ZLE12_NIB ) out = fd_zle12_put_vint_small( out, L-FD_ZLE12_NIB );
    if( Z>=FD_ZLE12_NIB ) out = fd_zle12_put_vint_small( out, Z-FD_ZLE12_NIB );
  } else {
    if( L>=FD_ZLE12_NIB ) out = fd_zle12_put_vint( out, L-FD_ZLE12_NIB );
    if( Z>=FD_ZLE12_NIB ) out = fd_zle12_put_vint( out, Z-FD_ZLE12_NIB );
  }
  fd_zle12_copy( out, lp, L, narrow );
  return out+L;
}

/* The encoders find zero runs with a pair mask: bit i is set iff bytes
   i and i+1 are both zero.  Adding the lowest set bit to the mask
   carries through the run's bits and sets the bit of the run's last
   byte, so tzcnt gives both ends of the run without a scan. */

/* fd_zle12_compress_tiny encodes 1<=len<=64 bytes from one load.  L and
   Z are at most 64, so extensions are one byte. */

static ulong
fd_zle12_compress_tiny( uchar *       FD_RESTRICT dst,
                        uchar const * FD_RESTRICT src,
                        ulong                     len ) {
# if FD_HAS_ASAN || FD_HAS_MSAN
  /* sanitizers would flag the read past len */
  wwb_t v = _mm512_maskz_loadu_epi8( _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)len ) ), src );
# else
  wwb_t v = wwb_ldu( src ); /* may read up to align_up( len, 64 ) */
# endif
  ulong d = (ulong)_mm512_testn_epi8_mask( v, v );
  /**/  d = _bzhi_u64( d & ( d>>1 ), (uint)(len-1UL) ); /* drop pairs past len */

  uchar * out = dst;
  ulong   lp  = 0UL; /* start of pending literals */

  /* The highest pair bit is len-2, so the carry stays in the word */
  while( d ) {
    ulong zp = (ulong)_tzcnt_u64( d );
    d += (ulong)_blsi_u64( d );
    ulong ze = (ulong)_tzcnt_u64( d )+1UL;
    ulong L  = zp-lp;
    ulong Z  = ze-zp;
    *out++ = (uchar)( ( fd_ulong_min( L, FD_ZLE12_NIB )<<4 ) | fd_ulong_min( Z, FD_ZLE12_NIB ) );
    *out   = (uchar)( L-FD_ZLE12_NIB ); out += (ulong)( L>=FD_ZLE12_NIB );
    *out   = (uchar)( Z-FD_ZLE12_NIB ); out += (ulong)( Z>=FD_ZLE12_NIB );
    fd_zle12_copy64( out, src+lp, L );
    out += L;
    lp   = ze;
    d    = (ulong)_blsr_u64( d ); /* clear the carry bit */
  }

  if( lp<len ) { /* last token */
    ulong L = len-lp;
    *out++ = (uchar)( fd_ulong_min( L, FD_ZLE12_NIB )<<4 );
    *out   = (uchar)( L-FD_ZLE12_NIB ); out += (ulong)( L>=FD_ZLE12_NIB );
    fd_zle12_copy64( out, src+lp, L );
    out += L;
  }
  return (ulong)( out-dst );
}

/* fd_zle12_compress_core encodes len>64 bytes.  Loads are unmasked and
   never read past end: a pair load at src needs src+64<end, and the
   last 1..64 bytes come from the word ending at end.  narrow is a
   compile time constant: 1 for len<=256 (short varints, no 256 byte
   skips). */

static inline __attribute__((always_inline)) ulong
fd_zle12_compress_core( uchar *       FD_RESTRICT dst,
                        uchar const * FD_RESTRICT src,
                        ulong                     len,
                        int                       narrow ) {
  uchar *       out = dst;
  uchar const * end = src+len;
  uchar const * lp  = src; /* start of pending literals */
  uchar const * zp;
  uchar const * ze;
  ulong         d;
  ulong         m;
  wwb_t         v;

find_zmask:
  v = wwb_or( wwb_ldu( src ), wwb_ldu( src+1UL ) );
  d = (ulong)_mm512_testn_epi8_mask( v, v );
  if( d ) goto found_zmask;
  if( narrow ) goto next_word;

  /* Check a second word before skipping ahead */
  src += 64UL;
  if( src+64UL>=end ) goto last_word;
  v = wwb_or( wwb_ldu( src ), wwb_ldu( src+1UL ) );
  d = (ulong)_mm512_testn_epi8_mask( v, v );
  if( d ) goto found_zmask;
  src += 64UL;

  /* long literal run: skip 256 bytes at a time */
  while( src+256UL<end ) {
    wwb_t v0 = wwb_or( wwb_ldu( src       ), wwb_ldu( src+  1UL ) );
    wwb_t v1 = wwb_or( wwb_ldu( src+ 64UL ), wwb_ldu( src+ 65UL ) );
    wwb_t v2 = wwb_or( wwb_ldu( src+128UL ), wwb_ldu( src+129UL ) );
    wwb_t v3 = wwb_or( wwb_ldu( src+192UL ), wwb_ldu( src+193UL ) );
    wwb_t t  = _mm512_min_epu8( _mm512_min_epu8( v0, v1 ), _mm512_min_epu8( v2, v3 ) );
    if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) {
      d = (ulong)_mm512_testn_epi8_mask( v0, v0 ); if( d ) goto found_zmask;
      src += 64UL;
      d = (ulong)_mm512_testn_epi8_mask( v1, v1 ); if( d ) goto found_zmask;
      src += 64UL;
      d = (ulong)_mm512_testn_epi8_mask( v2, v2 ); if( d ) goto found_zmask;
      src += 64UL;
      d = (ulong)_mm512_testn_epi8_mask( v3, v3 );
      goto found_zmask;
    }
    src += 256UL;
  }
  if( src+64UL<end ) goto find_zmask;
  goto last_word;

next_word:
  src += 64UL;
  if( src+64UL<end ) goto find_zmask;
last_word:
  if( src>=end ) goto last_literals;
  /* highest pair bit is end-src-2, so the carry stays in the word */
  v = wwb_ldu( end-64UL );
  d = (ulong)_mm512_testn_epi8_mask( v, v ) >> ( 64UL-(ulong)( end-src ) );
  d &= d>>1;
  if( !d ) goto last_literals;
found_zmask:
  zp = src+_tzcnt_u64( d );
  if( FD_UNLIKELY( __builtin_add_overflow( d, (ulong)_blsi_u64( d ), &d ) ) ) goto find_zend;
  ze  = src+_tzcnt_u64( d )+1UL;
  out = fd_zle12_emit( out, lp, zp, ze, narrow );
  lp  = ze;
  d   = (ulong)_blsr_u64( d ); /* clear the carry bit */
  if( d ) goto found_zmask;
  goto next_word;

find_zend: /* the run continues into the next word */
  src += 64UL;
  if( src+64UL>=end ) goto find_zend_last;
find_zend_loop:
  v = wwb_ldu( src );
  m = (ulong)_mm512_test_epi8_mask( v, v );
  if( m ) goto found_zend;
  src += 64UL;
  if( !narrow ) {
    /* long zero run: skip 256 bytes at a time */
    while( src+256UL<end ) {
      wwb_t v0 = wwb_ldu( src       );
      wwb_t v1 = wwb_ldu( src+ 64UL );
      wwb_t v2 = wwb_ldu( src+128UL );
      wwb_t v3 = wwb_ldu( src+192UL );
      wwb_t t  = wwb_or( wwb_or( v0, v1 ), wwb_or( v2, v3 ) );
      if( FD_UNLIKELY( _mm512_test_epi8_mask( t, t ) ) ) {
        m = (ulong)_mm512_test_epi8_mask( v0, v0 ); if( m ) goto found_zend;
        src += 64UL;
        m = (ulong)_mm512_test_epi8_mask( v1, v1 ); if( m ) goto found_zend;
        src += 64UL;
        m = (ulong)_mm512_test_epi8_mask( v2, v2 ); if( m ) goto found_zend;
        src += 64UL;
        m = (ulong)_mm512_test_epi8_mask( v3, v3 );
        goto found_zend;
      }
      src += 256UL;
    }
  }
  if( src+64UL<end ) goto find_zend_loop;
find_zend_last:
  v  = wwb_ldu( end-64UL );
  m  = (ulong)_mm512_test_epi8_mask( v, v ) >> ( 64UL-(ulong)( end-src ) );
  ze = src+fd_ulong_min( _tzcnt_u64( m ), (ulong)( end-src ) ); /* may be end */
  goto emit_zend;
found_zend:
  ze = src+_tzcnt_u64( m );
emit_zend:
  out = fd_zle12_emit( out, lp, zp, ze, narrow );
  lp  = ze;
  src = ze;
  if( src+64UL<end ) goto find_zmask;
  goto last_word;

last_literals:
  if( lp<end ) {
    ulong L = (ulong)( end-lp );
    *out++ = (uchar)( fd_ulong_min( L, FD_ZLE12_NIB )<<4 );
    if( L>=FD_ZLE12_NIB ) out = narrow ? fd_zle12_put_vint_small( out, L-FD_ZLE12_NIB )
                                       : fd_zle12_put_vint      ( out, L-FD_ZLE12_NIB );
    fd_zle12_copy( out, lp, L, narrow );
    out += L;
  }
  return (ulong)( out-dst );
}

static ulong
fd_zle12_compress_small( uchar *       FD_RESTRICT dst,
                         uchar const * FD_RESTRICT src,
                         ulong                     len ) {
  return fd_zle12_compress_core( dst, src, len, 1 );
}

static ulong __attribute__((noinline))
fd_zle12_compress_big( uchar *       FD_RESTRICT dst,
                       uchar const * FD_RESTRICT src,
                       ulong                     len ) {
  return fd_zle12_compress_core( dst, src, len, 0 );
}

#else /* !FD_HAS_AVX512: emit a single literal token */

static ulong
fd_zle12_compress_tiny( uchar *       FD_RESTRICT dst,
                        uchar const * FD_RESTRICT src,
                        ulong                     len ) {
  uchar * out = dst;
  *out++ = (uchar)( fd_ulong_min( len, FD_ZLE12_NIB )<<4 );
  if( len>=FD_ZLE12_NIB ) {
    ulong v = len-FD_ZLE12_NIB;
    while( v>=0x80UL ) { *out++ = (uchar)( v | 0x80UL ); v >>= 7; }
    *out++ = (uchar)v;
  }
  fd_memcpy( out, src, len );
  return (ulong)( out-dst )+len;
}

#define fd_zle12_compress_small fd_zle12_compress_tiny
#define fd_zle12_compress_big   fd_zle12_compress_tiny

#endif /* FD_HAS_AVX512 */

ulong
fd_zle12_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz      ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<= 64UL ) ) return fd_zle12_compress_tiny ( dst, src, data_sz );
  if( FD_LIKELY  ( data_sz<=256UL ) ) return fd_zle12_compress_small( dst, src, data_sz );
  return                                     fd_zle12_compress_big  ( dst, src, data_sz );
}

/* The decoder runs until comp is exhausted.  Lengths read from comp are
   only ever compared against remaining space, never added to a pointer
   first. */

long
fd_zle12_decompress( void *       FD_RESTRICT data,
                     ulong                    data_sz,
                     void const * FD_RESTRICT comp,
                     ulong                    comp_sz ) {
  uchar *       out     = (uchar *)data;
  uchar *       dst_end = out+data_sz;
  uchar const * src     = (uchar const *)comp;
  uchar const * src_end = src+comp_sz;

  if( FD_UNLIKELY( !comp_sz ) ) return 0L;

  do {
    ulong tok = (ulong)*src++;
    ulong L   = tok>>4;
    ulong Z   = tok & FD_ZLE12_NIB;
    if( FD_UNLIKELY( L==FD_ZLE12_NIB ) ) {
      ulong x;
      src = fd_zle12_get_vint( src, src_end, &x );
      if( FD_UNLIKELY( !src ) ) return FD_ZLE12_ERR_STREAM;
      L += x;
    }
    if( FD_UNLIKELY( Z==FD_ZLE12_NIB ) ) {
      ulong x;
      src = fd_zle12_get_vint( src, src_end, &x );
      if( FD_UNLIKELY( !src ) ) return FD_ZLE12_ERR_STREAM;
      Z += x;
    }

    /* L,Z < 2^35+15: no overflow */
    if( FD_UNLIKELY( L>(ulong)( src_end-src ) ) ) return FD_ZLE12_ERR_STREAM;
    if( FD_UNLIKELY( L+Z>(ulong)( dst_end-out ) ) ) return FD_ZLE12_ERR_SPACE;

#   if FD_HAS_AVX512
    if( FD_LIKELY( L+Z<=64UL ) ) {
      /* one store: the masked load zero fills past L */
      __mmask64 lm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)L     ) );
      __mmask64 om = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)(L+Z) ) );
      _mm512_mask_storeu_epi8( (void *)out, om, _mm512_maskz_loadu_epi8( lm, src ) );
    } else {
      fd_zle12_copy( out,   src, L, 0 );
      fd_zle12_zero( out+L,      Z );
    }
#   else
    fd_memcpy( out,   src, L );
    fd_memset( out+L, 0,   Z );
#   endif

    out += L+Z;
    src += L;
  } while( src<src_end );

  return (long)( out-(uchar *)data );
}

char const *
fd_zle12_strerror( long res ) {
  switch( res ) {
  case FD_ZLE12_ERR_STREAM: return "malformed compressed data";
  case FD_ZLE12_ERR_SPACE:  return "out of buffer space";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
