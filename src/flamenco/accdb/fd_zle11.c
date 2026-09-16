#include "fd_zle11.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format (identical to fd_zle8): a stream of tokens, each producing
   L literal bytes then Z zero bytes.

     L         field
     literals  L bytes verbatim
     Z         field, omitted in the last token

   A field is one byte if <0xFF, else 0xFF followed by varint( v-0xFF ).

   The encoder breaks the literal run at every zero run of at least
   FD_ZLE11_MIN_Z bytes, or at least 2 bytes if it ends the input.

   The output never exceeds data_sz+data_sz/386+6 bytes: a token saves
   at least one byte unless L needs an escape, which takes a literal run
   of at least 255 bytes plus a zero run of 3 (see fd_zle8.c). */

#define FD_ZLE11_ESC       (0xFFUL) /* field escape marker */
#define FD_ZLE11_MIN_Z     (3UL)    /* shortest zero run worth a token */
#define FD_ZLE11_VARINT_SH (42)     /* max varint shift of a valid field */

/* fd_zle11_put_int writes field n at out and returns the cursor past it.
   If narrow, n<0xFF. */

static inline __attribute__((always_inline)) uchar *
fd_zle11_put_int( uchar * out,
                  ulong   n,
                  int     narrow ) {
  if( narrow || FD_LIKELY( n<FD_ZLE11_ESC ) ) {
    out[0] = (uchar)n;
    return out+1UL;
  }

  *out++ = (uchar)FD_ZLE11_ESC;
  n     -= FD_ZLE11_ESC;
  while( n>=0x80UL ) {
    *out++ = (uchar)( n | 0x80UL );
    n    >>= 7;
  }
  out[0] = (uchar)n;
  return out+1UL;
}

/* fd_zle11_get_int reads the varint following an escape marker at src.
   Sets *n to the field value and returns the cursor past it.  Returns
   NULL if the varint is truncated or too long. */

static inline __attribute__((always_inline)) uchar const *
fd_zle11_get_int( uchar const * src,
                  uchar const * src_end,
                  ulong *       n ) {
  ulong v = 0UL;
  ulong k = 0UL;
  uchar b;
  do {
    if( FD_UNLIKELY( ( src==src_end ) | ( k>FD_ZLE11_VARINT_SH ) ) ) return NULL;
    b  = *src++;
    v |= (ulong)( b & 0x7f )<<k;
    k += 7UL;
  } while( b & 0x80 );
  *n = v+FD_ZLE11_ESC;
  return src;
}

#if FD_HAS_AVX512

/* The copy and zero helpers below are unrolled on purpose: clang turns
   loops here into memcpy/memset calls. */

/* fd_zle11_copy64 copies sz<=64 bytes. */

static inline void
fd_zle11_copy64( uchar *       FD_RESTRICT d,
                 uchar const * FD_RESTRICT s,
                 ulong                     sz ) {
  __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
  _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
}

/* fd_zle11_copy copies sz bytes without touching bytes past d+sz or
   s+sz.  If narrow, sz<=254. */

static inline __attribute__((always_inline)) void
fd_zle11_copy( uchar *       FD_RESTRICT d,
               uchar const * FD_RESTRICT s,
               ulong                     sz,
               int                       narrow ) {
  if( FD_LIKELY( sz<=64UL ) ) {
    fd_zle11_copy64( d, s, sz );
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

/* fd_zle11_zero zeroes sz bytes without touching bytes past d+sz.
   Unlike fd_zle11_copy, it tests word by word, as most decoded zero
   runs of 65..256 bytes need only two stores. */

static inline void
fd_zle11_zero( uchar * d,
               ulong   sz ) {
  wwb_t z = wwb_zero();
  if( FD_LIKELY( sz<=64UL ) ) {
    _mm512_mask_storeu_epi8( (void *)d, _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ), z );
    return;
  }

  if( FD_LIKELY( sz<=256UL ) ) {
    /* eager branching instead of overlapping writes - measured to be faster */
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

/* The encoders find zero runs with a pair mask: bit i is set iff bytes
   i and i+1 are both zero.  Adding the lowest set bit to the mask
   carries through the run's bits and sets the bit of the run's last
   byte, so ctz gives both ends of the run without a scan. */

/* fd_zle11_compress_tiny encodes 1<=len<=64 bytes from one load. */

static ulong
fd_zle11_compress_tiny( uchar *       FD_RESTRICT dst,
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

  uchar * out  = dst;
  ulong   base = 0UL; /* start of pending literals */

  /* The highest pair bit is len-2, so the carry stays in the word.
     d==0 runs once with a run [64,65) that is never emitted. */
  do {
    ulong zpos = (ulong)_tzcnt_u64( d );
    d += (ulong)_blsi_u64( d );
    ulong zend = (ulong)_tzcnt_u64( d )+1UL;
    ulong Z    = zend-zpos;
    if( FD_LIKELY( ( Z>=FD_ZLE11_MIN_Z ) | ( zend==len ) ) ) {
      ulong L = zpos-base;
      out[0] = (uchar)L;
      fd_zle11_copy64( out+1UL, src+base, L );
      out[ L+1UL ] = (uchar)Z;
      out += L+2UL;
      base = zend;
    }
    d = (ulong)_blsr_u64( d ); /* clear the carry bit */
  } while( d );

  if( base<len ) { /* last token */
    ulong L = len-base;
    out[0] = (uchar)L;
    fd_zle11_copy64( out+1UL, src+base, L );
    out += L+1UL;
  }
  return (ulong)( out-dst );
}

/* fd_zle11_pairs returns the pair mask of the 64 byte word at p.  Reads
   65 bytes. */

static inline __attribute__((always_inline)) ulong
fd_zle11_pairs( uchar const * p ) {
  wwb_t v = wwb_or( wwb_ldu( p ), wwb_ldu( p+1UL ) );
  return (ulong)_mm512_testn_epi8_mask( v, v );
}

/* fd_zle11_emit writes the token for literals [base,zpos) and zeros
   [zpos,zend).  Returns the cursor past it. */

static inline __attribute__((always_inline)) uchar *
fd_zle11_emit( uchar *       out,
               uchar const * base,
               uchar const * zpos,
               uchar const * zend,
               int           narrow ) {
  ulong L = (ulong)( zpos-base );
  out = fd_zle11_put_int( out, L, narrow );
  fd_zle11_copy( out, base, L, narrow );
  return fd_zle11_put_int( out+L, (ulong)( zend-zpos ), narrow );
}

/* fd_zle11_zero_end returns the first non-zero byte in [src,end), or
   end if there is none.  Assumes src<end.  Reads nothing past end.

   Loops here and in fd_zle11_compress_core are bottom-tested on
   purpose: top-tested loops make clang spill a register in the literal
   copy. */

static inline __attribute__((always_inline)) uchar const *
fd_zle11_zero_end( uchar const * src,
                   uchar const * end,
                   int           narrow ) {
  if( src+64UL<end ) do {
    /* potential short run: test first 64 bytes */
    wwb_t v = wwb_ldu( src );
    ulong m = (ulong)_mm512_test_epi8_mask( v, v );
    if( m ) return src+_tzcnt_u64( m );
    src += 64UL;
    if( narrow ) continue;

    /* long run: skip 256 bytes at a time */
    while( src+256UL<end ) {
      wwb_t v0 = wwb_ldu( src       );
      wwb_t v1 = wwb_ldu( src+ 64UL );
      wwb_t v2 = wwb_ldu( src+128UL );
      wwb_t v3 = wwb_ldu( src+192UL );
      wwb_t t  = wwb_or( wwb_or( v0, v1 ), wwb_or( v2, v3 ) );
      if( FD_UNLIKELY( _mm512_test_epi8_mask( t, t ) ) ) {
        m = (ulong)_mm512_test_epi8_mask( v0, v0 ); if( m ) return src+       _tzcnt_u64( m );
        m = (ulong)_mm512_test_epi8_mask( v1, v1 ); if( m ) return src+ 64UL+_tzcnt_u64( m );
        m = (ulong)_mm512_test_epi8_mask( v2, v2 ); if( m ) return src+128UL+_tzcnt_u64( m );
        m = (ulong)_mm512_test_epi8_mask( v3, v3 );    return src+192UL+_tzcnt_u64( m );
      }
      src += 256UL;
    }
  } while( src+64UL<end );

  /* last 1..64 bytes, from the word ending at end */
  ulong rem = (ulong)( end-src );
  wwb_t v   = wwb_ldu( end-64UL );
  ulong m   = (ulong)_mm512_test_epi8_mask( v, v ) >> ( 64UL-rem );
  return src+fd_ulong_min( _tzcnt_u64( m ), rem );
}

/* fd_zle11_compress_core encodes n>64 bytes.  narrow is a compile time
   constant: 1 for n<255 (fields fit a byte, no 256 byte skips). */

static inline __attribute__((always_inline)) ulong
fd_zle11_compress_core( uchar *       FD_RESTRICT dst,
                        uchar const * FD_RESTRICT src,
                        ulong                     n,
                        int                       narrow ) {
  uchar *       out  = dst;
  uchar const * end  = src+n;
  uchar const * base = src; /* start of pending literals */

  /* Walk the input one word at a time while a full pair mask
     (src+65<=end) fits. */
  do {
    ulong d = fd_zle11_pairs( src );
    if( !d ) { /* not FD_UNLIKELY: an out of line block is slower here */
      /* Check a second word before skipping ahead.  Most accounts are
         too short for a skip to pay off. */
      src += 64UL;
      if( FD_UNLIKELY( src+64UL>=end ) ) break;
      d = fd_zle11_pairs( src );
      if( !d ) {
        src += 64UL;
        if( narrow ) continue;

        /* long literal run: skip 256 bytes at a time */
        while( src+256UL<end ) {
          wwb_t v0 = wwb_or( wwb_ldu( src       ), wwb_ldu( src+  1UL ) );
          wwb_t v1 = wwb_or( wwb_ldu( src+ 64UL ), wwb_ldu( src+ 65UL ) );
          wwb_t v2 = wwb_or( wwb_ldu( src+128UL ), wwb_ldu( src+129UL ) );
          wwb_t v3 = wwb_or( wwb_ldu( src+192UL ), wwb_ldu( src+193UL ) );
          wwb_t t  = _mm512_min_epu8( _mm512_min_epu8( v0, v1 ), _mm512_min_epu8( v2, v3 ) );
          if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) {
            d = (ulong)_mm512_testn_epi8_mask( v0, v0 ); if( d ) break;
            src += 64UL;
            d = (ulong)_mm512_testn_epi8_mask( v1, v1 ); if( d ) break;
            src += 64UL;
            d = (ulong)_mm512_testn_epi8_mask( v2, v2 ); if( d ) break;
            src += 64UL;
            d = (ulong)_mm512_testn_epi8_mask( v3, v3 ); /* nonzero */
            break;
          }
          src += 256UL;
        }
        if( !d ) continue;
      }
    }

    /* Emit every run in this word */
    for(;;) {
      uchar const * zpos = src+_tzcnt_u64( d );
      if( FD_UNLIKELY( __builtin_add_overflow( d, (ulong)_blsi_u64( d ), &d ) ) ) {
        /* run continues into the next word */
        src += 64UL;
        uchar const * zend = fd_zle11_zero_end( src, end, narrow );
        if( ( (ulong)( zend-zpos )>=FD_ZLE11_MIN_Z ) | ( zend==end ) ) {
          out  = fd_zle11_emit( out, base, zpos, zend, narrow );
          base = zend;
          if( zend==end ) return (ulong)( out-dst );
        }
        src = zend;
        break;
      }

      uchar const * zend = src+_tzcnt_u64( d )+1UL; /* <end */
      if( (ulong)( zend-zpos )>=FD_ZLE11_MIN_Z ) {
        out  = fd_zle11_emit( out, base, zpos, zend, narrow );
        base = zend;
      }

      d = (ulong)_blsr_u64( d ); /* clear the carry bit */
      if( !d ) { src += 64UL; break; }
    }
  } while( src+64UL<end );

  /* Last 1..64 bytes, from the word ending at end.  Its highest bit has
     no pair, so the carry stays in the word.  d==0 runs once with a run
     past end that is never emitted. */
  ulong rem = (ulong)( end-src );
  wwb_t v   = wwb_ldu( end-64UL );
  ulong d   = (ulong)_mm512_testn_epi8_mask( v, v ) >> ( 64UL-rem );
  /**/  d  &= d>>1;

  do {
    uchar const * zpos = src+_tzcnt_u64( d );
    d += (ulong)_blsi_u64( d );
    uchar const * zend = src+_tzcnt_u64( d )+1UL;
    if( ( (ulong)( zend-zpos )>=FD_ZLE11_MIN_Z ) | ( zend==end ) ) {
      out  = fd_zle11_emit( out, base, zpos, zend, narrow );
      base = zend;
    }
    d = (ulong)_blsr_u64( d );
  } while( d );

  if( base<end ) { /* last token */
    ulong L = (ulong)( end-base );
    out = fd_zle11_put_int( out, L, narrow );
    fd_zle11_copy( out, base, L, narrow );
    out += L;
  }
  return (ulong)( out-dst );
}

static ulong
fd_zle11_compress_small( uchar *       FD_RESTRICT dst,
                         uchar const * FD_RESTRICT src,
                         ulong                     n ) {
  return fd_zle11_compress_core( dst, src, n, 1 );
}

static ulong __attribute__((noinline))
fd_zle11_compress_big( uchar *       FD_RESTRICT dst,
                       uchar const * FD_RESTRICT src,
                       ulong                     n ) {
  return fd_zle11_compress_core( dst, src, n, 0 );
}

#else /* !FD_HAS_AVX512: emit a single literal token */

static inline void
fd_zle11_copy( uchar *       FD_RESTRICT d,
               uchar const * FD_RESTRICT s,
               ulong                     sz,
               int                       narrow ) {
  (void)narrow;
  fd_memcpy( d, s, sz );
}

static inline void
fd_zle11_zero( uchar * d,
               ulong   sz ) {
  fd_memset( d, 0, sz );
}

static ulong
fd_zle11_compress_tiny( uchar *       FD_RESTRICT dst,
                        uchar const * FD_RESTRICT src,
                        ulong                     len ) {
  uchar * out = fd_zle11_put_int( dst, len, 0 );
  fd_memcpy( out, src, len );
  return (ulong)( out-dst )+len;
}

#define fd_zle11_compress_small fd_zle11_compress_tiny
#define fd_zle11_compress_big   fd_zle11_compress_tiny

#endif /* FD_HAS_AVX512 */

ulong
fd_zle11_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz       ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<=64UL  ) ) return fd_zle11_compress_tiny ( dst, src, data_sz );
  if( FD_LIKELY  ( data_sz< 255UL ) ) return fd_zle11_compress_small( dst, src, data_sz );
  return                                     fd_zle11_compress_big  ( dst, src, data_sz );
}

/* The decoder runs until comp is exhausted; the last token is the one
   whose literals reach comp_sz.  Lengths read from comp are only ever
   compared against remaining space, never added to a pointer first. */

long
fd_zle11_decompress( void *       FD_RESTRICT data,
                     ulong                    data_sz,
                     void const * FD_RESTRICT comp,
                     ulong                    comp_sz ) {
  uchar *       out     = (uchar *)data;
  uchar *       dst_end = out+data_sz;
  uchar const * src     = (uchar const *)comp;
  uchar const * src_end = src+comp_sz;

  if( FD_UNLIKELY( !comp_sz ) ) return 0L;

  for(;;) {
    /* read literal count L */
    ulong L = (ulong)*src++;
    if( FD_UNLIKELY( L==FD_ZLE11_ESC ) ) {
      src = fd_zle11_get_int( src, src_end, &L );
      if( FD_UNLIKELY( !src ) ) return FD_ZLE11_ERR_STREAM;
    }
    
    /* copy L literals from stream */
    if( FD_UNLIKELY( L>(ulong)( dst_end-out ) ) ) return FD_ZLE11_ERR_SPACE;
    if( FD_UNLIKELY( L>(ulong)( src_end-src ) ) ) return FD_ZLE11_ERR_STREAM;
    fd_zle11_copy( out, src, L, 0 );
    out += L;
    src += L;
    if( src==src_end ) break; /* last token */

    /* read zero count Z */
    ulong Z = (ulong)*src++;
    if( FD_UNLIKELY( Z==FD_ZLE11_ESC ) ) {
      src = fd_zle11_get_int( src, src_end, &Z );
      if( FD_UNLIKELY( !src ) ) return FD_ZLE11_ERR_STREAM;
    }

    /* write Z zeroes */
    if( FD_UNLIKELY( Z>(ulong)( dst_end-out ) ) ) return FD_ZLE11_ERR_SPACE;
    fd_zle11_zero( out, Z );
    out += Z;
    if( src==src_end ) break;
  }

  return (long)( out-(uchar *)data );
}

char const *
fd_zle11_strerror( long res ) {
  switch( res ) {
  case FD_ZLE11_ERR_STREAM: return "malformed compressed data";
  case FD_ZLE11_ERR_SPACE:  return "out of buffer space";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
