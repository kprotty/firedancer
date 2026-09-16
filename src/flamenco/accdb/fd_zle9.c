#include "fd_zle9.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format -- fd_zle8's, bit for bit (see fd_zle8.c)

     frame := token*
     token := L-field literal[L] [ Z-field ]
     field := b:u8 ; if b==0xFF: b += varint

   Interior zero runs are framed iff Z>=ZLE9_MIN_Z; a run that ends the
   input is framed for any Z>=2.  Runs are maximal and greedy, so the
   token stream is a pure function of the input and fd_zle9_compress
   produces exactly fd_zle8_compress's output.

   Encoder (>64 B)
   ------------------------------------------------------------------

   Written as explicit branches over a pointer cursor, after
   fd_zle_pseudocode3.  `src` walks the input in 64 B words, `base` is
   the start of the literals not yet emitted.

     scan       Pair mask of the word at src (bit i: src[i]==src[i+1]==0)
                from one OR of two overlapping loads.  Empty -> next
                word; after two empty words (big only) stride 256 B.
     run        Walk the pair mask: adding the lowest bit carries through
                the run, and the bit the carry lands on is the run's last
                zero byte.  Carry out of bit 63 -> long_zero.  Every run
                in a word is resolved from that word's one load.
     long_zero  The run reaches the next word: find the first non-zero
                byte a word at a time (big: 256 B strides after one
                all-zero word).
     scan_last  <65 B left: one end-aligned load resolves every remaining
     lz_last    candidate / the run's end.

   Only scan_last and lz_last know where the input ends, so the loops
   above carry no tail masks and never read past data+data_sz.

   Two instantiations of zle9_compress_core: small (65..254 B, where
   L,Z<0xFF so fields are raw bytes and the output bound cannot be hit)
   and big.

   Measured and rejected (see ZLE_BENCH.md): restarting the scan at each
   run's end instead of walking the word's remaining pair bits (mix
   encode -27%). */

#define ZLE9_ESC   (0xFFUL) /* field escape marker: 0xFF then varint( v-0xFF ) */
#define ZLE9_MIN_Z (3UL)    /* shortest interior zero run worth framing */

/* ==================================================================
   Field codec */

static inline int
zle9_vget( uchar const * s,
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

/* zle9_get reads one field at *pi.  Returns 0 (leaving pi/out
   untouched) on truncated or overflowing input. */

static inline int
zle9_get( uchar const * s,
          ulong         slen,
          ulong *       pi,
          ulong *       out ) {
  ulong i = *pi;
  if( FD_UNLIKELY( i>=slen ) ) return 0;
  ulong v = (ulong)s[i++];
  if( FD_UNLIKELY( v==ZLE9_ESC ) ) {
    ulong ext;
    if( FD_UNLIKELY( !zle9_vget( s, slen, &i, &ext ) ) ) return 0;
    v = ZLE9_ESC+ext;
  }
  *pi  = i;
  *out = v;
  return 1;
}

/* zle9_put writes one field at out and returns the cursor past it (at
   most 6 bytes).  Each loop pass writes the byte staged by the previous
   one (first the escape marker) and stages the next 7-bit group with
   its continuation bit set; the last group is written with it cleared. */

static inline uchar *
zle9_put( uchar * out,
          ulong   n ) {
  if( FD_LIKELY( n<ZLE9_ESC ) ) { out[0] = (uchar)n; return out+1UL; }
  n -= ZLE9_ESC;
  ulong v = ZLE9_ESC;
  do {
    *out++ = (uchar)v;
    v      = ( n & 0x7fUL ) | 0x80UL;
    n    >>= 7;
  } while( n );
  out[0] = (uchar)( v & 0x7fUL );
  return out+1UL;
}

#if FD_HAS_AVX512

/* ==================================================================
   Copy / zero helpers.  Straight-line stores: a loop here folds to a
   memcpy/memset PLT call (see ZLE_TECHNIQUES.md). */

/* zle9_copy64 copies sz<=64 bytes with one masked load/store. */

static inline void
zle9_copy64( uchar *       FD_RESTRICT d,
             uchar const * FD_RESTRICT s,
             ulong                     sz ) {
  __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
  _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
}

/* zle9_copy copies sz bytes, never writing at or past d+sz.

   always_inline: left to clang it was emitted out of line in
   zle9_compress_big (a call per token at two of its emit sites);
   inlining it is +1.1% mix encode, up to +8% on 600 B-2 KiB shapes.
   (An unmasked 64 B store for the encoder's sz<=64 copies, relying on
   output slack, measured no faster than this masked one.) */

static inline __attribute__((always_inline)) void
zle9_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  if( FD_LIKELY  ( sz<= 64UL ) ) { zle9_copy64( d, s, sz ); return; }
  if( FD_UNLIKELY( sz> 256UL ) ) { fd_memcpy( d, s, sz );   return; }

  ulong tail  = sz-64UL;
  ulong mid_a = fd_ulong_min(  64UL, tail );
  ulong mid_b = fd_ulong_min( 128UL, tail );
  wwb_stu( d,       wwb_ldu( s       ) );
  wwb_stu( d+mid_a, wwb_ldu( s+mid_a ) );
  wwb_stu( d+mid_b, wwb_ldu( s+mid_b ) );
  wwb_stu( d+tail,  wwb_ldu( s+tail  ) );
}

/* zle9_zero zeroes sz bytes (decoder only).  Deliberately not
   zle9_copy's tiering: decoded zero runs cluster at 65..128 B, where
   this needs 2 stores behind a well predicted compare rather than 4
   (copy's tiering measured -13% decode on meteora dlmm bins). */

#define ZLE9_ZERO_THRESH (512UL) /* past here fd_memset's ERMS/AVX path wins */

static inline void
zle9_zero( uchar * d,
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
  if( FD_LIKELY( sz<ZLE9_ZERO_THRESH ) ) {
    ulong k = 0UL;
    do { wwb_stu( d+k, z ); k += 64UL; } while( k+64UL<=sz );
    wwb_stu( d+sz-64UL, z ); /* overlaps, stays in range */
    return;
  }
  fd_memset( d, 0, sz );
}

/* ==================================================================
   Scan primitives.  Unmasked full-word loads. */

/* zle9_pairs returns a mask with bit i set iff p[i] and p[i+1] are both
   zero.  Reads p[0,65). */

static inline ulong
zle9_pairs( uchar const * p ) {
  wwb_t v = wwb_or( wwb_ldu( p ), wwb_ldu( p+1UL ) );
  return (ulong)_mm512_testn_epi8_mask( v, v );
}

/* zle9_zeros / zle9_nonzeros return a mask with bit i set iff p[i] is /
   is not zero.  Read p[0,64). */

static inline ulong
zle9_zeros( uchar const * p ) {
  wwb_t v = wwb_ldu( p );
  return (ulong)_mm512_testn_epi8_mask( v, v );
}

static inline ulong
zle9_nonzeros( uchar const * p ) {
  wwb_t v = wwb_ldu( p );
  return (ulong)_mm512_test_epi8_mask( v, v );
}

/* ==================================================================
   Token emitters.  Both write L-field, the L literals at base, Z-field
   and return the cursor past the token.

   always_inline: left to clang, zle9_emit_var (and a 256 B stride test
   helper) were emitted out of line and the big path lost 3-6% encode. */

/* zle9_emit_u8: L,Z<ZLE9_ESC, fields are single bytes. */

static inline __attribute__((always_inline)) uchar *
zle9_emit_u8( uchar *       out,
              uchar const * base,
              ulong         L,
              ulong         Z ) {
  out[0] = (uchar)L;
  zle9_copy( out+1UL, base, L );
  out[ L+1UL ] = (uchar)Z;
  return out+L+2UL;
}

/* zle9_emit_var: escaping fields. */

static inline __attribute__((always_inline)) uchar *
zle9_emit_var( uchar *       out,
               uchar const * base,
               ulong         L,
               ulong         Z ) {
  out = zle9_put( out, L );
  zle9_copy( out, base, L );
  return zle9_put( out+L, Z );
}

/* ==================================================================
   zle9_compress_tiny -- 1<=len<=64.  fd_zle_pseudocode3's
   zle_compress_tiny with MIN_Z: the whole input is one masked load and
   every run falls out of bit twiddling on its pair mask. */

static ulong
zle9_compress_tiny( uchar *       FD_RESTRICT dst,
                    uchar const * FD_RESTRICT src,
                    ulong                     len ) {
  __mmask64 km = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)len ) );
  __m512i   v  = _mm512_maskz_loadu_epi8( km, src );
  ulong     m  = (ulong)_mm512_cmpeq_epi8_mask( v, _mm512_setzero_si512() );
  ulong     d  = _bzhi_u64( m & ( m>>1 ), (uint)(len-1UL) ); /* no pairs into the zeroed pad */

  uchar * out  = dst;
  ulong   base = 0UL;

  while( d ) {
    ulong zs = (ulong)__builtin_ctzll( d );
    ulong x  = d + (ulong)_blsi_u64( d );
    ulong ze = fd_ulong_min( (ulong)_tzcnt_u64( x )+1UL, len ); /* x==0: run reaches the pad */
    d        = (ulong)_blsr_u64( x );

    ulong Z = ze-zs;
    if( FD_UNLIKELY( (Z<ZLE9_MIN_Z) & (ze<len) ) ) continue;

    ulong L = zs-base;
    out[0] = (uchar)L;
    zle9_copy64( out+1UL, src+base, L );
    out[ L+1UL ] = (uchar)Z;
    out  += L+2UL;
    base  = ze;
  }

  if( base<len ) { /* terminal lone-L token */
    ulong L = len-base;
    out[0] = (uchar)L;
    zle9_copy64( out+1UL, src+base, L );
    out += L+1UL;
  }

  return (ulong)( out-dst );
}

/* ==================================================================
   zle9_compress_core -- len>64.  narrow is a compile-time constant:
   1 for 65..254 B (single byte fields, no bound check, no 256 B
   strides), 0 otherwise. */

static inline __attribute__((always_inline)) ulong
zle9_compress_core( uchar *       FD_RESTRICT dst,
                    uchar const * FD_RESTRICT src,
                    ulong                     n,
                    int                       narrow ) {
  uchar *       out  = dst;
  uchar *       cap  = dst+FD_ZLE9_COMPRESS_BOUND( n )-16UL;
  uchar const * end  = src+n;
  uchar const * base = src; /* literals not yet emitted start here */
  uchar const * zs;         /* current zero run is [zs,ze) */
  uchar const * ze;
  ulong         d;          /* pair mask of the word being walked */
  ulong         m;          /* non-zero mask (long_zero) */
  ulong         rem;        /* bytes left before end */
  wwb_t         a0, a1, a2, a3; /* 256 B stride words: pair words (lit_hit), raw loads (lz_hit) */

  /* EMIT frames the zero run [zs,ze) behind the literals [base,zs).
     The big bound check is unreachable (fd_zle8.c "Output bound") and
     degrades to one trailing literal token. */

# define EMIT( zs, ze ) do {                                                    \
    uchar const * _zs = (zs);                                                   \
    uchar const * _ze = (ze);                                                   \
    ulong         _L  = (ulong)( _zs-base );                                    \
    if( narrow ) {                                                              \
      out = zle9_emit_u8( out, base, _L, (ulong)( _ze-_zs ) );                  \
    } else {                                                                    \
      if( FD_UNLIKELY( out+_L+16UL>cap ) ) goto trailing;                       \
      out = zle9_emit_var( out, base, _L, (ulong)( _ze-_zs ) );                 \
    }                                                                           \
    base = _ze;                                                                 \
  } while(0)

scan: /* word at src; needs src+65<=end */
  if( FD_UNLIKELY( src+65UL>end ) ) goto scan_last;
  d = zle9_pairs( src );
  if( FD_LIKELY( d ) ) goto run;
  src += 64UL;

  if( FD_UNLIKELY( src+65UL>end ) ) goto scan_last;
  d = zle9_pairs( src );
  if( FD_LIKELY( d ) ) goto run;
  src += 64UL;

  if( !narrow ) {
    /* Long literal span: stride 256 B until one of four words has a
       pair, then lit_hit resolves that word from a0..a3.  Kept inline:
       behind a helper returning the mask as ulong, clang rotates this
       loop and reloads end from the stack (-1% mix encode). */
    while( src+257UL<=end ) {
      a0 = wwb_or( wwb_ldu( src       ), wwb_ldu( src+  1UL ) );
      a1 = wwb_or( wwb_ldu( src+ 64UL ), wwb_ldu( src+ 65UL ) );
      a2 = wwb_or( wwb_ldu( src+128UL ), wwb_ldu( src+129UL ) );
      a3 = wwb_or( wwb_ldu( src+192UL ), wwb_ldu( src+193UL ) );
      wwb_t t = _mm512_min_epu8( _mm512_min_epu8( a0, a1 ), _mm512_min_epu8( a2, a3 ) );
      if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) goto lit_hit;
      src += 256UL;
    }
  }
  goto scan;

run: /* lowest bit of d starts a run */
  zs = src+(ulong)__builtin_ctzll( d );
  if( FD_UNLIKELY( __builtin_add_overflow( d, (ulong)_blsi_u64( d ), &d ) ) ) goto long_zero;
  ze = src+(ulong)__builtin_ctzll( d )+1UL; /* <=src+64<end: never terminal */
  if( (ulong)( ze-zs )>=ZLE9_MIN_Z ) EMIT( zs, ze );
  d = (ulong)_blsr_u64( d ); /* drop the bit the carry landed on */
  if( d ) goto run;
  src += 64UL;
  goto scan;

long_zero: /* zeros cover [zs,src+64] */
  src += 64UL;
lz_scan:
  if( FD_UNLIKELY( src+64UL>=end ) ) goto lz_last;
  m = zle9_nonzeros( src );
  if( m ) {
    ze = src+(ulong)__builtin_ctzll( m );
    goto lz_emit;
  }
  src += 64UL;
  if( !narrow ) {
    /* Long zero span: stride 256 B until one of four words has a
       non-zero byte, then lz_hit resolves it from a0..a3. */
    while( src+256UL<end ) {
      a0 = wwb_ldu( src       );
      a1 = wwb_ldu( src+ 64UL );
      a2 = wwb_ldu( src+128UL );
      a3 = wwb_ldu( src+192UL );
      wwb_t t = wwb_or( wwb_or( a0, a1 ), wwb_or( a2, a3 ) );
      if( FD_UNLIKELY( _mm512_test_epi8_mask( t, t ) ) ) goto lz_hit;
      src += 256UL;
    }
  }
  goto lz_scan;
lz_last:
  /* scan_last reloads the word at end-64 rather than reusing m: reuse
     measured -6..-7% encode on token-2022 (ZLE_BENCH.md). */
  rem = (ulong)( end-src ); /* 1..64 */
  m   = zle9_nonzeros( end-64UL ) >> ( 64UL-rem );
  ze  = src+fd_ulong_min( (ulong)_tzcnt_u64( m ), rem ); /* m==0: zeros reach end */
lz_emit:
  if( ( (ulong)( ze-zs )>=ZLE9_MIN_Z ) | ( ze==end ) ) EMIT( zs, ze );
  src = ze;
  goto scan;

scan_last: /* <65 B left */
  rem = (ulong)( end-src ); /* 0..64 */
  if( FD_UNLIKELY( !rem ) ) goto trailing;
  d  = zle9_zeros( end-64UL ) >> ( 64UL-rem );
  d &= d>>1; /* top zero has no pair: no carry can leave the word */
last_run:
  if( !d ) goto trailing;
  zs = src+(ulong)__builtin_ctzll( d );
  d += (ulong)_blsi_u64( d );
  ze = src+(ulong)__builtin_ctzll( d )+1UL;
  d  = (ulong)_blsr_u64( d );
  if( ( (ulong)( ze-zs )>=ZLE9_MIN_Z ) | ( ze==end ) ) EMIT( zs, ze );
  goto last_run;

trailing: /* terminal lone-L token */
  if( base<end ) {
    ulong L = (ulong)( end-base );
    if( narrow ) *out++ = (uchar)L;
    else         out    = zle9_put( out, L );
    zle9_copy( out, base, L );
    out += L;
  }
  return (ulong)( out-dst );

lit_hit:
  /* A literal stride found a pair in one of a0..a3: take d for that
     word from its load (each word ends by src+257<=end, as run needs).
     Placed here, cold, rather than inside the stride loop: inline, it
     grew zle9_compress_big's hot layout and cost 4-5% encode on shapes
     that never stride (meteora dlmm bins, metaplex edition marker). */
  d = (ulong)_mm512_testn_epi8_mask( a0, a0 ); if( d ) goto run;
  src += 64UL;
  d = (ulong)_mm512_testn_epi8_mask( a1, a1 ); if( d ) goto run;
  src += 64UL;
  d = (ulong)_mm512_testn_epi8_mask( a2, a2 ); if( d ) goto run;
  src += 64UL;
  d = (ulong)_mm512_testn_epi8_mask( a3, a3 ); /* non-zero: the stride test proved it */
  goto run;

lz_hit:
  /* Likewise for a zero stride: the first non-zero byte is in one of
     a0..a3, all before end, so the run is interior (lz_emit). */
  m = (ulong)_mm512_test_epi8_mask( a0, a0 ); if( m ) goto lz_hit_word;
  src += 64UL;
  m = (ulong)_mm512_test_epi8_mask( a1, a1 ); if( m ) goto lz_hit_word;
  src += 64UL;
  m = (ulong)_mm512_test_epi8_mask( a2, a2 ); if( m ) goto lz_hit_word;
  src += 64UL;
  m = (ulong)_mm512_test_epi8_mask( a3, a3 ); /* non-zero: the stride test proved it */
lz_hit_word:
  ze = src+(ulong)__builtin_ctzll( m );
  goto lz_emit;

# undef EMIT
}

static ulong
zle9_compress_small( uchar *       FD_RESTRICT dst,
                     uchar const * FD_RESTRICT src,
                     ulong                     n ) {
  return zle9_compress_core( dst, src, n, 1 );
}

static ulong __attribute__((noinline))
zle9_compress_big( uchar *       FD_RESTRICT dst,
                   uchar const * FD_RESTRICT src,
                   ulong                     n ) {
  return zle9_compress_core( dst, src, n, 0 );
}

#else /* !FD_HAS_AVX512 -- a single literal token is a valid frame for
         any input; these stubs only keep the translation unit building */

static inline void
zle9_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  fd_memcpy( d, s, sz );
}

static inline void
zle9_zero( uchar * d,
           ulong   sz ) {
  fd_memset( d, 0, sz );
}

static ulong
zle9_compress_tiny( uchar *       FD_RESTRICT dst,
                    uchar const * FD_RESTRICT src,
                    ulong                     len ) {
  uchar * out = zle9_put( dst, len );
  fd_memcpy( out, src, len );
  return (ulong)( out-dst )+len;
}

#define zle9_compress_small zle9_compress_tiny
#define zle9_compress_big   zle9_compress_tiny

#endif /* FD_HAS_AVX512 */

ulong
fd_zle9_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz       ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<=64UL  ) ) return zle9_compress_tiny ( dst, src, data_sz );
  if( FD_LIKELY  ( data_sz< 255UL ) ) return zle9_compress_small( dst, src, data_sz );
  return                                     zle9_compress_big  ( dst, src, data_sz );
}

/* ==================================================================
   Decoder -- fd_zle8's */

long
fd_zle9_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       o    = (uchar *)data;
  uchar *       oend = o+data_sz;
  uchar const * s    = (uchar const *)comp;
  ulong         i    = 0UL;

  while( o<oend ) {

    ulong L;
    if( FD_UNLIKELY( !zle9_get( s, comp_sz, &i, &L ) ) ) return FD_ZLE9_ERR_CORRUPT;
    if( FD_UNLIKELY( ( L>(ulong)( oend-o ) ) | ( L>comp_sz-i ) ) ) return FD_ZLE9_ERR_CORRUPT;
    zle9_copy( o, s+i, L ); o += L; i += L;
    if( o==oend ) break; /* terminal token: no Z field */

    ulong Z;
    if( FD_UNLIKELY( !zle9_get( s, comp_sz, &i, &Z ) ) ) return FD_ZLE9_ERR_CORRUPT;
    if( FD_UNLIKELY( Z>(ulong)( oend-o ) ) ) return FD_ZLE9_ERR_CORRUPT;
    zle9_zero( o, Z ); o += Z;
  }

  if( FD_UNLIKELY( i!=comp_sz ) ) return FD_ZLE9_ERR_CORRUPT; /* exact consumption */
  return (long)data_sz;
}

char const *
fd_zle9_strerror( long res ) {
  switch( res ) {
  case FD_ZLE9_ERR_SPACE:   return "out of buffer space";
  case FD_ZLE9_ERR_CORRUPT: return "malformed compressed data";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
