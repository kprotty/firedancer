#include "fd_zle10.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format -- fd_zle8's, bit for bit (see fd_zle8.c)

     frame := token*
     token := L-field literal[L] [ Z-field ]
     field := b:u8 ; if b==0xFF: b += varint

   Interior zero runs are framed iff Z>=ZLE10_MIN_Z; a run that ends the
   input is framed for any Z>=2.  Runs are maximal and greedy, so the
   token stream is a pure function of the input and fd_zle10_compress
   produces exactly fd_zle8_compress's output.

   This file is fd_zle_pseudocode4 transcribed: the labels below are its
   labels, in its order.

   Encoder (>64 B)
   ------------------------------------------------------------------

   `src` walks the input in 64 B words, `base` is the start of the
   literals not yet emitted.

     zero_scan  Pair mask of the word at src (bit i: src[i]==src[i+1]==0)
                from one OR of two overlapping loads.  Empty -> next
                word; after one empty word (big only) stride 256 B.
     run        Walk the pair mask: adding the lowest bit carries through
                the run, and the bit the carry lands on is the run's last
                zero byte.  Carry out of bit 63 -> long_zero.  Every run
                in a word is resolved from that word's one load.
     long_zero  The run reaches the next word: find the first non-zero
                byte a word at a time (big: 256 B strides after one
                all-zero word).
     zero_scan_last  <65 B left: one end-aligned load resolves every
     long_zero_last  remaining candidate / the run's end.

   Only the _last blocks know where the input ends, so the loops above
   carry no tail masks and never read past data+data_sz.

   Two instantiations of zle10_compress_core: small (65..254 B, where
   L,Z<0xFF so fields are raw bytes) and big.

   No emit site checks for output space.  Per token the frame grows by
   |L-field|+|Z-field|-Z, which is -1 for the unescaped case, <=-249
   once Z escapes, and at worst +1 per 386 input bytes when L escapes
   (L=383,Z=3); with the terminal lone-L token's <=6 bytes that is
   out <= in_sz+in_sz/386+6, against FD_ZLE10_COMPRESS_BOUND's
   in_sz+in_sz/64+128 (fd_zle8.c "Output bound"). */

#define ZLE10_ESC   (0xFFUL) /* field escape marker: 0xFF then varint( v-0xFF ) */
#define ZLE10_MIN_Z (3UL)    /* shortest interior zero run worth framing */

#define ZLE10_VARINT_MAX_SH (42) /* no valid field needs an 8th 7-bit group */

/* ==================================================================
   Field codec */

/* zle10_put_int writes one field at out and returns the cursor past it
   (at most 6 bytes).  Each loop pass writes the byte staged by the
   previous one (first the escape marker) and stages the next 7-bit
   group with its continuation bit set; the last group is written with
   it cleared.  narrow callers pass n<ZLE10_ESC and fold to a store. */

static inline __attribute__((always_inline)) uchar *
zle10_put_int( uchar * out,
               ulong   n,
               int     narrow ) {
  if( narrow || FD_LIKELY( n<ZLE10_ESC ) ) { out[0] = (uchar)n; return out+1UL; }
  n -= ZLE10_ESC;
  ulong v = ZLE10_ESC;
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

/* zle10_copy64 copies sz<=64 bytes with one masked load/store. */

static inline void
zle10_copy64( uchar *       FD_RESTRICT d,
              uchar const * FD_RESTRICT s,
              ulong                     sz ) {
  __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
  _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
}

/* zle10_copy copies sz bytes, never writing at or past d+sz.  narrow
   callers pass sz<=254 and drop the huge test. */

static inline __attribute__((always_inline)) void
zle10_copy( uchar *       FD_RESTRICT d,
            uchar const * FD_RESTRICT s,
            ulong                     sz,
            int                       narrow ) {
  if( FD_LIKELY( sz<=64UL ) ) { zle10_copy64( d, s, sz ); return; }
  if( !narrow && FD_UNLIKELY( sz>256UL ) ) { fd_memcpy( d, s, sz ); return; }

  ulong tail  = sz-64UL;
  ulong mid_a = fd_ulong_min(  64UL, tail );
  ulong mid_b = fd_ulong_min( 128UL, tail );
  wwb_stu( d,       wwb_ldu( s       ) );
  wwb_stu( d+tail,  wwb_ldu( s+tail  ) );
  wwb_stu( d+mid_a, wwb_ldu( s+mid_a ) );
  wwb_stu( d+mid_b, wwb_ldu( s+mid_b ) );
}

/* zle10_zero zeroes sz bytes (decoder only).  Deliberately not
   zle10_copy's tiering: decoded zero runs cluster at 65..128 B, where
   this needs 2 stores behind a well predicted compare rather than a
   loop (measured: pumpswap amm +37%, meteora dlmm bins +27% decode
   against the counted loop, +1.6% on the mix).  Past 256 B the loop
   costs nothing, and past ZLE10_ZERO_THRESH fd_memset's ERMS/AVX path
   wins. */

#define ZLE10_ZERO_THRESH (512UL)

static inline void
zle10_zero( uchar * d,
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
  if( FD_UNLIKELY( sz>=ZLE10_ZERO_THRESH ) ) { fd_memset( d, 0, sz ); return; }
  ulong p = 0UL;
  do {
    wwb_stu( d+p, z );
    p += 64UL;
  } while( p+64UL<sz );
  wwb_stu( d+sz-64UL, z ); /* overlaps, stays in range */
}

/* ==================================================================
   zle10_compress_tiny -- 1<=len<=64.  The whole input is one 64 B
   load and every run falls out of bit twiddling on its pair mask. */

static ulong
zle10_compress_tiny( uchar *       FD_RESTRICT dst,
                     uchar const * FD_RESTRICT src,
                     ulong                     len ) {
  wwb_t     v  = wwb_ldu( src ); /* readable to align_up( len, 64 ) */
  ulong     d  = (ulong)_mm512_testn_epi8_mask( v, v );
  /**/      d  = _bzhi_u64( d & ( d>>1 ), (uint)(len-1UL) ); /* no pairs touching bytes past len */

  uchar * out  = dst;
  ulong   base = 0UL;
  ulong   zpos, zend, Z, L;

loop:
  zpos = (ulong)_tzcnt_u64( d ); /* d==0: 64, Z falls out below */
  d   += (ulong)_blsi_u64( d );
  zend = (ulong)_tzcnt_u64( d )+1UL; /* pair bits end by len-2: the carry lands by len-1 */

  Z = zend-zpos;
  if( FD_UNLIKELY( ( Z<ZLE10_MIN_Z ) & ( zend!=len ) ) ) goto loop_skip;
  L = zpos-base;
  out[0] = (uchar)L;
  zle10_copy64( out+1UL, src+base, L );
  out[ L+1UL ] = (uchar)Z;
  out += L+2UL;
  base = zend;

loop_skip:
  d = (ulong)_blsr_u64( d );
  if( d ) goto loop;
  if( base==len ) return (ulong)( out-dst );

  /* terminal lone-L token */
  L = len-base;
  out[0] = (uchar)L;
  zle10_copy64( out+1UL, src+base, L );
  out += L+1UL;
  return (ulong)( out-dst );
}

/* ==================================================================
   zle10_compress_core -- len>64.  narrow is a compile-time constant:
   1 for 65..254 B (single byte fields, no 256 B strides), 0 otherwise. */

static inline __attribute__((always_inline)) ulong
zle10_compress_core( uchar *       FD_RESTRICT dst,
                     uchar const * FD_RESTRICT src,
                     ulong                     n,
                     int                       narrow ) {
  uchar *       out  = dst;
  uchar const * end  = src+n;
  uchar const * base = src; /* literals not yet emitted start here */
  uchar const * zpos = src; /* current zero run is [zpos,zend) */
  uchar const * zend;
  ulong         d;          /* pair mask of the word being walked */
  ulong         m;          /* non-zero mask (long_zero) */
  ulong         off;        /* run end, relative to src (long_zero) */
  ulong         rem;        /* bytes left before end */
  ulong         Z, L;
  wwb_t         v, t;
  wwb_t         v0, v1, v2, v3;

# define EMIT( _zpos, _zend ) do {                              \
    ulong _L = (ulong)( (_zpos)-base );                         \
    out = zle10_put_int( out, _L, narrow );                     \
    zle10_copy( out, base, _L, narrow );                        \
    out = zle10_put_int( out+_L, (ulong)( (_zend)-(_zpos) ), narrow ); \
    base = (_zend);                                             \
  } while(0)

zero_scan: /* word at src; needs src+65<=end */
  if( FD_UNLIKELY( src+64UL>=end ) ) goto zero_scan_last;
  v = wwb_or( wwb_ldu( src ), wwb_ldu( src+1UL ) );
  d = (ulong)_mm512_testn_epi8_mask( v, v );
  if( FD_LIKELY( d ) ) goto run;
  src += 64UL;

  /* A second word before committing to the 256 B stride.  Striding after
     one empty word measured -13% on the mix: the account shapes that
     dominate it (spl-token 165 B, stake 200 B, ...) never reach a stride
     and pay for the test, and this unrolls their scan besides
     (spl-token +33%, stake +31%, spl-token mint nofz +48% encode). */
  if( FD_UNLIKELY( src+64UL>=end ) ) goto zero_scan_last;
  v = wwb_or( wwb_ldu( src ), wwb_ldu( src+1UL ) );
  d = (ulong)_mm512_testn_epi8_mask( v, v );
  if( FD_LIKELY( d ) ) goto run;
  src += 64UL;

  if( narrow || src+256UL>=end ) goto zero_scan;

zero_scan_big:
  /* Long literal span: stride 256 B until one of four words has a pair,
     then zero_scan_found_big resolves that word from v0..v3. */
  v0 = wwb_or( wwb_ldu( src       ), wwb_ldu( src+  1UL ) );
  v1 = wwb_or( wwb_ldu( src+ 64UL ), wwb_ldu( src+ 65UL ) );
  v2 = wwb_or( wwb_ldu( src+128UL ), wwb_ldu( src+129UL ) );
  v3 = wwb_or( wwb_ldu( src+192UL ), wwb_ldu( src+193UL ) );

  t = _mm512_min_epu8( _mm512_min_epu8( v0, v1 ), _mm512_min_epu8( v2, v3 ) );
  if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) goto zero_scan_found_big;
  src += 256UL;
  if( FD_LIKELY( src+256UL<end ) ) goto zero_scan_big;
  goto zero_scan;

zero_scan_found_big: /* each word ends by src+257<=end, as run needs */
  d = (ulong)_mm512_testn_epi8_mask( v0, v0 ); if( d ) goto run;
  src += 64UL;
  d = (ulong)_mm512_testn_epi8_mask( v1, v1 ); if( d ) goto run;
  src += 64UL;
  d = (ulong)_mm512_testn_epi8_mask( v2, v2 ); if( d ) goto run;
  src += 64UL;
  d = (ulong)_mm512_testn_epi8_mask( v3, v3 ); /* non-zero: the stride test proved it */

run: /* lowest bit of d starts a run */
  zpos = src+(ulong)__builtin_ctzll( d );
  if( FD_UNLIKELY( __builtin_add_overflow( d, (ulong)_blsi_u64( d ), &d ) ) ) goto long_zero;
  zend = src+(ulong)__builtin_ctzll( d )+1UL; /* <=src+64<end: never terminal */
  Z    = (ulong)( zend-zpos );
  if( Z<ZLE10_MIN_Z ) goto run_skip;
  EMIT( zpos, zend );

run_skip:
  d = (ulong)_blsr_u64( d ); /* drop the bit the carry landed on */
  if( d ) goto run;
  src += 64UL;
  goto zero_scan;

long_zero_found_big:
  m = (ulong)_mm512_test_epi8_mask( v0, v0 ); off = (ulong)_tzcnt_u64( m ); if( m ) goto long_zero_emit;
  src += 64UL;
  m = (ulong)_mm512_test_epi8_mask( v1, v1 ); off = (ulong)_tzcnt_u64( m ); if( m ) goto long_zero_emit;
  src += 64UL;
  m = (ulong)_mm512_test_epi8_mask( v2, v2 ); off = (ulong)_tzcnt_u64( m ); if( m ) goto long_zero_emit;
  src += 64UL;
  m = (ulong)_mm512_test_epi8_mask( v3, v3 ); off = (ulong)_tzcnt_u64( m ); /* non-zero: proven */
  goto long_zero_emit;

long_zero_big:
  /* Long zero span: stride 256 B until one of four words has a non-zero
     byte, then long_zero_found_big resolves it from v0..v3. */
  v0 = wwb_ldu( src       );
  v1 = wwb_ldu( src+ 64UL );
  v2 = wwb_ldu( src+128UL );
  v3 = wwb_ldu( src+192UL );

  t = wwb_or( wwb_or( v0, v1 ), wwb_or( v2, v3 ) );
  if( FD_UNLIKELY( _mm512_test_epi8_mask( t, t ) ) ) goto long_zero_found_big;
  src += 256UL;
  if( FD_LIKELY( src+256UL<end ) ) goto long_zero_big;
  goto long_zero_next;

long_zero: /* zeros cover [zpos,src+64] */
  src += 64UL;
long_zero_next: /* zeros cover [zpos,src) */
  if( FD_UNLIKELY( src+64UL>=end ) ) goto long_zero_last;

long_zero_scan:
  v   = wwb_ldu( src );
  m   = (ulong)_mm512_test_epi8_mask( v, v );
  off = (ulong)_tzcnt_u64( m );
  if( m ) goto long_zero_emit;
  src += 64UL;
  if( !narrow && src+256UL<end ) goto long_zero_big;
  if( src+64UL<end ) goto long_zero_scan;

long_zero_last:
  rem = (ulong)( end-src ); /* 1..64 */
  v   = wwb_ldu( end-64UL );
  m   = (ulong)_mm512_test_epi8_mask( v, v ) >> ( 64UL-rem );
  off = fd_ulong_min( (ulong)_tzcnt_u64( m ), rem ); /* m==0: zeros reach end */

long_zero_emit:
  src += off;
  zend = src;
  Z    = (ulong)( zend-zpos );
  if( ( Z<ZLE10_MIN_Z ) & ( zend!=end ) ) goto zero_scan;
  EMIT( zpos, zend );
  if( FD_LIKELY( src+64UL<end ) ) goto zero_scan;
  if( src>=end ) goto done; /* the run ended the input: base==end */

zero_scan_last: /* <65 B left, rem in [1,64] */
  rem = (ulong)( end-src );
  v   = wwb_ldu( end-64UL );
  d   = (ulong)_mm512_testn_epi8_mask( v, v ) >> ( 64UL-rem );
  d  &= d>>1; /* top zero has no pair: no carry can leave the word */

last_run:
  zpos = src+(ulong)_tzcnt_u64( d ); /* d==0: 64, Z falls out below */
  d   += (ulong)_blsi_u64( d );
  zend = src+(ulong)_tzcnt_u64( d )+1UL;
  Z    = (ulong)( zend-zpos );
  if( ( Z<ZLE10_MIN_Z ) & ( zend!=end ) ) goto last_run_skip;
  EMIT( zpos, zend );

last_run_skip:
  d = (ulong)_blsr_u64( d );
  if( d ) goto last_run;

  /* trailing: terminal lone-L token (fall-through only) */
  if( base>=end ) goto done;
  L   = (ulong)( end-base );
  out = zle10_put_int( out, L, narrow );
  zle10_copy( out, base, L, narrow );
  out += L;

done:
  return (ulong)( out-dst );

# undef EMIT
}

static ulong
zle10_compress_small( uchar *       FD_RESTRICT dst,
                      uchar const * FD_RESTRICT src,
                      ulong                     n ) {
  return zle10_compress_core( dst, src, n, 1 );
}

static ulong __attribute__((noinline))
zle10_compress_big( uchar *       FD_RESTRICT dst,
                    uchar const * FD_RESTRICT src,
                    ulong                     n ) {
  return zle10_compress_core( dst, src, n, 0 );
}

#else /* !FD_HAS_AVX512 -- a single literal token is a valid frame for
         any input; these stubs only keep the translation unit building */

static inline void
zle10_copy( uchar *       FD_RESTRICT d,
            uchar const * FD_RESTRICT s,
            ulong                     sz,
            int                       narrow ) {
  (void)narrow;
  fd_memcpy( d, s, sz );
}

static inline void
zle10_zero( uchar * d,
            ulong   sz ) {
  fd_memset( d, 0, sz );
}

static ulong
zle10_compress_tiny( uchar *       FD_RESTRICT dst,
                     uchar const * FD_RESTRICT src,
                     ulong                     len ) {
  uchar * out = zle10_put_int( dst, len, 0 );
  fd_memcpy( out, src, len );
  return (ulong)( out-dst )+len;
}

#define zle10_compress_small zle10_compress_tiny
#define zle10_compress_big   zle10_compress_tiny

#endif /* FD_HAS_AVX512 */

ulong
fd_zle10_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz       ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<=64UL  ) ) return zle10_compress_tiny ( dst, src, data_sz );
  if( FD_LIKELY  ( data_sz< 255UL ) ) return zle10_compress_small( dst, src, data_sz );
  return                                     zle10_compress_big  ( dst, src, data_sz );
}

/* ==================================================================
   Decoder

   Input driven: the frame ends when comp is exhausted, and the terminal
   token is the one whose literals run to comp_sz (it carries no Z
   field).  Every bound is tested as a subtraction so no length read off
   the wire is ever added to a pointer. */

long
fd_zle10_decompress( void *       FD_RESTRICT data,
                     ulong                    data_sz,
                     void const * FD_RESTRICT comp,
                     ulong                    comp_sz ) {
  uchar *       out;
  uchar *       dst_end;
  uchar const * src;
  uchar const * src_end;
  ulong         L, Z, k, v;
  uchar         b;

  if( FD_UNLIKELY( !comp_sz ) ) return 0L;

  /* data_sz and comp_sz describe the caller's own buffers, so (as in
     fd_zle, fd_zle8 and fd_zle9) they are a precondition rather than
     something to test -- testing them for wrap measured -1.1% on the
     mix and -9% on the small account shapes, for a check that only ever
     catches a caller bug.  Every length read off the *wire* below is
     bounded by subtraction, never by adding it to a pointer. */

  out     = (uchar *)data;
  dst_end = out+data_sz;
  src     = (uchar const *)comp;
  src_end = src+comp_sz;

loop:
  L = (ulong)*src++;
  if( FD_LIKELY( L<ZLE10_ESC ) ) goto copy_lit;
  k = 0UL;
  v = 0UL;
read_varint_L:
  if( FD_UNLIKELY( ( src==src_end ) | ( k>ZLE10_VARINT_MAX_SH ) ) ) goto err_stream;
  b  = *src++;
  v |= (ulong)( b & 0x7f )<<k;
  k += 7UL;
  if( b & 0x80 ) goto read_varint_L;
  L = v+ZLE10_ESC;

copy_lit:
  if( FD_UNLIKELY( L>(ulong)( dst_end-out ) ) ) goto err_space;
  if( FD_UNLIKELY( L>(ulong)( src_end-src ) ) ) goto err_stream;
  zle10_copy( out, src, L, 0 );
  out += L;
  src += L;
  if( src==src_end ) goto done; /* terminal token: no Z field */

  Z = (ulong)*src++;
  if( FD_LIKELY( Z<ZLE10_ESC ) ) goto write_zeroes;
  k = 0UL;
  v = 0UL;
read_varint_Z:
  if( FD_UNLIKELY( ( src==src_end ) | ( k>ZLE10_VARINT_MAX_SH ) ) ) goto err_stream;
  b  = *src++;
  v |= (ulong)( b & 0x7f )<<k;
  k += 7UL;
  if( b & 0x80 ) goto read_varint_Z;
  Z = v+ZLE10_ESC;

write_zeroes:
  if( FD_UNLIKELY( Z>(ulong)( dst_end-out ) ) ) goto err_space;
  zle10_zero( out, Z );
  out += Z;
  if( src<src_end ) goto loop;

done:
  return (long)( out-(uchar *)data );

err_stream: return FD_ZLE10_ERR_STREAM;
err_space:  return FD_ZLE10_ERR_SPACE;
}

char const *
fd_zle10_strerror( long res ) {
  switch( res ) {
  case FD_ZLE10_ERR_STREAM: return "malformed compressed data";
  case FD_ZLE10_ERR_SPACE:  return "out of buffer space";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
