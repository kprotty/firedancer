#include "../../util/fd_util.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* bench_zle7_vget -- isolated microbenchmark for zle7_vget (the LEB128
   varint reader fd_zle7.c's type.fill escape uses), per ZLE_TECHNIQUES.md's
   recommendation to measure a varint pack/unpack idea alone before it's
   built into anything.

   Candidate B below is fd_zle_ideas.h's `zle::varint_get` masked-load path
   (ZLE_VARINT_MASKED), transcribed to this file's C style, per the user's
   request:

     k = bzhi(~0, slen)
     x = vmovq(maskz_load(k, s))
     e = ~x & 0x8080808080808080
     b = tzcnt(e) + 1
     m = bzhi(0x7f7f7f7f7f7f7f7f, b)
     v = pext(x, m)
     s += b/8

   fd_zle_ideas.h's own comment on that path already claims it measured
   13-20% slower than a plain unconditional 8-byte load with a scalar
   fallback tail -- but that number isn't attributed to this box, and
   this file's job is to check it here rather than trust it secondhand.

   Not wired into fd_zle7.c -- this is call-count parity only (a single
   decode of a short buffer), not a full round trip. */

/* ==================================================================
   Candidate A: fd_zle7.c's zle7_vget, verbatim. */

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

#if FD_HAS_AVX512

/* ==================================================================
   Candidate B: masked 8-byte load + pext, per the pseudocode above.

   Matches zle7_vget's semantics exactly: max 7 bytes (49 payload bits --
   zle7_vget's sh>42 cap), fails (returns 0) if no terminator byte is
   found within min(slen-i,8) real bytes, or if the terminator's byte
   position would need bytes past slen (the maskz_load zero-fills lanes
   past `avail`, which would otherwise look like a false terminator). */

static inline int
zle7_vget_pext( uchar const * s,
                 ulong         slen,
                 ulong *       pi,
                 ulong *       out ) {
  ulong i     = *pi;
  ulong rem   = i<slen ? slen-i : 0UL;
  ulong avail = rem<8UL ? rem : 8UL;

  __mmask16 k = (__mmask16)_bzhi_u32( 0xFFFFU, (uint)avail );
  __m128i   x128 = _mm_maskz_loadu_epi8( k, s+i );
  ulong     x  = (ulong)_mm_cvtsi128_si64( x128 );

  ulong e = (~x) & 0x8080808080808080UL;
  if( FD_UNLIKELY( !e ) ) return 0; /* no stop byte in the available window */

  ulong b      = (ulong)_tzcnt_u64( e ) + 1UL;
  ulong nbytes = b>>3;
  if( FD_UNLIKELY( nbytes>avail || nbytes>7UL ) ) return 0; /* truncated, or would need an 8th byte */

  ulong m = _bzhi_u64( 0x7f7f7f7f7f7f7f7fUL, (uint)b );
  ulong v = _pext_u64( x, m );

  *pi  = i+nbytes;
  *out = v;
  return 1;
}

#endif /* FD_HAS_AVX512 */

#if FD_HAS_AVX512

/* ==================================================================
   Correctness: cross-check against zle7_vget (ground truth) over
   structured varints of every byte length 1..8 (8 must fail on both --
   see the sh>42 cap) plus pure random fuzz, truncated and untruncated. */

static void
check( fd_rng_t * rng ) {
  uchar buf[ 32 ];

  for( ulong trial=0UL; trial<20000000UL; trial++ ) {
    ulong nbytes = 1UL + fd_rng_uint( rng )%8UL; /* 1..8 -- 8 must fail */
    ulong expect = 0UL;
    int   sh     = 0;
    for( ulong k=0UL; k<nbytes; k++ ) {
      uchar b = (uchar)( fd_rng_uint( rng ) & 0x7f );
      int   last = ( k+1UL==nbytes );
      buf[ k ] = last ? b : (uchar)( b | 0x80U );
      if( !last ) { expect |= (ulong)b<<sh; sh += 7; }
      else          expect |= (ulong)b<<sh;
    }
    ulong extra = fd_rng_uint( rng )%4UL;
    for( ulong j=0UL; j<extra; j++ ) buf[ nbytes+j ] = (uchar)fd_rng_uint( rng );
    ulong slen = nbytes+extra;

    /* 1 in 4: truncate before the real terminator to exercise the
       "ran out of bytes" failure path on both sides. */
    if( nbytes>1UL && (fd_rng_uint( rng )%4UL)==0UL ) slen = 1UL+fd_rng_uint( rng )%(nbytes-1UL);

    ulong i1=0UL, i2=0UL, o1=0UL, o2=0UL;
    int r1 = zle7_vget     ( buf, slen, &i1, &o1 );
    int r2 = zle7_vget_pext( buf, slen, &i2, &o2 );
    if( FD_UNLIKELY( r1!=r2 ) )
      FD_LOG_ERR(( "mismatch (return): nbytes=%lu slen=%lu r1=%d r2=%d", nbytes, slen, r1, r2 ));
    if( r1 ) {
      if( FD_UNLIKELY( i1!=i2 || o1!=o2 || (slen>=nbytes && o1!=expect) ) )
        FD_LOG_ERR(( "mismatch (value): nbytes=%lu slen=%lu i1=%lu i2=%lu o1=%lu o2=%lu expect=%lu",
                     nbytes, slen, i1, i2, o1, o2, expect ));
    }
  }

  /* pure garbage, no structure, including slen==0 and slen>8 */
  for( ulong trial=0UL; trial<20000000UL; trial++ ) {
    ulong slen = fd_rng_uint( rng )%16UL;
    for( ulong j=0UL; j<slen; j++ ) buf[j] = (uchar)fd_rng_uint( rng );
    ulong i1=0UL, i2=0UL, o1=0UL, o2=0UL;
    int r1 = zle7_vget     ( buf, slen, &i1, &o1 );
    int r2 = zle7_vget_pext( buf, slen, &i2, &o2 );
    if( FD_UNLIKELY( r1!=r2 || (r1 && (i1!=i2 || o1!=o2)) ) )
      FD_LOG_ERR(( "fuzz mismatch: slen=%lu r1=%d r2=%d i1=%lu i2=%lu o1=%lu o2=%lu",
                   slen, r1, r2, i1, i2, o1, o2 ));
  }

  FD_LOG_NOTICE(( "correctness: pext variant agrees with zle7_vget on %d cases", 40000000 ));
}

/* ==================================================================
   Throughput: fixed-length varint streams (one byte length at a time,
   matching how the fill escape actually calls this -- one decode per
   frame, not a loop of them) plus a mixed/geometric workload standing
   in for the real distribution (almost always 1-2 bytes: this only
   fires when n==63 escapes to n-63, and the escape range needed in
   practice is small). */

#define N_CASE (1UL<<16)

static uchar in_buf[ N_CASE*16UL ] __attribute__((aligned(64)));

static ulong volatile sink;

static void
fill_fixed( fd_rng_t * rng,
            ulong      nbytes ) {
  for( ulong c=0UL; c<N_CASE; c++ ) {
    uchar * b = in_buf + c*16UL;
    for( ulong k=0UL; k<nbytes; k++ ) {
      uchar x = (uchar)( fd_rng_uint( rng ) & 0x7f );
      b[k] = ( k+1UL==nbytes ) ? x : (uchar)( x | 0x80U );
    }
    fd_memset( b+nbytes, 0xff, 16UL-nbytes ); /* nonzero pad: would break a masked-load bug that reads past nbytes */
  }
}

static void
fill_mixed( fd_rng_t * rng ) {
  for( ulong c=0UL; c<N_CASE; c++ ) {
    uchar * b = in_buf + c*16UL;
    /* geometric-ish: P(1B)=75%, P(2B)=~19%, tail out to 7B */
    ulong nbytes = 1UL;
    while( nbytes<7UL && (fd_rng_uint( rng )%4UL)==0UL ) nbytes++;
    for( ulong k=0UL; k<nbytes; k++ ) {
      uchar x = (uchar)( fd_rng_uint( rng ) & 0x7f );
      b[k] = ( k+1UL==nbytes ) ? x : (uchar)( x | 0x80U );
    }
    fd_memset( b+nbytes, 0xff, 16UL-nbytes );
  }
}

#define DIM (fd_log_style_dim())
#define RST (fd_log_style_normal())

static void
bench_one( char const * name,
           ulong        iter ) {
  ulong pi, out;

  for( ulong i=0UL; i<iter/8UL+1UL; i++ ) {
    uchar const * b = in_buf + (i%N_CASE)*16UL;
    pi = 0UL; zle7_vget     ( b, 16UL, &pi, &out ); sink += out;
    pi = 0UL; zle7_vget_pext( b, 16UL, &pi, &out ); sink += out;
  }

  long t0 = fd_log_wallclock();
  for( ulong i=0UL; i<iter; i++ ) {
    uchar const * b = in_buf + (i%N_CASE)*16UL;
    pi = 0UL;
    zle7_vget( b, 16UL, &pi, &out );
    sink += out;
  }
  long t1 = fd_log_wallclock();
  for( ulong i=0UL; i<iter; i++ ) {
    uchar const * b = in_buf + (i%N_CASE)*16UL;
    pi = 0UL;
    zle7_vget_pext( b, 16UL, &pi, &out );
    sink += out;
  }
  long t2 = fd_log_wallclock();

  double na = (double)(t1-t0)/(double)iter;
  double nb = (double)(t2-t1)/(double)iter;

  FD_LOG_NOTICE(( "%-16s %s|%s scalar %6.3f ns/call (%6.1f Mcall/s) %s|%s pext %6.3f ns/call (%6.1f Mcall/s) %s|%s %+6.1f%%",
                  name, DIM, RST, na, 1000./na, DIM, RST, nb, 1000./nb, DIM, RST, 100.*(na-nb)/na ));
}

int
main( int     argc,
      char ** argv ) {
  fd_boot( &argc, &argv );

  fd_rng_t _rng[1]; fd_rng_t * rng = fd_rng_join( fd_rng_new( _rng, 5150U, 0UL ) );
  FD_TEST( rng );

  FD_LOG_NOTICE(( "-- correctness --" ));
  check( rng );

  FD_LOG_NOTICE(( "-- throughput (cache-hot, %lu distinct calls cycled) --", N_CASE ));
  ulong const iter = 1UL<<26;

  fill_fixed( rng, 1UL ); bench_one( "1 byte",   iter );
  fill_fixed( rng, 2UL ); bench_one( "2 byte",   iter );
  fill_fixed( rng, 3UL ); bench_one( "3 byte",   iter );
  fill_fixed( rng, 4UL ); bench_one( "4 byte",   iter );
  fill_fixed( rng, 5UL ); bench_one( "5 byte",   iter );
  fill_fixed( rng, 6UL ); bench_one( "6 byte",   iter );
  fill_fixed( rng, 7UL ); bench_one( "7 byte",   iter );
  fill_mixed( rng );      bench_one( "mixed",    iter );

  FD_LOG_NOTICE(( "sink=%lu", sink ));

  fd_rng_delete( fd_rng_leave( rng ) );
  fd_halt();
  return 0;
}

#else /* !FD_HAS_AVX512 */

int
main( int     argc,
      char ** argv ) {
  fd_boot( &argc, &argv );
  FD_LOG_NOTICE(( "skip: FD_HAS_AVX512 not set" ));
  fd_halt();
  return 0;
}

#endif /* FD_HAS_AVX512 */
