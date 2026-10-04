#include "fd_zle12.h"
#include "fd_zle12_ref.h"
#include "../../util/fd_util.h"

#include <stdlib.h>

/* test_zle12 checks fd_zle12_compress against the byte at a time
   reference in fd_zle12_ref.h, round trips every case, and checks the
   n+n/145+6 bound and the 7 byte scratch limit. */

static uchar in  [ (8UL<<20)+64UL ] __attribute__((aligned(64)));
static uchar comp[ FD_ZLE12_COMPRESS_BOUND( 8UL<<20 ) ] __attribute__((aligned(64)));
static uchar comp8[ FD_ZLE12_COMPRESS_BOUND( 8UL<<20 ) ] __attribute__((aligned(64)));
static uchar out [ (8UL<<20)+64UL ] __attribute__((aligned(64)));

static void
round_trip( ulong n,
            char const * tag ) {
  fd_memset( in+n, 0xff, fd_ulong_align_up( n, 64UL )+64UL-n );
  fd_memset( comp, 0xcc, sizeof(comp) );
  ulong cs = fd_zle12_compress( comp, in, n );
  ulong c8 = fd_zle12_ref_compress( comp8, in, n );
  if( cs!=c8 || memcmp( comp, comp8, cs ) ) {
    FD_LOG_ERR(( "ref/zle12 output mismatch: %s n=%lu c12=%lu ref=%lu", tag, n, cs, c8 ));
  }
  FD_TEST( cs<=FD_ZLE12_COMPRESS_BOUND( n ) );
  FD_TEST( cs<=n+n/145UL+6UL );
  /* the encoder writes at most 7 scratch bytes past its output */
  ulong canary_end = fd_ulong_min( FD_ZLE12_COMPRESS_BOUND( n )+64UL, sizeof(comp) );
  for( ulong i=cs+7UL; i<canary_end; i++ ) FD_TEST( comp[i]==0xcc );
  fd_memset( out, 0x55, n );
  long ds = fd_zle12_decompress( out, n, comp, cs );
  if( ds!=(long)n || memcmp( out, in, n ) ) {
    FD_LOG_ERR(( "round trip failed: %s n=%lu cs=%lu ds=%ld", tag, n, cs, ds ));
  }
}

int
main( int     argc,
      char ** argv ) {
  fd_boot( &argc, &argv );
  fd_rng_t _rng[1]; fd_rng_t * rng = fd_rng_join( fd_rng_new( _rng, 1234U, 0UL ) );

  /* empty */
  FD_TEST( fd_zle12_compress( comp, in, 0UL )==0UL );
  FD_TEST( fd_zle12_decompress( out, 0UL, comp, 0UL )==0L );

  /* all zero, every size 1..64: the degenerate [ L=0, Z=n ] frame,
     which is both a leading-L=0 token and a terminal-Z one */
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "all-zero" );

  /* all nonzero -- no zero-byte mask bit is ever set, so the whole
     input must come out as one lone-L token */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "incompressible" );

  /* uniform random: occasional zero bytes, mostly isolated and so
     mostly isolated */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "uniform" );

  /* isolated single zeros between literals: X 0 X 0 X ... -- no zero
     pair, so this must come out as one lone-L
     token -- and it is the shape zle12_compress_tiny's change 1
     (masking d to len-1) exists for: unmasked, a trailing zero byte
     adjacent to the padding is framed as a phantom Z=1 run */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i&1UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "x0x0" );

  /* X 0 0 repeated: maximum frame density, one run every 3 bytes */
  for( ulong i=0UL; i<sizeof(in); i+=3UL ) { in[i]=1U; in[i+1]=0U; in[i+2]=0U; }
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "x00" );

  /* every input ends in a real zero run that reaches len-1 (the exact
     shape zle12_compress_tiny's change 1 exists for) */
  for( ulong n=2UL; n<=64UL; n++ ) {
    for( ulong i=0UL; i<n; i++ ) in[i] = (uchar)( (i+1UL<n) ? ( fd_rng_uint(rng)|1U ) : 0U );
    /* force at least the last 2 bytes zero so it's a real run,
       not an isolated trailing zero */
    if( n>=2UL ) { in[n-1] = 0U; in[n-2] = 0U; }
    round_trip( n, "trailing-zero-run" );
  }

  /* every input starts with a run of len-2 zero bytes then 2 literal
     bytes -- exercises a run whose candidate bits start at 0 */
  for( ulong n=3UL; n<=64UL; n++ ) {
    fd_memset( in, 0, n-2UL );
    in[n-2] = (uchar)( fd_rng_uint(rng)|1U );
    in[n-1] = (uchar)( fd_rng_uint(rng)|1U );
    round_trip( n, "leading-zero-run" );
  }

  /* structured: alternating runs of random length, many trials, sizes
     1..64 so every run-count/boundary combination gets exercised */
  for( ulong trial=0UL; trial<200000UL; trial++ ) {
    ulong n = 1UL + fd_rng_uint( rng ) % 64UL;
    ulong j = 0UL;
    int zero = (int)( fd_rng_uint( rng ) & 1U );
    while( j<n ) {
      ulong run = 1UL + fd_rng_uint( rng ) % ( zero ? 12UL : 8UL );
      run = fd_ulong_min( run, n-j );
      if( zero ) fd_memset( in+j, 0, run );
      else       for( ulong k=0UL; k<run; k++ ) in[j+k] = (uchar)( fd_rng_uint( rng ) | 1U );
      j += run; zero = !zero;
    }
    round_trip( n, "structured" );
  }

  /* >64B, incompressible: one lone-L token whose L field escapes past
     0xFF for the larger sizes */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  round_trip( 65UL,   "big-fallback" );
  round_trip( 100UL,  "big-fallback" );
  round_trip( 1000UL, "big-fallback" );
  round_trip( 4096UL, "big-fallback" );

  /* >64B, all zero: a single [ L=0, Z=n ] token, reaching the 256B
     skip-ahead and zle12_spill's word-at-a-time spill */
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=65UL; n<=4096UL; n++ ) round_trip( n, "all-zero-more" );
  for( ulong n=4096UL; n<=(1UL<<20); n<<=1 ) round_trip( n, "all-zero-more-big" );

  /* uniform random >64B (occasional zeros, mostly isolated) */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  for( ulong n=65UL; n<=2048UL; n++ ) round_trip( n, "uniform-more" );
  round_trip( 4UL<<20, "uniform-more-big" );

  /* structured: alternating runs of random length, many trials, across
     the 64B tiny/more boundary and well past it */
  for( ulong trial=0UL; trial<20000UL; trial++ ) {
    ulong n = 1UL + fd_rng_uint( rng ) % 8000UL;
    ulong j = 0UL;
    int zero = (int)( fd_rng_uint( rng ) & 1U );
    while( j<n ) {
      ulong run = 1UL + fd_rng_uint( rng ) % ( zero ? 40UL : 20UL );
      run = fd_ulong_min( run, n-j );
      if( zero ) fd_memset( in+j, 0, run );
      else       for( ulong k=0UL; k<run; k++ ) in[j+k] = (uchar)( fd_rng_uint( rng ) | 1U );
      j += run; zero = !zero;
    }
    round_trip( n, "structured-more" );
  }

  /* pathological: X 0 0 repeated past 64B (max frame density for
     the token stream) */
  for( ulong i=0UL; i<sizeof(in); i+=3UL ) { in[i]=1U; in[i+1]=0U; in[i+2]=0U; }
  for( ulong n=65UL; n<=6000UL; n+=3UL ) round_trip( n, "x00-more" );

  /* isolated single zeros between literals past 64B: no zero pair,
     so the whole input is one lone-L token */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i&1UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=65UL; n<=4000UL; n++ ) round_trip( n, "x0x0-more" );

  /* a long literal span followed by a huge zero run -- exercises
     zle12_spill and the 256B skip-ahead in both directions */
  for( ulong i=0UL; i<600UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+600UL, 0, (8UL<<20)-600UL );
  round_trip( 4UL<<20, "spill" );

  /* runs that straddle the ZLE12_TOK_ESC (0xFF) 1-byte/varint boundary
     in both L and Z fields -- the one path the earlier <=64B-only test
     could never reach (L,Z<=64<0xFF always there) */
  for( ulong trial=0UL; trial<3000UL; trial++ ) {
    ulong j = 0UL;
    int   zero = (int)( fd_rng_uint( rng ) & 1U );
    while( j<200000UL ) {
      ulong run = 240UL + fd_rng_uint( rng ) % 40UL; /* 240..279, brackets 255 */
      if( fd_rng_uint( rng ) & 3U ) run = 1UL + fd_rng_uint( rng ) % 12UL; /* mostly short */
      run = fd_ulong_min( run, 200000UL-j );
      if( zero ) fd_memset( in+j, 0, run );
      else       for( ulong k=0UL; k<run; k++ ) in[j+k] = (uchar)( fd_rng_uint( rng ) | 1U );
      j += run; zero = !zero;
    }
    round_trip( 65UL + fd_rng_uint( rng ) % ( j-64UL ), "esc-boundary" );
  }

  /* a zero run whose varint extension needs >1 byte (Z well past 255,
     well past ZLE12_TOK_ESC's first varint group) */
  fd_memset( in, 0, 8UL<<20 );
  for( ulong i=0UL; i<7UL; i++ ) in[i] = 1U;
  round_trip( 6UL<<20, "multibyte-varint-z" );

  /* a single literal run long enough to need the same for L (delta
     L-0xFF >= 128), isolating the per-token L varint escape */
  for( ulong i=0UL; i<2000UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+2000UL, 0, (1UL<<20)-2000UL );
  round_trip( 1UL<<20, "multibyte-varint-l" );
  for( ulong i=0UL; i<20000UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  for( ulong n=20002UL; n<20400UL; n+=7UL ) round_trip( n, "3byte-varint-l" );

  /* The shape fd_zle7 cannot encode at all: a multi-MiB incompressible
     span followed by a big zero run.  fd_zle7's type.more header holds
     the literal plane's size in 22 bits, so past 2^22-1 plane bytes it
     falls back to a verbatim copy and gives up the (much larger) zero
     run entirely.  fd_zle12 has no size field, so these must compress
     -- assert that, not just that they round trip. */
  {
    ulong tail = 2UL<<20;
    for( ulong lit=(4UL<<20)-64UL; lit<=(4UL<<20)+64UL; lit+=128UL ) {
      for( ulong i=0UL; i<lit; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
      fd_memset( in+lit, 0, tail );
      round_trip( lit+tail, "no-size-cap" );
      FD_TEST( fd_zle12_compress( comp, in, lit+tail ) < lit+tail );
    }
  }

  /* dense sweep across the tiny/small/big dispatch edges (64, 65, 254,
     255) and the scan_last / lz_last window: short runs of 1..6 so
     nibble escapes, 2-byte terminal runs and word-straddling runs are all
     common */
  for( ulong trial=0UL; trial<400000UL; trial++ ) {
    ulong n = 1UL + fd_rng_uint( rng ) % 520UL;
    ulong j = 0UL;
    int zero = (int)( fd_rng_uint( rng ) & 1U );
    while( j<n ) {
      ulong run = 1UL + fd_rng_uint( rng ) % 6UL;
      if( !( fd_rng_uint( rng ) & 15U ) ) run = 1UL + fd_rng_uint( rng ) % 300UL;
      run = fd_ulong_min( run, n-j );
      if( zero ) fd_memset( in+j, 0, run );
      else       for( ulong k=0UL; k<run; k++ ) in[j+k] = (uchar)( fd_rng_uint( rng ) | 1U );
      j += run; zero = !zero;
    }
    round_trip( n, "dense-sweep" );
  }

# if FD_HAS_ASAN || FD_HAS_MSAN
  /* sanitizer builds accept inputs without padding */
  for( ulong n=1UL; n<=64UL; n++ ) {
    uchar * exact = malloc( n );
    FD_TEST( exact );
    for( ulong i=0UL; i<n; i++ ) exact[i] = (uchar)( fd_rng_uint( rng ) & 1U );
    ulong cs = fd_zle12_compress( comp,  exact, n );
    ulong c8 = fd_zle12_ref_compress( comp8, exact, n );
    FD_TEST( cs==c8 && !memcmp( comp, comp8, cs ) );
    free( exact );
  }
# endif

  /* decompressor must be safe on arbitrary junk, including sizes that
     reach well into the token loop */
  for( ulong trial=0UL; trial<200000UL; trial++ ) {
    ulong cs = fd_rng_uint( rng ) % 256UL;
    for( ulong i=0UL; i<cs; i++ ) comp[i] = (uchar)fd_rng_uint( rng );
    ulong ds_sz = 1UL + fd_rng_uint( rng ) % 4096UL;
    long r = fd_zle12_decompress( out, ds_sz, comp, cs );
    FD_TEST( r<0L || (r<=(long)ds_sz) );
  }

  fd_rng_delete( fd_rng_leave( rng ) );
  FD_LOG_NOTICE(( "pass" ));
  fd_halt();
  return 0;
}
