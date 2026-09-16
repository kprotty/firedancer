#include "fd_zle4.h"
#include "../../util/fd_util.h"

static uchar in  [ (8UL<<20)+64UL ] __attribute__((aligned(64)));
static uchar comp[ FD_ZLE4_COMPRESS_BOUND( 8UL<<20 ) ] __attribute__((aligned(64)));
static uchar out [ (8UL<<20)+64UL ] __attribute__((aligned(64)));

static void
round_trip( ulong n,
            char const * tag ) {
  fd_memset( in+n, 0xff, fd_ulong_align_up( n, 64UL )+64UL-n );
  fd_memset( comp, 0xcc, sizeof(comp) );
  ulong cs = fd_zle4_compress( comp, in, n );
  FD_TEST( cs<=FD_ZLE4_COMPRESS_BOUND( n ) );
  fd_memset( out, 0x55, n );
  long ds = fd_zle4_decompress( out, n, comp, cs );
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
  FD_TEST( fd_zle4_compress( comp, in, 0UL )==0UL );
  FD_TEST( fd_zle4_decompress( out, 0UL, comp, 0UL )==0L );

  /* all zero, all sizes across the small/big boundary */
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=1UL; n<=4096UL; n++ ) round_trip( n, "all-zero" );
  for( ulong n=4096UL; n<=(1UL<<20); n<<=1 ) round_trip( n, "all-zero-big" );

  /* all nonzero (incompressible -> stored) */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  for( ulong n=1UL; n<=1024UL; n++ ) round_trip( n, "incompressible" );
  round_trip( 1UL<<20, "incompressible-big" );

  /* uniform random (has occasional zeros) */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  for( ulong n=1UL; n<=2048UL; n++ ) round_trip( n, "uniform" );
  round_trip( 8UL<<20, "uniform-8m" );

  /* structured: alternating runs of random length */
  for( ulong trial=0UL; trial<4000UL; trial++ ) {
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
    round_trip( n, "structured" );
  }

  /* pathological: X 0 0 repeated (max frame density) */
  for( ulong i=0UL; i<sizeof(in); i+=3UL ) { in[i]=1U; in[i+1]=0U; in[i+2]=0U; }
  for( ulong n=3UL; n<=6000UL; n+=3UL ) round_trip( n, "x00" );

  /* isolated single zeros between literals: X 0 X 0 X ... */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i&1UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=1UL; n<=4000UL; n++ ) round_trip( n, "x0x0" );

  /* a long literal span followed by a huge zero run (skip-ahead + spill) */
  for( ulong i=0UL; i<600UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+600UL, 0, (8UL<<20)-600UL );
  round_trip( 4UL<<20, "spill" );

  /* runs that straddle the 1-byte / u32 control encoding boundary
     (250..260) in both fields, small and large inputs */
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
    round_trip( 1UL + fd_rng_uint( rng ) % j, "ext-boundary" );
  }

  /* a >4 GiB-length field is impossible (FD_ZLE4_MAX_SZ), but exercise a
     multi-MiB single zero run (u32 ext) and a multi-MiB literal run */
  fd_memset( in, 0, 8UL<<20 );
  for( ulong i=0UL; i<7UL; i++ ) in[i] = 1U;
  round_trip( 6UL<<20, "u32-zero-ext" );
  for( ulong i=0UL; i<(6UL<<20); i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+(6UL<<20), 0, (2UL<<20) );
  round_trip( 8UL<<20, "u32-lit-ext" );

  /* decompressor must be safe on arbitrary junk */
  for( ulong trial=0UL; trial<200000UL; trial++ ) {
    ulong cs = fd_rng_uint( rng ) % 256UL;
    for( ulong i=0UL; i<cs; i++ ) comp[i] = (uchar)fd_rng_uint( rng );
    ulong ds = 1UL + fd_rng_uint( rng ) % 4096UL;
    long r = fd_zle4_decompress( out, ds, comp, cs );
    FD_TEST( r<0L || (r<=(long)ds) );
  }

  fd_rng_delete( fd_rng_leave( rng ) );
  FD_LOG_NOTICE(( "pass" ));
  fd_halt();
  return 0;
}
