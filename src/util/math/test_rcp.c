#include "../fd_util.h"
#include "fd_rcp.h"

static void
test_rcp( ulong d,
          ulong x ) {
  fd_rcp_t rcp = fd_rcp( d );
  ulong    q   = fd_rcp_div( rcp, x );
  ulong    r   = fd_rcp_mod( rcp, x );
  if( FD_UNLIKELY( (q!=x/d) | (r!=x%d) ) ) {
    FD_LOG_ERR(( "FAIL: x %lu d %lu q %lu (exp %lu) r %lu (exp %lu) mul %lu", x, d, q, x/d, r, x%d, rcp.mul ));
  }
}

int
main( int     argc,
      char ** argv ) {

  fd_boot( &argc, &argv );

  fd_rng_t _rng[1]; fd_rng_t * rng = fd_rng_join( fd_rng_new( _rng, 0U, 0UL ) );

  /* Small divisors */

  for( ulong d=1UL; d<=1024UL; d++ ) {
    for( ulong i=0UL; i<100000UL; i++ ) test_rcp( d, fd_rng_ulong( rng ) );
  }

  /* Random divisors near powers of 2 and uniform */

  for( ulong iter=0UL; iter<10000000UL; iter++ ) {
    uint  ctl = fd_rng_uint( rng );
    int   w   = (int)(ctl & 63U);
    ulong d;
    switch( (ctl>>6) & 3U ) {
    case 0:  d = 1UL<<w;                        break;
    case 1:  d = (1UL<<w)-1UL;                  break;
    case 2:  d = (1UL<<w)+1UL;                  break;
    default: d = fd_rng_ulong( rng )>>(63-w);   break;
    }
    d = fd_ulong_max( d, 1UL );
    for( ulong j=0UL; j<8UL; j++ ) test_rcp( d, fd_rng_ulong( rng ) );
  }

  /* d==0 is treated as d==1 */

  fd_rcp_t rcp0 = fd_rcp( 0UL );
  FD_TEST( rcp0.d==1UL );
  FD_TEST( fd_rcp_div( rcp0, ULONG_MAX )==ULONG_MAX );
  FD_TEST( fd_rcp_mod( rcp0, ULONG_MAX )==0UL );

  /* Edge cases */

  ulong edge_d[] = { 1UL, 2UL, 3UL, 7UL, 1UL<<63, (1UL<<63)-1UL, (1UL<<63)+1UL, (1UL<<62)+1UL, ULONG_MAX-1UL, ULONG_MAX };
  ulong edge_x[] = { 0UL, 1UL, ULONG_MAX, ULONG_MAX-1UL, 1UL<<63, 1UL<<32 };
  for( ulong i=0UL; i<sizeof(edge_d)/sizeof(edge_d[0]); i++ ) {
    for( ulong j=0UL; j<sizeof(edge_x)/sizeof(edge_x[0]); j++ ) test_rcp( edge_d[ i ], edge_x[ j ] );
  }

  fd_rng_delete( fd_rng_leave( rng ) );

  FD_LOG_NOTICE(( "pass" ));
  fd_halt();
  return 0;
}
