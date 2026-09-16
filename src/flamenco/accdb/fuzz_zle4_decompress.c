/* fuzz_zle4_decompress attacks fd_zle4_decompress with arbitrary bytes:
   it must never read or write out of bounds and must classify every
   input as ok / ERR_SPACE / ERR_CORRUPT.  Successful decodes are then
   re-compressed and round-tripped. */

#if !FD_HAS_HOSTED
#error "This target requires FD_HAS_HOSTED"
#endif

#include "fd_zle4.h"
#include "../../util/fd_util.h"
#include "../../util/sanitize/fd_fuzz.h"

#include <stdlib.h>

#define MAX_SZ (1UL<<20)

int
LLVMFuzzerInitialize( int  *   argc,
                      char *** argv ) {
  putenv( "FD_LOG_BACKTRACE=0" );
  setenv( "FD_LOG_PATH", "", 0 );
  fd_boot( argc, argv );
  atexit( fd_halt );
  fd_log_level_core_set( 3 );
  return 0;
}

static void
decode_at( uchar const * comp,
           ulong         comp_sz,
           ulong         data_sz ) {

  /* exact-sized allocations so an overrun cannot hide in slack */
  uchar * cbuf = malloc( fd_ulong_max( comp_sz, 1UL ) );
  uchar * data = malloc( fd_ulong_max( data_sz, 1UL ) );
  FD_TEST( cbuf && data );
  if( comp_sz ) memcpy( cbuf, comp, comp_sz );
  memset( data, 0xcc, fd_ulong_max( data_sz, 1UL ) );

  long res = fd_zle4_decompress( data, data_sz, cbuf, comp_sz );
  FD_TEST( res<=(long)data_sz );
  FD_TEST( ( res>=0L ) | ( res==FD_ZLE4_ERR_SPACE ) | ( res==FD_ZLE4_ERR_CORRUPT ) );
  FD_TEST( fd_zle4_strerror( res ) );

  if( FD_LIKELY( res>=0L ) ) {
    ulong   n     = (ulong)res;
    ulong   span  = fd_ulong_align_up( fd_ulong_max( n, 1UL ), 64UL );
    uchar * pad   = aligned_alloc( 64UL, span );
    uchar * comp2 = malloc( FD_ZLE4_COMPRESS_BOUND( n ) );
    uchar * out2  = malloc( fd_ulong_max( n, 1UL ) );
    FD_TEST( pad && comp2 && out2 );
    if( n ) memcpy( pad, data, n );
    memset( pad+n, 0xff, span-n );

    ulong csz = fd_zle4_compress( comp2, pad, n );
    FD_TEST( csz<=FD_ZLE4_COMPRESS_BOUND( n ) );
    FD_TEST( fd_zle4_decompress( out2, n, comp2, csz )==(long)n );
    FD_TEST( 0==memcmp( out2, data, n ) );

    free( out2 ); free( comp2 ); free( pad );
  }

  free( data ); free( cbuf );
}

int
LLVMFuzzerTestOneInput( uchar const * data,
                        ulong         data_sz ) {
  if( FD_UNLIKELY( data_sz>MAX_SZ ) ) data_sz = MAX_SZ;

  ulong cap;
  if( FD_UNLIKELY( !data_sz ) ) cap = 0UL;
  else switch( data[0] & 3U ) {
    case 0:  cap = 0UL;                          break;
    case 1:  cap = (ulong)data[ data_sz-1UL ];   break;
    case 2:  cap = data_sz;                      break;
    default: cap = MAX_SZ;                       break;
  }

  decode_at( data, data_sz, cap );

  FD_FUZZ_MUST_BE_COVERED;
  return 0;
}
