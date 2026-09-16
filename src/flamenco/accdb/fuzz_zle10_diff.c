/* fuzz_zle10_diff differentially fuzzes fd_zle10_compress against
   fd_zle8_compress.  The two share a wire format and fd_zle10 claims
   byte-identical output, so the check is exact: same size, same bytes.
   It also round trips through fd_zle10_decompress, asserts fd_zle8.c's
   proven csz<=sz+sz/386+6 bound (not sz+6: four 383+ B literals split
   by 3-zero runs, then a 383+ B tail, is sz+7) and that the encoder
   writes nothing past its output (checked to 64 B beyond the bound).
   Inputs are placed at a varying offset with poisoned padding. */

#if !FD_HAS_HOSTED
#error "This target requires FD_HAS_HOSTED"
#endif

#include "fd_zle8.h"
#include "fd_zle10.h"
#include "../../util/fd_util.h"
#include "../../util/sanitize/fd_fuzz.h"

#include <stdlib.h>

#define MAX_SZ (1UL<<20)

static uchar comp8[ FD_ZLE8_COMPRESS_BOUND( MAX_SZ ) ] __attribute__((aligned(64)));
static uchar comp9[ FD_ZLE10_COMPRESS_BOUND( MAX_SZ )+64UL ] __attribute__((aligned(64)));
static uchar out  [ MAX_SZ ];

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
check_at( uchar const * data,
          ulong         sz,
          ulong         off ) {
  ulong   span = off + fd_ulong_align_up( fd_ulong_max( sz, 1UL ), 64UL )+64UL;
  uchar * buf  = aligned_alloc( 64UL, fd_ulong_align_up( span, 64UL ) );
  FD_TEST( buf );
  uchar * src = buf+off;
  memcpy( src, data, sz );
  memset( src+sz, 0xff, span-off-sz ); /* pad must not affect the output */

  ulong bound9 = FD_ZLE10_COMPRESS_BOUND( sz );
  memset( comp9, 0xa5, bound9+64UL );

  ulong c8 = fd_zle8_compress( comp8, src, sz );
  ulong c9 = fd_zle10_compress( comp9, src, sz );
  FD_TEST( c9<=bound9 );
  for( ulong i=c9; i<bound9+64UL; i++ ) FD_TEST( comp9[i]==0xa5 );
  FD_TEST( c9<=sz+sz/386UL+6UL );
  if( FD_UNLIKELY( c8!=c9 || memcmp( comp8, comp9, c9 ) ) )
    FD_LOG_ERR(( "zle8/zle10 output mismatch: sz %lu off %lu c8 %lu c9 %lu", sz, off, c8, c9 ));

  long r = fd_zle10_decompress( out, sz, comp9, c9 );
  if( FD_UNLIKELY( r!=(long)sz || ( sz && memcmp( out, src, sz ) ) ) )
    FD_LOG_ERR(( "round trip mismatch: sz %lu off %lu csz %lu r %ld", sz, off, c9, r ));

  free( buf );
}

int
LLVMFuzzerTestOneInput( uchar const * data,
                        ulong         data_sz ) {
  if( FD_UNLIKELY( data_sz>MAX_SZ ) ) data_sz = MAX_SZ;
  ulong hash = fd_ulong_hash( data_sz ^ ( data_sz ? ( ( (ulong)data[0]<<8 ) | (ulong)data[data_sz-1] ) : 0UL ) );
  check_at( data, data_sz, 0UL         );
  check_at( data, data_sz, hash & 63UL );
  FD_FUZZ_MUST_BE_COVERED;
  return 0;
}
