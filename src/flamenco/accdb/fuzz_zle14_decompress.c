/* fuzz_zle14_decompress attacks fd_zle14_decompress with arbitrary
   bytes.  It must never write past data+data_sz (an exact sized heap
   buffer, so ASan sees any overrun), must never read off the pages the
   blob lies on (the blob sits against a guard page), must return either
   a length in [0,data_sz] or an FD_ZLE14_ERR_* code, and must leave
   whatever it reports untouched by a second, identical call.  Output it
   does report is re-compressed and round tripped.

   The blob does not get an exact sized allocation because the decoder
   is documented to read up to 8 bytes past its control cursor and 128
   past its literal cursor.  Those reads are what makes it fast: they
   are bounded by the page rather than by the byte, and the guard page
   is what holds the codec to that.  The input picks how much of the
   page is left after the blob, so the corpus covers both the strict
   boundary, where the decoder must fall back to its exact paths, and
   the slack that lets its sloppy paths run. */

#if !FD_HAS_HOSTED
#error "This target requires FD_HAS_HOSTED"
#endif

#include "fd_zle14.h"
#include "../../util/fd_util.h"
#include "../../util/sanitize/fd_fuzz.h"

#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define MAX_SZ (1UL<<20)

/* blob_at returns a place for a sz byte blob with slack bytes of the
   mapping left after it and a PROT_NONE page after that. */

static uchar *
blob_at( ulong sz,
         ulong slack ) {
  static uchar * map;
  static ulong   cap;
  ulong page = (ulong)sysconf( _SC_PAGESIZE );
  if( FD_UNLIKELY( !map ) ) {
    cap = fd_ulong_align_up( MAX_SZ+page, page );
    map = mmap( NULL, cap+page, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0 );
    FD_TEST( map!=MAP_FAILED );
    FD_TEST( !mprotect( map+cap, page, PROT_NONE ) );
  }
  FD_TEST( sz+slack<=cap );
  return map + cap - slack - sz;
}

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
           ulong         data_sz,
           ulong         slack ) {

  /* the blob against a guard page, the output exact sized */
  uchar * cbuf = blob_at( comp_sz, slack );
  uchar * data = malloc( fd_ulong_max( data_sz, 1UL ) );
  uchar * copy = malloc( fd_ulong_max( data_sz, 1UL ) );
  FD_TEST( data && copy );
  if( comp_sz ) memcpy( cbuf, comp, comp_sz );
  memset( data, 0xcc, fd_ulong_max( data_sz, 1UL ) );

  long res = fd_zle14_decompress( data, data_sz, cbuf, comp_sz );
  FD_TEST( res>=FD_ZLE14_ERR_SPACE );
  FD_TEST( res<=(long)data_sz );

  if( res>=0L ) {
    memcpy( copy, data, (ulong)res );

    /* deterministic */
    memset( data, 0x55, fd_ulong_max( data_sz, 1UL ) );
    FD_TEST( fd_zle14_decompress( data, data_sz, cbuf, comp_sz )==res );
    FD_TEST( 0==memcmp( data, copy, (ulong)res ) );

    /* re-compressing what came out must round trip */
    ulong   n     = (ulong)res;
    ulong   bound = FD_ZLE14_COMPRESS_BOUND( n );
    uchar * src   = aligned_alloc( 64UL, fd_ulong_align_up( n+64UL, 64UL ) );
    uchar * c2    = malloc( bound+64UL );
    uchar * d2    = malloc( fd_ulong_max( n, 1UL ) );
    FD_TEST( src && c2 && d2 );
    if( n ) memcpy( src, copy, n );
    memset( src+n, 0xff, fd_ulong_align_up( n+64UL, 64UL )-n );
    ulong cs = fd_zle14_compress( c2, src, n );
    FD_TEST( cs<=bound );
    FD_TEST( fd_zle14_decompress( d2, n, c2, cs )==(long)n );
    FD_TEST( 0==memcmp( d2, copy, n ) );
    free( src ); free( c2 ); free( d2 );
  }

  free( data ); free( copy );
}

int
LLVMFuzzerTestOneInput( uchar const * data,
                        ulong         data_sz ) {
  if( FD_UNLIKELY( data_sz>MAX_SZ ) ) data_sz = MAX_SZ;
  /* the first two bytes pick the output capacity and the blob's distance
     from the guard page, the rest is the blob */
  ulong cap   = 0UL;
  ulong slack = 0UL;
  if( data_sz>=2UL ) {
    cap     = ( ( (ulong)data[0]<<8 ) | (ulong)data[1] )*7UL;
    slack   = (ulong)data[0] % 129UL;  /* 0: the blob ends at the guard */
    data   += 2UL;
    data_sz -= 2UL;
  }
  decode_at( data, data_sz, cap,         slack );
  decode_at( data, data_sz, data_sz*8UL, slack ); /* generous capacity */
  FD_FUZZ_MUST_BE_COVERED;
  return 0;
}
