/* fuzz_zle4_diff differentially fuzzes fd_zle4_compress: it compresses
   the fuzzer input, then decodes the result with an independent scalar
   transcription of the wire format as well as with fd_zle4_decompress,
   and checks that both reproduce the input exactly.  The scalar decoder
   is what pins the encoder's output format. */

#if !FD_HAS_HOSTED
#error "This target requires FD_HAS_HOSTED"
#endif

#include "fd_zle4.h"
#include "../../util/fd_util.h"
#include "../../util/sanitize/fd_fuzz.h"

#include <stdlib.h>

#define MAX_SZ (1UL<<20)

/* ref_decompress: a scalar, byte-at-a-time transcription of the fd_zle4
   wire format.  Returns the decoded size, or -1 on malformed input. */

static long
ref_decompress( uchar *       d,
                ulong         data_sz,
                uchar const * s,
                ulong         comp_sz ) {
  if( !comp_sz ) return data_sz==0UL ? 0L : -1L;
  if( s[0]==0 ) { /* stored */
    ulong n = comp_sz-1UL;
    if( n>data_sz ) return -1L;
    memcpy( d, s+1UL, n );
    return (long)n;
  }
  if( s[0]!=1 || comp_sz<2UL ) return -1L;

  ulong hdr = 2UL, C = s[1];
  if( C==0xFFUL ) {
    if( comp_sz<6UL ) return -1L;
    C   = (ulong)s[2] | ((ulong)s[3]<<8) | ((ulong)s[4]<<16) | ((ulong)s[5]<<24);
    hdr = 6UL;
  }
  if( hdr+C>comp_sz ) return -1L;

  ulong         lit_sz = comp_sz-hdr-C;
  uchar const * lp     = s+hdr;
  uchar const * lpend  = lp+lit_sz;
  uchar const * cp     = lpend;
  uchar const * ce     = s+comp_sz;
  ulong         o      = 0UL;

  while( cp<ce ) {
    ulong ll, zl;
    if( cp[0]!=0xFF ) {
      if( cp+2UL>ce ) return -1L;
      ll = cp[0]; zl = cp[1]; cp += 2UL;
    } else {
      if( cp+9UL>ce ) return -1L;
      ll = (ulong)cp[1] | ((ulong)cp[2]<<8) | ((ulong)cp[3]<<16) | ((ulong)cp[4]<<24);
      zl = (ulong)cp[5] | ((ulong)cp[6]<<8) | ((ulong)cp[7]<<16) | ((ulong)cp[8]<<24);
      cp += 9UL;
    }
    if( ll>(ulong)(lpend-lp) || ll>data_sz-o || zl>data_sz-o-ll ) return -1L;
    for( ulong i=0UL; i<ll; i++ ) d[o++] = *lp++;
    for( ulong i=0UL; i<zl; i++ ) d[o++] = 0;
  }
  if( o!=data_sz || lp!=lpend ) return -1L;
  return (long)o;
}

static uchar comp[ FD_ZLE4_COMPRESS_BOUND( MAX_SZ ) ] __attribute__((aligned(64)));
static uchar out1[ MAX_SZ ];
static uchar out2[ MAX_SZ ];

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
  ulong   span = off + fd_ulong_align_up( fd_ulong_max( sz, 1UL ), 64UL );
  uchar * buf  = aligned_alloc( 64UL, fd_ulong_align_up( span, 64UL ) );
  FD_TEST( buf );
  uchar * src = buf+off;
  memcpy( src, data, sz );
  memset( src+sz, 0xff, span-off-sz ); /* pad must not affect the output */

  ulong csz = fd_zle4_compress( comp, src, sz );
  FD_TEST( csz<=FD_ZLE4_COMPRESS_BOUND( sz ) );

  long r1 = ref_decompress   ( out1, sz, comp, csz );
  long r2 = fd_zle4_decompress( out2, sz, comp, csz );
  if( FD_UNLIKELY( r1!=(long)sz || r2!=(long)sz ||
                   ( sz && ( memcmp( out1, src, sz ) || memcmp( out2, src, sz ) ) ) ) )
    FD_LOG_ERR(( "round trip mismatch: sz %lu off %lu csz %lu ref %ld avx %ld", sz, off, csz, r1, r2 ));

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
