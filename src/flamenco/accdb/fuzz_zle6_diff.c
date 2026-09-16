/* fuzz_zle6_diff differentially fuzzes fd_zle6_compress: it compresses
   the fuzzer input, then decodes the result with an independent scalar
   transcription of the wire format as well as with fd_zle6_decompress,
   and checks that both reproduce the input exactly.  The scalar decoder
   is what pins the encoder's output format. */

#if !FD_HAS_HOSTED
#error "This target requires FD_HAS_HOSTED"
#endif

#include "fd_zle6.h"
#include "../../util/fd_util.h"
#include "../../util/sanitize/fd_fuzz.h"

#include <stdlib.h>

#define MAX_SZ (1UL<<20)

/* ref_decompress: a scalar, byte-at-a-time transcription of the fd_zle6
   wire format.  Returns the decoded size, or -1 on malformed input. */

static long
ref_decompress( uchar *       d,
                ulong         data_sz,
                uchar const * s,
                ulong         comp_sz ) {
  if( comp_sz<1UL ) return data_sz==0UL ? 0L : -1L;

  uchar b0 = s[0];
  if( b0&0x80U ) { /* stored */
    ulong n = comp_sz-1UL;
    if( n>data_sz ) return -1L;
    memcpy( d, s+1UL, n );
    return (long)n;
  }

  ulong hdr, L;
  ulong c0 = (ulong)b0 & 0x7FUL;
  if( c0!=0x7FUL ) { hdr = 1UL; L = c0; }
  else {
    if( comp_sz<4UL ) return -1L;
    L   = (ulong)s[1] | ((ulong)s[2]<<8) | ((ulong)s[3]<<16);
    hdr = 4UL;
  }
  if( hdr+L>comp_sz ) return -1L;
  uchar const * lp    = s+hdr;
  uchar const * lpend = lp+L;
  uchar const * cp    = lpend;
  uchar const * ce    = s+comp_sz;
  ulong         o     = 0UL;

  while( o<data_sz ) {
    if( cp>=ce ) return -1L;
    ulong ll = *cp++;
    if( ll==0xFFUL ) {
      if( cp+4UL>ce ) return -1L;
      ll  = (ulong)cp[0] | ((ulong)cp[1]<<8) | ((ulong)cp[2]<<16) | ((ulong)cp[3]<<24);
      cp += 4UL;
    }
    if( ll>(ulong)(lpend-lp) || ll>data_sz-o ) return -1L;
    for( ulong i=0UL; i<ll; i++ ) d[o++] = *lp++;
    if( o==data_sz ) break; /* terminal token: no Z field */

    if( cp>=ce ) return -1L;
    ulong zb = *cp++;
    ulong zl;
    if( zb!=0xFFUL ) zl = zb+2UL;
    else {
      if( cp+4UL>ce ) return -1L;
      ulong ext = (ulong)cp[0] | ((ulong)cp[1]<<8) | ((ulong)cp[2]<<16) | ((ulong)cp[3]<<24);
      cp += 4UL;
      zl  = ext+2UL;
    }
    if( zl>data_sz-o ) return -1L;
    for( ulong i=0UL; i<zl; i++ ) d[o++] = 0;
  }
  if( lp!=lpend || cp!=ce ) return -1L;
  return (long)o;
}

static uchar comp[ FD_ZLE6_COMPRESS_BOUND( MAX_SZ ) ] __attribute__((aligned(64)));
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

  ulong csz = fd_zle6_compress( comp, src, sz );
  FD_TEST( csz<=FD_ZLE6_COMPRESS_BOUND( sz ) );

  long r1 = ref_decompress   ( out1, sz, comp, csz );
  long r2 = fd_zle6_decompress( out2, sz, comp, csz );
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
