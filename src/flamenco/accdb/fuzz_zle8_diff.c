/* fuzz_zle8_diff differentially fuzzes fd_zle8_compress: it compresses
   the fuzzer input, then decodes the result both with an independent
   byte-at-a-time transcription of the wire format and with
   fd_zle8_decompress, and checks that both reproduce the input exactly.
   The scalar decoder is what pins the encoder's output format.

   It additionally asserts the format's headroom claim directly -- that
   the encoder never writes more than FD_ZLE8_COMPRESS_BOUND, and never
   more than sz+sz/386+6 -- since fd_zle8, unlike fd_zle5/6/7, has no
   verbatim fallback frame standing behind that bound.  (Not sz+6: four
   383+ B literals split by 3-zero runs, then a 383+ B tail, is sz+7.) */

#if !FD_HAS_HOSTED
#error "This target requires FD_HAS_HOSTED"
#endif

#include "fd_zle8.h"
#include "../../util/fd_util.h"
#include "../../util/sanitize/fd_fuzz.h"

#include <stdlib.h>

#define MAX_SZ (1UL<<20)

static int
ref_vget( uchar const * s,
          ulong         slen,
          ulong *       pi,
          ulong *       out ) {
  ulong v  = 0UL;
  ulong i  = *pi;
  int   sh = 0;
  for(;;) {
    if( (i>=slen) | (sh>42) ) return 0;
    uchar b = s[i++];
    v |= (ulong)( b & 0x7f )<<sh;
    if( !(b & 0x80) ) break;
    sh += 7;
  }
  *pi  = i;
  *out = v;
  return 1;
}

static int
ref_get( uchar const * s,
         ulong         slen,
         ulong *       pi,
         ulong *       out ) {
  ulong i = *pi;
  if( i>=slen ) return 0;
  ulong v = (ulong)s[i++];
  if( v==0xFFUL ) {
    ulong ext;
    if( !ref_vget( s, slen, &i, &ext ) ) return 0;
    v = 0xFFUL+ext;
  }
  *pi  = i;
  *out = v;
  return 1;
}

/* ref_decompress: a scalar, byte-at-a-time transcription of the fd_zle8
   wire format.  Returns the decoded size, or -1 on malformed input. */

static long
ref_decompress( uchar *       d,
                ulong         data_sz,
                uchar const * s,
                ulong         comp_sz ) {
  ulong o = 0UL;
  ulong i = 0UL;
  while( o<data_sz ) {
    ulong L;
    if( !ref_get( s, comp_sz, &i, &L ) ) return -1L;
    if( ( L>data_sz-o ) | ( L>comp_sz-i ) ) return -1L;
    for( ulong k=0UL; k<L; k++ ) d[ o+k ] = s[ i+k ];
    o += L; i += L;
    if( o==data_sz ) break;

    ulong Z;
    if( !ref_get( s, comp_sz, &i, &Z ) ) return -1L;
    if( Z>data_sz-o ) return -1L;
    for( ulong k=0UL; k<Z; k++ ) d[ o+k ] = (uchar)0;
    o += Z;
  }
  if( i!=comp_sz ) return -1L;
  return (long)o;
}

static uchar comp[ FD_ZLE8_COMPRESS_BOUND( MAX_SZ ) ] __attribute__((aligned(64)));
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
  ulong   span = off + fd_ulong_align_up( fd_ulong_max( sz, 1UL ), 64UL )+64UL;
  uchar * buf  = aligned_alloc( 64UL, fd_ulong_align_up( span, 64UL ) );
  FD_TEST( buf );
  uchar * src = buf+off;
  memcpy( src, data, sz );
  memset( src+sz, 0xff, span-off-sz ); /* pad must not affect the output */

  ulong csz = fd_zle8_compress( comp, src, sz );
  FD_TEST( csz<=FD_ZLE8_COMPRESS_BOUND( sz ) );
  FD_TEST( csz<=sz+sz/386UL+6UL ); /* the real bound -- see "Output bound" in fd_zle8.c */

  long r1 = ref_decompress    ( out1, sz, comp, csz );
  long r2 = fd_zle8_decompress( out2, sz, comp, csz );
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
