/* fuzz_zle7_diff differentially fuzzes fd_zle7_compress: it compresses
   the fuzzer input, then decodes the result with an independent scalar
   transcription of the wire format (all four frame types -- raw, fill,
   last, more) as well as with fd_zle7_decompress, and checks that both
   reproduce the input exactly.  The scalar decoder is what pins the
   encoder's output format. */

#if !FD_HAS_HOSTED
#error "This target requires FD_HAS_HOSTED"
#endif

#include "fd_zle7.h"
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
ref_tok_get( uchar const * s,
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

/* ref_decompress: a scalar, byte-at-a-time transcription of the fd_zle7
   wire format.  Returns the decoded size, or -1 on malformed input. */

static long
ref_decompress( uchar *       d,
                ulong         data_sz,
                uchar const * s,
                ulong         comp_sz ) {
  if( comp_sz<1UL ) return data_sz==0UL ? 0L : -1L;

  uchar b0   = s[0];
  ulong type = (ulong)b0 & 0x3UL;
  ulong ext  = (ulong)b0 >> 2;

  switch( type ) {

  case 0UL: { /* raw */
    if( 1UL+ext>comp_sz ) return -1L;
    ulong n = comp_sz-1UL-ext;
    if( n!=data_sz ) return -1L;
    memcpy( d, s+1UL+ext, n );
    return (long)n;
  }

  case 1UL: { /* fill */
    ulong n   = ext;
    ulong hdr = 1UL;
    if( n==63UL ) {
      ulong i = 1UL, ex;
      if( !ref_vget( s, comp_sz, &i, &ex ) ) return -1L;
      n   = 63UL+ex;
      hdr = i;
    }
    n += 1UL;
    if( hdr!=comp_sz || n!=data_sz ) return -1L;
    memset( d, 0, n );
    return (long)n;
  }

  case 2UL: { /* last */
    if( comp_sz<3UL ) return -1L;
    if( s[1]==0xFFU || s[2]==0xFFU ) return -1L;

    uchar const * lp    = s+3UL;
    ulong         o     = 0UL;
    ulong         bp    = comp_sz;
    ulong         avail = comp_sz-3UL;

    ulong L0 = s[1], Z0 = s[2];
    if( L0>avail || L0>data_sz-o ) return -1L;
    memcpy( d+o, lp, L0 ); o += L0; lp += L0; avail -= L0;
    if( Z0>data_sz-o ) return -1L;
    memset( d+o, 0, Z0 ); o += Z0;

    for(;;) {
      if( !avail ) return -1L;
      bp -= 1UL; avail -= 1UL;
      ulong L = s[bp];
      if( L==0xFFUL || L>avail || L>data_sz-o ) return -1L;
      memcpy( d+o, lp, L ); o += L; lp += L; avail -= L;
      if( o==data_sz ) break;

      if( !avail ) return -1L;
      bp -= 1UL; avail -= 1UL;
      ulong Z = s[bp];
      if( Z==0xFFUL || Z>data_sz-o ) return -1L;
      memset( d+o, 0, Z ); o += Z;
    }
    if( avail!=0UL ) return -1L;
    return (long)o;
  }

  default: { /* more */
    if( comp_sz<3UL ) return -1L;
    ulong size_ext = (ulong)s[1] | ( (ulong)s[2]<<8 );
    ulong Lsz      = ( size_ext<<6 ) | ext;
    if( 3UL+Lsz>comp_sz ) return -1L;

    uchar const * lp    = s+3UL;
    uchar const * lpend = lp+Lsz;
    ulong         o     = 0UL;
    ulong         i     = (ulong)( lpend-s );

    while( o<data_sz ) {
      ulong ll;
      if( !ref_tok_get( s, comp_sz, &i, &ll ) ) return -1L;
      if( ll>(ulong)(lpend-lp) || ll>data_sz-o ) return -1L;
      memcpy( d+o, lp, ll ); o += ll; lp += ll;
      if( o==data_sz ) break;

      ulong zl;
      if( !ref_tok_get( s, comp_sz, &i, &zl ) ) return -1L;
      if( zl>data_sz-o ) return -1L;
      memset( d+o, 0, zl ); o += zl;
    }
    if( (lp!=lpend) | (i!=comp_sz) ) return -1L;
    return (long)o;
  }
  }
}

static uchar comp[ FD_ZLE7_COMPRESS_BOUND( MAX_SZ ) ] __attribute__((aligned(64)));
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

  ulong csz = fd_zle7_compress( comp, src, sz );
  FD_TEST( csz<=FD_ZLE7_COMPRESS_BOUND( sz ) );

  long r1 = ref_decompress   ( out1, sz, comp, csz );
  long r2 = fd_zle7_decompress( out2, sz, comp, csz );
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
