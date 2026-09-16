#include "fd_zle2.h"
#include "../../util/fd_util.h"

/* test_zle2: differential test for the fd_zle2 prototype encoder
   (see fd_zle2.h).  fd_zle2_compress targets the same wire format as
   fd_zle_compress and decodes with the existing fd_zle_decompress, so
   every case here checks that it round trips.

   It is usually, but not always, byte-for-byte identical to what
   fd_zle_compress produces: both make the same greedy (leftmost,
   MIN_Z=2, same escape threshold) decisions, EXCEPT that fd_zle2
   additionally splits a literal run that would outgrow its
   FD_ZLE2_LIT_CAP-byte scratch buffer (see fd_zle2.c) into an extra
   interstitial Z=0 frame -- fd_zle_big has no such cap, since it
   copies literal bytes straight from src instead of staging them.
   That trade costs a byte or two of ratio on long literal runs, not
   correctness, so diff_case only asserts the output round trips and
   stays within a generous distance of fd_zle_compress's size; it does
   not require byte-identical output. */

#define MAX_SZ (1UL<<20)

static uchar in   [ MAX_SZ ] __attribute__((aligned(64)));
static uchar comp2[ FD_ZLE_COMPRESS_BOUND( MAX_SZ )+1UL ] __attribute__((aligned(64))); /* +1: diff_case's sentinel byte */
static uchar comp1[ FD_ZLE_COMPRESS_BOUND( MAX_SZ ) ] __attribute__((aligned(64)));
static uchar out  [ MAX_SZ ] __attribute__((aligned(64)));

static void
diff_case( uchar const * src,
           ulong         sz ) {
  fd_memset( comp2, 0xcc, FD_ZLE_COMPRESS_BOUND( sz )+1UL );
  ulong sz2 = fd_zle2_compress( comp2, src, sz );
  FD_TEST( sz2<=FD_ZLE_COMPRESS_BOUND( sz ) );
  FD_TEST( comp2[ FD_ZLE_COMPRESS_BOUND( sz ) ]==0xcc ); /* no write past bound */

  ulong sz1 = fd_zle_compress( comp1, src, sz );

  /* at most one interstitial split per FD_ZLE2_LIT_CAP (256 B) of
     input, each costing a handful of bytes at most */

  FD_TEST( sz2<=sz1+8UL*( sz/256UL+2UL ) );

  fd_memset( out, 0xcc, sz );
  FD_TEST( fd_zle_decompress( out, sz, comp2, sz2 )==(long)sz );
  FD_TEST( 0==memcmp( out, src, sz ) );
}

int
main( int     argc,
      char ** argv ) {
  fd_boot( &argc, &argv );

  fd_rng_t _rng[1]; fd_rng_t * rng = fd_rng_join( fd_rng_new( _rng, 0U, 0UL ) );

  /* empty input */

  FD_TEST( fd_zle2_compress( comp2, in, 0UL )==0UL );

  /* known vectors */

  fd_memcpy( in, "AB\0\0\0CD", 7UL );
  diff_case( in, 7UL );

  fd_memset( in, 0, 2UL );
  diff_case( in, 2UL );

  fd_memset( in, 0, 100UL );
  diff_case( in, 100UL );

  in[0] = 0;
  diff_case( in, 1UL );

  /* all zeros, various sizes -- spans many words and exercises the
     zero-run extension loop repeatedly */

  fd_memset( in, 0, MAX_SZ );
  for( ulong sz=2UL; sz<=MAX_SZ; sz<<=1 ) diff_case( in, sz );

  /* zero-free input: escapes, exercising the escape path */

  for( ulong j=0UL; j<MAX_SZ; j++ ) in[j] = (uchar)( 1U+( fd_rng_uint( rng )%255U ) );
  for( ulong sz=1UL; sz<=MAX_SZ; sz<<=1 ) diff_case( in, sz );

  /* max frame density: 00 00 X repeated */

  for( ulong j=0UL; j<MAX_SZ; j++ ) in[j] = ( j%3UL==2UL ) ? (uchar)0xff : (uchar)0;
  diff_case( in, MAX_SZ );

  /* SPL-token-shaped: 165 bytes, zero runs interleaved with pubkeys */

  fd_memset( in, 0, 165UL );
  for( ulong j= 0UL; j< 32UL; j++ ) in[j] = (uchar)fd_rng_uint( rng );
  for( ulong j=32UL; j< 64UL; j++ ) in[j] = (uchar)fd_rng_uint( rng );
  in[64] = 1; in[105] = 1;
  diff_case( in, 165UL );

  /* one long literal run, swept across every offset and length that
     can touch a 64 B word boundary or the FD_ZLE2_LIT_CAP (256 B)
     scratch-flush boundary -- this is the case fd_zle2 changes the
     most relative to fd_zle_big, so it gets the most direct coverage */

  for( ulong sz=1UL; sz<=1024UL; sz++ ) {
    for( ulong off=0UL; off<sz; off++ ) {
      ulong len = sz-off;
      if( !( ( off%64UL )<=1UL || ( off%64UL )>=63UL ||
             ( off%256UL )<=1UL || ( off%256UL )>=255UL ||
             off==0UL || off==sz-1UL ) ) continue;
      for( ulong j=0UL; j<sz; j++ ) in[j] = (uchar)( fd_rng_uint( rng ) | 1U );
      for( ulong j=off; j<off+len && j<sz; j++ ) in[j] = 0;
      diff_case( in, sz );
    }
  }

  /* every size 0..2048 at a spread of zero densities, so runs land on
     and straddle both the 64 B word and 256 B scratch-cap boundary at
     every possible phase */

  static uint const dens[] = { 0U, 1U, 8U, 64U, 128U, 192U, 248U, 255U, 256U };
  for( ulong sz=0UL; sz<=2048UL; sz++ ) {
    for( ulong k=0UL; k<sizeof(dens)/sizeof(dens[0]); k++ ) {
      for( ulong j=0UL; j<sz; j++ )
        in[j] = ( fd_rng_uint( rng )&255U )<dens[k] ? (uchar)0 : (uchar)( fd_rng_uint( rng ) | 1U );
      diff_case( in, sz );
    }
  }

  /* larger sizes, random run-structured shapes */

  for( ulong t=0UL; t<300UL; t++ ) {
    ulong sz = 1UL + (ulong)( fd_rng_uint( rng )%200000U );
    uint  d  = fd_rng_uint( rng )%257U;
    ulong j  = 0UL;
    while( j<sz ) {
      ulong run = 1UL + (ulong)( fd_rng_uint( rng )%600U );
      int   z   = ( fd_rng_uint( rng )&255U )<d;
      for( ulong q=0UL; q<run && j<sz; q++, j++ ) in[j] = z ? (uchar)0 : (uchar)( fd_rng_uint( rng ) | 1U );
    }
    diff_case( in, sz );
  }

  fd_rng_delete( fd_rng_leave( rng ) );

  FD_LOG_NOTICE(( "pass" ));
  fd_halt();
  return 0;
}
