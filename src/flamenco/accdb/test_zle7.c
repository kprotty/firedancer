#include "fd_zle7.h"
#include "../../util/fd_util.h"

/* fd_zle7 implements all four frame types now: type.raw/fill/last for
   data_sz<=64 (zle7_compress_small) and type.more for data_sz>64
   (zle7_compress_more) -- see the wire format comment in fd_zle7.c. */

static uchar in  [ (8UL<<20)+64UL ] __attribute__((aligned(64)));
static uchar comp[ FD_ZLE7_COMPRESS_BOUND( 8UL<<20 ) ] __attribute__((aligned(64)));
static uchar out [ (8UL<<20)+64UL ] __attribute__((aligned(64)));

static void
round_trip( ulong n,
            char const * tag ) {
  fd_memset( in+n, 0xff, fd_ulong_align_up( n, 64UL )+64UL-n );
  fd_memset( comp, 0xcc, sizeof(comp) );
  ulong cs = fd_zle7_compress( comp, in, n );
  FD_TEST( cs<=FD_ZLE7_COMPRESS_BOUND( n ) );
  fd_memset( out, 0x55, n );
  long ds = fd_zle7_decompress( out, n, comp, cs );
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
  FD_TEST( fd_zle7_compress( comp, in, 0UL )==0UL );
  FD_TEST( fd_zle7_decompress( out, 0UL, comp, 0UL )==0L );

  /* all zero, every size 1..64 (exercises the FILL fast path,
     including its one escape case at n==64) */
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "all-zero" );

  /* all nonzero -- every byte set so no zero-byte mask bit is ever
     set, exercises RAW */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "incompressible" );

  /* uniform random (occasional zero bytes, mostly isolated -> RAW or
     LAST with short runs) */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "uniform" );

  /* isolated single zeros between literals: X 0 X 0 X ... -- every
     zero byte is below MIN_Z, so this should always hit RAW even
     though the padding-merge bug (fix 1) would otherwise manufacture
     a phantom run out of a trailing zero byte */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i&1UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "x0x0" );

  /* X 0 0 repeated: maximum frame density, one run every 3 bytes */
  for( ulong i=0UL; i<sizeof(in); i+=3UL ) { in[i]=1U; in[i+1]=0U; in[i+2]=0U; }
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "x00" );

  /* every input ends in a real zero run that reaches len-1 (the exact
     shape fix 1's padding-merge bug corrupts if it regresses) */
  for( ulong n=2UL; n<=64UL; n++ ) {
    for( ulong i=0UL; i<n; i++ ) in[i] = (uchar)( (i+1UL<n) ? ( fd_rng_uint(rng)|1U ) : 0U );
    /* force at least the last 2 bytes zero so it's a real MIN_Z run,
       not an isolated trailing zero */
    if( n>=2UL ) { in[n-1] = 0U; in[n-2] = 0U; }
    round_trip( n, "trailing-zero-run" );
  }

  /* every input starts with a run of len-2 zero bytes then 2 literal
     bytes -- exercises a run whose candidate bits start at 0 */
  for( ulong n=3UL; n<=64UL; n++ ) {
    fd_memset( in, 0, n-2UL );
    in[n-2] = (uchar)( fd_rng_uint(rng)|1U );
    in[n-1] = (uchar)( fd_rng_uint(rng)|1U );
    round_trip( n, "leading-zero-run" );
  }

  /* structured: alternating runs of random length, many trials, sizes
     1..64 so every run-count/boundary combination gets exercised */
  for( ulong trial=0UL; trial<200000UL; trial++ ) {
    ulong n = 1UL + fd_rng_uint( rng ) % 64UL;
    ulong j = 0UL;
    int zero = (int)( fd_rng_uint( rng ) & 1U );
    while( j<n ) {
      ulong run = 1UL + fd_rng_uint( rng ) % ( zero ? 12UL : 8UL );
      run = fd_ulong_min( run, n-j );
      if( zero ) fd_memset( in+j, 0, run );
      else       for( ulong k=0UL; k<run; k++ ) in[j+k] = (uchar)( fd_rng_uint( rng ) | 1U );
      j += run; zero = !zero;
    }
    round_trip( n, "structured" );
  }

  /* >64B, incompressible: type.more's mid-scan bailout should fall
     back to type.raw, same as before this session's type.more work */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  round_trip( 65UL,   "big-fallback" );
  round_trip( 100UL,  "big-fallback" );
  round_trip( 1000UL, "big-fallback" );
  round_trip( 4096UL, "big-fallback" );

  /* >64B, all zero: now the real type.more single-token path (a run
     that reaches the 256B skip-ahead and fd_zle_spill's word-at-a-time
     spill for the smaller sizes) */
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=65UL; n<=4096UL; n++ ) round_trip( n, "all-zero-more" );
  for( ulong n=4096UL; n<=(1UL<<20); n<<=1 ) round_trip( n, "all-zero-more-big" );

  /* uniform random >64B (occasional zeros -> a mix of RAW and MORE) */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  for( ulong n=65UL; n<=2048UL; n++ ) round_trip( n, "uniform-more" );
  round_trip( 4UL<<20, "uniform-more-big" );

  /* structured: alternating runs of random length, many trials, across
     the 64B tiny/more boundary and well past it */
  for( ulong trial=0UL; trial<20000UL; trial++ ) {
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
    round_trip( n, "structured-more" );
  }

  /* pathological: X 0 0 repeated past 64B (max frame density for
     type.more's token stream) */
  for( ulong i=0UL; i<sizeof(in); i+=3UL ) { in[i]=1U; in[i+1]=0U; in[i+2]=0U; }
  for( ulong n=65UL; n<=6000UL; n+=3UL ) round_trip( n, "x00-more" );

  /* isolated single zeros between literals past 64B: every zero byte
     is below MIN_Z, so the whole input is one literal run (RAW, or a
     MORE frame with a single terminal L field, whichever wins) */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i&1UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=65UL; n<=4000UL; n++ ) round_trip( n, "x0x0-more" );

  /* a long literal span followed by a huge zero run -- exercises
     zle7_spill and the 256B skip-ahead in both directions */
  for( ulong i=0UL; i<600UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+600UL, 0, (8UL<<20)-600UL );
  round_trip( 4UL<<20, "spill" );

  /* runs that straddle the ZLE7_TOK_ESC (0xFF) 1-byte/varint boundary
     in both L and Z fields -- the one path the earlier <=64B-only test
     could never reach (L,Z<=64<0xFF always there) */
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
    round_trip( 65UL + fd_rng_uint( rng ) % ( j-64UL ), "esc-boundary" );
  }

  /* a zero run whose varint extension needs >1 byte (Z well past 255,
     well past ZLE7_TOK_ESC's first varint group) */
  fd_memset( in, 0, 8UL<<20 );
  for( ulong i=0UL; i<7UL; i++ ) in[i] = 1U;
  round_trip( 6UL<<20, "multibyte-varint-z" );

  /* a single literal run long enough to need the same for L (delta
     L-0xFF >= 128), but small enough that the literal *plane* (Lsz,
     capped separately at 2^22 -- see the size-cap test below) is nowhere
     near its limit -- isolates the per-token L varint escape itself
     from that unrelated cap. */
  for( ulong i=0UL; i<2000UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+2000UL, 0, (1UL<<20)-2000UL );
  round_trip( 1UL<<20, "multibyte-varint-l" );

  /* literal-plane size at and past the type.more 22-bit size field's
     cap (2^22-1 = 4194303, see fd_zle7.c's ZLE7_SIZE_MAX).  An
     incompressible span this large followed by a big zero run would
     otherwise be a huge type.more win (the zero run alone dwarfs the
     literal bytes it would cost to store it), so this specifically
     isolates the header-field-width fallback from the ordinary
     "doesn't beat raw" one already covered above: a literal span that
     just fits leaves fs<n, cp-cp0 small, comfortably passing the
     mid-scan economic bailout, so only the Lsz>ZLE7_SIZE_MAX check in
     zle7_finish_more decides whether it goes out as type.more or
     type.raw. */
  {
    ulong tail = 2UL<<20;
    for( ulong lit=(4UL<<20)-64UL; lit<=(4UL<<20)+64UL; lit+=128UL ) {
      for( ulong i=0UL; i<lit; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
      fd_memset( in+lit, 0, tail );
      round_trip( lit+tail, "size-cap" );
    }
  }

  /* decompressor must be safe on arbitrary junk, including sizes that
     reach into the type.more parsing path */
  for( ulong trial=0UL; trial<200000UL; trial++ ) {
    ulong cs = fd_rng_uint( rng ) % 256UL;
    for( ulong i=0UL; i<cs; i++ ) comp[i] = (uchar)fd_rng_uint( rng );
    ulong ds_sz = 1UL + fd_rng_uint( rng ) % 4096UL;
    long r = fd_zle7_decompress( out, ds_sz, comp, cs );
    FD_TEST( r<0L || (r<=(long)ds_sz) );
  }

  fd_rng_delete( fd_rng_leave( rng ) );
  FD_LOG_NOTICE(( "pass" ));
  fd_halt();
  return 0;
}
