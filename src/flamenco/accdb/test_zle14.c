#include "fd_zle14.h"
#include "fd_zle14_ref.h"
#include "../../util/fd_util.h"

#include <stdlib.h>

/* test_zle14 checks fd_zle14_compress byte for byte against the scalar
   reference in fd_zle14_ref.h, round trips every case through both
   decoders, and checks the compress bound and the encoder's write
   slack.  fd_zle14_compress writes up to 63 bytes of the chunk's
   literal copy past its output, which the bound covers. */

#define MAX_SZ (8UL<<20)

static uchar in   [ MAX_SZ+64UL ] __attribute__((aligned(64)));
static uchar comp [ FD_ZLE14_COMPRESS_BOUND( MAX_SZ )+64UL ] __attribute__((aligned(64)));
static uchar compr[ FD_ZLE14_COMPRESS_BOUND( MAX_SZ )+64UL ] __attribute__((aligned(64)));
static uchar out  [ MAX_SZ+64UL ] __attribute__((aligned(64)));

/* Walls around both buffers.  The decoder emits a group with sloppy 64
   byte stores wherever it has the slack for them, so a canary outside
   the buffers is what catches one that runs long. */

#define WALL (512UL)
#define WCAP (8192UL)
static uchar wout[ WALL+WCAP+WALL ] __attribute__((aligned(64)));
static uchar wcmp[ WALL+WCAP+WALL ] __attribute__((aligned(64)));

static void
walled_decode( uchar const * comp,
               ulong         comp_sz,
               ulong         cap ) {
  FD_TEST( comp_sz<=WCAP && cap<=WCAP );
  fd_memset( wout, 0xA5, sizeof(wout) );
  fd_memset( wcmp, 0x5A, sizeof(wcmp) );
  fd_memcpy( wcmp+WALL, comp, comp_sz );
  long r = fd_zle14_decompress( wout+WALL, cap, wcmp+WALL, comp_sz );
  FD_TEST( r>=FD_ZLE14_ERR_SPACE );
  FD_TEST( r<=(long)cap );
  for( ulong i=0UL; i<WALL; i++ ) {
    if( wout[i]!=0xA5 || wout[WALL+cap+i]!=0xA5 )
      FD_LOG_ERR(( "decoder wrote outside data: comp_sz=%lu cap=%lu r=%ld", comp_sz, cap, r ));
    if( wcmp[i]!=0x5A || wcmp[WALL+comp_sz+i]!=0x5A )
      FD_LOG_ERR(( "decoder wrote into comp: comp_sz=%lu cap=%lu r=%ld", comp_sz, cap, r ));
  }
}

static void
round_trip( ulong        n,
            char const * tag ) {
  fd_memset( in+n, 0xff, fd_ulong_align_up( n, 64UL )+64UL-n ); /* pad must not leak */
  fd_memset( comp, 0xcc, FD_ZLE14_COMPRESS_BOUND( n )+64UL );
  ulong cs = fd_zle14_compress( comp, in, n );
  ulong cr = fd_zle14_ref_compress( compr, in, n );
  if( FD_UNLIKELY( cs!=cr || memcmp( comp, compr, cs ) ) )
    FD_LOG_ERR(( "ref/zle14 mismatch: %s n=%lu asm=%lu ref=%lu", tag, n, cs, cr ));
  FD_TEST( cs<=FD_ZLE14_COMPRESS_BOUND( n ) );
  /* nothing past the output but the literal copy's 63 byte slack */
  for( ulong i=cs+64UL; i<FD_ZLE14_COMPRESS_BOUND( n )+64UL; i++ ) FD_TEST( comp[i]==0xcc );

  fd_memset( out, 0x55, n+64UL );
  long ds = fd_zle14_decompress( out, n, comp, cs );
  if( FD_UNLIKELY( ds!=(long)n || memcmp( out, in, n ) ) )
    FD_LOG_ERR(( "round trip failed: %s n=%lu cs=%lu ds=%ld", tag, n, cs, ds ));
  for( ulong i=n; i<n+64UL; i++ ) FD_TEST( out[i]==0x55 ); /* no write past data_sz */

  /* the reference decoder agrees */
  fd_memset( out, 0x55, n );
  FD_TEST( fd_zle14_ref_decompress( out, n, comp, cs )==(long)n );
  FD_TEST( 0==memcmp( out, in, n ) );

  /* a capacity one byte short must be reported, not written */
  if( n>1UL ) FD_TEST( fd_zle14_decompress( out, n-1UL, comp, cs )<0L );

  /* extra capacity must not change the answer */
  fd_memset( out, 0x55, n+64UL );
  FD_TEST( fd_zle14_decompress( out, n+64UL, comp, cs )==(long)n );
  FD_TEST( 0==memcmp( out, in, n ) );
}

int
main( int     argc,
      char ** argv ) {
  fd_boot( &argc, &argv );
  fd_rng_t _rng[1]; fd_rng_t * rng = fd_rng_join( fd_rng_new( _rng, 1234U, 0UL ) );

  /* empty */
  FD_TEST( fd_zle14_compress( comp, in, 0UL )==0UL );
  FD_TEST( fd_zle14_decompress( out, 0UL, comp, 0UL )==0L );

  /* all zero, every size 1..64: the degenerate [ L=0, Z=n ] frame,
     which is both a leading-L=0 token and a terminal-Z one */
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "all-zero" );

  /* all nonzero -- no zero-byte mask bit is ever set, so the whole
     input must come out as one lone-L token */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "incompressible" );

  /* uniform random: occasional zero bytes, mostly isolated and so
     mostly isolated */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "uniform" );

  /* isolated single zeros between literals: X 0 X 0 X ... -- no zero
     pair, so this must come out as one lone-L
     token -- and it is the shape zle14_compress_tiny's change 1
     (masking d to len-1) exists for: unmasked, a trailing zero byte
     adjacent to the padding is framed as a phantom Z=1 run */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i&1UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "x0x0" );

  /* X 0 0 repeated: maximum frame density, one run every 3 bytes */
  for( ulong i=0UL; i<sizeof(in); i+=3UL ) { in[i]=1U; in[i+1]=0U; in[i+2]=0U; }
  for( ulong n=1UL; n<=64UL; n++ ) round_trip( n, "x00" );

  /* every input ends in a real zero run that reaches len-1 (the exact
     shape zle14_compress_tiny's change 1 exists for) */
  for( ulong n=2UL; n<=64UL; n++ ) {
    for( ulong i=0UL; i<n; i++ ) in[i] = (uchar)( (i+1UL<n) ? ( fd_rng_uint(rng)|1U ) : 0U );
    /* force at least the last 2 bytes zero so it's a real run,
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
  for( ulong trial=0UL; trial<40000UL; trial++ ) {
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

  /* >64B, incompressible: one lone-L token whose L field escapes past
     0xFF for the larger sizes */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  round_trip( 65UL,   "big-fallback" );
  round_trip( 100UL,  "big-fallback" );
  round_trip( 1000UL, "big-fallback" );
  round_trip( 4096UL, "big-fallback" );

  /* >64B, all zero: a single [ L=0, Z=n ] token, reaching the 256B
     skip-ahead and zle14_spill's word-at-a-time spill */
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=65UL; n<=4096UL; n+=3UL ) round_trip( n, "all-zero-more" );
  for( ulong n=4096UL; n<=(1UL<<20); n<<=1 ) round_trip( n, "all-zero-more-big" );

  /* uniform random >64B (occasional zeros, mostly isolated) */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)fd_rng_uint( rng );
  for( ulong n=65UL; n<=2048UL; n+=3UL ) round_trip( n, "uniform-more" );
  round_trip( 4UL<<20, "uniform-more-big" );

  /* structured: alternating runs of random length, many trials, across
     the 64B tiny/more boundary and well past it */
  for( ulong trial=0UL; trial<4000UL; trial++ ) {
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
     the token stream) */
  for( ulong i=0UL; i<sizeof(in); i+=3UL ) { in[i]=1U; in[i+1]=0U; in[i+2]=0U; }
  for( ulong n=65UL; n<=6000UL; n+=63UL ) round_trip( n, "x00-more" );

  /* isolated single zeros between literals past 64B: no zero pair,
     so the whole input is one lone-L token */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i&1UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=65UL; n<=4000UL; n+=7UL ) round_trip( n, "x0x0-more" );

  /* a long literal span followed by a huge zero run -- exercises
     zle14_spill and the 256B skip-ahead in both directions */
  for( ulong i=0UL; i<600UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+600UL, 0, (8UL<<20)-600UL );
  round_trip( 4UL<<20, "spill" );

  /* runs that straddle the ZLE14_TOK_ESC (0xFF) 1-byte/varint boundary
     in both L and Z fields -- the one path the earlier <=64B-only test
     could never reach (L,Z<=64<0xFF always there) */
  for( ulong trial=0UL; trial<40UL; trial++ ) {
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
     well past ZLE14_TOK_ESC's first varint group) */
  fd_memset( in, 0, 8UL<<20 );
  for( ulong i=0UL; i<7UL; i++ ) in[i] = 1U;
  round_trip( 6UL<<20, "multibyte-varint-z" );

  /* a single literal run long enough to need the same for L (delta
     L-0xFF >= 128), isolating the per-token L varint escape */
  for( ulong i=0UL; i<2000UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  fd_memset( in+2000UL, 0, (1UL<<20)-2000UL );
  round_trip( 1UL<<20, "multibyte-varint-l" );
  for( ulong i=0UL; i<20000UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  for( ulong n=20002UL; n<20400UL; n+=7UL ) round_trip( n, "3byte-varint-l" );

  /* block boundary cases: every offset of a 2 byte zero run across a
     64 byte boundary, and every size around a chunk span */
  for( ulong n=128UL; n<=200UL; n+=7UL ) {
    for( ulong p=0UL; p+2UL<=n; p++ ) {
      for( ulong i=0UL; i<n; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
      in[p] = 0U; in[p+1UL] = 0U;
      round_trip( n, "run-astride-block" );
    }
  }
  fd_memset( in, 0, sizeof(in) );
  for( ulong n=2000UL; n<=2200UL; n+=3UL ) round_trip( n, "chunk-span-zero" );
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  for( ulong n=2000UL; n<=2200UL; n+=3UL ) round_trip( n, "chunk-span-lit" );

  /* a long zero run crossing many chunk spans, which the encoder joins
     into one chunk */
  fd_memset( in, 0, sizeof(in) );
  for( ulong i=0UL; i<70UL; i++ ) in[i] = (uchar)( fd_rng_uint( rng ) | 1U );
  round_trip( 1UL<<20, "zero-tail-1M" );
  round_trip( 8UL<<20, "zero-tail-8M" );

  /* dense texture: every block hits mask mode */
  for( ulong i=0UL; i<sizeof(in); i++ ) in[i] = (uchar)( (i%3UL) ? 0U : (fd_rng_uint(rng)|1U) );
  for( ulong n=64UL; n<=4096UL; n+=64UL ) round_trip( n, "mask-mode" );
  round_trip( 1UL<<20, "mask-mode-big" );

  /* Malformed streams: the decoder must report or absorb them, never run
     off its buffers.  The first case is one fuzz_zle14_decompress found:
     the header's control size field holds 9 bits, so a stream can claim
     511 control bytes, and the decoder's padded copy of that region has
     to be sized for the field rather than for what the encoder emits. */
  {
    static uchar bad[ 4096 ];
    /* ctl_sz=511, last=1, rem=64, and a blob far too short to hold it */
    fd_memset( bad, 0xff, sizeof(bad) );
    for( ulong n=2UL; n<=600UL; n++ ) {
      long r = fd_zle14_decompress( out, 64UL, bad, n );
      FD_TEST( r>=FD_ZLE14_ERR_SPACE && r<=64L );
      r = fd_zle14_decompress( out, sizeof(out)-64UL, bad, n );
      FD_TEST( r>=FD_ZLE14_ERR_SPACE && r<=(long)(sizeof(out)-64UL) );
    }
    /* ctl_sz=0 is not a legal chunk */
    bad[0] = 0x00; bad[1] = 0x00;
    FD_TEST( fd_zle14_decompress( out, sizeof(out), bad, 8UL )==FD_ZLE14_ERR_STREAM );
    /* a chunk that claims more control than the blob holds */
    bad[0] = 0x81; bad[1] = 0x40; /* ctl_sz=128, last=1 */
    FD_TEST( fd_zle14_decompress( out, sizeof(out), bad, 8UL )==FD_ZLE14_ERR_STREAM );
    /* one all zero block, one byte long */
    bad[0] = 0x81; bad[1] = 0x00; /* ctl_sz=1, rem=1, last=1 */
    bad[2] = 0xC0;                /* one all zero block */
    FD_TEST( fd_zle14_decompress( out, 1UL, bad, 3UL )==1L );
    FD_TEST( out[0]==0 );
    FD_TEST( fd_zle14_decompress( out, 0UL, bad, 3UL )==FD_ZLE14_ERR_SPACE );
    /* a run of 64 all zero blocks: 4033 bytes, and nothing less */
    bad[2] = 0xFF;
    FD_TEST( fd_zle14_decompress( out, 8192UL, bad, 3UL )==4033L );
    FD_TEST( fd_zle14_decompress( out, 4032UL, bad, 3UL )==FD_ZLE14_ERR_SPACE );
    /* random bytes, walled: no crash, a legal answer, nothing written
       outside the output.  The textures matter -- position bytes with bit
       6 set, mask mode codes and run codes reach different paths. */
    for( ulong trial=0UL; trial<300000UL; trial++ ) {
      ulong n    = 1UL + fd_rng_ulong( rng )%2000UL;
      ulong cap  = fd_rng_ulong( rng )%4096UL;
      uint  mode = fd_rng_uint( rng )%5U;
      for( ulong i=0UL; i<n; i++ ) {
        uint b = fd_rng_uint( rng );
        switch( mode ) {
        case 0: bad[i] = (uchar)b;                                    break;
        case 1: bad[i] = (uchar)( b | 0x80U );                        break;
        case 2: bad[i] = (uchar)( b & 0x3FU );                        break;
        case 3: bad[i] = (uchar)( ( b & 0x3FU ) | 0x40U );            break;
        case 4: bad[i] = (uchar)( (b&1U) ? 0x7FU : (uchar)(b>>1) );   break;
        }
      }
      if( n>=2UL && (fd_rng_uint( rng )&1U) ) { /* a plausible header */
        ulong ctl_sz = 1UL + fd_rng_ulong( rng )%fd_ulong_min( n, 511UL );
        ulong h = (ctl_sz<<7) | ((fd_rng_ulong( rng )%64UL)<<1) | (fd_rng_uint( rng )&1U);
        bad[0] = (uchar)h; bad[1] = (uchar)(h>>8);
      }
      walled_decode( bad, n, cap );
    }
  }

  fd_rng_delete( fd_rng_leave( rng ) );
  FD_LOG_NOTICE(( "pass" ));
  fd_halt();
  return 0;
}
