/* ans_test - correctness of the ANS layer, and what an encode step
   costs under each way of removing the division. */

#include "model.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails = 0;

static void
check( int          ok,
       char const * name,
       char const * detail ) {
  printf( "  %-52s %s   %s\n", name, ok ? "ok  " : "FAIL", detail );
  if( !ok ) fails++;
}

static ulong rs = 12345UL;
static double
urand( void ) {
  rs = rs*6364136223846793005UL + 1442695040888963407UL;
  return (double)(rs>>11)*(1.0/9007199254740992.0);
}

static double
now( void ) {
  struct timespec ts;
  clock_gettime( CLOCK_MONOTONIC, &ts );
  return (double)ts.tv_sec + 1e-9*(double)ts.tv_nsec;
}

#define D (784)

/* A code length must equal the sum of the symbol information contents.
   That identity is what makes an encode pass a linear layer. */

static void
test_codelen( void ) {
  ans_tans_level_t const * bank = ans_tans_bank();
  ans_rans_sym_t   const * tab  = ans_rans_symtab();
  double wt = 0.0, wr = 0.0, tot = 0.0, sum_t = 0.0;

  for( int trial=0; trial<500; trial++ ) {
    int  lev[D]; uint frq[D]; uint bit[D];
    double ref_t = 0.0, ref_r = 0.0;
    for( int i=0; i<D; i++ ) {
      float w = (float)( (urand()*2.0-1.0)*ANS_W_MAX );
      lev[i] = ans_w_to_level( w );
      frq[i] = ans_p_to_freq( ans_w_to_p( w ), ANS_RANS_M );
      bit[i] = urand()<0.2;
      uint ft = ans_level_freq( lev[i] );
      ref_t += -log2( (double)( bit[i] ? ft     : ANS_TANS_SIZE-ft     )/(double)ANS_TANS_SIZE );
      ref_r += -log2( (double)( bit[i] ? frq[i] : ANS_RANS_M   -frq[i] )/(double)ANS_RANS_M   );
    }
    ans_acc_t at, ar;
    ans_tans_init( &at ); ans_rans_init( &ar );
    for( int i=0; i<D; i++ ) {
      ans_tans_put( &at, &bank[ lev[i] ], bit[i] );
      ans_rans_put( &ar, &tab[ 2u*frq[i] + bit[i] ] );
    }
    double gt = (double)ans_tans_codelen_q8( &at )/256.0;
    double gr = (double)ans_rans_codelen_q8( &ar )/256.0;
    sum_t += gt-ref_t;
    if( fabs(gt-ref_t)>wt ) wt = fabs(gt-ref_t);
    if( fabs(gr-ref_r)>wr ) wr = fabs(gr-ref_r);
    tot += ref_r;
  }
  char buf[ 128 ];
  double mean = tot/500.0;
  snprintf( buf, sizeof(buf), "worst %.3f bits over %d symbols (%.3f%% of %.0f)", wr, D, 100.0*wr/mean, mean );
  check( wr<1.0, "rans code length == sum of -log2 p", buf );
  snprintf( buf, sizeof(buf), "mean %+.3f%%, worst %.3f bits over %d symbols (of %.0f)",
            100.0*(sum_t/500.0)/mean, wt, D, mean );
  /* the tANS approximation scales with the table, so must the bound */
  check( wt<8.0*(4096.0/(double)ANS_TANS_SIZE), "tans code length == sum of -log2 p", buf );
}

/* The reciprocal must reproduce the divide for every frequency, not
   just most of them: a single mismatch is a corrupt state. */

static void
test_reciprocal( void ) {
  ans_rans_sym_t const * tab = ans_rans_symtab();
  ulong bad = 0UL, cases = 0UL;
  for( uint f=1u; f<ANS_RANS_M; f++ ) {
    ans_rans_sym_t const * sym = &tab[ 2u*f + 1u ];   /* start 0, freq f */
    for( int k=0; k<32; k++ ) {
      uint x = ANS_RANS_L + (uint)( urand()*(double)( sym->x_max-ANS_RANS_L ) );
      ans_acc_t a = { x, 0u }, b = { x, 0u };
      ans_rans_put( &a, sym );
      ans_rans_put_div( &b, sym, f, 0u );
      bad += (ulong)( a.state!=b.state || a.nbit!=b.nbit );
      cases++;
    }
  }
  char buf[ 128 ];
  snprintf( buf, sizeof(buf), "%lu mismatches over %lu (freq,state) cases", bad, cases );
  check( bad==0UL, "mulhi reciprocal == integer divide", buf );

  ulong bad2 = 0UL, cases2 = 0UL;
  for( uint s=1u; s<ANS_RANS_M_BITS; s++ ) {
    uint f = 1u<<s;
    ans_rans_sym_t r, p;
    ans_rans_sym_init      ( &r, 0u, f );
    ans_rans_sym_init_shift( &p, 0u, f );
    for( int k=0; k<256; k++ ) {
      uint x = ANS_RANS_L + (uint)( urand()*(double)( r.x_max-ANS_RANS_L ) );
      ans_acc_t a = { x, 0u }, b = { x, 0u };
      ans_rans_put      ( &a, &r );
      ans_rans_put_shift( &b, &p );
      bad2 += (ulong)( a.state!=b.state || a.nbit!=b.nbit );
      cases2++;
    }
  }
  snprintf( buf, sizeof(buf), "%lu mismatches over %lu cases", bad2, cases2 );
  check( bad2==0UL, "shift path == reciprocal for pow2 frequencies", buf );
}

/* Striping the features across lanes must not change the answer. */

static void
test_interleave( void ) {
  ans_tans_level_t const * bank = ans_tans_bank();
  double worst = 0.0;
  for( int trial=0; trial<200; trial++ ) {
    int lev[D]; uint bit[D];
    for( int i=0; i<D; i++ ) {
      lev[i] = ans_w_to_level( (float)( (urand()*2.0-1.0)*ANS_W_MAX ) );
      bit[i] = urand()<0.2;
    }
    ans_acc_t one;
    ans_tans_init( &one );
    for( int i=0; i<D; i++ ) ans_tans_put( &one, &bank[ lev[i] ], bit[i] );

    ans_acc_t lane[ ANS_LANES ];
    for( int k=0; k<ANS_LANES; k++ ) ans_tans_init( &lane[k] );
    for( int i=0; i<D; i++ ) ans_tans_put( &lane[ i%ANS_LANES ], &bank[ lev[i] ], bit[i] );
    int sum = 0;
    for( int k=0; k<ANS_LANES; k++ ) sum += ans_tans_codelen_q8( &lane[k] );

    double d = fabs( (double)( sum-ans_tans_codelen_q8( &one ) ) )/256.0;
    if( d>worst ) worst = d;
  }
  char buf[ 128 ];
  snprintf( buf, sizeof(buf), "%d lanes, worst disagreement %.3f bits", ANS_LANES, worst );
  check( worst<8.0, "interleaved lanes sum to the serial code length", buf );
}

/* Picking the shortest code by comparing raw (nbit,state) pairs must
   agree with comparing the code lengths themselves. */

static void
test_lexicographic( void ) {
  ans_tans_level_t const * bank = ans_tans_bank();
  ulong bad = 0UL, near = 0UL, n = 0UL;
  for( int trial=0; trial<20000; trial++ ) {
    ans_acc_t a[ 2 ];
    double    ref[ 2 ];
    for( int c=0; c<2; c++ ) {
      ans_tans_init( &a[c] );
      ref[c] = 0.0;
      for( int i=0; i<64; i++ ) {
        int  lv = ans_w_to_level( (float)( (urand()*2.0-1.0)*ANS_W_MAX ) );
        uint b  = urand()<0.5;
        uint ft = ans_level_freq( lv );
        ref[c] += -log2( (double)( b ? ft : ANS_TANS_SIZE-ft )/(double)ANS_TANS_SIZE );
        ans_tans_put( &a[c], &bank[ lv ], b );
      }
    }
    int lex = ans_acc_lt( &a[0], &a[1] );
    int tru = ref[0]<ref[1];
    n++;
    if( lex!=tru ) { bad++; if( fabs(ref[0]-ref[1])<1.0 ) near++; }
  }
  char buf[ 160 ];
  snprintf( buf, sizeof(buf), "%lu/%lu disagree, %lu of those within 1 bit", bad, n, near );
  check( bad==near, "lexicographic (nbit,state) orders by code length", buf );
}

static void
test_roundtrip( void ) {
  ans_model_t * m = ans_model_new( 64u, 8u, 2u, 4u );
  for( ulong k=0UL; k<8UL*64UL; k++ ) m->w1[k]   = (float)( urand()*2.0-1.0 );
  for( ulong k=0UL; k<8UL*2UL;  k++ ) m->taur[k] = (float)( urand()*10.0 );
  for( ulong k=0UL; k<4UL*m->n_feat; k++ ) m->w2[k] = (float)( urand()*2.0-1.0 );
  ans_model_refresh( m );
  int ok = ans_model_save( m, "/tmp/ans_test_model.ans" )==0;
  ans_model_t * r = ans_model_load( "/tmp/ans_test_model.ans" );
  ok = ok && r && r->n_in==m->n_in && r->n_hid==m->n_hid && r->n_thr==m->n_thr
          && r->n_out==m->n_out && r->n_feat==m->n_feat
          && !memcmp( r->w1,   m->w1,   sizeof(float)*8UL*64UL )
          && !memcmp( r->taur, m->taur, sizeof(float)*8UL*2UL  )
          && !memcmp( r->w2,   m->w2,   sizeof(float)*4UL*m->n_feat );
  check( ok, "model save/load round trip", "" );
  remove( "/tmp/ans_test_model.ans" );
  ans_model_free( m );
  if( r ) ans_model_free( r );
}

/* The blocked routines must agree exactly with each other and with a
   plain per-neuron encode.  n_grp is deliberately not a multiple of
   the unroll factor so the remainder path is exercised. */

static void
test_blocked( void ) {
  ans_tans_level_t const * bank = ans_tans_bank();
  uint const n_in = D, n_grp = 7u, nn = n_grp*ANS_BLK;

  uchar * lv  = malloc( (ulong)n_grp*n_in*ANS_BLK );
  uchar * bit = malloc( n_in );
  int   * ref = malloc( sizeof(int)*nn );
  int   * a   = malloc( sizeof(int)*nn );
  int   * b   = malloc( sizeof(int)*nn );
  int   * c   = malloc( sizeof(int)*nn );
  for( ulong k=0UL; k<(ulong)n_grp*n_in*ANS_BLK; k++ )
    lv[k] = (uchar)ans_w_to_level( (float)( (urand()*2.0-1.0)*ANS_W_MAX ) );
  for( uint i=0u; i<n_in; i++ ) bit[i] = (uchar)( urand()<0.2 );

  /* reference: one accumulator per neuron, plain ans_tans_put */
  for( uint h=0u; h<nn; h++ ) {
    ans_acc_t acc;
    ans_tans_init( &acc );
    for( uint i=0u; i<n_in; i++ )
      ans_tans_put( &acc, &bank[ lv[ ( (ulong)( h/ANS_BLK )*n_in + i )*ANS_BLK + h%ANS_BLK ] ], bit[i] );
    ref[h] = ans_tans_codelen_q8( &acc );
  }
  ans_tans_blk_scalar       ( lv, bit, n_in, n_grp, a );
  ans_tans_blk_avx512       ( lv, bit, n_in, n_grp, b );
  ans_tans_blk_avx512_packed( lv, bit, n_in, n_grp, c );

  ulong da = 0UL, db = 0UL, dc = 0UL;
  for( uint h=0u; h<nn; h++ ) {
    da += (ulong)( a[h]!=ref[h] );
    db += (ulong)( b[h]!=ref[h] );
    dc += (ulong)( c[h]!=ref[h] );
  }
  char buf[ 160 ];
  snprintf( buf, sizeof(buf), "%u neurons, %s", nn,
            ans_has_avx512() ? "avx512 present" : "SCALAR FALLBACK, not testing vector code" );
  check( da==0UL && db==0UL && dc==0UL,
         "blocked scalar/avx512/avx512-packed == per-neuron encode", buf );
  free( lv ); free( bit ); free( ref ); free( a ); free( b ); free( c );
}

/* What an encode step costs.  Eight lanes, as inference runs them, so
   the dependent chain through the state is not what is being timed.
   Every variant streams one byte of weight per step and indexes a
   shared table, so they are priced on the same memory traffic. */

#define BENCH_N (1u<<21)

#define BENCH_RUN( which, stmt ) do {                                  \
    ans_acc_t a[ ANS_LANES ];                                          \
    for( int k=0; k<ANS_LANES; k++ )                                   \
      if( (which)==3 ) ans_tans_init( &a[k] ); else ans_rans_init( &a[k] ); \
    double t0 = now();                                                 \
    for( uint i=0u; i<BENCH_N; i+=ANS_LANES )                          \
      for( uint k=0u; k<ANS_LANES; k++ ) { uint j = i+k; stmt; }       \
    double dt = now()-t0;                                              \
    if( dt<best[which] ) best[which] = dt;                             \
    for( int k=0; k<ANS_LANES; k++ ) sink += a[k].state + a[k].nbit;   \
  } while( 0 )

static void
bench( void ) {
  ans_tans_level_t const * bank = ans_tans_bank();
  ans_rans_sym_t   const * tab  = ans_rans_symtab();

  uchar  * lev = malloc( BENCH_N );
  ushort * frq = malloc( sizeof(ushort)*BENCH_N );
  uchar  * bit = malloc( BENCH_N );
  uchar  * pix = malloc( BENCH_N );
  ans_rans_sym_t pw[ 16 ];                       /* pow2 freqs, bench only */
  for( uint s=1u; s<16u; s++ ) ans_rans_sym_init_shift( &pw[s], 0u, 1u<<s );
  pw[0] = pw[1];
  for( uint i=0u; i<BENCH_N; i++ ) {
    float w = (float)( (urand()*2.0-1.0)*ANS_W_MAX );
    lev[i] = (uchar)ans_w_to_level( w );
    frq[i] = (ushort)ans_p_to_freq( ans_w_to_p( w ), ANS_RANS_M );
    bit[i] = (uchar)( urand()<0.2 );
    /* price the shift path on the frequency each step actually uses,
       rounded to a power of two: a mismatched frequency would change
       how often the state renormalizes and time something else */
    uint f = bit[i] ? frq[i] : ANS_RANS_M-frq[i];
    uint s = 31u - (uint)__builtin_clz( f );
    if( s<15u && ( f-(1u<<s) ) > ( (1u<<(s+1u))-f ) ) s++;
    pix[i] = (uchar)( s<1u ? 1u : s );
  }

  printf( "\n  encode step cost, %u steps, %d lanes\n", BENCH_N, ANS_LANES );
  ulong  sink    = 0UL;
  double best[4] = { 1e30, 1e30, 1e30, 1e30 };
  char const * name[4] = { "rans, integer divide",
                           "rans, mulhi reciprocal",
                           "rans, shift (pow2 freq)",
                           "tans, table lookup" };
  for( int rep=0; rep<7; rep++ ) {
    BENCH_RUN( 0, { uint f = bit[j] ? frq[j] : ANS_RANS_M-frq[j];
                    uint c = bit[j] ? 0u     : frq[j];
                    ans_rans_put_div( &a[k], &tab[ 2u*frq[j]+bit[j] ], f, c ); } );
    BENCH_RUN( 1, ans_rans_put      ( &a[k], &tab[ 2u*frq[j]+bit[j] ] ) );
    BENCH_RUN( 2, ans_rans_put_shift( &a[k], &pw[ pix[j] ] ) );
    BENCH_RUN( 3, ans_tans_put      ( &a[k], &bank[ lev[j] ], bit[j] ) );
  }
  for( int which=0; which<4; which++ )
    printf( "    %-26s %6.2f ns/step   %7.1f Msteps/s   %5.2fx\n", name[which],
            1e9*best[which]/(double)BENCH_N, 1e-6*(double)BENCH_N/best[which],
            best[0]/best[which] );
  printf( "    (sink %lu)\n", sink );
  free( lev ); free( frq ); free( bit ); free( pix );

  /* Across-neuron backends, on the shape inference actually runs:
     512 neurons over 784 features. */
  uint const n_in = 784u, n_grp = 32u, iters = 24u;
  uchar * lv2 = malloc( (ulong)n_grp*n_in*ANS_BLK );
  uchar * bt2 = malloc( n_in );
  int   * o2  = malloc( sizeof(int)*(ulong)n_grp*ANS_BLK );
  for( ulong k=0UL; k<(ulong)n_grp*n_in*ANS_BLK; k++ )
    lv2[k] = (uchar)ans_w_to_level( (float)( (urand()*2.0-1.0)*ANS_W_MAX ) );
  for( uint i=0u; i<n_in; i++ ) bt2[i] = (uchar)( urand()<0.2 );
  ulong steps = (ulong)n_grp*ANS_BLK*n_in;

  printf( "\n  across-neuron tans, %u neurons x %u features, unroll %d\n",
          n_grp*ANS_BLK, n_in, ANS_AVX_UNROLL );
  char const * bn[3] = { "blocked scalar", "avx512, 8 MiB lut", "avx512, 4 MiB lut" };
  double bb[3] = { 1e30, 1e30, 1e30 };
  for( uint rep=0u; rep<5u; rep++ )
    for( int w=0; w<3; w++ ) {
      double t0 = now();
      for( uint it=0u; it<iters; it++ ) {
        if     ( w==0 ) ans_tans_blk_scalar       ( lv2, bt2, n_in, n_grp, o2 );
        else if( w==1 ) ans_tans_blk_avx512       ( lv2, bt2, n_in, n_grp, o2 );
        else            ans_tans_blk_avx512_packed( lv2, bt2, n_in, n_grp, o2 );
      }
      double dt = ( now()-t0 )/(double)iters;
      if( dt<bb[w] ) bb[w] = dt;
      sink += (ulong)o2[0];
    }
  for( int w=0; w<3; w++ )
    printf( "    %-26s %6.2f ns/step   %7.1f Msteps/s   %5.2fx\n", bn[w],
            1e9*bb[w]/(double)steps, 1e-6*(double)steps/bb[w], bb[0]/bb[w] );
  printf( "    (sink %lu)\n", sink );
  free( lv2 ); free( bt2 ); free( o2 );
}


/* uABS.  The vector path must agree with the scalar one exactly, and
   the relaxed step - a rounded reciprocal in place of the exact
   division - must track the true bijective codec and the ideal code
   length.  See ans_uabs.h for what the relaxation gives up. */

static void
test_uabs( void ) {
  uint const n_in = D, n_grp = 7u, nn = n_grp*ANS_BLK;

  ushort * r0  = malloc( sizeof(ushort)*(ulong)n_grp*n_in*ANS_BLK );
  ushort * r1  = malloc( sizeof(ushort)*(ulong)n_grp*n_in*ANS_BLK );
  float  * w   = malloc( sizeof(float )*(ulong)n_grp*n_in*ANS_BLK );
  uchar  * bit = malloc( n_in );
  int    * ref = malloc( sizeof(int)*nn );
  int    * a   = malloc( sizeof(int)*nn );
  int    * b   = malloc( sizeof(int)*nn );
  for( ulong k=0UL; k<(ulong)n_grp*n_in*ANS_BLK; k++ ) {
    w [k] = (float)( (urand()*2.0-1.0)*ANS_W_MAX );
    r0[k] = ans_uabs_r( w[k], 0u );
    r1[k] = ans_uabs_r( w[k], 1u );
  }
  for( uint i=0u; i<n_in; i++ ) bit[i] = (uchar)( urand()<0.2 );

  /* reference: one accumulator per neuron, plain ans_uabs_put */
  for( uint h=0u; h<nn; h++ ) {
    ans_acc_t acc;
    ans_uabs_init( &acc );
    for( uint i=0u; i<n_in; i++ ) {
      ulong o = ( (ulong)( h/ANS_BLK )*n_in + i )*ANS_BLK + h%ANS_BLK;
      ans_uabs_put( &acc, bit[i] ? r1[o] : r0[o] );
    }
    ref[h] = ans_uabs_codelen_q8( &acc );
  }
  ans_uabs_blk_scalar( r0, r1, bit, n_in, n_grp, a );
  ans_uabs_blk_avx512( r0, r1, bit, n_in, n_grp, b );

  ulong da = 0UL, db = 0UL;
  for( uint h=0u; h<nn; h++ ) { da += (ulong)( a[h]!=ref[h] ); db += (ulong)( b[h]!=ref[h] ); }
  char buf[ 160 ];
  snprintf( buf, sizeof(buf), "%u neurons, %s", nn,
            ans_has_avx512() ? "avx512 present" : "SCALAR FALLBACK, not testing vector code" );
  check( da==0UL && db==0UL, "uabs blocked scalar/avx512 == per-neuron encode", buf );

  /* code length: relaxed step, exact bijective step, ideal */
  double er = 0.0, ee = 0.0;
  for( uint h=0u; h<nn; h++ ) {
    ans_acc_t ar, ae;
    double    ideal = 0.0;
    ans_uabs_init( &ar ); ans_uabs_init( &ae );
    for( uint i=0u; i<n_in; i++ ) {
      ulong o  = ( (ulong)( h/ANS_BLK )*n_in + i )*ANS_BLK + h%ANS_BLK;
      double p = (double)ans_w_to_p( w[o] );
      if( !bit[i] ) p = 1.0-p;
      uint f = (uint)ans_p_to_freq( ans_w_to_p( w[o] ), ANS_RANS_M );
      ideal += -log2( p );
      ans_uabs_put      ( &ar, bit[i] ? r1[o] : r0[o] );
      ans_uabs_exact_put( &ae, f, bit[i] );
    }
    er += ( (double)ans_uabs_codelen_q8( &ar )/256.0 - ideal )/ideal;
    ee += ( (double)ans_uabs_codelen_q8( &ae )/256.0 - ideal )/ideal;
  }
  er = 100.0*er/(double)nn;
  ee = 100.0*ee/(double)nn;
  snprintf( buf, sizeof(buf), "relaxed %+.4f%%, exact %+.4f%% (tans is -0.079%% at log 12)", er, ee );
  check( fabs( er )<0.05 && fabs( ee )<0.05, "uabs code length excess", buf );

  /* the exact codec is a bijection; the relaxed one is not, and is not
     claimed to be */
  ulong bad = 0UL, n = 0UL;
  for( uint f=256u; f<65280u; f+=1021u )
    for( uint x=ANS_UABS_L; x<ANS_UABS_L+4096u; x++ )
      for( uint sym=0u; sym<2u; sym++ ) {
        ans_acc_t acc = { x, 0u };
        ans_uabs_exact_put( &acc, f, sym );
        ulong M  = 1UL<<16;
        uint  y  = acc.state;
        uint  c  = (uint)( ( (ulong)y*f + M-1UL )>>16 );
        uint  c1 = (uint)( ( (ulong)( y+1u )*f + M-1UL )>>16 );
        uint  s2 = c1-c;
        uint  z  = s2 ? c : y-c;
        n++;
        if( s2!=sym || z!=( x>>acc.nbit ) ) bad++;
      }
  snprintf( buf, sizeof(buf), "%lu states x probabilities", n );
  check( !bad, "uabs exact step decodes back to (symbol, state)", buf );

  free( r0 ); free( r1 ); free( w ); free( bit ); free( ref ); free( a ); free( b );
}

int
main( int     argc,
      char ** argv ) {
  int do_bench = 1;
  for( int i=1; i<argc; i++ ) if( !strcmp( argv[i], "--no-bench" ) ) do_bench = 0;

  printf( "ans_test  tans log %d, %d levels   rans M 2^%d, L 2^23\n\n",
          ANS_TANS_LOG, ANS_TANS_LEVEL_CNT, ANS_RANS_M_BITS );
  test_codelen();
  test_reciprocal();
  test_interleave();
  test_blocked();
  test_uabs();
  test_lexicographic();
  test_roundtrip();
  if( do_bench ) bench();

  printf( "\n%s\n", fails ? "FAILED" : "all tests passed" );
  return fails ? 1 : 0;
}
