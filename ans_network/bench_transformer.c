/* bench_transformer - price a binarized transformer's weight matmuls
   as ANS encode passes against fp32 and int8 doing the same arithmetic.

   See transformer.h for what is and is not an encode pass here, and
   TRANSFORMER_PLAN.md for the question.  Three sections:

     --check   every kernel against a scalar reference, exactly where
               the reference is integer.  Nothing is timed before this
               passes.
     --dump    the fp32 reference forward pass on a tiny config, as
               bit vectors and logits, for check_ref.py to reproduce
     default   phase 1, the linear layers alone at batch 1 and batch
               32, and phase 2, the whole forward pass

   Every accumulated result is consumed into a volatile.  Two
   benchmarks in this directory have already been eaten by dead code
   elimination; an impossible throughput is that, not a discovery. */

#include "transformer.h"
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
now( void ) {
  struct timespec ts;
  clock_gettime( CLOCK_MONOTONIC, &ts );
  return (double)ts.tv_sec + 1e-9*(double)ts.tv_nsec;
}

static uint pad64( uint n ) { return (n+63u) & ~63u; }

static void *
xalloc( ulong sz ) {
  sz = ( sz+63UL ) & ~63UL;
  void * p = aligned_alloc( 64UL, sz );
  if( !p ) { fprintf( stderr, "oom\n" ); exit( 1 ); }
  memset( p, 0, sz );
  return p;
}

static volatile double sink;

/* Correctness **********************************************************/

static int fails = 0;

static void
check( int          ok,
       char const * what,
       char const * detail ) {
  printf( "  %-46s %s   %s\n", what, ok ? "ok  " : "FAIL", detail );
  if( !ok ) fails++;
}

static void
kernel_checks( void ) {
  uint const n_in = 256u, s_in = pad64( n_in ), n_out = 48u, nb = 5u;

  xf_seed( 777UL );
  /* a linear of this shape, built by the same compile path the model
     uses, so the three representations are the ones being timed */
  xf_lin_t L;
  xf_lin_new( &L, n_out, n_in );

  uchar * bit = xalloc( (ulong)nb*s_in );
  float * xf  = xalloc( sizeof(float)*(ulong)nb*s_in );
  for( uint b=0u; b<nb; b++ )
    for( uint i=0u; i<n_in; i++ ) {
      uchar v = (uchar)( xf_rnd()&1UL );
      bit[ (ulong)b*s_in + i ] = v;
      xf [ (ulong)b*s_in + i ] = (float)v;
    }

  printf( "kernel checks\n" );

  /* fp32 */
  /* fp32 sums in a different order than a double accumulator does, so
     the error is measured against the rms of the outputs rather than
     against a single one, which can sit near zero */
  float * y  = xalloc( sizeof(float)*(ulong)nb*n_out );
  float * y2 = xalloc( sizeof(float)*(ulong)nb*n_out );
  double emax = 0.0, rms = 0.0;
  for( uint b=0u; b<nb; b++ ) {
    xf_fgemv( L.w, n_out, s_in, xf + (ulong)b*s_in, y + (ulong)b*n_out );
    for( uint o=0u; o<n_out; o++ ) {
      double s = 0.0;
      for( uint i=0u; i<s_in; i++ ) s += (double)L.w[ (ulong)o*s_in + i ]*(double)xf[ (ulong)b*s_in + i ];
      double e = fabs( s - (double)y[ (ulong)b*n_out + o ] );
      rms += s*s;
      if( e>emax ) emax = e;
    }
  }
  rms = sqrt( rms/(double)( nb*n_out ) );
  char buf[ 128 ];
  snprintf( buf, sizeof(buf), "max err %.2e, %.1e of rms %.2f", emax, emax/rms, rms );
  check( emax/rms<1e-4, "fp32 gemv == scalar dot", buf );

  xf_fgemm( L.w, n_out, s_in, xf, nb, y2 );
  emax = 0.0;
  for( ulong i=0UL; i<(ulong)nb*n_out; i++ ) {
    double e = fabs( (double)y2[i]-(double)y[i] );
    if( e>emax ) emax = e;
  }
  snprintf( buf, sizeof(buf), "max err %.2e, %.1e of rms %.2f", emax, emax/rms, rms );
  check( emax/rms<1e-4, "fp32 gemm == fp32 gemv", buf );

  /* int8 - integer, so exact */
  int * iy  = xalloc( sizeof(int)*(ulong)nb*n_out );
  int * iy2 = xalloc( sizeof(int)*(ulong)nb*n_out );
  ulong bad = 0UL;
  for( uint b=0u; b<nb; b++ ) {
    xf_i8gemv( L.q, n_out, s_in, bit + (ulong)b*s_in, iy + (ulong)b*n_out );
    for( uint o=0u; o<n_out; o++ ) {
      int s = 0;
      for( uint i=0u; i<s_in; i++ ) s += (int)L.q[ (ulong)o*s_in + i ]*(int)bit[ (ulong)b*s_in + i ];
      bad += (ulong)( s!=iy[ (ulong)b*n_out + o ] );
    }
  }
  snprintf( buf, sizeof(buf), "%lu of %u differ", bad, nb*n_out );
  check( !bad, "int8 vnni gemv == scalar dot", buf );

  xf_i8gemm( L.q, n_out, s_in, bit, nb, iy2 );
  bad = 0UL;
  for( ulong i=0UL; i<(ulong)nb*n_out; i++ ) bad += (ulong)( iy2[i]!=iy[i] );
  snprintf( buf, sizeof(buf), "%lu of %u differ", bad, nb*n_out );
  check( !bad, "int8 vnni gemm == int8 vnni gemv", buf );

  /* uABS across tokens against the shipped across-neuron kernel, which
     is the reference that matters: same reciprocals, same step order,
     so the two must agree bit for bit */
  uchar * tb  = xalloc( (ulong)16*s_in );
  for( ulong i=0UL; i<(ulong)16*s_in; i++ ) tb[i] = (uchar)( i%s_in<n_in ? ( xf_rnd()&1UL ) : 0UL );
  ushort * msk = xalloc( sizeof(ushort)*s_in );
  for( uint i=0u; i<s_in; i++ ) {
    uint m = 0u;
    for( uint t=0u; t<16u; t++ ) m |= (uint)tb[ (ulong)t*s_in + i ]<<t;
    msk[i] = (ushort)m;
  }
  int * ref = xalloc( sizeof(int)*(ulong)16*( (n_out+15u)&~15u ) );
  int * tva = xalloc( sizeof(int)*(ulong)n_out*16 );
  int * tvs = xalloc( sizeof(int)*(ulong)n_out*16 );
  for( uint t=0u; t<16u; t++ )
    ans_uabs_blk_avx512( L.r[0], L.r[1], tb + (ulong)t*s_in, s_in, L.n_grp,
                         ref + (ulong)t*( (n_out+15u)&~15u ) );
  xf_uabs_tok_scalar( L.rt[0], L.rt[1], msk, s_in, n_out, tvs );
  xf_uabs_tok_avx512( L.rt[0], L.rt[1], msk, s_in, n_out, tva );
  ulong bad_s = 0UL, bad_v = 0UL;
  for( uint o=0u; o<n_out; o++ )
    for( uint t=0u; t<16u; t++ ) {
      int rr = ref[ (ulong)t*( (n_out+15u)&~15u ) + o ];
      bad_s += (ulong)( tvs[ (ulong)o*16+t ]!=rr );
      bad_v += (ulong)( tva[ (ulong)o*16+t ]!=rr );
    }
  snprintf( buf, sizeof(buf), "scalar %lu, avx512 %lu of %u differ", bad_s, bad_v, n_out*16u );
  check( !bad_s && !bad_v, "uabs across tokens == uabs across neurons", buf );

  /* the identity itself: -codelen is the float pre-activation */
  int * cl = xalloc( sizeof(int)*(ulong)L.n_grp*16 );
  ans_uabs_blk_avx512( L.r[0], L.r[1], bit, s_in, L.n_grp, cl );
  double eabs = 0.0, esd = 0.0, mean = 0.0;
  for( uint o=0u; o<n_out; o++ ) {
    double zf = (double)y[o] - (double)L.bias_f[o];
    double za = -(double)cl[o]/256.0;
    mean += zf;
    if( fabs( za-zf )>eabs ) eabs = fabs( za-zf );
  }
  mean /= n_out;
  for( uint o=0u; o<n_out; o++ ) {
    double zf = (double)y[o] - (double)L.bias_f[o];
    esd += ( zf-mean )*( zf-mean );
  }
  esd = sqrt( esd/n_out );
  snprintf( buf, sizeof(buf), "max |dz| %.4f bits, %.2f%% of sd %.2f", eabs, 100.0*eabs/esd, esd );
  check( eabs<0.5, "uabs codelen == fp32 pre-activation", buf );

  int * iq = xalloc( sizeof(int)*n_out );
  xf_i8gemv( L.q, n_out, s_in, bit, iq );
  eabs = 0.0;
  for( uint o=0u; o<n_out; o++ ) {
    double zf = (double)y[o] - (double)L.bias_f[o];
    double zq = (double)L.qs[o]*(double)iq[o] - (double)L.bias_q[o];
    if( fabs( zq-zf )>eabs ) eabs = fabs( zq-zf );
  }
  snprintf( buf, sizeof(buf), "max |dz| %.4f bits, %.2f%% of sd %.2f", eabs, 100.0*eabs/esd, esd );
  check( eabs<2.0, "int8 dot == fp32 pre-activation", buf );

  free( bit ); free( xf ); free( y ); free( y2 ); free( iy ); free( iy2 );
  free( tb ); free( msk ); free( ref ); free( tva ); free( tvs ); free( cl ); free( iq );
  xf_lin_free( &L );
}

/* Phase 1: the linear layers alone *************************************/

/* Every weight matrix in the model, in forward order.  A pass over all
   of them is one token's worth of matmul: 12*d_model^2 per layer plus
   the output projection, 12.6 M weights at the default shape, which is
   50 MiB of fp32 and well past any cache. */

#define LIN_MAX (256)

struct p1 {
  xf_lin_t const * lin[ LIN_MAX ];
  uint             n_lin;
  uchar *          bit;   /* [batch][s_max] */
  float *          xf;    /* [batch][s_max] */
  ushort *         msk;   /* [s_max] */
  float *          fy;    /* [batch][n_out_max] */
  int *            iy;
  int *            cl;    /* [n_out_max*16] */
  uint             s_max;
  uint             batch;
  int              kern;
};

#define K_UABS1 (0)
#define K_F32V  (1)
#define K_I8V   (2)
#define K_UABSB (3)
#define K_UABST (4)
#define K_F32M  (5)
#define K_I8M   (6)

static void
tok_transpose( uchar const * bit,
               uint          s_max,
               uint          n_in,
               ushort *      msk ) {
  for( uint i=0u; i<n_in; i++ ) {
    uint m = 0u;
    for( uint t=0u; t<16u; t++ ) m |= (uint)bit[ (ulong)t*s_max + i ]<<t;
    msk[i] = (ushort)m;
  }
}

static double
run_p1( void * a ) {
  struct p1 * c   = a;
  double      acc = 0.0;
  uint        B   = c->batch;
  for( uint li=0u; li<c->n_lin; li++ ) {
    xf_lin_t const * L = c->lin[li];
    switch( c->kern ) {

      case K_UABS1:
        ans_uabs_blk_avx512( L->r[0], L->r[1], c->bit, L->s_in, L->n_grp, c->cl );
        acc += (double)c->cl[0];
        break;

      case K_UABSB:
        for( uint b=0u; b<B; b++ ) {
          ans_uabs_blk_avx512( L->r[0], L->r[1], c->bit + (ulong)b*c->s_max,
                               L->s_in, L->n_grp, c->cl );
          acc += (double)c->cl[0];
        }
        break;

      case K_UABST:
        for( uint b=0u; b<B; b+=16u ) {
          tok_transpose( c->bit + (ulong)b*c->s_max, c->s_max, L->s_in, c->msk );
          xf_uabs_tok_avx512( L->rt[0], L->rt[1], c->msk, L->s_in, L->n_out, c->cl );
          acc += (double)c->cl[0];
        }
        break;

      case K_F32V:
        xf_fgemv( L->w, L->n_out, L->s_in, c->xf, c->fy );
        acc += (double)c->fy[0];
        break;

      case K_F32M:
        xf_fgemm( L->w, L->n_out, L->s_in, c->xf, B, c->fy );
        acc += (double)c->fy[0];
        break;

      case K_I8V:
        xf_i8gemv( L->q, L->n_out, L->s_in, c->bit, c->iy );
        acc += (double)c->iy[0];
        break;

      default:
        xf_i8gemm( L->q, L->n_out, L->s_in, c->bit, B, c->iy );
        acc += (double)c->iy[0];
        break;
    }
  }
  sink += acc;
  return (double)( c->kern<K_UABSB ? 1u : B );
}

/* Best of three, each trial long enough to be stable.  Same protocol
   as bench_float.c. */

static double
bench( double (*fn)( void * ),
       void *  arg,
       double  secs ) {
  double best = 1e30;
  for( int trial=0; trial<3; trial++ ) {
    double t0 = now(), work = 0.0;
    do work += fn( arg ); while( now()-t0 < secs/3.0 );
    double dt = now()-t0;
    if( dt/work < best ) best = dt/work;
  }
  return best;
}

/* Phase 2: the whole forward pass **************************************/

struct p2 { xf_run_t * r; int mode; };

static double
run_p2( void * a ) {
  struct p2 * c = a;
  xf_forward( c->r, c->mode, 0 );
  sink += (double)c->r->logit[0];
  return (double)c->r->x->cfg.seq;
}

/* The plane stride diagnostic ******************************************

   A uABS group's reciprocals are [group][feature][16], so the stride
   between the groups a chain runs is n_in*32 bytes.  With n_in a
   multiple of 64 - which every natural transformer width is - that is a
   multiple of 16 KiB, and the eight chains in flight land in one L1
   set.  This measures what that costs by padding n_in by 16, which is
   3% more steps and breaks the alias.  It is a property of the layout,
   not of the transformer, so it is reported as a footnote rather than
   used in the tables. */

static void
stride_probe( double mib ) {
  printf( "uabs plane stride, %.1f MiB of reciprocals per plane pair\n", mib );
  printf( "  %-10s %8s %14s %10s\n", "n_in", "n_grp", "Gweight/s", "vs pow2" );
  uint nn[4] = { 512u, 528u, 2048u, 2064u };
  double base = 0.0;
  for( uint j=0u; j<4u; j++ ) {
    uint  n_in  = nn[j];
    uint  n_grp = (uint)( (ulong)( mib*1048576.0 )/( 4UL*n_in*ANS_BLK ) );
    n_grp &= ~7u;                  /* a multiple of the kernel's chain count,
                                      or the one-group tail path dominates */
    if( !n_grp ) n_grp = 8u;
    ulong np    = (ulong)n_grp*n_in*ANS_BLK;
    ushort * r0 = xalloc( sizeof(ushort)*np );
    ushort * r1 = xalloc( sizeof(ushort)*np );
    uchar *  bt = xalloc( n_in );
    int *    o  = xalloc( sizeof(int)*(ulong)n_grp*ANS_BLK );
    for( ulong i=0UL; i<np; i++ ) {
      r0[i] = ans_uabs_r( xf_rndw( -2.0f, 2.0f ), 0u );
      r1[i] = ans_uabs_r( xf_rndw( -2.0f, 2.0f ), 1u );
    }
    for( uint i=0u; i<n_in; i++ ) bt[i] = (uchar)( xf_rnd()&1UL );
    ulong  w = (ulong)n_grp*ANS_BLK*n_in;
    double best = 1e30;
    for( int t=0; t<3; t++ ) {
      double t0 = now(); ulong n = 0UL;
      do { ans_uabs_blk_avx512( r0, r1, bt, n_in, n_grp, o ); n++; } while( now()-t0<0.3 );
      double dt = ( now()-t0 )/(double)n;
      if( dt<best ) best = dt;
    }
    sink += (double)o[0];
    double g = 1e-9*(double)w/best;
    if( !(j&1u) ) base = g;
    printf( "  %-10u %8u %14.2f %9.2fx\n", n_in, n_grp, g, g/base );
    free( r0 ); free( r1 ); free( bt ); free( o );
  }
}

/* main *****************************************************************/

static void
dump( void ) {
  /* A shape small enough for a pure Python implementation to follow,
     with the same seed, so the fp32 reference is checked against code
     that shares nothing with it. */
  xf_cfg_t cfg = { .d_model = 16u, .n_head = 2u, .d_ff = 32u,
                   .n_layer = 2u, .seq = 6u, .vocab = 16u };
  xf_t *     x = xf_new( &cfg, 12345UL );
  xf_run_t * r = xf_run_new( x );
  printf( "tok" );
  for( uint t=0u; t<cfg.seq; t++ ) printf( " %u", r->tok[t] );
  printf( "\n" );
  xf_set_trace( stdout );
  xf_forward( r, XF_LIN_F32, 1 );
  xf_set_trace( NULL );
  for( uint t=0u; t<cfg.seq; t++ ) {
    printf( "logit" );
    for( uint c=0u; c<cfg.vocab; c++ ) printf( " %.6e", (double)r->logit[ (ulong)t*cfg.vocab + c ] );
    printf( "\n" );
  }
  xf_run_free( r ); xf_free( x );
}

int
main( int     argc,
      char ** argv ) {
  uint   d_model = 512u, n_head = 8u, d_ff = 2048u, n_layer = 4u;
  uint   seq = 256u, vocab = 256u, batch = 32u;
  double secs = 1.5;
  int    do_check = 0, do_dump = 0;
  double do_stride = 0.0;

  for( int i=1; i<argc; i++ ) {
    if     ( !strcmp(argv[i],"--check") ) do_check = 1;
    else if( !strcmp(argv[i],"--dump" ) ) do_dump  = 1;
    else if( !strcmp(argv[i],"--stride") && i+1<argc ) do_stride = atof(argv[++i]);
    else if( !strcmp(argv[i],"--d"     ) && i+1<argc ) d_model = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--ff"    ) && i+1<argc ) d_ff    = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--layers") && i+1<argc ) n_layer = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--seq"   ) && i+1<argc ) seq     = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--batch" ) && i+1<argc ) batch   = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--secs"  ) && i+1<argc ) secs    = atof(argv[++i]);
    else { fprintf( stderr, "usage: %s [--check] [--dump] [--d N] [--ff N] [--layers N] "
                            "[--seq N] [--batch N] [--secs S] [--stride MiB]\n", argv[0] ); return 1; }
  }

  if( do_dump ) { dump(); return 0; }
  if( do_stride>0.0 ) { xf_seed( 99UL ); stride_probe( do_stride ); return 0; }

  if( do_check ) { kernel_checks(); printf( "\n%s\n", fails ? "FAILED" : "all kernel checks passed" ); return fails ? 1 : 0; }

  xf_cfg_t cfg = { d_model, n_head, d_ff, n_layer, seq, vocab };
  xf_t *     x = xf_new( &cfg, 20260918UL );
  xf_run_t * r = xf_run_new( x );

  printf( "d_model %u  n_head %u  d_ff %u  n_layer %u  seq %u  vocab %u  batch %u\n",
          d_model, n_head, d_ff, n_layer, seq, vocab, batch );
  printf( "%lu weights in the linear layers   fp32 %.1f MiB   uabs %.1f MiB (2 planes x 2 B)"
          "   int8 %.1f MiB\n\n",
          x->n_weight, (double)(x->n_weight*4UL)/1048576.0,
          (double)(x->n_weight*4UL)/1048576.0, (double)x->n_weight/1048576.0 );

  kernel_checks();
  if( fails ) { printf( "\nFAILED - not timing anything\n" ); return 1; }
  printf( "\n" );

  /* Phase 1 ***********************************************************/

  uint s_max = pad64( d_ff>d_model ? d_ff : d_model );
  uint o_max = d_ff>d_model ? d_ff : d_model;

  struct p1 c = { .s_max = s_max, .batch = batch };
  c.bit = xalloc( (ulong)batch*s_max );
  c.xf  = xalloc( sizeof(float)*(ulong)batch*s_max );
  c.msk = xalloc( sizeof(ushort)*s_max );
  c.fy  = xalloc( sizeof(float)*(ulong)batch*o_max );
  c.iy  = xalloc( sizeof(int  )*(ulong)batch*o_max );
  c.cl  = xalloc( sizeof(int  )*(ulong)o_max*16 );
  for( uint b=0u; b<batch; b++ )
    for( uint i=0u; i<s_max; i++ ) {
      uchar v = (uchar)( xf_rnd()&1UL );
      c.bit[ (ulong)b*s_max + i ] = v;
      c.xf [ (ulong)b*s_max + i ] = (float)v;
    }

  /* Two footprints, because they are bound by different things and the
     conclusion differs between them.  Hot is one d_model x d_model
     matrix, which is 1 MiB of fp32 and L2 resident, so nothing is
     waiting on memory and what shows is the instruction ratio.  Cold is
     every weight in the model once per token, 48 MiB, which is what
     autoregressive decode actually does. */

  struct { char const * name; uint n_lin; ulong nw; } foot[2];
  foot[0].name = "hot: one d_model x d_model matrix, L2 resident";
  foot[0].n_lin = 1u;
  foot[0].nw = (ulong)d_model*d_model;
  foot[1].name = "cold: every weight in the model, streamed";
  foot[1].n_lin = 0u;                              /* filled in below */
  foot[1].nw = x->n_weight;

  xf_lin_t const * all[ LIN_MAX ];
  uint n_all = 0u;
  if( 6u*n_layer+1u>LIN_MAX ) { fprintf( stderr, "LIN_MAX too small\n" ); return 1; }
  for( uint l=0u; l<n_layer; l++ ) {
    xf_layer_t const * L = &x->layer[l];
    all[ n_all++ ] = &L->wq; all[ n_all++ ] = &L->wk;
    all[ n_all++ ] = &L->wv; all[ n_all++ ] = &L->wo;
    all[ n_all++ ] = &L->w1; all[ n_all++ ] = &L->w2;
  }
  all[ n_all++ ] = &x->wout;
  foot[1].n_lin = n_all;

  struct { char const * name; int kern; int batched; } kk[] = {
    { "uabs encode, across neurons",  K_UABS1, 0 },
    { "fp32 avx512 gemv",             K_F32V,  0 },
    { "int8 vnni gemv",               K_I8V,   0 },
    { "uabs encode, across neurons",  K_UABSB, 1 },
    { "uabs encode, across tokens",   K_UABST, 1 },
    { "fp32 avx512 gemm",             K_F32M,  1 },
    { "int8 vnni gemm",               K_I8M,   1 },
  };

  printf( "phase 1: the linear layers alone\n" );
  for( uint f=0u; f<2u; f++ ) {
    memcpy( c.lin, all, sizeof(xf_lin_t const *)*foot[f].n_lin );
    c.n_lin = foot[f].n_lin;
    printf( "\n  %s\n  %lu weights per token, uabs %.2f MiB, fp32 %.2f MiB, int8 %.2f MiB\n",
            foot[f].name, foot[f].nw,
            (double)(foot[f].nw*4UL)/1048576.0, (double)(foot[f].nw*4UL)/1048576.0,
            (double)foot[f].nw/1048576.0 );
    printf( "  %-32s %10s %10s %12s\n", "kernel", "batch", "us/token", "Gweight/s" );
    for( uint i=0u; i<7u; i++ ) {
      c.kern = kk[i].kern;
      double t = bench( run_p1, &c, secs );
      char bs[ 16 ];
      snprintf( bs, sizeof(bs), "%u", kk[i].batched ? batch : 1u );
      printf( "  %-32s %10s %10.1f %12.1f\n", kk[i].name, bs, 1e6*t,
              1e-9*(double)foot[f].nw/t );
    }
  }

  /* Phase 2 ***********************************************************/

  /* The sites, in the order binz writes them, so a flip can be
     attributed to the place it happened. */

  ulong per_layer = (ulong)seq*( 3UL*d_model + d_ff );
  ulong blog_cap  = (ulong)n_layer*per_layer + (ulong)seq*d_model;
  uchar * bl_f = xalloc( blog_cap );
  uchar * bl_x = xalloc( blog_cap );
  float * lg_f = xalloc( sizeof(float)*(ulong)seq*vocab );

  xf_forward( r, XF_LIN_F32, 1 );          /* calibrate the thresholds */

  xf_bitlog( bl_f, blog_cap );
  xf_forward( r, XF_LIN_F32, 0 );
  ulong bl_n = xf_bitlog_n();
  xf_bitlog( NULL, 0UL );
  memcpy( lg_f, r->logit, sizeof(float)*(ulong)seq*vocab );

  double t_fwd[ XF_LIN_CNT ];
  ulong  flip[ XF_LIN_CNT ][ 8 ][ 5 ];   /* [mode][layer][site], site 4 = final */
  ulong  top1[ XF_LIN_CNT ];
  double lerr[ XF_LIN_CNT ];

  struct p2 p2 = { .r = r };
  for( int mode=0; mode<XF_LIN_CNT; mode++ ) {
    p2.mode = mode;
    t_fwd[ mode ] = bench( run_p2, &p2, secs );
    memset( flip[mode], 0, sizeof(flip[mode]) );
    if( n_layer>8u ) { fprintf( stderr, "flip table holds 8 layers\n" ); return 1; }
    top1[ mode ] = 0UL; lerr[ mode ] = 0.0;
    if( mode==XF_LIN_NONE ) continue;

    xf_bitlog( bl_x, blog_cap );
    xf_forward( r, mode, 0 );
    xf_bitlog( NULL, 0UL );
    for( uint l=0u; l<n_layer; l++ ) {
      ulong o = (ulong)l*per_layer;
      ulong sz[4] = { (ulong)seq*d_model, (ulong)seq*d_model, (ulong)seq*d_model, (ulong)seq*d_ff };
      for( uint st=0u; st<4u; st++ ) {
        for( ulong i=0UL; i<sz[st]; i++ ) flip[mode][l][st] += (ulong)( bl_x[o+i]!=bl_f[o+i] );
        o += sz[st];
      }
    }
    ulong o = (ulong)n_layer*per_layer;
    for( ulong i=0UL; i<(ulong)seq*d_model; i++ )
      flip[mode][0][4] += (ulong)( bl_x[o+i]!=bl_f[o+i] );

    double s2 = 0.0;
    for( uint t=0u; t<seq; t++ ) {
      uint af = 0u, ax = 0u;
      for( uint v=1u; v<vocab; v++ ) {
        if( lg_f    [ (ulong)t*vocab+v ]>lg_f    [ (ulong)t*vocab+af ] ) af = v;
        if( r->logit[ (ulong)t*vocab+v ]>r->logit[ (ulong)t*vocab+ax ] ) ax = v;
      }
      top1[ mode ] += (ulong)( af==ax );
      for( uint v=0u; v<vocab; v++ ) {
        double d = (double)r->logit[ (ulong)t*vocab+v ] - (double)lg_f[ (ulong)t*vocab+v ];
        lerr[ mode ] += d*d;
        s2 += (double)lg_f[ (ulong)t*vocab+v ]*(double)lg_f[ (ulong)t*vocab+v ];
      }
    }
    lerr[ mode ] = sqrt( lerr[mode]/s2 );
  }

  double t_rem = t_fwd[ XF_LIN_NONE ];
  printf( "\nphase 2: the whole forward pass, seq %u, one token at a time through each\n"
          "matrix - the weights are reused across the sequence, so this is prefill shaped\n\n", seq );
  printf( "  %-14s %10s %10s %10s %10s %10s %10s\n",
          "path", "us/token", "matmul", "remainder", "matmul %", "top1", "logit err" );
  for( int mode=0; mode<XF_LIN_CNT; mode++ ) {
    if( mode==XF_LIN_NONE ) {
      printf( "  %-14s %10.1f %10s %10.1f %10s %10s %10s\n",
              xf_lin_name[mode], 1e6*t_fwd[mode], "-", 1e6*t_rem, "-", "-", "-" );
      continue;
    }
    double mm = t_fwd[mode]-t_rem;
    printf( "  %-14s %10.1f %10.1f %10.1f %9.1f%% %9.1f%% %10.3f\n",
            xf_lin_name[mode], 1e6*t_fwd[mode], 1e6*mm, 1e6*t_rem, 100.0*mm/t_fwd[mode],
            100.0*(double)top1[mode]/(double)seq, lerr[mode] );
  }

  /* A flip at the first site is the quantizer alone: both paths read the
     same stream there.  Every later site reads a stream the earlier
     flips have already moved, so what those columns show is how much
     the binarized model amplifies a perturbation, not how good the
     coder is. */

  printf( "\n  bit flips against the fp32 path, %% of decisions at each site\n" );
  char const * sname[5] = { "stream -> Q,K,V", "context -> O", "stream -> FF1",
                            "FF1 -> FF2", "final -> logits" };
  ulong ssz[5] = { (ulong)seq*d_model, (ulong)seq*d_model, (ulong)seq*d_model,
                   (ulong)seq*d_ff,    (ulong)seq*d_model };
  for( int mode=XF_LIN_ANS; mode<=XF_LIN_I8; mode++ ) {
    printf( "  %-20s", xf_lin_name[mode] );
    for( uint l=0u; l<n_layer; l++ ) printf( " %9s%u", "layer ", l );
    printf( "\n" );
    for( uint st=0u; st<4u; st++ ) {
      printf( "  %-20s", sname[st] );
      for( uint l=0u; l<n_layer; l++ )
        printf( " %9.3f%%", 100.0*(double)flip[mode][l][st]/(double)ssz[st] );
      printf( "\n" );
    }
    printf( "  %-20s %9.3f%%\n", sname[4],
            100.0*(double)flip[mode][0][4]/(double)ssz[4] );
  }
  printf( "  (%lu decisions in total)\n", bl_n );

  free( c.bit ); free( c.xf ); free( c.msk ); free( c.fy ); free( c.iy ); free( c.cl );
  free( bl_f ); free( bl_x ); free( lg_f );
  xf_run_free( r ); xf_free( x );
  return 0;
}
