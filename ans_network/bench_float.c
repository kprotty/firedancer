/* bench_float - price the ANS backends against the float model they
   were quantized from, on the same machine and the same kernel.

   infer.c already prints a "float (sparse dot)" row, but that row is
   not a like for like comparison: it walks only the set pixels of an
   image (about 19% of them on MNIST) while every ANS backend walks all
   784 features of every neuron.  It is the right reference for
   accuracy and the wrong one for speed.

   Here the float side is what an optimized implementation of the same
   network would actually be: a dense fp32 GEMV over the whole weight
   matrix with AVX-512 FMA, several rows in flight, which is the same
   arithmetic the ANS encoders perform in the log domain.  Both the
   single image (latency) and the batched (throughput) forms are
   measured, because weight traffic is what this size of layer is bound
   by and a batch amortizes it.

   Every row computes the same network; the class agreement column
   checks that. */

#include "model.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

static double
now( void ) {
  struct timespec ts;
  clock_gettime( CLOCK_MONOTONIC, &ts );
  return (double)ts.tv_sec + 1e-9*(double)ts.tv_nsec;
}

static uint pad16( uint n ) { return (n+15u) & ~15u; }

/* Dense fp32 kernels ***************************************************/

/* Rows in flight.  An FMA chain is 4 deep and two issue per cycle, so
   one accumulator per row needs 8 rows to cover the latency. */

#define GEMV_R (8)

/* y[n_row] = w[n_row][stride] . x[stride].  stride is a multiple of 16
   and the columns past n_col hold zeros, so no tail loop is needed. */

static void
fgemv( float const * w,
       uint          n_row,
       uint          stride,
       float const * x,
       float *       y ) {
#if defined(__AVX512F__)
  uint r = 0u;
  for( ; r+GEMV_R<=n_row; r+=GEMV_R ) {
    __m512 a[ GEMV_R ];
    for( uint u=0u; u<GEMV_R; u++ ) a[u] = _mm512_setzero_ps();
    for( uint i=0u; i<stride; i+=16u ) {
      __m512 xv = _mm512_loadu_ps( x+i );
      for( uint u=0u; u<GEMV_R; u++ )
        a[u] = _mm512_fmadd_ps( _mm512_loadu_ps( w + (ulong)(r+u)*stride + i ), xv, a[u] );
    }
    for( uint u=0u; u<GEMV_R; u++ ) y[ r+u ] = _mm512_reduce_add_ps( a[u] );
  }
  for( ; r<n_row; r++ ) {
    __m512 a = _mm512_setzero_ps();
    for( uint i=0u; i<stride; i+=16u )
      a = _mm512_fmadd_ps( _mm512_loadu_ps( w + (ulong)r*stride + i ), _mm512_loadu_ps( x+i ), a );
    y[ r ] = _mm512_reduce_add_ps( a );
  }
#else
  for( uint r=0u; r<n_row; r++ ) {
    float s = 0.0f;
    for( uint i=0u; i<stride; i++ ) s += w[ (ulong)r*stride + i ]*x[i];
    y[ r ] = s;
  }
#endif
}

/* y[n_img][n_row] = w[n_row][stride] . x[n_img][stride], blocked so a
   weight vector loaded once serves GEMM_B images. */

#define GEMM_R (4)
#define GEMM_B (4)

static void
fgemm( float const * w,
       uint          n_row,
       uint          stride,
       float const * x,
       uint          n_img,
       float *       y ) {
#if defined(__AVX512F__)
  for( uint b=0u; b<n_img; b+=GEMM_B ) {
    uint nb = n_img-b<GEMM_B ? n_img-b : GEMM_B;
    for( uint r=0u; r<n_row; r+=GEMM_R ) {
      uint nr = n_row-r<GEMM_R ? n_row-r : GEMM_R;
      __m512 a[ GEMM_R ][ GEMM_B ];
      for( uint u=0u; u<nr; u++ )
        for( uint v=0u; v<nb; v++ ) a[u][v] = _mm512_setzero_ps();
      for( uint i=0u; i<stride; i+=16u ) {
        __m512 xv[ GEMM_B ], wv;
        for( uint v=0u; v<nb; v++ ) xv[v] = _mm512_loadu_ps( x + (ulong)(b+v)*stride + i );
        for( uint u=0u; u<nr; u++ ) {
          wv = _mm512_loadu_ps( w + (ulong)(r+u)*stride + i );
          for( uint v=0u; v<nb; v++ ) a[u][v] = _mm512_fmadd_ps( wv, xv[v], a[u][v] );
        }
      }
      for( uint u=0u; u<nr; u++ )
        for( uint v=0u; v<nb; v++ )
          y[ (ulong)(b+v)*n_row + r+u ] = _mm512_reduce_add_ps( a[u][v] );
    }
  }
#else
  for( uint b=0u; b<n_img; b++ ) fgemv( w, n_row, stride, x + (ulong)b*stride, y + (ulong)b*n_row );
#endif
}

/* The float network ****************************************************/

/* Padded copies of the two weight matrices, plus the dense float input
   the GEMV wants.  This is the layout an optimized float
   implementation would hold the model in; building it is setup, not
   part of any measurement. */

struct fnet {
  uint    n_in,  s_in;    /* s_* are the 16-padded strides */
  uint    n_hid, n_thr;
  uint    n_out, n_feat, s_feat;
  float * w1;             /* [n_hid][s_in]   */
  float * w2;             /* [n_out][s_feat] */
  float * taur;           /* [n_hid][n_thr]  */
  float * bias2;          /* [n_out]         */
  float * hid;            /* [n_hid] scratch */
  float * feat;           /* [s_feat] scratch */
  float * cl;             /* [n_out] scratch */
};

typedef struct fnet fnet_t;

static fnet_t *
fnet_new( ans_model_t const * m ) {
  fnet_t * f  = calloc( 1UL, sizeof(fnet_t) );
  f->n_in     = m->n_in;      f->s_in   = pad16( m->n_in );
  f->n_hid    = m->n_hid;     f->n_thr  = m->n_thr;
  f->n_out    = m->n_out;     f->n_feat = m->n_feat;
  f->s_feat   = pad16( m->n_feat );

  if( m->n_hid ) {
    f->w1 = aligned_alloc( 64UL, sizeof(float)*(ulong)m->n_hid*f->s_in );
    memset( f->w1, 0, sizeof(float)*(ulong)m->n_hid*f->s_in );
    for( uint h=0u; h<m->n_hid; h++ )
      memcpy( f->w1 + (ulong)h*f->s_in, m->w1 + (ulong)h*m->n_in, sizeof(float)*m->n_in );
    f->taur = m->taur;
    f->hid  = malloc( sizeof(float)*m->n_hid );
  }
  f->w2 = aligned_alloc( 64UL, sizeof(float)*(ulong)m->n_out*f->s_feat );
  memset( f->w2, 0, sizeof(float)*(ulong)m->n_out*f->s_feat );
  for( uint c=0u; c<m->n_out; c++ )
    memcpy( f->w2 + (ulong)c*f->s_feat, m->w2 + (ulong)c*m->n_feat, sizeof(float)*m->n_feat );
  f->bias2 = m->bias2;
  f->feat  = aligned_alloc( 64UL, sizeof(float)*f->s_feat );
  f->cl    = malloc( sizeof(float)*m->n_out );
  return f;
}

static void
fnet_free( fnet_t * f ) {
  free( f->w1 ); free( f->w2 ); free( f->hid ); free( f->feat ); free( f->cl );
  free( f );
}

/* One image, dense float, start to finish.  x is the 16-padded 0/1
   image.  Matches ans_model_predict up to summation order. */

static int
fnet_predict( fnet_t *      f,
              float const * x ) {
  if( f->n_hid ) {
    fgemv( f->w1, f->n_hid, f->s_in, x, f->hid );
    memset( f->feat, 0, sizeof(float)*f->s_feat );
    for( uint h=0u; h<f->n_hid; h++ )
      for( uint j=0u; j<f->n_thr; j++ )
        f->feat[ h*f->n_thr + j ] = (float)( -f->hid[h] >= f->taur[ h*f->n_thr + j ] );
  } else {
    memcpy( f->feat, x, sizeof(float)*f->s_feat );
  }
  f->feat[ f->n_feat-1u ] = 1.0f;                 /* the class prior feature */

  fgemv( f->w2, f->n_out, f->s_feat, f->feat, f->cl );

  int   best    = 0;
  float best_cl = 0.0f;
  for( uint c=0u; c<f->n_out; c++ ) {
    float cl = f->bias2[c] - f->cl[c];
    if( !c || cl<best_cl ) { best_cl = cl; best = (int)c; }
  }
  return best;
}

/* Benchmark ************************************************************/

static double
bench( double (*fn)( void * ),
       void *  arg,
       double  secs ) {
  double best = 1e30;
  for( int trial=0; trial<3; trial++ ) {
    double t0 = now(), work = 0.0;
    do work += fn( arg ); while( now()-t0 < secs/3.0 );
    double dt = now()-t0;
    if( dt/work < best ) best = dt/work;   /* seconds per image */
  }
  return best;
}

struct ctx {
  ans_net_t *   net;
  fnet_t *      fl;
  ans_model_t * m;
  ans_data_t *  te;
  float *       xf;      /* [n][s_in] dense float images */
  float *       ybuf;    /* batch output */
  uint          n;
  uint          batch;
  int           backend;
  int *         pred;
};

static double
run_ans( void * a ) {
  struct ctx * c = a;
  for( uint s=0u; s<c->n; s++ )
    c->pred[s] = ans_net_predict( c->net, c->te->bit + (ulong)s*c->te->dim, c->backend );
  return (double)c->n;
}

static double
run_float( void * a ) {
  struct ctx * c = a;
  for( uint s=0u; s<c->n; s++ )
    c->pred[s] = fnet_predict( c->fl, c->xf + (ulong)s*c->fl->s_in );
  return (double)c->n;
}

static double
run_sparse( void * a ) {
  struct ctx * c = a;
  for( uint s=0u; s<c->n; s++ )
    c->pred[s] = ans_model_predict( c->m, c->te->idx + c->te->off[s],
                                    c->te->off[s+1]-c->te->off[s], NULL );
  return (double)c->n;
}

/* Hidden layer only, so the vectorized ANS kernel is priced against
   the float kernel and nothing else. */

static double
run_ans_hid( void * a ) {
  struct ctx * c = a;
  for( uint s=0u; s<c->n; s++ ) {
    switch( c->backend ) {
      case ANS_BACKEND_TANS_BLK:
        ans_tans_blk_scalar( c->net->l1v, c->te->bit + (ulong)s*c->te->dim,
                             c->net->n_in, c->net->n_grp, c->net->cl ); break;
      case ANS_BACKEND_TANS_AVX512:
        ans_tans_blk_avx512( c->net->l1v, c->te->bit + (ulong)s*c->te->dim,
                             c->net->n_in, c->net->n_grp, c->net->cl ); break;
      case ANS_BACKEND_UABS:
        ans_uabs_blk_avx512( c->net->u1[0], c->net->u1[1],
                             c->te->bit + (ulong)s*c->te->dim,
                             c->net->n_in, c->net->n_grp, c->net->cl ); break;
      default:
        ans_tans_blk_avx512_packed( c->net->l1v, c->te->bit + (ulong)s*c->te->dim,
                                    c->net->n_in, c->net->n_grp, c->net->cl ); break;
    }
  }
  return (double)c->n;
}

static double
run_float_hid( void * a ) {
  struct ctx * c = a;
  for( uint s=0u; s<c->n; s++ )
    fgemv( c->fl->w1, c->fl->n_hid, c->fl->s_in, c->xf + (ulong)s*c->fl->s_in, c->fl->hid );
  return (double)c->n;
}

static double
run_float_hid_batch( void * a ) {
  struct ctx * c = a;
  for( uint s=0u; s<c->n; s+=c->batch ) {
    uint nb = c->n-s<c->batch ? c->n-s : c->batch;
    fgemm( c->fl->w1, c->fl->n_hid, c->fl->s_in,
           c->xf + (ulong)s*c->fl->s_in, nb, c->ybuf );
  }
  return (double)c->n;
}

/* Hidden layer, float, over the set pixels only - the sparse shape the
   infer.c reference uses, isolated to the same layer. */

static double
run_sparse_hid( void * a ) {
  struct ctx * c = a;
  for( uint s=0u; s<c->n; s++ ) {
    uint const * idx = c->te->idx + c->te->off[s];
    uint         cnt = c->te->off[s+1]-c->te->off[s];
    for( uint h=0u; h<c->fl->n_hid; h++ )
      c->fl->hid[h] = ans_model_rel( c->m->w1 + (ulong)h*c->m->n_in, idx, cnt );
  }
  return (double)c->n;
}

int
main( int     argc,
      char ** argv ) {
  char const * dir    = "data";
  char const * path   = "model.ans";
  uint         thresh = 128u;
  uint         limit  = 2000u;
  uint         batch  = 64u;
  double       secs   = 1.5;

  for( int i=1; i<argc; i++ ) {
    if     ( !strcmp(argv[i],"--data")   && i+1<argc ) dir    = argv[++i];
    else if( !strcmp(argv[i],"--model")  && i+1<argc ) path   = argv[++i];
    else if( !strcmp(argv[i],"--thresh") && i+1<argc ) thresh = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--limit")  && i+1<argc ) limit  = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--batch")  && i+1<argc ) batch  = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--secs")   && i+1<argc ) secs   = atof(argv[++i]);
    else { fprintf( stderr, "usage: %s [--data DIR] [--model PATH] [--thresh N] "
                            "[--limit N] [--batch N] [--secs S]\n", argv[0] ); return 1; }
  }

  ans_model_t * m = ans_model_load( path );
  if( !m ) { fprintf( stderr, "cannot load %s\n", path ); return 1; }
  if( !m->n_hid ) { fprintf( stderr, "%s has no hidden layer\n", path ); return 1; }

  ans_data_t te;
  int synth;
  if( ans_data_load( &te, dir, 1, thresh, &synth ) ) return 1;
  uint n = limit && limit<te.n ? limit : te.n;

  ans_net_t * net = ans_net_compile( m );
  fnet_t *    fl  = fnet_new( m );

  float * xf = aligned_alloc( 64UL, sizeof(float)*(ulong)n*fl->s_in );
  memset( xf, 0, sizeof(float)*(ulong)n*fl->s_in );
  for( uint s=0u; s<n; s++ )
    for( uint i=0u; i<m->n_in; i++ )
      xf[ (ulong)s*fl->s_in + i ] = (float)te.bit[ (ulong)s*te.dim + i ];

  struct ctx c = { .net = net, .fl = fl, .m = m, .te = &te, .xf = xf, .n = n,
                   .batch = batch, .backend = ANS_BACKEND_TANS_AVX512P,
                   .pred = malloc( sizeof(int)*n ),
                   .ybuf = malloc( sizeof(float)*(ulong)batch*m->n_hid ) };

  ulong mac_hid = (ulong)m->n_hid*m->n_in;
  ulong mac_all = mac_hid + (ulong)m->n_out*m->n_feat;

  printf( "model %s   %u -> %u units x %u thr -> %u   %u samples, %s\n",
          path, m->n_in, m->n_hid, m->n_thr, m->n_out, n, synth ? "synthetic" : "mnist" );
  printf( "hidden layer %lu weights   fp32 %.2f MiB   tans levels %.2f MiB + %lu MiB lut   "
          "uabs %.2f MiB, no lut\n\n",
          mac_hid, (double)(mac_hid*4UL)/1048576.0, (double)mac_hid/1048576.0,
          (ulong)ANS_TANS_LEVEL_CNT*2UL*ANS_TANS_SIZE*sizeof(ushort)/1048576UL,
          (double)(mac_hid*4UL)/1048576.0 );

  /* Correctness: the float network and the reference must agree. */
  int * ref = malloc( sizeof(int)*n );
  for( uint s=0u; s<n; s++ )
    ref[s] = ans_model_predict( m, te.idx + te.off[s], te.off[s+1]-te.off[s], NULL );

  printf( "hidden layer kernel only (%lu weights per image)\n", mac_hid );
  printf( "  %-34s %10s %12s %12s\n", "kernel", "us/image", "Mweight/s", "vs avx512" );

  struct { char const * name; double (*fn)( void * ); int be; } hid[] = {
    { "ans tans avx512, 4 MiB lut",  run_ans_hid,         ANS_BACKEND_TANS_AVX512P },
    { "ans uabs avx512, no table",   run_ans_hid,         ANS_BACKEND_UABS         },
    { "ans tans avx512, 8 MiB lut",  run_ans_hid,         ANS_BACKEND_TANS_AVX512  },
    { "ans tans blocked scalar",     run_ans_hid,         ANS_BACKEND_TANS_BLK     },
    { "float fp32 avx512 gemv",      run_float_hid,       -1                       },
    { "float fp32 avx512 gemm",      run_float_hid_batch, -1                       },
    { "float sparse dot (set px)",   run_sparse_hid,      -1                       },
  };
  double t_ans_hid = 0.0;
  for( uint k=0u; k<sizeof(hid)/sizeof(hid[0]); k++ ) {
    if( hid[k].be>=0 ) c.backend = hid[k].be;
    double t = bench( hid[k].fn, &c, secs );
    if( !k ) t_ans_hid = t;
    char note[ 32 ];
    if( !k ) snprintf( note, sizeof(note), "1.00x" );
    else     snprintf( note, sizeof(note), "%.2fx", t_ans_hid/t );
    char nm[ 48 ];
    if( hid[k].fn==run_float_hid_batch ) snprintf( nm, sizeof(nm), "%s, batch %u", hid[k].name, batch );
    else                                 snprintf( nm, sizeof(nm), "%s", hid[k].name );
    printf( "  %-34s %10.2f %12.0f %12s\n", nm, 1e6*t, 1e-6*(double)mac_hid/t, note );
  }

  printf( "\nwhole network, one image at a time (%lu weights per image)\n", mac_all );
  printf( "  %-34s %10s %12s %12s %10s\n", "path", "us/image", "Mweight/s", "vs avx512", "agree" );

  c.backend = ANS_BACKEND_TANS_AVX512P;
  double t_ans = bench( run_ans, &c, secs );
  ulong  ag_ans = 0UL; for( uint s=0u; s<n; s++ ) ag_ans += (ulong)( c.pred[s]==ref[s] );
  printf( "  %-34s %10.2f %12.0f %12s %9.2f%%\n", "ans tans avx512, 4 MiB lut",
          1e6*t_ans, 1e-6*(double)mac_all/t_ans, "1.00x", 100.0*(double)ag_ans/(double)n );

  c.backend = ANS_BACKEND_UABS;
  double t_u = bench( run_ans, &c, secs );
  ulong  ag_u = 0UL; for( uint s=0u; s<n; s++ ) ag_u += (ulong)( c.pred[s]==ref[s] );
  printf( "  %-34s %10.2f %12.0f %11.2fx %9.2f%%\n", "ans uabs avx512, no table",
          1e6*t_u, 1e-6*(double)mac_all/t_u, t_ans/t_u, 100.0*(double)ag_u/(double)n );

  double t_fl = bench( run_float, &c, secs );
  ulong  ag_fl = 0UL; for( uint s=0u; s<n; s++ ) ag_fl += (ulong)( c.pred[s]==ref[s] );
  printf( "  %-34s %10.2f %12.0f %11.2fx %9.2f%%\n", "float fp32 dense (avx512 gemv)",
          1e6*t_fl, 1e-6*(double)mac_all/t_fl, t_ans/t_fl, 100.0*(double)ag_fl/(double)n );

  double t_sp = bench( run_sparse, &c, secs );
  ulong  ag_sp = 0UL; for( uint s=0u; s<n; s++ ) ag_sp += (ulong)( c.pred[s]==ref[s] );
  printf( "  %-34s %10.2f %12s %11.2fx %9.2f%%\n", "float sparse dot (infer.c ref)",
          1e6*t_sp, "-", t_ans/t_sp, 100.0*(double)ag_sp/(double)n );

  free( ref ); free( c.pred ); free( c.ybuf ); free( xf );
  fnet_free( fl ); ans_net_free( net ); ans_model_free( m ); ans_data_free( &te );
  return 0;
}
