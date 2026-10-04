#include "transformer.h"
#include "ans_avx512.h"   /* ANS_BLK */
#include "model.h"        /* ans_softplus2 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

char const * const xf_lin_name[ XF_LIN_CNT ] = { "fp32", "uabs", "int8", "no matmul" };

/* Deterministic weights ************************************************/

static ulong xf_s = 0x123456789abcdefUL;

void  xf_seed( ulong s ) { xf_s = s; }

ulong
xf_rnd( void ) {
  ulong z = ( xf_s += 0x9E3779B97F4A7C15UL );
  z = ( z ^ ( z>>30 ) )*0xBF58476D1CE4E5B9UL;
  z = ( z ^ ( z>>27 ) )*0x94D049BB133111EBUL;
  return z ^ ( z>>31 );
}

/* 24 bits of mantissa and a power of two denominator, so the result is
   exact in fp32 and a double implementation gets the same value. */

float
xf_rndw( float lo,
         float hi ) {
  uint u = (uint)( xf_rnd()>>40 );
  return lo + ( hi-lo )*( (float)u/16777216.0f );
}

/* Allocation and compile ***********************************************/

static uint pad64( uint n ) { return (n+63u) & ~63u; }

static void *
xalloc( ulong sz ) {
  sz = ( sz+63UL ) & ~63UL;            /* aligned_alloc wants a multiple */
  void * p = aligned_alloc( 64UL, sz );
  if( !p ) { fprintf( stderr, "oom\n" ); exit( 1 ); }
  memset( p, 0, sz );
  return p;
}

/* One weight matrix in all three representations.  The uABS planes are
   built exactly as ans_net_compile builds u1: r[sym][(g*s_in+i)*BLK+k]
   for neuron g*BLK+k. */

void
xf_lin_new( xf_lin_t * L,
            uint       n_out,
            uint       n_in ) {
  L->n_out = n_out;
  L->n_in  = n_in;
  L->s_in  = pad64( n_in );
  L->n_grp = ( n_out+ANS_BLK-1u )/ANS_BLK;

  ulong np = (ulong)n_out*L->s_in;
  L->w      = xalloc( sizeof(float )*np );
  L->bias_f = xalloc( sizeof(float )*n_out );
  L->q      = xalloc( sizeof(schar )*(ulong)L->n_grp*ANS_BLK*L->s_in );
  L->qs     = xalloc( sizeof(float )*n_out );
  L->bias_q = xalloc( sizeof(float )*n_out );
  for( uint s=0u; s<2u; s++ ) {
    L->r [s] = xalloc( sizeof(ushort)*(ulong)L->n_grp*L->s_in*ANS_BLK );
    L->rt[s] = xalloc( sizeof(uint  )*np );
  }

  for( uint o=0u; o<n_out; o++ )
    for( uint i=0u; i<n_in; i++ ) L->w[ (ulong)o*L->s_in + i ] = xf_rndw( -2.0f, 2.0f );

  for( uint o=0u; o<n_out; o++ ) {
    float mx = 0.0f;
    for( uint i=0u; i<L->s_in; i++ ) {
      float a = fabsf( L->w[ (ulong)o*L->s_in + i ] );
      if( a>mx ) mx = a;
    }
    L->qs[o] = mx>0.0f ? mx/127.0f : 1.0f;
  }

  for( uint o=0u; o<n_out; o++ ) {
    uint  g = o/ANS_BLK, k = o%ANS_BLK;
    float bf = 0.0f, bq = 0.0f;
    for( uint i=0u; i<L->s_in; i++ ) {
      float w = L->w[ (ulong)o*L->s_in + i ];
      ushort r0 = ans_uabs_r( w, 0u ), r1 = ans_uabs_r( w, 1u );
      L->r[0][ ( (ulong)g*L->s_in + i )*ANS_BLK + k ] = r0;
      L->r[1][ ( (ulong)g*L->s_in + i )*ANS_BLK + k ] = r1;
      L->rt[0][ (ulong)o*L->s_in + i ] = r0;
      L->rt[1][ (ulong)o*L->s_in + i ] = r1;

      int qv = (int)lrintf( w/L->qs[o] );
      if( qv> 127 ) qv =  127;
      if( qv<-127 ) qv = -127;
      L->q[ (ulong)o*L->s_in + i ] = (schar)qv;

      bf += ans_softplus2( w );
      bq += ans_softplus2( L->qs[o]*(float)qv );
    }
    L->bias_f[o] = bf;
    L->bias_q[o] = bq;
  }
}

void
xf_lin_free( xf_lin_t * L ) {
  free( L->w ); free( L->bias_f ); free( L->q ); free( L->qs ); free( L->bias_q );
  for( uint s=0u; s<2u; s++ ) { free( L->r[s] ); free( L->rt[s] ); }
}

xf_t *
xf_new( xf_cfg_t const * cfg,
        ulong            seed ) {
  xf_seed( seed );
  xf_t * x = calloc( 1UL, sizeof(xf_t) );
  x->cfg    = *cfg;
  x->d_head = cfg->d_model/cfg->n_head;
  x->layer  = calloc( cfg->n_layer, sizeof(xf_layer_t) );

  x->emb = xalloc( sizeof(float)*(ulong)cfg->vocab*cfg->d_model );
  x->pos = xalloc( sizeof(float)*(ulong)cfg->seq  *cfg->d_model );
  for( ulong i=0UL; i<(ulong)cfg->vocab*cfg->d_model; i++ ) x->emb[i] = xf_rndw( -1.0f, 1.0f );
  for( ulong i=0UL; i<(ulong)cfg->seq  *cfg->d_model; i++ ) x->pos[i] = xf_rndw( -1.0f, 1.0f );

  for( uint l=0u; l<cfg->n_layer; l++ ) {
    xf_layer_t * L = &x->layer[l];
    xf_lin_new( &L->wq, cfg->d_model, cfg->d_model );
    xf_lin_new( &L->wk, cfg->d_model, cfg->d_model );
    xf_lin_new( &L->wv, cfg->d_model, cfg->d_model );
    xf_lin_new( &L->wo, cfg->d_model, cfg->d_model );
    xf_lin_new( &L->w1, cfg->d_ff,    cfg->d_model );
    xf_lin_new( &L->w2, cfg->d_model, cfg->d_ff    );
    L->tau_qkv = xalloc( sizeof(float)*cfg->d_model );
    L->tau_o   = xalloc( sizeof(float)*cfg->d_model );
    L->tau_f1  = xalloc( sizeof(float)*cfg->d_model );
    L->tau_f2  = xalloc( sizeof(float)*cfg->d_ff    );
    x->n_weight += 4UL*(ulong)cfg->d_model*cfg->d_model + 2UL*(ulong)cfg->d_ff*cfg->d_model;
  }
  xf_lin_new( &x->wout, cfg->vocab, cfg->d_model );
  x->tau_out = xalloc( sizeof(float)*cfg->d_model );
  x->n_weight += (ulong)cfg->vocab*cfg->d_model;
  return x;
}

void
xf_free( xf_t * x ) {
  for( uint l=0u; l<x->cfg.n_layer; l++ ) {
    xf_layer_t * L = &x->layer[l];
    xf_lin_free( &L->wq ); xf_lin_free( &L->wk ); xf_lin_free( &L->wv );
    xf_lin_free( &L->wo ); xf_lin_free( &L->w1 ); xf_lin_free( &L->w2 );
    free( L->tau_qkv ); free( L->tau_o ); free( L->tau_f1 ); free( L->tau_f2 );
  }
  xf_lin_free( &x->wout );
  free( x->tau_out ); free( x->layer ); free( x->emb ); free( x->pos );
  free( x );
}

/* fp32 kernels *********************************************************/

/* Verbatim from bench_float.c: eight rows in flight covers a 4 deep
   FMA chain at two per cycle, and the 64-padded stride means no tail. */

#ifndef GEMV_R
#define GEMV_R (8)
#endif

void
xf_fgemv( float const * w,
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

#define GEMM_R (4)
#define GEMM_B (4)

void
xf_fgemm( float const * w,
          uint          n_row,
          uint          stride,
          float const * x,
          uint          n_b,
          float *       y ) {
#if defined(__AVX512F__)
  for( uint b=0u; b<n_b; b+=GEMM_B ) {
    uint nb = n_b-b<GEMM_B ? n_b-b : GEMM_B;
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
  for( uint b=0u; b<n_b; b++ ) xf_fgemv( w, n_row, stride, x + (ulong)b*stride, y + (ulong)b*n_row );
#endif
}

/* int8 VNNI kernels ****************************************************/

/* vpdpbusd takes unsigned activations and signed weights, which is
   exactly the shape here: the activations are binary, so they are the
   u8 side with no quantization error at all, and the weights are the
   s8 side.  64 MACs per instruction against uABS's 16 per six. */

#ifndef I8_R
#define I8_R (8)
#endif

void
xf_i8gemv( schar const * w,
           uint          n_row,
           uint          stride,
           uchar const * x,
           int *         y ) {
#if defined(__AVX512VNNI__)
  uint r = 0u;
  for( ; r+I8_R<=n_row; r+=I8_R ) {
    __m512i a[ I8_R ];
    for( uint u=0u; u<I8_R; u++ ) a[u] = _mm512_setzero_si512();
    for( uint i=0u; i<stride; i+=64u ) {
      __m512i xv = _mm512_loadu_si512( (void const *)( x+i ) );
      for( uint u=0u; u<I8_R; u++ )
        a[u] = _mm512_dpbusd_epi32( a[u], xv,
                 _mm512_loadu_si512( (void const *)( w + (ulong)(r+u)*stride + i ) ) );
    }
    for( uint u=0u; u<I8_R; u++ ) y[ r+u ] = _mm512_reduce_add_epi32( a[u] );
  }
  for( ; r<n_row; r++ ) {
    __m512i a = _mm512_setzero_si512();
    for( uint i=0u; i<stride; i+=64u )
      a = _mm512_dpbusd_epi32( a, _mm512_loadu_si512( (void const *)( x+i ) ),
            _mm512_loadu_si512( (void const *)( w + (ulong)r*stride + i ) ) );
    y[ r ] = _mm512_reduce_add_epi32( a );
  }
#else
  for( uint r=0u; r<n_row; r++ ) {
    int s = 0;
    for( uint i=0u; i<stride; i++ ) s += (int)w[ (ulong)r*stride + i ]*(int)x[i];
    y[ r ] = s;
  }
#endif
}

#define I8M_R (4)
#define I8M_B (4)

void
xf_i8gemm( schar const * w,
           uint          n_row,
           uint          stride,
           uchar const * x,
           uint          n_b,
           int *         y ) {
#if defined(__AVX512VNNI__)
  for( uint b=0u; b<n_b; b+=I8M_B ) {
    uint nb = n_b-b<I8M_B ? n_b-b : I8M_B;
    for( uint r=0u; r<n_row; r+=I8M_R ) {
      uint nr = n_row-r<I8M_R ? n_row-r : I8M_R;
      __m512i a[ I8M_R ][ I8M_B ];
      for( uint u=0u; u<nr; u++ )
        for( uint v=0u; v<nb; v++ ) a[u][v] = _mm512_setzero_si512();
      for( uint i=0u; i<stride; i+=64u ) {
        __m512i xv[ I8M_B ], wv;
        for( uint v=0u; v<nb; v++ )
          xv[v] = _mm512_loadu_si512( (void const *)( x + (ulong)(b+v)*stride + i ) );
        for( uint u=0u; u<nr; u++ ) {
          wv = _mm512_loadu_si512( (void const *)( w + (ulong)(r+u)*stride + i ) );
          for( uint v=0u; v<nb; v++ ) a[u][v] = _mm512_dpbusd_epi32( a[u][v], xv[v], wv );
        }
      }
      for( uint u=0u; u<nr; u++ )
        for( uint v=0u; v<nb; v++ )
          y[ (ulong)(b+v)*n_row + r+u ] = _mm512_reduce_add_epi32( a[u][v] );
    }
  }
#else
  for( uint b=0u; b<n_b; b++ ) xf_i8gemv( w, n_row, stride, x + (ulong)b*stride, y + (ulong)b*n_row );
#endif
}

/* uABS across tokens ***************************************************/

/* The shipped kernel puts 16 neurons in a vector and reloads every
   reciprocal for every token.  Turning the axis around - lane t is
   token t, one neuron at a time - reads a weight once per 16 tokens,
   which is the only thing a batch has to amortize here.  It costs two
   broadcasts and a blend per step instead of one 32 byte load, so it
   trades issue slots for traffic; section 1 of FINDINGS.md is the
   prediction that this is a bad trade. */

#ifndef XF_TOK_CHAIN
#define XF_TOK_CHAIN (8)
#endif

void
xf_uabs_tok_scalar( uint const *   rt0,
                    uint const *   rt1,
                    ushort const * msk,
                    uint           n_in,
                    uint           n_out,
                    int *          out ) {
  for( uint o=0u; o<n_out; o++ ) {
    for( uint t=0u; t<16u; t++ ) {
      ans_acc_t a;
      ans_uabs_init( &a );
      for( uint i=0u; i<n_in; i++ ) {
        uint const * pl = ( msk[i]>>t )&1u ? rt1 : rt0;
        ans_uabs_put( &a, pl[ (ulong)o*n_in + i ] );
      }
      out[ (ulong)o*16u + t ] = ans_uabs_codelen_q8( &a );
    }
  }
}

#if defined(__AVX512F__)

typedef struct { __m512i st, lz; } tacc_t;

static inline tacc_t
tstep( uint const * r0,
       uint const * r1,
       __mmask16    m,
       tacc_t       a ) {
  __m512i r  = _mm512_mask_blend_epi32( m, _mm512_set1_epi32( (int)*r0 ),
                                           _mm512_set1_epi32( (int)*r1 ) );
  __m512i q  = _mm512_mullo_epi32( a.st, r );
  __m512i lz = _mm512_lzcnt_epi32( q );
  __m512i sh = _mm512_sub_epi32( _mm512_set1_epi32( 32-ANS_UABS_LOG-1 ), lz );
  a.st = _mm512_srlv_epi32( q, sh );
  a.lz = _mm512_add_epi32( a.lz, lz );
  return a;
}

static inline void
treadout( tacc_t a,
          uint   n_in,
          int *  out ) {
  uint sb[ 16 ], lz[ 16 ];
  _mm512_storeu_si512( (void *)sb, a.st );
  _mm512_storeu_si512( (void *)lz, a.lz );
  for( uint t=0u; t<16u; t++ ) {
    ans_acc_t c = { sb[t], ( 32u-ANS_UABS_LOG-1u-ANS_UABS_RSHIFT )*n_in - lz[t] };
    out[ t ] = ans_uabs_codelen_q8( &c );
  }
}

static inline void
trun( uint const *   rt0,
      uint const *   rt1,
      ushort const * msk,
      uint           n_in,
      uint           nu,
      int *          out ) {
  tacc_t a[ XF_TOK_CHAIN ];
  for( uint u=0u; u<nu; u++ ) {
    a[u].st = _mm512_set1_epi32( ANS_UABS_L );
    a[u].lz = _mm512_setzero_si512();
  }
  for( uint i=0u; i<n_in; i++ ) {
    __mmask16 m = (__mmask16)msk[i];
    for( uint u=0u; u<nu; u++ )
      a[u] = tstep( rt0 + (ulong)u*n_in + i, rt1 + (ulong)u*n_in + i, m, a[u] );
  }
  for( uint u=0u; u<nu; u++ ) treadout( a[u], n_in, out + (ulong)u*16u );
}

void
xf_uabs_tok_avx512( uint const *   rt0,
                    uint const *   rt1,
                    ushort const * msk,
                    uint           n_in,
                    uint           n_out,
                    int *          out ) {
  uint o = 0u;
  for( ; o+XF_TOK_CHAIN<=n_out; o+=XF_TOK_CHAIN )
    trun( rt0+(ulong)o*n_in, rt1+(ulong)o*n_in, msk, n_in, XF_TOK_CHAIN, out+(ulong)o*16u );
  for( ; o<n_out; o++ )
    trun( rt0+(ulong)o*n_in, rt1+(ulong)o*n_in, msk, n_in, 1u,           out+(ulong)o*16u );
}

#else

void
xf_uabs_tok_avx512( uint const * rt0, uint const * rt1, ushort const * msk,
                    uint n_in, uint n_out, int * out ) {
  xf_uabs_tok_scalar( rt0, rt1, msk, n_in, n_out, out );
}

#endif

/* Forward pass *********************************************************/

/* Everything from here down is identical for all three linear kernels.
   It is fp32 and it is deliberately unoptimized beyond being
   vectorized, because its only job is to be the same on every path and
   to be measurable as a remainder. */

static float
dotf( float const * a,
      float const * b,
      uint          n ) {
#if defined(__AVX512F__)
  __m512 s = _mm512_setzero_ps();
  uint   i = 0u;
  for( ; i+16u<=n; i+=16u ) s = _mm512_fmadd_ps( _mm512_loadu_ps( a+i ), _mm512_loadu_ps( b+i ), s );
  if( i<n ) {
    __mmask16 m = (__mmask16)( ( 1u<<( n-i ) )-1u );
    s = _mm512_fmadd_ps( _mm512_maskz_loadu_ps( m, a+i ), _mm512_maskz_loadu_ps( m, b+i ), s );
  }
  return _mm512_reduce_add_ps( s );
#else
  float s = 0.0f;
  for( uint i=0u; i<n; i++ ) s += a[i]*b[i];
  return s;
#endif
}

static void
axpyf( float *       y,
       float const * x,
       float         c,
       uint          n ) {
#if defined(__AVX512F__)
  __m512 cv = _mm512_set1_ps( c );
  uint   i  = 0u;
  for( ; i+16u<=n; i+=16u )
    _mm512_storeu_ps( y+i, _mm512_fmadd_ps( cv, _mm512_loadu_ps( x+i ), _mm512_loadu_ps( y+i ) ) );
  if( i<n ) {
    __mmask16 m = (__mmask16)( ( 1u<<( n-i ) )-1u );
    _mm512_mask_storeu_ps( y+i, m,
      _mm512_fmadd_ps( cv, _mm512_maskz_loadu_ps( m, x+i ), _mm512_maskz_loadu_ps( m, y+i ) ) );
  }
#else
  for( uint i=0u; i<n; i++ ) y[i] += c*x[i];
#endif
}

static void
layernorm( float *       o,
           float const * v,
           uint          n ) {
  float m = 0.0f;
  for( uint i=0u; i<n; i++ ) m += v[i];
  m /= (float)n;
  float s = 0.0f;
  for( uint i=0u; i<n; i++ ) { float d = v[i]-m; s += d*d; }
  float inv = 1.0f/sqrtf( s/(float)n + 1e-5f );
  for( uint i=0u; i<n; i++ ) o[i] = ( v[i]-m )*inv;
}

/* A threshold is the mean of what it compares, taken over the
   sequence.  The model is untrained, so this is what keeps about half
   of every bit vector set; a trained model would learn it.  It is
   resolved on the fp32 path and then used by all three, so the three
   binarize against the same numbers. */

static void
calib_tau( float *       tau,
           float const * v,
           uint          n_tok,
           uint          n,
           uint          stride ) {
  for( uint i=0u; i<n; i++ ) {
    float s = 0.0f;
    for( uint t=0u; t<n_tok; t++ ) s += v[ (ulong)t*stride + i ];
    tau[i] = s/(float)n_tok;
  }
}

/* Binarize into every path's native layout at once, so no path pays a
   conversion the others do not. */

/* xf_trace makes the fp32 reference auditable from outside: every
   binarization site prints its bit vector, so check_ref.py can be
   compared site by site rather than only on the logits. */

static FILE * xf_trace = NULL;

void xf_set_trace( FILE * f ) { xf_trace = f; }

/* Every binarization site's bits, appended in forward order, so two
   runs can be differenced site by site.  A bit that flips is the only
   way a quantized path can diverge from the float one here: everything
   downstream of a site reads the bits and nothing else. */

static uchar * xf_blog     = NULL;
static ulong   xf_blog_n   = 0UL;
static ulong   xf_blog_cap = 0UL;

void
xf_bitlog( uchar * buf,
           ulong   cap ) {
  xf_blog = buf; xf_blog_cap = cap; xf_blog_n = 0UL;
}

ulong xf_bitlog_n( void ) { return xf_blog_n; }

static void
binz( float const * v,
      float const * tau,
      uint          n,
      uint          s_in,
      uchar *       bit,
      float *       xf ) {
  for( uint i=0u; i<n; i++ ) {
    uchar b = (uchar)( v[i]>=tau[i] );
    bit[i] = b;
    xf [i] = (float)b;
  }
  memset( bit+n, 0, s_in-n );
  memset( xf +n, 0, sizeof(float)*( s_in-n ) );
  if( xf_blog && xf_blog_n+n<=xf_blog_cap ) {
    memcpy( xf_blog+xf_blog_n, bit, n );
    xf_blog_n += n;
  }
  if( xf_trace ) {
    for( uint i=0u; i<n; i++ ) fputc( bit[i] ? '1' : '0', xf_trace );
    fputc( '\n', xf_trace );
  }
}

/* z = w.x - sum_i softplus2( w_i ), the pre-activation of the unit
   whose ANS code length is -z.  The three paths differ only here. */

static void
lin_eval( xf_lin_t const * L,
          uchar const *    bit,
          float const *    xf,
          int              mode,
          float *          scr,
          int *            cl,
          float *          z ) {
  switch( mode ) {

    /* The remainder: everything except W x, so that subtracting it from
       a real path gives the matmul's share of the forward pass without
       having to assume which regime the weights were in. */
    case XF_LIN_NONE:
      memset( z, 0, sizeof(float)*L->n_out );
      break;

    case XF_LIN_ANS:
      ans_uabs_blk_avx512( L->r[0], L->r[1], bit, L->s_in, L->n_grp, cl );
      for( uint o=0u; o<L->n_out; o++ ) z[o] = -(float)cl[o]*(1.0f/256.0f);
      break;

    case XF_LIN_I8:
      xf_i8gemv( L->q, L->n_out, L->s_in, bit, (int *)scr );
      for( uint o=0u; o<L->n_out; o++ )
        z[o] = L->qs[o]*(float)( (int *)scr )[o] - L->bias_q[o];
      break;

    default:
      xf_fgemv( L->w, L->n_out, L->s_in, xf, scr );
      for( uint o=0u; o<L->n_out; o++ ) z[o] = scr[o] - L->bias_f[o];
      break;
  }
}

xf_run_t *
xf_run_new( xf_t const * x ) {
  xf_cfg_t const * c = &x->cfg;
  uint sf = pad64( c->d_ff );
  xf_run_t * r = calloc( 1UL, sizeof(xf_run_t) );
  r->x     = x;
  ulong nd = (ulong)c->seq*c->d_model;
  r->res   = xalloc( sizeof(float)*nd );
  r->nrm   = xalloc( sizeof(float)*nd );
  r->q     = xalloc( sizeof(float)*nd );
  r->k     = xalloc( sizeof(float)*nd );
  r->v     = xalloc( sizeof(float)*nd );
  r->ctx   = xalloc( sizeof(float)*nd );
  r->att   = xalloc( sizeof(float)*c->seq );
  r->ff    = xalloc( sizeof(float)*(ulong)c->seq*c->d_ff );
  r->z     = xalloc( sizeof(float)*sf );
  r->cl    = xalloc( sizeof(int  )*sf );
  r->xf32  = xalloc( sizeof(float)*(ulong)c->seq*sf );
  r->bit   = xalloc( sizeof(uchar)*(ulong)c->seq*sf );
  r->logit = xalloc( sizeof(float)*(ulong)c->seq*c->vocab );
  r->tok   = xalloc( sizeof(uint )*c->seq );
  for( uint t=0u; t<c->seq; t++ ) r->tok[t] = (uint)( xf_rnd()%c->vocab );
  return r;
}

void
xf_run_free( xf_run_t * r ) {
  free( r->res ); free( r->nrm ); free( r->q ); free( r->k ); free( r->v );
  free( r->ctx ); free( r->att ); free( r->ff ); free( r->z ); free( r->cl );
  free( r->xf32 ); free( r->bit ); free( r->logit ); free( r->tok );
  free( r );
}

void
xf_forward( xf_run_t * r,
            int        mode,
            int        calib ) {
  xf_t const * x  = r->x;
  uint d  = x->cfg.d_model, S = x->cfg.seq, F = x->cfg.d_ff;
  uint dh = x->d_head,      nh = x->cfg.n_head;
  uint sd = pad64( d ),     sf = pad64( F );
  float iscale = 1.0f/sqrtf( (float)dh );

  for( uint t=0u; t<S; t++ )
    for( uint i=0u; i<d; i++ )
      r->res[ (ulong)t*d + i ] = x->emb[ (ulong)r->tok[t]*d + i ] + x->pos[ (ulong)t*d + i ];

  for( uint l=0u; l<x->cfg.n_layer; l++ ) {
    xf_layer_t const * L = &x->layer[l];

    /* stream -> Q, K, V */

    for( uint t=0u; t<S; t++ ) layernorm( r->nrm + (ulong)t*d, r->res + (ulong)t*d, d );
    if( calib ) calib_tau( L->tau_qkv, r->nrm, S, d, d );
    for( uint t=0u; t<S; t++ )
      binz( r->nrm + (ulong)t*d, L->tau_qkv, d, sd, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf );
    for( uint t=0u; t<S; t++ ) {
      uchar const * b = r->bit  + (ulong)t*sf;
      float const * f = r->xf32 + (ulong)t*sf;
      lin_eval( &L->wq, b, f, mode, r->z, r->cl, r->q + (ulong)t*d );
      lin_eval( &L->wk, b, f, mode, r->z, r->cl, r->k + (ulong)t*d );
      lin_eval( &L->wv, b, f, mode, r->z, r->cl, r->v + (ulong)t*d );
    }

    /* attention.  Q and K are normalized per head before the score:
       a code length is hundreds of bits, and a dot product of two of
       them saturates any softmax.  It is the same operation on every
       path. */

    for( uint t=0u; t<S; t++ )
      for( uint h=0u; h<nh; h++ ) {
        layernorm( r->q + (ulong)t*d + h*dh, r->q + (ulong)t*d + h*dh, dh );
        layernorm( r->k + (ulong)t*d + h*dh, r->k + (ulong)t*d + h*dh, dh );
      }

    memset( r->ctx, 0, sizeof(float)*(ulong)S*d );
    for( uint h=0u; h<nh; h++ )
      for( uint t=0u; t<S; t++ ) {
        float mx = -1e30f;
        for( uint s=0u; s<=t; s++ ) {
          float sc = iscale*dotf( r->q + (ulong)t*d + h*dh, r->k + (ulong)s*d + h*dh, dh );
          r->att[s] = sc;
          if( sc>mx ) mx = sc;
        }
        float sum = 0.0f;
        for( uint s=0u; s<=t; s++ ) { r->att[s] = expf( r->att[s]-mx ); sum += r->att[s]; }
        float inv = 1.0f/sum;
        for( uint s=0u; s<=t; s++ )
          axpyf( r->ctx + (ulong)t*d + h*dh, r->v + (ulong)s*d + h*dh, r->att[s]*inv, dh );
      }

    /* context -> O, residual */

    if( calib ) calib_tau( L->tau_o, r->ctx, S, d, d );
    for( uint t=0u; t<S; t++ )
      binz( r->ctx + (ulong)t*d, L->tau_o, d, sd, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf );
    for( uint t=0u; t<S; t++ ) {
      lin_eval( &L->wo, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf, mode, r->z, r->cl, r->z+sf/2u );
      for( uint i=0u; i<d; i++ ) r->res[ (ulong)t*d + i ] += r->z[ sf/2u + i ];
    }

    /* stream -> FF1 -> FF2, residual.  The FF1 binarization is the
       nonlinearity: it is where a ReLU would be. */

    for( uint t=0u; t<S; t++ ) layernorm( r->nrm + (ulong)t*d, r->res + (ulong)t*d, d );
    if( calib ) calib_tau( L->tau_f1, r->nrm, S, d, d );
    for( uint t=0u; t<S; t++ )
      binz( r->nrm + (ulong)t*d, L->tau_f1, d, sd, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf );
    for( uint t=0u; t<S; t++ )
      lin_eval( &L->w1, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf, mode, r->z, r->cl,
                r->ff + (ulong)t*F );

    if( calib ) calib_tau( L->tau_f2, r->ff, S, F, F );
    for( uint t=0u; t<S; t++ )
      binz( r->ff + (ulong)t*F, L->tau_f2, F, sf, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf );
    for( uint t=0u; t<S; t++ ) {
      lin_eval( &L->w2, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf, mode, r->z, r->cl, r->z+sf/2u );
      for( uint i=0u; i<d; i++ ) r->res[ (ulong)t*d + i ] += r->z[ sf/2u + i ];
    }
  }

  for( uint t=0u; t<S; t++ ) layernorm( r->nrm + (ulong)t*d, r->res + (ulong)t*d, d );
  if( calib ) calib_tau( x->tau_out, r->nrm, S, d, d );
  for( uint t=0u; t<S; t++ )
    binz( r->nrm + (ulong)t*d, x->tau_out, d, sd, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf );
  for( uint t=0u; t<S; t++ )
    lin_eval( &x->wout, r->bit + (ulong)t*sf, r->xf32 + (ulong)t*sf, mode, r->z, r->cl,
              r->logit + (ulong)t*x->cfg.vocab );
}
