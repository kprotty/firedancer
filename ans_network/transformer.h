#ifndef HEADER_ans_network_transformer_h
#define HEADER_ans_network_transformer_h

/* A binarized transformer whose weight matmuls are ANS encode passes.

   The identity this project is built on is that an ANS encoder's code
   length is a linear unit's pre-activation (see README.md).  A
   transformer is the harder test of it, because only some of a
   transformer is a matmul against static weights.  Three things here
   cannot be an encode pass and are kept in fp32:

     attention scores   QK^T multiplies two activations, and an ANS
                        probability is a static parameter - making one
                        a runtime value costs a division per element
     softmax, LayerNorm they need exp and sqrt
     residuals          signed accumulation, and the stream has to stay
                        real valued for the normalizations to mean
                        anything

   What is left is the six weight matmuls per layer - Q, K, V, O, FF1,
   FF2 - plus the output projection, and those are what this prices.
   ANS needs a *binary* input, so the stream is binarized before each
   of them: every matmul reads a bit vector produced by comparing a
   normalized activation against one learned threshold per unit, which
   is the n_thr=1 case of the hidden layer in model.h.

   The comparison must therefore be the same binarized transformer with
   float matmuls, not an ordinary float transformer, or it measures
   binarization rather than ANS.  Every path here runs identical
   numerics except for how W x is evaluated:

     XF_LIN_F32   fp32 AVX-512 GEMV/GEMM, the kernels from bench_float.c
     XF_LIN_ANS   uABS encode, ans_uabs_blk_avx512
     XF_LIN_I8    int8 VNNI (vpdpbusd), the baseline that matters - a
                  tANS level is one byte, so int8 weights carry the
                  same information at half the bytes of a reciprocal

   Nothing is trained.  Speed does not depend on the weight values, and
   fidelity is the question of whether the integer pass reproduces the
   float pass on the same weights, which needs no task metric. */

#include "ans.h"
#include "ans_uabs.h"

#include <stdio.h>

typedef signed char schar;

/* Shape.  d_model=512, n_head=8, d_ff=2048, n_layer=4 is 12*d_model^2
   weights per layer, so the weights of one layer are 6 MiB in fp32 and
   the matmuls dominate whatever surrounds them. */

struct xf_cfg {
  uint d_model;
  uint n_head;
  uint d_ff;
  uint n_layer;
  uint seq;
  uint vocab;
};

typedef struct xf_cfg xf_cfg_t;

/* One weight matrix, compiled into all three representations.

   The fp32 rows are padded to a multiple of 64 with zeros so no kernel
   needs a tail loop; the padded features hold w=0, which is p=0.5, so
   the uABS planes hold R=2^RSHIFT/0.5 there and each padded step adds
   exactly one bit.  That constant lands in bias the same way on every
   path, so the three agree over the padded width exactly as they do
   over the real one.

   Each path carries the bias its own arithmetic implies, resolved from
   its own rounded weights, for the reason ans_net_compile resolves a
   threshold per quantizer: the input independent term is a sum over
   n_in of softplus2, and rounding the weights moves it. */

struct xf_lin {
  uint     n_out;
  uint     n_in;
  uint     s_in;      /* n_in padded to 64 */
  uint     n_grp;     /* n_out/ANS_BLK */

  float *  w;         /* [n_out][s_in]            fp32 log2 odds */
  float *  bias_f;    /* [n_out] sum_i softplus2( w ) */

  ushort * r[2];      /* [2][n_grp][s_in][ANS_BLK]  uABS planes, one per
                         symbol, blocked across neurons */
  uint *   rt[2];     /* [2][n_out][s_in]           the same reciprocals
                         unblocked and widened to 32 bits, so the
                         across-token kernel broadcasts one from memory */

  schar *  q;         /* [n_out][s_in]            int8 weights */
  float *  qs;        /* [n_out]                  per row scale */
  float *  bias_q;    /* [n_out] sum_i softplus2( qs*q ) */
};

typedef struct xf_lin xf_lin_t;

struct xf_layer {
  xf_lin_t wq, wk, wv, wo, w1, w2;
  float *  tau_qkv;   /* [d_model] binarization thresholds, stream -> QKV */
  float *  tau_o;     /* [d_model] context -> O */
  float *  tau_f1;    /* [d_model] stream -> FF1 */
  float *  tau_f2;    /* [d_ff]    FF1 -> FF2, the nonlinearity */
};

typedef struct xf_layer xf_layer_t;

struct xf {
  xf_cfg_t     cfg;
  uint         d_head;
  xf_layer_t * layer;
  float *      emb;      /* [vocab][d_model] */
  float *      pos;      /* [seq][d_model]   */
  xf_lin_t     wout;     /* [vocab][d_model] */
  float *      tau_out;  /* [d_model] */
  ulong        n_weight; /* weights in all the linear layers */
};

typedef struct xf xf_t;

/* How a linear layer is evaluated.  Everything else is identical. */

#define XF_LIN_F32  (0)
#define XF_LIN_ANS  (1)
#define XF_LIN_I8   (2)
#define XF_LIN_NONE (3)   /* skip the matmul: times the fp32 remainder */
#define XF_LIN_CNT  (4)

extern char const * const xf_lin_name[ XF_LIN_CNT ];

/* Deterministic weights.  splitmix64, and the float conversion is
   exact in fp32, so check_ref.py can reproduce the model bit for bit
   and the fp32 forward pass can be checked against an implementation
   that shares no code with it. */

void  xf_seed( ulong s );
ulong xf_rnd ( void );
float xf_rndw( float lo, float hi );

xf_t * xf_new ( xf_cfg_t const * cfg, ulong seed );
void   xf_free( xf_t * x );

/* Scratch for one forward pass over a whole sequence. */

struct xf_run {
  xf_t const * x;
  float * res;     /* [seq][d_model] residual stream */
  float * nrm;     /* [seq][d_model] */
  float * q, * k, * v, * ctx;   /* [seq][d_model] */
  float * att;     /* [seq] */
  float * ff;      /* [seq][d_ff] */
  float * z;       /* [max(n_out)] one linear's output */
  int *   cl;      /* [max(n_out)] Q8 code lengths */
  float * xf32;    /* [s_in_max] float 0/1 activation */
  uchar * bit;     /* [seq][d_ff] binarized activations */
  float * logit;   /* [seq][vocab] */
  uint *  tok;     /* [seq] */
};

typedef struct xf_run xf_run_t;

xf_run_t * xf_run_new ( xf_t const * x );
void       xf_run_free( xf_run_t * r );

/* xf_forward runs the whole sequence.  mode selects the linear kernel;
   every other operation is fp32 and identical across modes.  calib
   sets each threshold to the mean of what it is about to compare, in
   forward order, which is what an untrained model needs to keep about
   half of every bit vector set. */

void xf_forward( xf_run_t * r, int mode, int calib );

/* xf_set_trace dumps every binarization site's bit vector to f, which
   is how the fp32 reference is checked against check_ref.py. */

void xf_set_trace( FILE * f );

/* xf_bitlog appends every binarization site's bits to buf in forward
   order; xf_bitlog_n reports how many were written.  Differencing two
   runs' logs is the fidelity measurement: a quantized path can only
   diverge from the float one by flipping a bit at a site. */

void  xf_bitlog  ( uchar * buf, ulong cap );
ulong xf_bitlog_n( void );

/* xf_lin_new compiles one weight matrix into all three
   representations from the current RNG stream. */

void xf_lin_new ( xf_lin_t * L, uint n_out, uint n_in );
void xf_lin_free( xf_lin_t * L );

/* Kernels.  Each is checked against a scalar reference in
   bench_transformer --check before anything is timed. */

void xf_fgemv( float const * w, uint n_row, uint stride, float const * x, float * y );
void xf_fgemm( float const * w, uint n_row, uint stride, float const * x, uint n_b, float * y );
void xf_i8gemv( schar const * w, uint n_row, uint stride, uchar const * x, int * y );
void xf_i8gemm( schar const * w, uint n_row, uint stride, uchar const * x, uint n_b, int * y );

/* xf_uabs_tok_avx512 is the batched form of the uABS encode: the
   vector axis is tokens rather than neurons, so one reciprocal is a
   scalar broadcast shared by 16 tokens and the input bit selecting the
   plane becomes a 16 bit mask.  That reads a weight once per 16 tokens
   instead of once per token - a 16x cut in weight traffic, which is
   the only thing a batch can amortize for this kernel.

   msk is [n_in] token masks, bit t of msk[i] being token t's feature
   i.  out is [n_out][16] Q8 code lengths. */

void
xf_uabs_tok_avx512( uint const * rt0,
                    uint const * rt1,
                    ushort const * msk,
                    uint n_in,
                    uint n_out,
                    int * out );

void
xf_uabs_tok_scalar( uint const * rt0,
                    uint const * rt1,
                    ushort const * msk,
                    uint n_in,
                    uint n_out,
                    int * out );

#endif /* HEADER_ans_network_transformer_h */
