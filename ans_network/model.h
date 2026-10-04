#ifndef HEADER_ans_network_model_h
#define HEADER_ans_network_model_h

#include "ans.h"
#include "ans_avx512.h"
#include "ans_uabs.h"

#include <math.h>

/* Network shape
   ============

   Every unit in this network is an ANS encoder with its own
   product-of-Bernoullis model over its binary inputs, and its
   pre-activation is the code length that encoder produces:

     codelen_u(x) = sum_i -log2( x_i ? p_ui : 1-p_ui )

   Parameters are stored as the log2 odds w_ui = log2( p_ui/(1-p_ui) ),
   because that is the weight of the equivalent linear unit:

     codelen_u(x) = sum_i softplus2( w_ui ) - sum_{i: x_i=1} w_ui
                  = -( w_u.x + b_u ),   softplus2(w) = log2( 1+2^w )

   so the bias is not free, it is pinned to the weights.  That costs
   nothing here: only differences of class code lengths matter, and the
   hidden layer has its own thresholds.

   Hidden layer.  Unit h emits n_thr binary features by comparing its
   code length against learned thresholds, which in the integer machine
   is n_thr integer compares and nothing else:

     t_hj(x) = [ codelen_h(x) >= tau_hj ]

   The threshold is stored relative to the unit's input independent
   term,  tau_hj = sum_i softplus2( w_hi ) + taur_hj,  so that

     t_hj(x) = [ -sum_{i: x_i=1} w_hi >= taur_hj ]

   This is only a reparameterization -- the machine still computes a
   whole code length and compares it against one absolute integer --
   but it matters twice.  Training: that input independent term is
   hundreds of bits and drifts as the weights move, far faster than an
   absolute threshold can track, which starves all but a handful of
   units.  Compiling: the term is resolved from the *quantized*
   weights, so each backend gets the threshold its own arithmetic
   implies.

   n_thr=1 gives a binarized unit; larger n_thr gives a coarse
   staircase.  n_hid=0 drops the layer and the output layer reads the
   input directly, which is a discriminatively trained naive Bayes.

   Output layer.  One encoder per class, over the hidden features plus
   one feature that is always set.  That constant feature contributes
   -log2 p to every class independently, which is exactly a free class
   prior -- the label encoded as a symbol of its own before the data --
   recovering the per class bias that pinning the bias to the weights
   would otherwise cost.

   The prediction is the class whose encoder emitted the fewest bits,
   and the softmax posterior is the normalized 2^-codelen. */

struct ans_model {
  uint    n_in;
  uint    n_hid;    /* 0 = no hidden layer */
  uint    n_thr;    /* thresholds per hidden unit */
  uint    n_out;
  uint    n_feat;   /* features the output layer sees, last is the
                       always-set prior feature */

  float * w1;       /* [n_hid][n_in]   log2 odds */
  float * taur;     /* [n_hid][n_thr]  thresholds, bits, relative to
                       the unit's input independent term */
  float * slope;    /* [n_hid]         STE slope, training only */
  float * w2;       /* [n_out][n_feat] log2 odds */

  float * bias1;    /* [n_hid] sum_i softplus2( w1 ), cached */
  float * bias2;    /* [n_out] sum_j softplus2( w2 ), cached */
};

typedef struct ans_model ans_model_t;

ans_model_t * ans_model_new ( uint n_in, uint n_hid, uint n_thr, uint n_out );
void          ans_model_free( ans_model_t * m );
int           ans_model_save( ans_model_t const * m, char const * path );
ans_model_t * ans_model_load( char const * path );

/* ans_model_refresh recomputes the cached biases.  Must be called
   after any change to w1 or w2. */

void ans_model_refresh( ans_model_t * m );

static inline float
ans_softplus2( float w ) {
  return w>0.0f ? w + log2f( 1.0f+exp2f( -w ) ) : log2f( 1.0f+exp2f( w ) );
}

/* ans_model_rel is the input dependent part of a unit's code length,
   -sum_{i set} w_i.  Adding the unit's cached bias gives the code
   length itself; the hidden layer never needs to. */

static inline float
ans_model_rel( float const * w,
               uint const *  idx,
               uint          cnt ) {
  float s = 0.0f;
  for( uint k=0u; k<cnt; k++ ) s += w[ idx[k] ];
  return -s;
}

/* ans_model_predict is the floating point reference the integer
   backends are checked against.  Writes the per class code length to
   codelen when non NULL. */

int
ans_model_predict( ans_model_t const * m,
                   uint const *        idx,
                   uint                cnt,
                   float *             codelen );

/* Quantized network
   ================= */

struct ans_net {
  uint     n_in;
  uint     n_hid;
  uint     n_thr;
  uint     n_out;
  uint     n_feat;

  int *    tau_q8;  /* [ANS_TAU_SET_CNT][n_hid][n_thr] absolute
                       thresholds in Q8 bits, one set per quantizer:
                       each rounds the weights differently and so
                       implies its own input independent term */

  ushort * f1;      /* [n_hid][n_in]    rANS frequency of the set bit */
  uchar *  l1;      /* [n_hid][n_in]    tANS level */
  uchar *  l1v;     /* [n_grp][n_in][ANS_BLK]  same levels, blocked for
                       the across-neuron backends */
  uint     n_grp;   /* ceil( n_hid / ANS_BLK ) */
  ushort * u1[2];   /* [2][n_grp][n_in][ANS_BLK]  uABS reciprocals,
                       one plane per symbol, blocked like l1v */
  ushort * f2;      /* [n_feat][n_out]  transposed: output units run
                       interleaved over the feature loop */
  uchar *  l2;      /* [n_feat][n_out] */
  ushort * u2[2];   /* [2][n_feat][n_out]  same, for the output layer */

  uchar *  feat;    /* [n_feat] scratch */
  int *    cl;      /* [n_grp*ANS_BLK] scratch */
};

typedef struct ans_net ans_net_t;

/* ans_rans_symtab returns the global table of encoder symbols indexed
   by [frequency][bit], so a weight costs 2 bytes in the model and the
   encode step needs no setup. */

ans_rans_sym_t const * ans_rans_symtab( void );

ans_net_t * ans_net_compile( ans_model_t const * m );
void        ans_net_free   ( ans_net_t * n );

/* Backends.  TANS and RANS stripe one neuron's features across
   ANS_LANES accumulators; the BLK/AVX512/UABS backends instead run
   ANS_BLK neurons side by side with one accumulator each, so they
   report the exact non interleaved code length.  The two structures
   can therefore disagree on a threshold by well under a bit.

   UABS is the one that needs no table: see ans_uabs.h. */

#define ANS_BACKEND_TANS         (0)
#define ANS_BACKEND_RANS         (1)
#define ANS_BACKEND_TANS_BLK     (2)
#define ANS_BACKEND_TANS_AVX512  (3)
#define ANS_BACKEND_TANS_AVX512P (4)
#define ANS_BACKEND_UABS         (5)
#define ANS_BACKEND_CNT          (6)

/* Threshold sets.  The tANS and uABS quantizers round weights their
   own way, and rANS a third; a compiled threshold belongs to one of
   the three, not to one backend. */

#define ANS_TAU_TANS    (0)
#define ANS_TAU_RANS    (1)
#define ANS_TAU_UABS    (2)
#define ANS_TAU_SET_CNT (3)

int ans_tau_set( int backend );

extern char const * const ans_backend_name[ ANS_BACKEND_CNT ];

/* ans_net_predict runs integer-only inference over the dense bit
   vector.  Returns the class whose encoder emitted the fewest bits. */

int
ans_net_predict( ans_net_t *   n,
                 uchar const * bit,
                 int           backend );

#endif /* HEADER_ans_network_model_h */
