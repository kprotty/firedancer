#ifndef HEADER_ans_network_ans_h
#define HEADER_ans_network_ans_h

/* ANS primitives for two symbol (Bernoulli) sources.

   A neuron in this network encodes its binary input vector under its
   own product-of-Bernoullis model.  The resulting code length is the
   neuron's pre-activation:

     codelen(x) = sum_i -log2( x_i ? p_i : 1-p_i )
                = -( w.x + b ),  w_i = log2( p_i/(1-p_i) )

   so an encode pass is a linear layer evaluated in the log domain.
   See README.md for the derivation.

   Two backends compute the same quantity:

     rans  32 bit rANS, byte renormalization, full 16 bit frequency
           resolution.  The x/freq in the encode step is replaced by a
           precomputed reciprocal (mulhi+shr).

     tans  table driven ANS.  Weights are quantized to a small set of
           probability levels sharing one state table each.  The encode
           step is an add, two shifts and a 16 bit load: no multiply
           and no divide, the renormalization being a shift by a bit
           count the table supplies.  It pays ~0.08% in code length for
           that, see README.md.

   Neither backend emits the coded bits.  Only the bit count and the
   final state are needed, and together they give the code length
   exactly:  codelen = nbit + log2( state / state_init ). */

#include <stdint.h>

typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;

/* Accumulator shared by both backends.  nbit counts the bits shifted
   out by renormalization, state holds the fractional remainder. */

struct ans_acc {
  uint state;
  uint nbit;
};

typedef struct ans_acc ans_acc_t;

/* Q8 (1/256 bit) fixed point log2, used to turn an accumulator into a
   scalar pre-activation.  ans_log2_q8 is exact to ~0.001 bit. */

#define ANS_LOG2_MANT_BITS (12)

extern ushort ans_log2_frac[ 1<<ANS_LOG2_MANT_BITS ];

static inline uint
ans_log2_q8( uint x ) {
  uint e = 31u - (uint)__builtin_clz( x );
  uint m = (uint)( ( (ulong)x << (31u-e) ) >> (31u-ANS_LOG2_MANT_BITS) )
         & ( (1u<<ANS_LOG2_MANT_BITS)-1u );
  return (e<<8) + ans_log2_frac[ m ];
}

/* rANS backend ********************************************************/

#define ANS_RANS_M_BITS (16)
#define ANS_RANS_M      (1u<<ANS_RANS_M_BITS)
#define ANS_RANS_L      (1u<<23)          /* state lower bound */
#define ANS_RANS_L_Q8   (23u<<8)

/* Encoder side symbol description.  Laid out so the hot path touches
   one 20 byte record.  Follows Giesen's rans_byte.h: the division by
   freq becomes a widening multiply by rcp_freq plus a shift, and for
   freq==1 the reciprocal trick degenerates to (x-1), absorbed by
   bias. */

struct ans_rans_sym {
  uint x_max;      /* renormalize while state >= x_max */
  uint rcp_freq;   /* floor( 2^(shift+31) / freq ) rounded up */
  uint bias;       /* cumulative freq, +M-1 when freq==1 */
  uint cmpl_freq;  /* M - freq */
  uint rcp_shift;
};

typedef struct ans_rans_sym ans_rans_sym_t;

void
ans_rans_sym_init( ans_rans_sym_t * sym,
                   uint             start,
                   uint             freq );

/* ans_rans_sym_init_shift prepares a symbol for ans_rans_put_shift.
   freq must be a power of two. */

void
ans_rans_sym_init_shift( ans_rans_sym_t * sym,
                         uint             start,
                         uint             freq );

/* ans_rans_put encodes one symbol.  Equivalent to

     state = (state/freq)*M + (state%freq) + start

   after renormalizing state down into [ x_max/M*freq, x_max ). */

static inline void
ans_rans_put( ans_acc_t *            acc,
              ans_rans_sym_t const * sym ) {
  uint x    = acc->state;
  uint nbit = acc->nbit;
  while( x>=sym->x_max ) { x >>= 8; nbit += 8u; }
  uint q = (uint)( ( (ulong)x * (ulong)sym->rcp_freq ) >> 32 ) >> sym->rcp_shift;
  acc->state = x + sym->bias + q*sym->cmpl_freq;
  acc->nbit  = nbit;
}

/* ans_rans_put_shift is the same step for a power of two freq, where
   the reciprocal multiply collapses to a shift.  rcp_shift holds
   log2(freq).  Used by the microbenchmark in test.c to price the
   multiply; the model quantizer does not constrain freq to powers of
   two because doing so would cost far more in weight resolution than
   the multiply is worth. */

static inline void
ans_rans_put_shift( ans_acc_t *            acc,
                    ans_rans_sym_t const * sym ) {
  uint x    = acc->state;
  uint nbit = acc->nbit;
  while( x>=sym->x_max ) { x >>= 8; nbit += 8u; }
  uint q = x >> sym->rcp_shift;
  acc->state = x + sym->bias + q*sym->cmpl_freq;
  acc->nbit  = nbit;
}

/* ans_rans_put_div is the textbook step, kept only as the reference
   the reciprocal path is checked and benchmarked against. */

static inline void
ans_rans_put_div( ans_acc_t *            acc,
                  ans_rans_sym_t const * sym,
                  uint                   freq,
                  uint                   start ) {
  uint x    = acc->state;
  uint nbit = acc->nbit;
  while( x>=sym->x_max ) { x >>= 8; nbit += 8u; }
  acc->state = (x/freq)*ANS_RANS_M + (x%freq) + start;
  acc->nbit  = nbit;
}

static inline void
ans_rans_init( ans_acc_t * acc ) {
  acc->state = ANS_RANS_L;
  acc->nbit  = 0u;
}

/* tANS backend ********************************************************/

/* Table log and level count are compile time so the hot loop indexes
   with shifts.  ANS_TANS_LEVEL_CNT levels span [-ANS_W_MAX,+ANS_W_MAX]
   of weight; each level owns one ANS_TANS_SIZE entry state table. */

#ifndef ANS_TANS_LOG
#define ANS_TANS_LOG       (12)
#endif
#define ANS_TANS_SIZE      (1u<<ANS_TANS_LOG)
#ifndef ANS_TANS_LEVEL_CNT
#define ANS_TANS_LEVEL_CNT (256)
#endif

/* Levels are stored one byte per weight, in ans_net_t's l1 and l1v.
   Past 256 they would silently wrap, so the limit is asserted rather
   than left to be discovered.  It costs nothing worth having: the
   code length error is set by the table log, not by the level count,
   and 256 levels already lands within 0.1 points of the float model. */

_Static_assert( ANS_TANS_LEVEL_CNT<=256, "level index must fit a uchar" );
_Static_assert( ANS_TANS_LOG<=14, "tANS state must fit a ushort" );
#define ANS_W_MAX          (8.0f)

struct ans_tans_sym {
  int delta_nbit;   /* (maxbit<<16) - (freq<<maxbit) */
  int delta_state;  /* cumulative freq - freq */
};

typedef struct ans_tans_sym ans_tans_sym_t;

/* One level: the shared state table plus the two symbol transforms.
   Both symbols index the same state table, which is what keeps the
   whole level bank at ANS_TANS_LEVEL_CNT*ANS_TANS_SIZE*2 bytes. */

struct ans_tans_level {
  ushort         state[ ANS_TANS_SIZE ];
  ans_tans_sym_t sym[ 2 ];
};

typedef struct ans_tans_level ans_tans_level_t;

/* ans_tans_bank builds, once, the level table for every quantized
   weight.  Returns a pointer to static storage. */

ans_tans_level_t const * ans_tans_bank( void );

/* ans_w_to_level quantizes a log2 odds weight to a level index, and
   ans_level_to_w gives the weight a level actually represents. */

int   ans_w_to_level( float w );
float ans_level_to_w( int level );

/* ans_level_freq is the table frequency a level actually resolves to,
   which is what the compiled thresholds must be derived from. */

uint  ans_level_freq( int level );

static inline void
ans_tans_put( ans_acc_t *              acc,
              ans_tans_level_t const * lvl,
              uint                     sym ) {
  ans_tans_sym_t const * t = &lvl->sym[ sym ];
  uint x  = acc->state;
  uint nb = (uint)( ( (int)x + t->delta_nbit ) >> 16 );
  acc->state = lvl->state[ (int)( x>>nb ) + t->delta_state ];
  acc->nbit += nb;
}

static inline void
ans_tans_init( ans_acc_t * acc ) {
  acc->state = ANS_TANS_SIZE;
  acc->nbit  = 0u;
}

/* Interleaving.  Every encode step depends on the state the previous
   one produced, so a single accumulator runs at the latency of a
   dependent L2 load (tans) or multiply (rans).  Splitting a neuron's
   features across ANS_LANES independent accumulators breaks that chain
   without changing the result: the stripes encode disjoint feature
   subsets, so their code lengths simply add.

   The output layer deliberately does not interleave.  Keeping one
   accumulator per class is what allows a class to be picked by
   ans_acc_lt, i.e. by comparing raw bit counts, with no logarithm
   anywhere in the decision. */

#define ANS_LANES (8)

/* Readout **************************************************************/

/* ans_acc_codelen_q8 turns an accumulator into a code length in Q8
   bits, relative to the backend's initial state. */

static inline int
ans_rans_codelen_q8( ans_acc_t const * acc ) {
  return (int)( acc->nbit<<8 ) + (int)ans_log2_q8( acc->state ) - (int)ANS_RANS_L_Q8;
}

static inline int
ans_tans_codelen_q8( ans_acc_t const * acc ) {
  return (int)( acc->nbit<<8 ) + (int)ans_log2_q8( acc->state ) - (int)(ANS_TANS_LOG<<8);
}

/* ans_acc_lt orders two accumulators by code length without computing
   any logarithm.  Both accumulators must come from the same backend,
   which keeps every state in one octave, so the raw state breaks ties
   on the bit count exactly.  This is what lets the output layer pick a
   class by "whichever encoder emitted the fewest bits". */

static inline int
ans_acc_lt( ans_acc_t const * a,
            ans_acc_t const * b ) {
  if( a->nbit!=b->nbit ) return a->nbit<b->nbit;
  return a->state<b->state;
}

/* Quantization helpers *************************************************/

/* ans_w_to_p maps a log2 odds weight to P(bit set), and ans_p_to_freq
   normalizes it to a frequency out of tot, clamped so neither symbol
   can vanish. */

float ans_w_to_p  ( float w );
uint  ans_p_to_freq( float p, uint tot );

#endif /* HEADER_ans_network_ans_h */
