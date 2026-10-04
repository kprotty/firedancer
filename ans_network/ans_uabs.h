#ifndef HEADER_ans_network_ans_uabs_h
#define HEADER_ans_network_ans_uabs_h

/* uABS - Duda's uniform binary ANS, which is what a two symbol
   alphabet admits and the other two backends do not use.

   With p = P(1) the codec is

     decode  s = ceil((x+1)p) - ceil(xp)
             x = s ? ceil(xp) : x - ceil(xp)
     encode  C(x,1) = floor( x/p )
             C(x,0) = ceil( (x+1)/(1-p) ) - 1

   so the state is *multiplied* by 1/p_s.  There is no state table and
   so no lookup: the encode step is a multiply and a renormalization.
   That matters here more than it would in a compressor.  The tANS
   backend spends one 16 lane gather per weight into a multi megabyte
   transition table, and a gather is the machine's slowest way to touch
   memory - measured on this core it caps at 2.4 G lanes/s from a 4 MiB
   table and 5.4 G even from 1 KiB, against 31 G weights/s for a dense
   fp32 GEMV over the same layer.  Replacing the lookup with arithmetic
   is what closes that gap.

   What is stored.  Both branches divide by a per weight constant, so
   the weight is kept as the reciprocal itself,

     R_s = round( 2^ANS_UABS_RSHIFT / p_s ),   one per symbol,

   a ushort, and the step becomes a multiply by R.  Two planes, one per
   symbol, so the input bit selects a plane and a group's 16
   reciprocals are one contiguous 32 byte load - no gather, and the bit
   stays a broadcast scalar exactly as in ans_avx512.h.

   The relaxation.  Using a rounded reciprocal for both branches rather
   than the exact C(x,0) above gives up decodability: the two symbols'
   images no longer tile the state space exactly, so this encoder is
   not a bijection and cannot be inverted.  Nothing here decodes - only
   the code length is read - and the code length is what the relaxation
   keeps: measured excess is +0.004%, against -0.079% for tANS at
   ANS_TANS_LOG=12.  ans_uabs_exact_put is the true bijective step,
   kept as the reference the relaxation is checked against.

   Accounting.  x stays in [ANS_UABS_L, 2*ANS_UABS_L), so a step is

     q  = x*R                  ANS_UABS_L*2^RSHIFT <= q < 2^32
     sh = bitlen(q) - LOG - 1  = RSHIFT - lzcnt(q)  + (32-LOG-1-RSHIFT)
     x  = q >> sh

   and since log2 R_s = RSHIFT - log2 p_s,

     sum -log2 p = log2( x_end/x_0 ) + sum( sh_i - RSHIFT )

   which is the same (bit count, state) pair the other backends read
   out, so ans_acc_lt still orders classes with no logarithm. */

#include "ans.h"

/* State octave.  LOG=15 is the largest that keeps x*R inside 32 bits
   for a ushort R, and the per step floor error is 2^-LOG, so it is
   also the most accurate. */

#define ANS_UABS_LOG    (15)
#define ANS_UABS_L      (1u<<ANS_UABS_LOG)
#define ANS_UABS_RSHIFT (8)

_Static_assert( ANS_UABS_LOG+16<=32, "x*R must fit 32 bits" );

/* ans_uabs_r quantizes one log2 odds weight to the reciprocal of the
   given symbol's probability.  Clamped to a ushort, which costs
   0.006 of the 8.0 weight range. */

ushort ans_uabs_r( float w, uint sym );

/* ans_uabs_w is the weight a pair of reciprocals actually represents,
   and ans_uabs_bit0 the code length a clear bit costs under r0 - which
   is what a compiled threshold has to be resolved against. */

double ans_uabs_bit0( ushort r0 );

static inline void
ans_uabs_init( ans_acc_t * acc ) {
  acc->state = ANS_UABS_L;
  acc->nbit  = 0u;
}

static inline void
ans_uabs_put( ans_acc_t * acc,
              uint        r ) {
  uint q  = acc->state*r;
  uint sh = (uint)( 31-__builtin_clz( q ) ) - ANS_UABS_LOG;
  acc->state = q>>sh;
  acc->nbit += sh - ANS_UABS_RSHIFT;
}

static inline int
ans_uabs_codelen_q8( ans_acc_t const * acc ) {
  return (int)( acc->nbit<<8 ) + (int)ans_log2_q8( acc->state ) - (int)(ANS_UABS_LOG<<8);
}

/* ans_uabs_exact_put is the bijective step, dividing rather than
   multiplying by a rounded reciprocal.  f is P(sym) as a fraction of
   2^16.  Reference only: ans_test checks the relaxed step against it
   and prices it. */

void ans_uabs_exact_put( ans_acc_t * acc, uint f, uint sym );

/* Across-neuron encode, the same axis and the same blocked layout as
   ans_avx512.h: r0 and r1 are [group][feature][ANS_BLK] reciprocal
   planes, bit is the dense input, out receives one Q8 code length per
   neuron.  Both routines produce identical results. */

void
ans_uabs_blk_scalar( ushort const * r0,
                     ushort const * r1,
                     uchar const *  bit,
                     uint           n_in,
                     uint           n_grp,
                     int *          out );

void
ans_uabs_blk_avx512( ushort const * r0,
                     ushort const * r1,
                     uchar const *  bit,
                     uint           n_in,
                     uint           n_grp,
                     int *          out );

#endif /* HEADER_ans_network_ans_uabs_h */
