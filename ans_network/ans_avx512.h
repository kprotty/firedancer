#ifndef HEADER_ans_network_ans_avx512_h
#define HEADER_ans_network_ans_avx512_h

/* Vectorized tANS encode.

   The scalar path stripes one neuron's features across ANS_LANES
   accumulators to break the dependency between successive encode
   steps.  That is the wrong axis to vectorize on: the levels of one
   neuron are contiguous but its input bits are not reused, and the
   lane code lengths have to be summed back together, which costs a
   little accuracy.

   These routines vectorize across *neurons* instead.  Lane k is
   neuron k, all lanes step through the same feature at the same time,
   and so

     - the input bit is one broadcast scalar per step, not a vector,
     - each lane holds one whole neuron's accumulator, so the result is
       the exact non interleaved code length,
     - the level bytes for the 16 neurons of a group are one aligned
       16 byte load, given the blocked layout below.

   Blocked level layout.  ans_net_compile stores hidden levels as
   [group][feature][16] so that walking features for a group of 16
   neurons is sequential.  Group g, feature i, lane k lives at
   lv[ ((g*n_in) + i)*16 + k ].

   Transition table.  The scalar encoder reaches the next state through
   two dependent loads (symbol transform, then state table).  A gather
   can only chase one, so both are folded into a single table indexed
   by level, symbol and current state:

     lut[ ((level*2 + sym) << ANS_TANS_LOG) + (state - ANS_TANS_SIZE) ]
       = ( emitted bits << 16 ) | next state

   Readout.  Every tANS state sits in [ANS_TANS_SIZE, 2*ANS_TANS_SIZE),
   so its exponent is always ANS_TANS_LOG and it cancels against the
   initial state, leaving

     codelen_q8 = ( nbit << 8 ) + ans_log2_frac[ state & 0xfff ] */

#include "ans.h"

#define ANS_BLK (16)   /* neurons per group */

/* Groups kept in flight by the vector loops.  A build knob: the best
   value is a property of the machine's load queue, not the algorithm. */

#ifndef ANS_AVX_UNROLL
#define ANS_AVX_UNROLL (4)
#endif

/* ans_tans_lut returns the flattened transition table, 8 MiB, built on
   first use.  ans_tans_lut_packed is the same table with both fields
   squeezed into 16 bits, 4 MiB, for the variant that trades a few
   vector ops against half the footprint. */

/* The packed table needs the bit count and the state offset to share
   16 bits, so it exists only while ANS_TANS_LOG leaves room for it. */

#define ANS_TANS_NBIT_BITS (4)
#define ANS_HAS_PACKED_LUT (ANS_TANS_LOG+ANS_TANS_NBIT_BITS<=16)

uint const *   ans_tans_lut       ( void );
ushort const * ans_tans_lut_packed( void );

/* Each routine computes n_grp*ANS_BLK code lengths in Q8 bits.  lv is
   the blocked level array, bit the dense input, out receives one code
   length per neuron.  All three produce identical results. */

void
ans_tans_blk_scalar( uchar const * lv,
                     uchar const * bit,
                     uint          n_in,
                     uint          n_grp,
                     int *         out );

void
ans_tans_blk_avx512( uchar const * lv,
                     uchar const * bit,
                     uint          n_in,
                     uint          n_grp,
                     int *         out );

void
ans_tans_blk_avx512_packed( uchar const * lv,
                            uchar const * bit,
                            uint          n_in,
                            uint          n_grp,
                            int *         out );

/* ans_has_avx512 reports whether the vector paths are real code rather
   than the scalar fallback. */

int ans_has_avx512( void );

#endif /* HEADER_ans_network_ans_avx512_h */
