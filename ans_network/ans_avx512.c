#include "ans_avx512.h"

#include <stdlib.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Transition tables ****************************************************/

static uint   * lut_full = NULL;
static ushort * lut_pack = NULL;

static void
lut_build( void ) {
  ans_tans_level_t const * bank = ans_tans_bank();
  ulong n = (ulong)ANS_TANS_LEVEL_CNT*2UL*ANS_TANS_SIZE;
  lut_full = malloc( sizeof(uint)*n );
  lut_pack = ANS_HAS_PACKED_LUT ? malloc( sizeof(ushort)*n ) : NULL;
  for( uint l=0u; l<ANS_TANS_LEVEL_CNT; l++ )
    for( uint s=0u; s<2u; s++ ) {
      ans_tans_sym_t const * t = &bank[l].sym[s];
      ulong o = ( (ulong)( l*2u+s ) )<<ANS_TANS_LOG;
      for( uint x=ANS_TANS_SIZE; x<2u*ANS_TANS_SIZE; x++ ) {
        uint nb = (uint)( ( (int)x + t->delta_nbit )>>16 );
        uint nx = bank[l].state[ (int)( x>>nb ) + t->delta_state ];
        lut_full[ o + x-ANS_TANS_SIZE ] = ( nb<<16 ) | nx;
#if ANS_HAS_PACKED_LUT
        lut_pack[ o + x-ANS_TANS_SIZE ] = (ushort)( ( nb<<ANS_TANS_LOG ) | ( nx-ANS_TANS_SIZE ) );
#endif
      }
    }
}

uint const *
ans_tans_lut( void ) {
  if( !lut_full ) lut_build();
  return lut_full;
}

ushort const *
ans_tans_lut_packed( void ) {
  if( !lut_pack ) lut_build();
  return lut_pack;
}

/* Scalar reference *****************************************************/

/* Same structure and same table as the vector paths, so that comparing
   against it prices the vectorization alone and not the change of
   axis. */

void
ans_tans_blk_scalar( uchar const * lv,
                     uchar const * bit,
                     uint          n_in,
                     uint          n_grp,
                     int *         out ) {
  uint const * lut = ans_tans_lut();
  for( uint g=0u; g<n_grp; g++ ) {
    ans_acc_t     a[ ANS_BLK ];
    uchar const * L = lv + (ulong)g*n_in*ANS_BLK;
    for( uint k=0u; k<ANS_BLK; k++ ) ans_tans_init( &a[k] );
    for( uint i=0u; i<n_in; i++ ) {
      uint b = bit[ i ];
      for( uint k=0u; k<ANS_BLK; k++ ) {
        uint e = lut[ ( (ulong)( L[ (ulong)i*ANS_BLK + k ]*2u + b )<<ANS_TANS_LOG )
                      + a[k].state - ANS_TANS_SIZE ];
        a[k].nbit  += e>>16;
        a[k].state  = e & 0xffffu;
      }
    }
    for( uint k=0u; k<ANS_BLK; k++ ) out[ (ulong)g*ANS_BLK + k ] = ans_tans_codelen_q8( &a[k] );
  }
}

#if defined(__AVX512F__)

int ans_has_avx512( void ) { return 1; }

/* Groups in flight.  One group alone leaves the whole vector waiting
   on the gather that produced its own next index, so the loop runs at
   gather latency; independent groups cover it.  ANS_AVX_UNROLL is a
   build knob because the best value is a property of the machine's
   load queue, not of the algorithm. */

static inline void
readout( __m512i st,
         __m512i nb,
         int *   out ) {
  uint sb[ ANS_BLK ], nn[ ANS_BLK ];
  _mm512_storeu_si512( (void *)sb, st );
  _mm512_storeu_si512( (void *)nn, nb );
  for( uint k=0u; k<ANS_BLK; k++ ) {
    ans_acc_t a = { sb[k], nn[k] };
    out[ k ] = ans_tans_codelen_q8( &a );
  }
}

/* One group's accumulators: 16 neurons' states and bit counts. */

typedef struct { __m512i st, nb; } acc16_t;

static inline acc16_t
acc16_init( void ) {
  acc16_t a = { _mm512_set1_epi32( ANS_TANS_SIZE ), _mm512_setzero_si512() };
  return a;
}

/* One encode step for one group.  bias carries the input bit, already
   broadcast and folded into the index:

     index = (level << TANS_LOG+1) + (bit << TANS_LOG) + state - TANS_SIZE
     entry = (bits emitted << 16) | next state */

static inline acc16_t
step_full( uint const *  lut,
           __m512i       bias,
           uchar const * lv,
           acc16_t       a ) {
  __m512i v = _mm512_cvtepu8_epi32( _mm_loadu_si128( (__m128i const *)lv ) );
  __m512i x = _mm512_add_epi32( _mm512_add_epi32( _mm512_slli_epi32( v, ANS_TANS_LOG+1 ), bias ), a.st );
  __m512i e = _mm512_i32gather_epi32( x, lut, 4 );
  a.nb = _mm512_add_epi32( a.nb, _mm512_srli_epi32( e, 16 ) );
  a.st = _mm512_and_si512( e, _mm512_set1_epi32( 0xffff ) );
  return a;
}

/* Same step against the half size table, where two entries share a
   dword and each lane shifts out the half its own state indexes:

     dword index = (level << TANS_LOG) + (bit << TANS_LOG-1) + offset/2
     entry       = (bits emitted << TANS_LOG) | (next state - TANS_SIZE) */

static inline acc16_t
step_pack( uint const *  lut,
           __m512i       bias,
           uchar const * lv,
           acc16_t       a ) {
  __m512i const sz = _mm512_set1_epi32( ANS_TANS_SIZE );
  __m512i v  = _mm512_cvtepu8_epi32( _mm_loadu_si128( (__m128i const *)lv ) );
  __m512i o  = _mm512_sub_epi32( a.st, sz );              /* 0 .. TANS_SIZE-1 */
  __m512i x  = _mm512_add_epi32( _mm512_add_epi32( _mm512_slli_epi32( v, ANS_TANS_LOG ), bias ),
                                 _mm512_srli_epi32( o, 1 ) );
  __m512i e  = _mm512_i32gather_epi32( x, lut, 4 );
  __m512i sh = _mm512_slli_epi32( _mm512_and_si512( o, _mm512_set1_epi32( 1 ) ), 4 );
  e = _mm512_and_si512( _mm512_srlv_epi32( e, sh ), _mm512_set1_epi32( 0xffff ) );
  a.nb = _mm512_add_epi32( a.nb, _mm512_srli_epi32( e, ANS_TANS_LOG ) );
  a.st = _mm512_add_epi32( _mm512_and_si512( e, _mm512_set1_epi32( ANS_TANS_SIZE-1 ) ), sz );
  return a;
}

/* Encode nu adjacent groups, walking their features together.  Every
   call site passes a constant nu, so the loops over it unroll and the
   accumulators stay in registers. */

static inline void
run_full( uint const *  lut,
          uchar const * lv,
          uchar const * bit,
          uint          n_in,
          uint          nu,
          int *         out ) {
  acc16_t       a[ ANS_AVX_UNROLL ];
  uchar const * L[ ANS_AVX_UNROLL ];
  for( uint u=0u; u<nu; u++ ) {
    a[u] = acc16_init();
    L[u] = lv + (ulong)u*n_in*ANS_BLK;
  }
  for( uint i=0u; i<n_in; i++ ) {
    __m512i bias = _mm512_set1_epi32( (int)( (uint)bit[i]<<ANS_TANS_LOG ) - ANS_TANS_SIZE );
    for( uint u=0u; u<nu; u++ )
      a[u] = step_full( lut, bias, L[u] + (ulong)i*ANS_BLK, a[u] );
  }
  for( uint u=0u; u<nu; u++ ) readout( a[u].st, a[u].nb, out + (ulong)u*ANS_BLK );
}

static inline void
run_pack( uint const *  lut,
          uchar const * lv,
          uchar const * bit,
          uint          n_in,
          uint          nu,
          int *         out ) {
  acc16_t       a[ ANS_AVX_UNROLL ];
  uchar const * L[ ANS_AVX_UNROLL ];
  for( uint u=0u; u<nu; u++ ) {
    a[u] = acc16_init();
    L[u] = lv + (ulong)u*n_in*ANS_BLK;
  }
  for( uint i=0u; i<n_in; i++ ) {
    __m512i bias = _mm512_set1_epi32( (int)( (uint)bit[i]<<(ANS_TANS_LOG-1) ) );
    for( uint u=0u; u<nu; u++ )
      a[u] = step_pack( lut, bias, L[u] + (ulong)i*ANS_BLK, a[u] );
  }
  for( uint u=0u; u<nu; u++ ) readout( a[u].st, a[u].nb, out + (ulong)u*ANS_BLK );
}

/* Whole groups at the build's unroll factor, then any leftover one at
   a time, so n_grp need not be a multiple of it. */

void
ans_tans_blk_avx512( uchar const * lv,
                     uchar const * bit,
                     uint          n_in,
                     uint          n_grp,
                     int *         out ) {
  uint const * lut    = ans_tans_lut();
  ulong        stride = (ulong)n_in*ANS_BLK;
  uint         g      = 0u;
  for( ; g+ANS_AVX_UNROLL<=n_grp; g+=ANS_AVX_UNROLL )
    run_full( lut, lv+(ulong)g*stride, bit, n_in, ANS_AVX_UNROLL, out+(ulong)g*ANS_BLK );
  for( ; g<n_grp; g++ )
    run_full( lut, lv+(ulong)g*stride, bit, n_in, 1u,             out+(ulong)g*ANS_BLK );
}

void
ans_tans_blk_avx512_packed( uchar const * lv,
                            uchar const * bit,
                            uint          n_in,
                            uint          n_grp,
                            int *         out ) {
#if !ANS_HAS_PACKED_LUT
  ans_tans_blk_avx512( lv, bit, n_in, n_grp, out );   /* no room to pack */
#else
  uint const * lut    = (uint const *)ans_tans_lut_packed();
  ulong        stride = (ulong)n_in*ANS_BLK;
  uint         g      = 0u;
  for( ; g+ANS_AVX_UNROLL<=n_grp; g+=ANS_AVX_UNROLL )
    run_pack( lut, lv+(ulong)g*stride, bit, n_in, ANS_AVX_UNROLL, out+(ulong)g*ANS_BLK );
  for( ; g<n_grp; g++ )
    run_pack( lut, lv+(ulong)g*stride, bit, n_in, 1u,             out+(ulong)g*ANS_BLK );
#endif
}

#else /* no AVX-512 */

int ans_has_avx512( void ) { return 0; }

void
ans_tans_blk_avx512( uchar const * lv, uchar const * bit, uint n_in, uint n_grp, int * out ) {
  ans_tans_blk_scalar( lv, bit, n_in, n_grp, out );
}

void
ans_tans_blk_avx512_packed( uchar const * lv, uchar const * bit, uint n_in, uint n_grp, int * out ) {
  ans_tans_blk_scalar( lv, bit, n_in, n_grp, out );
}

#endif
