#include "ans_uabs.h"
#include "ans_avx512.h"   /* ANS_BLK */

#include <math.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Quantization *********************************************************/

ushort
ans_uabs_r( float w,
            uint  sym ) {
  float p = ans_w_to_p( w );
  if( !sym ) p = 1.0f-p;
  double r = ldexp( 1.0, ANS_UABS_RSHIFT )/(double)p;
  if( r<(double)(1u<<ANS_UABS_RSHIFT) ) r = (double)(1u<<ANS_UABS_RSHIFT);
  if( r>65535.0                       ) r = 65535.0;
  return (ushort)lrint( r );
}

double
ans_uabs_bit0( ushort r0 ) {
  return log2( (double)r0 ) - (double)ANS_UABS_RSHIFT;
}

/* Exact codec **********************************************************/

void
ans_uabs_exact_put( ans_acc_t * acc,
                    uint        f,
                    uint        sym ) {
  uint  x  = acc->state;
  uint  nb = acc->nbit;
  ulong M  = 1UL<<16;
  for(;;) {
    ulong y = sym ? ( ( (ulong)x<<16 )/(ulong)f )
                  : ( ( ( ( (ulong)(x+1u) )<<16 ) + ( M-(ulong)f ) - 1UL )/( M-(ulong)f ) - 1UL );
    if( y < 2UL*ANS_UABS_L ) { acc->state = (uint)y; acc->nbit = nb; return; }
    x >>= 1; nb++;
  }
}

/* Scalar reference *****************************************************/

/* Same structure and same planes as the vector path, so comparing
   against it prices the vectorization alone. */

void
ans_uabs_blk_scalar( ushort const * r0,
                     ushort const * r1,
                     uchar const *  bit,
                     uint           n_in,
                     uint           n_grp,
                     int *          out ) {
  for( uint g=0u; g<n_grp; g++ ) {
    ans_acc_t a[ ANS_BLK ];
    for( uint k=0u; k<ANS_BLK; k++ ) ans_uabs_init( &a[k] );
    for( uint i=0u; i<n_in; i++ ) {
      ushort const * R = ( bit[i] ? r1 : r0 ) + ( (ulong)g*n_in + i )*ANS_BLK;
      for( uint k=0u; k<ANS_BLK; k++ ) ans_uabs_put( &a[k], R[k] );
    }
    for( uint k=0u; k<ANS_BLK; k++ ) out[ (ulong)g*ANS_BLK + k ] = ans_uabs_codelen_q8( &a[k] );
  }
}

/* Vector ***************************************************************/

#if defined(__AVX512F__)

/* Groups in flight.  Deeper than the tANS kernel's four: that one is
   waiting on gather latency, this one is issuing six cheap ops per
   step and needs more independent work to fill the ports. */

#ifndef ANS_UABS_UNROLL
#define ANS_UABS_UNROLL (8)
#endif

/* One group's 16 accumulators.  lz accumulates leading zero counts
   rather than shift counts: sum(sh-RSHIFT) = sum(LOG+1-RSHIFT-lzcnt),
   so the per step constant is folded into one subtraction at readout
   and the step keeps to six instructions. */

typedef struct { __m512i st, lz; } uacc_t;

static inline uacc_t
uacc_init( void ) {
  uacc_t a = { _mm512_set1_epi32( ANS_UABS_L ), _mm512_setzero_si512() };
  return a;
}

static inline uacc_t
ustep( ushort const * R,
       uacc_t         a ) {
  __m512i r  = _mm512_cvtepu16_epi32( _mm256_loadu_si256( (__m256i const *)R ) );
  __m512i q  = _mm512_mullo_epi32( a.st, r );
  __m512i lz = _mm512_lzcnt_epi32( q );
  __m512i sh = _mm512_sub_epi32( _mm512_set1_epi32( 32-ANS_UABS_LOG-1 ), lz );
  a.st = _mm512_srlv_epi32( q, sh );
  a.lz = _mm512_add_epi32( a.lz, lz );
  return a;
}

static inline void
ureadout( uacc_t a,
          uint   n_in,
          int *  out ) {
  uint sb[ ANS_BLK ], lz[ ANS_BLK ];
  _mm512_storeu_si512( (void *)sb, a.st );
  _mm512_storeu_si512( (void *)lz, a.lz );
  for( uint k=0u; k<ANS_BLK; k++ ) {
    ans_acc_t c = { sb[k], ( 32u-ANS_UABS_LOG-1u-ANS_UABS_RSHIFT )*n_in - lz[k] };
    out[ k ] = ans_uabs_codelen_q8( &c );
  }
}

static inline void
urun( ushort const * r0,
      ushort const * r1,
      uchar const *  bit,
      uint           n_in,
      uint           nu,
      int *          out ) {
  uacc_t a[ ANS_UABS_UNROLL ];
  for( uint u=0u; u<nu; u++ ) a[u] = uacc_init();
  for( uint i=0u; i<n_in; i++ ) {
    ushort const * pl = ( bit[i] ? r1 : r0 ) + (ulong)i*ANS_BLK;
    for( uint u=0u; u<nu; u++ ) a[u] = ustep( pl + (ulong)u*n_in*ANS_BLK, a[u] );
  }
  for( uint u=0u; u<nu; u++ ) ureadout( a[u], n_in, out + (ulong)u*ANS_BLK );
}

void
ans_uabs_blk_avx512( ushort const * r0,
                     ushort const * r1,
                     uchar const *  bit,
                     uint           n_in,
                     uint           n_grp,
                     int *          out ) {
  ulong stride = (ulong)n_in*ANS_BLK;
  uint  g      = 0u;
  for( ; g+ANS_UABS_UNROLL<=n_grp; g+=ANS_UABS_UNROLL )
    urun( r0+(ulong)g*stride, r1+(ulong)g*stride, bit, n_in, ANS_UABS_UNROLL, out+(ulong)g*ANS_BLK );
  for( ; g<n_grp; g++ )
    urun( r0+(ulong)g*stride, r1+(ulong)g*stride, bit, n_in, 1u,              out+(ulong)g*ANS_BLK );
}

#else

void
ans_uabs_blk_avx512( ushort const * r0, ushort const * r1, uchar const * bit,
                     uint n_in, uint n_grp, int * out ) {
  ans_uabs_blk_scalar( r0, r1, bit, n_in, n_grp, out );
}

#endif
