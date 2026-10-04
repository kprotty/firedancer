#include "ans.h"

#include <math.h>

ushort ans_log2_frac[ 1<<ANS_LOG2_MANT_BITS ];

float
ans_w_to_p( float w ) {
  /* p = 1/(1+2^-w), evaluated on whichever side avoids overflow */
  if( w>=0.0f ) { float t = exp2f( -w ); return 1.0f/(1.0f+t); }
  float t = exp2f( w );
  return t/(1.0f+t);
}

uint
ans_p_to_freq( float p,
               uint  tot ) {
  long f = lrintf( p*(float)tot );
  if( f<1L            ) f = 1L;
  if( f>(long)tot-1L  ) f = (long)tot-1L;
  return (uint)f;
}

/* rANS ****************************************************************/

void
ans_rans_sym_init( ans_rans_sym_t * sym,
                   uint             start,
                   uint             freq ) {
  sym->x_max     = ( (ANS_RANS_L>>ANS_RANS_M_BITS)<<8 )*freq;
  sym->cmpl_freq = ANS_RANS_M - freq;
  if( freq<2u ) {
    /* q = (x*~0u)>>32 = x-1 for x>=1, and the missing (M-1) folds
       into bias, giving x*M + start exactly */
    sym->rcp_freq  = ~0u;
    sym->rcp_shift = 0u;
    sym->bias      = start + ANS_RANS_M - 1u;
  } else {
    uint shift = 0u;
    while( freq>(1u<<shift) ) shift++;
    sym->rcp_freq  = (uint)( ( ( 1UL<<(shift+31) ) + freq-1UL )/freq );
    sym->rcp_shift = shift-1u;
    sym->bias      = start;
  }
}

void
ans_rans_sym_init_shift( ans_rans_sym_t * sym,
                         uint             start,
                         uint             freq ) {
  sym->x_max     = ( (ANS_RANS_L>>ANS_RANS_M_BITS)<<8 )*freq;
  sym->cmpl_freq = ANS_RANS_M - freq;
  sym->rcp_freq  = 0u;
  sym->rcp_shift = 31u - (uint)__builtin_clz( freq );
  sym->bias      = start;
}

/* tANS ****************************************************************/

static ans_tans_level_t ans_tans_level[ ANS_TANS_LEVEL_CNT ];

float
ans_level_to_w( int level ) {
  return -ANS_W_MAX + 2.0f*ANS_W_MAX*(float)level/(float)(ANS_TANS_LEVEL_CNT-1);
}

int
ans_w_to_level( float w ) {
  long k = lrintf( (w+ANS_W_MAX)*(float)(ANS_TANS_LEVEL_CNT-1)/(2.0f*ANS_W_MAX) );
  if( k<0L                          ) k = 0L;
  if( k>(long)ANS_TANS_LEVEL_CNT-1L ) k = (long)ANS_TANS_LEVEL_CNT-1L;
  return (int)k;
}

uint
ans_level_freq( int level ) {
  return ans_p_to_freq( ans_w_to_p( ans_level_to_w( level ) ), ANS_TANS_SIZE );
}

static inline uint
highbit32( uint v ) {
  return 31u - (uint)__builtin_clz( v );
}

/* tans_build lays out one level's state table: FSE's CTable
   construction specialized to two symbols.  Spread the symbols over
   the table, then sort the slots by symbol to get the next state
   array.

   The spread is Bresenham rather than FSE's odd-stride walk.  FSE's
   stride is tuned for multi symbol alphabets; on the skewed binary
   sources here it scatters the rare symbol unevenly and costs ~0.34%
   in code length, against ~0.08% for an even spread. */

static void
tans_build( ans_tans_level_t * lvl,
            uint               freq_on ) {
  uint const size = ANS_TANS_SIZE;
  uint cnt[ 2 ] = { size-freq_on, freq_on };

  uchar symtab[ ANS_TANS_SIZE ];
  for( uint u=0u; u<size; u++ )
    symtab[ u ] = (uchar)( ( (u+1u)*freq_on )/size > ( u*freq_on )/size );

  uint cum[ 2 ] = { 0u, cnt[0] };
  for( uint u=0u; u<size; u++ ) {
    uint s = symtab[ u ];
    lvl->state[ cum[s]++ ] = (ushort)( size+u );
  }

  uint total = 0u;
  for( uint s=0u; s<2u; s++ ) {
    uint f      = cnt[ s ];
    uint maxbit = ANS_TANS_LOG - ( f>1u ? highbit32( f-1u ) : 0u );
    lvl->sym[ s ].delta_nbit  = (int)( maxbit<<16 ) - (int)( f<<maxbit );
    lvl->sym[ s ].delta_state = (int)total - (int)f;
    total += f;
  }
}

ans_tans_level_t const *
ans_tans_bank( void ) {
  return ans_tans_level;
}

__attribute__((constructor)) static void
ans_init( void ) {
  for( uint m=0u; m<(1u<<ANS_LOG2_MANT_BITS); m++ ) {
    double x = 1.0 + (double)m/(double)(1u<<ANS_LOG2_MANT_BITS);
    ans_log2_frac[ m ] = (ushort)lrint( 256.0*log2( x ) );
  }
  for( int k=0; k<ANS_TANS_LEVEL_CNT; k++ )
    tans_build( &ans_tans_level[ k ], ans_level_freq( k ) );
}
