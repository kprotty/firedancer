#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Float model **********************************************************/

ans_model_t *
ans_model_new( uint n_in,
               uint n_hid,
               uint n_thr,
               uint n_out ) {
  ans_model_t * m = calloc( 1UL, sizeof(ans_model_t) );
  if( !m ) return NULL;
  m->n_in   = n_in;
  m->n_hid  = n_hid;
  m->n_thr  = n_hid ? n_thr : 0u;
  m->n_out  = n_out;
  /* +1 for the always-set feature that carries the class prior */
  m->n_feat = ( n_hid ? n_hid*n_thr : n_in ) + 1u;

  if( n_hid ) {
    m->w1    = calloc( (ulong)n_hid*n_in,  sizeof(float) );
    m->taur  = calloc( (ulong)n_hid*n_thr, sizeof(float) );
    m->slope = calloc( n_hid,              sizeof(float) );
    m->bias1 = calloc( n_hid,              sizeof(float) );
  }
  m->w2    = calloc( (ulong)n_out*m->n_feat, sizeof(float) );
  m->bias2 = calloc( n_out,                  sizeof(float) );
  return m;
}

void
ans_model_free( ans_model_t * m ) {
  if( !m ) return;
  free( m->w1 ); free( m->taur ); free( m->slope ); free( m->w2 );
  free( m->bias1 ); free( m->bias2 );
  free( m );
}

void
ans_model_refresh( ans_model_t * m ) {
  for( uint h=0u; h<m->n_hid; h++ ) {
    float const * w = m->w1 + (ulong)h*m->n_in;
    float s = 0.0f;
    for( uint i=0u; i<m->n_in; i++ ) s += ans_softplus2( w[i] );
    m->bias1[ h ] = s;
  }
  for( uint c=0u; c<m->n_out; c++ ) {
    float const * w = m->w2 + (ulong)c*m->n_feat;
    float s = 0.0f;
    for( uint j=0u; j<m->n_feat; j++ ) s += ans_softplus2( w[j] );
    m->bias2[ c ] = s;
  }
}

int
ans_model_predict( ans_model_t const * m,
                   uint const *        idx,
                   uint                cnt,
                   float *             codelen ) {
  uint nf = m->n_feat;
  uint fidx[ nf ];
  uint fcnt = 0u;

  if( m->n_hid ) {
    for( uint h=0u; h<m->n_hid; h++ ) {
      float r = ans_model_rel( m->w1 + (ulong)h*m->n_in, idx, cnt );
      for( uint j=0u; j<m->n_thr; j++ )
        if( r>=m->taur[ (ulong)h*m->n_thr + j ] ) fidx[ fcnt++ ] = h*m->n_thr + j;
    }
  } else {
    for( uint k=0u; k<cnt; k++ ) fidx[ fcnt++ ] = idx[ k ];
  }
  fidx[ fcnt++ ] = nf-1u;                       /* the class prior feature */

  int   best    = 0;
  float best_cl = 0.0f;
  for( uint c=0u; c<m->n_out; c++ ) {
    float cl = m->bias2[c] + ans_model_rel( m->w2 + (ulong)c*nf, fidx, fcnt );
    if( codelen ) codelen[ c ] = cl;
    if( !c || cl<best_cl ) { best_cl = cl; best = (int)c; }
  }
  return best;
}

/* Serialization ********************************************************/

#define ANS_MAGIC "ANSNET01"

int
ans_model_save( ans_model_t const * m,
                char const *        path ) {
  FILE * f = fopen( path, "wb" );
  if( !f ) return -1;
  uint hdr[ 4 ] = { m->n_in, m->n_hid, m->n_thr, m->n_out };
  int ok = fwrite( ANS_MAGIC, 1UL, 8UL, f )==8UL
        && fwrite( hdr, sizeof(uint), 4UL, f )==4UL;
  if( ok && m->n_hid )
    ok = fwrite( m->w1,  sizeof(float), (ulong)m->n_hid*m->n_in,  f )==(ulong)m->n_hid*m->n_in
      && fwrite( m->taur, sizeof(float), (ulong)m->n_hid*m->n_thr, f )==(ulong)m->n_hid*m->n_thr;
  if( ok )
    ok = fwrite( m->w2, sizeof(float), (ulong)m->n_out*m->n_feat, f )==(ulong)m->n_out*m->n_feat;
  fclose( f );
  return ok ? 0 : -1;
}

ans_model_t *
ans_model_load( char const * path ) {
  FILE * f = fopen( path, "rb" );
  if( !f ) return NULL;
  char magic[ 8 ];
  uint hdr[ 4 ];
  if( fread( magic, 1UL, 8UL, f )!=8UL || memcmp( magic, ANS_MAGIC, 8UL )
   || fread( hdr, sizeof(uint), 4UL, f )!=4UL ) { fclose( f ); return NULL; }

  ans_model_t * m = ans_model_new( hdr[0], hdr[1], hdr[2], hdr[3] );
  if( !m ) { fclose( f ); return NULL; }
  int ok = 1;
  if( m->n_hid )
    ok = fread( m->w1,  sizeof(float), (ulong)m->n_hid*m->n_in,  f )==(ulong)m->n_hid*m->n_in
      && fread( m->taur, sizeof(float), (ulong)m->n_hid*m->n_thr, f )==(ulong)m->n_hid*m->n_thr;
  if( ok )
    ok = fread( m->w2, sizeof(float), (ulong)m->n_out*m->n_feat, f )==(ulong)m->n_out*m->n_feat;
  fclose( f );
  if( !ok ) { ans_model_free( m ); return NULL; }
  ans_model_refresh( m );
  return m;
}

/* Quantized network ****************************************************/

char const * const ans_backend_name[ ANS_BACKEND_CNT ] = {
  "tans (lanes, no mul)",
  "rans (mulhi reciprocal)",
  "tans (blocked scalar)",
  "tans (avx512, 8MiB lut)",
  "tans (avx512, 4MiB lut)",
  "uabs (avx512, no table)"
};

int
ans_tau_set( int backend ) {
  if( backend==ANS_BACKEND_RANS ) return ANS_TAU_RANS;
  if( backend==ANS_BACKEND_UABS ) return ANS_TAU_UABS;
  return ANS_TAU_TANS;
}

static ans_rans_sym_t * rans_symtab = NULL;

ans_rans_sym_t const *
ans_rans_symtab( void ) {
  if( rans_symtab ) return rans_symtab;
  rans_symtab = malloc( sizeof(ans_rans_sym_t)*2UL*ANS_RANS_M );
  for( uint f=0u; f<ANS_RANS_M; f++ ) {
    uint ff = f ? f : 1u;                      /* never let a symbol vanish */
    ans_rans_sym_init( &rans_symtab[ 2UL*f + 1UL ], 0u, ff );
    ans_rans_sym_init( &rans_symtab[ 2UL*f + 0UL ], ff, ANS_RANS_M-ff );
  }
  return rans_symtab;
}

ans_net_t *
ans_net_compile( ans_model_t const * m ) {
  ans_net_t * n = calloc( 1UL, sizeof(ans_net_t) );
  if( !n ) return NULL;
  n->n_in   = m->n_in;
  n->n_hid  = m->n_hid;
  n->n_thr  = m->n_thr;
  n->n_out  = m->n_out;
  n->n_feat = m->n_feat;

  if( m->n_hid ) {
    ulong sz = (ulong)m->n_hid*m->n_in;
    n->f1     = malloc( sizeof(ushort)*sz );
    n->l1     = malloc( sz );
    n->n_grp  = ( m->n_hid + ANS_BLK-1u )/ANS_BLK;
    n->l1v    = calloc( (ulong)n->n_grp*m->n_in*ANS_BLK, 1UL );
    n->cl     = malloc( sizeof(int)*(ulong)n->n_grp*ANS_BLK );
    n->tau_q8 = malloc( sizeof(int)*(ulong)ANS_TAU_SET_CNT*m->n_hid*m->n_thr );
    for( uint s=0u; s<2u; s++ )
      n->u1[s] = calloc( (ulong)n->n_grp*m->n_in*ANS_BLK, sizeof(ushort) );
    for( ulong k=0UL; k<sz; k++ ) {
      float w  = m->w1[ k ];
      n->f1[k] = (ushort)ans_p_to_freq( ans_w_to_p( w ), ANS_RANS_M );
      n->l1[k] = (uchar)ans_w_to_level( w );
    }
    /* Blocked copy of the same levels: group, then feature, then the
       16 neurons of the group, so a group walks features linearly. */
    for( uint h=0u; h<m->n_hid; h++ )
      for( uint i=0u; i<m->n_in; i++ ) {
        ulong o = ( (ulong)( h/ANS_BLK )*m->n_in + i )*ANS_BLK + h%ANS_BLK;
        float w = m->w1[ (ulong)h*m->n_in + i ];
        n->l1v[ o ] = n->l1[ (ulong)h*m->n_in + i ];
        for( uint s=0u; s<2u; s++ ) n->u1[s][ o ] = ans_uabs_r( w, s );
      }

    /* Resolve each threshold against the input independent term the
       quantized weights actually produce.  The two backends round to
       different frequencies, so they do not agree on that term and
       each needs its own absolute threshold. */
    ulong ntau = (ulong)m->n_hid*m->n_thr;
    for( uint h=0u; h<m->n_hid; h++ ) {
      double bt = 0.0, br = 0.0, bu = 0.0;
      for( uint i=0u; i<m->n_in; i++ ) {
        uint ft = ans_level_freq( n->l1[ (ulong)h*m->n_in + i ] );
        uint fr = n->f1[ (ulong)h*m->n_in + i ];
        bt -= log2( (double)( ANS_TANS_SIZE-ft )/(double)ANS_TANS_SIZE );
        br -= log2( (double)( ANS_RANS_M  -fr )/(double)ANS_RANS_M   );
        bu += ans_uabs_bit0( ans_uabs_r( m->w1[ (ulong)h*m->n_in + i ], 0u ) );
      }
      for( uint j=0u; j<m->n_thr; j++ ) {
        double t = m->taur[ (ulong)h*m->n_thr + j ];
        n->tau_q8[ ANS_TAU_TANS*ntau + (ulong)h*m->n_thr + j ] = (int)lrint( 256.0*( bt+t ) );
        n->tau_q8[ ANS_TAU_RANS*ntau + (ulong)h*m->n_thr + j ] = (int)lrint( 256.0*( br+t ) );
        n->tau_q8[ ANS_TAU_UABS*ntau + (ulong)h*m->n_thr + j ] = (int)lrint( 256.0*( bu+t ) );
      }
    }
  }

  /* the output layer runs its classes interleaved over one feature
     loop, so transpose its weights to [feature][class] */
  ulong sz2 = (ulong)m->n_out*m->n_feat;
  n->f2 = malloc( sizeof(ushort)*sz2 );
  n->l2 = malloc( sz2 );
  for( uint s=0u; s<2u; s++ ) n->u2[s] = malloc( sizeof(ushort)*sz2 );
  for( uint c=0u; c<m->n_out; c++ )
    for( uint j=0u; j<m->n_feat; j++ ) {
      float w = m->w2[ (ulong)c*m->n_feat + j ];
      n->f2[ (ulong)j*m->n_out + c ] = (ushort)ans_p_to_freq( ans_w_to_p( w ), ANS_RANS_M );
      n->l2[ (ulong)j*m->n_out + c ] = (uchar)ans_w_to_level( w );
      for( uint s=0u; s<2u; s++ ) n->u2[s][ (ulong)j*m->n_out + c ] = ans_uabs_r( w, s );
    }

  n->feat = malloc( m->n_feat );
  ans_rans_symtab();
  ans_tans_bank();
  if( m->n_hid ) ans_tans_lut();          /* build before anything is timed */
  return n;
}

void
ans_net_free( ans_net_t * n ) {
  if( !n ) return;
  free( n->f1 ); free( n->l1 ); free( n->l1v ); free( n->f2 ); free( n->l2 );
  for( uint s=0u; s<2u; s++ ) { free( n->u1[s] ); free( n->u2[s] ); }
  free( n->tau_q8 ); free( n->feat ); free( n->cl );
  free( n );
}

/* Integer inference ****************************************************/

/* Both helpers stripe the features across ANS_LANES accumulators so the
   encode steps do not form one dependent chain.  The stripes cover
   disjoint features, so their code lengths add. */

static inline int
codelen_tans( uchar const * lv,
              uchar const * bit,
              uint          n ) {
  ans_tans_level_t const * bank = ans_tans_bank();
  ans_acc_t a[ ANS_LANES ];
  for( uint k=0u; k<ANS_LANES; k++ ) ans_tans_init( &a[k] );
  uint i = 0u;
  for( ; i+ANS_LANES<=n; i+=ANS_LANES )
    for( uint k=0u; k<ANS_LANES; k++ )
      ans_tans_put( &a[k], &bank[ lv[i+k] ], bit[i+k] );
  for( ; i<n; i++ ) ans_tans_put( &a[0], &bank[ lv[i] ], bit[i] );
  int cl = 0;
  for( uint k=0u; k<ANS_LANES; k++ ) cl += ans_tans_codelen_q8( &a[k] );
  return cl;
}

static inline int
codelen_rans( ushort const * fq,
              uchar const *  bit,
              uint           n ) {
  ans_rans_sym_t const * tab = ans_rans_symtab();
  ans_acc_t a[ ANS_LANES ];
  for( uint k=0u; k<ANS_LANES; k++ ) ans_rans_init( &a[k] );
  uint i = 0u;
  for( ; i+ANS_LANES<=n; i+=ANS_LANES )
    for( uint k=0u; k<ANS_LANES; k++ )
      ans_rans_put( &a[k], &tab[ 2u*fq[i+k] + bit[i+k] ] );
  for( ; i<n; i++ ) ans_rans_put( &a[0], &tab[ 2u*fq[i] + bit[i] ] );
  int cl = 0;
  for( uint k=0u; k<ANS_LANES; k++ ) cl += ans_rans_codelen_q8( &a[k] );
  return cl;
}

int
ans_net_predict( ans_net_t *   n,
                 uchar const * bit,
                 int           backend ) {
  uint nf    = n->n_feat;
  uint nbase = nf-1u;

  if( n->n_hid ) {
    int const * tau = n->tau_q8 + (ulong)ans_tau_set( backend )*n->n_hid*n->n_thr;
    switch( backend ) {
      case ANS_BACKEND_TANS_BLK:
        ans_tans_blk_scalar        ( n->l1v, bit, n->n_in, n->n_grp, n->cl ); break;
      case ANS_BACKEND_TANS_AVX512:
        ans_tans_blk_avx512        ( n->l1v, bit, n->n_in, n->n_grp, n->cl ); break;
      case ANS_BACKEND_TANS_AVX512P:
        ans_tans_blk_avx512_packed ( n->l1v, bit, n->n_in, n->n_grp, n->cl ); break;
      case ANS_BACKEND_UABS:
        ans_uabs_blk_avx512( n->u1[0], n->u1[1], bit, n->n_in, n->n_grp, n->cl ); break;
      default:
        for( uint h=0u; h<n->n_hid; h++ )
          n->cl[h] = backend==ANS_BACKEND_TANS
                   ? codelen_tans( n->l1 + (ulong)h*n->n_in, bit, n->n_in )
                   : codelen_rans( n->f1 + (ulong)h*n->n_in, bit, n->n_in );
        break;
    }
    for( uint h=0u; h<n->n_hid; h++ )
      for( uint j=0u; j<n->n_thr; j++ )
        n->feat[ h*n->n_thr + j ] = (uchar)( n->cl[h] >= tau[ h*n->n_thr + j ] );
  } else {
    memcpy( n->feat, bit, nbase );
  }
  n->feat[ nbase ] = 1u;                        /* the class prior feature */
  uchar const * feat = n->feat;

  /* One accumulator per class, not interleaved: that is what lets the
     winner be read off the raw bit counts with no logarithm. */
  ans_acc_t a[ n->n_out ];
  if( backend==ANS_BACKEND_UABS ) {
    for( uint c=0u; c<n->n_out; c++ ) ans_uabs_init( &a[c] );
    for( uint j=0u; j<nf; j++ ) {
      ushort const * R = n->u2[ feat[j] ] + (ulong)j*n->n_out;
      for( uint c=0u; c<n->n_out; c++ ) ans_uabs_put( &a[c], R[c] );
    }
  } else if( backend!=ANS_BACKEND_RANS ) {
    ans_tans_level_t const * bank = ans_tans_bank();
    for( uint c=0u; c<n->n_out; c++ ) ans_tans_init( &a[c] );
    for( uint j=0u; j<nf; j++ ) {
      uchar const * lv = n->l2 + (ulong)j*n->n_out;
      uint          b  = feat[ j ];
      for( uint c=0u; c<n->n_out; c++ ) ans_tans_put( &a[c], &bank[ lv[c] ], b );
    }
  } else {
    ans_rans_sym_t const * tab = ans_rans_symtab();
    for( uint c=0u; c<n->n_out; c++ ) ans_rans_init( &a[c] );
    for( uint j=0u; j<nf; j++ ) {
      ushort const * fq = n->f2 + (ulong)j*n->n_out;
      uint           b  = feat[ j ];
      for( uint c=0u; c<n->n_out; c++ ) ans_rans_put( &a[c], &tab[ 2u*fq[c] + b ] );
    }
  }

  int best = 0;
  for( uint c=1u; c<n->n_out; c++ ) if( ans_acc_lt( &a[c], &a[best] ) ) best = (int)c;
  return best;
}
