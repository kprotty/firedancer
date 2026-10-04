#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint
be32( uchar const * p ) {
  return ((uint)p[0]<<24) | ((uint)p[1]<<16) | ((uint)p[2]<<8) | (uint)p[3];
}

static uchar *
slurp( char const * path,
       ulong *      sz ) {
  FILE * f = fopen( path, "rb" );
  if( !f ) return NULL;
  fseek( f, 0L, SEEK_END );
  long n = ftell( f );
  fseek( f, 0L, SEEK_SET );
  uchar * b = malloc( (ulong)n );
  if( !b || fread( b, 1UL, (ulong)n, f )!=(ulong)n ) { free( b ); fclose( f ); return NULL; }
  fclose( f );
  *sz = (ulong)n;
  return b;
}

/* build_sparse derives the set feature list from the dense view */

static void
build_sparse( ans_data_t * d ) {
  d->off = malloc( sizeof(uint)*(d->n+1UL) );
  ulong tot = 0UL;
  for( ulong k=0UL; k<(ulong)d->n*d->dim; k++ ) tot += d->bit[k];
  d->idx = malloc( sizeof(uint)*(tot?tot:1UL) );
  ulong w = 0UL;
  for( uint s=0u; s<d->n; s++ ) {
    d->off[ s ] = (uint)w;
    uchar const * b = d->bit + (ulong)s*d->dim;
    for( uint i=0u; i<d->dim; i++ ) if( b[i] ) d->idx[ w++ ] = i;
  }
  d->off[ d->n ] = (uint)w;
}

/* synth generates a linearly separable bag-of-bits problem: each class
   owns a random set of features it turns on with high probability. */

static void
synth( ans_data_t * d,
       uint         n,
       int          test ) {
  d->n = n; d->dim = 256u; d->n_class = 10u;
  d->bit   = calloc( (ulong)n*d->dim, 1UL );
  d->label = malloc( n );
  ulong rs = test ? 99UL : 7UL;
  static float proto[ 10 ][ 256 ];
  ulong ps = 4242UL;
  for( uint c=0u; c<10u; c++ ) for( uint i=0u; i<256u; i++ ) {
    ps = ps*6364136223846793005UL + 1442695040888963407UL;
    proto[c][i] = ( (float)(ps>>40) / 16777216.0f )*0.6f + 0.05f;
  }
  for( uint s=0u; s<n; s++ ) {
    rs = rs*6364136223846793005UL + 1442695040888963407UL;
    uint c = (uint)( (rs>>33) % 10UL );
    d->label[ s ] = (uchar)c;
    for( uint i=0u; i<256u; i++ ) {
      rs = rs*6364136223846793005UL + 1442695040888963407UL;
      float u = (float)(rs>>40) / 16777216.0f;
      d->bit[ (ulong)s*256UL + i ] = (uchar)( u<proto[c][i] );
    }
  }
  build_sparse( d );
}

int
ans_data_load( ans_data_t * d,
               char const * dir,
               int          test,
               uint         thresh,
               int *        synthetic ) {
  char ip[ 512 ], lp[ 512 ];
  snprintf( ip, sizeof(ip), "%s/%s-images-idx3-ubyte", dir, test ? "t10k"   : "train" );
  snprintf( lp, sizeof(lp), "%s/%s-labels-idx1-ubyte", dir, test ? "t10k"   : "train" );

  ulong isz, lsz;
  uchar * ib = slurp( ip, &isz );
  uchar * lb = slurp( lp, &lsz );
  if( !ib || !lb ) {
    free( ib ); free( lb );
    *synthetic = 1;
    synth( d, test ? 10000u : 60000u, test );
    return 0;
  }
  *synthetic = 0;

  if( isz<16UL || be32( ib )!=0x803u || lsz<8UL || be32( lb )!=0x801u ) {
    fprintf( stderr, "bad idx magic in %s\n", ip );
    free( ib ); free( lb );
    return -1;
  }
  uint n    = be32( ib+4 );
  uint rows = be32( ib+8 );
  uint cols = be32( ib+12 );
  if( be32( lb+4 )<n || isz<16UL+(ulong)n*rows*cols ) {
    fprintf( stderr, "truncated idx file\n" );
    free( ib ); free( lb );
    return -1;
  }

  d->n = n; d->dim = rows*cols; d->n_class = 10u;
  d->bit   = malloc( (ulong)n*d->dim );
  d->label = malloc( n );
  for( ulong k=0UL; k<(ulong)n*d->dim; k++ ) d->bit[ k ] = (uchar)( ib[ 16UL+k ]>=thresh );
  memcpy( d->label, lb+8, n );
  build_sparse( d );

  free( ib ); free( lb );
  return 0;
}

void
ans_data_free( ans_data_t * d ) {
  free( d->bit ); free( d->idx ); free( d->off ); free( d->label );
  memset( d, 0, sizeof(*d) );
}
