/* ans_infer - run a trained model over the test set with each backend
   and report accuracy, agreement and throughput. */

#include "model.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
now( void ) {
  struct timespec ts;
  clock_gettime( CLOCK_MONOTONIC, &ts );
  return (double)ts.tv_sec + 1e-9*(double)ts.tv_nsec;
}

int
main( int     argc,
      char ** argv ) {
  char const * dir    = "data";
  char const * path   = "model.ans";
  uint         thresh = 128u;
  uint         limit  = 0u;
  int          only   = -1;

  for( int i=1; i<argc; i++ ) {
    if     ( !strcmp(argv[i],"--data")   && i+1<argc ) dir    = argv[++i];
    else if( !strcmp(argv[i],"--model")  && i+1<argc ) path   = argv[++i];
    else if( !strcmp(argv[i],"--thresh") && i+1<argc ) thresh = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--limit")  && i+1<argc ) limit  = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--backend")&& i+1<argc ) only   = atoi(argv[++i]);
    else { fprintf( stderr, "usage: %s [--data DIR] [--model PATH] [--thresh N] [--limit N]\n"
                            "          [--backend N]\n",
                    argv[0] ); return 1; }
  }

  ans_model_t * m = ans_model_load( path );
  if( !m ) { fprintf( stderr, "cannot load %s\n", path ); return 1; }

  ans_data_t te;
  int synth;
  if( ans_data_load( &te, dir, 1, thresh, &synth ) ) return 1;
  uint n = limit && limit<te.n ? limit : te.n;

  ans_net_t * net = ans_net_compile( m );

  if( m->n_hid ) printf( "model %s   shape %u -> %u units x %u thresholds -> %u\n",
                         path, m->n_in, m->n_hid, m->n_thr, m->n_out );
  else           printf( "model %s   shape %u -> %u  (linear, no hidden layer)\n",
                         path, m->n_in, m->n_out );
  printf( "test  %s  %u samples\n\n", synth ? "synthetic" : "mnist", n );

  ulong  enc = (ulong)n*( (ulong)m->n_hid*m->n_in + (ulong)m->n_out*m->n_feat );
  int *  ref = malloc( sizeof(int)*n );

  double t0 = now();
  for( uint s=0u; s<n; s++ )
    ref[s] = ans_model_predict( m, te.idx + te.off[s], te.off[s+1]-te.off[s], NULL );
  double t_ref = now()-t0;

  ulong hit_ref = 0UL;
  for( uint s=0u; s<n; s++ ) hit_ref += (ulong)( ref[s]==(int)te.label[s] );

  printf( "%-24s %8s %10s %10s %12s\n", "backend", "accuracy", "vs float", "us/image", "Mencode/s" );
  printf( "%-24s %7.2f%% %10s %10.1f %12s\n", "float (sparse dot)",
          100.0*(double)hit_ref/(double)n, "-", 1e6*t_ref/(double)n, "-" );

  for( int be=0; be<ANS_BACKEND_CNT; be++ ) {
    if( !m->n_hid && be>=ANS_BACKEND_TANS_BLK ) continue;   /* no hidden layer to block */
    if( only>=0 && be!=only ) continue;
    ulong  hit = 0UL, agree = 0UL;
    double t1  = now();
    for( uint s=0u; s<n; s++ ) {
      int p = ans_net_predict( net, te.bit + (ulong)s*te.dim, be );
      hit   += (ulong)( p==(int)te.label[s] );
      agree += (ulong)( p==ref[s] );
    }
    double dt = now()-t1;
    printf( "%-24s %7.2f%% %9.2f%% %10.1f %12.1f\n",
            ans_backend_name[be],
            100.0*(double)hit/(double)n, 100.0*(double)agree/(double)n,
            1e6*dt/(double)n, 1e-6*(double)enc/dt );
  }

  ulong wc = (ulong)m->n_hid*m->n_in + (ulong)m->n_out*m->n_feat;
  ulong lt = (ulong)ANS_TANS_LEVEL_CNT*2UL*ANS_TANS_SIZE;
  printf( "\n%lu weights, %lu encode steps per image.  No float is touched below "
          "ans_net_compile.\n", wc, wc );
  printf( "  weights       tans %lu KiB (1 B/weight)   rans %lu KiB (2 B/weight)\n",
          wc/1024UL, wc*sizeof(ushort)/1024UL );
  printf( "  shared tables tans lanes %lu KiB   rans %lu KiB   "
          "avx512 lut %lu MiB   packed %lu MiB\n",
          (ulong)ANS_TANS_LEVEL_CNT*ANS_TANS_SIZE*sizeof(ushort)/1024UL,
          2UL*ANS_RANS_M*sizeof(ans_rans_sym_t)/1024UL,
          lt*sizeof(uint)/(1024UL*1024UL), lt*sizeof(ushort)/(1024UL*1024UL) );
  if( !ans_has_avx512() ) printf( "  (built without AVX-512: vector rows ran the scalar fallback)\n" );

  free( ref );
  ans_net_free( net );
  ans_model_free( m );
  ans_data_free( &te );
  return 0;
}
