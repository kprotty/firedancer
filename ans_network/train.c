/* ans_train - train an ANS coded network by ordinary backprop.

   The forward pass is exactly the quantity the integer machine
   computes, so there is no separate "training model": a unit's output
   is the code length its encoder produces,

     codelen_u(x) = sum_i softplus2( w_ui ) - sum_{i: x_i=1} w_ui

   whose gradient is the textbook logistic one,

     d codelen_u / d w_ui = sigma( w_ui ) - x_i

   The second term is sparse in the input and the first does not depend
   on the sample, so a minibatch costs O( set bits ) per unit rather
   than O( inputs ):

     grad w_ui = sigma( w_ui ) * sum_b d_bu  -  sum_b d_bu x_bi

   The hidden layer is cheaper still.  Its thresholds are stored
   relative to sum_i softplus2( w_hi ) (see model.h), so that term
   cancels out of every hidden derivative and the dense half of the
   gradient disappears with it.

   Thresholds are stepped, so they get a straight through estimator:
   hard compare forward, logistic slope backward. */

#include "model.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ulong rng_s = 1UL;

static inline float
rng_f( void ) {
  rng_s = rng_s*6364136223846793005UL + 1442695040888963407UL;
  return (float)(rng_s>>40) / 16777216.0f;
}

/* Adam ****************************************************************/

struct adam { float * m; float * v; ulong n; };

static void
adam_init( struct adam * a,
           ulong         n ) {
  a->n = n;
  a->m = calloc( n, sizeof(float) );
  a->v = calloc( n, sizeof(float) );
}

static void
adam_step( struct adam * a,
           float *       w,
           float const * g,
           float         lr,
           float         b1t,
           float         b2t,
           float         clip ) {
  float const b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  for( ulong k=0UL; k<a->n; k++ ) {
    a->m[k] = b1*a->m[k] + (1.0f-b1)*g[k];
    a->v[k] = b2*a->v[k] + (1.0f-b2)*g[k]*g[k];
    float mh = a->m[k]/(1.0f-b1t);
    float vh = a->v[k]/(1.0f-b2t);
    float x  = w[k] - lr*mh/(sqrtf(vh)+eps);
    if( clip>0.0f ) { if( x>clip ) x = clip; if( x<-clip ) x = -clip; }
    w[k] = x;
  }
}

static double
evaluate( ans_model_t const * m,
          ans_data_t const *  d ) {
  ulong hit = 0UL;
  #pragma omp parallel for reduction(+:hit) schedule(static)
  for( uint s=0u; s<d->n; s++ ) {
    int p = ans_model_predict( m, d->idx + d->off[s], d->off[s+1]-d->off[s], NULL );
    hit += (ulong)( p==(int)d->label[s] );
  }
  return (double)hit/(double)d->n;
}

int
main( int     argc,
      char ** argv ) {
  char const * dir     = "data";
  char const * out     = "model.ans";
  uint         n_hid   = 256u;
  uint         n_thr   = 1u;
  uint         epochs  = 20u;
  uint         batch   = 128u;
  uint         thresh  = 128u;
  float        lr      = 0.02f;
  int          verbose = 0;

  for( int i=1; i<argc; i++ ) {
    if     ( !strcmp(argv[i],"--data")    && i+1<argc ) dir    = argv[++i];
    else if( !strcmp(argv[i],"--out")     && i+1<argc ) out    = argv[++i];
    else if( !strcmp(argv[i],"--hidden")  && i+1<argc ) n_hid  = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--thr")     && i+1<argc ) n_thr  = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--epochs")  && i+1<argc ) epochs = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--batch")   && i+1<argc ) batch  = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--lr")      && i+1<argc ) lr     = (float)atof(argv[++i]);
    else if( !strcmp(argv[i],"--thresh")  && i+1<argc ) thresh = (uint)atoi(argv[++i]);
    else if( !strcmp(argv[i],"--seed")    && i+1<argc ) rng_s  = (ulong)atol(argv[++i]);
    else if( !strcmp(argv[i],"--verbose")             ) verbose = 1;
    else { fprintf( stderr, "usage: %s [--data DIR] [--out PATH] [--hidden N] [--thr N]\n"
                            "          [--epochs N] [--batch N] [--lr F] [--thresh N]\n"
                            "          [--seed N] [--verbose]\n", argv[0] ); return 1; }
  }
  if( n_hid==0u ) n_thr = 0u;

  ans_data_t tr, te;
  int synth_tr, synth_te;
  if( ans_data_load( &tr, dir, 0, thresh, &synth_tr )
   || ans_data_load( &te, dir, 1, thresh, &synth_te ) ) return 1;
  printf( "%s  train %u  test %u  dim %u  classes %u\n",
          synth_tr ? "synthetic (no idx files found)" : "mnist",
          tr.n, te.n, tr.dim, tr.n_class );

  ans_model_t * m  = ans_model_new( tr.dim, n_hid, n_thr, tr.n_class );
  uint          nf = m->n_feat;
  if( n_hid ) printf( "shape %u -> %u units x %u thresholds -> %u  (%lu + %lu weights)\n",
                      m->n_in, n_hid, n_thr, m->n_out,
                      (ulong)n_hid*m->n_in, (ulong)m->n_out*nf );
  else        printf( "shape %u -> %u  linear  (%lu weights)\n",
                      m->n_in, m->n_out, (ulong)m->n_out*nf );

  for( ulong k=0UL; k<(ulong)n_hid*m->n_in; k++ ) m->w1[k] = rng_f()-0.5f;
  for( ulong k=0UL; k<(ulong)m->n_out*nf;   k++ ) m->w2[k] = (rng_f()-0.5f)*0.1f;
  ans_model_refresh( m );

  float * p2   = malloc( sizeof(float)*(ulong)m->n_out*nf );
  float * rel  = n_hid ? malloc( sizeof(float)*(ulong)batch*n_hid ) : NULL;
  float * cl2  = malloc( sizeof(float)*(ulong)batch*m->n_out );
  float * d1   = n_hid ? malloc( sizeof(float)*(ulong)batch*n_hid ) : NULL;
  float * d2   = malloc( sizeof(float)*(ulong)batch*m->n_out );
  float * dt   = malloc( sizeof(float)*(ulong)batch*nf );
  uchar * tfe  = malloc( (ulong)batch*nf );
  float * gw1  = n_hid ? malloc( sizeof(float)*(ulong)n_hid*m->n_in ) : NULL;
  float * gw2  = malloc( sizeof(float)*(ulong)m->n_out*nf );
  float * gtau = n_hid ? malloc( sizeof(float)*(ulong)n_hid*n_thr ) : NULL;
  float * gsl  = n_hid ? malloc( sizeof(float)*n_hid ) : NULL;
  uint  * perm = malloc( sizeof(uint)*tr.n );
  for( uint s=0u; s<tr.n; s++ ) perm[s] = s;

  /* Threshold init.  The scale of a unit's output is set by the data,
     not by the weight init, so place each threshold at a quantile of
     what the unit actually emits and take the straight through slope
     from the spread. */
  if( n_hid ) {
    uint    nw = 1024u<tr.n ? 1024u : tr.n;
    float * rw = malloc( sizeof(float)*(ulong)nw*n_hid );
    #pragma omp parallel for schedule(static)
    for( uint h=0u; h<n_hid; h++ )
      for( uint b=0u; b<nw; b++ )
        rw[ (ulong)b*n_hid + h ] = ans_model_rel( m->w1 + (ulong)h*m->n_in,
                                                  tr.idx+tr.off[b], tr.off[b+1]-tr.off[b] );
    for( uint h=0u; h<n_hid; h++ ) {
      float mu = 0.0f, sd = 0.0f;
      for( uint b=0u; b<nw; b++ ) mu += rw[ (ulong)b*n_hid + h ];
      mu /= (float)nw;
      for( uint b=0u; b<nw; b++ ) { float e = rw[(ulong)b*n_hid+h]-mu; sd += e*e; }
      sd = sqrtf( sd/(float)nw ) + 1e-6f;
      for( uint j=0u; j<n_thr; j++ )
        m->taur[ (ulong)h*n_thr + j ] = mu + sd*( 2.0f*((float)(j+1u)/(float)(n_thr+1u)) - 1.0f );
      m->slope[ h ] = 4.0f/sd;
    }
    free( rw );
  }

  struct adam ad_w1, ad_w2, ad_tau, ad_sl;
  if( n_hid ) { adam_init( &ad_w1,  (ulong)n_hid*m->n_in );
                adam_init( &ad_tau, (ulong)n_hid*n_thr );
                adam_init( &ad_sl,  n_hid ); }
  adam_init( &ad_w2, (ulong)m->n_out*nf );
  float b1t = 1.0f, b2t = 1.0f;

  double best = 0.0;
  for( uint ep=0u; ep<epochs; ep++ ) {
    for( uint s=tr.n; s>1u; s-- ) {
      uint j = (uint)( (ulong)( rng_f()*(float)s ) % s );
      uint t = perm[s-1u]; perm[s-1u] = perm[j]; perm[j] = t;
    }
    double loss = 0.0; ulong hit = 0UL, seen = 0UL;

    for( uint b0=0u; b0+batch<=tr.n; b0+=batch ) {
      uint const * pb = perm + b0;

      /* ---- forward, hidden ---- */
      memset( tfe, 0, (ulong)batch*nf );
      if( n_hid ) {
        #pragma omp parallel for schedule(static)
        for( uint h=0u; h<n_hid; h++ ) {
          float const * w = m->w1 + (ulong)h*m->n_in;
          for( uint b=0u; b<batch; b++ ) {
            uint s = pb[b];
            rel[ (ulong)b*n_hid + h ] = ans_model_rel( w, tr.idx+tr.off[s], tr.off[s+1]-tr.off[s] );
          }
        }
        for( uint b=0u; b<batch; b++ )
          for( uint h=0u; h<n_hid; h++ )
            for( uint j=0u; j<n_thr; j++ )
              tfe[ (ulong)b*nf + h*n_thr + j ] =
                (uchar)( rel[(ulong)b*n_hid+h] >= m->taur[(ulong)h*n_thr+j] );
      } else {
        for( uint b=0u; b<batch; b++ ) {
          uint s = pb[b];
          for( uint k=tr.off[s]; k<tr.off[s+1]; k++ ) tfe[ (ulong)b*nf + tr.idx[k] ] = 1u;
        }
      }
      for( uint b=0u; b<batch; b++ ) tfe[ (ulong)b*nf + nf-1u ] = 1u;   /* class prior */

      /* ---- forward, output ---- */
      #pragma omp parallel for schedule(static)
      for( uint c=0u; c<m->n_out; c++ ) {
        float const * w = m->w2 + (ulong)c*nf;
        float *       p = p2    + (ulong)c*nf;
        float         sp = 0.0f;
        for( uint j=0u; j<nf; j++ ) {
          float wj = w[j];
          if( wj>0.0f ) { float t = exp2f(-wj); sp += wj+log2f(1.0f+t); p[j] = 1.0f/(1.0f+t); }
          else          { float t = exp2f( wj); sp +=    log2f(1.0f+t); p[j] = t/(1.0f+t);    }
        }
        m->bias2[c] = sp;
        for( uint b=0u; b<batch; b++ ) {
          float         s = 0.0f;
          uchar const * t = tfe + (ulong)b*nf;
          for( uint j=0u; j<nf; j++ ) if( t[j] ) s += w[j];
          cl2[ (ulong)b*m->n_out + c ] = sp - s;
        }
      }

      /* ---- loss ---- */
      for( uint b=0u; b<batch; b++ ) {
        float * cl = cl2 + (ulong)b*m->n_out;
        float   mx = -cl[0]; int arg = 0;
        for( uint c=1u; c<m->n_out; c++ ) if( -cl[c]>mx ) { mx = -cl[c]; arg = (int)c; }
        float sum = 0.0f;
        for( uint c=0u; c<m->n_out; c++ ) sum += expf( -cl[c]-mx );
        uint y = tr.label[ pb[b] ];
        loss += (double)( cl[y]+mx+logf(sum) );
        hit  += (ulong)( arg==(int)y );
        for( uint c=0u; c<m->n_out; c++ )
          d2[ (ulong)b*m->n_out + c ] = ( (c==y) ? 1.0f : 0.0f ) - expf( -cl[c]-mx )/sum;
      }
      seen += batch;
      float scale = 1.0f/(float)batch;

      /* ---- backward, output ---- */
      #pragma omp parallel for schedule(static)
      for( uint c=0u; c<m->n_out; c++ ) {
        float * g = gw2 + (ulong)c*nf;
        float   D = 0.0f;
        for( uint b=0u; b<batch; b++ ) D += d2[ (ulong)b*m->n_out + c ];
        float const * p = p2 + (ulong)c*nf;
        for( uint j=0u; j<nf; j++ ) g[j] = p[j]*D;
        for( uint b=0u; b<batch; b++ ) {
          float         e = d2[ (ulong)b*m->n_out + c ];
          uchar const * t = tfe + (ulong)b*nf;
          for( uint j=0u; j<nf; j++ ) if( t[j] ) g[j] -= e;
        }
        for( uint j=0u; j<nf; j++ ) g[j] *= scale;
      }

      if( n_hid ) {
        for( uint b=0u; b<batch; b++ ) {
          float * o = dt + (ulong)b*nf;
          for( uint j=0u; j<nf; j++ ) o[j] = 0.0f;
          for( uint c=0u; c<m->n_out; c++ ) {
            float         e = d2[ (ulong)b*m->n_out + c ];
            float const * w = m->w2 + (ulong)c*nf;
            for( uint j=0u; j<nf; j++ ) o[j] -= e*w[j];
          }
        }
        #pragma omp parallel for schedule(static)
        for( uint h=0u; h<n_hid; h++ ) {
          float sl = m->slope[h];
          float gs = 0.0f;
          for( uint j=0u; j<n_thr; j++ ) gtau[ (ulong)h*n_thr + j ] = 0.0f;
          for( uint b=0u; b<batch; b++ ) {
            float r   = rel[ (ulong)b*n_hid + h ];
            float acc = 0.0f;
            for( uint j=0u; j<n_thr; j++ ) {
              float d  = r - m->taur[ (ulong)h*n_thr + j ];
              float sg = 1.0f/(1.0f+expf(-d*sl));
              float dd = sg*(1.0f-sg);
              float go = dt[ (ulong)b*nf + h*n_thr + j ];
              acc += go*dd*sl;
              gtau[ (ulong)h*n_thr + j ] -= go*dd*sl*scale;
              gs  += go*dd*d*scale;
            }
            d1[ (ulong)b*n_hid + h ] = acc;
          }
          gsl[ h ] = gs;
        }
        /* d rel / d w1_hi = -x_i, so this gradient is entirely sparse */
        #pragma omp parallel for schedule(static)
        for( uint h=0u; h<n_hid; h++ ) {
          float * g = gw1 + (ulong)h*m->n_in;
          memset( g, 0, sizeof(float)*m->n_in );
          for( uint b=0u; b<batch; b++ ) {
            uint  s = pb[b];
            float e = d1[ (ulong)b*n_hid + h ]*scale;
            for( uint k=tr.off[s]; k<tr.off[s+1]; k++ ) g[ tr.idx[k] ] -= e;
          }
        }
      }

      b1t *= 0.9f; b2t *= 0.999f;
      adam_step( &ad_w2, m->w2, gw2, lr, b1t, b2t, ANS_W_MAX );
      if( n_hid ) {
        adam_step( &ad_w1,  m->w1,    gw1,  lr,        b1t, b2t, ANS_W_MAX );
        adam_step( &ad_tau, m->taur,  gtau, lr*10.0f,  b1t, b2t, 0.0f );
        adam_step( &ad_sl,  m->slope, gsl,  lr*0.01f,  b1t, b2t, 0.0f );
        /* the hard compare ignores the sign of the slope but the
           straight through gradient does not, so a slope that crossed
           zero would push every weight the wrong way */
        for( uint h=0u; h<n_hid; h++ ) if( m->slope[h]<1e-5f ) m->slope[h] = 1e-5f;
      }
    }

    ans_model_refresh( m );
    double acc = evaluate( m, &te );
    int    imp = acc>best;
    if( imp ) { best = acc; ans_model_save( m, out ); }
    printf( "epoch %2u  loss %.4f  train %.4f  test %.4f%s\n",
            ep+1u, loss/(double)seen, (double)hit/(double)seen, acc, imp ? "  *" : "" );
    if( verbose && n_hid ) {
      uint    nw   = 2000u<tr.n ? 2000u : tr.n;
      uint    live = 0u;
      float * rate = calloc( nf, sizeof(float) );
      #pragma omp parallel for schedule(static)
      for( uint h=0u; h<n_hid; h++ )
        for( uint b=0u; b<nw; b++ ) {
          float r = ans_model_rel( m->w1+(ulong)h*m->n_in, tr.idx+tr.off[b], tr.off[b+1]-tr.off[b] );
          for( uint j=0u; j<n_thr; j++ )
            if( r>=m->taur[(ulong)h*n_thr+j] ) rate[ h*n_thr+j ] += 1.0f;
        }
      for( uint j=0u; j<nf-1u; j++ ) {
        float r = rate[j]/(float)nw;
        if( r>0.02f && r<0.98f ) live++;
      }
      float wabs = 0.0f;
      for( ulong k=0UL; k<(ulong)n_hid*m->n_in; k++ ) wabs += fabsf( m->w1[k] );
      printf( "          live feat %u/%u  mean|w1| %.3f\n",
              live, nf-1u, wabs/(float)((ulong)n_hid*m->n_in) );
      free( rate );
    }
    fflush( stdout );
  }

  printf( "best test accuracy %.4f, saved to %s\n", best, out );
  ans_model_free( m );
  ans_data_free( &tr ); ans_data_free( &te );
  return 0;
}
