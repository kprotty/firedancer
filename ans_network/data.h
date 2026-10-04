#ifndef HEADER_ans_network_data_h
#define HEADER_ans_network_data_h

#include "ans.h"

/* A binarized image set.  bit is the dense 0/1 view the inference path
   walks; idx/off is the same data as a sparse list of set features,
   which is what makes both training passes O(set bits). */

struct ans_data {
  uint     n;        /* samples */
  uint     dim;      /* features per sample */
  uchar *  bit;      /* [n][dim] */
  uint *   idx;      /* set feature indices, concatenated */
  uint *   off;      /* [n+1] offsets into idx */
  uchar *  label;    /* [n] */
  uint     n_class;
};

typedef struct ans_data ans_data_t;

/* ans_data_load reads MNIST idx files from dir, binarizing pixels at
   thresh.  Returns 0 on success.  If the files are absent it falls
   back to a synthetic separable set so the project runs without the
   download, and sets *synth. */

int
ans_data_load( ans_data_t * d,
               char const * dir,
               int          test,
               uint         thresh,
               int *        synth );

void ans_data_free( ans_data_t * d );

#endif /* HEADER_ans_network_data_h */
