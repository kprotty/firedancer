#ifndef HEADER_fd_flamenco_accdb_fd_zle9_h
#define HEADER_fd_flamenco_accdb_fd_zle9_h

/* fd_zle9 is fd_zle8's wire format with a rewritten encoder.

   The output is byte-identical to fd_zle8_compress for every input, so
   ratio, decoder and output bound are fd_zle8's.  What differs is the
   encoder's control flow: instead of fd_zle8's absolute-cursor scan
   helpers (zle8_cand / zle8_zmask / zle8_spill), it is written in the
   explicit branch style of fd_zle_pseudocode3 -- a pointer cursor, a
   full-word pair scan, a separate zero-run spill loop entered on carry
   out of the candidate mask, and a single end-aligned load for the last
   <=64 bytes -- with dedicated tiny (<=64 B), small (65..254 B, no field
   escape possible) and big instantiations.  See fd_zle9.c. */

#include "../../util/fd_util_base.h"

#define FD_ZLE9_COMPRESS_BOUND( in_sz ) ( (in_sz) + (in_sz)/64UL + 128UL )

#define FD_ZLE9_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE9_ERR_SPACE   (-1L) /* out of output buffer space */
#define FD_ZLE9_ERR_CORRUPT (-2L) /* malformed compressed data */

FD_PROTOTYPES_BEGIN

/* fd_zle9_compress compresses data_sz bytes at data into comp and
   returns the number of compressed bytes written (0 when data_sz==0).
   comp must have room for FD_ZLE9_COMPRESS_BOUND( data_sz ) bytes.
   data_sz <= FD_ZLE9_MAX_SZ.  The readable span starting at data must
   cover fd_ulong_align_up( data_sz, 64 ) bytes (the tiny path does a
   masked load; larger inputs never read past data+data_sz). */

ulong
fd_zle9_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz );

/* fd_zle9_decompress: same contract as fd_zle8_decompress. */

long
fd_zle9_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz );

char const *
fd_zle9_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle9_h */
