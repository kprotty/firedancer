#ifndef HEADER_fd_flamenco_accdb_fd_zle4_h
#define HEADER_fd_flamenco_accdb_fd_zle4_h

/* fd_zle4.h is an experimental redesign of fd_zle (zero run length
   coding) that is free to break the wire format in pursuit of raw
   encode/decode throughput on AVX-512.

   Key differences from fd_zle:

   - Planar layout.  The compressed blob is a small header, a "control
     plane" (a fixed-stride stream of (literal_len, zero_len) frame
     descriptors), then a "literal plane" (every literal byte, packed
     contiguously).  The decoder walks the control plane with no literal
     bytes in the way, which keeps its instruction stream tight and lets
     it validate the whole frame list up front and then emit every frame
     with zero per-frame safety branches.

   - Looser bound.  FD_ZLE4_COMPRESS_BOUND is 3*in_sz/2 + 32 rather than
     in_sz + 8.  The encoder never runs a per-frame "would this frame
     overflow the bound" check (fd_zle's is a measured 13-21% of its
     encode time on frame-heavy inputs); instead it falls back to a
     single stored (raw) frame if the compressed form is not smaller
     than the input.

   - The decoder is told the decompressed size out of band (data_sz),
     same as fd_zle.  The compressed blob does not store it. */

#include "../../util/fd_util_base.h"

#define FD_ZLE4_OVERHEAD (32UL)
#define FD_ZLE4_COMPRESS_BOUND( in_sz ) (FD_ZLE4_OVERHEAD + (in_sz) + (in_sz)/2UL)

/* FD_ZLE4_MAX_SZ is the largest input fd_zle4_compress accepts. */

#define FD_ZLE4_MAX_SZ (0xFFFFFFF0UL) /* ~4 GiB, lengths are u32 on the wire */

#define FD_ZLE4_ERR_SPACE   (-1L) /* out of buffer space */
#define FD_ZLE4_ERR_CORRUPT (-2L) /* malformed compressed data */

FD_PROTOTYPES_BEGIN

/* fd_zle4_compress compresses data_sz bytes at data.  Writes up to
   FD_ZLE4_COMPRESS_BOUND( data_sz ) compressed bytes to comp.  Returns
   the number of compressed bytes written.  data_sz<=FD_ZLE4_MAX_SZ.

   data has no alignment requirement, but the readable span starting at
   data must cover fd_ulong_align_up( data_sz, 64 ) bytes.  The tail
   padding does not affect the output. */

ulong
fd_zle4_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz );

/* fd_zle4_decompress takes an fd_zle4 compressed blob at comp.  Writes
   exactly data_sz decompressed bytes to data on success.  Returns the
   number of bytes decompressed (>=0) on success, or a negative
   FD_ZLE4_ERR_* number on failure.  data, data_sz have no alignment
   requirements; no bytes past data+data_sz are written.  comp/comp_sz
   have no alignment requirements and no OOB reads are done.  Safe (but
   possibly slow) given arbitrary input. */

long
fd_zle4_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz );

/* fd_zle4_strerror returns a static-lifetime cstr describing a
   fd_zle4_decompress error code. */

char const *
fd_zle4_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle4_h */
