#ifndef HEADER_fd_flamenco_accdb_fd_zle11_h
#define HEADER_fd_flamenco_accdb_fd_zle11_h

/* fd_zle11 is a zero run length compressor for Solana account data.  It
   produces the same output as fd_zle8 and fd_zle10. */

#include "../../util/fd_util_base.h"

#define FD_ZLE11_COMPRESS_BOUND( in_sz ) ( (in_sz) + (in_sz)/64UL + 128UL )

#define FD_ZLE11_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE11_ERR_STREAM (-1L) /* malformed compressed data */
#define FD_ZLE11_ERR_SPACE  (-2L) /* out of buffer space */

FD_PROTOTYPES_BEGIN

/* fd_zle11_compress compresses data_sz bytes at data.  Writes up to
   FD_ZLE11_COMPRESS_BOUND( data_sz ) bytes to comp.  Returns the number
   of bytes written.  data_sz<=FD_ZLE11_MAX_SZ.

   data has no alignment requirement, but the readable span starting at
   data must cover fd_ulong_align_up( data_sz, 64 ) bytes.  The padding
   does not affect the output. */

ulong
fd_zle11_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz );

/* fd_zle11_decompress decompresses comp_sz bytes at comp.  Writes up to
   data_sz bytes to data.  Returns the number of bytes decompressed
   (>=0) on success, or a negative FD_ZLE11_ERR_* code on failure.
   data_sz is a capacity: callers that know the decompressed size must
   check the return value.  No alignment requirements.  Safe given
   arbitrary input: never reads past comp+comp_sz or writes past
   data+data_sz.

   The caller assumes all data_sz output bytes are invalidated. */

long
fd_zle11_decompress( void *       FD_RESTRICT data,
                     ulong                    data_sz,
                     void const * FD_RESTRICT comp,
                     ulong                    comp_sz );

/* fd_zle11_strerror returns a static-lifetime cstr describing a return
   code of fd_zle11_decompress. */

char const *
fd_zle11_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle11_h */
