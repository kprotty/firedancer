#ifndef HEADER_fd_flamenco_accdb_fd_zle14_h
#define HEADER_fd_flamenco_accdb_fd_zle14_h

/* fd_zle14 is a zero run length compressor for Solana account data.

   Unlike fd_zle .. fd_zle13, which code one token per zero run and
   therefore walk the output one variable length field at a time,
   fd_zle14 codes the input in fixed 64 byte blocks.  Every block
   advances the input by 64 bytes and the output by an amount known
   before the block's bytes are touched, so neither encoder nor decoder
   carries a pointer chase through the data.  See fd_zle14.c for the
   wire format. */

#include "../../util/fd_util_base.h"

/* FD_ZLE14_COMPRESS_BOUND is the largest output fd_zle14_compress can
   produce.  A block costs at most 67 bytes: three 2 byte zero runs put it
   in mask mode, which is 9 bytes of code for 58 literals.  A chunk of 32
   such blocks costs 2 more, and the literal copy at the end of a chunk
   stores up to 63 bytes of slack past the output. */

#define FD_ZLE14_COMPRESS_BOUND( in_sz ) ( (in_sz) + (in_sz)/20UL + 128UL )

#define FD_ZLE14_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE14_ERR_STREAM (-1L) /* malformed compressed data */
#define FD_ZLE14_ERR_SPACE  (-2L) /* out of buffer space */

FD_PROTOTYPES_BEGIN

/* fd_zle14_compress compresses data_sz bytes at data.  Writes up to
   FD_ZLE14_COMPRESS_BOUND( data_sz ) bytes to comp.  Returns the number
   of bytes written.  data_sz<=FD_ZLE14_MAX_SZ.

   data has no alignment requirement, but the readable span starting at
   data must cover fd_ulong_align_up( data_sz, 64 ) bytes.  The padding
   does not affect the output.  comp must be writable for the full
   bound: the last chunk's literal copy overshoots the output by up to
   63 bytes. */

ulong
fd_zle14_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz );

/* fd_zle14_decompress decompresses comp_sz bytes at comp.  Writes up to
   data_sz bytes to data.  Returns the number of bytes decompressed
   (>=0) on success, or a negative FD_ZLE14_ERR_* code on failure.
   data_sz is a capacity: callers that know the decompressed size must
   check the return value.  No alignment requirements.  Safe given
   arbitrary input: never reads past comp+comp_sz or writes past
   data+data_sz.

   The caller assumes all data_sz output bytes are invalidated. */

long
fd_zle14_decompress( void *       FD_RESTRICT data,
                     ulong                    data_sz,
                     void const * FD_RESTRICT comp,
                     ulong                    comp_sz );

/* fd_zle14_strerror returns a static-lifetime cstr describing a return
   code of fd_zle14_decompress. */

char const *
fd_zle14_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle14_h */
