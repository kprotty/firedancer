#ifndef HEADER_fd_flamenco_accdb_fd_zle10_h
#define HEADER_fd_flamenco_accdb_fd_zle10_h

/* fd_zle10 is fd_zle_pseudocode4's structure with each piece resolved
   by measurement against fd_zle9 (mainnet-shaped mix, bench_zle10):
   enc +13.7%, dec +0.9%.

   fd_zle10_compress is byte-identical to fd_zle8_compress /
   fd_zle9_compress for every input, so ratio and output bound are
   fd_zle8's.  Relative to fd_zle9:

     - pseudocode4's goto state machine, narrow-templated copy, and
       subtraction-form decoder bounds;
     - no per-token output bound check.  The bound is proven (fd_zle8.c
       "Output bound": out <= in_sz+in_sz/386+6 against a bound of
       in_sz+in_sz/64+128), so fd_zle9's check is unreachable, and it
       measured -3.8% encode;
     - fd_zle9's two-word scan before committing to a 256 B stride
       (+13% mix encode over striding after one) and fd_zle9's
       straight-line zero tier ladder (+1.6% mix decode over a counted
       loop);
     - the decoder takes data_sz as a capacity rather than an exact
       size, after fd_zle's contract (see fd_zle10_decompress).

   See fd_zle10.c. */

#include "../../util/fd_util_base.h"

#define FD_ZLE10_COMPRESS_BOUND( in_sz ) ( (in_sz) + (in_sz)/64UL + 128UL )

#define FD_ZLE10_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE10_ERR_STREAM (-1L) /* malformed compressed data */
#define FD_ZLE10_ERR_SPACE  (-2L) /* data_sz cannot hold the stream */

FD_PROTOTYPES_BEGIN

/* fd_zle10_compress compresses data_sz bytes at data into comp and
   returns the number of compressed bytes written (0 when data_sz==0).
   comp must have room for FD_ZLE10_COMPRESS_BOUND( data_sz ) bytes.
   data_sz <= FD_ZLE10_MAX_SZ.  The readable span starting at data must
   cover fd_ulong_align_up( data_sz, 64 ) bytes (the tiny path does an
   unmasked 64 B load; larger inputs never read past data+data_sz). */

ulong
fd_zle10_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz );

/* fd_zle10_decompress expands an fd_zle10 blob at comp/comp_sz into
   data.  Writes up to data_sz bytes and returns the number of bytes
   decompressed (>=0), or a negative FD_ZLE10_ERR_* code.  data_sz is a
   capacity, not an assertion: a caller that knows the decompressed size
   must compare it against the return value.  The caller assumes all
   data_sz output bytes are invalidated.

   Safe against arbitrary input: never reads out of bounds on
   comp/comp_sz, never writes past data+data_sz.  data, data_sz, comp
   and comp_sz have no alignment requirements. */

long
fd_zle10_decompress( void *       FD_RESTRICT data,
                     ulong                    data_sz,
                     void const * FD_RESTRICT comp,
                     ulong                    comp_sz );

/* fd_zle10_strerror returns a static-lifetime cstr describing an
   fd_zle10_decompress return code. */

char const *
fd_zle10_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle10_h */
