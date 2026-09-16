#ifndef HEADER_fd_flamenco_accdb_fd_zle6_h
#define HEADER_fd_flamenco_accdb_fd_zle6_h

/* fd_zle6 is a zero-run-length codec for Solana account data.  It keeps
   fd_zle5's scanning core (zero mask, candidate/carry-propagate run
   resolution, 256B skip-aheads, tiny/small/big size dispatch) and the
   literal-plane-then-control-stream layout, but changes the control
   token itself: `fd_zle_pseudocode`'s independently-escaping (L, Z)
   token instead of fd_zle5's jointly-escaping (ll, zl) frame.  See the
   wire format comment in fd_zle6.c for the exact layout and the
   deviations from the pseudocode this implements (both documented and
   deliberate -- the pseudocode had two underspecified corners: the
   size==0 stored sentinel collides with an all-zero-literal frame, and
   the universal Z+2 bias leaves no code point for "no trailing zero
   run").

   The workload, ratio/bound rationale and AVX-512 scanning technique
   are otherwise identical to fd_zle5 -- see ZLE4_DESIGN.md -- so this
   header only documents what's different. */

#include "../../util/fd_util_base.h"

/* FD_ZLE6_OVERHEAD / FD_ZLE6_COMPRESS_BOUND: same generous slack as
   fd_zle5 (a per-token bound test, not a per-frame one, pays for it). */

#define FD_ZLE6_OVERHEAD (32UL)
#define FD_ZLE6_COMPRESS_BOUND( in_sz ) (FD_ZLE6_OVERHEAD + (in_sz) + (in_sz)/2UL)

/* FD_ZLE6_MAX_SZ is the largest input fd_zle6_compress accepts.  Token
   length extensions are u32 on the wire, so the limit is just under
   4 GiB (matches fd_zle5). */

#define FD_ZLE6_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE6_ERR_SPACE   (-1L) /* out of output buffer space */
#define FD_ZLE6_ERR_CORRUPT (-2L) /* malformed compressed data */

FD_PROTOTYPES_BEGIN

/* fd_zle6_compress compresses data_sz bytes at data into comp and
   returns the number of compressed bytes written (0 when data_sz==0).
   comp must have room for FD_ZLE6_COMPRESS_BOUND( data_sz ) bytes.
   data_sz <= FD_ZLE6_MAX_SZ.

   data has no alignment requirement, but the readable span starting at
   data must cover fd_ulong_align_up( data_sz, 64 ) bytes: the scan
   always loads full 64-byte words and masks the tail, so the padding is
   read but never affects the output. */

ulong
fd_zle6_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz );

/* fd_zle6_decompress expands an fd_zle6 blob at comp/comp_sz into data,
   writing exactly data_sz bytes on success.  Returns the number of
   bytes written (>=0), or a negative FD_ZLE6_ERR_* code on failure.

   Safe against arbitrary input: never reads out of bounds on
   comp/comp_sz, never writes past data+data_sz.  data, data_sz, comp
   and comp_sz have no alignment requirements. */

long
fd_zle6_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz );

/* fd_zle6_strerror returns a static-lifetime cstr describing an
   fd_zle6_decompress return code. */

char const *
fd_zle6_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle6_h */
