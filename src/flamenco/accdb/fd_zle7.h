#ifndef HEADER_fd_flamenco_accdb_fd_zle7_h
#define HEADER_fd_flamenco_accdb_fd_zle7_h

/* fd_zle7 is a prototype zero-run-length codec for Solana account data,
   implementing the wire format sketched in `fd_zle_pseudocode2`: a
   1-byte header (`type:u2, ext:u6`) picking one of four frame kinds
   (raw / fill / last / more), rather than fd_zle5/6's single fixed
   frame shape.  See the wire format comment at the top of fd_zle7.c.

   All four frame types are implemented: data_sz<=64 uses the `raw` /
   `fill` / `last` small encoder, data_sz>64 uses the `more` encoder
   (a planar literal-plane-then-token-stream frame, fd_zle5/6-style
   scan machinery, delta-varint-escaping tokens) -- see the wire format
   comment and zle7_compress_more's comment in fd_zle7.c for both.
   The `more` encoder falls back to a `raw` frame whenever the
   compressed form doesn't actually win, or its literal plane exceeds
   the 22-bit size field's ~4 MiB cap (see FD_ZLE7_COMPRESS_BOUND
   below), so the API is total for any data_sz<=FD_ZLE7_MAX_SZ. */

#include "../../util/fd_util_base.h"

/* FD_ZLE7_OVERHEAD / FD_ZLE7_COMPRESS_BOUND: same generous slack as
   fd_zle5/6 (a per-token bound test, not a per-frame one, pays for
   it). */

#define FD_ZLE7_OVERHEAD (32UL)
#define FD_ZLE7_COMPRESS_BOUND( in_sz ) (FD_ZLE7_OVERHEAD + (in_sz) + (in_sz)/2UL)

/* FD_ZLE7_MAX_SZ is the largest input fd_zle7_compress accepts. */

#define FD_ZLE7_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE7_ERR_SPACE   (-1L) /* out of output buffer space */
#define FD_ZLE7_ERR_CORRUPT (-2L) /* malformed compressed data */

FD_PROTOTYPES_BEGIN

/* fd_zle7_compress compresses data_sz bytes at data into comp and
   returns the number of compressed bytes written (0 when data_sz==0).
   comp must have room for FD_ZLE7_COMPRESS_BOUND( data_sz ) bytes.
   data_sz <= FD_ZLE7_MAX_SZ.

   data has no alignment requirement, but the readable span starting at
   data must cover fd_ulong_align_up( data_sz, 64 ) bytes: the small
   path always loads a full 64-byte word and masks the tail, so the
   padding is read but never affects the output. */

ulong
fd_zle7_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz );

/* fd_zle7_decompress expands an fd_zle7 blob at comp/comp_sz into data,
   writing exactly data_sz bytes on success.  Returns the number of
   bytes written (>=0), or a negative FD_ZLE7_ERR_* code on failure.

   Safe against arbitrary input: never reads out of bounds on
   comp/comp_sz, never writes past data+data_sz.  data, data_sz, comp
   and comp_sz have no alignment requirements. */

long
fd_zle7_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz );

/* fd_zle7_strerror returns a static-lifetime cstr describing an
   fd_zle7_decompress return code. */

char const *
fd_zle7_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle7_h */
