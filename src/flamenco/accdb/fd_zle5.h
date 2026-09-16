#ifndef HEADER_fd_flamenco_accdb_fd_zle5_h
#define HEADER_fd_flamenco_accdb_fd_zle5_h

/* fd_zle5 is a zero-run-length codec for Solana account data.  It is a
   cleaned-up restatement of fd_zle4: same algorithm, same wire format,
   more comments.  See ZLE4_DESIGN.md for the full rationale; the short
   version follows.

   The workload
   ------------
   Account data is mostly zeros (unset option fields, padded slabs,
   fresh allocations).  The non-zero bytes -- pubkeys, balances,
   fixed-point fields -- do not compress, so the codec only answers one
   question: where are the zero runs and how long are they?  There is no
   entropy stage and no attempt to find repeats in the non-zero bytes.

   Planar layout
   -------------
   The compressed blob is

       [header] [literal plane] [control plane]

   The literal plane is every byte that is not part of an encoded zero
   run, concatenated in input order, nothing else.  The control plane is
   a dense array of (literal_len, zero_len) frame descriptors, also in
   input order.  Keeping the two apart means the decoder's inner loop is
   a tight parse of fixed-stride records with no literal bytes in the
   instruction stream.

   Bound
   -----
   FD_ZLE5_COMPRESS_BOUND is 3*in_sz/2 + 32, looser than a byte-for-byte
   codec would need.  The slack pays for dropping fd_zle's per-frame
   "would this frame overflow the output" check (a measured 13-21% of
   its encode time on frame-heavy inputs).  Instead the encoder falls
   back to a single stored (raw) frame whenever the compressed form
   would not be strictly smaller than the input.

   Decompressed size
   -----------------
   Not stored in the blob.  The caller passes it out of band (data_sz),
   exactly as fd_zle does; the accdb layer already knows every account's
   length. */

#include "../../util/fd_util_base.h"

/* FD_ZLE5_OVERHEAD is the fixed slack term in the compress bound.
   FD_ZLE5_COMPRESS_BOUND( n ) is an upper bound on the number of bytes
   fd_zle5_compress may write for an n byte input. */

#define FD_ZLE5_OVERHEAD (32UL)
#define FD_ZLE5_COMPRESS_BOUND( in_sz ) (FD_ZLE5_OVERHEAD + (in_sz) + (in_sz)/2UL)

/* FD_ZLE5_MAX_SZ is the largest input fd_zle5_compress accepts.  Frame
   lengths are u32 on the wire, so the limit is just under 4 GiB. */

#define FD_ZLE5_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE5_ERR_SPACE   (-1L) /* out of output buffer space */
#define FD_ZLE5_ERR_CORRUPT (-2L) /* malformed compressed data */

FD_PROTOTYPES_BEGIN

/* fd_zle5_compress compresses data_sz bytes at data into comp and
   returns the number of compressed bytes written (0 when data_sz==0).
   comp must have room for FD_ZLE5_COMPRESS_BOUND( data_sz ) bytes.
   data_sz <= FD_ZLE5_MAX_SZ.

   data has no alignment requirement, but the readable span starting at
   data must cover fd_ulong_align_up( data_sz, 64 ) bytes: the scan
   always loads full 64-byte words and masks the tail, so the padding is
   read but never affects the output. */

ulong
fd_zle5_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz );

/* fd_zle5_decompress expands an fd_zle5 blob at comp/comp_sz into data,
   writing exactly data_sz bytes on success.  Returns the number of
   bytes written (>=0), or a negative FD_ZLE5_ERR_* code on failure.

   Safe against arbitrary input: never reads out of bounds on
   comp/comp_sz, never writes past data+data_sz.  data, data_sz, comp
   and comp_sz have no alignment requirements. */

long
fd_zle5_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz );

/* fd_zle5_strerror returns a static-lifetime cstr describing an
   fd_zle5_decompress return code. */

char const *
fd_zle5_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle5_h */
