#ifndef HEADER_fd_flamenco_accdb_fd_zle8_h
#define HEADER_fd_flamenco_accdb_fd_zle8_h

/* fd_zle8 is a prototype zero-run-length codec for Solana account data
   using a *headerless, fully interleaved* wire format:

     frame := token*
     token := L-field literal[L] [ Z-field ]
     field := u8 b; if b==0xFF: b += varint

   i.e. "copy L bytes, then write Z zero bytes, repeat until the output
   is data_sz bytes long".  See the wire format comment at the top of
   fd_zle8.c for the exact decoder loop and the terminal-token rule.

   Versus fd_zle7 (and fd_zle5/6), which split each frame into a
   contiguous *literal plane* followed by a separate token stream, this
   format has no frame header at all, no frame type byte, no literal
   plane size field, no token-stream relocation, and no raw fallback
   path.  Three concrete consequences:

     - 3 fewer bytes on every >64 B input, 1 fewer on every <=64 B one.
       On the single most common account shape on chain (a 165 B SPL
       token account) that is 74 B out instead of fd_zle7's 78 B.

     - the encoder writes strictly forward and monotonically, so the
       output buffer bound is ~in_sz rather than fd_zle7's 1.5*in_sz
       (which is dictated by where fd_zle7 must provisionally park its
       token stream, not by any output it can actually produce).

     - there is no size field to overflow, so no input shape falls off
       a ratio cliff into a verbatim copy.

   The costs are that a fully incompressible input is 1..5 bytes over
   raw (one L field) rather than exactly 1, and that an all-zero input
   <=64 B is 2 bytes rather than fd_zle7's 1 (type.fill).  Both are
   strictly smaller than the savings above on every shape in the
   mainnet mix.  */

#include "../../util/fd_util_base.h"

/* FD_ZLE8_COMPRESS_BOUND: unlike fd_zle5/6/7's generous in_sz*1.5, this
   is a real bound on the encoder's output, derived in the "Output
   bound" comment in fd_zle8.c: every token whose fields both fit in one
   byte shrinks the output relative to the input, and a token that
   escapes a field to a varint has already consumed >=255 input bytes to
   pay for it.  in_sz/64 is ~24x the worst case escape density; 128 is
   slack for the final token plus a 64 B store tail. */

#define FD_ZLE8_COMPRESS_BOUND( in_sz ) ( (in_sz) + (in_sz)/64UL + 128UL )

/* FD_ZLE8_MAX_SZ is the largest input fd_zle8_compress accepts. */

#define FD_ZLE8_MAX_SZ (0xFFFFFFF0UL)

#define FD_ZLE8_ERR_SPACE   (-1L) /* out of output buffer space */
#define FD_ZLE8_ERR_CORRUPT (-2L) /* malformed compressed data */

FD_PROTOTYPES_BEGIN

/* fd_zle8_compress compresses data_sz bytes at data into comp and
   returns the number of compressed bytes written (0 when data_sz==0).
   comp must have room for FD_ZLE8_COMPRESS_BOUND( data_sz ) bytes.
   data_sz <= FD_ZLE8_MAX_SZ.

   data has no alignment requirement, but (as in fd_zle5/6/7) the
   readable span starting at data must cover
   fd_ulong_align_up( data_sz, 64 ) bytes: the scan always loads full
   64-byte words and masks the tail, so the padding is read but never
   affects the output. */

ulong
fd_zle8_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz );

/* fd_zle8_decompress expands an fd_zle8 blob at comp/comp_sz into data,
   writing exactly data_sz bytes on success.  Returns the number of
   bytes written (>=0), or a negative FD_ZLE8_ERR_* code on failure.

   Safe against arbitrary input: never reads out of bounds on
   comp/comp_sz, never writes past data+data_sz.  data, data_sz, comp
   and comp_sz have no alignment requirements. */

long
fd_zle8_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz );

/* fd_zle8_strerror returns a static-lifetime cstr describing an
   fd_zle8_decompress return code. */

char const *
fd_zle8_strerror( long res );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle8_h */
