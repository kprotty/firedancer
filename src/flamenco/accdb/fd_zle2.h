#ifndef HEADER_fd_flamenco_accdb_fd_zle2_h
#define HEADER_fd_flamenco_accdb_fd_zle2_h

/* fd_zle2.h is an experimental fd_zle encoder.  It targets the exact
   same wire format as fd_zle (see fd_zle.h / fd_zle.c) -- output that
   fd_zle_decompress decodes -- but restructures the compressor's
   inner loop.

   fd_zle_compress's general path (fd_zle_big) loads each 64 B word of
   the input twice: once to test it for zero bytes while hunting for
   run boundaries, and a second time -- from src, at fd_zle_emit --
   once a literal run's extent is finally known.  fd_zle2_compress
   instead keeps the vector register loaded for the zero test and
   stores it straight into a small stack scratch buffer as it scans,
   so every literal byte is read out of src exactly once, regardless
   of which frame it ends up belonging to.  See fd_zle2.c for the
   scratch buffer bookkeeping and what this trades away: fd_zle_big's
   256 B skip-ahead through very long uniform zero runs, and (since
   the scratch buffer is bounded) byte-identical output with fd_zle on
   inputs with a literal run longer than FD_ZLE2_LIT_CAP, which
   fd_zle2 instead splits with an extra interstitial frame.

   This is a prototype: no scalar fallback, not wired into any
   production path, decodes with the existing fd_zle_decompress. */

#include "fd_zle.h"

FD_PROTOTYPES_BEGIN

/* fd_zle2_compress: see fd_zle_compress.  Same contract: writes up to
   FD_ZLE_COMPRESS_BOUND( data_sz ) bytes to comp and returns the
   number written; data_sz<=FD_ZLE_MAX_SZ; the readable span at data
   must cover fd_ulong_align_up( data_sz, 64 ) bytes. */

ulong
fd_zle2_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz );

FD_PROTOTYPES_END

#endif /* HEADER_fd_flamenco_accdb_fd_zle2_h */
