#ifndef HEADER_fd_flamenco_accdb_fd_zle12_ref_h
#define HEADER_fd_flamenco_accdb_fd_zle12_ref_h

/* fd_zle12_ref_compress is a byte at a time reference for
   fd_zle12_compress (test only).  Returns the output size. */

#include "../../util/bits/fd_bits.h"

static inline uchar *
fd_zle12_ref_put_vint( uchar * out,
                       ulong   v ) {
  while( v>=0x80UL ) { *out++ = (uchar)( v | 0x80UL ); v >>= 7; }
  *out++ = (uchar)v;
  return out;
}

static inline uchar *
fd_zle12_ref_token( uchar *       out,
                    uchar const * lit,
                    ulong         L,
                    ulong         Z ) {
  *out++ = (uchar)( ( fd_ulong_min( L, 15UL )<<4 ) | fd_ulong_min( Z, 15UL ) );
  if( L>=15UL ) out = fd_zle12_ref_put_vint( out, L-15UL );
  if( Z>=15UL ) out = fd_zle12_ref_put_vint( out, Z-15UL );
  for( ulong i=0UL; i<L; i++ ) *out++ = lit[i];
  return out;
}

static inline ulong
fd_zle12_ref_compress( uchar *       dst,
                       uchar const * src,
                       ulong         n ) {
  uchar * out = dst;
  ulong   lp  = 0UL;
  ulong   i   = 0UL;
  while( i<n ) {
    if( src[i] ) { i++; continue; }
    ulong j = i;
    while( j<n && !src[j] ) j++;
    if( j-i>=2UL ) {
      out = fd_zle12_ref_token( out, src+lp, i-lp, j-i );
      lp  = j;
    }
    i = j;
  }
  if( lp<n ) out = fd_zle12_ref_token( out, src+lp, n-lp, 0UL );
  return (ulong)( out-dst );
}

#endif /* HEADER_fd_flamenco_accdb_fd_zle12_ref_h */
