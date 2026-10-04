#ifndef HEADER_fd_flamenco_accdb_fd_zle14_ref_h
#define HEADER_fd_flamenco_accdb_fd_zle14_ref_h

/* Scalar reference for fd_zle14 (test only).  fd_zle14_ref_compress
   produces byte for byte what fd_zle14.c produces, and
   fd_zle14_ref_decompress inverts it.  Both are written for clarity;
   the format itself is documented in fd_zle14.c. */

#include "../../util/bits/fd_bits.h"

#define FD_ZLE14_REF_SPAN  (2048UL) /* input bytes per chunk span */
#define FD_ZLE14_REF_MASK  (   5UL) /* transitions at which mask mode wins */

/* fd_zle14_ref_keep returns the literal mask of the bs<=64 byte block at
   s: bit i is clear iff byte i belongs to a coded zero run, i.e. a run
   of at least 3 zeros that lies entirely inside the block. */

static inline ulong
fd_zle14_ref_keep( uchar const * s,
                   ulong         bs ) {
  ulong full = bs>=64UL ? ~0UL : (1UL<<bs)-1UL;
  ulong z    = 0UL;
  for( ulong i=0UL; i<bs; i++ ) z |= ((ulong)( s[i]==0 ))<<i;
  ulong r2 = z & (z>>1);
  ulong b  = ( r2 | (r2<<1) ) & full;
  return (~b) & full;
}

static inline ulong
fd_zle14_ref_compress( uchar *       dst,
                       uchar const * src,
                       ulong         n ) {
  uchar   lit[ 2176 ];
  uchar * out = dst;
  ulong   pos = 0UL;
  ulong   rem = n ? ((n-1UL)&63UL)+1UL : 0UL;

  if( FD_UNLIKELY( !n ) ) return 0UL;

  while( pos<n ) { /* one chunk */
    uchar * hdr    = out; out += 2UL;
    uchar * ctl    = out;
    ulong   lit_sz = 0UL;
    ulong   span;

  again: /* (re)set the chunk's block span: whole blocks, and never more
            literals than the chunk's scratch still holds */
    span = pos + fd_ulong_min( (FD_ZLE14_REF_SPAN-lit_sz) & ~63UL, (n-pos) & ~63UL );

    while( pos<span ) { /* full blocks */
      ulong keep = fd_zle14_ref_keep( src+pos, 64UL );
      ulong b    = ~keep;
      ulong T    = b ^ (b<<1);
      ulong t    = (ulong)fd_ulong_popcnt( T );

      if( !t ) { /* a run of all literal blocks */
        ulong cnt = 0UL;
        for(;;) {
          for( ulong i=0UL; i<64UL; i++ ) lit[ lit_sz++ ] = src[ pos+i ];
          pos += 64UL;
          cnt++;
          if( pos>=span ) break;
          if( fd_zle14_ref_keep( src+pos, 64UL )!=~0UL ) break;
        }
        *out++ = (uchar)( 0x3FUL + cnt );
        continue;
      }

      if( !keep ) { /* a run of all zero blocks: up to the format's 64 */
        ulong zlim = pos + fd_ulong_min( 64UL*64UL, (n-pos) & ~63UL );
        ulong cnt  = 0UL;
        for(;;) {
          pos += 64UL;
          cnt++;
          if( pos>=zlim ) break;
          if( fd_zle14_ref_keep( src+pos, 64UL ) ) break;
        }
        *out++ = (uchar)( 0xBFUL + cnt );
        continue;
      }

      if( t>=FD_ZLE14_REF_MASK ) { /* mask mode */
        *out++ = (uchar)0x7F;
        for( ulong i=0UL; i<8UL; i++ ) *out++ = (uchar)( keep>>(8*i) );
      } else { /* position list */
        ulong m = T;
        for( ulong i=0UL; i<t; i++ ) {
          ulong p = (ulong)fd_ulong_find_lsb( m );
          m &= m-1UL;
          *out++ = (uchar)( p | ( i==t-1UL ? 0x80UL : 0UL ) );
        }
      }
      for( ulong i=0UL; i<64UL; i++ ) if( keep & (1UL<<i) ) lit[ lit_sz++ ] = src[ pos+i ];
      pos += 64UL;
    }

    if( pos<n && n-pos<64UL ) { /* the short last block, in this chunk */
      ulong bs   = n-pos;
      ulong full = (1UL<<bs)-1UL;
      ulong keep = fd_zle14_ref_keep( src+pos, bs );
      ulong b    = (~keep) & full;
      ulong T    = ( b ^ (b<<1) ) & full;
      ulong t    = (ulong)fd_ulong_popcnt( T );
      ulong nz   = 0UL;
      for( ulong i=0UL; i<bs; i++ ) nz |= ((ulong)( src[pos+i]!=0 ))<<i;
      if     ( !nz   ) { *out++ = (uchar)0xC0; keep = 0UL; } /* all zero, any size */
      else if( !t    ) *out++ = (uchar)0x40;
      else if( t>=FD_ZLE14_REF_MASK ) {
        *out++ = (uchar)0x7F;
        for( ulong i=0UL; i<8UL; i++ ) *out++ = (uchar)( keep>>(8*i) );
      } else {
        ulong m = T;
        for( ulong i=0UL; i<t; i++ ) {
          ulong p = (ulong)fd_ulong_find_lsb( m );
          m &= m-1UL;
          *out++ = (uchar)( p | ( i==t-1UL ? 0x80UL : 0UL ) );
        }
      }
      for( ulong i=0UL; i<bs; i++ ) if( keep & (1UL<<i) ) lit[ lit_sz++ ] = src[ pos+i ];
      pos = n;
    } else if( pos<n ) { /* the span ran out: keep the chunk if its fields can hold more */
      if( (ulong)( out-ctl )<200UL && lit_sz+64UL<=FD_ZLE14_REF_SPAN ) goto again;
    }

    ulong ctl_sz = (ulong)( out-ctl );
    ulong last   = pos>=n;
    ulong r      = last ? rem : 64UL;
    ulong h      = (ctl_sz<<7) | ((r-1UL)<<1) | last;
    hdr[0] = (uchar)h; hdr[1] = (uchar)(h>>8);
    for( ulong i=0UL; i<lit_sz; i++ ) *out++ = lit[i];
  }

  return (ulong)( out-dst );
}

/* fd_zle14_ref_decompress mirrors fd_zle14_decompress.  Trusts its
   input except for the bounds checks the real decoder makes. */

static inline long
fd_zle14_ref_decompress( uchar *       dst,
                         ulong         dst_sz,
                         uchar const * src,
                         ulong         src_sz ) {
  uchar const * p    = src;
  uchar const * pend = src+src_sz;
  uchar *       out  = dst;
  uchar *       oend = dst+dst_sz;

  if( FD_UNLIKELY( !src_sz ) ) return 0L;

  while( p<pend ) {
    if( FD_UNLIKELY( pend-p<2L ) ) return -1L;
    ulong h      = (ulong)p[0] | ((ulong)p[1]<<8);
    ulong ctl_sz = h>>7;
    ulong rem    = ((h>>1)&63UL)+1UL;
    ulong last   = h&1UL;
    uchar const * ctl  = p+2;
    uchar const * cend = ctl+ctl_sz;
    if( FD_UNLIKELY( !ctl_sz || cend>pend ) ) return -1L;
    uchar const * lit  = cend;

    while( ctl<cend ) {
      ulong c    = (ulong)*ctl;
      ulong keep = 0UL;
      ulong cnt  = 1UL;
      int   zero = 0, plain = 0;

      if( !(c & 0x40UL) ) { /* position list */
        ulong t = 0UL;
        ulong z = 0UL;
        for(;;) {
          if( FD_UNLIKELY( ctl+t>=cend ) ) return -1L;
          ulong v = (ulong)ctl[t];
          z ^= ~0UL<<(v&63UL);
          t++;
          if( v & 0x80UL ) break;
          if( FD_UNLIKELY( t>8UL ) ) return -1L;
        }
        ctl  += t;
        keep  = ~z;
      } else if( c==0x7FUL ) { /* mask mode */
        if( FD_UNLIKELY( cend-ctl<9L ) ) return -1L;
        for( ulong i=0UL; i<8UL; i++ ) keep |= ((ulong)ctl[1+i])<<(8*i);
        ctl += 9;
      } else if( c<0x7FUL ) { /* all literal run */
        cnt = c-0x3FUL; ctl++; plain = 1; keep = ~0UL;
      } else { /* all zero run */
        cnt = c-0xBFUL; ctl++; zero = 1;
      }
      (void)plain;

      for( ulong k=0UL; k<cnt; k++ ) {
        ulong bs = ( last && ctl>=cend && k==cnt-1UL ) ? rem : 64UL;
        if( FD_UNLIKELY( (ulong)(oend-out)<bs ) ) return -2L;
        ulong full = bs>=64UL ? ~0UL : (1UL<<bs)-1UL;
        ulong kp   = zero ? 0UL : ( keep & full );
        ulong nlit = (ulong)fd_ulong_popcnt( kp );
        if( FD_UNLIKELY( (ulong)(pend-lit)<nlit ) ) return -1L;
        for( ulong i=0UL; i<bs; i++ ) out[i] = (uchar)( (kp & (1UL<<i)) ? *lit++ : 0 );
        out += bs;
      }
    }
    p = lit;
  }

  return (long)( out-dst );
}

#endif /* HEADER_fd_flamenco_accdb_fd_zle14_ref_h */
