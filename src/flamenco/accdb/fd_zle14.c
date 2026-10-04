#include "fd_zle14.h"

char const *
fd_zle14_strerror( long res ) {
  switch( res ) {
  case FD_ZLE14_ERR_STREAM: return "malformed compressed data";
  case FD_ZLE14_ERR_SPACE:  return "out of buffer space";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}

#if FD_HAS_AVX512

#include "../../util/simd/fd_avx512.h"

/* WIRE FORMAT

   The input is cut into blocks of 64 bytes; the last block holds the
   remaining 1..64.  Blocks are grouped into chunks, and a chunk keeps
   its control bytes and its literal bytes in two runs rather than
   interleaved, so that neither side has to walk a variable length field
   to find where the next one starts:

     blob   := chunk+                       (empty input -> empty blob)
     chunk  := hdr[2] ctl[ctl_sz] lit[...]
     hdr    := u16le: ctl_sz<<7 | (rem-1)<<1 | last
     ctl    := group+                       (covers this chunk's blocks)
     lit    := the literal bytes of this chunk's blocks, in order

   rem is the size of the blob's very last block (1..64) and last is set
   on the chunk that holds it.  A chunk takes blocks until its control
   field would not hold another span of them (ctl_sz reaches 200, and a
   span adds at most 288) or its 2048 byte literal run is full, so
   ctl_sz<=488 and the three fields fit one u16.

   A group describes one block, or a run of trivial blocks.  Its first
   byte c selects the form:

     c&0x40==0  position list: c and the bytes that follow are byte
                positions inside the block, bit 7 set on the last one.
                Positions are strictly increasing and alternate: the
                first starts a zero run, the second ends it, and so on.
                The block's literals are the bytes outside those runs,
                read from lit in order.  At most 4 positions (below).
     c==0x7F    mask mode: the next 8 bytes are a little endian bit mask
                of the block's literal bytes.
     0x40..0x7E (c-0x3F) blocks, 1..63, that are all literal.
     0xC0..0xFF (c-0xBF) blocks, 1..64, that are all zero.

   A zero run is coded only if it is at least 2 bytes long and lies
   entirely inside its block; single zeros, and the 1 byte tails of runs
   that straddle a block boundary, stay literals.  The coder switches to
   mask mode once a block has 5 transitions, so a position list is never
   longer than 4 bytes and a group never longer than 9.  The mask costs
   9 bytes against a list's 5 to 8 -- 0.2% of the output -- and buys a
   group whose length neither side has to find by scanning.

   The block is the unit of work on both sides: the encoder advances the
   input by 64 bytes per block and the decoder advances the output by 64
   bytes per block, while the other cursor advances by a popcount of the
   block's mask, never by a length read out of the stream.  That is
   where the speed comes from: the dense shapes carry no serial chain
   through the data at all. */

#define ZLE14_SPAN (2048UL) /* literal bytes a chunk can hold */
#define ZLE14_MASK (   5UL) /* transitions at which mask mode wins */

/* zle14_iota is the byte index of every lane, the source the encoder
   compresses a transition mask down into a position list from. */

static uchar const zle14_iota[ 64 ] __attribute__((aligned(64))) = {
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
  16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
  32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
  48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
};

ulong
fd_zle14_compress( void *       FD_RESTRICT comp,
                   void const * FD_RESTRICT data,
                   ulong                    data_sz ) {

  /* The chunk's literals are built here and copied out at the chunk's
     end: 2048 of span, 64 for the block stores that write a whole
     block's worth regardless of the mask, and 64 for the short last
     block, which the span does not budget for. */

  uchar lit_buf[ ZLE14_SPAN+128UL ] __attribute__((aligned(64)));

  uchar const * src;
  uchar const * src_end;
  uchar const * blk_end;  /* the span's last block boundary */
  uchar *       out;      /* control cursor */
  uchar *       ctl;      /* this chunk's control start */
  uchar *       lit;      /* literal scratch cursor */
  __m512i       iota;
  __m512i       v;        /* the block's bytes */
  ulong         z, r2, b, T, keep, t, span, hdr_last;

  if( FD_UNLIKELY( !data_sz ) ) return 0UL;

  src      = (uchar const *)data;
  src_end  = src + data_sz;
  out      = (uchar *)comp;
  iota     = _mm512_load_si512( zle14_iota );
  hdr_last = ( ((data_sz-1UL)&63UL)<<1 ) | 1UL; /* (rem-1)<<1 | last */

chunk:
  out += 2UL;                                   /* the chunk header */
  ctl  = out;
  lit  = lit_buf;
  span = fd_ulong_min( (ulong)(src_end-src) & ~63UL, ZLE14_SPAN );
  goto span_short;

span_next:                                      /* a later span of the same chunk */
  {
    ulong avail = (ulong)(src_end-src) & ~63UL;         /* whole blocks left */
    ulong room  = (ulong)( lit_buf+ZLE14_SPAN - lit );  /* literal scratch still free */
    span = avail<=room ? avail : ( room & ~63UL );
  }

span_short:
  blk_end = src + span;
  if( FD_UNLIKELY( src>=blk_end ) ) goto tail;  /* no full block left */

blk:                                            /* a full block at src */
  v  = _mm512_loadu_si512( src );
  z  = (ulong)_mm512_testn_epi8_mask( v, v );   /* zero bytes */
  r2 = z & (z>>1);                              /* 2 zeros start here */
  if( FD_UNLIKELY( !r2 ) ) goto alllit;

blk_r2:                                         /* r2 nonzero, v the block */
  b = r2 | (r2<<1);                             /* byte is in a coded run */
  if( FD_UNLIKELY( b==~0UL ) ) goto allzero;
  T    = b ^ (b<<1);                            /* transitions */
  keep = ~b;
  t    = (ulong)fd_ulong_popcnt( T );
  if( t>=ZLE14_MASK ) {                         /* mask mode */
    out[0] = (uchar)0x7F;
    FD_STORE( ulong, out+1, keep );
    out += 9UL;
  } else if( t==1UL ) {
    /* One zero run, reaching the end of the block: one control byte,
       and the literals are the block's own prefix, so neither needs a
       compress. */
    ulong p0 = (ulong)fd_ulong_find_lsb( T );
    *out++ = (uchar)( p0 | 0x80UL );
    _mm512_storeu_si512( lit, v );
    lit += p0;
    goto blk_next;
  } else {
    _mm512_mask_compressstoreu_epi8( out, (__mmask64)T, iota );      /* the positions */
    out[ t-1UL ] = (uchar)( (ulong)fd_ulong_find_msb( T ) | 0x80UL );       /* terminate the list */
    out += t;
  }
  _mm512_mask_compressstoreu_epi8( lit, (__mmask64)keep, v );
  lit += (ulong)fd_ulong_popcnt( keep );

blk_next:
  src += 64UL;
  if( FD_LIKELY( src<blk_end ) ) goto blk;

blkdone:                                        /* the chunk's span is done */
  if( src>=src_end                       ) goto chunk_end;
  if( (ulong)(src_end-src)<64UL          ) goto tail;       /* a short last block, same chunk */
  if( (ulong)(out-ctl)>=200UL            ) goto chunk_end;  /* a span adds at most 288 bytes */
  if( lit>=lit_buf+ZLE14_SPAN-63UL       ) goto chunk_end;  /* the literal scratch is full */
  goto span_next;

  /* A block with no coded zero run.  Runs of them share one group, and
     the loop's test doubles as the next block's, so falling out of it
     re-enters the block body with the mask already in hand. */

alllit:
  {
    ulong cnt = 0UL;
    for(;;) {
      _mm512_storeu_si512( lit, v );
      lit += 64UL;
      src += 64UL;
      cnt++;
      if( src>=blk_end ) { *out++ = (uchar)( 63UL+cnt ); goto blkdone; }
      v  = _mm512_loadu_si512( src );
      z  = (ulong)_mm512_testn_epi8_mask( v, v );
      r2 = z & (z>>1);
      if( r2 ) break;
    }
    *out++ = (uchar)( 63UL+cnt );
  }
  goto blk_r2;

  /* An all zero block.  Scan 256 bytes at a time for the end of the
     run, out to the 64 blocks one group can code, regardless of where
     the chunk's span ends -- the run costs the chunk no literal
     scratch. */

allzero:
  {
    uchar const * run = src;
    uchar const * lim = src + fd_ulong_min( (ulong)(src_end-src) & ~63UL, 64UL*64UL );
    src += 64UL;
    while( src+256UL<=lim ) {
      __m512i o = _mm512_loadu_si512( src );
      o = _mm512_or_si512( o, _mm512_loadu_si512( src+ 64UL ) );
      o = _mm512_or_si512( o, _mm512_loadu_si512( src+128UL ) );
      o = _mm512_or_si512( o, _mm512_loadu_si512( src+192UL ) );
      if( _mm512_test_epi8_mask( o, o ) ) break;
      src += 256UL;
    }
    while( src<lim ) {
      __m512i w = _mm512_loadu_si512( src );
      if( _mm512_test_epi8_mask( w, w ) ) break;
      src += 64UL;
    }
    *out++ = (uchar)( 191UL + (ulong)(src-run)/64UL );  /* blocks in the run */
  }
  if( src<blk_end ) goto blk;
  goto blkdone;

  /* The last block, 1..63 bytes.  All zero is tested first: an account
     that ends in zeros is the common case.  The load reads up to
     align_up( data_sz, 64 ). */

tail:
  {
    ulong bs   = (ulong)( src_end-src );
    ulong full = _bzhi_u64( ~0UL, (uint)bs );
    ulong m;

    v = _mm512_loadu_si512( src );
    m = (ulong)_mm512_test_epi8_mask( v, v );   /* nonzero bytes */
    src = src_end;

    if( !(m & full) ) { *out++ = (uchar)0xC0; goto chunk_end; }  /* all zero, any size */

    z  = (~m) & full;
    r2 = z & (z>>1);
    if( !r2 ) {                                 /* no coded zero run */
      *out++ = (uchar)0x40;
      _mm512_mask_compressstoreu_epi8( lit, (__mmask64)full, v ); /* the whole block */
      lit += bs;
      goto chunk_end;
    }

    b    = ( r2 | (r2<<1) ) & full;
    T    = ( b ^ (b<<1) ) & full;
    keep = (~b) & full;
    t    = (ulong)fd_ulong_popcnt( T );
    if( t>=ZLE14_MASK ) {
      out[0] = (uchar)0x7F;
      FD_STORE( ulong, out+1, keep );
      out += 9UL;
    } else {
      _mm512_mask_compressstoreu_epi8( out, (__mmask64)T, iota );
      out[ t-1UL ] = (uchar)( (ulong)fd_ulong_find_msb( T ) | 0x80UL );
      out += t;
    }
    _mm512_mask_compressstoreu_epi8( lit, (__mmask64)keep, v );
    lit += (ulong)fd_ulong_popcnt( keep );
  }

chunk_end:
  {
    ulong ctl_sz = (ulong)( out-ctl );
    ulong h      = ( ctl_sz<<7 ) | ( src>=src_end ? hdr_last : 126UL ); /* 126: rem 64, last 0 */
    ulong lit_sz = (ulong)( lit-lit_buf );

    FD_STORE( ushort, ctl-2, (ushort)h );

    if( lit_sz<=64UL ) {
      if( lit_sz ) _mm512_storeu_si512( out, _mm512_load_si512( lit_buf ) ); /* one block covers most chunks */
    } else {
      for( ulong i=0UL; i<lit_sz; i+=64UL )
        _mm512_storeu_si512( out+i, _mm512_load_si512( lit_buf+i ) );
    }
    out += lit_sz;
  }
  if( src<src_end ) goto chunk;

  return (ulong)( out-(uchar *)comp );
}

/* A group is emitted with sloppy 64 byte stores when it is not the last
   group of its chunk and both cursors have a block of slack.  Otherwise
   it goes through the mask path, which expands the block into one
   register and stores exactly its bytes, so nothing is ever read past
   comp+comp_sz or written past data+data_sz. */

long
fd_zle14_decompress( void *       FD_RESTRICT data,
                     ulong                    data_sz,
                     void const * FD_RESTRICT comp,
                     ulong                    comp_sz ) {

  /* A chunk whose control bytes end within 8 of the blob is run over a
     padded copy: ctl_sz's 511 max, rounded up by a block because the
     copy stores 64 at a time. */

  uchar ctl_buf[ 576 ];

  uchar * const dst = (uchar *)data;
  uchar * const out_end  = dst + data_sz;
  uchar const * const blob_end = (uchar const *)comp + comp_sz;
  __m512i const zero = _mm512_setzero_si512();

  uchar *       out;
  uchar const * p;
  uchar const * ctl;
  uchar const * ctl_end;
  uchar const * lit;
  uchar const * nxt;
  ulong         lit_lim;  /* the literal cursor's sloppy read limit */
  ulong         bs_last;  /* the size of this chunk's last block */
  ulong         w;        /* the 8 control bytes at ctl */
  ulong         c, keep, bs, nlit, cnt, tm1;

  if( FD_UNLIKELY( !comp_sz ) ) return 0L;

  out = dst;
  p   = (uchar const *)comp;

  /* The sloppy paths read up to 128 bytes past the literal cursor.
     When the blob's last byte is at least that far from the end of its
     page, those reads touch no page the blob does not already touch,
     and the cursor only has to stay inside the blob. */

  lit_lim = (ulong)blob_end - ( (((ulong)(blob_end-1)) & 4095UL)>=3968UL ? 128UL : 0UL );

chunk:                                          /* p < blob_end */
  ctl = p+2UL;
  if( FD_UNLIKELY( ctl>blob_end ) ) goto estream;
  {
    ulong h      = (ulong)FD_LOAD( ushort, p );
    ulong ctl_sz = h>>7;
    if( FD_UNLIKELY( !ctl_sz ) ) goto estream;
    lit = ctl + ctl_sz;
    if( FD_UNLIKELY( lit>blob_end ) ) goto estream;
    ctl_end = lit;
    bs_last = (h&1UL) ? ( ((h>>1)&63UL)+1UL ) : 64UL; /* the blob's last block */
  }
  /* the 8 byte control reads must fit */
  if( FD_UNLIKELY( (ulong)ctl_end-120UL > lit_lim ) ) goto ctlcopy;
  w = fd_ulong_load_8( ctl );

grp:                                            /* ctl < ctl_end, w the control bytes at ctl */
  c = w & 0xFFUL;
  if( c & 0x40UL ) goto special;

  /* position list */
  tm1 = ((ulong)_tzcnt_u64( w & 0x8080808080808080UL ))>>3;  /* t-1 */
  w  &= 0x3F3F3F3F3F3F3F3FUL;                   /* a corrupt stream can set bit 6 */
  nxt = ctl + tm1 + 1UL;
  if( FD_UNLIKELY( nxt>=ctl_end                ) ) goto pos_exact; /* the chunk's last group */
  if( FD_UNLIKELY( (ulong)(out_end-out)<128UL  ) ) goto pos_exact; /* output slack */
  if( FD_UNLIKELY( (ulong)lit>lit_lim          ) ) goto pos_exact; /* literal slack */
  ctl = nxt;

  /* Dispatched by hand rather than by a switch: a jump table here is an
     indirect branch on the control stream, which predicts badly.  Two
     zero runs is the common case, so it goes first. */

  if( FD_LIKELY( tm1==2UL ) ) goto t3;
  if( tm1<2UL ) { if( tm1 ) goto t2; goto t1; }
  if( tm1==3UL ) goto t4;
  bs = 64UL;                                    /* only a corrupt stream */
  goto pos_keep;

t3: { /* two zero runs, the second reaching the end of the block */
    ulong   p0 = w & 0xFFUL, p1 = (w>>8) & 0xFFUL, p2 = (w>>16) & 0xFFUL;
    __m512i v0 = _mm512_loadu_si512( lit );
    __m512i v1 = _mm512_loadu_si512( lit+p0 );
    w = fd_ulong_load_8( ctl );                 /* the next group, early */
    _mm512_storeu_si512( out,    v0   );
    _mm512_storeu_si512( out+p0, zero );
    _mm512_storeu_si512( out+p1, v1   );
    _mm512_storeu_si512( out+p2, zero );
    lit += p0 + p2 - p1;
    out += 64UL;
    goto grp;
  }

t1: { /* one zero run reaching the end of the block */
    ulong p0 = w & 0xFFUL;
    w = fd_ulong_load_8( ctl );
    _mm512_storeu_si512( out,    _mm512_loadu_si512( lit ) );
    _mm512_storeu_si512( out+p0, zero );
    lit += p0;
    out += 64UL;
    goto grp;
  }

t2: { /* one zero run inside the block */
    ulong   p0 = w & 0xFFUL, p1 = (w>>8) & 0xFFUL;
    __m512i v0 = _mm512_loadu_si512( lit );
    __m512i v1 = _mm512_loadu_si512( lit+p0 );
    w = fd_ulong_load_8( ctl );
    _mm512_storeu_si512( out,    v0   );
    _mm512_storeu_si512( out+p0, zero );
    _mm512_storeu_si512( out+p1, v1   );
    lit += p0 + 64UL - p1;
    out += 64UL;
    goto grp;
  }

t4: { /* two zero runs inside the block */
    ulong   p0 = w & 0xFFUL, p1 = (w>>8) & 0xFFUL, p2 = (w>>16) & 0xFFUL, p3 = (w>>24) & 0xFFUL;
    __m512i v0 = _mm512_loadu_si512( lit );
    __m512i v1 = _mm512_loadu_si512( lit+p0 );
    ulong   k;
    w = fd_ulong_load_8( ctl );
    _mm512_storeu_si512( out,    v0   );
    _mm512_storeu_si512( out+p0, zero );
    _mm512_storeu_si512( out+p1, v1   );
    _mm512_storeu_si512( out+p2, zero );
    k = ( p0 + p2 - p1 ) & 63UL;                /* a corrupt stream can wrap it */
    _mm512_storeu_si512( out+p3, _mm512_loadu_si512( lit+k ) );
    lit += k + 64UL - p3;
    out += 64UL;
    goto grp;
  }

  /* A position list emitted exactly, when the sloppy paths have no
     room.  The literal mask is the xor of ( ~0 << p ) over the list,
     complemented.  Five or more positions: only a corrupt stream gets
     here, since the coder switches to mask mode at five transitions. */

pos_exact:                                      /* nxt = control after the group */
  bs  = nxt<ctl_end ? 64UL : bs_last;           /* the chunk's last group */
  ctl = nxt;

pos_keep:
  {
    ulong n = fd_ulong_min( tm1+1UL, 8UL );
    ulong m = 0UL;
    for( ulong i=0UL; i<n; i++ ) m ^= ~0UL<<( (w>>(8*i)) & 63UL );
    keep = ~m;
  }
  goto emit;

  /* Special groups.  One all zero block and one all literal block are
     half of all groups, so they get their own short paths. */

special:
  if( c==0xC0UL ) goto z1;
  if( c> 0xC0UL ) goto zrun;
  if( c==0x40UL ) goto l1;
  if( c==0x7FUL ) goto mask;

lrun:
  cnt = ( c-0x3FUL )<<6;                        /* literal bytes */
  nxt = ctl+1UL;
  ctl = nxt;
  if( FD_UNLIKELY( nxt>=ctl_end ) ) cnt += bs_last-64UL; /* the blob's last block may be short */
  if( FD_UNLIKELY( out+cnt>out_end  ) ) goto espace;
  if( FD_UNLIKELY( lit+cnt>blob_end ) ) goto estream;
  {
    ulong whole = cnt & ~63UL;
    ulong i     = 0UL;
    for( ; i+256UL<=whole; i+=256UL ) {
      __m512i v0 = _mm512_loadu_si512( lit+i      );
      __m512i v1 = _mm512_loadu_si512( lit+i+ 64UL );
      __m512i v2 = _mm512_loadu_si512( lit+i+128UL );
      __m512i v3 = _mm512_loadu_si512( lit+i+192UL );
      _mm512_storeu_si512( out+i,       v0 );
      _mm512_storeu_si512( out+i+ 64UL, v1 );
      _mm512_storeu_si512( out+i+128UL, v2 );
      _mm512_storeu_si512( out+i+192UL, v3 );
    }
    for( ; i<whole; i+=64UL ) _mm512_storeu_si512( out+i, _mm512_loadu_si512( lit+i ) );
    if( cnt & 63UL ) {
      __mmask64 m = (__mmask64)_bzhi_u64( ~0UL, (uint)( cnt & 63UL ) );
      _mm512_mask_storeu_epi8( out+whole, m, _mm512_maskz_loadu_epi8( m, lit+whole ) );
    }
    lit += cnt;
    out += cnt;
  }
  goto grp_next;

zrun:
  cnt = ( c-0xBFUL )<<6;                        /* zero bytes */
  nxt = ctl+1UL;
  ctl = nxt;
  if( FD_UNLIKELY( nxt>=ctl_end ) ) cnt += bs_last-64UL;
  if( FD_UNLIKELY( out+cnt>out_end ) ) goto espace;
  {
    ulong whole = cnt & ~63UL;
    ulong i     = 0UL;
    for( ; i+256UL<=whole; i+=256UL ) {
      _mm512_storeu_si512( out+i,       zero );
      _mm512_storeu_si512( out+i+ 64UL, zero );
      _mm512_storeu_si512( out+i+128UL, zero );
      _mm512_storeu_si512( out+i+192UL, zero );
    }
    for( ; i<whole; i+=64UL ) _mm512_storeu_si512( out+i, zero );
    if( cnt & 63UL )
      _mm512_mask_storeu_epi8( out+whole, (__mmask64)_bzhi_u64( ~0UL, (uint)( cnt & 63UL ) ), zero );
    out += cnt;
  }
  goto grp_next;

z1:                                             /* one all zero block */
  nxt = ctl+1UL;
  if( FD_UNLIKELY( nxt>=ctl_end ) ) goto z1_last;
  if( FD_UNLIKELY( (ulong)(out_end-out)<128UL ) ) goto zrun;
  ctl = nxt;
  w   = fd_ulong_load_8( ctl );
  _mm512_storeu_si512( out, zero );
  out += 64UL;
  goto grp;

l1:                                             /* one all literal block */
  nxt = ctl+1UL;
  if( FD_UNLIKELY( nxt>=ctl_end ) ) goto l1_last;
  if( FD_UNLIKELY( (ulong)(out_end-out)<128UL ) ) goto lrun;
  if( FD_UNLIKELY( (ulong)lit>lit_lim          ) ) goto lrun;
  ctl = nxt;
  w   = fd_ulong_load_8( ctl );
  _mm512_storeu_si512( out, _mm512_loadu_si512( lit ) );
  lit += 64UL;
  out += 64UL;
  goto grp;

  /* The same two as the chunk's last group: the block may be short, so
     they store exactly bs_last bytes. */

z1_last:
  ctl = nxt;
  if( FD_UNLIKELY( out+bs_last>out_end ) ) goto espace;
  _mm512_mask_storeu_epi8( out, (__mmask64)_bzhi_u64( ~0UL, (uint)bs_last ), zero );
  out += bs_last;
  goto chunk_next;

l1_last:
  ctl = nxt;
  if( FD_UNLIKELY( out+bs_last>out_end  ) ) goto espace;
  if( FD_UNLIKELY( lit+bs_last>blob_end ) ) goto estream;
  {
    __mmask64 m = (__mmask64)_bzhi_u64( ~0UL, (uint)bs_last );
    _mm512_mask_storeu_epi8( out, m, _mm512_maskz_loadu_epi8( m, lit ) );
  }
  lit += bs_last;
  out += bs_last;
  goto chunk_next;

mask:                                           /* the block's literal mask is in the stream */
  nxt = ctl+9UL;
  if( FD_UNLIKELY( nxt>ctl_end ) ) goto estream;
  keep = fd_ulong_load_8( ctl+1UL );
  ctl  = nxt;
  if( FD_UNLIKELY( nxt==ctl_end ) ) { bs = bs_last; goto emit; } /* the chunk's last group */
  bs = 64UL;
  if( FD_UNLIKELY( (ulong)(out_end-out)<128UL ) ) goto emit;
  if( FD_UNLIKELY( (ulong)lit>lit_lim          ) ) goto emit;
  nlit = (ulong)fd_ulong_popcnt( keep );
  w    = fd_ulong_load_8( ctl );
  _mm512_storeu_si512( out, _mm512_maskz_expandloadu_epi8( (__mmask64)keep, lit ) );
  lit += nlit;
  out += 64UL;
  goto grp;

  /* Emit one block: keep the literal mask, bs the block size.  A full
     block needs no mask on its store and no clip on its keep, which is
     the common case. */

emit:
  if( FD_LIKELY( bs==64UL ) ) {
    if( FD_UNLIKELY( out+64UL>out_end ) ) goto espace;
    nlit = (ulong)fd_ulong_popcnt( keep );
    if( FD_UNLIKELY( lit+nlit>blob_end ) ) goto estream;
    _mm512_storeu_si512( out, _mm512_maskz_expandloadu_epi8( (__mmask64)keep, lit ) );
    lit += nlit;
    out += 64UL;
  } else {                                      /* the blob's last block */
    ulong full;
    if( FD_UNLIKELY( out+bs>out_end ) ) goto espace;
    full  = _bzhi_u64( ~0UL, (uint)bs );
    keep &= full;
    nlit  = (ulong)fd_ulong_popcnt( keep );
    if( FD_UNLIKELY( lit+nlit>blob_end ) ) goto estream;
    _mm512_mask_storeu_epi8( out, (__mmask64)full,
                             _mm512_maskz_expandloadu_epi8( (__mmask64)keep, lit ) );
    lit += nlit;
    out += bs;
  }

grp_next:
  if( FD_UNLIKELY( ctl>=ctl_end ) ) goto chunk_next;
  w = fd_ulong_load_8( ctl );
  goto grp;

chunk_next:
  p = lit;
  if( p<blob_end ) goto chunk;
  return (long)( out-dst );

  /* The control bytes end within 8 of the blob: run the group loop over
     a padded copy instead. */

ctlcopy:
  {
    ulong ctl_sz = (ulong)( ctl_end-ctl );
    ulong i      = 0UL;
    for( ; ctl_sz-i>=64UL; i+=64UL )
      _mm512_storeu_si512( ctl_buf+i, _mm512_loadu_si512( ctl+i ) );
    _mm512_storeu_si512( ctl_buf+i,
                         _mm512_maskz_loadu_epi8( (__mmask64)_bzhi_u64( ~0UL, (uint)( ctl_sz-i ) ),
                                                 ctl+i ) );
    ctl     = ctl_buf;
    ctl_end = ctl_buf + ctl_sz;
    w       = fd_ulong_load_8( ctl );
  }
  goto grp;

espace:
  return FD_ZLE14_ERR_SPACE;
estream:
  return FD_ZLE14_ERR_STREAM;
}

#endif /* FD_HAS_AVX512 */
