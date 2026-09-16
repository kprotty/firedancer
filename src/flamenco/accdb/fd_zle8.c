#include "fd_zle8.h"

#if FD_HAS_AVX512
#include "../../util/simd/fd_avx512.h"
#endif

/* Wire format
   ==========================================================

   A frame is a bare sequence of tokens.  There is no header of any
   kind -- no type byte, no size field, no plane boundary:

     frame := token*
     token := L-field literal[L] [ Z-field ]
     field := b:u8 ; if b==0xFF: b += varint   (LEB128, added to the
                                                0xFF base, fd_zle7's
                                                convention)

   The decoder knows data_sz (the container stores it, as it does for
   every other fd_zle variant) and drives the loop off it:

     o = data; end = data+data_sz;
     while( o<end ) {
       L = get_field();  copy L bytes from the stream to o;  o += L;
       if( o==end ) break;              // terminal token: no Z field
       Z = get_field();  write Z zeros to o;                 o += Z;
     }
     // the stream must be exactly consumed

   Note the two independent ways a frame ends, which together are what
   make the format header-free *and* free of any end marker:

     - the input ends with literals: the final token's L copy brings o
       to end, and its Z field is simply absent (the `if( o==end )
       break` above).  Same terminal-lone-L rule fd_zle7's type.last
       and type.more already use.

     - the input ends with zeros: the final token's Z field brings o to
       end and the `while( o<end )` test ends the frame.  This costs no
       wire bytes at all, which is why an all-zero input encodes as the
       two bytes [ L=0, Z=n ] with no special case in either direction.

   How this differs from fd_zle5/6/7
   ------------------------------------------------------------------

   Those all use a *planar* frame: a header, then every literal byte
   concatenated into one contiguous plane, then the (L,Z) token stream.
   That buys a decoder whose literal copies all come from one linearly
   advancing cursor -- but it costs the encoder a size field for the
   plane (fd_zle7: a fixed 3-byte header holding a 22-bit size), and it
   forces the token stream to be written somewhere provisional and then
   relocated once the plane's real extent is known (fd_zle7's
   zle7_finish_more).  That in turn is what forces fd_zle5/6/7's
   output buffer bound up to 1.5*in_sz: the bound describes where the
   *scratch* token stream sits (dst+3+n), not any output the encoder
   can actually emit -- every one of those encoders bails to a verbatim
   copy long before it could write in_sz+1 bytes.

   Interleaving removes all three.  There is nothing to size, so there
   is nothing to relocate, so the encoder writes strictly forward and
   the bound is a real one (see "Output bound" below).  It also removes
   fd_zle7's 22-bit size cap, which made a >4 MiB literal plane fall
   back to a verbatim copy no matter how much the input's zero runs
   would otherwise have saved.

   The wire cost is 3 bytes less per >64 B frame and 1 byte less per
   <=64 B frame.  That is not noise at account sizes: the most common
   account on chain is a 165 B SPL token account with runs
   { 70 lit, 38 zero, 1 lit, 56 zero }, which is

     fd_zle8:  [70] 70B [38] [1] 1B [56]                    =  74 B
     fd_zle7:  hdr3 + 71B plane + (70,38)(1,56) tokens      =  78 B

   and fd_zle7's own 3 header bytes are 4.1% of that 78.

   What it costs
   ------------------------------------------------------------------

   1. A fully incompressible input costs 1 L field (1..5 bytes) where
      fd_zle7's type.raw costs exactly 1.  For n<255 they are equal; the
      worst case is 5 bytes at n near FD_ZLE8_MAX_SZ.

   2. An all-zero input <=64 B costs 2 bytes ([0][n]) where fd_zle7's
      type.fill costs 1.  Past 64 B fd_zle7 pays 3 header bytes and
      loses this one too.

   3. A decoder cannot pre-parse the control stream independently of
      the literals: finding the next field requires having consumed the
      preceding L literal bytes.  The planar format lets a decoder walk
      the token plane first and learn every output offset without
      touching literal data.  No fd_zle decoder actually does this --
      they are all a serial copy/memset chain -- but it is the one real
      structural advantage the planar layout has, and it is the thing
      to revisit if decode ever gets vectorized ahead of encode.

   Output bound
   ------------------------------------------------------------------

   Let P be input bytes consumed by emitted tokens and W bytes written.
   For a token with L<0xFF and Z<0xFF, W grows by L+2 and P by L+Z, and
   the encoder only emits a run when Z>=ZLE8_MIN_Z==3 (or when the run
   ends the input, where the token has no following L field to pay
   for -- see the emit test in zle8_compress_big), so W-P strictly
   decreases.  A token that escapes a field has L>=0xFF or Z>=0xFF and
   so consumed >=255 input bytes to pay for its <=5 varint bytes; the
   densest escaping shape is L=383,Z=3 repeated, which grows W-P by 1
   per 386 input bytes.  Adding the final lone-L token's <=6 bytes,

     W <= in_sz + in_sz/386 + 6

   FD_ZLE8_COMPRESS_BOUND is in_sz + in_sz/64 + 128, ~24x that margin.
   The encoder additionally range-checks against the bound each token
   (zle8_compress_big's `out+L+16>cap` test) and degrades to a single
   trailing literal token rather than trusting the proof -- one
   perfectly predicted compare, and unlike fd_zle5/6/7's equivalent
   bailout it is unreachable rather than routinely taken.

   MIN_Z
   ------------------------------------------------------------------

   ZLE8_MIN_Z is 3, not fd_zle5/6/7's 2.  Framing an interior run of
   exactly 2 zero bytes costs 2 wire bytes (the run's own Z field, plus
   the L field of the literal segment it splits off, which would not
   exist if the run stayed merged into the literals) to save the 2 zero
   bytes: exactly break-even on ratio, and strictly worse on speed --
   an extra token, an extra unpredictable branch, and one more frame
   for the decoder to serialize on.  MIN_Z=3 is therefore >= MIN_Z=2 on
   both axes.  The exception is a run that *ends* the input: it has no
   following L field, so it costs 1 to save Z, and is worth framing for
   any Z>=2.  Both fd_zle5/6/7 would take the same win. */

#define ZLE8_ESC   (0xFFUL) /* field escape marker: 0xFF then varint( v-0xFF ) */
#define ZLE8_MIN_Z (3UL)    /* shortest interior zero run worth framing -- see above */

/* ==================================================================
   Varint (LEB128) and field codec.  Same shape as fd_zle7's. */

static inline int
zle8_vget( uchar const * s,
           ulong         slen,
           ulong *       pi,
           ulong *       out ) {
  ulong v  = 0UL;
  ulong i  = *pi;
  int   sh = 0;
  for(;;) {
    if( FD_UNLIKELY( (i>=slen) | (sh>42) ) ) return 0;
    uchar b = s[i++];
    v |= (ulong)( b & 0x7f )<<sh;
    if( !(b & 0x80) ) break;
    sh += 7;
  }
  *pi  = i;
  *out = v;
  return 1;
}

static inline ulong
zle8_vput( uchar * p,
           ulong   v ) {
  ulong k = 0UL;
  do {
    p[k] = (uchar)( v & 0x7fUL );
    v >>= 7;
    if( v ) p[k] = (uchar)( p[k] | 0x80U );
    k++;
  } while( v );
  return k;
}

/* zle8_get reads one field (an L or a Z) at *pi.  Returns 0 (leaving
   pi/out untouched) on truncated or overflowing input. */

static inline int
zle8_get( uchar const * s,
          ulong         slen,
          ulong *       pi,
          ulong *       out ) {
  ulong i = *pi;
  if( FD_UNLIKELY( i>=slen ) ) return 0;
  ulong v = (ulong)s[i++];
  if( FD_UNLIKELY( v==ZLE8_ESC ) ) {
    ulong ext;
    if( FD_UNLIKELY( !zle8_vget( s, slen, &i, &ext ) ) ) return 0;
    v = ZLE8_ESC+ext;
  }
  *pi  = i;
  *out = v;
  return 1;
}

/* zle8_put writes one field at cp, the encode-side counterpart of
   zle8_get.  Writes at most 6 bytes (1 marker + a 5-byte varint, which
   covers deltas up to 35 bits, comfortably above FD_ZLE8_MAX_SZ). */

static inline uchar *
zle8_put( uchar * cp,
          ulong   v ) {
  if( FD_LIKELY( v<ZLE8_ESC ) ) { cp[0] = (uchar)v; return cp+1UL; }
  cp[0] = (uchar)ZLE8_ESC;
  return cp+1UL+zle8_vput( cp+1UL, v-ZLE8_ESC );
}

#if FD_HAS_AVX512

/* zle8_copy -- literal copy, byte-exact (never touches d+sz or past
   it).  Transcribed unchanged from fd_zle7's zle7_copy so that a
   fd_zle7 / fd_zle8 benchmark isolates the wire format rather than the
   copy strategy; see its comment there for the (64,256] tier's
   derivation.

   (Future work, and the reason this is a separate helper: because
   fd_zle8's output cursor only ever moves forward and
   FD_ZLE8_COMPRESS_BOUND leaves >=64 B of slack past the largest
   output, the sz<=64 tier could be an *unmasked* 64-byte store -- the
   overrun lands on bytes the next field write or the return length
   makes dead.  That is not available to a planar encoder whose literal
   plane abuts a live token stream.) */

#define ZLE8_ZERO_THRESH (512UL) /* past here fd_memset's ERMS/AVX path wins */

static inline void
zle8_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  if( FD_LIKELY( sz<=64UL ) ) {
    __mmask64 mm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) );
    _mm512_mask_storeu_epi8( (void *)d, mm, _mm512_maskz_loadu_epi8( mm, s ) );
    return;
  }
  if( FD_UNLIKELY( sz>256UL ) ) { fd_memcpy( d, s, sz ); return; }

  ulong tail  = sz-64UL;
  ulong mid_a = fd_ulong_min(  64UL, tail );
  ulong mid_b = fd_ulong_min( 128UL, tail );
  wwb_stu( d,        wwb_ldu( s        ) );
  wwb_stu( d+mid_a,  wwb_ldu( s+mid_a  ) );
  wwb_stu( d+mid_b,  wwb_ldu( s+mid_b  ) );
  wwb_stu( d+tail,   wwb_ldu( s+tail   ) );
}

/* zle8_zero -- the decoder's zero-run writer, transcribed from
   fd_zle.c's fd_zle_zero (see its comment for why the <=256 B tiers are
   straight-line rather than a loop: a loop there folds to a memset PLT
   call, vzeroupper and all, in the middle of the decode loop).  The
   decoder needs this for the same reason the shipping codec does --
   account zero runs are tens of bytes, where a call into memset costs
   more than the store it saves. */

static inline void
zle8_zero( uchar * d,
           ulong   sz ) {
  wwb_t z = wwb_zero();
  if( FD_LIKELY( sz<=64UL ) ) {
    _mm512_mask_storeu_epi8( (void *)d, _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)sz ) ), z );
    return;
  }
  if( FD_LIKELY( sz<=256UL ) ) {
    wwb_stu( d, z );
    if( FD_LIKELY( sz<=128UL ) ) { wwb_stu( d+sz-64UL, z ); return; }
    wwb_stu( d+64UL, z );
    if( sz<=192UL ) { wwb_stu( d+sz-64UL, z ); return; }
    wwb_stu( d+128UL, z );
    wwb_stu( d+sz-64UL, z );
    return;
  }
  if( FD_LIKELY( sz<ZLE8_ZERO_THRESH ) ) {
    ulong k = 0UL;
    do { wwb_stu( d+k, z ); k += 64UL; } while( k+64UL<=sz );
    wwb_stu( d+sz-64UL, z ); /* overlaps, stays in range */
    return;
  }
  fd_memset( d, 0, sz );
}

/* zle8_zmask / zle8_cand / zle8_spill -- the scan primitives, also
   transcribed unchanged from fd_zle7 (which took them from fd_zle5/6;
   see ZLE4_DESIGN.md for the derivation).  Only the emit side below
   is new. */

static inline ulong
zle8_zmask( uchar const * s,
            ulong         p,
            ulong         n ) {
  ulong m = wwb_eq( wwb_ldu( s+p ), wwb_zero() );
  if( FD_LIKELY( p+64UL<=n ) ) return m;
  if( p>=n                  ) return 0UL;
  return m & ( ~0UL>>( 64UL-( n-p ) ) );
}

static inline ulong
zle8_cand( uchar const * s,
           ulong         p,
           ulong         n ) {
  if( FD_LIKELY( p+64UL<n ) ) {
    wwb_t v = wwb_or( wwb_ldu( s+p ), wwb_ldu( s+p+1UL ) );
    return (ulong)_mm512_testn_epi8_mask( v, v );
  }
  ulong m = zle8_zmask( s, p, n );
  return m & ( m>>1 );
}

static inline ulong
zle8_spill( uchar const * s,
            ulong         n,
            ulong *       pp,
            ulong *       pm,
            ulong         len ) {
  ulong p = *pp+64UL;
  for(;;) {
    if( p>=n ) { *pp = p; *pm = 0UL; return len; }

    ulong t = zle8_zmask( s, p, n );
    if( FD_UNLIKELY( t!=~0UL ) ) {
      ulong ext = (ulong)__builtin_ctzll( ~t );
      *pp = p;
      *pm = t & ~( ( 1UL<<ext )-1UL );
      return len+ext;
    }

    p += 64UL; len += 64UL;
    while( p+256UL<=n ) {
      uchar const * q = s+p;
      wwb_t v = wwb_or( wwb_or( wwb_ldu( q       ), wwb_ldu( q+ 64UL ) ),
                        wwb_or( wwb_ldu( q+128UL ), wwb_ldu( q+192UL ) ) );
      if( FD_UNLIKELY( _mm512_test_epi8_mask( v, v ) ) ) break;
      p += 256UL; len += 256UL;
    }
  }
}

/* ==================================================================
   zle8_compress_tiny -- 1<=len<=64, fd_zle_pseudocode3's
   zle_compress_tiny.

   The whole input is one masked 64-byte load, so the zero mask, the
   run-start candidate mask and every run boundary come out of scalar
   bit twiddling on a single `d` with no second pass over memory and no
   spill machinery.  L and Z are both <=64<ZLE8_ESC here, so no field
   can escape and every token is exactly 2 bytes plus its literals --
   zle8_put is never called on this path.

   Four changes from the literal pseudocode:

   1. `d` is masked to bzhi(d,len-1).  `v` is a maskz load, so every
      lane past len reads as a zero byte and the adjacency mask sees
      phantom candidates there.  The pseudocode's `if Zpos>=len: goto
      finish` catches candidates that start at or past len, but not the
      one starting at len-1 (a real trailing zero byte adjacent to the
      first padding byte): that one survives and is framed as a Z=1
      run, which is decodable but costs a byte instead of saving one.
      Masking removes the whole class.

   2. The run-end computation reads `blsr(d+blsi(d))`, not the
      pseudocode's separate `d += blsi(d)` ... `d = blsr(d)`, which
      recomputes tzcnt on a value it has already consumed.  Same
      result; see the trace on ze below.

   3. The pseudocode's `cp = vshufb(v, vaddb(iota,bcast(Lpos)))` cannot
      work as written: VPSHUFB is lane-local (4 independent 16-byte
      lanes), so it cannot shift a 64-byte register down by an
      arbitrary Lpos.  The instruction that can is VPERMB, which needs
      AVX512_VBMI -- not in this project's assumed baseline (see
      config/machine/, and fd_zle_ideas.h's own VBMI2 caveat).  A
      masked reload from src+base costs the same L1 hit, needs no VBMI,
      and drops the iota register and the Lpos->vector dependency
      chain, so that is what this does.

   4. Runs shorter than ZLE8_MIN_Z are skipped rather than framed (see
      the MIN_Z discussion at the top of this file); `base` is left
      alone so the literal segment simply continues through them. */

static ulong
zle8_compress_tiny( uchar *       FD_RESTRICT dst,
                    uchar const * FD_RESTRICT src,
                    ulong                     len ) {
  __mmask64 km = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)len ) );
  __m512i   v  = _mm512_maskz_loadu_epi8( km, src );
  ulong     m  = (ulong)_mm512_cmpeq_epi8_mask( v, _mm512_setzero_si512() );
  ulong     d  = _bzhi_u64( m & ( m>>1 ), (uint)(len-1UL) ); /* change 1 */

  uchar * out  = dst;
  ulong   base = 0UL;

  while( d ) {

    /* d's bits zs..zs+r-1 are set and bit zs+r is clear iff input
       bytes zs..zs+r are zero and byte zs+r+1 is not, so adding
       blsi(d) carries the run away and leaves bit zs+r as x's lowest
       set bit: ze = tzcnt(x)+1 is the run's exclusive end, and
       blsr(x) clears that borrowed bit to leave the candidates above
       the run untouched.  x==0 means the carry left bit 63, i.e. the
       run reaches the padding: tzcnt gives 64, ze clamps to len, and
       d becomes 0 so the loop ends. */

    ulong zs = (ulong)__builtin_ctzll( d );
    ulong x  = d + (ulong)_blsi_u64( d );
    ulong ze = fd_ulong_min( (ulong)__builtin_ctzll( x )+1UL, len );
    d        = (ulong)_blsr_u64( x ); /* change 2 */

    ulong Z = ze-zs;
    if( FD_UNLIKELY( (Z<ZLE8_MIN_Z) & (ze<len) ) ) continue; /* change 4 */

    ulong     L  = zs-base;
    __mmask64 lm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)L ) );
    out[0] = (uchar)L;
    _mm512_mask_storeu_epi8( (void *)( out+1 ), lm,
                             _mm512_maskz_loadu_epi8( lm, src+base ) ); /* change 3 */
    out[ L+1UL ] = (uchar)Z;
    out  += L+2UL;
    base  = ze;
  }

  if( base<len ) { /* terminal lone-L token, no Z field */
    ulong     L  = len-base;
    __mmask64 lm = _cvtu64_mask64( _bzhi_u64( ~0UL, (uint)L ) );
    out[0] = (uchar)L;
    _mm512_mask_storeu_epi8( (void *)( out+1 ), lm,
                             _mm512_maskz_loadu_epi8( lm, src+base ) );
    out += L+1UL;
  }

  return (ulong)( out-dst );
}

/* ==================================================================
   zle8_compress_big -- len>64.  Same scan as fd_zle7's
   zle7_compress_more (absolute byte cursor `p`, carry-propagate run
   resolution, 256 B skip-ahead that resolves the hit word straight out
   of the a0..a3 registers instead of rescanning); the emit side is the
   whole difference.

   fd_zle7's emit maintains two cursors -- a literal plane at
   dst+3 and a token stream provisionally parked at dst+3+n -- and
   finishes by relocating the second onto the end of the first once the
   plane's size is known, then writing that size into a 3-byte header.
   Here there is exactly one cursor, `out`, and the function ends by
   returning out-dst.  No zle8_finish, no size field, no relocation
   copy, no raw fallback, and the dead store traffic to dst+3+n (which
   for a compressible input is a second, entirely separate set of cache
   lines from the literal plane) is gone with it. */

static ulong __attribute__((noinline))
zle8_compress_big( uchar *       FD_RESTRICT dst,
                   uchar const * FD_RESTRICT src,
                   ulong                     n ) {
  uchar * out = dst;
  uchar * cap = dst+FD_ZLE8_COMPRESS_BOUND( n )-16UL;

  ulong   fs  = 0UL; /* start of the literal segment being accumulated */
  ulong   p   = 0UL;
  ulong   m   = 0UL;
  ulong   d   = zle8_cand( src, 0UL, n );

  for(;;) {

    if( !d ) {
      p += 64UL;
      for(;;) {
        if( FD_UNLIKELY( p>=n ) ) goto trailing;
        d = zle8_cand( src, p, n );
        if( d ) break;
        p += 64UL;

        while( p+257UL<=n ) {
          uchar const * q  = src+p;
          wwb_t a0 = wwb_or( wwb_ldu( q       ), wwb_ldu( q+  1UL ) );
          wwb_t a1 = wwb_or( wwb_ldu( q+ 64UL ), wwb_ldu( q+ 65UL ) );
          wwb_t a2 = wwb_or( wwb_ldu( q+128UL ), wwb_ldu( q+129UL ) );
          wwb_t a3 = wwb_or( wwb_ldu( q+192UL ), wwb_ldu( q+193UL ) );
          wwb_t t  = _mm512_min_epu8( _mm512_min_epu8( a0, a1 ), _mm512_min_epu8( a2, a3 ) );
          if( FD_UNLIKELY( _mm512_testn_epi8_mask( t, t ) ) ) {
            d = (ulong)_mm512_testn_epi8_mask( a0, a0 ); if( d ) goto search_done;
            p += 64UL; d = (ulong)_mm512_testn_epi8_mask( a1, a1 ); if( d ) goto search_done;
            p += 64UL; d = (ulong)_mm512_testn_epi8_mask( a2, a2 ); if( d ) goto search_done;
            p += 64UL; d = (ulong)_mm512_testn_epi8_mask( a3, a3 );
            goto search_done; /* nonzero: a0..a2 ruled out, t proved one of the four has it */
          }
          p += 256UL;
        }
      }
    }
    search_done:;

    ulong lo = (ulong)_blsi_u64( d );
    ulong b  = (ulong)__builtin_ctzll( d );
    ulong zs = p+b;   /* run start, absolute */
    ulong x  = d + lo;
    ulong len, ze;
    if( FD_LIKELY( x ) ) {
      ze  = p+(ulong)__builtin_ctzll( x )+1UL;
      len = ze-zs;
      d   = (ulong)_blsr_u64( x );
    } else {
      len = zle8_spill( src, n, &p, &m, 64UL-b );
      ze  = zs+len;
      ulong nn = ( p+64UL<n ) ? ( zle8_zmask( src, p+64UL, n ) & 1UL ) : 0UL;
      d = m & ( ( m>>1 ) | ( nn<<63 ) );
    }

    /* Frame the run if it pays: >=ZLE8_MIN_Z, or any length>=2 when it
       ends the input, since a terminal run's Z field has no following
       L field to pay for.  (len>=2 always -- a candidate bit means two
       adjacent zero bytes -- so no lower guard is needed.)  When it
       doesn't pay, fs is left alone and the literal segment absorbs
       the zeros. */

    if( FD_LIKELY( ( len>=ZLE8_MIN_Z ) | ( ze>=n ) ) ) {
      ulong L = zs-fs;

      /* Unreachable given FD_ZLE8_COMPRESS_BOUND (see "Output bound"),
         but a codec should not have a buffer bound that rests only on
         a proof.  Degrading to the trailing lone-L token below is
         always in bounds: out-dst <= fs + (bound-n) at every token
         boundary, so out+6+(n-fs) <= dst+bound. */
      if( FD_UNLIKELY( out+L+16UL>cap ) ) goto trailing;

      out = zle8_put( out, L );
      zle8_copy( out, src+fs, L ); out += L;
      out = zle8_put( out, len );
      fs  = ze;
    }
  }

trailing:
  if( fs<n ) {
    ulong L = n-fs;
    out = zle8_put( out, L );
    zle8_copy( out, src+fs, L ); out += L;
  }

  return (ulong)( out-dst );
}

#else /* !FD_HAS_AVX512 -- Firedancer requires AVX-512; these stubs only
         exist so the translation unit still builds elsewhere.  A single
         literal token is a valid frame for any input, which is exactly
         the scalar fallback this format wants: no type byte to pick, no
         plane to size. */

static inline void
zle8_copy( uchar *       FD_RESTRICT d,
           uchar const * FD_RESTRICT s,
           ulong                     sz ) {
  fd_memcpy( d, s, sz );
}

static inline void
zle8_zero( uchar * d,
           ulong   sz ) {
  fd_memset( d, 0, sz );
}

static ulong
zle8_compress_tiny( uchar *       FD_RESTRICT dst,
                    uchar const * FD_RESTRICT src,
                    ulong                     len ) {
  uchar * out = zle8_put( dst, len );
  fd_memcpy( out, src, len );
  return (ulong)( out-dst )+len;
}

static ulong
zle8_compress_big( uchar *       FD_RESTRICT dst,
                   uchar const * FD_RESTRICT src,
                   ulong                     n ) {
  return zle8_compress_tiny( dst, src, n );
}

#endif /* FD_HAS_AVX512 */

ulong
fd_zle8_compress( void *       FD_RESTRICT comp,
                  void const * FD_RESTRICT data,
                  ulong                    data_sz ) {
  uchar *       dst = (uchar *)comp;
  uchar const * src = (uchar const *)data;
  if( FD_UNLIKELY( !data_sz      ) ) return 0UL;
  if( FD_LIKELY  ( data_sz<=64UL ) ) return zle8_compress_tiny( dst, src, data_sz );
  return                                    zle8_compress_big ( dst, src, data_sz );
}

/* ==================================================================
   Decoder -- a scalar loop over the tokens, driving fd_zle.c's
   AVX-512 copy/zero helpers rather than fd_memcpy/fd_memset (which at
   account-sized runs cost a PLT call and a vzeroupper per frame -- the
   single biggest decode effect measured on the shipping codec, well
   above anything the wire format does).  One cursor into comp, where
   fd_zle7's type.more decoder needs two (a literal-plane cursor and a
   token cursor, touching two separate sets of cache lines throughout). */

long
fd_zle8_decompress( void *       FD_RESTRICT data,
                    ulong                    data_sz,
                    void const * FD_RESTRICT comp,
                    ulong                    comp_sz ) {
  uchar *       o    = (uchar *)data;
  uchar *       oend = o+data_sz;
  uchar const * s    = (uchar const *)comp;
  ulong         i    = 0UL;

  while( o<oend ) {

    ulong L;
    if( FD_UNLIKELY( !zle8_get( s, comp_sz, &i, &L ) ) ) return FD_ZLE8_ERR_CORRUPT;
    if( FD_UNLIKELY( ( L>(ulong)( oend-o ) ) | ( L>comp_sz-i ) ) ) return FD_ZLE8_ERR_CORRUPT;
    zle8_copy( o, s+i, L ); o += L; i += L;
    if( o==oend ) break; /* terminal token: no Z field */

    ulong Z;
    if( FD_UNLIKELY( !zle8_get( s, comp_sz, &i, &Z ) ) ) return FD_ZLE8_ERR_CORRUPT;
    if( FD_UNLIKELY( Z>(ulong)( oend-o ) ) ) return FD_ZLE8_ERR_CORRUPT;
    zle8_zero( o, Z ); o += Z;
  }

  /* exact consumption: no trailing bytes the frame never claimed */
  if( FD_UNLIKELY( i!=comp_sz ) ) return FD_ZLE8_ERR_CORRUPT;
  return (long)data_sz;
}

char const *
fd_zle8_strerror( long res ) {
  switch( res ) {
  case FD_ZLE8_ERR_SPACE:   return "out of buffer space";
  case FD_ZLE8_ERR_CORRUPT: return "malformed compressed data";
  case 0L:                  return "empty";
  default:                  return res>0L ? "ok" : "unknown";
  }
}
