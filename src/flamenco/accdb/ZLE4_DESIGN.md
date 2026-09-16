# fd_zle4 — algorithm and design

`fd_zle4` is a zero‑run‑length codec for Solana account data. It is an
experimental redesign of `fd_zle` that is free to break the wire format
in pursuit of raw encode/decode throughput on AVX‑512. This document
explains how it works, end to end.

Files: `fd_zle4.c` / `fd_zle4.h`, benched by `bench_zle4.c`, tested by
`test_zle4.c`, fuzzed by `fuzz_zle4_diff.c` and
`fuzz_zle4_decompress.c`.

---

## 1. What problem it solves

Solana account data is dominated by zeros. A full mainnet snapshot
(slot 442,351,854, 1.076 B accounts, 343 GB) compresses to ~30 % of its
raw size with nothing more than run‑length coding of the zero bytes. The
non‑zero bytes are effectively incompressible (pubkeys, balances,
fixed‑point fields), so a zero‑RLE codec only has to answer one
question well: **where are the zero runs, and how long are they?**

`fd_zle4` is that codec, specialised hard for the shape of the workload:

* Most accounts are small. `spl-token` accounts alone (165 bytes) are
  615 M of 1.076 B accounts. The 65–256 byte band is 83 % of all
  accounts and 40 % of all bytes. Per‑call fixed overhead matters more
  than asymptotic throughput.
* A handful of account layouts (spl‑token, token‑2022, the Metaplex
  records, the Serum/OpenBook slabs) cover the overwhelming majority of
  bytes. `bench_zle4.c` hand‑writes their run patterns and weights the
  benchmark to reproduce the snapshot's byte and account distribution to
  <0.1 %.
* The machine is AMD EPYC 9555P (Zen 5), AVX‑512
  F/BW/CD/DQ/IFMA/VBMI/VL, built with clang. Firedancer is x86‑64 Linux
  only and assumes TSO; `fd_zle4` assumes AVX‑512 (there is a scalar
  fallback purely so the file builds and fuzzes on other hosts).

It is **not** a general compressor. It never looks for repeats in the
non‑zero bytes, has no entropy stage, and a run of non‑zero bytes is
copied verbatim.

---

## 2. Relationship to `fd_zle`

`fd_zle4` keeps `fd_zle`'s scanning core almost verbatim and throws away
its wire format.

| | `fd_zle` | `fd_zle4` |
|---|---|---|
| layout | **interleaved**: `(token, [ext], L literal bytes, [ext])` frames back to back | **planar**: `[header][literal plane][control plane]` |
| frame | 1 token byte, nibble‑packs `min(L,15)` and `min(Z,15)`, varint escape when a field ≥ 15 | 2 bytes `(u8 ll, u8 zl)`, or 9 bytes `(0xFF, u32 ll, u32 zl)` |
| bound | `in + 8`, guaranteed by a **per‑frame** "would this overflow" test | `3·in/2 + 32`, guaranteed by **one** end‑of‑stream "is it actually smaller" test |
| scanning | zero‑mask + carry‑propagate run resolution + 256‑B skip‑aheads | identical |

The per‑frame bound test in `fd_zle` is a measured 13–21 % of its encode
time on frame‑heavy inputs (half the branch, half the extra loop‑exit
edge that blocks the compiler from inlining the skip‑aheads). Removing
it is the main reason `fd_zle4` exists. The planar layout is the second
reason: it makes the decoder's inner loop a tight parse with no literal
bytes in the instruction stream.

Result on the mainnet‑weighted mix (same box, `--no-synth`, best of 5):

| | ratio | encode | decode |
|---|---|---|---|
| `fd_zle` (upstream) | 30.01 % | 16.5 GB/s | 24.0 GB/s |
| `fd_zle4` | 31.20 % | 18.9 GB/s | 32.1 GB/s |

The ~1.2 pp ratio cost is the price of 2‑byte frames instead of the
nibble token.

---

## 3. Wire format

```
[0]                u8   mode            0 = stored, 1 = zle

mode 0  (stored, the incompressible / tiny fallback):
  [1 .. 1+n)            raw bytes                       comp_sz = n + 1

mode 1  (zle):
  [1]              u8   c0              control-plane size C, or 0xFF
  [2 .. 6)        u32LE C               present only when c0 == 0xFF
  [h .. h+L)           literal plane    every literal byte, packed
  [h+L .. comp_sz)     control plane    C bytes
                       h = 2  (c0 != 0xFF)  or  6  (c0 == 0xFF)
                       L = comp_sz - h - C
```

The decompressed size `n` is **not** stored; the caller passes it out of
band (`data_sz`), exactly as `fd_zle` does. The accdb layer already
knows every account's length.

### 3.1 The two planes

* **Literal plane** — every byte that is *not* part of an encoded zero
  run, concatenated in input order. Nothing else: no lengths, no
  framing.
* **Control plane** — a stream of frames, each describing one
  `(literal run, zero run)` pair in input order:

  ```
  2-byte frame:  (u8 ll, u8 zl)                       ll, zl in 0..254
  9-byte frame:  (0xFF, u32LE ll, u32LE zl)           either value >= 255
  ```

  One test of the lead byte (`ll == 0xFF`?) distinguishes the two. A
  nibble‑packed single‑byte token (`(min(ll,15)<<4)|min(zl,15)` + ext)
  was tried and lost: the dominant `spl-token` layout has 38–70 byte
  runs that always overflow the nibble, so it paid an extra ext byte and
  two taken branches on the most common shape.

Invariants over all frames: `Σ(ll + zl) == n` and `Σ ll == L`.

### 3.2 Why planar

The decoder walks the control plane as a dense little array with the
literal bytes nowhere near it. Its inner loop is: read 2 bytes,
bounds‑check, emit, advance two pointers. No variable‑length token to
parse, no literal bytes to step over, no branch on ext in the common
case. The literal plane is consumed by a second cursor that only ever
moves forward by `ll`.

The trade is a fixed cost the interleaved format does not have: the
control plane has to be *placed* after the literal plane, whose size is
not known until the last frame. See §5.3 and §5.4.

### 3.3 Sub‑`MIN_Z` runs stay literal

`ZLE4_MIN_Z = 2`. A zero run of length 1 costs one literal‑plane byte if
left alone, or 2 control bytes if framed — so the encoder never frames a
1‑byte zero run; that zero goes into the literal plane like any other
byte and the decoder copies it out. (`MIN_Z` also bounds the number of
frames: every frame consumes ≥ 2 input bytes, so ≤ `n/2` frames.)

---

## 4. The scanning core

Everything downstream operates on a **zero mask**, never on the bytes
directly. For a 64‑byte word at offset `off`:

```c
m = wwb_eq( wwb_ldu(s+off), wwb_zero() );   // vpcmpeqb vs 0 -> 64-bit mask
```

`m` bit *i* is set iff byte *i* of the word is zero. `zle4_zmask` also
masks off bits past `n` for the final partial word (and returns 0 for a
word entirely past the end). This much is shared by both size tiers.

From here the two tiers diverge:

* **`fd_zle4_small`** walks each mask directly (`ZLE4_WALK`, §5.1): find
  the lowest zero bit, count the consecutive ones above it, that is a
  run. Simple, and for ≤ 4 words the whole scan lives in registers.
* **`fd_zle4_big`** uses the candidate/carry‑propagate resolver below
  plus the 256‑byte skip‑aheads, because it also has to handle runs that
  span many words and long incompressible stretches efficiently.

### 4.1 Run‑start candidates (`fd_zle4_big`)

`d` bit *i* is set iff bytes *i* and *i+1* are both zero — i.e. *i*
begins a run of ≥ 2 zeros. For one maximal zero run covering bits
`[p, q]` (with `q ≥ p+1`), `d` holds the contiguous span `[p, q-1]`.

The scalar form is `d = m & ((m>>1) | (nz<<63))`, where `nz` is bit 0 of
the *next* word's mask. But `m>>1` is just the zero‑mask of the bytes
starting one further along, so the whole thing is a single unaligned
compare (`zle4_cand`):

```c
wwb_t p = wwb_or( wwb_ldu( s+off ), wwb_ldu( s+off+1 ) );
d       = _mm512_testn_epi8_mask( p, p );     // vptestnmb
```

`or(v, v<<8bytes)` is zero at byte *i* iff `src[off+i]` and
`src[off+i+1]` are both zero; `vptestnmb` turns that into the 64‑bit
mask directly. The cross‑word carry is free — the `+1` load pulls the
next word's low byte into lane 63 — so there is no separate neighbour
mask, no `nz`, and no scalar shift/or/and. `m` itself is now only
materialised on the (rare) spill path. The last word of the input falls
back to the scalar form: the `+1` load would read one byte past the
caller's guaranteed `align_up(n,64)` span.

The literal skip‑ahead (§4.3) already computes the same
`or(ldu(p), ldu(p+1))` quantity, so the scan runs on one primitive.

### 4.2 Carry‑propagate run resolution (`fd_zle4_big`)

Resolving that run's length is one add, no loop, no branch:

```c
ulong lo  = d & (~d + 1UL);          // isolate lowest candidate bit: 1<<p
ulong b   = ctzll(d);                //   p  (for the run's start offset)
ulong x   = d + lo;                  // add 1<<p to the ones-span [p, q-1]
ulong len = popcountll(d ^ x);       // d^x is exactly bits [p, q]  ->  q-p+1
d         = x & (x - 1UL);           // drop bit q; other runs' candidates remain
```

Adding `1<<p` to a run of set bits `[p, q-1]` carries all the way
through and sets bit `q` (one past the run's last byte). `d ^ x` is then
exactly the span `[p, q]` (the run bits, plus the new carry bit, with
everything above `q` unchanged and cancelling), so its popcount is the
run length. Using `popcount(d ^ x)` rather than `ctz(x) - b + 1` keeps
`len` off the `b = ctz(d)` dependency chain. `x & (x-1)` clears exactly
the bit this iteration consumed, so multiple runs packed into one word
resolve one per iteration without re‑deriving `d`.

`ctz(x) - b + 1` was the earlier form and is equivalent; the spill test
below still keys on `x == 0`, which `popcount` leaves untouched. (Doing
the whole thing as `d = (d >> (p+len)) << (p+len)` to lean on `shrx`
looks tighter but loses the `x == 0` spill signal and shifts by 64 when
a run reaches bit 63.)

**Spill.** If the run reaches bit 63, `d + lo` carries out of the word
and `x == 0`. Then `zle4_spill` continues the run through following
words: word‑at‑a‑time while partial, then **256 bytes at a time** while
whole —

```c
while( (w<<6)+256 <= n ) {
  v = OR( ldu(p), ldu(p+64), ldu(p+128), ldu(p+192) );
  if( _mm512_test_epi8_mask(v, v) ) break;   // some nonzero byte in 256 B
  w += 4;  len += 256;                        // else skip the whole block
}
```

A 14 KB zero run (Serum slab) costs ~57 of these OR/test sequences, not
14 000 byte tests. This is why `serum v3 slab` encodes at ~380 GB/s.

### 4.3 Literal skip‑ahead

The mirror image, in the "no candidate in this word" path of
`fd_zle4_big`: if there is no adjacent‑zero‑*pair* anywhere in the next
256 bytes there is no run to find there, so skip it —

```c
a0 = OR( ldu(p),      ldu(p+1)      );   // a0[i]==0  iff  src[p+i]==src[p+i+1]==0
a1 = OR( ldu(p+64),   ldu(p+65)     );
a2 = OR( ldu(p+128),  ldu(p+129)    );
a3 = OR( ldu(p+192),  ldu(p+193)    );
t  = min_epu8( min(a0,a1), min(a2,a3) );
if( _mm512_testn_epi8_mask(t, t) ) break;   // t has a zero -> a pair exists
w += 4;                                       // else no pair in 256 B, skip
```

This keeps `uniform random 8 MiB` (never compresses) scanning at
~25 GB/s instead of paying a per‑byte cost.

---

## 5. Encoder

```c
fd_zle4_compress(comp, data, data_sz):
  data_sz == 0           -> return 0
  data_sz <= 256         -> fd_zle4_small(...)      // 1..4 mask words
  else                   -> fd_zle4_big(...)
```

`data` may be unaligned but the readable span must cover
`align_up(data_sz, 64)` — the scan always loads full 64‑byte words and
`zle4_zmask` masks the tail. Padding bytes never affect the output.

### 5.1 `fd_zle4_small` — the ≤ 256 B fast path

This path handles the band that dominates the workload, so it is
straight‑line with no loop over words and no size check in the hot path.

1. Load 1–4 zero masks `m0..m3` directly into registers; mask the final
   partial word with `tmask`.
2. `switch(words)` into one of four unrolled bodies. Each body runs
   `ZLE4_WALK` once per word, with a per‑word `EXT` argument that
   describes how a run reaching bit 63 continues into the next word(s):

   ```
   words==3:
     ZLE4_WALK(m0,   0, { EXT(m1); if(full) EXT(m2); })
     ZLE4_WALK(m1,  64,   EXT(m2) )
     ZLE4_WALK(m2, 128,   nop )
   ```

3. `ZLE4_WALK(m, wb, EXT)` — for each zero run in mask `m`:

   ```c
   b   = ctzll(m);              // run starts at bit b, absolute offset wb+b
   hi  = m >> b;
   len = ~hi ? ctzll(~hi)       // run length inside this word
             : 64 - b;          // ...or it fills the rest of the word
   clear those bits from m;
   if( b+len == 64 ) EXT;       // extend across the word boundary; may set `full`
   if( len >= MIN_Z ) {
     _L = zs - fs;                          // literal run = gap since last frame end
     zle4_copy(lp, src+fs, _L); lp += _L;   // stream literals to the literal plane
     cp[0] = _L; cp[1] = len; cp += 2;      // 2-byte control frame
     emax  = max(emax, _L, len);
     fs    = zs + len;                      // advance the frame-end cursor
   }
   ```

4. Trailing literal run: if `fs < n`, emit `(n - fs, 0)`.
5. `zle4_copy(lp, ctrl, C)` — the control plane was built in a stack
   buffer `ctrl[256]`; copy it to sit right after the literal plane.
6. Decide: if `emax >= 255` (a run needs the 9‑byte escape — vanishingly
   rare for ≤ 256 B) **or** `2 + L + C >= n + 1` (not actually smaller),
   throw all of the above away and write a **stored** frame instead.
   Otherwise write `mode = 1`, `c0 = C`, return `2 + L + C`.

`fd_zle4_small` never emits a 9‑byte frame — it punts to stored first.
So its control plane is pure 2‑byte frames, and any control plane large
enough to matter (`2 + L + C >= n + 1`) already fails the size check and
becomes a stored frame. When it *doesn't*, `C < 255`, the header is
always 2 bytes, and the literal plane sits at a fixed offset `dst + 2`.

### 5.2 `fd_zle4_big` — everything larger

`__attribute__((noinline))`, so the small path is not polluted by its
register pressure.

* Literal plane grows up from `dst + 6` (a fixed 6‑byte header).
* Control plane grows up from `dst + 6 + n` — past anywhere the literal
  plane could reach — and is **slid down** to abut the literal plane at
  the very end (`zle4_copy(lp, cp0, C)`; `dst < src`, so the forward
  overlap is safe).
* The main loop uses the carry‑propagate resolver (§4.2), `zle4_spill`
  (§4.2), and the literal skip‑ahead (§4.3). For each resolved run of
  `len >= MIN_Z`:

  ```c
  L = zs - fs;
  if( (lp-lp0)+L + (cp-cp0)+10 >= n  ||  cp+12 >= cpx )
      return fd_zle4_stored(dst, src, n);     // can't win, or buffer nearly full
  zle4_copy(lp, src+fs, L); lp += L;
  cp = zle4_emit(cp, L, len);                 // 2- or 9-byte frame
  fs = zs + len;
  ```

* `zle4_emit` writes a 2‑byte frame when `(ll | zl) < 255`, else the
  9‑byte `(0xFF, u32, u32)` form — so a multi‑megabyte zero run stays a
  single frame and decodes to a single `memset`.
* At the end: trailing frame, then the "is it actually smaller" check
  (`6 + L + C >= n + 1` ⇒ stored), then write `mode = 1`, `c0 = 0xFF`,
  `u32 C`, slide the control plane down, return `6 + L + C`.

There is **no per‑frame bound check**. The two guards above are (a) "stop
if we already lost" and (b) "stop if the output buffer is nearly full"
— both rare, both predicted‑not‑taken, neither on the critical path of a
compressible input.

### 5.3 The bound

`FD_ZLE4_COMPRESS_BOUND(n) = 32 + n + n/2`.

While the encoder is building the planar layout in `comp`, the peak
footprint is: literal plane ≤ `n` bytes at `dst+6`, control plane
growing from `dst+6+n`. The `cpx` guard
(`cpx = dst + BOUND(n) - 16`) makes the encoder bail to `stored`
(`n + 1` bytes) before the control plane can cross `BOUND(n)`. So both
the in‑progress planar write and the stored fallback fit inside
`BOUND(n)`. The looseness (`1.5n` vs `fd_zle`'s `n + 8`) is what buys
the "no per‑frame test".

### 5.4 The tiny‑account encode cost

For a 20–40 byte account the planar format's fixed cost — building the
control plane in `ctrl[]` and then the `zle4_copy(lp, ctrl, C)`
placement store — is a real fraction of the whole call, so shapes like
`metaplex edition` (20 B) and `metaplex token record` (32 B) encode
30–45 % slower than `fd_zle`. They are ~3 % of mix encode time, so the
mix still nets +12 %. Recovering them would need a hybrid (interleaved
under ~64 B, planar above); not worth the code.

---

## 6. Decoder

```c
fd_zle4_decompress(data, data_sz, comp, comp_sz):
```

Safe against arbitrary input: no OOB reads on `comp`, no writes past
`data + data_sz`, every input classified `ok` / `ERR_SPACE` /
`ERR_CORRUPT`. Verified by `fuzz_zle4_decompress` (LLVMFuzzer + ASan).

1. `mode = s[0]`. `stored` ⇒ one `memcpy` (checked against `data_sz`).
2. Parse the header: `C = s[1]`; if `C == 0xFF`, `C = u32 s[2..6]`,
   `hdr = 6`. Check `hdr + C <= comp_sz`.
3. Derive the layout:
   ```
   lit_sz = comp_sz - hdr - C
   lp     = s + hdr            lpend = lp + lit_sz     // literal plane
   cp     = lpend              ce    = s + comp_sz     // control plane
   o      = data               oend  = data + data_sz  // output cursor
   ```
4. Walk the control plane:
   ```c
   while( cp < ce ) {
     ll = cp[0];
     if( ll != 0xFF ) { check cp+2<=ce;  zl = cp[1];        cp += 2; }
     else             { check cp+9<=ce;  ll = u32; zl = u32; cp += 9; }
     if( ll > lpend-lp  ||  ll+zl > oend-o ) return ERR_CORRUPT;
     zle4_frame(o, lp, ll, zl);
     o  += ll + zl;
     lp += ll;
   }
   if( o != oend || lp != lpend ) return ERR_CORRUPT;
   return data_sz;
   ```

### 6.1 `zle4_frame` — one masked store in the common case

```c
zle4_frame(o, lp, ll, zl):
  tot = ll + zl;
  if( tot <= 64 ) {                                  // the overwhelmingly common account frame
    ml = bzhi(~0, ll);                               // low ll bits
    mt = bzhi(~0, tot);                              // low tot bits
    _mm512_mask_storeu_epi8(o, mt, _mm512_maskz_loadu_epi8(ml, lp));
    return;
  }
  zle4_copy(o,      lp, ll);
  zle4_zero(o + ll,     zl);
```

For a frame that fits in one 64‑byte vector, the masked load reads `ll`
literal bytes and **zero‑fills positions `[ll, 64)` for free**, and the
masked store writes exactly `tot` bytes. Literal copy + zero fill in one
instruction, no branch on `ll` vs `zl`.

Deliberately *not* extended to `tot > 64` with more inline masked stores:
tried, it cost ~2 GB/s on the mix. Bloating `zle4_frame` pessimises the
common single‑store path's register allocation; the `tot > 64` frames
are better served by the out‑of‑line `zle4_copy` / `zle4_zero` calls,
which keep the decoder loop small.

### 6.2 A two‑pass decoder was slower

An earlier version validated the entire control plane first, then
emitted every frame with zero per‑frame safety branches. It was ~5 %
slower: the second parse of the control plane cost more than the
predicted‑not‑taken per‑frame branch it removed. So decode is a single
fused parse+emit pass.

Also tried and reverted: block‑wise prefix‑sum parallel decode (store→
load round trip through offset arrays costs more than the serial
`o += ll+zl` chain), 4‑wide unroll of the offset chain (decode is
throughput‑bound on parse+store uops, not latency‑bound), unconditional
64‑byte overcopy (write amplification on small zero runs).

---

## 7. `zle4_copy` / `zle4_zero` and the memcpy‑fold trap

These two helpers copy a frame's literal bytes / write its zero run.
They are on **every frame** of both the encoder and the decoder.

The natural implementation is a 64‑byte‑chunk loop plus a masked tail.
clang's loop‑idiom pass folds that loop into `call memcpy@PLT` — **even
on a 6–70 byte `spl-token` literal run**. Inside an otherwise
all‑vector function that means `vzeroupper`, a pile of caller‑saved
register spills/reloads, and the call itself: ~15–20 cycles, roughly
half of a small‑account encode. `objdump -dr fd_zle.o` on the upstream
codec shows 3 such calls in `fd_zle_big`, 3/4/5 in `fd_zle_w2/w3/w4`,
and 2 `memset@PLT` in the decoder.

`fd_zle4` writes them as **straight‑line masked stores, one branch per
64‑byte chunk, capped at 256 bytes** (`sz > 256` still calls
`fd_memcpy` — rare, amortised, and clang would emit the call anyway):

```c
if( sz <= 64  ) { one masked store; return; }
if( sz > 256  ) { fd_memcpy(d, s, sz); return; }
                  full store d[0..64);
if( sz <= 128 ) { masked store d[64..sz); return; }
                  full store d[64..128);
if( sz <= 192 ) { masked store d[128..sz); return; }
                  full store d[128..192);
                  masked store d[192..sz);
```

Every full 64‑byte load is gated by a size test that proves the source
run is at least that long, so there is no over‑read of the compressed
blob on decode. This one change is worth **+6 % encode / +17 % decode**
on the mix — it is also the entire content of the standalone `fd_zle.c`
patch, which gets the same win without the format change.

An `__asm__ volatile("" : "+r"(k))` barrier on the loop induction
variable also defeats the fold with a one‑line diff, but benched ~2–4 %
slower: it forces the `sz > 256` copies off `memcpy`'s ERMS path and
leaves the decoder's per‑frame copy as a real counted loop.

---

## 8. Worked example — `spl-token` account (165 B)

Run pattern `{ 70 literal, 38 zero, 1 literal, 56 zero }` (mint + owner
+ amount, then the delegate / is_native / close_authority COptions,
nearly all unset). `n = 165`, `words = 3`.

### Encode (`fd_zle4_small`, `words == 3`)

```
m0 = zmask(src[0..64))   = 0            (bytes 0..69 are all literal)
m1 = zmask(src[64..128)) : bits 6..43 set (the 38-byte run),
                           bit 44 clear (the 1 literal byte),
                           bits 45..63 set (start of the 56-byte run)
m2 = zmask(src[128..165)) & tmask : bits 0..36 set (rest of the 56-byte run)

WALK(m1, 64):
  run #1: b=6, len=38  -> L = (64+6) - 0 = 70
          zle4_copy(lp, src+0, 70)          -> literal plane += 70
          ctrl <- (70, 38)                  -> fs = 108
  run #2: b=45, len reaches bit 63 -> EXT(m2) adds 37 -> len = 56
          L = (64+45) - 108 = 1
          zle4_copy(lp, src+108, 1)         -> literal plane += 1
          ctrl <- (1, 56)                   -> fs = 165

fs == n, no trailing frame.
L = 71, C = 4, emax = 70.  2 + 71 + 4 = 77 < 166  -> keep.
zle4_copy(lp, ctrl, 4)     // place control plane
dst[0]=1, dst[1]=4         // mode, C
```

Compressed: `77` bytes (`[1][4] | 71 literal bytes | 70,38,1,56`).
Ratio 46.7 %.

### Decode

```
mode=1, C=4, hdr=2, lit_sz = 77-2-4 = 71
lp=s+2 .. s+73 ; cp=s+73 .. s+77 ; o=data .. data+165

frame (70,38): tot=108 > 64  -> zle4_copy(o, lp, 70) [full + masked-6]
                                 zle4_zero(o+70, 38) [masked]
               o += 108 ; lp += 70
frame (1,56):  tot=57  <= 64  -> ONE masked store:
                                 maskz_load(low 1 bit, lp) stored under low 57 bits
               o += 57 ; lp += 1

cp == ce ; o == data+165 ; lp == s+73    -> ok, return 165
```

---

## 9. Performance summary (mainnet‑weighted mix, this box)

| shape class | encode vs `fd_zle` | decode vs `fd_zle` |
|---|---|---|
| `spl-token` / `token-2022` (the bulk) | +6 … +21 % | +24 … +45 % |
| 600 B – 4 KB records (Metaplex, Meteora, tails) | +6 … +21 % | +40 … +70 % |
| giant zero‑run shapes (Serum, Raydium pools) | ±2 % | ±5 % |
| sub‑64 B accounts | **−30 … −50 %** | +10 … +20 % |
| `metaplex edition marker` (282 B, 2 % ratio) | −16 % | −10 % |
| **mix** | **+15 %** | **+34 %** |
| ratio | 31.20 % (vs 30.01 %) | |

The 600 B – 4 KB band gained another ~+8 % encode (address‑lookup‑table
+21 %, Metaplex metadata +11 %, Meteora DAMM +9 %) from the `zle4_cand`
vector candidate mask and the `popcount` run‑length in §4.1 / §4.2.

Decode is a broad win because the planar control loop plus the
single‑store frame emit plus the `zle4_copy`/`zle4_zero` fix all
compound. Encode is up on everything that amortises the planar placement
store and down on the accounts too small to.

---

## 10. Things that were tried and rejected

From `ZLE_TECHNIQUES.md` (full detail there):

* Building the literal plane with `vpcompressb` — needs a separate pass
  because the plane offset isn't known until the control plane is sized;
  the pass costs more than the per‑frame copies.
* Fixed 2‑byte control with runs capped at 255 (branchless parse) —
  shreds the big zero‑run shapes (one 14 KB run becomes 58 tiny
  memsets).
* Nibble‑packed 1‑byte token — the dominant `spl-token` runs always
  overflow the nibble.
* Decode overcopy, block prefix‑sum parallel decode, 4‑wide decode
  unroll, tighter fused‑store gate — all neutral‑to‑negative.
* Inlining `zle4_frame`'s `tot > 64` path — pessimises the common
  single‑store frame.

The one open direction: a hybrid interleaved/planar format that uses
interleaved frames under ~64 B to recover the tiny‑account encode cost.
More code for a sub‑1 % mix gain.
