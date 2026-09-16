# fd_zle: techniques, and why the pseudocode direction underperformed

Written after the `fd_zle2` prototype (see `ZLE_BENCH.md`) measured 5-70%
slower encode than `fd_zle` across every shape in `bench_zle`, despite
implementing the "read src once" idea from `fd_zle_pseudocode`/`fd_zle_ideas.h`
correctly. This doc covers what `fd_zle.c` actually does and why, then uses
that regression to explain which parts of the pseudocode direction are likely
to have the same problem.

Machine: AMD EPYC 9555P (Zen 5 / "Turin" generation), AVX-512 F/BW/CD/DQ/IFMA/VBMI/VL.

## What fd_zle.c does

### Wire format

A stream of frames, each `(token byte, [lit_ext varint], L literal bytes,
[zero_ext varint])`. Token packs `min(L,15)` and `min(Z,15)` into one byte;
either field escapes to a trailing varint if its true value is `>=15`. See
the comment at the top of `fd_zle.c` for the exact bit layout.

### Two-tier dispatch on input size

`fd_zle_compress` picks one of five encoders based on `data_sz`:

- `fd_zle_w1`/`w2`/`w3`/`w4` for `<=64`/`128`/`192`/`256` bytes
- `fd_zle_big` for everything larger

The four small encoders are the same code shape at four compile-time word
counts (`FD_ZLE_WALK`/`FD_ZLE_EXT` macros, unrolled per call site). This
matters because **most real account data is small** — per `bench_zle.c`'s
mainnet fit, spl-token accounts (165B) alone are 615M of 1.08B accounts —
and a fixed compile-time word count means the masks for the whole input live
in registers with zero loop overhead and zero dead work on words past the
end. There's no size check inside the hot path, no loop counter, nothing to
mispredict on the words themselves.

### The zero-byte mask

Every word test is the same primitive: `wwb_eq(wwb_ldu(s+off), wwb_zero())`
→ `vpcmpeqb` against a zero vector, producing a 64-bit mask (bit *i* set iff
byte *i* is zero). Everything downstream operates on this mask, not on the
bytes themselves — the raw bytes only get touched again at literal-copy
time, by design (see below).

### Per-run resolution: a carry-propagate trick, not a loop

`fd_zle_big` finds zero-run start candidates as `d = m & ((m>>1)|(nz<<63))`
— bit *i* set iff bytes *i* and *i+1* are both zero, so for one contiguous
run of length *L* starting at *p*, `d` has bits `[p, p+L-2]` set as one
contiguous span. Resolving the whole run's length is then one addition:

```c
ulong lo = d & (~d+1UL);        /* isolate the lowest candidate bit */
ulong x  = d + lo;              /* carry ripples through the whole span */
len = ctz(x) - b + 1UL;         /* the carry-out lands one past the run's end */
d   = x & (x-1UL);              /* drop that one bit, next run's candidates remain */
```

Adding `1<<p` to a contiguous run of 1-bits `[p, p+L-2]` carries through
all of them and sets bit `p+L-1`. One `ctz` and one subtraction recovers the
exact run length — no scan, no branch, and multiple runs packed into the
same word resolve one at a time without re-deriving `d` per run (`x &
(x-1)` clears exactly the bit this iteration consumed).

### The dominant lever: 256B-at-a-time skip-ahead, in *both* directions

This is the mechanism that makes `serum v3 slab` (14.5KB, 99.9% zero) hit
**320 GB/s** and `raydium clmm ticks` hit **228 GB/s** in the baseline — and
whose absence is exactly what cratered those two shapes by 65-70% in
`fd_zle2`.

`fd_zle_spill` skips through long *zero* runs:

```c
while( (w<<6)+256UL<=n ) {
  wwb_t v = wwb_or( wwb_or(wwb_ldu(p), wwb_ldu(p+64)),
                    wwb_or(wwb_ldu(p+128), wwb_ldu(p+192)) );
  if( _mm512_test_epi8_mask(v,v) ) break;   /* found a nonzero byte somewhere in 256B */
  w += 4UL; len += 256UL;                    /* else the whole 256B was zero: skip it */
}
```

Four loads, three ORs, one mask test — one instruction sequence answers
"is any of these 256 bytes nonzero" for the *entire* block. If the answer
is no, the scan advances 256 bytes for the cost of that one sequence,
independent of how long the actual run is. A 14.5KB zero run costs roughly
14500/256 ≈ 57 of these, not 14500 individual byte tests.

The same shape exists for long *literal* (incompressible) spans, via
`or(p[i],p[i+1])` reduced with `min_epu8` across four 64B lanes: if no
adjacent-zero-pair exists anywhere in 256B, there's no run to find there
either, and the scan skips ahead the same way. This is why `uniform random
8 MiB` (which never compresses at all) still scans at a respectable
25 GB/s in the baseline instead of paying a per-byte cost.

### Copy/zero: masked stores, with a threshold-based libc fallback for zero

`fd_zle_copy` is a 64B-chunk `wwb_stu`/`wwb_ldu` loop with one masked tail
store — always vectorized, no size threshold. `fd_zle_zero` is vectorized
only below `FD_ZLE_ZERO_THRESH` (512B); past that it calls `fd_memset`,
because libc's `rep stosb`/ERMS path beats a hand-rolled AVX-512 loop once
the fixed cost of the store-string setup is amortized over enough bytes.

### The escape bound

`esc_sz = 1 + ext(n) + n` and `lim = dst + esc_sz` are computed once up
front; every frame emission checks `(ulong)o + frame_sz(...) >= lim` before
writing, and bails to a single flat literal frame over the whole input if
so. This is what keeps `FD_ZLE_COMPRESS_BOUND(n) = n + 8` a hard guarantee
without per-frame bookkeeping beyond one comparison.

## The fd_zle2 lesson: what actually costs cycles here

`fd_zle2` implemented the pseudocode's core idea — keep the vector register
already loaded for the zero-byte test, stream it into a scratch buffer, and
never re-read `src` for literal bytes — faithfully and correctly (see
`ZLE_BENCH.md` for the two ordering bugs that took to get there). It was
still 11-70% slower encoding on every single shape in `bench_zle`.

Why: **`bench_zle` is explicitly a cache-hot benchmark** (its own file
header says so), so the "second read" `fd_zle2` avoids was always an L1
hit — on this core that's ~4-5 cycles of latency, fully pipelined, with
throughput for two loads per cycle. A modern out-of-order engine hides a
redundant hot load almost for free; it does not hide a branch mispredict
(~15-20 cycles) or a lost 256-bytes-per-instruction skip-ahead. `fd_zle2`
traded a free load for:

1. Losing the small-input size-tiered dispatch (no `w1`..`w4` equivalent) —
   general-loop fixed overhead dominates every small shape.
2. Losing the 256B skip-ahead in both directions — the exact shapes that
   depend on it (`serum v3 slab`, `raydium clmm ticks`, `raydium amm v4
   pool`) regressed 35-70%.
3. Adding a mandatory store-to-scratch, plus the deferred-write bookkeeping
   (`have_stage`/`stage_v`, re-evaluated every run that crosses a word
   boundary) the correctness fixes required — net new branches and a real
   register, in exchange for eliminating work that cost nothing.

The general lesson: **on this codebase, load count is not the scarce
resource under a cache-hot benchmark; branches and skip-ahead throughput
are.** Any idea whose selling point is "fewer loads" needs to be checked
against that before it's trusted.

## Reading fd_zle_ideas.h / fd_zle_pseudocode against that lesson

Going through the pseudocode's techniques with the above in mind:

- **`sel_`/`tz256`'s inline-asm cmov trick** — aimed squarely at removing a
  branch (the file's own comment: GCC flattens the ternary into a
  mispredicting linear chain). This targets the resource that turned out to
  matter. Worth trying in isolation.

- **`edges4`'s XOR-shift boundary bitmap** — a different way to find run
  boundaries than fd_zle's candidate/carry-propagate trick, similar order of
  work per run. Not obviously faster or slower without measuring; it isn't
  a load-count play, so it doesn't carry `fd_zle2`'s specific failure mode,
  but it also isn't free — it needs the same isolated A/B `fd_zle2` should
  have gotten *before* being wired into a full alternate encoder.

- **`encode_dense`'s `vpcompressb`-into-a-stack-buffer, then vector-subtract
  the boundaries** — this is the same *shape* of idea `fd_zle2` tried
  (materialize into a buffer instead of resolving in registers), and the
  pseudocode file's own comment on `compress_small`'s rejected merged path
  says it directly: *"the vpcompressb round trip through a stack buffer is
  pure loss — the store-to-load forward stall costs more than the handful
  of boundaries it materializes."* That risk doesn't go away just because
  `encode_dense` applies the pattern to bigger blocks instead of small ones
  — it needs the identical scrutiny that sentence already gave the small
  path, and `fd_zle2` is a second, independent data point that "should be
  faster because it does less" doesn't hold on this hardware without
  checking.

- **Branchless varint via `pdep`/`pext`/`bzhi`** — elegant, and correctly
  targets a genuinely branchy, ~30-40%-taken decision per the file's own
  comment. Its own caveat — `pdep`/`pext` are "microcoded and slow on Zen
  1/2" — likely does *not* apply on this EPYC 9555P (Zen 5 has single-cycle
  `pdep`/`pext`), but "likely does not apply" is exactly the kind of
  assumption that just cost `fd_zle2` 29% on the mix. A five-minute isolated
  throughput microbenchmark of the varint pack/unpack alone, on this box,
  before it's built into anything, is cheap insurance.

- **The file's own "MEASURED AND REJECTED" note on the merged
  `compress_small`** (a single path for all *N* in 1..256, instead of four
  templated instantiations) is the right model to generalize: it already
  found that "fewer branches in theory" lost 9-49% in practice once
  measured, for the same underlying reason `fd_zle2` lost — removing a
  templated/unrolled fast path costs more than the mispredict it was meant
  to save.

## Recommendation for the next attempt

Don't build a full `fd_zleN.c` and benchmark it whole against `fd_zle`
again — that's what let `fd_zle2`'s two real wins (deferred-write
correctness aside) get buried under two accidental losses (dispatch,
skip-ahead) that had nothing to do with the idea being tested. Instead,
swap *one* mechanism into the existing `fd_zle_big`/`w1`-`w4` structure at a
time — e.g. replace just the candidate/carry-propagate run resolution with
`edges4`, keeping the skip-ahead and size dispatch untouched — and run
`bench_zle` on that alone. That isolates whether the new trick wins on its
own terms, instead of re-deriving (at higher cost) the lesson this round
already paid for.

## Two isolated experiments, taken up on that recommendation (2026-09-06)

Two follow-ups, both measured in isolation *before* touching
`fd_zle_big`'s real control flow — exactly the discipline the section
above asks for, applied this time instead of just recommended.

### 1. Streaming literal copy into `fd_zle_big` specifically — measured, not built

The `fd_zle2` postmortem blamed the redundant-load premise on cache-hot
`bench_zle`, but that argument is about the *whole* benchmark; it says
nothing about whether some *subset* of `fd_zle_big`'s literal bytes
still has a real redundant load worth removing. That's measurable
directly from what `fd_zle_compress` actually emits, no new encoder
needed: a throwaway tool (`measure_zle`, a `bench_zle` copy that walks
the frame stream fd_zle_compress produces to bucket literal bytes by
run length, mix-weighted) found:

| bucket (fd_zle_big-eligible accounts, sz>256B only) | share of literal bytes |
|---|---|
| ≤15B (fits token nibble) | 33.1% |
| 16-64B (one word) | 25.5% |
| 65-256B (2-4 words) | 41.3% |
| >256B (reaches the 256B skip-ahead) | 0.16% |

So 41% of `fd_zle_big`'s literal bytes sit in exactly the range where
`fd_zle_spill`'s 256B skip-ahead never engages, and every word gets
loaded once for the candidate scan (`fd_zle_zmask`) and reloaded later
by `fd_zle_copy` — a real, non-trivial redundant-load target,
concentrated in shapes like `address lookup table` (94% incompressible,
gets absorbed by MIN_Z=2 filtering into one long literal span) and
`raydium amm v4 info`. This is a materially different situation from
`fd_zle2`'s "everywhere, mostly negligible" claim.

The header-size problem that blocks streaming naturally (you can't
start writing literal bytes until you know how many header bytes
precede them, but you don't know the header size — 1 byte or 1+varint —
until the run closes) does have a clean, wire-format-*compatible* fix:
reserve `max_ext = fd_zle_ext(n)` bytes for the varint up front (already
computed once per `fd_zle_big` call as part of `esc_sz` — the worst
case width for any `L<=n`), write literal bytes optimistically at `o+1`,
and the one time a run's length first crosses 15 mid-scan, shift the
(at most 14) bytes written so far forward by `max_ext` and continue
streaming from the new fixed offset. Padding a varint to a fixed width
with trailing continuation bytes (`0x80`, zero value bits) decodes
identically under the *existing* `fd_zle_vget` — no decoder change
needed. So the format obstacle is solvable.

It wasn't built, because a microbenchmark (`micro_lit`) tested the
mechanism it depends on — replacing two separable passes (a scan loop
that loads-and-discards, then a copy loop that loads-and-stores) with
one fused loop (load, test, store per word) — for exactly the 65-256B
sizes above, and it's uniformly *slower*, not faster:

| size (words) | two-pass | fused/streamed | delta |
|---|---|---|---|
| 65B (2w) | 0.98 ns | 0.96 ns | -1.8% |
| 100B (2w) | 0.88 ns | 1.14 ns | **+29%** |
| 150B (3w) | 1.17 ns | 1.27 ns | +8.8% |
| 200B (4w) | 1.41 ns | 1.78 ns | **+26%** |
| 256B (4w) | 1.42 ns | 1.77 ns | **+25%** |

Two clean, independent loops (one all-loads-and-compares, one
all-loads-and-stores) pipeline better separately than one loop with a
longer per-iteration dependency chain (load → compare → store, serialized)
— the same free-standing-loops-beat-a-fused-loop shape as a classic
loop-fission-vs-fusion tradeoff. Given this, wiring the shift/streaming
scheme into `fd_zle_big`'s actual candidate/spill state machine (itself
real, `fd_zle2`-grade correctness risk: partial words, mid-word run
starts, spill crossing word boundaries) would be paying real
implementation risk for a mechanism that already loses in the best case
for it. **Verdict: dead end, and now dead on evidence instead of on
priors — the format problem was solvable, the arithmetic wasn't there.**

### 2. `tz256`/`sel_`'s branchless 4-word probe — measured against fd_zle_big's actual loop

ZLE_TECHNIQUES.md's first pass flagged this one as correctly targeting
branch mispredicts and "worth trying in isolation." Tried: a
microbenchmark (`micro_branch`) pitted `fd_zle_big`'s actual "!d"
word-advancement loop (`for(;;){ m=zmask(w); d=cand(m,1); if(d) break; w++; }`,
early-exit, branchy) against a faithful `sel_`/`tz256` port that finds
the same first-candidate-bit across the same 4 words branchlessly, with
the candidate position randomized per call (a pool of 4096 pre-filled
256B blocks, cycled unpredictably) specifically so the branch predictor
couldn't just memorize one repeated pattern:

| P(no candidate in 256B) | branchy (early-exit) | branchless (tz256) | delta |
|---|---|---|---|
| 0% | 2.46 ns | 3.68 ns | **+49%** |
| 25% | 2.53 ns | 3.67 ns | +45% |
| 50% | 2.61 ns | 3.68 ns | +41% |
| 90% | 2.71 ns | 3.68 ns | +36% |

Branchless loses everywhere, even fully mispredicted. The reason:
`tz256` always loads and tests all 4 words unconditionally (that's the
point of being branchless), while `fd_zle_big`'s early-exit loop
averages under 4 word-scans whenever the candidate is found before the
last word — and the *savings from not touching the remaining words at
all* beats the misprediction cost of the branch that decides to stop.
`tz256`/`first_zero_run` make sense when you've *already* unconditionally
loaded all 4 words for another reason (e.g. right after the skip-ahead
OR-reduction, which does exactly that) — not as a replacement for a
loop whose whole design point is to avoid touching words it doesn't
need. **Verdict: also a dead end, and a useful correction to the first
pass's take on it** — "targets the right resource" (branches) doesn't
imply "beats a well-written early-exit loop"; that also has to be
measured, which the earlier reading of `fd_zle_ideas.h` didn't yet do.

Both experiments used the recommended methodology (isolate the
mechanism, measure it standalone, don't build the whole encoder) and
both came back negative — which is itself the useful result: it says
the remaining unexplored techniques in `fd_zle_ideas.h` (`edges4`'s
run-boundary bitmap on its own terms, not as a `tz256` proxy; the
`pdep`/`pext` varint) are now the more informative next isolated tests,
since the two most structurally promising-looking ones just failed on
this hardware.

## Third pass: the last two named techniques, plus the escape check (2026-09-07)

Baseline this session: mix enc **15.93 GB/s**, dec 23.0 GB/s (same tree,
same box, `--no-synth`).

### 3. `pdep`/`pext` branchless varint in the emitter — MEASURED, REJECTED

Isolated microbenchmark (`micro_emit`, scratch) of `fd_zle_emit` /
`fd_zle_emit_small` against branchless (full pdep 8-byte store),
branchless-ALU (4-byte store, from `emit_seq_small`), and hybrid
(L branchy, Z branchless) variants, over three run-shape mixes:

| variant | dense-small (Z ~40% esc) | dlmm-like (L never esc, Z ~55%) | L-heavy mix |
|---|---|---|---|
| small branchy (baseline) | 5.00 ns/frame | 2.76 | 8.18 |
| small branchless-ALU | 3.13 | 3.21 | 4.54 |
| small hybrid (L branchy, Z branchless) | **2.44** | 2.52 | 6.02 |

In isolation the hybrid looked like a 10-50% win. **Wired into
`fd_zle_emit_small` it lost — mix 15.93 → 15.41 GB/s, and the small
shapes cratered** (metaplex edition 3.61 → 2.28, tail 1-64 6.75 → 5.39,
pump.fun 4.22 → 3.38). Why the microbench mispredicted reality: its run
mixes had Z ≥ 15 at 40-55%, but the *real* small shapes
(`run_metaplex_edition = {1,8,1,10}`, `run_spl_token_acct`, the tail
bands) almost never escape — Z is 2-14 and L is 1-14. So the branch is
predicted-not-taken essentially always and costs ~0, while the
branchless form pays an unconditional 2-byte store + mask ALU on every
single frame. This is the *fourth* independent confirmation of the same
lesson: **the branch fd_zle chose to keep is the one that predicts; a
"branchless" rewrite only helps the rare shape where it doesn't, and
loses the common one.**

### 4. `edges4` / `vpcompressb` boundary bitmap — MEASURED WHOLE, REJECTED

`fd_zle_ideas.h`'s `zle::compress` *is* a complete alternate encoder
built on `edges4` (XOR-shift run-boundary bitmap) + `vpcompressb` into a
stack `pos[]` + a scalar walk of the offsets — and it already carries
the size-tiered templated small path (`compress_small<FILTER,1..4>`) and
both skip-aheads (`all_zero_256`, `no_zero_pair_256`), so unlike
`fd_zle2` it is a *fair* whole-encoder comparison. Built standalone
(`bench_ideas.cpp`, scratch), round-trip verified on every shape, ratios
identical to fd_zle to 0.01%:

| shape | fd_zle | ideas.h | shape | fd_zle | ideas.h |
|---|---|---|---|---|---|
| metaplex edition | 3.61 | 1.91 (-47%) | metaplex metadata | 15.69 | 7.55 (-52%) |
| spl-token account | 13.61 | 13.42 | raydium amm v4 info | 6.77 | 3.50 (-48%) |
| stake account | 10.40 | 7.09 (-32%) | meteora dlmm bins | 19.33 | 9.90 (-49%) |
| metaplex edition marker | 22.37 | 9.03 (-60%) | openbook v1 queue | 105 | 51 (-51%) |
| address lookup table | 16.10 | 9.42 (-41%) | raydium clmm ticks | 219 | 130 (-41%) |

**Uniformly 20-60% slower, no shape wins.** Toggling the build flags
(`ZLE_SMALL_WALK=1` to use fd_zle's exact walk shape, `ZLE_SMALL_FILTER`,
etc.) doesn't move it (metaplex edition 1.89 with `ZLE_SMALL_WALK`).
So `edges4` *on its own terms* — as the doc's second pass asked for —
is a dead end too: the open/close boundary-pairing state machine plus
the `vpcompressb` store-to-load round trip lose to fd_zle's in-register
`d + (d & -d)` carry-propagate resolution, which needs no materialized
boundary list at all. Every mechanism in `fd_zle_ideas.h` is now
measured and rejected on this hardware.

### The escape check: a real 15-20% cost on frame-heavy shapes, but no safe way found to remove it

`fd_zle_big`'s per-frame `if( o+L+15>=lim && o+frame_sz>=lim ) goto
escape;` (the FD_ZLE_COMPRESS_BOUND guarantee) is **not free**. Deleting
it entirely (measured via `prof`, a scratch single-shape harness on the
real `fd_zle.o`):

| shape | with check | no check |
|---|---|---|
| raydium amm v4 info | 6.74 GB/s | 7.62 (+13%) |
| dlmm-like 237 KB tail | 7.89 GB/s | 9.51 (+21%) |

The cost splits two ways, isolated by replacing `goto escape` with a
side-effect-only `sink++` (branch kept, exit edge gone): the **branch**
alone costs ~half, and the **second loop exit** costs the other half —
the `goto escape` edge blocks clang from folding `fd_zle_spill` and the
256 B skip-aheads into the frame loop (with the edge gone it inlines
them, reassociates the OR-reductions into `vpternlogq`, and tightens the
whole nest).

Attempts to recover it *safely* (the bound must still hold for
adversarial input — a `143 literal / 2 zero` repeat expands past
`n + 8`):

- **cheaper arithmetic** (`slack = lim-15` hoisted; integer `o-dst` vs
  pointer `lim`): no change or slightly worse — it isn't the ALU.
- **incremental `credit = fs-(o-dst)` gate**, skip the check while
  `credit` proves safety: −16% (the per-frame credit update + extra
  loop-carried register cost more than the check).
- **`o >= odanger || L >= 64` gate**: −3%.
- **~4 KiB episodes**, check-free inner loop when `credit >= 192`,
  exact-check loop otherwise (two macro-instantiated loop bodies, no
  exit edge in the fast one): **mix 15.75 (−1%)**. The big tails gained
  (`tail 64K-1M` 7.21 → 7.50, `tail >1M` 8.55 → 8.97) but every
  mid-size shape lost the episode/`wend`/dual-loop overhead, and those
  dominate the mix.
- **`__attribute__((always_inline))` on `fd_zle_spill` only** (keeps the
  check, just forces the inlining the check was blocking): **mix 15.93
  → 16.14 (+1.3%)**, no ratio change, `serum v3 slab` 318 → 249 (a
  skip-ahead-bound shape whose codegen the forced inline happens to
  pessimize — but it's ~0.2% of mix time so it doesn't register).
  Decode wobbles ±1.5% (pure binary-layout noise; `fd_zle_spill` is
  encode-only). This is the only net-positive result and it is
  marginal.

**Verdict:** the escape check is a genuine, correctly-identified
bottleneck (15-20% on the frame-heavy shapes, which are ~35% of mix
encode time via the big tails), but it resists safe removal — the
guarantee it provides has to be paid for somewhere, and every
relocation of the cost tried here lands within noise of break-even on
the mix. `always_inline` on `fd_zle_spill` is a real +1.3% and the
lowest-risk change available, if a small win is wanted; it was left
unapplied pending a call on whether +1.3% justifies touching the file.

### Where this leaves it

Every technique in `fd_zle_ideas.h` / `fd_zle_pseudocode` is now
measured and rejected: streaming literal copy (`fd_zle2`, pass 1),
`tz256`/`sel_` branchless probe (pass 2), the `pdep`/`pext` varint
(pass 3), and the `edges4`/`vpcompressb` encoder whole (pass 3). The
one structural opportunity the passes *surfaced* rather than imported —
the escape check — is real but not cheaply bankable. fd_zle as written
is at a local optimum for this hardware and benchmark; further encode
gains would need either a different wire format (e.g. one whose bound
guarantee doesn't need a per-frame test) or a workload where the
cache-hot assumption breaks.

## fd_zle3: a from-scratch redesign, wire format free (2026-09-07)

New mandate: *"we're free to break the wire format. The goal is just to
make the fastest zero RLE compressor/decompressor possible on an AVX-512
machine."*  Ratio may drift to 50% (from 30%), bound to ~1.5n (from
n+8).  `fd_zle.c` stays pristine; the work lives in `fd_zle3.{c,h}`,
`bench_zle3.c`, `test_zle3.c`.

### Design

**Planar layout.**  `[header][literal plane][control plane]`.  The
control plane (the run-length stream) is a contiguous block with *no
literal bytes interleaved*, so the decoder's inner loop is a tight
parse+emit with nothing in the way, and the plane's extent bounds every
literal read implicitly.

**Frame = 2 bytes.**  `(u8 ll, u8 zl)`, or `(0xFF, u32 ll, u32 zl)` for
a run that overflows a byte.  One lead-byte test per frame covers both
lengths; a run of any length still decodes to a single `memset`.

**No per-frame bound check.**  The encoder emits a stored frame
(`[0x00][raw]`) whenever the compressed form is not smaller than the
input, decided *once* at the end, so fd_zle_big's per-frame
`o+frame_sz>=lim` test (a measured 13-21% of its encode time on
frame-heavy shapes, above) is simply gone.  `FD_ZLE3_COMPRESS_BOUND` is
`3n/2 + 32`.

**Decode = one fused pass.**  Parse a frame, bounds-check it (predicted
not-taken), emit it.  When a frame fits in one 64-byte vector
(`ll+zl <= 64`, the common account case) it is a *single* masked store —
the masked load zero-fills past `ll`, so the trailing zeros are free.
An earlier two-pass "validate the whole control plane, then blast with
zero safety branches" version was 5% slower: the second parse of the
control plane cost more than the per-frame predicted branch it removed.

**Encode = one fused pass.**  fd_zle's size-tiered dispatch
(`w1..w4` / `big`) and its candidate/carry-propagate run resolution and
256-B skip-aheads are kept verbatim; only the emit changed — literal
bytes stream to the literal plane and `(ll,zl)` to the control plane as
the scan finds each run.

### Result (mainnet mix, `bench_zle3 --no-synth`, best of 14; box is noisy downward)

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle  | 30.01% | 15.94 GB/s | 23.07 GB/s |
| fd_zle3 | 31.20% | 16.22 GB/s (+1.8%) | 28.68 GB/s (+24.3%) |

(median over the 14 runs: fd_zle3 16.16 / 28.5, fd_zle 15.91 / 22.95.)

Decode is broadly **+20% to +69%** on every shape from ~600 B upward
(`metaplex metadata` +31%, `meteora damm` +48%, `raydium amm v4 info`
+51%, `tail 4K-16K` +69%, `tail 64K-1M` +55%, `tail >1M` +57%) — the
planar control stream and the fused single-store frame are the levers.
Encode is roughly flat on the mix: the big/frame-heavy shapes gain
5-16%, but the sub-64-byte shapes (`metaplex edition` -31%, `metaplex
token record` -47%, `tail 1-64` -41%) regress because the planar format
has a fixed cost — one masked store to place the control plane after the
literal plane — that dominates a 20-byte account.  Those shapes are
~3% of mix encode time; `spl-token` (615M accounts, ~38% of mix encode
time) is +3%.

Known decode regressions: `tail 65-256` -12% and `spl-token mint nofz`
-8% (many-small-frame shapes where the planar per-frame parse loses to
fd_zle's nibble token) — outweighed in the mix by everything else.

### fd_zle3 sub-experiments that were tried and reverted

- **`vpcompressb` to build the literal plane** (one op/word extracts all
  non-zero bytes) instead of a per-frame `zle3_copy` — needs a separate
  pass over the input because the literal-plane offset isn't known until
  the control plane is sized; the separate pass costs more than the
  per-frame copies it replaces.  Fused copy-from-`src` wins (matches the
  `fd_zle2` finding: a hot re-read is nearly free).
- **Fixed 2-byte control, runs capped at 255** (branchless parse, no
  escape) — shreds the big zero-run shapes: `serum v3 slab`'s one 14 KB
  zero run becomes 58 `(0,255)` frames = 58 tiny memsets instead of one,
  decode -67%.  The escape has to stay.
- **Nibble-packed token** `(min(ll,15)<<4)|min(zl,15)` + ext — better
  ratio and 1 byte/frame for texture-heavy shapes, but the dominant
  `spl-token` has 38-70 byte runs that *always* overflow the nibble, so
  it paid an extra ext byte and two taken branches per frame: mix enc
  15.3 / dec 25.6, both worse.
- **Decode overcopy** (unconditional 64-B stores, exact only near the
  buffer end) — write amplification on small zero runs (a 5-byte zero
  run costs a 64-byte store that the next frame overwrites); `tail
  64K-1M` decode -37%.
- **Block-wise prefix-sum parallel decode** (record every frame's
  output/literal offset, then emit all frame bodies independently) —
  the store→load round trip through the offset arrays costs more than
  the serial `o += ll+zl` chain it breaks; mix dec 17.8 (-30%).
- **4-wide decode unroll** (chain 4 frames' offsets in registers, no
  arrays) — no gain and a small loss.  The decode is throughput-bound
  (parse + store uops), not latency-bound on the `o` chain, so adding
  ILP doesn't help.
- **`zle3_frame` fused-store gated tighter** (`&& ll>=zl`, `&& zl<=24`,
  etc. on top of `ll+zl<=64`) — every extra `&&` cost more than it
  saved; the plain `ll+zl<=64` gate is best.

### Where fd_zle3 stands

A planar, escape-check-free format is a real decode win (+25% on the
mix, far more on the shapes that matter for snapshot load) at a
~neutral encode cost and ~1 pp of ratio.  The remaining decode
regression (`tail 65-256`) and the tiny-account encode regression are
both inherent to putting the control stream in its own plane — the
per-frame parse and the one control-plane placement store are fixed
costs a very small account can't amortize.  Recovering those would mean
a hybrid (interleaved format under ~64 B, planar above) — more code for
a sub-1% mix gain.

## The copy helper was compiling to a memcpy PLT call (2026-09-07) — REAL WIN, applied to fd_zle

Every prior encode pass targeted the escape check or branchless
microtricks.  None looked at how clang lowers `fd_zle_copy` /
`zle3_copy` — the 64-B-chunk `for( ; k+64<=sz; ... )` loop that copies a
frame's literal bytes.  clang's loop-idiom pass folds that loop into a
`call memcpy@PLT`, **even on the 6-70 byte literal runs of the small
account fast paths** (`fd_zle_w2/w3/w4`, `fd_zle3_small`).  In the
middle of an otherwise all-vector function that means `vzeroupper` + a
pile of caller-saved-register spills/reloads + the call itself —
`objdump -dr fd_zle.o` shows 3 such calls in `fd_zle_big`, 3 in
`fd_zle_w2`, 4 in `w3`, 5 in `w4`, and 2 `memset` calls in
`fd_zle_decompress` from `fd_zle_zero`.

Fix: replace the loop with straight-line masked stores, one branch per
64-B chunk, capped at 256 B (`sz>256` still calls `fd_memcpy` — those
are the rare big-literal shapes where the call is amortized and clang
was going to emit it anyway).  Same for `fd_zle_zero` at `sz<=256`.  No
format change, no ratio change.

Alternative considered: an `__asm__ volatile("" : "+r"(k))` barrier on
the loop induction variable, which also makes the loop opaque to
loop-idiom (much smaller diff — keeps the original loop).  It works, but
benched **~2% slower encode / ~4% slower decode** than the straight-line
unroll on this box: the barrier also blocks the `sz>256` copies from
`fd_memcpy`'s ERMS path, and it leaves the decoder's per-frame copy as a
real loop (init + branch per frame) where the predicated `sz<=256`
straight-line form is 1-2 overlapping stores with no loop. Straight-line
kept.

**Result — applied to the shipping `fd_zle.c` (`bench_zle --no-synth`,
clang-19, best-of-8; box noisy downward on encode):**

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle before | 30.01% | 15.9 GB/s | 23.0 GB/s |
| fd_zle after  | 30.01% | 16.9 GB/s (+6%) | 26.9 GB/s (+17%) |

Decode carries the win because `fd_zle_copy`/`fd_zle_zero` are on every
frame of `fd_zle_decompress`: `spl-token account` dec 15.5 → 25.6
(+65%), `tail 65-256` dec 21.3 → 35.4 (+66%), `token-2022` dec 13.2 →
18.4 (+39%), `pumpswap amm` dec 32 → 43, `metaplex metadata` dec 21.4 →
27.7.  Encode is +2-5% on the small shapes (`spl-token` enc 14.1 → 14.8)
and flat on the big ones.  `test_zle` passes; `fuzz_zle_diff` and
`fuzz_zle_decompress` clean at multi-million runs.

The identical fix on `fd_zle3` (as `fd_zle4.c`, a straight copy plus
this change) moves its mix from 16.2 / 28.5 to **17.9 / 31.0 GB/s**
(enc +10%, dec +9%), same 31.20% ratio.  So the two codecs' post-fix
standing is:

| codec | ratio | enc | dec |
|---|---|---|---|
| fd_zle (patched) | 30.01% | 16.9 | 26.9 |
| fd_zle4 = fd_zle3 + fix | 31.20% | 17.9 | 31.0 |

### fd_zle4 dead ends (2026-09-07)

- **Inlining `zle4_frame`'s `tot>64` path** (2-4 straight-line masked
  stores per 64-B output chunk instead of the `zle4_copy` + `zle4_zero`
  calls) — *lost*, mix dec 30.6 → 28.4 and made encode unstable.  The
  decoder loop stays tighter with `zle4_copy`/`zle4_zero` left as
  out-of-line calls; bloating `zle4_frame` inline pessimizes the common
  `tot<=64` single-store path's register allocation and code size.  The
  copy-helper fix wants to *shrink* the helper's body enough to inline,
  not move the whole frame emit inline.

## fd_zle6: independently-escaping tokens, per `fd_zle_pseudocode` (2026-09-11)

`fd_zle_pseudocode` sketched a wire format described as a "sequence of
frames" of `[size:u32, literals, tokens]`, with a token being an
`(L, Z)` pair where *each field escapes to a u32 independently* (unlike
fd_zle5's control frame, which escapes both fields together as soon as
either is >= 0xFF) and `Z` carries a uniform `+2` bias. The pseudocode
itself was incomplete (`compress_tiny`/`small`/`big` bodies are `...`)
and had two corners that don't actually work as literally written:

- `size==0` doubling as the stored-fallback sentinel collides with the
  only legitimate encoding of an all-zero input (its literal-plane size
  really is 0) — a case `bench_zle`'s own `all-zero N` shapes exercise.
- A universal `Z += 2` bias means no token can represent `Z==0`, so
  there is no way to write the trailing-literal-with-no-zero-run frame
  fd_zle5 writes as `(ll, zl=0)`.

Implemented as `fd_zle6.{c,h}` (`bench_zle6.c`, `test_zle6.c`,
`fuzz_zle6_diff.c`, `fuzz_zle6_decompress.c`), keeping fd_zle5's
scanning core, tiny/small/big dispatch (this session also split
fd_zle5's single <=256B "small" tier into a leaner tiny (<64B) path per
the pseudocode's three-way split, since that's the band ZLE4_DESIGN.md
§5.4 already flagged as the planar format's weakest) and
literal-plane-then-slide-down-control-stream mechanics verbatim, and
changing only the token itself. The two gaps above are resolved by (a)
a `0xFFFFFFFF` header sentinel instead of `0`, and (b) letting the
*last* token in the stream omit its `Z` field entirely: the decoder
already knows `data_sz`, so right after copying a token's `L` literal
bytes it can tell whether output is complete and a `Z` field exists at
all, with no wire marker needed. Verified with `test_zle6` and ~50M execs of
`fuzz_zle6_diff` (encoder output against an independent scalar decoder)
plus ~17M of `fuzz_zle6_decompress` (arbitrary-bytes safety), both
clean.

**Result (mainnet mix, `bench_zle`/`bench_zle6 --no-synth`, clang-19,
best of 3, same box):**

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle (current) | 30.01% | 16.85 GB/s | 26.77 GB/s |
| fd_zle6 | 31.67% (+1.66pp) | 17.90 GB/s (+6.2%) | 29.28 GB/s (+9.4%) |

Encode and decode both win on the mix, for the same reason fd_zle4/5
already won: independent-field escaping gives a shorter token in the
common asymmetric case (a big literal span with a short zero run, or
vice versa, costs 6 bytes instead of fd_zle5's 9) at the cost of a
second predicted-not-taken branch per token, and the terminal-token
rule removes a wire byte on any frame that ends without a zero run.

**But ratio is worse than both fd_zle (30.01%) and fd_zle4/5 (31.20%)
— this is a real regression, not just a rounding difference — and it's
concentrated exactly where ZLE4_DESIGN.md §5.4 already predicted a
planar format would hurt.** The fixed 4-byte `u32` header costs far
more, proportionally, on a tiny account than fd_zle5's 2-byte
(mode+`c0`) header: `metaplex edition` (20B) ratio goes 20.00%
(fd_zle) → 50.00% (fd_zle6); `wormhole 1B` 200% → 500%; `tail 1-64`
86.21% → 100.00% (i.e. every ≤64B literal-only input now costs *more*
than storing it raw, because the 4-byte header alone exceeds fd_zle's
whole per-frame token). This isn't the independent-escaping idea's
fault — it's the cost of committing to a `u32` header at all, which
this session's implementation chose for a fixed-width, patch-back-only
header (no branch on header size the way fd_zle4/5's `c0`/`0xFF`
hybrid needs). The mid-to-large shapes that dominate mix *time* (`spl-token
account` 46.06%→47.88%, 615M/1.08B accounts) more than pay for it in
the aggregate mix ratio (+1.66pp), but any per-shape or per-account
accounting would show tiny accounts losing real bytes.

**Where this leaves it:** the core hypothesis — independent-field
escaping beats fd_zle4/5's joint escape — reads as a genuine, if modest,
win on both encode and decode GB/s, consistent enough with the "shorter
common-case token wins" pattern already established in this document to
trust without further isolation. The header-size choice is the
confound: a hybrid header (fd_zle5's `c0`/`0xFF` style, sized to the
actual literal-plane length like fd_zle4/5 already do) instead of a flat
`u32` would very likely recover most or all of the 1.66pp against
fd_zle and probably beat fd_zle4/5's 31.20% too, without touching the
token change measured here — untried in this session, next candidate
if this is picked back up.

### fd_zle6_big: the pseudocode's streaming literal copy, tried directly this time — MEASURED, REJECTED, and two real bugs on the way (2026-09-11)

`fd_zle_pseudocode`'s comment block describes scanning in chunks and,
for every clean chunk, "copy `c` to output frame.literals and continue
scanning" — i.e. stream literal bytes into the output *during* the
scan, rather than fd_zle5/6's deferred bulk `zle6_copy(lp,src+fs,L)` at
each run's close. `ZLE_TECHNIQUES.md`'s own "Pass 1" (this doc, second
section) already measured this exact idea in isolation for the
interleaved format and found it an 8-29% loss; asked directly whether
that verdict holds when wired into `fd_zle6_big` specifically, with the
refinement of issuing the store *before* the compare (decoupling it
from the branch, not gating it on the branch's outcome) rather than
after confirming a word clean.

Implemented directly in `fd_zle6_big`: preemptively `zle6_copy` each
word's bytes to the literal plane immediately after loading it, ahead
of the `zle6_cand` check that says whether a run starts there; the
256B skip-ahead's own `wwb_ldu` loads (already needed for the
clean-pair check) reused as the store's source instead of letting
`zle6_copy` reload the block a second time.

**Two real correctness bugs surfaced getting this right, both caught
by `test_zle6`/`fuzz_zle6_diff`, neither hypothetical:**

1. **Orphaned literal bytes.** The first attempt advanced a single `fs`
   cursor on every eagerly-flushed word. But the wire format only lets
   the *decoder* reach literal-plane bytes via a token's `L` field —
   there's no independent "the literal plane also has this many valid
   bytes" signal. Flushing several words' worth of literal content with
   no run in between, then finally hitting a run, computed `L = zs-fs`
   using an `fs` that eager-flushing had *already* walked up to `zs`
   -- so the token claimed `L=0` while thousands of physically-correct
   bytes sat in the literal plane with nothing referencing them.
   Fixed by splitting into two cursors: `fs` (span start, frozen until
   a token is actually emitted) and `flushed` (how far eager-copying
   has progressed within the span, reset to `fs` each time a token
   closes it) — the token's `L` always uses `fs`, so it always claims
   the whole span regardless of how it was physically written.

2. **Resurrected candidates.** Fixing bug 1's sibling issue (a
   *dropped*-literal variant, where a premature `w++` skipped a
   spill's landing word entirely) by letting the search loop re-examine
   that landing word introduced a worse problem: `zle6_cand` re-scans a
   word from scratch, with no notion that bytes before `flushed` are
   already spoken for by the run just closed. On a word where that run
   ended, its own tail can still pair up under a fresh scan, producing
   a "new" candidate that starts *before* `fs` — caught as `zs < fs` on
   an 8 MiB uniform-random shape, word 20944 in, table stakes for how
   rare this needs real scale to hit. Fixed by unconditionally flushing
   a word's already-known-clean remainder (from the rebuilt mask, no
   rescan) before advancing to the next, genuinely unexamined word, and
   only ever calling `zle6_cand` on words that have never been checked.

Verified after both fixes: `test_zle6` passes; ~155M combined
`fuzz_zle6_diff`/`fuzz_zle6_decompress` executions at the default 4 KiB
cap plus another ~73M at `-max_len=262144` specifically to reach the
word-count where bug 2 lived, all clean.

**Result (mainnet mix, best of 8, same box):**

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle6 (deferred bulk copy, i.e. shipping) | 30.82% | 17.29 GB/s | 28.5 GB/s |
| fd_zle6_big with preemptive per-word store | 30.82% | 16.27 GB/s (**-5.9%**) | 28.5 GB/s (unaffected — decode doesn't touch this) |

**Rejected, reverted to the deferred-copy version.** Same verdict as
Pass 1 and as `fd_zle2`'s postmortem, now confirmed a third time in a
third shape (interleaved format, whole alternate encoder, and now
`fd_zle6_big`'s scan loop specifically): under `bench_zle`'s cache-hot
assumption, a bulk copy sized once per run beats the same bytes moved
via many smaller per-word stores, even with the stores decoupled from
the branch and the redundant reloads eliminated. The fixed per-call
overhead of each extra store (register pressure, a longer dependency
chain per iteration even when the store's address doesn't depend on
the compare) outweighs whatever the decoupling saves. The two bugs
above are the more durable output of this pass: both are exactly the
kind of bookkeeping trap any future streaming-copy attempt on this
codebase — in `fd_zle6` or elsewhere — will hit again if it splits
"where a token says the literal span ends" from "where bytes have
physically been written" without keeping them as strictly separate
cursors.

### fd_zle6 follow-up: the hybrid header, done two ways (2026-09-11)

Picked back up same-day: fold fd_zle5's `c0`/`0xFF` header directly into
the stored/zle flag bit, instead of a separate mode byte or the flat
`u32` above — `b0` bit 7 is the stored flag, bits 6..0 are `c0`: direct
`L` (0..126) in the common case, `0x7F` escapes to 3 more bytes (`L` up
to 2^24-1, comfortably above Solana's 10 MiB `MAX_PERMITTED_DATA_LENGTH`
and every literal plane `bench_zle6` has ever produced). 1 byte in the
common case, same 4-byte worst case as before.

**First cut regressed encode.** The natural way to write this is: scan
writes the literal plane at the worst-case offset `dst+4` exactly as the
flat-header version did (so the scan itself doesn't need to know the
final header size), then at the end, *if* `L < 0x7F`, shift the whole
(literal, control) block left by 3 to close the gap. Correct (verified:
`test_zle6`, ~150M combined fuzz execs across every variant in this
section, zero mismatches), but mix encode dropped to **15.7-15.9 GB/s**
— an 11% loss from the flat header's 17.90, on the branch that now
fires for nearly every small account. Swapping the shift's
implementation (libc `memmove`, correct for any overlap, vs a
hand-rolled straight-line `zle6_shift3` proven safe for this specific
fixed 3-byte left-shift by the same "write footprint never reaches an
unread byte" argument as fd_zle's carry-propagate resolver) made *no*
measurable difference — confirming the cost was the extra pass over the
block, not an uninlined-call artifact (the trap this file's own
memcpy-fold comment already documents for `zle6_copy`). Paying a real
second pass over the data on the *common* branch, to save 3 header
bytes, is a bad trade regardless of how that pass is written.

**Fix: flip which branch pays for the shift.** Write the literal plane
optimistically at `dst+1` (the 1-byte-header offset) from the start;
only the *rare* branch (`L >= 0x7F`) pays a shift, and it shifts
*right* (`zle6_shift3r`, high-to-low, same safety argument mirrored).
For the tiny tier (`n < 64`, so `L < 64` always) that branch is
unreachable — the size band with the most accounts by far never shifts
at all.

**Result (mainnet mix, `bench_zle`/`bench_zle6 --no-synth`, clang-19,
best of 8, same box):**

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle (current) | 30.01% | 16.85 GB/s | 26.77 GB/s |
| fd_zle4/5 (documented) | 31.20% | 17.9 GB/s | 31.0 GB/s |
| fd_zle6, flat `u32` header | 31.67% | 17.90 GB/s | 29.28 GB/s |
| fd_zle6, hybrid header (shift-left, first cut) | 30.82% | 15.89 GB/s | 28.53 GB/s |
| fd_zle6, hybrid header (shift-right, fixed) | **30.82%** | **17.29 GB/s** | **28.5 GB/s** |

The fix recovers essentially all of the shift's encode cost (15.89 →
17.29, back above plain fd_zle) while keeping the ratio win intact
(30.82%, better than fd_zle4/5's 31.20% *and* fd_zle's 30.01% is now the
only thing it doesn't beat, by 0.81pp — down from the flat header's
1.66pp). So fd_zle6 now wins on all three axes against fd_zle (ratio
close, encode +2.6%, decode +6.5%) and wins ratio against fd_zle4/5,
but decode still trails fd_zle4/5's documented 31.0 by about 8% — that
gap is the independent-escape token's second predicted-not-taken branch
per token (flagged as a real cost when the token format was first
measured, above), not something the header layout touches.

## fd_zle7 escape varint: `pdep`/`pext` masked-load decode, isolated (2026-09-13)

fd_zle7's `type.fill` escape (`zle7_vget`, a plain LEB128 byte-loop that
only fires when the fill run hits the encoder's 63-and-escape boundary —
i.e. the input is exactly 64 bytes and all zero) is a varint call site the
third pass didn't isolate-test: that pass measured `pdep`/`pext` on the
*emit* side (`fd_zle_emit`'s L/Z fields, in `fd_zle.c`) and rejected it.
This is the *decode* side, in a different file (`fd_zle7`, not `fd_zle`)
— worth its own measurement rather than assuming the emit-side verdict
transfers.

Candidate: `fd_zle_ideas.h`'s `ZLE_VARINT_MASKED` `varint_get` — masked
8-byte load, `tzcnt` locates the terminator, `pext` gathers the 7-bit
payload groups — transcribed to C in a scratch file (`bench_zle7_vget.c`,
uncommitted, not wired into `fd_zle7.c`):

```c
k = bzhi(~0, avail)              // avail = min(slen-i, 8)
x = vmovq(maskz_load(k, s+i))
e = ~x & 0x8080808080808080
b = tzcnt(e) + 1
m = bzhi(0x7f7f7f7f7f7f7f7f, b)
v = pext(x, m)
i += b/8
```

with the same 7-byte/49-bit cap as `zle7_vget`'s `sh>42` (`nbytes>7` is
rejected to match, so a failure on one side is a failure on the other).

**Correctness:** cross-checked against `zle7_vget` (return value, cursor
advance, decoded value) over 20M structured varints of every length 1-8
(8 must fail on both, matching the shared cap) plus 20M pure-garbage-byte
fuzz including truncated buffers — zero mismatches.

**Throughput** (`bench_zle7_vget`, clang-19, this box, cache-hot, 2^26
calls cycling 64K distinct buffers so it isn't one hot static branch):

| varint length | scalar `zle7_vget` | pext/masked-load | delta |
|---|---|---|---|
| 1 byte | 0.316 ns | 0.889 ns | scalar 3.2x faster |
| 2 byte | 0.525 ns | 0.891 ns | scalar +70% |
| 3 byte | 0.745 ns | 0.889 ns | scalar +19% |
| 4 byte | 0.986 ns | 0.889 ns | pext +10% |
| 5 byte | 1.169 ns | 0.892 ns | pext +24% |
| 6 byte | 1.391 ns | 0.892 ns | pext +36% |
| 7 byte | 1.654 ns | 0.892 ns | pext +46% |
| mixed (length random per call, geometric ~75%/19%/5%/1%…) | 2.145 ns | 0.889 ns | **pext +59%** |

The pext variant's cost is flat regardless of length — no data-dependent
branch, just `tzcnt`/`pext` on a fixed-size masked load. The scalar loop
costs ~0.22 ns/extra byte, matching its branch count, and wins outright on
same-length-repeated benchmarks up to 3 bytes — consistent with every
other branch-vs-branchless result in this file (the branch that predicts
wins). The interesting result is `mixed`: even though the distribution is
75% 1-byte, scalar's loop-exit branch is a single static branch whose trip
count changes on every call, which the predictor can't lock onto — its
per-call cost there (2.145 ns) is *worse than its own fixed 7-byte number*
(1.654 ns), and pext's flat 0.889 ns wins by 59%.

This is the opposite of `fd_zle_ideas.h`'s own comment on this exact idea
("MEASURED 13-20% SLOWER") — that number isn't attributed to this box, and
was almost certainly taken under a same-length-repeated calling pattern
(where the scalar branch predicts perfectly), not the random-length-
per-call pattern real chain data would produce across many small-account
decodes hitting this same branch.

**Not applied.** `zle7_vget` is reachable only from `type.fill`'s `n==63`
escape — an input that is *exactly* 64 bytes and all-zero — so real call
frequency is near zero regardless of which variant wins; this was measured
purely because the mechanism itself (masked-load `pdep`/`pext` varint
*decode*, as opposed to pass 3's *encode*-side rejection) hadn't been
isolated on this hardware yet. If `fd_zle7` grows a hot varint decode path
later (e.g. a real `type.more` big-input scan), the `mixed`-workload
result above — branchless wins once call-to-call length is unpredictable —
is the one to start from, not the same-length-repeated numbers.

## fd_zle_big's run-length arithmetic: dropping a cancelling `b` term (2026-09-13)

User-suggested while reading `fd_zle.c`'s run-resolution snippet: `zs =
(w<<6)+b` and `len = ctz(x)-b+1` are both computed from `b = ctz(d)`,
and the subsequent `fs = zs+len` re-adds what `len` just subtracted --
`fs = (w<<6)+b + (ctz(x)-b+1) = (w<<6)+ctz(x)+1`, which never actually
depends on `b`. The old form spent a data-dependent subtract-then-add
on the loop-carried chain into `fs` recomputing a value that was one
constant-offset add away the whole time. Fix: compute `ze =
(w<<6)+ctz(x)+1` directly (base+index+constant-disp, a single LEA-able
add once `ctz(x)` resolves) and set `fs = ze` with no further
arithmetic; `len` becomes `ze-zs` -- still needed for the `MIN_Z` test
and the emitted Z field, but off the `fs` dependency chain instead of
gating it.

`test_zle` passes. `bench_zle --no-synth`, clang-19, same box, stashed
A/B (baseline vs patched `fd_zle.c` alone, 6 runs each):

| | encode (GB/s) | decode (GB/s) |
|---|---|---|
| baseline | 14.24, 15.98, 16.00, 14.27, 15.96, 15.93 (bimodal) | 23.01, 22.99, 23.16, 19.78, 23.14, 19.78 |
| patched | 16.56, 16.56, 16.43, 16.57, 16.56, 16.57 (tight) | 27.07, 27.11, 27.10, 27.07, 27.14, 27.13 |

Best-of-6 encode: 16.00 → 16.57 GB/s (+3.6%). The more notable effect
is variance: baseline alternates between two clusters (~14.3 and
~16.0) run to run, while the patched version lands in a single tight
band every time -- removing one data-dependent op from the
loop-carried chain apparently also removed whatever made the loop
susceptible to falling into the slower of two codegen/scheduling
states. Decode's shift (~23 → ~27, decode code itself untouched) is
the same pure binary-layout noise this doc has already attributed to
encode-side changes before (see the `always_inline`/`fd_zle_spill`
result above: "Decode wobbles ±1.5%... `fd_zle_spill` is
encode-only").

Follow-up, same session: replaced the `d & (~d+1UL)` / `x & (x-1UL)`
idioms in this loop (both here and in the two equivalent spots in
`fd_zle7_compress_more`/`zle7_compress_small`) with `_blsi_u64`/
`_blsr_u64` directly. `test_zle`/`test_zle7` pass, and best-of-6
`bench_zle --no-synth` mix encode is 16.51-16.59 GB/s -- indistinguishable
from the pre-intrinsic 16.43-16.60 GB/s above. **No measured throughput
change**: clang-19 already pattern-matches both idioms into `blsi`/`blsr`
on this BMI2 target, so the intrinsics are a readability/intent change
(the operation being performed is named instead of spelled out), not a
performance one. Recorded so a future pass doesn't re-measure the same
null result expecting a win.

## fd_zle7 `type.more`: the big-input path, implemented (2026-09-13)

`fd_zle7` was a <=64B-only prototype (`type.raw`/`fill`/`last`, see
`fd_zle_pseudocode2`); `type.more` -- the chained/planar big-input
frame -- was still `todo()`, with `fd_zle7_compress` falling back to a
whole-input `type.raw` frame for any `data_sz>64`. This session
implements it: `zle7_compress_more`, gated behind the same
`data_sz<=64` / `>64` dispatch fd_zle5/6 already use, applying their
exact scan machinery (zero mask, run-start candidate mask,
carry-propagate run resolution -- including the `ze` form above -- and
the 256-byte skip-ahead in both directions) to `fd_zle_pseudocode2`'s
`type.more` wire format instead of fd_zle6's.

Two format properties made this simpler than fd_zle6_big, not harder:

- **The header is fixed-width** (1 type/ext byte + `u16 size_ext` = 3
  bytes, always) instead of fd_zle6's variable c0/0xFF header. The
  literal plane's final address is known before the scan starts, so
  there is no zle6-style "shift the literal plane once the header
  width is known" step at all -- only the (much smaller) token stream
  ever moves, from a provisional slot past the literal plane's
  worst-case extent down to abut its real end, exactly the
  non-overlapping-by-construction slide `zle6_finish` already does for
  its control stream (the same "every in-flight bailout check keeps
  `Lsz+C` a safety margin below `n`" invariant makes the final copy's
  source and destination ranges provably disjoint here too).
- **Tokens escape by delta, not by replacement**: `fd_zle_pseudocode2`
  specifies `if L==0xFF: L += read_varint(...)` (this file's original
  convention, already live in `type.fill`'s `n==63` escape) rather than
  fd_zle6's fixed 4-byte absolute replacement. A shared `zle7_tok_get`
  decode helper (byte, then `zle7_vget` only if that byte is 0xFF) and
  `zle7_emit_l`/`zle7_emit` encode helpers implement one field each,
  used for both the `L` and `Z` fields of every token.

The 22-bit `size` field (`(size_ext<<6)|ext`) caps the literal plane at
2^22-1 (~4 MiB); same idea as fd_zle6's 24-bit cap being "comfortably
above every literal plane this codec has ever produced" but with less
headroom -- an input whose literal plane doesn't fit (only reachable
with several-MiB of incompressible data) falls back to `type.raw`, same
code path as "the more-frame just doesn't win."

Kept identical to fd_zle6: the terminal-lone-`L`-field rule (the token
that completes `data_sz` output may omit its `Z` field, decoder infers
it), `ZLE7_MIN_Z==2`, and the mid-scan bailout-to-raw check used both
for "is this economically worth it" and as the overlap-safety argument
for the final token-stream slide. Not carried over from fd_zle6: the
`+2` Z bias (zle7's Z field is the literal run length, no bias, matching
`type.last`'s existing token semantics) and the fixed-4-byte escape
(delta-varint instead, per `fd_zle_pseudocode2`, see above).

## fd_zle7's `type.more` scan, rewritten from the fd_zle5/6 copy (2026-09-13)

User-driven rewrite: `zle7_compress_more` had been carried over verbatim
from fd_zle5/6's word-index/`words`-bound scan shape. Three changes,
all structural rather than micro-optimizations:

1. **One absolute byte cursor, not a word index plus a separately
   carried word count.** Every helper (`zle7_zmask`, `zle7_cand`,
   `zle7_spill`) now takes/returns an absolute byte position `p`
   directly, instead of a word index `w` that every call site had to
   convert via `w<<6` (fd_zle5/6 do this at least half a dozen times
   per scan) and a `words = (n+63)>>6` bound carried alongside it. The
   end check becomes `p>=n`, in the same units as everything else,
   and `words` disappears as a variable entirely.

2. **The 256B skip-ahead resolves its own answer instead of
   rediscovering it.** The skip-ahead computes `a0..a3` (each word
   OR'd with its shifted neighbor) to answer "is there a run-start
   candidate anywhere in these 256 bytes" via a MIN-reduction. The old
   code, on a "yes," just `break`s and lets the per-word `zle7_cand`
   loop above figure out *which* word by reloading and re-ORing --
   up to 3 of the 4 already-computed words redone, and the word that
   comes up empty first also silently walks into re-triggering the
   *next* 256B group scan, re-touching this group's own tail. Now the
   "yes" branch runs `_mm512_testn_epi8_mask` on `a0`, then `a1`, then
   `a2`, then `a3` (each a single instruction on a register already in
   hand, no reload) and takes whichever is nonzero first -- the
   redundant work is just gone.

3. **`zle7_copy`'s 65-256B tier is branchless.** Replaced the
   `sz>128`/`sz>192` conditional ladder with 4 unconditional
   loads+stores at `{0, mid_a, mid_b, tail}` where `tail=sz-64` and
   `mid_a/mid_b = min(64/128, tail)` -- every offset is provably
   `<=sz-64` by construction (verified by hand for sz spanning every
   boundary in (64,256]), so this never touches a byte at or past
   `d+sz`, and `objdump` confirms clang lowers the two `min`s to
   `cmovb` with no branch. (`fd_zle_pseudocode2`'s own `zle_copy`/
   `zle_zero` sketch this same 4-store shape but with an offset
   expression -- `(len/2)*64` -- that's dimensionally wrong for `len`
   in byte units (it would compute offsets in the thousands for a
   256-byte buffer); this derives the correct offsets for the same
   intent instead of transcribing the pseudocode's version.)

**Correctness:** `test_zle7` passes (unchanged, already covers the
`type.more` scan/spill/skip-ahead/escape-boundary shapes). Re-fuzzed
both harnesses post-rewrite: `fuzz_zle7_diff` ~20M execs,
`fuzz_zle7_decompress` ~3.5M execs, both clean (higher coverage than
the pre-rewrite baseline run -- 126/180 features vs 119/172 -- from the
new branch shapes, as expected, not a regression signal).

**Performance:** built `bench_zle7.c` (mechanical copy of
`bench_zle6.c`, the mainnet-mix harness fd_zle7 didn't have yet). A/B
via a manual before/after swap of just this scan code (fd_zle7.c isn't
git-tracked on this branch, so no stash checkpoint existed --
reconstructed the pre-rewrite version from earlier in this session
instead), `bench_zle7 --no-synth`, clang-19, same box, best of 10 runs
each:

| | mix ratio | encode | decode |
|---|---|---|---|
| old (fd_zle5/6-shaped scan) | 31.38% | 15.36 GB/s | (unchanged code; ~14-16 GB/s either side, pure layout noise) |
| rewritten (this pass) | 31.38% | 16.15 GB/s (+5.1%) | ″ |

Same ratio (the wire format didn't change, only how the encoder finds
the runs), decode untouched (its ~14-17 GB/s spread on both sides of
this change is the same binary-layout noise this doc has already
attributed to unrelated encode-side edits earlier). fd_zle7's decoder
remains the plain scalar path noted in fd_zle7.c's own comment ("not on
the performance-critical path this prototype is actually about yet"),
which is why encode (16.15 GB/s) and decode (~15-16 GB/s) now sit close
to each other here -- unlike fd_zle6's decode consistently beating its
encode -- and is the next obvious place to spend effort if this codec
is picked back up.

