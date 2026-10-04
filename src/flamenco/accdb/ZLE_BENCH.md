# fd_zle bench baseline

Baseline numbers before starting encoder optimization work. Branch: `ripatel/zle`.

## Machine

- CPU: AMD EPYC 9555P 64-Core Processor
- AVX-512: F, BW, CD, DQ, IFMA, VBMI, VL
- Compiler: clang-19 (GCC 15.2.0 is rejected by `config/machine/native.mk` due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=123002)

## Build & run

```
make -j BUILDDIR=clang CC=clang-19 bench_zle
./build/clang/unit-test/bench_zle
```

## Weighted mix (mainnet-shaped, matches real chain ratio to ~0.1%)

- Ratio: **30.01%** of raw (chain: 30.01%)
- Encode: **16.46 GB/s** (0.0608 ns/B)
- Decode: **20.36 GB/s** (0.0491 ns/B)

## Per-shape (selected)

| case | size | ratio | enc GB/s | dec GB/s |
|---|---|---|---|---|
| metaplex edition | 20 | 20.00% | 3.74 | 3.93 |
| pump.fun curve | 49 | 81.63% | 4.37 | 6.14 |
| metaplex token record | 32 | 34.38% | 8.57 | 9.86 |
| wormhole 1 B | 1 | 200.00% | 0.28 | 0.34 |
| spl-token account | 165 | 46.06% | 14.10 | 15.50 |
| token-2022 account | 170 | 44.71% | 12.94 | 13.18 |
| spl-token mint | 82 | 90.24% | 7.12 | 7.94 |
| spl-token mint nofz | 82 | 56.10% | 7.38 | 16.73 |
| pumpswap amm | 137 | 31.39% | 18.02 | 29.36 |
| stake account | 200 | 66.00% | 10.78 | 8.73 |
| metaplex metadata | 607 | 39.70% | 16.26 | 21.42 |
| metaplex metadata v1.3 | 679 | 32.70% | 17.74 | 20.81 |
| metaplex edition marker | 282 | 2.13% | 23.17 | 40.30 |
| address lookup table | 568 | 94.19% | 16.77 | 20.11 |
| raydium amm v4 info | 752 | 60.24% | 7.01 | 10.35 |
| openbook v1 queue | 3228 | 3.28% | 107.91 | 125.43 |
| meteora damm v2 pool | 1112 | 26.98% | 13.41 | 20.62 |
| raydium amm v4 pool | 2208 | 3.08% | 86.29 | 143.91 |
| serum v3 slab | 14524 | 0.12% | 323.67 | 260.23 |
| meteora dlmm bins | 10136 | 12.95% | 19.94 | 31.47 |
| raydium clmm ticks | 10240 | 0.58% | 206.12 | 239.87 |
| serum v3 queue 64K | 65500 | 0.03% | 176.70 | 196.10 |
| serum v3 slab 64K | 65548 | 0.03% | 179.84 | 194.87 |
| serum v3 queue 256K | 262156 | 0.01% | 186.32 | 215.43 |
| tail 1-64 | 29 | 86.21% | 7.00 | 8.42 |
| tail 65-256 | 140 | 60.00% | 12.88 | 21.29 |
| tail 257-1K | 456 | 48.25% | 9.01 | 13.38 |
| tail 1K-4K | 2175 | 15.91% | 19.82 | 31.01 |
| tail 4K-16K | 7771 | 15.88% | 21.53 | 33.57 |
| tail 16K-64K | 29668 | 25.67% | 9.41 | 11.79 |
| tail 64K-1M | 237237 | 22.48% | 7.45 | 10.91 |
| tail >1M | 1958702 | 26.63% | 8.91 | 10.91 |

## Synthetic bounds (not in the mainnet mix)

| case | size | ratio | enc GB/s | dec GB/s |
|---|---|---|---|---|
| uniform random 8 MiB | 8388608 | 100.00% | 25.38 | 56.21 |
| all-zero 64 B | 64 | 3.12% | 20.23 | 23.31 |
| all-zero 256 B | 256 | 1.17% | 34.86 | 56.35 |
| all-zero 1 KiB | 1024 | 0.29% | 94.20 | 191.87 |
| all-zero 4 KiB | 4096 | 0.07% | 205.98 | 255.41 |
| all-zero 16 KiB | 16384 | 0.02% | 298.33 | 275.73 |
| all-zero 64 KiB | 65536 | 0.01% | 178.10 | 205.30 |
| all-zero 1 MiB | 1048576 | 0.00% | 186.17 | 183.34 |
| all-zero 8 MiB | 8388608 | 0.00% | 136.90 | 101.72 |

## Observations for future encoder work

- Small inputs (<200B, e.g. metaplex edition, token record) are the slowest in GB/s terms — likely per-call fixed overhead (mask setup, `fd_zle_w1`/`w2` dispatch) dominates rather than throughput. These are also extremely common on-chain (spl-token account alone is 615M of 1.08B accounts), so fixed overhead matters more than peak GB/s here.
- The 1-byte case (0.28 GB/s) is the floor — pure call/branch overhead, not representative of steady-state throughput.
- Best throughput cases (raydium clmm ticks, serum slabs/queues, all-zero mid-size) are long runs where `fd_zle_big`'s 256B-skip fast path engages — encode reaches 100-320 GB/s.
- All-zero throughput peaks at 16 KiB (298 GB/s encode) then *drops* at 64 KiB+ (178, 186, 137 GB/s) — worth investigating (cache/TLB effects on `comp`/`in` buffers at MAX_SZ=8MiB, or `fd_zle_zero`'s `FD_ZLE_ZERO_THRESH=512` fallback to `fd_memset` beyond the AVX512 loop).
- `raydium amm v4 info` (752B, 75 dense runs) is the worst realistic ratio/branch-heavy shape at only 7.01 GB/s encode — called out in bench_zle.c comments as "the worst common shape for a run coder, both for ratio and for the unpredictable branch per frame."

## fd_zle2 prototype: result (2026-09-05)

`fd_zle2` (`fd_zle2.c`, benchmarked via `bench_zle2`, built the same way as `bench_zle`) tried avoiding fd_zle_big's second, emit-time reload of literal bytes from `src` by keeping the vector register already loaded for the zero-byte test and streaming it straight into a small (256B) stack scratch buffer as the scan progresses, only touching `src` once per byte. Same wire format, same greedy decisions, ratio matches fd_zle to the same 30.01%.

**Verdict: encode is uniformly slower than fd_zle, from -5% to -70% depending on shape. No case wins.** Decode is unchanged (identical code path, `fd_zle_decompress`).

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle (mix) | 30.01% | 15.94 GB/s | 23.12 GB/s |
| fd_zle2 (mix) | 30.01% | **11.35 GB/s (-29%)** | 22.89 GB/s (noise) |

Worst regressions, fd_zle → fd_zle2 encode GB/s:

| case | fd_zle | fd_zle2 | delta |
|---|---|---|---|
| serum v3 slab | 317.46 | 93.50 | **-70%** |
| raydium clmm ticks | 227.76 | 79.68 | **-65%** |
| all-zero 16 KiB | 250.29 | 97.19 | -61% |
| uniform random 8 MiB (escape path) | 25.20 | 10.82 | -57% |
| all-zero 64 B | 19.61 | 8.54 | -56% |
| metaplex edition | 3.63 | 1.90 | -48% |
| stake account | 10.44 | 5.44 | -48% |
| openbook v1 queue | 104.86 | 58.72 | -44% |
| metaplex token record | 8.31 | 3.71 | -55% |

Best (still losses): metaplex metadata v1.3 -5%, metaplex edition marker -10%, meteora dlmm bins -12%.

**Why it didn't pay off:**

1. `bench_zle`/`bench_zle2` are explicitly cache-hot benchmarks (see the file header) — `src` is already resident, so there was no cold reload to avoid. The extra store-to-scratch-then-reload round trip is pure added work with no corresponding cache miss it's saving here.
2. fd_zle2 has no equivalent of `fd_zle_w1`..`w4` (fd_zle's size-tiered dispatch for ≤256B inputs) — every call pays the general loop's fixed per-word overhead, and small inputs (spl-token accounts, the most common shape on chain at 615M/1.08B accounts) are exactly where that dispatch mattered most.
3. fd_zle2 has no equivalent of `fd_zle_big`'s 4-word (256B) OR-and-test skip-ahead through long uniform zero runs — that skip is *why* fd_zle is fastest on precisely the shapes (serum slabs, clmm ticks, raydium pool) that regressed most.
4. The deferred-write bookkeeping the correctness fixes required (`have_stage`/`stage_v`, re-checked every run that crosses a word boundary) adds a real branch and register on top of all of the above.

## fd_zle7 `zle7_vget`: pext/masked-load varint decode, isolated (2026-09-13)

Full writeup in `ZLE_TECHNIQUES.md`. Summary: isolated microbenchmark
(`bench_zle7_vget.c`, uncommitted, not wired into `fd_zle7.c`) of
`fd_zle7.c`'s `zle7_vget` (scalar LEB128 loop, only reached by the
`type.fill` `n==63` escape — a 64-byte all-zero input) against
`fd_zle_ideas.h`'s masked-load `pdep`/`pext` `varint_get`. Correctness:
40M cross-checked cases (structured + fuzz), zero mismatches.

| varint length | scalar | pext/masked-load | delta |
|---|---|---|---|
| 1 byte | 0.316 ns | 0.889 ns | scalar 3.2x faster |
| 3 byte | 0.745 ns | 0.889 ns | scalar +19% |
| 4 byte | 0.986 ns | 0.889 ns | pext +10% |
| 7 byte | 1.654 ns | 0.892 ns | pext +46% |
| mixed (length random per call) | 2.145 ns | 0.889 ns | **pext +59%** |

Scalar wins same-length-repeated benchmarks up to 3 bytes (predicted
branch beats fixed vector overhead); pext wins once call-to-call length is
unpredictable, because the scalar loop's exit branch stops predicting.
Not applied — the escape this feeds is essentially never hit in practice.

**Conclusion:** the "read src once" idea, as implemented, is not a win under this benchmark's cache-hot assumption. Next steps if pursuing further: (a) benchmark under cold-cache conditions (large working set that doesn't fit L2/L3) where a saved reload might actually show up, since that's the premise the idea depends on; (b) port fd_zle_big's skip-ahead and fd_zle's size-tiered dispatch into fd_zle2 before drawing conclusions on the core idea in isolation; or (c) treat this prototype as a dead end and look elsewhere for encoder wins.

## fd_zle8: headerless interleaved wire format (2026-09-15)

`fd_zle8` (`fd_zle8.c`, benchmarked via `bench_zle8`) replaces
fd_zle5/6/7's *planar* frame (header, then one contiguous literal
plane, then the token stream) with a **headerless, fully interleaved**
one, per `fd_zle_pseudocode3`:

```
frame := token*
token := L-field literal[L] [ Z-field ]
field := b:u8 ; if b==0xFF: b += varint
```

No type byte, no literal-plane size field, no token-stream relocation,
no verbatim-copy fallback frame.  A frame ends either because the final
token's L copy reaches data_sz (its Z field is simply absent) or
because the final Z fills to data_sz -- neither costs a wire byte.
Encoder scan machinery is fd_zle7's, unchanged, so the comparison
below isolates the format plus two policy changes noted under it.

All three benchmarked in the same session on the same machine
(EPYC 9555P, clang-19, `make -j BUILDDIR=clang CC=clang-19`), mix line,
3 runs each:

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle (shipping) | **30.01%** | 16.3 GB/s | 27.1 GB/s |
| fd_zle7 (planar)  | 31.38% | 16.1 GB/s | 15.8 GB/s |
| fd_zle8 (interleaved) | 30.40% | **19.4 GB/s (+19%)** | **34.2 GB/s (+26%)** |

**vs fd_zle7 it is better on all three axes** -- ratio 31.38% -> 30.40%,
encode +20%, decode +2.1x.  Three separate things pay for that:

1. **3 bytes of header gone from every >64 B frame** (fd_zle7's type/ext
   byte plus its u16 literal-plane size), 1 byte from every <=64 B one.
   At account sizes that is not noise: a 165 B spl-token account, the
   most common account on chain at 615M/1.08B, is 74 B out vs
   fd_zle7's 78 B, and 3 of those 4 bytes are the header.
2. **No relocation.** fd_zle7 parks its token stream at `dst+3+n` and
   copies it down once the plane size is known; that dead store traffic
   goes to a second, entirely separate set of cache lines from the
   literal plane, on every frame.
3. **No size cap.** fd_zle7's 22-bit plane-size field forces any input
   whose literal plane exceeds ~4 MiB to a verbatim copy, giving up
   arbitrarily large zero runs.  There is no field to overflow here.

The output-buffer bound drops with it: **`in_sz + in_sz/64 + 128`, vs
fd_zle5/6/7's `1.5 * in_sz`**.  The 1.5x in those is not a bound on any
output they can produce, it is where the provisional token stream sits.
Here the encoder writes strictly forward, so the bound is real and
provable -- `fuzz_zle8_diff` asserts the tighter `csz <= sz+sz/386+6` (the proof's bound; `sz+6` as first written was wrong, see the fd_zle9 fuzz note) on every
input.

### Where it still loses to shipping fd_zle: the nibble-packed token

The remaining 0.39pp of ratio is **entirely** fd_zle's 1-byte token
(`(min(L,15)<<4) | min(Z,15)`) vs this format's 2 separate u8 fields.
It shows up exactly where runs are short and dense, and nowhere else:

| case | fd_zle | fd_zle8 |
|---|---|---|
| metaplex edition (20 B, runs 1/8/1/10) | 20.00% | 30.00% |
| pump.fun curve (49 B, 10 runs) | 81.63% | 85.71% |
| stake account (200 B, 14 runs) | 66.00% | 68.00% |
| raydium amm v4 info (752 B, 75 runs) | 60.24% | 63.96% |
| tail >1M | 26.63% | 30.55% |
| spl-token account (165 B, 2 runs) | 46.06% | **45.45%** |
| metaplex token record, token-2022, spl-token mint nofz, tail 1-64 | — | **tie** |

Wherever both fields fit in a nibble, fd_zle pays 1 control byte per
run and fd_zle8 pays 2.  Wherever they don't (spl-token account: L=70,
Z=38), fd_zle pays 1 + two varints and fd_zle8's flat u8 pair wins.

**Next step:** fd_zle's wire format is *already* interleaved and
headerless -- the user-proposed format is fd_zle's format with the
nibble packing removed.  So the encode/decode win measured here is
attributable to the scan/emit machinery and to MIN_Z (below), not to
interleaving per se, and the two are composable: port fd_zle8's
encoder onto fd_zle's existing nibble token and the expected result is
fd_zle's 30.01% ratio at ~19 GB/s encode.  That is a bit-compatible
change to the shipping codec, not a new format.

### MIN_Z=3 is free ratio and +22% decode (applies to fd_zle5/6/7 too)

fd_zle5/6/7 all use MIN_Z=2.  For a 2-byte token that is exactly
break-even: framing an interior run of 2 zero bytes costs its own Z
field plus the L field of the literal segment it splits off -- 2 bytes
to save 2 bytes.  Measured directly by flipping the constant:

| ZLE8_MIN_Z | ratio | encode | decode |
|---|---|---|---|
| 2 | 30.40% | 19.17 GB/s | 15.37 GB/s |
| 3 | **30.40%** | 19.27 GB/s | **18.75 GB/s (+22%)** |

(both rows pre-date the decoder change below, hence the lower decode
numbers.)  Identical ratio to the basis point, +22% decode from the
frames that stop existing.  The exception, which fd_zle8 takes, is a
run that *ends* the input: it has no following L field, so it is worth
framing at Z>=2.

### The decode measurement was confounded by memcpy, not the format

First measurement had fd_zle8 decode at 18.75 GB/s vs fd_zle's 27.1 --
a 31% loss that looked like a real cost of the interleaved layout.  It
was not.  fd_zle8's decoder was calling `fd_memcpy`/`fd_memset` per
frame where fd_zle drives `fd_zle_copy`/`fd_zle_zero`; at
account-sized runs the PLT call and vzeroupper cost more than the
stores.  Transcribing those two helpers into fd_zle8 took decode
**18.75 -> 34.31 GB/s**, from 31% behind fd_zle to 26% ahead.  Same
effect already recorded for the shipping encoder (+17% dec); worth
being alert to whenever a prototype decoder is compared against it.

### Correctness

`test_zle8` (structured + boundary + escape shapes, ~1.3M round trips)
plus, under clang ASan:

- `fuzz_zle8_diff` -- encode, then decode with both an independent
  scalar transcription of the wire format and `fd_zle8_decompress`;
  also asserts `csz <= sz+6` (too tight -- corrected to `sz+sz/386+6` later, see the fd_zle9 fuzz note).  **31M execs, 0 findings.**
- `fuzz_zle8_decompress` -- arbitrary bytes at exact-sized allocations.
  **3.9M execs, 0 findings.**

## fd_zle9: explicit-branch encoder on fd_zle8's format (2026-09-15)

`fd_zle9` (`fd_zle9.c`, `bench_zle9`) keeps fd_zle8's wire format and
decoder and rewrites only the encoder, in `fd_zle_pseudocode3`'s
explicit branch style instead of fd_zle8's `zle8_cand` / `zle8_zmask` /
`zle8_spill` helpers: a pointer cursor, one `or(load(src), load(src+1))`
pair mask per word, a carry out of bit 63 jumping to a whole-word zero
spill loop, and one end-aligned load for the last <=64 B (the only
place the input end is ever tested).  Separate instantiations for
<=64 B (fd_zle8's tiny, unchanged), 65..254 B (raw u8 fields, no bound
guard) and >=255 B.  Output is **byte-identical to fd_zle8** for every
input, so ratio and decode are unchanged by construction.

Same session, `taskset -c 40`, `--no-synth`, alternating runs:

| | ratio | encode | decode |
|---|---|---|---|
| fd_zle8 | 30.40% | 13.26 / 13.24 GB/s | 23.3 GB/s |
| fd_zle9 | 30.40% | **14.76 / 14.74 GB/s (+11.3%)** | 23.3 GB/s |

(This session's fd_zle8 encode is well below the 19.4 GB/s recorded
above; only same-session numbers are comparable.)

Encode by shape (median of 3):

| case | size | fd_zle8 | fd_zle9 | |
|---|---|---|---|---|
| spl-token account | 165 | 10.24 | 11.77 | +15% |
| token-2022 account | 170 | 9.00 | 10.25 | +14% |
| spl-token mint / nofz | 82 | 5.41 / 6.15 | 6.17 / 7.94 | +14% / +29% |
| pumpswap amm | 137 | 12.22 | 15.15 | +24% |
| metaplex metadata / v1.3 | 607 / 679 | 13.77 / 14.85 | 15.76 / 17.43 | +14% / +17% |
| meteora damm v2 / dlmm bins | 1112 / 10136 | 11.96 / 14.46 | 13.80 / 16.58 | +15% |
| tail 65-256 | 140 | 10.76 | 12.51 | +16% |
| tail 64K-1M / >1M | | 7.40 / 8.50 | 8.15 / 9.20 | +10% / +8% |
| raydium amm v4 info, tail 257-1K | | 5.86 / 8.06 | 5.83 / 7.87 | -0.5% / -2.4% |
| <=64 B shapes, multi-KiB zero slabs | | | | ±1% |

### Rejected: one run per load (`ZLE9_RELOAD_PER_RUN 1`)

The later pseudocode3 small-path draft restarts the scan at each run's
end instead of walking the remaining candidate bits of the same word.
Mix encode **10.7 GB/s, -27% vs walking the bits**.  It wins only where
a word holds at most one framed run (spl-token +8.5%, token-2022 +7.4%,
pumpswap +4.4%) and loses 30-59% wherever words hold several (raydium
amm info -48%, metaplex metadata -39%, all tails >=257 B -36..-59%).
The switch was later removed from the source.

### Rejected: zero writer with the copy helper's tiering

Four clamped overlapping stores for every 65..256 B zero run and
`fd_memset` past 256 (the copy helper's shape) instead of fd_zle's
branchy 2/3/4-store tiers and 512 B loop.  Decode: meteora dlmm bins
-13%, pumpswap amm -9%, mix flat to slightly down.  Decoder zero runs
cluster at 65..128 B, where the branchy writer needs 2 stores behind a
well predicted compare.  Reverted.

### Adopted: pseudocode3's `zle_put_vint`

Same bytes as fd_zle8's `zle8_put` + `zle8_vput`; the pseudocode's
`return out` must be `return out + 1`.  Cold path (fields >=255 only),
no measurable effect.

### Correctness

- `test_zle9`: test_zle8's suite plus a dense 1..520 B sweep across the
  64/65/254/255 dispatch edges (~1.7M round trips), every round trip
  also asserting output byte-identical to `fd_zle8_compress`.  Passes
  with both `ZLE9_RELOAD_PER_RUN` settings.
- `fuzz_zle9_diff` (clang ASan): byte-identical to fd_zle8, round trip
  through `fd_zle9_decompress`, `csz <= sz+6` (since corrected to `sz+sz/386+6`), random offset and
  poisoned pad.  **43.0M execs (300 s, max_len 4096), 0 findings.**

### Cleanup, and two codegen traps it hit

The zle9 encoder was refactored for readability (emitters and scan
tests pulled into helpers, the `ZLE9_RELOAD_PER_RUN` switch removed)
under a hard rule: same machine code on the hot paths.  Checked by
diffing `objdump -d` of `zle9_compress_small` / `zle9_compress_big`
against the pre-cleanup build, then benchmarking.  Two helpers that
look free were not:

1. **Plain `static inline` helpers were emitted out of line.**  clang
   kept the 256 B stride test and the escaping token emitter as
   separate functions, so the big path paid a call per stride and per
   token: **-3..-6% encode on every >=282 B shape**.  Fixed with
   `__attribute__((always_inline))`.
2. **The 256 B literal stride test must stay inline in the loop.**
   Behind an always_inline helper returning the mask as `ulong`, clang
   rotates the loop (`jne` instead of `je` after the `kortest`) and
   reloads `end` from the stack each stride: still **-1.1% mix,
   -2..-5% big shapes**.  Restoring the original expression inside the
   loop gives byte-identical `zle9_compress_big`.

Final state: `fd_zle9_compress` and `zle9_compress_big` disassemble
identically to the pre-cleanup build, `zle9_compress_small` differs
only in `lea` operand order.  Mix encode 14.75 -> 14.77..14.80 GB/s,
no shape outside ±2%.

### Literal stride hit: resolve from the loads, at a cold label

On a 256 B literal stride hit, the encoder used to `break` back to
`scan`, which reloads the words word by word (and, for a hit in the
last two words, re-runs the 4-word stride test first).  Now `d` comes
straight from the four pair words already loaded (`a0..a3`), as in
fd_zle8.  A model of the control flow shows this path is rare on the
mix: only address lookup table and tail 4K-16K hit it, once per call.

Placement mattered more than the saved loads.  With the resolution
inline in the stride loop, `zle9_compress_big` grew 456 -> 490 insns
and shapes that never stride lost encode (edition marker -3.8%, dlmm
bins -3.7%, clmm ticks -2.1%; mix 14.78 -> 14.75).  Moved to a
`lit_hit:` label after the function's return (a0..a3 hoisted to
function scope), those recover and the mix is flat to up:

| | mix enc | edition marker | dlmm bins | ALT | serum slab |
|---|---|---|---|---|---|
| pre-change | 14.78 | 19.14 | 16.38 | 15.85 | 171.65 |
| inline exit | 14.75 | 18.42 | 15.77 | 15.98 | 171.27 |
| cold `lit_hit` (kept) | **14.79** | 19.06 | 16.24 | 16.12 | 177.97 |

Same session, fd_zle8 at 13.27 GB/s: fd_zle9 is +11.5% mix encode.

Correctness: `test_zle9` passes (output byte-identical to fd_zle8);
`fuzz_zle9_diff` under ASan, 38.3M execs (300 s), 0 findings.

### Long-zero stride hit: resolve from the loads (kept)

The 256 B zero stride in `lz_scan` used to `break` on a hit back to
`lz_scan`, which reloads word by word and, unless the hit is in the
first word, re-enters the 4-word stride test over mostly the same
bytes.  Now the stride keeps its four raw loads (`a0..a3`, shared with
`lit_hit`) and a cold `lz_hit:` label past the return picks the first
non-zero word.  `zle9_nonzeros4` is gone (the stride test is written
in the loop, per trap 2 above).  `zle9_compress_small` is unchanged
(0 differing insns); `zle9_compress_big` 0x846 -> 0x8d1 bytes.

### Rejected: reuse lz_last's end word in scan_last

After `lz_last`, control always reaches `scan_last`, which reloads the
same end-aligned word `lz_last` just loaded (its zero mask is `~m`).
Two ways to skip that reload, both on top of the stride change:

- **duplicate:** `lz_last` does its own `EMIT` and jumps to `last_run`
  with `d = ~m >> ...`.  Second inlined `EMIT` (copy tiers + memcpy
  call site): `zle9_compress_small` 0x618 -> 0x769 bytes.
- **flag:** `lz_last` falls through to the one `lz_emit` as before and
  sets `tail`; `scan_last` takes `tail ? ~m : zeros( end-64 )`.
  Small path 0x618 -> 0x67c.

Median of 5 alternating rounds, core 40, encode GB/s vs pre-change:

| case | pre | stride only (kept) | + reuse, duplicate | + reuse, flag |
|---|---|---|---|---|
| **mix** | 14.78 | **+0.3%** | -0.5% | -2.4% |
| token-2022 account | 10.33 | +0.1% | -6.5% | -7.3% |
| tail 65-256 | 12.54 | +0.0% | -8.1% | -6.1% |
| spl-token account | 11.81 | +0.1% | +0.5% | -3.2% |
| pumpswap amm / mint nofz | 15.25 / 8.01 | 0 / 0 | +2.5% / +0.1% | -5.0% / -5.6% |
| raydium amm v4 pool | 64.38 | +5.3% | +10.7% | +5.5% |
| raydium clmm ticks | 128.38 | +5.1% | +6.9% | +4.0% |
| serum v3 slab / queue 64K / 256K | | +3.6 / +5.8 / +5.3% | +4.7 / +5.6 / +5.5% | +2.3 / +6.6 / +5.7% |
| meteora dlmm bins | 16.23 | +1.7% | +1.8% | -0.1% |
| worst other shape | | -1.2% (tail 1K-4K) | -1.4% | -3.3% |

The reuse targets token-2022 (one hit per call) and loses most exactly
there.  Once more, the saved load of a cache-hot word is worth less
than the code it takes to avoid it on the small path.  Reverted; a
comment at `lz_last` points here.

### Bench core 40 was SMT-contended

Every fd_zle9 number above was taken with `taskset -c 40`.  cpu40's SMT
sibling is cpu104, which the week-long firedancer cluster on this box
keeps at 100%.  The same binary on core 56 (sibling 120, both idle):
mix encode **22.77 GB/s vs 14.8 GB/s on core 40**, and roughly 1 run in
4 on core 40 dipped to ~12.3 GB/s.  Numbers in the sections above are
comparable only with each other; from here on the bench core is 56.
Check `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`
and idle both threads before benchmarking.  (Core 56 also showed
occasional disturbed runs, so still use medians and >=5 rounds.)

### Fuzzer bug: `csz <= sz+6` was tighter than the format

A 300 s run of `fuzz_zle9_diff` failed `FD_TEST( c9<=sz+6UL )` on a
3024 B input: four literal runs of 418..967 B, each followed by exactly
3 zeros, then a 431 B tail.  Each >=383 B literal has a 3 byte L field
(+1 each over the 3 zeros it frames) and the tail's field is 3 bytes
(+3), so the correct output is **sz+7**.  An independent model of the
token stream gives the same 3031 B.  fd_zle8.c's own proof says
`W <= n + n/386 + 6`; the fuzzers (zle8's, commented "the real bound",
and zle9's copy of it) had dropped the n/386 term.  Both now assert
`sz+sz/386+6`, and the input replays clean on both.  **Not a codec
bug**: fd_zle8 and fd_zle9 both produce the right 3031 B.  The earlier
31-43M exec runs never assembled that shape.

`test_zle9` and `fuzz_zle9_diff` now also check that the encoder writes
nothing past its output (canary bytes from csz to 64 B past the bound).

### Encoder literal copy: masked store was not the cost, the call was

TODO candidate 4 was an unmasked 64 B store for the encoder's <=64 B
literal copies (masked stores to memory being slow on Zen), which is
safe because every token starts within n/386 bytes of the input it
consumed.  Benchmarking it exposed something else: **clang had emitted
the plain `static inline` `zle9_copy` out of line in
`zle9_compress_big`**, so two of its emit sites paid a `call` per
token (trap 1 above, again).  The small path and the decoder inlined
it.

Variants on top of the long-zero stride change (c1).  First on core 40
(contended): unmasked everywhere is -1.1% mix (tail 65-256 -13%);
unmasked big-path only == the same with a masked store (both -0.1%),
so the store kind is irrelevant.  Then core 56, median of 7, encode
vs c1:

| case | c1 GB/s | always_inline copy, unmasked | always_inline copy, masked (kept) |
|---|---|---|---|
| **mix** | 22.77 | +1.1% | **+1.1%** |
| metaplex metadata / v1.3 | 23.15 / 28.19 | +8.2% / +4.7% | +8.3% / +4.7% |
| raydium amm v4 pool / info | 125.0 / 8.59 | +7.4% / +2.4% | +7.6% / +2.3% |
| openbook v1 queue | 173.8 | +4.2% | +3.9% |
| meteora dlmm bins / clmm ticks | 21.49 / 215.1 | +3.9% / +3.9% | +4.1% / +3.2% |
| tail 257-1K / 16K-64K | 11.11 / 12.33 | +3.7% / +2.7% | +3.8% / +2.6% |
| serum v3 slab | 440.2 | **-6.6%** | +1.5% |
| serum v3 queue 64K / slab 64K / 256K | | -1.4 / -2.0 / -2.4% | -2.5 / -3.2 / -1.1% |
| address lookup table | 25.29 | -1.3% | -1.2% |
| <=254 B shapes | | ±0.4% | ±0.4% |

Kept: `zle9_copy` marked `always_inline`, masked stores unchanged, so
buffer safety does not start leaning on the slack proof.
`zle9_compress_big` 0x8d1 -> 0xa18 bytes; `zle9_compress_small`,
`fd_zle9_compress` and `fd_zle9_decompress` are byte-identical to c1.
Unmasked store rejected: no gain, one shape -6.6%.

Correctness: `test_zle9` passes; `fuzz_zle9_diff` (ASan, fixed bound +
canary) 8.93M execs in 120 s, 0 findings.

### Change 2 re-tested on the clean core: still rejected

The lz_last word reuse was rejected on contended core 40, so it was
re-applied (duplicate-EMIT form) on top of the final code and benched
on core 56, median of 7:

| case | final | + reuse |
|---|---|---|
| **mix** | 22.64 | **-0.8%** |
| tail 65-256 | 23.53 | -8.2% |
| token-2022 account (the target) | 14.11 | -1.1% |
| openbook v1 queue / serum v3 slab | | -5.5% / -5.4% |
| metaplex metadata | | -4.4% |
| address lookup table / edition marker | | +5.3% / +3.7% |

`zle9_compress_small` grows 0x618 -> 0x769.  Stays out.

### fd_zle9 session result (core 56)

Pre-session baseline -> final (long-zero stride resolved at `lz_hit`,
`zle9_copy` always_inline), median of 7, clean baseline rounds:
**mix encode 22.43 -> 22.64 GB/s (+0.9%)**; 20.1 -> 22.9 GB/s on a
5-round run whose baseline had two disturbed rounds.  Largest shape
gains: raydium amm v4 pool +21%, openbook v1 queue +21%, serum v3 slab
+25%, clmm ticks +10%, metaplex metadata +8.5%, tail 257-1K +6.6%.
Shapes <=254 B run byte-identical code (their ±2-3% swings are layout
noise).  Decode unchanged (`fd_zle9_decompress` byte-identical), ratio
unchanged (output byte-identical to fd_zle8).

## fd_zle10: fd_zle9's encoder without the goto state machine (2026-09-15)

`fd_zle10.{c,h}` + `test_zle10.c` / `bench_zle10.c` / `fuzz_zle10_diff.c`
(renamed copies of zle9's).  Same wire format, field codec, copy
helpers, tiny path and decoder as fd_zle9.  The encoder's 26 gotos
and 12 labels become:

- a **word loop** (`while( src+65<=end )`) that loads the pair mask;
  a pairless word checks the next word inline, and a second pairless
  word (big only) calls `zle10_lit_stride`;
- a **run loop** (`for(;;)` over the word's pair bits).  A carry out
  of bit 63 calls `zle10_zero_end` (last-word, word-at-a-time and
  256 B stride cases, all via `return`), sets `src = ze` and breaks;
- a **tail** block (end-aligned load) and the trailing token.

The one goto left is EMIT's big-path bound bail to `trailing` (TODO
candidate 3).  Output byte-identical to fd_zle8 (`test_zle10`).

Three versions, core 56, median of 9 alternating rounds (v2 kept):

| case | zle9 GB/s | v2 (kept) | v3 |
|---|---|---|---|
| **mix** | 22.74 | **+0.2%** | -0.8% |
| spl-token account / token-2022 | 20.04 / 14.25 | +3.7% / -2.0% | +1.0% / -2.2% |
| spl-token mint / nofz | 8.96 / 12.10 | +6.2% / +7.9% | +6.1% / +7.8% |
| pumpswap amm / stake account | 27.04 / 11.49 | +6.0% / +3.9% | +2.6% / +3.9% |
| tail 65-256 | 23.49 | **-8.0%** | -10.3% |
| metaplex metadata | 24.95 | -3.7% | -5.9% |
| address lookup table | 24.88 | **-13.9%** | -8.9% |
| openbook v1 queue | 180.3 | -7.7% | -4.9% |
| serum v3 slab | 433.2 | -19.3% | -20.2% |
| raydium amm v4 pool | 133.9 | -2.6% | -8.8% |
| tail 257-1K / 4K-16K | 11.38 / 28.26 | -4.6% / -3.0% | -3.3% / -3.5% |
| multi-KiB zero slabs, clmm, dlmm | | ±1% | -1..-3% |

(Single big shapes swing ±5-15% between whole runs here, e.g. serum
v3 slab; the mix and the small shapes are stable.)

- **v1** kept an `empty` counter (stride after two pairless words) and
  a `next` cursor for the run loop's exit.  Mix -0.9%, ALT -17%.
  Disassembly: both locals cost registers; the word loop spilled
  `src+1` to the stack (`-0x38(%rbp)`) on every pair word and tested
  the counter on every pairless word.
- **v2** removes both: the second word check is inline, and the run
  loop assigns `src` at its two exits.  Stride hits resolve inline
  (branchless 4-bit ctz for literals, a branch per word for zeros).
- **v3** = v2 with both stride hits resolved by reloading word by
  word (smaller code).  Recovers some ALT, loses amm v4 pool and the
  mix.

What the loops cost: fd_zle9 put both stride-hit resolutions at cold
labels past the return (trap: an inline exit cost 4-5% on shapes that
never stride), and there is no structured way to place a block there.
Stride-heavy big shapes are where zle10 still loses; the small path
got 7% smaller (0x618 -> 0x5c9) and faster on token/mint accounts but
slower on tail 65-256.  perf over the whole bench shows the same hot
instructions in both small paths (end-word `cmovae`, masked copy,
Z-field store), plus reloads of the `-1` mask constant in zle10.

## fd_zle14: block coded, written in assembly (2026-09-26)

`fd_zle14.S` + `fd_zle14.h` + `fd_zle14_ref.h` (scalar reference, test
only) + `test_zle14.c` / `bench_zle14.c` / `fuzz_zle14_diff.c` /
`fuzz_zle14_decompress.c`.  A new wire format, and both routines written
in x86-64 assembly rather than intrinsics.

### Why a new format

Every codec up to fd_zle13 walks a token stream: where the next token
starts depends on the length field of the current one, so both sides
carry a serial chain through the data -- a load, a field decode, a
pointer add -- once per run.  On the dense shapes that chain *is* the
cost: fd_zle11 decodes `tail 64K-1M` at 3 bytes/cycle where its uop count
alone would allow eight.

fd_zle14 codes the input in **64 byte blocks**.  Every block advances the
input (encode) or the output (decode) by exactly 64 bytes, and the other
cursor advances by a popcount of the block's literal mask, so neither
side ever reads a length to find out where the next unit begins.  Blocks
are grouped into chunks that keep their control bytes and their literal
bytes in two runs rather than interleaved, so the control stream can be
walked without stepping over literals.  Dense blocks store the literal
mask itself (9 bytes), sparse ones a list of the zero runs' byte
positions (1..4), and a run of all-zero or all-literal blocks costs one
byte for up to 64 of them.  The format is documented at the top of
`fd_zle14.S`.

Two parameter choices matter and both were measured:

- **MIN_Z=2, not 3.**  With block local runs, coding a 2 byte run costs
  two position bytes and saves two literals, and the extra transitions
  push more blocks into mask mode, which is both smaller and faster.  It
  is *better* on ratio (30.93% vs 31.18% at the time) as well as one
  shift/and pair cheaper per block.
- **Mask mode at 5 transitions, not 9.**  A mask is 9 bytes against a
  list's 5 to 8, which costs 0.2% of output, and buys a group whose
  length neither side has to scan for: decode +6%.  4 and 6 both
  measured worse.

### Results (EPYC 9555P, core 56, clang-19, median of 7 whole-bench runs)

| codec | ratio | encode | decode |
|---|---|---|---|
| fd_zle (committed) | 30.01% | 16.1 GB/s | 30.0 GB/s |
| fd_zle11 (fastest before) | 30.40% | 27.5 GB/s | 35.0 GB/s |
| fd_zle13 | 30.45% | 25.5 GB/s | 31.4 GB/s |
| **fd_zle14** | **31.15%** | **37.7 GB/s** | **46.0 GB/s** |

vs fd_zle11: **encode +37%, decode +31%**, ratio +2.5% relative.
vs the committed fd_zle: encode **2.34x**, decode **1.53x**.
The mix's memcpy roofline is 82 GB/s, so the encoder runs at 46% of it
and the decoder at 56%.  (Run to run spread on a quiet core is ~1%; with
anything else running on the socket decode swings 40-47.)

Per shape, median of 3 whole-bench runs, weight = share of fd_zle11's
mix time:

| case | size | weight | enc | dec |
|---|---|---|---|---|
| spl-token account | 165 | 30% | 0.93x | 0.93x |
| tail 64K-1M | 237237 | 17% | **5.7x** | **5.0x** |
| tail 257-1K | 456 | 9% | 2.2x | 2.0x |
| metaplex metadata | 607 | 8% | 1.6x | 1.3x |
| token-2022 account | 170 | 6% | 0.99x | 0.92x |
| tail >1M | 1958702 | 6% | 4.2x | 3.5x |
| tail 4K-16K | 7771 | 5% | 2.4x | 1.7x |
| tail 65-256 | 140 | 4% | 0.98x | 0.65x |
| spl-token mint | 82 | 4% | 1.2x | 1.7x |
| tail 1K-4K | 2175 | 3% | 2.8x | 2.6x |
| meteora dlmm bins | 10136 | 2% | 1.4x | 0.81x |
| metaplex metadata v1.3 | 679 | 2% | 1.8x | 1.7x |
| tail 16K-64K | 29668 | 1% | **5.0x** | **4.5x** |
| meteora damm v2 pool | 1112 | 0.6% | 2.0x | 2.0x |
| raydium amm v4 info | 752 | 0.5% | 3.2x | 3.4x |
| serum v3 slab | 14524 | 0.6% | 0.62x | 0.98x |
| tail 1-64 / token record | 29 / 32 | 1% | 0.40x | 0.63x |

Everything from 257 bytes up wins, by 1.3x to 5.7x.  Accounts around 165
bytes are at 0.93x and the tiniest shapes at 0.4-0.65x, and since
`spl-token account` alone is 30% of the weighted time, that is what keeps
the mix at +37%/+34% rather than 3x.

### Why the small sparse accounts do not win

A 165 byte account is 3 blocks, so fd_zle14 runs 3 group dispatches
where fd_zle11 runs 2 token emits, and the work per group cannot go much
below 30 instructions (16 of dispatch: the terminator scan, the boundary
tests and the kind/count branch; 14-18 of emit).  Measured for
`spl-token account`, decode: 157 instructions in 28 cycles against
fd_zle11's 132 in 25.  Adding a dedicated small-input path was costed
and rejected -- its gate, its own framing and its copy-out come to about
what the general path already spends.

The encoder's small-account gap has a different shape: it scans with
AVX-512 either way, and what it loses is framing (2 byte chunk header,
the header computation, and the literal copy out of the chunk's scratch)
against fd_zle11's near zero.

### Traps measured along the way

- **Straight-line mask builds beat a generic one by a mile.**  The exact
  emit path (used for the chunk's last group and whenever the sloppy
  stores have no room) built the literal mask with an 8 slot bzhi/xor
  loop plus a "force unused slots" step: 23 instructions.  Replacing it
  with one straight-line sequence per list length -- three instructions
  per position, no force step, since the length is known -- took decode
  from 36.6 to 41.8 GB/s (**+13%**).  Keeping an 8 slot fallback for the
  lengths only a corrupt stream can produce costs nothing.
- **Splitting the emit on "is this block 64 bytes" was worth 7%.**  A
  full block needs no clip on its keep mask and no mask on its store;
  only the blob's last block does.  41.8 -> 44.8 GB/s.
- **A read-modify-write on bytes a `vpcompressb` just stored costs ~8
  cycles.**  Terminating the position list with `orb $0x80,-1(%rdi,%rcx)`
  right after the compress store cost 8.7 of 46 cycles on a 165 byte
  encode.  Recomputing the last position (`lzcnt`, `xor $63`, `or
  $0x80`) and storing that byte plainly over the compressed one is three
  instructions more and eight cycles less.
- **Store-to-load forwarding does not happen when the load is wider than
  the store.**  The chunk's literal copy (scratch -> output) costs 12
  cycles on a 1 byte input and 10 on a 32 byte one, nearly all stall,
  because a 64 byte load covers bytes a masked compress store wrote.  It
  amortizes to 3 cycles by 165 bytes and to nothing above that, which is
  why the scratch stayed -- and why "decode the tail into a stack block
  so the sloppy paths stay legal" was rejected: the copy back out pays
  the same stall.
- **A hot register is worth more than two instructions.**  Moving the
  chunk's control end (2 uses per group) into a stack slot to save a
  push/pop pair cost 12% of decode.  Moving the literal read limit (1
  use) gained 2%, and the last block's size (1 use per chunk) gained 3%.
- **Single-shape A/B is misleading.**  Short exact paths for a one-block
  zero or literal run as a chunk's last group measured 5 cycles *slower*
  on `spl-token account` alone and 3% *faster* on the mix.  Only medians
  of 5-9 whole-bench runs are worth acting on: single mix runs swing 5%,
  single shapes much more, and the decode mix here is bimodal at ~39 and
  ~46 GB/s depending on where the code lands.
- **The bench overlaps calls.**  Nothing carries a dependency from one
  `fd_zle14_decompress` to the next, so the loop measures throughput,
  not latency: instruction count predicted every result far better than
  dependency chains did.  The decoder's per-group control load chain
  looked like the problem for a long time and was not (software
  pipelining that load into the previous group's emit bought 4%, where
  the instruction cuts above bought 25%).
- **The sloppy stores still earn their keep.**  With the exact path down
  to ~25 instructions, routing *every* position list through it (and
  deleting the sloppy paths and their two boundary tests) measured 41.0
  against the hybrid's 45.7.  Overlapping 64 byte stores beat one
  `vpexpandb` wherever there is room for them.
- Zen 5 byte granular compress/expand are not cheap: `vpcompressb` to
  memory is 3 cycles of throughput, to a register 5, and `vpexpandb` 5
  either way (measured).  A block's literals cost one `vpcompressb`,
  which is why the encoder tops out at ~4 cycles per dense block -- it
  is compress limited, not ALU limited.  Where a block can dodge the
  compress entirely it is worth doing: a block whose single zero run
  reaches its end (7% of groups) has one control byte and a literal run
  that is just the block's own prefix, so both go out with a plain store,
  and that alone is +1.8% of encode.
- **ASan cannot see hand-written assembly, so the decoder needed a canary
  harness.**  Two out of bounds writes survived 100M+ fuzz executions
  because libFuzzer's ASan instrumentation only covers compiled C: the
  faults only surfaced later, as "CHECK failed: asan_allocator.cpp" when
  the corrupted heap metadata was freed, with inputs that did not
  reproduce standalone.  A 30 line harness that puts 512 byte 0xA5 walls
  either side of both buffers and checks them after every call found the
  second one in seconds and shrank it to 12 significant bytes.  Any
  assembly codec in this tree wants one.
- **A corrupt stream can set bits the coder never sets.**  The decoder's
  sloppy emit paths store 64 bytes at each segment offset, which is legal
  while the offsets are the 0..63 the format allows -- but a position byte
  with bit 6 set reads as 64..127, and the store then reached 191 bytes
  into a block with 128 bytes of slack.  One `andq` of the whole control
  word against 0x3F3F3F3F3F3F3F3F, right after the terminator scan, pins
  every position in range and pays for itself by replacing the per
  position clamps the paths already had.  The same class of bug bit the
  third literal run's offset in the four position path, where p2-p1 can
  go negative and wrap the load address below the literal cursor.
- **A decoder scratch has to be sized for the field, not for what the
  encoder does with it.**  The chunk header's control size is 9 bits, so
  a stream can claim 511 control bytes; the decoder's padded copy of that
  region (used when the 8 byte control reads would run past the blob) was
  384 bytes, and 511 of them walked over the saved registers and the
  return address.  `fuzz_zle14_decompress` found it in 500k execs once
  the encoder started emitting chunks long enough to make the copy path
  common -- a reminder that the decoder's bounds come from the format's
  field widths, never from the encoder's habits.
- **Chunks should end on their fields, not on a byte count.**  Ending a
  chunk every 2048 input bytes made long zero runs pay a header every
  2 KiB; letting a chunk take another span while its control field has
  room, and letting a zero run scan out to the 64 blocks one group can
  code, cut `serum v3 queue 64K` from 0.17% to 0.05% of raw and lifted
  `openbook v1 queue` encode 30%.  The control field is 9 bits, so the
  chunk has to stop at ctl_sz 200: a span can add 288 bytes of control
  and 488 still fits.  (Found by fuzzing: at 480 it overflowed and the
  stream was unparseable.)

### Correctness

`test_zle14` checks the assembly byte for byte against
`fd_zle14_ref_compress` over every shape it could think of (all sizes
1..64 of six textures, runs astride block boundaries, chunk span edges,
zero tails to 8 MiB, mask-mode texture, 40k structured random cases),
round trips through both decoders, and checks the compress bound, that
nothing but the literal copy's slack is written past the output, that
nothing is written past `data_sz`, that one byte less capacity is
reported rather than written, and that extra capacity changes nothing.
`fuzz_zle14_diff` (ASan, differential + round trip + bound + canary) ran
over 100M execs across the session with no findings, and
`fuzz_zle14_decompress` (ASan, arbitrary bytes into exact sized heap
buffers, plus a re-compress round trip of whatever it accepts) found
nothing.

### Left on the table

- The decoder's group dispatch is 16 instructions and roughly half of
  the small-account cost; a jump table on the first control byte trades
  two of them for an indirect branch, untested.
- The encoder still spends ~21 instructions per chunk on the header and
  the literal copy.  Most of that is avoidable if the literal count
  moves into the header instead of the control count, which needs a
  third header byte (1.3% of ratio on a 165 byte account).
- fd_zle14 has no scalar fallback: `Local.mk` only builds it under
  `FD_HAS_AVX512`, and the API is AVX-512 only for now.
