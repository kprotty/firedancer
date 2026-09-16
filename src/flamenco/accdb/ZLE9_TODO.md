# fd_zle9: next steps

Handoff for the next session working on `fd_zle9.c`.  History and all
numbers so far are in `ZLE_BENCH.md` (sections starting "fd_zle9").

## Current state (2026-09-15, after session 2)

- `fd_zle9.{c,h}`: fd_zle8's wire format and decoder, explicit-branch
  encoder (labels `scan`, `run`, `long_zero`, `lz_scan`, `lz_last`,
  `lz_emit`, `scan_last`, `last_run`, `trailing`, cold `lit_hit` and
  `lz_hit`).  Output is **byte-identical to fd_zle8** for every input.
- Session 2 kept: candidate 1 (`lz_hit`) and `always_inline` on
  `zle9_copy`.  Rejected: candidate 2, unmasked store (candidate 4).
- Tests: `test_zle9` (zle8 byte equality + no writes past output),
  `fuzz_zle9_diff` (same, ASan, fixed bound).  `bench_zle9`.  Nothing
  committed.
- Mix encode on **core 56**: final 22.64 GB/s vs pre-session 22.43.
  Decode unchanged (~33.8).
- Binaries (do not rebuild over them), `build/clang/unit-test/`:
  - `bench_zle9_final`    -- current source
  - `bench_zle9_baseline` -- start of session 2
  - `bench_zle8_baseline`

## Candidates (user approved starting these)

Planned grouping: **1+2 as one change, 4 as a separate change.  3 needs
the user's decision first** -- ask before doing it.

### 1. Long-zero stride hit: resolve from the loads -- DONE (kept, mix +0.3%, big zero shapes +4-6%)

Where: `lz_scan`'s `if( !narrow )` loop calling `zle9_nonzeros4`.  On a
hit it `break`s back to `lz_scan`, which reloads and re-tests word by
word.  Do what `lit_hit` does for literals: keep the four loads
(z0..z3), test the OR, and on a hit test each word in turn, set
`m`/`ze` for the first non-zero word and go to `lz_emit`.  Put the
resolution at a **cold label after the return** (see pitfall C).

Expected: fires on meteora dlmm bins 33x/call, raydium amm v4 pool 3,
raydium clmm ticks 2, metaplex metadata / v1.3 / meteora damm 1.  Saves
one load + test per hit; small win on dlmm, flat elsewhere.

### 2. Reuse lz_last's end word in scan_last -- REJECTED (core 40: token-2022 -6.5..-7.3%; clean core 56 re-test: mix -0.8%, tail 65-256 -8%)

Where: `lz_last` loads `end-64` for the non-zero mask.  When the run
ends before `end`, `lz_emit` sets `src = ze` and control always lands
in `scan_last` (ze >= end-64, so src+65 > end), which loads `end-64`
again for the zero mask -- the complement of the same 64 bytes.

Fix: in `lz_last` keep the unshifted mask `M = zle9_nonzeros( end-64 )`.
After the emit decision, go directly to the `scan_last` logic with
`rem = end-ze`, `d = ~M >> (64-rem)`, `d &= d>>1`, then `last_run`
(handle rem==0 -> `trailing`).  Keep `lz_scan`'s path unchanged.  Watch
that `EMIT` can `goto trailing` in the big instantiation.

Expected: once per call on **token-2022 account (85M accounts, 170 B,
small/narrow path)**, plus openbook queue, raydium amm v4 pool, serum
slab shapes.  Best mix-weighted candidate.

### 3. Big-path output bound check (ASK USER FIRST)

`EMIT`'s `if( FD_UNLIKELY( out+_L+16UL>cap ) ) goto trailing;` runs per
token for >=255 B inputs but is unreachable per fd_zle8.c's "Output
bound" proof (fuzzer also asserts the proof's csz <= sz+sz/386+6).  zle8 kept it
deliberately so buffer safety does not rest only on a proof.  Removing
it is a safety-vs-speed policy call, not a cleanup.

### 4. Unmasked store for encoder literal copies <=64 B -- REJECTED as such; found the real cost (out-of-line zle9_copy), fixed with always_inline, see ZLE_BENCH.md

`zle9_copy64` masks both the load and the store.  In the **encoder**
the masked load already zero-fills past L, and output has >=64 B slack
past any token (out <= dst+n+1 at token boundaries, bound >= n+128), so
the store can be an unmasked `wwb_stu`.  Masked stores to memory are
comparatively slow on AMD Zen, and this runs every token (tiny, small
and big paths, plus `trailing`).  **Not applicable to the decoder**
(`zle9_copy` / `zle9_zero` there must not write past d+sz), so the
encoder needs its own helper.  Verify the slack argument for the tiny
path and `trailing` before relying on it.

## Session 2 notes (2026-09-15, later)

- **Bench core:** core 40's SMT sibling (cpu104) is kept 100% busy by the
  long-running cluster.  It cost ~35% throughput (mix 14.8 vs 22.8 GB/s)
  and caused ~1 in 4 runs to dip.  Use core 56 (sibling 120), check both
  are idle first.  All numbers from core 40 are comparable only to each
  other.
- **Fuzzer bound was wrong:** `fuzz_zle{8,9}_diff` asserted `csz<=sz+6`;
  the proof allows `sz+sz/386+6` and a 3024 B input really produces sz+7.
  Both fuzzers fixed.  Not a codec bug.
- `test_zle9` / `fuzz_zle9_diff` now also check that the encoder writes
  nothing past its output.
- Candidate 3 (big-path bound check) is still open and still needs the
  user.  Note for that decision: when the check fires, `trailing` writes
  *more* than the refused token, and those bytes are in bounds only
  because of the same proof.  So the check adds no safety the proof
  doesn't already give.

## How to verify each change

1. `make -j64 BUILDDIR=clang CC=clang-19 test_zle9 bench_zle9` (no
   warnings), run `./build/clang/unit-test/test_zle9 --log-path ''`.
2. Codegen: `nm -S` sizes of `zle9_compress_small` / `zle9_compress_big`
   and a normalized `objdump -d --no-show-raw-insn` diff against the
   baseline binary.  Normalize with
   `sed -E 's/^ *[0-9a-f]+:\t//; s/ *(#|<).*//; s/0x[0-9a-f]+/ADDR/g; s/\b[0-9a-f]{4,}\b/ADDR/g'`.
   Know which functions *should* change.
3. Bench: alternate baseline vs new, >=3 rounds (5 when deciding),
   `taskset -c 56 <bin> --no-synth --log-path ''` (not 40, see session 2 notes), compare the `mix`
   line and per-shape medians.  **Check shapes that do not take the new
   path** -- layout regressions show up there.  One disturbed round per
   session is common; use medians.
4. Fuzz: `make -j32 BUILDDIR=clang-fuzz-asan CC=clang-19 EXTRAS="fuzz asan" fuzz_zle9_diff`,
   then `build/clang-fuzz-asan/fuzz-test/fuzz_zle9_diff -max_total_time=300 -max_len=4096 <corpus>`
   (the binary is a file, not a directory).  Don't fuzz during a bench,
   or pin it to a distant core (e.g. `taskset -c 100`).
5. Update `ZLE_BENCH.md` with the result, including rejected attempts.

## Pitfalls already hit (don't repeat)

- **A.** Plain `static inline` helpers used by `zle9_compress_core` got
  emitted out of line by clang (-3..-6% big shapes).  Use
  `__attribute__((always_inline))`.
- **B.** The 256 B literal stride test must stay written inside its
  loop.  Behind a helper returning the mask as `ulong`, clang rotated
  the loop and reloaded `end` from the stack (-1% mix).
- **C.** A rare multi-statement exit placed inline in a hot loop cost
  ~4% on shapes that never take it (dlmm bins, edition marker) through
  layout alone; `FD_UNLIKELY` did not help.  Cold label past the
  return fixed it (`lit_hit`).
- **D.** Rejected, don't retry: one-run-per-load rescan (-27% mix);
  `zle9_zero` with copy's 4-store tiering (-13% decode dlmm bins).
- **E.** Writing `_kortestz_mask64_u8` explicitly gains nothing: clang
  already emits `kortestq` for `if( (ulong)mask )` and only `kmovq`s
  where the bits are used.
