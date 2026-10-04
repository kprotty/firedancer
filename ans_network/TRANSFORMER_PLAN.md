# Instructions: build a transformer on the ANS backends and price it

You are picking this up cold. Read `FINDINGS.md` first — it is the
verdict from the work that led here — then `README.md` for the network
and its backends. This file is the brief for the next experiment.

**The question.** Does the identity this project is built on (an ANS
encoder's code length *is* a linear unit's pre-activation) hold up on a
transformer, and how does it price against the same transformer with
ordinary float matmuls?

**The short answer you should expect.** It will probably lose, and the
point is to establish that with numbers rather than argument. Section 7
says what result would mean you have a bug rather than a finding.

## 1. What already exists, and what you must not rebuild

`ans_network/` is standalone: its own Makefile, nothing wired into the
Firedancer build, no shared headers. Keep the new work the same way.

| | |
|---|---|
| `ans_uabs.{h,c}` | uABS, the fast backend. `ans_uabs_blk_avx512` **is already a binary-input GEMV.** |
| `ans_avx512.{h,c}` | across-neuron tANS, 12x slower than uABS. Not worth using here. |
| `model.{h,c}` | model, quantizer, `ans_net_compile`, `ans_net_predict` |
| `bench_float.c` | the fp32 AVX-512 GEMV/GEMM baselines, already written and validated |
| `test.c` | `make check`; `test_uabs` is the pattern for testing a new kernel |

The coder is done. The work here is the transformer around it.

```c
/* n_grp*16 code lengths in Q8 bits, one per output unit.
   r0,r1 are [group][feature][16] ushort reciprocal planes,
   bit is the dense binary input of length n_in. */
void ans_uabs_blk_avx512( ushort const * r0, ushort const * r1,
                          uchar const * bit, uint n_in, uint n_grp, int * out );
```

A linear layer `y = W x` with binary `x` is one call. Build the weight
planes with `ans_uabs_r( w, sym )` in the blocked layout
`[(g*n_in + i)*16 + k]`, exactly as `ans_net_compile` does for `u1`.

## 2. The design, and the three things that cannot be ANS

ANS computes `sum_i -log2(p_i)` over a **binary** input under **static**
per-weight probabilities. That constrains the architecture, and the
constraints are not negotiable:

**Activations must be binarized.** So this is a binarized transformer.
Use the existing mechanism: compare a unit's code length against
`n_thr` learned thresholds to emit `n_thr` bits (`n_thr=1` is a
binarized unit, larger is a coarse staircase). See the "Hidden layer"
comment in `model.h`.

**Attention scores cannot be ANS.** `QK^T` multiplies two *activations*.
ANS's probabilities are static parameters; making one a runtime value
costs a division per element on the encode side (see `ans_uabs.h` — the
reciprocals are precomputed precisely because the weights are static).
Compute scores conventionally.

**Softmax, LayerNorm and residuals cannot be ANS.** They need `exp`,
`sqrt` and signed accumulation. Keep them fp32.

So the honest object is: **a binarized transformer in which the six
weight matmuls per layer (Q, K, V, O, FF1, FF2) are ANS encode passes,
and everything else is conventional.** The float comparison must be the
*same binarized transformer with float matmuls* — otherwise you are
measuring binarization, not ANS. See section 6.

Suggested shape, sized so the weights exceed L2 and the matmuls
dominate: `d_model=512, n_head=8, d_ff=2048, n_layer=4, seq=256,
vocab=256` (byte level). That is `12*d^2` weights per layer, ~12.6 M
total.

## 3. You almost certainly do not need to train anything

Do not write a transformer backward pass in C. The questions that matter
are answerable without it:

- **Speed** needs only a forward pass with weights of a realistic
  distribution. Random `w ~ U(-2,2)` log-odds is fine; the kernel's cost
  does not depend on the values.
- **Fidelity** is "does the ANS forward pass reproduce the float forward
  pass on the same weights" — the `vs float` agreement column `infer.c`
  already reports. Measure per-layer relative error and final logit
  KL/top-1 agreement.

Training is Phase 3 and optional. If you do want a task metric, train in
float first and post-training-quantize — `README.md` measures that at
≤0.17 points on the MNIST model, so it is not where the risk is.

## 4. Build it in this order, and stop early if Phase 1 settles it

**Phase 0 — harness.** `transformer.{h,c}`: config struct, weight
allocation, fp32 reference forward pass (no ANS). Verify it against a
20-line NumPy/Python check on a fixed seed so you know the reference is
right before anything is compared to it. Add to the Makefile as
`ans_transformer`, matching how `ans_bench_float` is wired in.

**Phase 1 — the measurement, and the actual deliverable.** Add the ANS
linear layer and benchmark *the linear layers alone*, per token:
`ans_uabs_blk_avx512` against the fp32 GEMV and GEMM already in
`bench_float.c`, plus int8 (section 6). Two regimes, and they will not
agree:

- **batch 1 / autoregressive decode** — weight-bandwidth bound, which is
  where uABS was competitive on the MNIST net
- **batch 32 / prefill** — compute bound, where it was not

If Phase 1 shows a large loss in both regimes, that is the answer. Write
it up and stop; do not build the rest to be thorough.

**Phase 2 — whole forward pass.** Assemble the full block with attention
and LayerNorm in fp32, and report end-to-end µs/token plus what fraction
of it the matmuls are. A large non-matmul remainder changes the
conclusion and is worth knowing.

**Phase 3 — optional, only if asked.** Train something. A char-level LM
on a small corpus gives bits/char, which is pleasingly also a code
length.

## 5. Measurement protocol

Copy it from `bench_float.c`; it is already correct there.

- **Pin to an idle core with an idle SMT sibling.** A Firedancer
  validator may be running on this box. Check `ps -eLo psr,pcpu` and
  `mpstat -P <core>,<sibling> 1 2`; core 56 (sibling 120) was idle when
  this was written, but verify. `taskset -c 56 ./ans_transformer`.
  Benchmarking on a core whose sibling is busy costs ~35% and is noisy.
- Best of 3+ trials, each running long enough to be stable (~1 s).
- Report µs/token, Mweight/s, and **which ceiling binds**, using the
  table in section 7. A number without a bound is not a result.
- `make -j` defaults to `CC=cc -O3 -march=native`. Do not mix compilers
  between the two sides of a comparison.

## 6. Baselines — get these right or the whole thing is worthless

| baseline | why it is the fair one |
|---|---|
| **fp32 AVX-512 GEMV/GEMM** | already in `bench_float.c`, reuse it verbatim |
| **int8 VNNI (`vpdpbusd`)** | **the one that matters.** A tANS level is one byte, so int8 weights carry the same information. VNNI does 64 int8 MACs per instruction against uABS's 16 weights per 6 instructions, at half the bytes. Expect it to win decisively. |
| binarized weights (XNOR + popcount) | **do not use as a baseline and do not binarize the weights.** ANS weights are multi-bit log-odds; a 1-bit-weight scheme is a different model with far worse quality, and no coder competes with 512 MACs per instruction pair. |

Both sides must run the **identical binarized numerics**. Same
thresholds, same activation bits, same everything except how `W x` is
evaluated. If the two models differ in quality, the comparison is
broken.

## 7. What to expect, and what would mean you have a bug

Measured on this machine (EPYC 9555P, core 56) during the work that
produced `FINDINGS.md`. These are properties of the core:

| | |
|---|---|
| fp32 AVX-512 FMA peak | 137.6 GMAC/s |
| stream read, 0.5 / 1.5 / 16 MiB | 264 / 139.5 / 129 GB/s |
| 16-lane gather, 1–16 KiB / 4 MiB table | 5.4 / 2.4 G lanes/s |
| uABS encode, across neurons | 33 G weights/s (6 instructions per 16 weights) |
| fp32 GEMV batch 1 / GEMM batch 64 | 29.5 / 66 G weights/s |

Predicted: **uABS within ~10% of fp32 at batch 1** (fp32 is bandwidth
bound there and wasting most of its FMA throughput), **uABS 2–5x slower
batched**, and **int8 VNNI ahead of both everywhere**.

Treat as a bug, not a discovery:

- uABS beating fp32 by a wide margin anywhere → the float baseline is
  not vectorizing, or is being eliminated.
- Either side's time not scaling with `d_model^2` → something is cached
  or hoisted out of the loop.
- uABS faster batched than at batch 1 → you are reusing state across
  tokens that should be independent.

## 8. Pitfalls that actually cost time here

- **Dead code elimination ate two benchmarks during this work.** If the
  accumulated result is never read, the whole loop vanishes and you get
  impossible throughput. Consume every accumulator — store it, or assign
  to a `volatile`. Sanity-check against a hand-computed op count.
- `aligned_alloc` requires the size to be a multiple of the alignment.
  `aligned_alloc(64, 784)` is UB and silently corrupted a benchmark.
- Group-unrolled loops need a remainder path. `for(g=0; g<n_grp; g+=U)`
  reads and writes out of bounds when `U` does not divide `n_grp`; see
  how `ans_uabs_blk_avx512` handles the tail.
- **Chain count is the dominant tuning knob**, not anything clever.
  `ANS_UABS_UNROLL=8` was worth 40% over 4 and over 10 (which spills).
  Sweep it for any new kernel before concluding anything about it.
- Add a `test_*` for every new kernel, following `test_uabs` in
  `test.c`: vector result must equal a scalar reference exactly. Do not
  benchmark a kernel you have not first proven correct.

## 9. Deliverable

A short section appended to `FINDINGS.md` — tables and a verdict, in the
voice of the ones already there. State what bound each number. A
negative result stated precisely is the expected and acceptable outcome;
the thing to avoid is an imprecise one.
