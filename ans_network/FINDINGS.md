# What pricing this thing against ordinary code actually showed

`README.md` documents the network and its backends. This is the
separate question of whether any of it is a good idea, measured rather
than argued. One core of an EPYC 9555P, core 56 with its SMT sibling
idle, `ans_bench_float` and the probes described below.

## 1. ANS is not an arithmetic unit

The network's premise is the identity in `README.md`: an ANS encoder's
code length *is* a linear unit's pre-activation, so a forward pass can
be an encode pass, in integers, with no floating point. That works, and
it is the wrong tool.

Hidden layer only, 512 units x 784 features, 401408 weights per image:

| kernel | µs/image | Mweight/s | bound by |
|---|---|---|---|
| `tans` avx512, 4 MiB lut | 170.7 | 2352 | gather throughput |
| `uabs` avx512, no table | 12.2 | 32930 | issue rate, ~2.9 ops/cycle |
| fp32 AVX-512 GEMV | 13.6 | 29456 | weight bandwidth, 118 GB/s |
| fp32 AVX-512 GEMM, batch 64 | 6.1 | 66307 | FMA, 48% of peak |

The best integer coder ties the float kernel at batch 1 and loses 2x
batched. The reason is not implementation slack on either side — both
single-image kernels sit at a machine limit, and section 3 measures
which one. It is that the step being performed is a log-domain
multiply-accumulate, and the machine has a single instruction that does
sixteen of those. uABS needs six instructions for the same sixteen.
Nothing closes a 6:1 instruction ratio.

uABS reaches parity at batch 1 only because the float kernel is
bandwidth-bound there and wasting most of its FMA throughput. Batching
hands that back: fp32 goes 13.6 -> 6.1 µs/image, uABS goes 13.2 -> 12.8
even after cutting its weight traffic 16x (see `README.md`, "Against an
optimized float model"). A 16x traffic cut buying 4% is the cleanest
evidence that instruction count, not memory, is what binds it.

**The narrow claim this supports:** ANS is not a substitute for a
matmul. It says nothing against ANS as a coder, because this project
never decoded — the direction ANS is actually known for went untested
from start to finish.

## 2. What ANS did win on, which was not speed

uABS beat tANS on accuracy while beating it 14x on speed, and beat the
float model it was quantized from:

| | tANS, log 12 | uABS | float |
|---|---|---|---|
| code length excess | −0.081% | **−0.002%** | — |
| agreement with the float model | 99.53% | **99.95%** | — |
| accuracy, 512 units / 256 x 4 | 96.53 / 97.05% | 96.66 / 97.02% | 96.63 / 96.99% |

A `ushort` reciprocal resolves a weight about 11x finer than the 256
level quantizer at its worst point, and there is no spread to get
wrong. The lesson is about quantizer design, not about ANS: the coder
that needed no table was also the one that needed no approximation.

## 3. The machine limits everything was measured against

Worth keeping because they are properties of the core, not of this
project, and they decided every question above.

| probe | result |
|---|---|
| fp32 AVX-512 FMA peak | 137.6 GMAC/s (2 x 16 lanes x 4.3 GHz) |
| stream read, 0.5 MiB (L2 resident) | 264 GB/s |
| stream read, 1.5 MiB | 139.5 GB/s |
| stream read, 16 MiB | 129 GB/s |
| 16-lane gather, 1–16 KiB table | 5.4 G lanes/s (2.97 ns, 12.8 cyc) |
| 16-lane gather, 1 MiB table | 3.3 G lanes/s |
| 16-lane gather, 4 MiB table | 2.4 G lanes/s (6.59 ns, 28.3 cyc) |
| 16-lane gather, 16 MiB table | 2.2 G lanes/s |

Gathers saturate at 6 independent streams; 4 gets within 8%.

The `tans` avx512 kernel runs at 2391 Mweight/s against the 2429 its
table size allows — 98% of the ceiling, i.e. no implementation slack.
The fp32 GEMV streams at 118 GB/s against 139.5 — 85%, and five
row/accumulator blockings land within 8% of each other. Both numbers in
section 1 are the machine's, not the code's.

## 4. Where ANS does belong

Ranked by how well the evidence here supports them. None of these are
measured; they are what the measurements imply.

**Anywhere the bits are wanted.** The entropy backend: FSE in Zstd,
LZFSE, Oodle, JPEG XL. The direction that matters there is decode — a
table lookup and a shift, no division — which is why ANS displaced
arithmetic coding, and which this project never exercised. The
`ANS_LANES` interleave is how real ANS codecs get SIMD throughput,
since one stream is inherently serial.

**uABS specifically: binary alphabets under a context model.** Model a
bit from context, code a bit. That is CABAC, LZMA's range coder, and
bit-plane coders. uABS replaces the range coder there and turns its
division into a multiply. In this tree: occupancy and tombstone
bitmaps, presence vectors, sparse index payloads.

**Determinism, in consensus.** Float summation is order dependent and
non-associative; integer ANS accumulation is bit exact on any machine.
Solana's runtime bans floats for that reason. This is the one setting
where ANS-as-an-ALU — the thing that lost on speed above — has an
argument, because the competition is not fp32 SIMD, it is a hand rolled
fixed point log table.

**Neural networks and ANS, the other way round.** Bits-back coding
(BB-ANS, HiLLoC, Bit-Swap), integer discrete flows: the network
supplies the probabilities and ANS codes real data, so the bitstream is
the point. That is the inverse of what this project built, and it is
where the two genuinely belong together.

## 5. The one result that transfers out of this directory

For any table-driven codec, `fd_zle` variants included:

> A per-symbol 16-lane gather caps at 2.4 G lanes/s from a 4 MiB table
> and 5.4 G even from 1 KiB, against ~31 G/s for streaming contiguous
> memory. Shrinking a table moves that ceiling; it does not remove it.

So if a hot loop's inner step is a gather, table size is a second order
knob. The first order question is whether the lookup can become
arithmetic at all — which is the whole of what uABS did here, and it
was worth 14x.

A caveat in the other direction, for `fd_zle` specifically: it is the
opposite design point, zero run length only with no entropy stage and
about 10x faster than Zstd at matching ratio on account data. Adding an
ANS stage trades away exactly what makes it good. If it fits anywhere
it is a second tier — snapshots written once and shipped to many peers,
where ratio is worth CPU — and the ratio gain would have to be measured
before any of it means anything.

## 6. The same question on a transformer

Section 1 priced one hidden layer. A transformer is the harder case,
because only part of one can be an encode pass at all: `QK^T`
multiplies two activations, and an ANS probability is a static
parameter, so scores, softmax, LayerNorm and the residuals stay fp32.
What is left is the six weight matmuls per layer plus the output
projection. `transformer.{h,c}` and `bench_transformer.c` build exactly
that — a binarized transformer, `d_model` 512, 8 heads, `d_ff` 2048, 4
layers, 12.7 M weights — and evaluate those matmuls three ways with
everything else held identical: uABS encode, fp32 AVX-512 GEMV/GEMM
(the kernels from `bench_float.c` verbatim), and int8 VNNI. Weights are
random; nothing is trained, because neither speed nor the fidelity
question depends on it.

The three paths really are the same network: `ans_transformer --check`
holds each kernel against a scalar reference, and the first
binarization site — the only one where all three read an identical
input — disagrees on 0 of 131072 decisions. `check_ref.py` reproduces
the fp32 reference forward pass in plain Python from the same seed,
matching all 1056 binarization decisions and the logits to 1.6e-7.

**The linear layers, L2 resident.** One `d_model x d_model` matrix, 1
MiB of weights, so nothing waits on memory and what shows is the
instruction ratio:

| kernel | batch | µs/token | Gweight/s | bound by |
|---|---|---|---|---|
| uabs encode, across neurons | 1 | 10.6 | 24.8 | issue rate, 73% of its 34.0 ceiling |
| fp32 avx512 gemv | 1 | 8.1 | **32.5** | load bandwidth, 130 of 196 GB/s |
| int8 vnni gemv | 1 | 1.9 | **136.5** | the horizontal reductions |
| uabs encode, across neurons | 32 | 10.2 | 25.7 | issue rate |
| uabs encode, across tokens | 32 | 8.1 | 32.2 | issue rate, at the ceiling |
| fp32 avx512 gemm | 32 | 4.1 | 63.5 | FMA, 46% of peak |
| int8 vnni gemm | 32 | 1.5 | **179.6** | the horizontal reductions |

**The linear layers, streamed.** Every weight in the model once per
token, 48.5 MiB, which is what autoregressive decode does:

| kernel | batch | µs/token | Gweight/s | GB/s | bound by |
|---|---|---|---|---|---|
| uabs encode, across neurons | 1 | 1458.8 | 8.7 | 26 | neither: see below |
| fp32 avx512 gemv | 1 | 878.0 | **14.5** | 59 | memory, at the stream rate |
| int8 vnni gemv | 1 | 109.6 | **116.0** | 116 | memory, 87% of L3 stream |
| uabs encode, across neurons | 32 | 688.6 | 18.5 | — | weight reuse, then issue rate |
| uabs encode, across tokens | 32 | 692.8 | 18.4 | 9.2 | issue rate |
| fp32 avx512 gemm | 32 | 263.1 | **48.3** | — | FMA, 35% of peak |
| int8 vnni gemm | 32 | 69.8 | **182.0** | 5 | the horizontal reductions |

uABS loses 1.7x at batch 1 and 2.6x batched to fp32, and 13x and 9.8x
to int8. The batch 1 loss is the interesting one, because on MNIST this
was a tie. Three ceilings, measured on this core, explain it:

| probe | result |
|---|---|
| single core sequential read, 0.5 / 1 / 4 MiB | 264 / 196 / 136 GB/s |
| single core sequential read, 12 / 48 / 192 MiB | 133 / 49.2 / 34.6 GB/s |
| uABS encode, 0.5 MiB of reciprocals | 34.0 Gweight/s, reproducing section 1 |

MNIST's hidden layer was 1.5 MiB. There uABS's 34 Gweight/s issue
ceiling and what fp32 could pull out of L2 were the same number, and
the two tied. At transformer scale the weights come from DRAM, and uABS
is worse *at memory* despite touching fewer bytes. It asks for 2 B per
weight against fp32's 4, but it reads a 32 byte half line from one of
two planes chosen per feature by the input bit, so a line is pulled
whenever either of its two features selects that plane: 3 B fetched per
2 B used, in streams that skip. It extracts 26 GB/s where the fp32 GEMV
gets 59 and a plain sequential read of the same footprint gets 49.
Fewer bytes do not help when the bytes are shaped wrong.

Two layout notes, both measured. The stride between the groups a chain
walks is `n_in*32` bytes, so any power-of-two width puts all eight
chains in one L1 set; padding `n_in` by 16 costs 3% more steps and buys
1.21x at 48 MiB and 1.37x at 1 MiB (`ans_transformer --stride 48`).
The tables use the natural layout. And chain count is again the only
tuning knob that matters: 8 is best for the across-neuron kernel except
in the streamed batch 1 case where 4 buys 14%, and 8 is best for the
new across-token kernel (4 and 12 lose 23%, 16 loses 42%) — the same
sweet spot section 1's kernel found.

The int8 kernels are the weakest-optimized of the three and their
numbers understate them: at `d_model` 512 a row is eight 64 byte loads
closed by a horizontal reduction, which is a comparable instruction
count, and neither memory (137 of 264 GB/s from 0.25 MiB) nor VNNI
issue (0.5 of 2 `vpdpbusd` per cycle) is near its limit. A blocked
kernel that reduces once per output block would widen the gap, not
close it.

**int8's win is not a cache artifact.** At 4 layers its 12.1 MiB fits
L3 while the other two do not. At 16 layers, 50.5 M weights, all three
stream and the ranking is exactly bytes per weight:

| kernel, 192.5 MiB fp32 / 48.1 MiB int8 | batch 1 | batch 32 |
|---|---|---|
| uabs encode | 7.1 Gweight/s | 19.1 |
| fp32 avx512 | 11.7 | 50.2 |
| int8 vnni | **53.0** | **164.6** |

**End to end.** The whole forward pass, sequence 256, with the fp32
remainder — LayerNorm, attention, softmax, residuals, binarization —
timed by running the same pass with the matmul removed:

| path | µs/token | matmul | remainder | matmul share |
|---|---|---|---|---|
| fp32 | 529.8 | 473.5 | 56.3 | 89.4% |
| uabs | 662.9 | 606.6 | 56.3 | 91.5% |
| int8 | **152.6** | 96.3 | 56.3 | 63.1% |

The remainder is 10.6% of the fp32 pass and 37% of the int8 one. That
is the ceiling on what any coder can win: at int8's speed the matmuls
have stopped being the problem, and a coder faster than int8
would be worth at most another 2.7x end to end.

**What ANS did win, again, was accuracy.** Same result as section 2, in
a second architecture. A `ushort` reciprocal resolves a weight about
11x finer than an int8 level, and the per-unit pre-activation error is
0.039 bits against int8's 0.112, 0.42% and 1.21% of the
pre-activation's own spread. In flipped binarization decisions against
the fp32 path, by site, layer 0 — the only layer whose input has not
already diverged:

| site | uabs | int8 | decisions |
|---|---|---|---|
| stream -> Q,K,V | 0.000% | 0.000% | 131072 |
| context -> O | **0.271%** | 0.413% | 131072 |
| stream -> FF1 | **2.979%** | 3.836% | 131072 |
| FF1 -> FF2 | **9.781%** | 11.877% | 524288 |

uABS is 25-30% fewer flips than int8 at every site, for free. It buys
nothing downstream: by layer 3 both paths flip 33-40% of decisions and
agree with the float model's top-1 on 2.7% and 0.8% of positions. That
is the binarized transformer being chaotic with untrained weights and
thresholds at the mean of a dense cluster, not a statement about either
quantizer — which is why only layer 0 is worth reading, and why the
site-by-site instrumentation was necessary to see it.

**Verdict.** The identity holds — the encoder reproduces the float
pre-activation to 0.42% of its spread, in integers, in a transformer —
and it is still the wrong tool. It was a tie at MNIST's layer size
because both kernels were at the same ceiling there; at transformer
size it loses 1.7x to fp32 and 13x to int8 at batch 1, and 2.6x and
9.8x batched. The batched form that turns the vector axis to tokens and
cuts weight traffic 16x measures 18.4 against 18.5 Gweight/s for the
form that does not — the same 4% nothing of section 1, for the same
reason. Nothing here changes section 1's claim or narrows it: ANS is
not a substitute for a matmul, and against VNNI it is not close.

The predictions in `TRANSFORMER_PLAN.md` were: uABS within ~10% of fp32
at batch 1, 2-5x behind batched, int8 ahead everywhere. The second and
third hold. The first was too generous, and the reason is the one thing
this exercise found that section 1 could not have: uABS's advantage in
bytes per weight is real and worthless, because its access pattern
cannot stream. The listed bug signatures were checked and none is
present — time scales with `d_model^2` at 256, 512 and 1024 (3.8-5.3x
per 4x weights, the excess being the footprint leaving L2), uABS is
faster batched only through weight reuse and not through any state
crossing tokens (the L2 resident batch is worth 4%, and the batched
kernel's output is bit-identical to the per-token one), and the float
baselines are at measured machine limits rather than unvectorized.
