# ans_network

A small neural network whose every unit is an ANS encoder. Training and
inference are both plain C. Nothing here is wired into the Firedancer
build — it lives at the tree root as a standalone project.

## The idea

Encoding a symbol with rANS maps the state `x -> x*M/f`, so the log of
the state accumulates code lengths. Encode a binary feature vector
under a product-of-Bernoullis model and the bits you get out are

```
codelen(x) = sum_i -log2( x_i ? p_i : 1-p_i )
           = -( w.x + b ),   w_i = log2( p_i/(1-p_i) ),  b = sum_i log2(1-p_i)
```

which is a linear layer, evaluated in the log domain, in integer
arithmetic, with no floating point anywhere. Three things follow.

**Negative weights are free.** `w_i` is a log-odds and sweeps all of ℝ
as `p_i` sweeps (0,1). No sign tricks, no two-accumulator scheme.

**The activation function is already there.** Renormalization counts
the bits shifted out of the state. Comparing that count against a
threshold is a step function, and is all the hidden layer does.

**Classification needs no logarithm.** Every accumulator sits in the
same octave, so comparing raw `(bit count, state)` pairs
lexicographically compares code lengths exactly. The predicted class is
literally *whichever encoder emitted the fewest bits*.

The network is 784 binary pixels → hidden units → 10 class encoders.
Set `--hidden 0` and the hidden layer disappears, leaving a
discriminatively trained naive Bayes.

## Results

MNIST, pixels binarized at 128. `float` is the sparse dot product the
identity above says the encoders compute; the other two are the integer
ANS backends. "vs float" is how often a backend picks the same class as
the float reference.

| model | float | tans | rans | vs float | weights |
|---|---|---|---|---|---|
| `--hidden 0` (linear) | 90.65% | 90.48% | 90.63% | 99.0 / 99.9% | 7.9 K |
| `--hidden 512 --thr 1` | 96.63% | 96.56% | 96.49% | 99.5 / 99.6% | 407 K |
| `--hidden 256 --thr 4` | 96.99% | 97.06% | 96.99% | 99.6 / 99.6% | 211 K |

Quantizing to the ANS tables costs at most 0.17 points. The `--thr 4`
model runs at 1.95 G encode steps/s single threaded, 108 µs per image
dense over every weight.

## Backends

Six, all computing the same code lengths. `tans`/`rans` stripe one
neuron's features across `ANS_LANES` accumulators; the `blk`/`avx512`/`uabs`
backends run `ANS_BLK` neurons side by side with one accumulator each. End to
end over the 10k test set, one core of an EPYC 9555P:

| backend | 512 units | 256 units x 4 | Mencode/s |
|---|---|---|---|
| float reference (sparse dot) | 30.8 µs | 17.6 µs | — |
| `tans` lanes | 429 µs | 247 µs | 855–947 |
| `rans` mulhi reciprocal | 1333 µs | 707 µs | 299–305 |
| `tans` blocked scalar | 620 µs | 333 µs | 633–656 |
| `tans` avx512, 8 MiB lut | 203 µs | 116 µs | 1822–2007 |
| `tans` avx512, 4 MiB lut | 187 µs | 108 µs | 1947–2179 |
| `uabs` avx512, no table | **15.6 µs** | **12.2 µs** | **17336–26080** |

All six land within 0.15 points of each other in accuracy. `uabs` is the
one that is not table driven at all, and is a factor of 12 faster for
it; the next two sections are why.

## Removing the division

rANS needs `x/freq`. Three ways out, all built and priced against each
other by `ans_test` (AMD EPYC 9555P, one idle core, 8-way interleaved):

| encode step | ns/step | speedup |
|---|---|---|
| `ans_rans_put_div` — integer divide | 6.39 | 1.00x |
| `ans_rans_put` — precomputed reciprocal, `mulhi`+`shr` | 4.45 | 1.44x |
| `ans_rans_put_shift` — power-of-two frequency, plain `shr` | 3.83 | 1.67x |
| `ans_tans_put` — table lookup | 1.40 | 4.56x |

The reciprocal (Giesen's `rans_byte.h` construction) recovers most of
the gap and is checked exactly against the divide for all 65535
frequencies. Forcing frequencies to powers of two buys a little more but
would cost far more in weight resolution than it is worth, so the model
quantizer does not do it — the shift path exists to price it.

tANS wins because the renormalization *is* a shift by a count the table
supplies: the step is an add, two shifts and a 16-bit load, no multiply
and no divide. It pays about 0.12% in code length for that, which is
within the noise of the weight quantization.

## AVX-512

`ans_avx512.c` vectorizes tANS only. The axis matters more than the
instruction set: the scalar path stripes one neuron's features across
lanes, but vectorizing across **neurons** instead makes the input bit a
broadcast scalar, makes the 16 level bytes one contiguous load (given
the blocked `[group][feature][16]` layout), and gives each lane a whole
neuron's accumulator — so the result is the exact non-interleaved code
length rather than a sum of lane code lengths.

Two things had to change for the gather:

**One table, not two.** The scalar encoder chases two dependent loads,
the symbol transform and then the state table. A gather can only chase
one, so both fold into `lut[((level*2+sym) << ANS_TANS_LOG) + state]`
holding `(bits << 16) | next state`. That is 8 MiB. Squeezing both
fields into 16 bits halves it to 4 MiB, and each lane then picks its
half of the gathered dword with a variable shift — four extra vector
ops that pay for themselves.

**Four groups in flight.** One group alone leaves the whole vector
waiting on the gather that produced its own next index. Isolated encode
step cost, 512 neurons over 784 features:

| `ANS_AVX_UNROLL` | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| Mencode/s (4 MiB lut) | 684 | 1252 | 1590 | **1947** | 1356 | 1921 |

4 is the default; 6 loses to register pressure. The leftover groups are
finished by the same code at one group wide, so `n_grp` need not be a
multiple of it. Measured against the blocked scalar path on the same
table, the vector version is **3.4x** (0.52 vs 1.77 ns/step); against
the previously fastest scalar backend it is **2.3x** end to end.

That is well short of 16x because the loop is bound by gather latency
into a multi-megabyte table, not by arithmetic. The table size is the
lever, and the next section measures it.

## uABS, or deleting the table

The alphabet here is binary, which rANS and tANS both treat as a
special case of something general. uABS is the coder built for it
directly. With `p = P(1)`:

```
decode  s = ceil((x+1)p) - ceil(xp)
        x = s ? ceil(xp) : x - ceil(xp)
encode  C(x,1) = floor( x/p )
        C(x,0) = ceil( (x+1)/(1-p) ) - 1
```

The state is *multiplied* by `1/p_s`. There is no state table, so there
is nothing to look up — and a lookup is exactly what the vectorized
tANS encoder is bound by. Measured on one core, a 16 lane gather
sustains 2.4 G lanes/s from a 4 MiB table and 5.4 G even from 1 KiB,
against 31 G weights/s for a dense fp32 GEMV over the same layer. No
table-driven design gets past that ceiling; shrinking the table only
moves it.

So the weight is stored as the reciprocal it is about to be multiplied
by, `R_s = round(2^8/p_s)`, a `ushort`, in two planes — one per symbol,
blocked `[group][feature][16]` like the levels. The input bit selects a
plane and stays a broadcast scalar, and a group's 16 reciprocals are one
contiguous 32 byte load. The whole step is six instructions:

```
q  = x*R          load 32 B, widen, multiply
sh = 16 - lzcnt(q)
x  = q >> sh      back into [2^15, 2^16)
nbit += sh - 8
```

and `sum -log2 p = log2(x_end/x_0) + sum(sh_i - 8)`, the same
`(bit count, state)` pair the other backends read out, so `ans_acc_lt`
still picks the class with no logarithm.

`ANS_UABS_UNROLL` is the number of groups kept in flight, and is a
build knob for the same reason `ANS_AVX_UNROLL` is. It wants to be
deeper than the tANS kernel's four — that one is waiting on gather
latency, this one is issuing cheap work and needs more of it to fill
the ports:

| `ANS_UABS_UNROLL` | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| Mweight/s | 23775 | 26227 | **33402** | 26606 |

**What it costs.** Using a rounded reciprocal for both branches rather
than the exact `C(x,0)` gives up the bijection: the two symbols' images
no longer tile the state space, so this encoder cannot be inverted.
Nothing here decodes — only the code length is read — and the code
length is what survives. `ans_uabs_exact_put` is the true bijective
step, and `ans_test` checks the relaxed one against it, prices both
against the ideal, and verifies the exact step round trips over 524288
(state, probability) cases.

**What it buys.** Accuracy, as well as speed. A `ushort` reciprocal
resolves the weight about eleven times finer than the 256 level
quantizer at its worst point and far finer typically, and there is no
spread to go wrong:

| | tANS, log 12 | uABS |
|---|---|---|
| code length excess | −0.081% | **−0.002%** |
| agreement with the float model | 99.53% | **99.95%** |
| accuracy, 512 units | 96.53% | 96.66% |
| accuracy, 256 x 4 | 97.05% | 97.02% |
| bytes per weight | 1 + 4 MiB lut | 2 read, 4 held |

The `ANS_TANS_LOG` and `ANS_TANS_LEVEL_CNT` knobs, and the whole
"lower precision is the win" tradeoff two sections down, simply do not
apply to it: there is no table to size.

## Against an optimized float model

The `float reference (sparse dot)` row above is the right reference for
accuracy and the wrong one for speed — it walks only the ~19% of pixels
that are set, while every ANS backend walks every weight. `ans_bench_float`
runs the honest comparison: the same network as a dense fp32 GEMV with
AVX-512 FMA, eight rows in flight, which is what an optimized
implementation of this model would be. Hidden layer only, 401408
weights per image, 512 units:

| kernel | µs/image | Mweight/s | bound by |
|---|---|---|---|
| `tans` avx512, 4 MiB lut | 170.7 | 2352 | gather throughput |
| **`uabs` avx512, no table** | **12.2** | **32930** | **issue rate, ~2.9 ops/cycle** |
| fp32 AVX-512 GEMV | 13.6 | 29456 | weight bandwidth, 118 GB/s |
| fp32 AVX-512 GEMM, batch 64 | 6.1 | 66307 | FMA, 48% of peak |
| fp32 sparse dot (set pixels) | 19.5 | 20629 | — |

Both single-image kernels are at a machine limit rather than an
implementation limit, and it is a different limit for each. The GEMV
streams 1.53 MiB of weights per image at 118 GB/s against a measured
139.5 GB/s ceiling for that footprint, and five row/accumulator
blockings land within 8% of each other. uABS issues six instructions
per 16 weights and reaches 7.7 weights/cycle, while touching half the
memory — 2 bytes per weight against 4.

So the integer coder edges out the float kernel at batch 1 and loses to
it 2x batched, because batching amortizes weight traffic and weight
traffic is not what binds uABS. Run the shipped kernel over a batch of
64 and it is flat at 13.2 µs/image — a reciprocal is reloaded for every
image, and nothing is reused. The genuinely batched form turns the
vector axis around, lane k being image k and one neuron at a time, so a
reciprocal becomes a scalar shared by 16 images and the plane select
becomes a blend against the batch's bit mask:

| uABS, batch 64 | µs/image | B/weight read |
|---|---|---|
| across neurons, per image | 13.2 | 2 |
| across images, 4 chains | 17.2 | 0.125 |
| across images, 8 chains | **12.8** | 0.125 |

A 16x cut in weight traffic buys 4%. That is the whole argument in one
line: uABS is bound by the six instructions it issues per 16 weights,
not by the memory it reads, so there is nothing for a batch to
amortize — while fp32 goes from 13.6 to 6.1 µs/image over the same
batch by amortizing exactly that. Chain count is what the batched form
is actually sensitive to; 8 is the sweet spot either way, and 16 spills.

End to end over the whole network the two are
a tie at 512 units (15.9 µs vs 15.7 µs) and the float model wins at
256 x 4 (12.3 µs vs 8.4 µs), where a fourth as many hidden units leave
the scalar threshold and output-layer loops a larger share of the work.

## Precision

Two knobs feed the table, and they do different things.
`ANS_TANS_LOG` sets the number of coder states, `ANS_TANS_LEVEL_CNT`
the number of distinct weights. Both are `-D` overridable; the table is
their product, `ANS_TANS_LEVEL_CNT * 2 * ANS_TANS_SIZE` entries.

`ANS_TANS_LOG` controls how faithfully the coder tracks the ideal code
length, and nothing else. Excess halves per bit of table, independent
of the level count:

| `ANS_TANS_LOG` | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|
| code length excess | −0.295% | −0.158% | −0.079% | −0.041% | −0.020% |
| accuracy (512-unit model) | 96.57% | 96.61% | 96.53% | 96.60% | 96.60% |

Accuracy does not move, because the network only needs code lengths
*relative* to each other and the error is a smooth systematic bias, not
noise. Paying for 13 or 14 is pure loss.

`ANS_TANS_LEVEL_CNT` is the one that touches accuracy, and it saturates
early. 32 levels costs about 0.4 points, consistently at every table
size; 64 and up are indistinguishable:

| levels | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| weight step (log2 odds) | 0.516 | 0.254 | 0.126 | 0.063 |
| accuracy | 96.22% | 96.68% | 96.54% | 96.53% |
| agreement with float | 98.33% | 98.93% | 99.27% | 99.53% |

Agreement with the float reference keeps climbing while accuracy does
not: past 64 levels the extra disagreements are on samples the float
model gets wrong anyway.

Speed tracks the table footprint almost exactly, which is what a
gather-bound loop should do. Both models, packed table:

| `LOG` / levels | table | 512-unit | 256x4 | accuracy |
|---|---|---|---|---|
| 12 / 256 (default) | 4.00 MiB | 2264 | 1926 | 96.53 / 97.05% |
| 12 / 64 | 1.00 MiB | 2835 | 2515 | 96.68 / 96.95% |
| 11 / 64 | 0.50 MiB | 3003 | 2787 | 96.66 / 96.90% |

So: **lower precision is the win, up to a floor.** Dropping to 64
levels shrinks the table 4x and buys 25–31% for no accuracy change
either model can distinguish (±0.15 points, under one standard error on
10k samples). The default stays at 256 because it is the most faithful
to the float model it was quantized from — `-DANS_TANS_LEVEL_CNT=64` is
a one-flag change when throughput matters more.

Levels are stored one byte per weight, so 512 would silently wrap. That
is now a `_Static_assert` rather than something to find by watching
accuracy fall to chance.

## What lzfse's FSE does, and why it is not used here

lzfse's encoder updates its state with no state table at all:

```c
int hi = s >= s0;  nbits = hi ? k : k-1;  delta = hi ? delta0 : delta1;
*pstate = delta + (s >> nbits);
```

Four `int16` per symbol, 4 KiB for the whole level bank here instead of
4 MiB — it would delete the gather that this code is bound by. It was
worth measuring, and it does not transfer.

The affine update is only valid if each symbol owns a *contiguous*
range of states, which is a spread choice, and the spread is exactly
what this application depends on. Encoding 784 symbols, each under a
different table (as every neuron does):

| spread | mean excess | worst |
|---|---|---|
| Bresenham (used here) | −0.081% | 2.1 bits |
| FSE odd-stride | −0.346% | 8.4 bits |
| contiguous (lzfse) | **−6.245%** | **137 bits** |

Reimplementing lzfse's affine encoder directly gives −6.229% / 139
bits, matching the contiguous row, which confirms the cost is the
spread and not the arithmetic.

For a compressor this barely matters: the bijection still round trips
and the bits still come out. Here the code length *is* the neuron's
output, and a 137-bit error against class separations of tens of bits
is fatal. A scattered spread keeps the state near-uniform whatever
table comes next; a contiguous one confines it to a narrow range, and
with the table changing every symbol it never settles. lzfse can make
that trade because it encodes long runs under one table. This network
never encodes two symbols under the same table in a row.

## Where the floats are, and are not

Nothing below `ans_net_compile` touches a float. The compiled network
holds levels as `uchar` (1 byte per weight), rANS frequencies as
`ushort`, and thresholds as `int` in Q8 bits. There are no biases at
inference at all: a hidden unit's input-independent term is resolved
into its integer threshold at compile time, and the output layer's is
simply what its encoder emits when the feature bits are clear.

The floats live in `ans_model_t` and the `.ans` file, which are the
*training* parameters. Two reasons they stay float there:

- Adam moves a weight by about `lr` = 0.02 in log-odds per step, against
  a level spacing of `2*ANS_W_MAX/255` = 0.063. A single step is a
  third of a quantum and would round to zero; near convergence, far
  less. The optimizer's moment estimates are continuous by construction
  too. This is the usual shadow-weight arrangement.
- A model reloaded to continue training would otherwise resume from the
  rounded weights and lose everything below the quantum.

The file could store levels instead and shrink 4x, at the cost of being
a terminal artifact. It is worth noting this is plain post-training
quantization — the forward pass trains on unquantized weights — and it
costs at most 0.17 points, so quantization-aware training would have
little left to recover.

## Build and run

```sh
make            # ans_train, ans_infer, ans_test, ans_bench_float
make data       # fetch MNIST (without it, everything runs on a synthetic set)
make check      # correctness + the encode step benchmark
make demo       # fetch, train, evaluate

./ans_train --hidden 256 --thr 4 --epochs 40 --verbose
./ans_infer  --model model.ans
./ans_bench_float --model model.ans      # ANS against a dense fp32 GEMV
```

`ans_train` takes `--hidden --thr --epochs --batch --lr --thresh --seed
--data --out --verbose`. Training is OpenMP parallel over units; build
with `OPENMP=` to drop the dependency. The AVX-512 paths are compiled
in when the target has them (`-march=native` by default) and fall back
to the blocked scalar path otherwise. `ANS_AVX_UNROLL`, `ANS_TANS_LOG`
and `ANS_TANS_LEVEL_CNT` are all `-D` overridable, e.g.
`make CFLAGS="-O3 -march=native -DANS_TANS_LEVEL_CNT=64"`.

## Files

| | |
|---|---|
| `ans.h`, `ans.c` | the rANS and tANS backends, table construction, Q8 log2 |
| `ans_avx512.h`, `ans_avx512.c` | across-neuron tANS: blocked scalar and two AVX-512 variants |
| `ans_uabs.h`, `ans_uabs.c` | uABS: the table-free binary coder, scalar and AVX-512 |
| `model.h`, `model.c` | model definition, float reference, quantizer, integer inference |
| `data.h`, `data.c` | MNIST idx loader, synthetic fallback |
| `train.c` | backprop + Adam, straight-through thresholds |
| `infer.c` | evaluate every backend, report accuracy and throughput |
| `test.c` | correctness suite and the encode step benchmark |
| `bench_float.c` | the same network as a dense fp32 GEMV, for the comparison above |

`TANS_CODEC.md` is separate: a standalone reference for building a correct
tANS codec (spread construction, encoder and decoder, the lzfse analysis).
Nothing in `ans_network` decodes, so none of it is wired in here.

`FINDINGS.md` is the verdict: what pricing all of this against an
optimized float model showed, the machine limits it was measured
against, and where ANS belongs instead.

## Design notes

**Thresholds are stored relative to the unit's bias.** A unit's code
length has a large input-independent part, `sum_i softplus2(w_i)`, on
the order of hundreds of bits, and it drifts as the weights train far
faster than an absolute threshold can follow. Left absolute, all but
~18 of 256 hidden units saturate and the network never beats the linear
model. Storing `tau = sum_i softplus2(w_i) + taur` and learning `taur`
fixes it (222 of 256 units live, 89% → 96%), and is only a
reparameterization: the machine still compares one whole code length
against one absolute integer. It also cancels the dense half of the
hidden gradient, leaving it entirely sparse.

**The compiled threshold is per backend.** The two backends round to
different frequencies and so disagree about that input-independent
term. `ans_net_compile` resolves each threshold against the frequencies
the backend actually got, rather than against the float weights.

**Inference interleaves 8 accumulators per unit.** Each encode step
depends on the state the last one produced, so one accumulator runs at
the latency of a dependent load. The stripes cover disjoint features, so
their code lengths just add. The *output* layer deliberately does not
interleave — keeping one accumulator per class is what allows the
lexicographic decision.

**tANS uses a Bresenham spread, not FSE's.** FSE's odd-stride walk is
tuned for multi-symbol alphabets; on the skewed binary sources here it
scatters the rare symbol unevenly and costs 0.34% in code length against
0.08% for an even spread. See the lzfse section for how far this can go
wrong.

**The output layer has one always-set feature.** Pinning the bias to the
weights would leave the classes without a free prior. A constant feature
contributes `-log2 p` to a class independently of everything else, which
is exactly the label encoded as a symbol of its own before the data.

## What this is and isn't

Accuracy is whatever the equivalent quantized linear/binarized model
gets; ANS does not make the network better. What it buys is mechanical:
integer inference with no floating point and (for tANS) no multiply or
divide, a vector form that reaches 2.2 G encode steps/s on one core, automatic dynamic range with no per-layer scale calibration,
weights executed in the form they are stored in, and a decision rule
that never leaves integer comparison. With uABS that costs nothing
against an optimized fp32 kernel at batch 1, and half the weight
memory. The forward pass is also a
bijection, which is the entry ticket to the established work in this
area — bits-back coding (BB-ANS, HiLLoC, Bit-Swap) and integer discrete
flows, where ANS and neural networks genuinely belong together.

The hidden layer is a binarized net trained with a straight-through
estimator, so it is bound by that, not by ANS: no batch norm, no
augmentation, no convolutions, ~40 epochs.
