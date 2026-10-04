# tANS codec notes

Reference for building a correct, fast tANS codec. This is **not** used by
`ans_network` — the network only ever encodes, discards the coded bits and
keeps the code length as its output, so it has no decoder. These notes came
out of designing one for a separate project, and the spread findings are the
same ones that drive `ans.c`, so they are kept here.

Everything below was verified numerically, not reasoned about on paper. The
checks are reproduced inline.

---

## 1. The correctness condition

A tANS decoder is valid iff, **for each symbol independently**, the
transitions it can produce tile the state space exactly once:

> for symbol `s`, the intervals `[e.state, e.state + 2^e.nbits)` over all
> slots carrying `s` must cover `[T, 2T)` with no gaps and no overlaps.

That single test catches essentially every construction bug. Worth keeping
as an assert in any table builder.

```python
def tiles(dec_tbl, freqs, table_log):
    T = 1 << table_log
    for s in range(len(freqs)):
        iv = sorted((e.state, e.state + (1 << e.nbits))
                    for e in dec_tbl if e.sym == s)
        cur = T
        for lo, hi in iv:
            if lo != cur: return False          # gap or overlap
            cur = hi
        if cur != 2*T: return False             # short or long
    return True
```

A second, weaker signal that is quick to eyeball: **`nbits` must take two
adjacent values per symbol** (`[k-1, k]`). If any symbol shows a single
value, fractional-bit coding is gone and what you have is a Huffman code.

---

## 2. The spread

Encoder and decoder must build from the **same** spread or nothing round
trips. Two representations, inverse to each other:

```
slot_sym[u]           # which symbol owns state slot u        -- decoder side
slot_of[cum[s] + j]   # the slot holding symbol s's j-th copy -- encoder side
```

### Even ("precise") spread — near optimal

```
spread(freqs):
    heap = min-heap of (key, sym, j)
    for s, f in enumerate(freqs):
        if f > 0: heap.push( key = 1/(2f), s, j = 0 )      # key = (2j+1)/(2f)
    for u in 0 .. T-1:
        (_, s, j) = heap.pop_min()
        slot_sym[u] = s
        j += 1
        if j < freqs[s]: heap.push( key = (2j+1)/(2*freqs[s]), s, j )
```

Compare keys by cross multiplication — `(2j+1)*f' < (2j'+1)*f` — to stay in
integers. For two symbols this degenerates to Bresenham, which is what
`ans.c` uses. O(T log n).

### Cheaper alternatives, and what they cost

FSE's odd-stride walk is O(T) and one line:

```
pos = (pos + (T>>1) + (T>>3) + 3) & (T-1)     # odd, so coprime with T
```

lzfse assigns each symbol a contiguous block and has no spread at all
(section 4).

Measured code length excess against the ideal, `table_log = 12`:

**Long stationary streams** (300k symbols, one table throughout — the normal
compressor case):

| alphabet | precise | fse-stride | contiguous |
|---|---|---|---|
| 2 symbols `[3000,1096]` | +0.000% | −0.000% | +2.371% |
| 2 symbols `[4000,96]` | +0.000% | +0.005% | +1.445% |
| 4 symbols | −0.000% | +0.000% | +1.379% |
| 8 symbols | +0.000% | +0.000% | +1.619% |
| 257 symbols, near flat | +0.000% | +0.000% | +0.000% |

Contiguous costs 1.4–2.4% on skewed alphabets and nothing at all when the
frequencies are all the same magnitude (one `k` for everyone, so the layout
stops mattering).

**A new table every symbol** (784 symbols, each under a different
distribution — what `ans_network` does, and the worst case for any spread):

| spread | mean excess | worst |
|---|---|---|
| Bresenham / precise | −0.081% | 2.1 bits |
| FSE odd-stride | −0.346% | 8.4 bits |
| contiguous | −6.245% | 137 bits |

The state never equilibrates, and a scattered spread is what keeps it
near-uniform whatever table comes next. Reimplementing lzfse's affine
encoder directly gives −6.229% / 139 bits, matching the contiguous row,
which confirms the cost is the spread and not the arithmetic.

---

## 3. The codec

Verified round trip on `[3000,1096]`, `[4000,96]`, `[1,4095]`, a 4-symbol and
an 8-symbol alphabet, under both the precise and fse-stride spreads. Bit
stream fully consumed in every case; code length within 0.001% of entropy
with the precise spread.

State lives in `[T, 2T)`, `T = 1<<table_log`.

### Tables

```
build(freqs):                       # requires sum(freqs) == 1<<table_log
    cum[s] = exclusive prefix sum of freqs
    slot_sym = spread(freqs)

    next = copy(cum)                                 # invert the spread
    for u in 0 .. T-1:
        s = slot_sym[u]
        slot_of[next[s]] = u
        next[s] += 1

    for sym, f in enumerate(freqs):                  # decoder table
        for j in 0 .. f-1:
            n = f + j                                # per-symbol counter, f .. 2f-1
            u = slot_of[cum[sym] + j]
            dec_tbl[u].sym   = sym
            dec_tbl[u].nbits = table_log - highbit(n)
            dec_tbl[u].state = n << dec_tbl[u].nbits # in [T,2T), low nbits zero

    for sym, f in enumerate(freqs):                  # encoder constants
        enc[sym].k     = table_log - highbit(f)
        enc[sym].limit = f << enc[sym].k             # in [T, 2T)
        enc[sym].bias  = cum[sym] - f
```

`n << nbits` lands in `[T, 2T)` directly, so there is no `- tableSize` to
undo — FSE needs it only because it indexes a `[0,T)` table and adds back.

### Encode

```
encode_init():  st = 1<<table_log

encode(st, sym):
    e  = enc[sym]
    nb = (st < e.limit) ? e.k - 1 : e.k              # two-valued, cmov
    bit_stream.write(bzhi(st, nb), nb)               # low nb bits of st
    n  = st >> nb                                    # lands in [f, 2f)
    return (1<<table_log) | slot_of[n + e.bias]

encode_flush(st):
    bit_stream.write(bzhi(st, table_log), table_log) # high bit is implied
```

`slot_of[n + e.bias]` is FSE's `stateTable[(x>>nb) + deltaFindState]`. The
spread permutation *is* the encoder table, nothing more.

### Decode

```
decode_init():  st = (1<<table_log) | bit_stream.read(table_log)

decode(st):
    e  = dec_tbl[bzhi(st, table_log)]
    st = e.state | bit_stream.read(e.nbits)
    return (st, e.sym)
```

`|` rather than `+` is safe: `e.state = n << nbits` has its low `nbits` zero
by construction (checked on every entry). It drops a carry chain from the
critical path. FSE writes `+` and does not need to.

`{sym, nbits, state}` packs into 32 bits — one aligned dword load per decode,
the same shape as lzfse's `{int8 k; uint8 symbol; int16 delta}`.

---

## 4. What lzfse does

lzfse computes **no spread**. `fse_init_decoder_table` walks `t++` straight
through, one contiguous block per symbol in symbol-index order; the comment
says so outright (*"Initialize all states S reached by this symbol: OFFSET <=
S < OFFSET + F"*). The only `step` in `lzfse_fse.c` is `highprec_step` in
`fse_normalize_freq`, which scales counts to sum to `nstates` — frequency
normalization, not slot assignment.

It still gets two `nbits` values per symbol, from a **range split** rather
than a spread:

```c
k  = clz(f) - clz(nstates);       // N <= (f<<k) < 2N
j0 = ((2*nstates) >> k) - f;
j <  j0:  nbits = k,    delta = ((f+j) << k) - N
j >= j0:  nbits = k-1,  delta = (j - j0) << (k-1)
```

`j0` is exactly the point that makes coverage come out right:
`j0·2^k + (f−j0)·2^(k−1) = 2^(k−1)·(j0+f) = 2^(k−1)·(2N>>k) = N`. Verified
valid by the tiling test for every frequency set tried.

Having no spread is precisely **why** its encoder can be affine:

```c
int hi = s >= s0;  nbits = hi ? k : k-1;  delta = hi ? delta0 : delta1;
*pstate = delta + (s >> nbits);        // no state table at all
```

Four `int16` per symbol instead of a T-entry permutation. With contiguous
ranges the slot lookup collapses to arithmetic; with a scattered spread it is
a genuine permutation and the table is unavoidable.

**You cannot have both.** That is the design axis:

| | table | code length |
|---|---|---|
| lzfse (contiguous) | 4 int16 per symbol | +1.4–2.4% skewed |
| FSE / zstd (spread) | T-entry permutation | ~0% |

---

## 5. Post-mortem: a broken first draft

Worth recording, because the failure is quiet and the expression looks right.

```
st    = (1<<table_log) + x            # x = GLOBAL slot index
nbits = clz(f) - clz(st)
state = (st - f) << nbits
```

Two bugs, one root cause.

**`nbits` came out constant per symbol.** `st = T + x` with `x < T` always has
its top bit at `table_log`, so `clz(st)` is a constant and
`nbits = table_log - highbit(f)` depends only on the symbol. One bit count per
symbol is a Huffman code. The symptom: per-symbol coverage becomes
`f << (table_log - highbit(f))`, which equals `T` only when `f` is an exact
power of two — so `freqs=[2048,2048]` was the one case that partly passed and
everything else showed gaps or overlaps.

**`state` was built from the global slot index**, putting states at
`[24000, 28384)` and `[512000, 518144)` instead of `[4096, 8192)`.

Both come from driving the shift with the global slot instead of the
per-symbol counter `n = f .. 2f-1`. The fix is to reassign the `clz`
arguments:

```
nbits = clz(f) - clz(st)              # wrong: clz(st) is constant
nbits = clz(n) - clz(1<<table_log)    # right
```

### The clz identity

`clz(1<<table_log)` is `W - 1 - table_log` for a W-bit `clz` (19 for
`table_log=12` on 32-bit, not 20). But the width cancels out of the whole
expression, so do not write it:

```
clz(n) - clz(1<<table_log) = (W-1 - highbit(n)) - (W-1 - table_log)
                           = table_log - highbit(n)
```

```c
nbits = table_log - highbit(n);              /* highbit(n) = 31 - __builtin_clz(n) */
nbits = __builtin_clz(n) + table_log - 31;   /* same, one clz */
```

Verified for every `table_log` 1..20 and every `n` in range, on both 32- and
64-bit `clz`. The direct form keeps the word width out of the expression, so
nothing silently breaks if `n` gets promoted to 64-bit.

---

## 6. Gotchas

- **ANS is LIFO.** Encode the message backwards so the decoder emits it
  forwards. The bit stream is a stack — the usual arrangement is the encoder
  writing into a buffer that grows downward while the decoder reads upward.
  The flushed state is written last and read first.
- **The decoder needs the symbol count**, or a terminator. Nothing in the
  stream says when to stop.
- **`sum(freqs)` must equal `1<<table_log` exactly.**
- **Symbols with `f == 0` get no slots and can never be encoded.** Skip them
  everywhere: `highbit(0)` is undefined, `__builtin_clz(0)` is UB, and `BSR`
  leaves its destination unmodified.
- **Table log is a pure fidelity knob.** Excess halves per bit and is
  independent of everything else: −0.295% / −0.158% / −0.079% / −0.041% /
  −0.020% at `table_log` 10 / 11 / 12 / 13 / 14.
