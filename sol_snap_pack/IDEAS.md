# sol_snap_pack — ideas for a snapshot specific account compressor

Goal: beat `zstd -1` (what Firedancer uses, `FD_BACKUP_ZSTD_LEVEL` in
`src/discof/backup/fd_backup.h`) on **both** ratio and throughput, for the
account payload of a Solana snapshot.

The codec is fed one `(slot, pubkey, owner, lamports, rent_epoch, executable,
data[])` tuple at a time and emits a chunked stream.  It is allowed to know
what a Solana account *is*; that is the entire point.

---

## 0. What the input actually looks like

An appendvec entry is a 136 byte header followed by `data_len` bytes, padded
to 8:

| off | sz | field | observed over all 1,178,617,724 accounts |
|----:|---:|-------|----------|
|   0 |  8 | write_version | **0 for every account** |
|   8 |  8 | data_len | 37,636 distinct values; len 165 alone is 52.3% |
|  16 | 32 | pubkey | uniformly random, irreducible, all distinct |
|  48 |  8 | lamports | the rent exempt minimum for `data_len` in 82.9% of accounts |
|  56 |  8 | rent_epoch | binary: 99.6% `u64::MAX`, 0.4% `0`, nothing else |
|  64 | 32 | owner | 47,834 distinct; top owner is 58.4% of all accounts |
|  96 |  1 | executable | 71,476 accounts in 1.18 billion |
| 104 | 32 | hash | **0 for every account** (accounts-lt-hash) |

So 40 of the 136 header bytes are constant zero on the wire (**43.9 GiB of
guaranteed zeros** across the snapshot) and another 7 are alignment padding.
Only 89 bytes carry anything, and 32 of those (the pubkey) are incompressible
by construction.

An order-0 model with the obvious contexts prices the whole header at
**~30.1 bytes/account, of which 29.4 is the pubkey**:

```
H(owner)                  3.02 bits
H(data_len | owner)       0.89 bits
H(lamports | owner,len)   1.65 bits
H(rent_epoch| owner,len)  0.02 bits
H(executable| owner,len)  0.00 bits
pubkey                  235.17 bits   (sorted set bound, 256 - log2 n + 1.44)
```

`zstd -1` on the same fields, split into per field streams, spends ~34.6
bytes/account.  So the header is *already* nearly free for zstd and the
remaining upside there is ~4.5 bytes/account (~2 GiB over the snapshot).
**The account data is where the bits are**: 72.1 of the 112.6 GiB archive.

---

## 1. Owner: builtin / dictionary / escape

Owners are a tiny, extremely skewed alphabet — 47,834 distinct, and the top
four (`Tokenkeg`, `Tokenz`, system, `metaqbxx`) cover 83.3% of all accounts.
Plan:

- A **static builtin table** baked into the codec (system, both token programs,
  stake, vote, BPF loaders, ATA, compute budget, address lookup table,
  sysvars, ...).  Known at compile time, costs nothing on the wire.
- A **dynamic dictionary** of owners discovered while feeding the compressor,
  emitted in the preamble (or as an escape-then-define stream, so that the
  encoder stays single pass).
- Encode the owner as an **ANS symbol** over `{builtin_id, dict_idx, escape}`.
  With H(owner) = 3.02 bits this is ~0.4 bytes/account.

The same dictionary machinery serves rule 3 below (pubkeys inside data), so it
should be built once and shared.

*Status: unmeasured as a scheme.*  H(owner) = 3.02 bits is measured and the
owner stream is 0.49 GiB at `zstd -1` today (`FINDINGS.md` §6), so the whole
section is worth at most ~0.4 GiB and nothing here has been priced separately.

## 2. rent_epoch, lamports, executable, data_len, slot

- `rent_epoch` is effectively a **single bit** today (`u64::MAX` for a live
  account, `0` for a zero-lamport tombstone), plus a rare escape for the
  handful of pre-`rent_epoch_freeze` accounts.  Code it as a binary ANS symbol
  contexted on `lamports == 0`.
- `lamports`: **82.9% of accounts hold exactly the rent exempt minimum**,
  `(128 + data_len) * 3480 * 2` (601 million accounts hold 2,039,280 lamports,
  the minimum for a 165 byte token account).  Code `lamports` as a symbol:
  `{rent_exempt_min, zero, group_modal_value, escape → varint}`.  Measured
  distinct-value counts show the top values in the whole snapshot are exactly
  the rent exempt minimums for the top data lengths.
- `executable` is implied by the owner (a loader) plus the data being an ELF;
  1 bit, in practice ~0 after ANS.
- `data_len` conditioned on the owner is ~1 bit (`Tokenkeg → 165` with
  p≈0.99).  Code as `{group modal len, small table, escape → varint}`.
- `slot` only changes between appendvecs, not between accounts.  It is a
  per-chunk field, not a per-account field: emit the slot once per run of
  accounts that share it (delta coded, zig zag).

Everything above is a couple of ANS streams and costs well under a byte per
account in total.

*Status: confirmed in aggregate* -- the five fields together price at 5.59 bits
= 0.70 B/account (`FINDINGS.md` §2), which is the number this section predicts;
the individual symbol alphabets above have not been tested one by one.  One
correction: `slot` is **not** cheap if accounts are stored in pubkey order,
where it is 18.7 bits/account (~2.6 GiB); it is only free while runs of
accounts share it (§3).

## 3. The pubkey floor, and what to do about it

A snapshot with `n` accounts must spend at least `log2(C(2^256, n)) ≈
n * (256 - log2 n + 1.44)` bits on the *set* of pubkeys.  At
n = 1,178,617,724 that is 28.41 bytes per account, i.e. **31.2 GiB of this
snapshot is pubkeys and nothing can be done about it** — over a quarter of the
112.6 GiB archive — except:

- Emit accounts **sorted by pubkey** and front code / delta code the sorted
  list.  This realises the `log2 n - 1.44` ≈ 3.6 bytes/account saving (~3.9 GiB
  on this snapshot) and costs nothing at decode time.  It also makes the
  snapshot directly usable as a sorted index.  Measured: clustering the records
  before splitting is a 6.8% net win overall, *including* the cost of the slot
  stream going from 0.9 MB to 26.6 MB on the sample (see `FINDINGS.md` §6).
- Sorting is *our* choice: a new snapshot format need not group accounts by
  appendvec/slot the way the Agave format does.  Sorting also helps every
  other stream (see the measured `--sort` column in `FINDINGS.md`: owner and
  data_len streams collapse to nearly nothing once records are clustered).

**Revised, measured** (`FINDINGS.md` §7): sorting is worth only **3.7% on the
data stream** (M4b vs M3), not the large win this section assumed, because the
free dictionary already prices a repeated key at almost nothing — so REPEAT
tokens mostly relabel references that were already cheap.  The case for sorting
now rests on three things that have nothing to do with the data stream: the
~3.9 GiB sorted-set saving above, the `(owner,len)` clustering that collapses
the metadata streams, and seekability.  That matters for the design, because a
KEYREF resolves an *index* into the pubkey array and never needs it sorted,
only agreed: an encoder that sorts within a 1-4 MiB chunk instead of globally
keeps nearly all of the win at bounded memory (front coding then saves
`log2(k)` rather than `log2(n)` bits per key, ~2 GiB instead of ~3.9 GiB).

## 4. Account data: it is zeros and pubkeys

Two measurements drive the whole data design:

- **72.2% of all account data bytes are zero** (232.3 of 321.9 GiB); zero runs
  of length >= 32 alone cover 67.4% of the data.  One dead program, Serum DEX
  v3, accounts for 88.6 GiB of that on its own (99.6% of its data is zeros);
  excluding it the rest of the snapshot is still 61.7% zeros.
- **20.5% of data bytes are 32 byte windows that match a pubkey the snapshot
  already carries** — measured against the fingerprints of all 1.178 billion
  account pubkeys, 1.87 references per account.  The reference distribution is
  extremely skewed: 13.47 bits of entropy on the sample, **14.96 bits measured
  over all 2,204,113,696 references in the snapshot**, top 16 keys are 40% of
  all references.

That is the entire thesis: **account data is mostly zeros and mostly pubkeys**,
and an LZ77 matcher with a 128 KiB window is a bad way to encode either.

**Warning, measured**: naive transposition is a trap.  Transposing the same
token accounts into 165 column streams made zstd *worse* (69.6 B/rec vs 59.5),
because splitting a 32 byte key across 32 streams destroys the only match LZ
could find.  Columnar is right for scalar fields and wrong for key fields; the
codec has to know which is which (§5).

Design: tokenize the data of each account into a stream of

```
ZERO_RUN (a run of zeros, run length ANS coded -- runs, not only aligned
          32 byte windows: ~20% of the data is zeros that a 32 byte token
          cannot swallow, `FINDINGS.md` §5)
KEYREF   (32 bytes that hit the global key dictionary, index ANS/varint coded)
ESCAPE   (32 bytes that look like a key but are not in the dictionary: emit
          them once, then reference them by rank in a second dictionary built
          from what has already been escaped -- see below)
LITERAL  (n residual bytes, entropy coded, or LZ'd for opaque blobs)
```

with the token type ANS coded in the context of `(account type, offset)`.  A
KEYREF costs **1.87 bytes** — measured unsampled over all 2,204,113,696 of them
(14.96 bits, `FINDINGS.md` §5) — versus 32 raw, and unlike an LZ match its cost
does not depend on how far away the previous occurrence was.  This is the main
structural advantage over zstd.

**ESCAPE is not an afterthought, it is the second largest term in the codec.**
Measured over all 719,051,458 token accounts (`FINDINGS.md` §10): the mint
field is a dictionary hit 99.96% of the time, but the *owner* field — the
wallet — only 62.36%, because a wallet that holds no SOL has no account of its
own.  Those 270,656,592 escapes are 32 raw bytes each (8.07 GiB) under the
vocabulary as first written, and they are only **127,933,600 distinct keys**.
Escaping each one once and referencing it by rank afterwards costs 4.17 GiB
instead — **3.90 GiB saved**, and the second dictionary is free to a streaming
encoder because it is built from bytes already emitted.  What is left is a
floor: 3.81 GiB of wallet identity, 86% of it a wallet that owns exactly one
token account and appears nowhere else in the snapshot.

How much that is worth, measured on the biggest group in the snapshot — the
165 byte SPL token accounts, 52.3% of all accounts:

| what | zstd -1 | zstd -9 | entropy of the field |
|---|---|---|---|
| whole records, as stored | 59.5 B/rec | 54.2 B/rec | — |
| the mint field alone (offset 0) | 22.9 B/rec | 16.9 B/rec | **1.92 B/rec** |
| the wallet field alone (offset 32) | 31.3 B/rec | 29.7 B/rec | **2.62 B/rec** |

zstd spends 17-23 bytes on each *mint* and ~30 bytes on each *wallet* because
the working set does not fit an LZ window, while the fields themselves carry
15.4 and 21.0 bits.  Those two fields are 64 of the 165 bytes, on 52.3% of all
accounts in the snapshot, and they are the whole ballgame.

**Corrected, unsampled** (`FINDINGS.md` §10).  The two field entropies in that
table are free-dictionary numbers and they assume the key *is* in the
dictionary.  For the mint that holds: it is a hit 99.96% of the time, so it
really does cost ~1.87 B against zstd's 17-23.  For the wallet it does not: it
is only a hit 62.36% of the time, and priced end to end -- reference when
resident, escape-then-rank when not -- the field costs **~7.4 B/account**, not
2.62.  So the wallet is still a 4x win over zstd rather than an 11x one, and
the mint is the bigger prize of the two.

## 5. Account type templates

Group accounts by `(owner, data_len)`.  The top ~200 groups cover the
overwhelming majority of accounts (SPL token 165, token-2022 170/182/179, the
AMM/CLMM pool layouts, stake 200, vote 3731, ...).  For each group the encoder
can learn a **template** and put it in the preamble:

- per offset: *always this constant byte* → costs nothing,
- per 32 byte aligned field: *this is a pubkey field* → KEYREF stream,
- per 8 byte field: *this is a u64 counter* → varint / delta stream,
- everything else → residual.

Templates can be learned by the snapshot creator in one stats pass (it has all
the accounts anyway) and shipped in a preamble of at most a few MB, amortised
over a ~100 GiB snapshot.  Measured: the top **1000** groups (not ~200) cover
90.7% of accounts, and the learned template alone gets the data stream to
0.451x `zstd -19` (`FINDINGS.md` §7).

**The adaptive XOR variant is dead.**  Measured two ways (`FINDINGS.md` §7):
XORing the whole record against the previous record of its group and zero-run
coding the result is **worse than plain `zstd -19` (1.245x)**, because it
disables the dictionary for the whole record and turns a pubkey field that
legitimately differs into high-entropy noise.  Done properly — keep the
dictionary, add a REPEAT token for a window equal to the previous record's —
it works but wins only 3.7%.  XOR delta coding the u64 template fields is a
clear loss as well (58.3 vs 31.2 MiB; delta beats raw on 3,640 of 23,844
slots).  Sorting by pubkey leaves adjacent records uncorrelated in those
fields, so XOR just doubles the alphabet.

**Hand written layouts beat learned templates, and this is now the design**
(`FINDINGS.md` §9).  11 layouts — `SPL Token account (165)`, `SPL Mint (82)`,
`SPL Multisig (355)`, `Token-2022` in both variants plus its TLV extensions,
`stake (200)`, `vote (3762)`, `AddressLookupTable`, `nonce (80)`, Metaplex
metadata — cover 74.2% of accounts and 48.9% of account data and beat the
learned template by **8.0% on the token groups and 9.9% over the whole data
stream**.  Three things the measurement settled:

- The statistical template is **blind by construction** to a field that is not
  8 byte aligned or whose dictionary hit rate dips below the threshold, and
  those are exactly the fields that pay: offsets 4 and 50 (mint and freeze
  authority), 76 and 133 (delegate, close authority), 12/44/92/124 (stake),
  1 and 33 (Metaplex).  Declaring the mint authority at offset 4 is most of why
  SPL mints go to 0.481x.
- **Dispatch order is declared layout, then learned template, then generic
  tokenizer.**  A layout must *compose* with the template, not replace it —
  handing every undeclared byte to the generic walker throws away the
  template's constant folding (Metaplex metadata has up to 2,840 constant
  bytes per record) and cost 1.2% of the whole data stream.
- A `COption::None` pubkey is **one decision symbol**, not a reference to the
  all-zero key.  36-99% of the optional pubkey fields are None, and the all-zero
  pubkey *is* an account, so a naive tokenizer codes them as real references.

One caveat that applies to every learned model here: an order-0 alphabet
measured on a 1/128 sample cannot cost more than `log2(records in the group)`
bits, so a field that is unique per record looks cheap however random it is —
13,614 of the 23,844 learned u64 slots are in that position, and a durable
nonce's blockhash gets charged *zero* bytes.  A declared layout has no such
exposure, which is a second reason to prefer one.

## 6. BPF programs / ELF

Executable accounts and `*Loader*`-owned data accounts are ELF files: a
standard header, a small set of section names, and a lot of structure that a
generic LZ does handle well.  They are also a small share of bytes.  Plan:
route "opaque" groups to a **generic LZ backend** (zstd with a trained
dictionary, or our own LZ), rather than trying to schema-model them.

**Measured, and this section is now the *whole* story of the residual**
(`FINDINGS.md` §11).  Once the layouts are in place, the loaders produce
**54.5% of the fallback's literal bytes** — 732 loader-owned accounts in the
1/128 sample, 93,696 in the snapshot, 83.1 MB of sampled data.  Given its own
stream, that ELF compresses **9.6x** (63,958,510 B to 6,637,305 at `zstd -19`)
against 2.4x for the rest of the fallback, so it is **0.79 GiB of the archive**.
Two consequences: give the loaders their own literal stream with its own LZ
window (splitting it out is worth 0.33 MB on the sample by itself), and
**deprioritise `zstd --train`** — 9.6x on unclustered ELF says the
intra-account redundancy is already being found and the entire prize is
0.79 GiB.  After the loaders the residual is NFT infrastructure (candy
machines, compressed-NFT merkle trees) and DEX order books; Meteora's DLMM
accounts are 29.6% key references and are the best candidate for the next
declared layout.

## 7. Chunking

Chunk = N accounts (target ~1-4 MiB of input, similar to zstd's block).  A
chunk must be independently decodable **given the preamble** (dictionaries and
templates), so that:

- decode is embarrassingly parallel across chunks (Firedancer loads snapshots
  on many cores; this matters as much as ratio),
- a chunk index in the footer gives random access to an account by pubkey
  (sorted order + chunk index = a seekable snapshot).

Per chunk, fields are stored **split by stream** (struct of arrays), each
stream entropy coded independently, which is also what makes the decoder
SIMD-friendly and branch-predictable.

*Status: untested, and one constraint added.*  Splitting by stream is measured
(4.2% for the metadata fields, `FINDINGS.md` §6, and 4.1% for the literal bytes
split per account type, §9), but chunking itself has not been measured at all.
The constraint from `PLAN.md` Phase 2: the encoder has to work without holding
the account set, so it buffers one chunk and sorts *within* it -- which is
sound because a KEYREF is an index into the pubkey array and never needs that
array sorted, only agreed.

## 8. Entropy coder

rANS with static per-chunk (or per-preamble) tables, interleaved 4-8 ways for
ILP, exactly like the existing `fd_zle` work.  Static tables allow SIMD decode;
adaptive models do not.  The ANS work in `ans_network/` and the `fd_zle*`
variants in `src/flamenco/accdb/` are directly reusable.

*Status: untested.*  Every "analytic" number in `FINDINGS.md` is an entropy, so
it assumes an entropy coder that reaches it; a real rANS with static tables
gives up a fraction of a percent to that bound, and the table cost itself has
not been charged anywhere yet.

---

## What we expect to win, and where

Originally estimated:

| component | of the snapshot | zstd -1 | proposed | mechanism |
|---|---|---|---|---|
| header, non pubkey | 149.3 GiB raw | 2.5 GiB | ~1.1 GiB | ANS + conditional models |
| pubkey | 35.1 GiB raw | 35.1 GiB | 31.2 GiB | sorted set coding |
| data: zeros | 72.2% of data | included below | ~2 GiB | ZERO32 token |
| data: key references | 20.5% of data | ~14 B each (est) | ~2.5 B each | free dictionary |
| data: residual | 27.0% of data | included below | ~30 GiB | templates + ANS |
| **total** | **475.2 GiB raw** | **112.6 GiB** | **~69 GiB** | |

Now measured rather than estimated (`FINDINGS.md` §7, §9, §10, §11), with the
data stream priced by the oracle as analytic bits plus a real `zstd -19` over
the literal streams:

| component | zstd -1 today | measured | mechanism |
|---|---|---|---|
| header, non pubkey, excl. slot | 2.4 GiB | ~0.8 GiB | ANS + conditional models (§2) |
| pubkey | 35.1 GiB | 31.2 GiB | sorted set coding, a hard floor |
| slot | 0.1 GiB | 0.1 - 2.6 GiB | free in appendvec order, 18.7 bits/account in pubkey order (§3) |
| account data | 72.1 GiB | **~19.7 GiB** | see below |
| manifest + framing | ~1 GiB | ~1 GiB | untouched |
| **total** | **112.6 GiB** | **~52.8 - 55.4 GiB** | **2.03 - 2.13x smaller** |

The slot row is the one place the ordering decision shows up in the budget, and
it is why §3's revision matters: global pubkey sorting buys 3.9 GiB on the
pubkey stream and gives back 2.5 GiB on the slot stream.

The account data row breaks down as (extrapolated x128 from the 1/128 sample,
except where it says unsampled):

| part | measured | with the escape dictionary |
|---|---:|---:|
| records a declared layout covers (74.2% of accounts, 48.9% of data) | 16.3 GiB | ~13.3 GiB |
| uncovered, excluding the loaders | 5.6 GiB | 5.6 GiB |
| loader owned ELF | 0.8 GiB | 0.8 GiB |
| **data stream** | **22.7 GiB** | **~19.7 GiB** |

and the mechanisms inside it: a key reference costs **1.87 B** (14.96 bits,
unsampled, §5); the token owner field's escapes cost **4.17 GiB** of which
3.81 GiB is an incompressible floor (unsampled, §10); a declared layout is
0.867x the learned template on the records it covers (§9); ELF compresses 9.6x
on its own stream (§11).

The header is nearly solved by zstd already; the **real levers are the global
key dictionary, the declared layouts and the escape dictionary** — in that
order of size — and the fourth lever is throughput, where a token/ANS design
should beat LZ match finding outright.  Two of the three estimates above were
pessimistic (the residual, badly so: ~30 GiB estimated against the data stream
landing at ~19.7 GiB in total) and one was optimistic: the key reference at
~2.5 B is really 1.87 B, but the *escapes* it did not anticipate cost 4.17 GiB.

---

## 9. Sketch of the wire format

```
header      magic, version, account count, chunk count, flags
preamble    builtin owner table id (compiled in, not transmitted)
            owner dictionary        (47,834 entries worst case = 1.5 MiB)
            group templates         (per (owner,len): field type map)
            ANS tables for each stream
pubkeys     the sorted pubkey array, front coded            <- also the dictionary
chunks[]    independently decodable, ~1-4 MiB of input each
footer      chunk index (first pubkey + offset per chunk), checksums
```

Revised by the measurements: the preamble carries **declared layout ids** (a
compiled-in table, not transmitted) alongside the learned templates, and the
decoder dispatches layout → template → generic tokenizer (§5).  The literal
bytes are **not one stream**: splitting them per account type before
compressing is worth 4.1% for free (homogeneous streams), and the loaders' ELF
gets its own stream with its own LZ window (§6).

Per chunk, one buffer per stream, each entropy coded on its own:

| stream | content | expected cost |
|---|---|---|
| `owner` | ANS symbol: builtin id / dict index / escape | 0.38 B/acct |
| `len` | ANS symbol conditioned on owner: modal / table / escape varint | 0.11 B/acct |
| `lam` | ANS symbol: rent exempt min / zero / group modal / escape varint | 0.21 B/acct |
| `rent` | ANS binary conditioned on `lamports==0` | ~0 |
| `exec` | ANS binary conditioned on owner | ~0 |
| `slot` | delta vs the chunk's base slot | ~2.3 B/acct if pubkey sorted |
| `tok` | per account data: `ZERO_RUN(n) / KEYREF / ESCAPE / LITERAL(n)`, ANS coded in the context of (layout field, or group+offset) | |
| `ref` | key reference: rank in the dictionary, ANS coded | **1.87 B/ref** measured, 1.87 refs/acct |
| `esc` | a 32 byte key that is not an account: emitted once, then a rank in the escape dictionary built from this stream | 32 B first time, ~2.7 B after |
| `lit[type]` | residual literal bytes, one stream per account type | |
| `lit[elf]` | loader owned accounts, LZ with its own window | 9.6x |

Decoding a chunk is: decode the scalar streams (pure ANS, no dependencies),
then walk the token stream writing zeros, dictionary copies and literals into
the account buffer.  Both phases are table driven and branch predictable.

**Dictionary bootstrap.**  The key dictionary is the snapshot's own pubkey
array, so decode is two phase: (1) decode `pubkeys` (parallel across chunks,
independent of everything else), (2) decode the chunks, resolving `ref` against
the now complete array.  No dictionary is transmitted and no chunk depends on
another chunk's *data*.

**The escape dictionary is the one exception** and it is a cheap one: it is
built from the `esc` stream itself, so it has to be decoded in order.  Keep it
per chunk (independent decode preserved, less sharing) or per `esc` stream
across the snapshot (all the sharing, one serial pass over a stream that is
~4 GiB and independent of everything else).  The measurement in §10 is the
global version; the chunk-local version has not been priced.

## 10. Why this should also be faster than zstd

*Nothing in this section has been measured.*  Every number in this document so
far is a size, and throughput is Phase 2's gate (>= 1.5 GB/s single core
encode, decode no slower than zstd's ~1.5 GB/s).  The reasoning:

- No match search.  zstd -1 spends its encode time in the hash chain / match
  finder; here the encoder does a 32 byte compare against a hash table (for
  KEYREF) and a zero test (for ZERO_RUN), both of which are single AVX-512
  instructions on the 64 byte granularity the data actually has.
- Static ANS tables allow interleaved, SIMD-friendly decode — the same
  structure as the `fd_zle` work in `src/flamenco/accdb/`.
- Streams are contiguous and homogeneous, so the decoder's inner loops are
  branch predictable in a way an LZ literal/match loop never is.
- Chunks are independent, so both directions scale linearly across cores; on
  this 128 core box a 100 GiB snapshot at even 1 GiB/s/core is under a second
  of CPU-bound work per pass, and the bottleneck moves to storage.
