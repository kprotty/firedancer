# Packing Solana snapshots

Notes towards a snapshot specific account compressor for Firedancer: one that
beats `zstd -1` (`FD_BACKUP_ZSTD_LEVEL` in `src/discof/backup/fd_backup.h`,
what Firedancer actually uses for snapshots) on **both** ratio and throughput.

This is the combined version of `FINDINGS.md`, `IDEAS.md` and `PLAN.md` in
`sol_snap_pack/`, as of 2026-09-18.  Part I is measured, Part II is design,
Part III is what to do next, and the appendix is how to reproduce any of it.

---

## The short version

The snapshot at slot 447,858,864 is 112.6 GiB of zstd over 475.2 GiB of tar
holding **1,178,617,724 accounts**.  Four measurements decide everything:

1. **A quarter of any optimal archive is pubkeys.**  The sorted-set floor for
   1.178e9 random 256 bit keys is 28.41 B/account = **31.2 GiB**, and nothing
   compresses them.  That is the hard bound on how good this can ever get.
2. **The rest of the header is already nearly free** — owner, data_len,
   lamports, rent_epoch and executable together carry 5.6 bits/account under
   the obvious conditional models, against the ~2.5 GiB `zstd -1` spends.
   Worth ~2 GiB, i.e. do it because it is easy.
3. **72.2% of account data bytes are zero.**  One dead program (Serum DEX v3)
   is 88.6 GiB of 99.6%-zero data on its own.
4. **20.41% of account data is copies of pubkeys the snapshot already
   carries** — 2,204,113,696 references, 203M distinct keys, 14.96 bits of
   entropy each.  **65.7 GiB of raw bytes worth 3.8 GiB**, and the dictionary
   is free because it is the snapshot's own pubkey array.

Point 4 is the thesis.  An LZ77 window cannot hold a 203 million key working
set, so zstd pays 17-31 bytes for references that carry 15-21 bits: on SPL
token accounts (52.3% of all accounts) it spends 22.9 B on each mint field and
31.3 B on each wallet field.  A codec whose native vocabulary is
`{zero run, key reference, literal}` addresses 92% of the data directly, with
table lookups instead of match search — which is why it should be **smaller and
faster at the same time**.

Budget: **112.6 GiB today → ~69 GiB projected (~1.6x)**, of which 31.2 GiB is
the irreducible pubkey floor.  The only soft number left in that projection is
the 86.1 GiB residual (§I.5), and pricing it is the next job (Part III).

---

## Part I — What the mainnet snapshot contains

Everything in Part I is measured, not estimated, unless it says "estimated".
Source: `snapshot-447858864-AjtJj28AjLeUffcyydNZZfXHJer8EtDVH22Rt53K6Dtq.tar.zst`
(the symlink `../snapshot.tar.zst`).  Raw reports are in `out/`:
`full_stats_pass2.json` (full pass, cross-checked against an independent second
pass in `full_stats3.json` — they agree exactly), `report_full.md` (rendered),
`corpstat_rand.txt`, `refscan_rand.txt`, `bench_full.txt`,
`bench_full_sorted.txt`.

### I.1 Shape of the archive

| metric | value |
|---|---|
| archive (zstd) | 120,906,157,860 B = **112.6 GiB** |
| uncompressed tar | 510,213,364,736 B = **475.2 GiB** |
| zstd ratio as shipped | **4.22x** |
| stored accounts | **1,178,617,724** |
| distinct account pubkeys | 1,178,617,687 fingerprints — i.e. **no duplicate accounts**, one entry per account |
| appendvec files | 426,972, slots 325,728,012 .. 447,858,864 |
| account headers | 149.3 GiB (136 B/acct, 31.4% of the raw bytes) |
| account data | 321.9 GiB (293 B/acct) |
| 8 byte alignment padding | 3.51 GiB |
| manifest + status cache | 0.20 GiB |
| distinct owners | 47,834 |
| distinct data_len values | 37,636 |
| distinct (owner, data_len) groups | 176,658 |

**Free bytes in the header.** `write_version` and the 32 byte account hash are
**zero for all 1.178 billion accounts** (the latter because of accounts-lt-hash).
That is 40 B/account = **43.9 GiB of guaranteed zeros**, plus 3.5 GiB of
padding, in a 475 GiB file.

### I.2 Per field entropy of the header

Measured on `rand.corpus` (the complete 1/128 pubkey-hash sample, 9,204,364
accounts), `corpstat`:

```
H(owner)                  3.02 bits      47,834 distinct, top owner = 58.4% of accounts
H(data_len | owner)       0.89 bits      len 165 alone is 52.3% of all accounts
H(lamports | owner,len)   1.66 bits      83.0% of accounts hold exactly the rent exempt minimum
H(rent_epoch| owner,len)  0.02 bits      99.6% u64::MAX, 0.4% zero, nothing else
H(executable| owner,len)  0.00 bits      71,476 executable accounts in 1.18 billion
pubkey                  234.3  bits      sorted set bound, irreducible
```

(The conditional entropies are "free model" numbers — the per group tables are
not charged for.  With 176,658 groups and 1.18e9 accounts the model is tiny
relative to the data, but the oracle in Part III should charge for it
honestly.)

So the whole 136 byte header is worth **~30.0 bytes/account, and 29.3 of those
are the pubkey**.  At 1.178e9 accounts the pubkey floor for the snapshot is
`n * (256 - log2 n + 1.44) / 8` = **31.2 GiB**, i.e. more than a quarter of the
entire current archive is pubkeys and is not compressible by anything.

Cross-checked against the producer side: Firedancer's own snapshot writer
(`fd_snapzp_tile.c:536`, `fd_backup_cache.c:299`) `memset`s the whole 136 byte
header and then fills only `pubkey`, `lamports`, `executable`, `data_len` and
`owner` — so it emits zero for the first 8 bytes, for the hash, *and* for
`rent_epoch`.  An Agave-produced snapshot (this one) has `rent_epoch =
u64::MAX` for 99.6% of accounts; a Firedancer-produced one has it zero
throughout.  Either way it is worth ~0 bits, but a codec must not hardcode
one producer's convention.

`lamports` deserves a special mention: the top values in the whole snapshot are
exactly the rent exempt minimums `(128 + data_len) * 3480 * 2`:

| lamports | accounts | = rent exempt min for |
|---|---|---|
| 2,039,280 | 601,105,421 | len 165 (SPL token account) |
| 2,074,080 | 80,181,805 | len 170 (token-2022) |
| 1,461,600 | 75,473,061 | len 82 (mint) |
| 1,030,080 | 23,035,093 | len 20 |
| 1,844,400 | 13,570,451 | len 137 |

### I.3 Owner distribution

| owner | accounts | share | data | zero share of its data |
|---|---|---|---|---|
| `TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA` | 688,702,235 | 58.4% | 100.08 GiB | 56.0% |
| `TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb` | 119,050,672 | 10.1% | 21.57 GiB | 57.2% |
| `11111111111111111111111111111111` | 90,891,926 | 7.7% | 0.37 GiB | 16.7% |
| `metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s` | 82,810,176 | 7.0% | 30.96 GiB | 65.8% |
| `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P` | 29,185,414 | 2.5% | 3.00 GiB | 58.1% |
| `srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX` | 4,463,154 | 0.4% | **88.62 GiB** | **99.63%** |

Owner concentration: top1 58.4%, top4 83.3%, top16 89.8%, top64 94.9%.

**The Serum outlier.** Serum DEX v3 (`srmqPvym...`) holds only 4.5M accounts but
**88.62 GiB of account data, 99.63% of which is zero** — 27.5% of all account
data in the snapshot is zero padding inside dead Serum market/queue accounts
(`data_len` up to 1,048,588).  Any codec that handles long zero runs well gets
this for free; it also means *average* statistics over "account data" are
dominated by one dead program and should always be reported with and without it.

### I.4 Account data: zeros

| metric | value |
|---|---|
| zero bytes | 232.3 GiB = **72.2% of all account data** |
| zero bytes excluding Serum | 61.7% of the remaining 233.3 GiB |
| bytes inside zero runs >= 8 | 223.0 GiB (69.3%) |
| bytes inside zero runs >= 32 | 216.8 GiB (67.4%) |
| trailing zero run | 60.8 GiB (18.9%) |
| accounts with no trailing zero at all | 151,464,348 (12.9%) |
| order-0 byte entropy of data | 3.96 bits/byte |
| order-0 entropy of the non-zero bytes | 7.93 bits/byte (37.9% of sampled bytes) |

The non-zero bytes are essentially incompressible under an order-0 byte model
(7.93 of 8 bits), which is the usual signature of *keys and hashes*: the
structure is at the 32 byte level, not the byte level.

### I.5 Account data: pubkey references — the main result

`refscan` matched every 4-byte-aligned 32 byte window of the sampled account
data against the fingerprints of **all 1.178 billion account pubkeys in the
snapshot** (complete 1/128 sample, 9,204,364 accounts, `out/refscan_rand.txt`):

```
zero windows        52.37% of data bytes
KEY references      20.61% of data bytes   (17,225,197 refs, 1.87 per account)
residual            27.02% of data bytes
distinct keys referenced   5,330,139
H(reference)               13.73 bits  = 1.72 bytes per reference
top 16 keys                40.0% of references
top 4096 keys              51.1% of references
top 1,048,576 keys         75.1% of references
```

(The 52.37% here and the 72.2% in §I.4 are not in conflict: `refscan` only counts
a *32 byte aligned window that is entirely zero* as a zero window, and its
greedy scan advances 4 bytes at a time otherwise, so short or misaligned zero
runs land in "residual".  §I.4 counts every zero byte.  The gap between the two,
~20% of the data, is zeros that a 32 byte token cannot swallow — which is why
the codec needs a zero *run* token, not only a zero *window* token.)

A second, independent measurement — counting *any* 32 byte blob that occurs
more than once inside the sample, whether or not it is an account pubkey —
covers **28.0% of data bytes** (154 million distinct blobs), leaving only
**17.4%** that is neither zeros nor a repeated 32 byte blob.  The two
measurements overlap, but together they say the same thing: **account data is
zeros and 32 byte keys, with a small remainder.**

#### The same measurement over the entire snapshot

`snapscan -R` repeated the scan over all 1,178,617,724 accounts against all
1.178 billion pubkeys (`out/full_stats_ref.json`), so nothing below is
sampled:

| metric | value |
|---|---|
| aligned all-zero 32 byte windows | 52.64% of data |
| **key references** | **2,204,113,696** = **20.41% of data = 65.7 GiB raw**, 1.87 per account |
| residual | 26.74% of data = 86.1 GiB |
| distinct keys referenced | **203,132,064** (17% of all accounts are pointed at by some other account) |
| H(reference) | **14.96 bits = 1.87 bytes** |
| those references, entropy coded | **3.8 GiB** |
| those references, as a flat 31 bit index into the pubkey array | 8.0 GiB |

**65.7 GiB of the snapshot is copies of pubkeys it already contains, and they
are worth 3.8 GiB.**  The 1/128 sample predicted 13.73 bits per reference and
the true figure is 14.96 — the sample was optimistic by 9%, as expected, and
the conclusion is unchanged.

The 203 million distinct referenced keys also settle the dictionary question:
transmitting a dictionary of them would cost 6.5 GB, so the dictionary *must*
be the snapshot's own pubkey array (free), with the frequency skew exploited by
ANS coding the rank rather than by shrinking the dictionary.

**One fifth of all account data is a copy of a pubkey that the snapshot is
already carrying somewhere else.**  The dictionary is therefore free: the
decoder can address any of them by index into the (sorted) pubkey array it has
already decoded, so a reference costs between ~1.7 bytes (skewed, ANS coded
rank) and ~3.8 bytes (a raw 30 bit index), against 32 bytes raw.

What zstd does with those bytes today, measured on the biggest group in the
snapshot — the 165 byte SPL token accounts (52.3% of all accounts):

| layout | zstd -1 | zstd -9 | order-0 entropy of the field |
|---|---|---|---|
| records as stored (165 B each) | **59.5 B/rec** | 54.2 B/rec | — |
| transposed to columns | 69.6 B/rec | 68.8 B/rec | — |
| just the mint field (offset 0, 32 B) | 22.9 B/rec | 16.9 B/rec | **1.92 B/rec** (15.38 bits, 1,451,754 distinct) |
| just the wallet field (offset 32, 32 B) | 31.3 B/rec | 29.7 B/rec | **2.62 B/rec** (20.98 bits, 3,574,366 distinct) |

Those last two rows are the whole argument in one place.  Over 4,796,143
sampled token accounts, the mint field carries 15.4 bits of information and
zstd spends 183 bits on it; the wallet field carries 21.0 bits and zstd spends
250.  The field entropies are "free dictionary" numbers, and the dictionary
*is* free here: mints and wallets are themselves accounts in the same snapshot.
Even pricing a reference at a raw index into the 1.18 billion entry pubkey
array (30.1 bits = 3.8 B) rather than at its entropy, the two fields cost
~7.6 B against zstd's 54.2, on 52.3% of all accounts in the snapshot.

The wallet number is the one to re-measure without sampling: in a 1/128 sample
3.57M of the 4.80M wallet references are unique, which inflates that field's
entropy; in the full snapshot each wallet's token accounts are all present.

Two things fall out of that table:

1. zstd spends **17-23 bytes on each mint reference and ~30 bytes on each
   wallet reference**, because an LZ77 window cannot hold the working set.  A
   dictionary reference costs ~2-4 bytes.  This is the single biggest
   structural win available.
2. **Naive transposition makes things worse**, not better (69.6 vs 59.5 B/rec):
   splitting a 32 byte key across 32 column streams destroys the only match LZ
   could find.  Columnar layout is right for scalar fields and wrong for key
   fields — a codec has to know which is which.

For reference, the field cardinality profile of the token group (`corpstat`)
shows where the entropy sits: offsets 0..64 (mint, owner, amount) are above the
64k cardinality cap, and **every offset from 68 to 165 carries less than 0.6
bits** — the remaining 97 bytes of a token account are, collectively, worth
about 10 bits.

### I.6 Baselines to beat

`corpsplit` + `bench_baseline.sh` on the complete 1/128 sample
(`rand.corpus`, 9,204,364 accounts = exactly 1/128 of the snapshot,
3.63 GB of tight array-of-structs records, 2.67 GB of account data):

| stream | raw | zstd -1 | zstd -3 | extrapolated to the snapshot (x128) |
|---|---|---|---|---|
| data | 2,674,206,708 | 604,863,382 | 585,055,549 | **72.1 GiB** (59.4 GiB at zstd -19) |
| pubkey | 294,539,648 | 294,546,402 | 294,546,402 | **35.1 GiB** (uncompressible) |
| lamports | 29,068,023 | 10,792,207 | 9,533,683 | 1.29 GiB |
| dlen | 16,122,669 | 5,497,209 | 5,308,862 | 0.66 GiB |
| slot | 9,672,378 | 919,120 | 926,476 | 0.11 GiB |
| owner (+dict) | 9,844,468 | 4,088,866 | 4,225,984 | 0.49 GiB |
| rent | 9,204,364 | 64,030 | 64,253 | 0.01 GiB |
| exec | 1,150,546 | 2,460 | 2,520 | ~0 |
| **total, field split** | 3,043,808,804 | **920,773,676** | 899,663,729 | **109.8 GiB** |
| same records as array-of-structs | 3,631,460,564 | 961,419,898 | 931,725,684 | 114.6 GiB |

The array-of-structs number (114.6 GiB extrapolated) reproduces the real
archive (112.6 GiB) to within 2%, which validates the sampling: **the
measurements above can be extrapolated**.

Splitting the records into per field streams is worth 4.2% on its own.
Clustering the records by `(owner, data_len, pubkey)` first (`corpsplit --sort`,
same sample) is worth another 6.8% *net*:

| stream | zstd -1, natural order | zstd -1, clustered |
|---|---|---|
| data | 604,863,382 | **532,736,486** (-11.9%) |
| lamports | 10,792,207 | 3,516,346 |
| dlen | 5,497,209 | 51,368 |
| owner | 3,759,343 | 31,008 |
| rent | 64,030 | 51,233 |
| slot | 919,120 | **26,639,326** (the penalty) |
| total | 920,773,676 | **857,898,302** |

Clustering destroys the slot stream: in appendvec order the slot is constant
for thousands of consecutive accounts, in pubkey order it is ~`log2(426,972)` =
18.7 bits of real entropy per account (~2.7 GiB over the snapshot).  It is
still a clear net win — and the slot cost is unavoidable in any format that
stores accounts in pubkey order rather than grouping them by the appendvec
they happen to live in.

#### Where the 112.6 GiB actually goes

| part | today (zstd -1) | floor / target | mechanism |
|---|---|---|---|
| account data | 72.1 GiB | ~36 GiB | zero tokens + key references + templates |
| pubkeys | 35.1 GiB | 31.2 GiB | sorted set coding (hard floor) |
| other header fields | 2.5 GiB | ~1.1 GiB | ANS with the §I.2 contexts |
| manifest + framing | ~1 GiB | ~1 GiB | untouched |
| **total** | **112.6 GiB** | **~69 GiB** | **~1.6x smaller** |

The ~36 GiB data target now rests on measured numbers for two of its three
terms: zero runs cost ~2% of their bytes (~2 GiB), the 2,204,113,696 key
references cost **3.8 GiB** entropy coded (8.0 GiB if coded as flat indices),
and the remaining term is the 86.1 GiB of residual, assumed to compress about
as well as zstd manages today (~2.9x → ~30 GiB).  **The residual is now the
only soft number in the budget** and is exactly what the Phase 1 oracle in
Part III has to pin down.

### I.7 What this implies for the design

1. **Pubkeys are 31% of an optimal archive** (31.2 GiB of ~100 GiB) and are
   untouchable except for the ~3.4 B/account that sorted-set coding recovers.
2. **The header is otherwise already almost free** and needs ~1 B/account.
   Chasing it is worth ~2 GiB; do it because it is easy, not because it matters.
3. **72% of account data is zeros** and another **20% is pubkey references**.
   A codec whose native vocabulary is `{zero run, key reference, literal}`
   addresses 92% of the data directly, and does it with table lookups instead
   of LZ match search — which is why it should be faster *and* smaller.
4. **The residual 27%** is where a conventional entropy stage still earns its
   keep; it is also where the per-`(owner,len)` templates should be aimed.
5. Never transpose blindly; distinguish key fields from scalar fields.

---

## Part II — How to compress it

The codec is fed one `(slot, pubkey, owner, lamports, rent_epoch, executable,
data[])` tuple at a time and emits a chunked stream.  It is allowed to know
what a Solana account *is*; that is the entire point.

### II.0 What the input actually looks like

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
**~30.0 bytes/account, of which 29.3 is the pubkey** (§I.2 has the per field
breakdown).  `zstd -1` on the same fields, split into per field streams, spends ~34.6
bytes/account.  So the header is *already* nearly free for zstd and the
remaining upside there is ~4.5 bytes/account (~2 GiB over the snapshot).
**The account data is where the bits are**: 72.1 of the 112.6 GiB archive.

---

### II.1 Owner: builtin / dictionary / escape

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

### II.2 rent_epoch, lamports, executable, data_len, slot

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

### II.3 The pubkey floor, and what to do about it

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
  stream going from 0.9 MB to 26.6 MB on the sample (see §I.6).
- Sorting is *our* choice: a new snapshot format need not group accounts by
  appendvec/slot the way the Agave format does.  Sorting also helps every
  other stream (see the measured `--sort` column in §I.6: owner and
  data_len streams collapse to nearly nothing once records are clustered).

### II.4 Account data: it is zeros and pubkeys

Two measurements drive the whole data design:

- **72.2% of all account data bytes are zero** (232.3 of 321.9 GiB); zero runs
  of length >= 32 alone cover 67.4% of the data.  One dead program, Serum DEX
  v3, accounts for 88.6 GiB of that on its own (99.6% of its data is zeros);
  excluding it the rest of the snapshot is still 61.7% zeros.
- **20.5% of data bytes are 32 byte windows that match a pubkey the snapshot
  already carries** — measured against the fingerprints of all 1.178 billion
  account pubkeys, 1.87 references per account.  The reference distribution is
  extremely skewed: 13.47 bits of entropy, top 16 keys are 40% of all
  references.

That is the entire thesis: **account data is mostly zeros and mostly pubkeys**,
and an LZ77 matcher with a 128 KiB window is a bad way to encode either.

**Warning, measured**: naive transposition is a trap.  Transposing the same
token accounts into 165 column streams made zstd *worse* (69.6 B/rec vs 59.5),
because splitting a 32 byte key across 32 streams destroys the only match LZ
could find.  Columnar is right for scalar fields and wrong for key fields; the
codec has to know which is which (§II.5).

Design: tokenize the data of each account into a stream of

```
ZERO32   (a 32 byte aligned run of zeros, run length ANS coded)
KEYREF   (32 bytes that hit the global key dictionary, index ANS/varint coded)
LITERAL  (n residual bytes, entropy coded, or LZ'd for opaque blobs)
```

with the token type ANS coded in the context of `(account type, offset)`.  A
KEYREF costs 1.7 bytes (ANS coded rank in the skewed distribution) to 3.8 bytes
(a raw index into the snapshot's own 1.18 billion entry pubkey array) versus 32
raw — and, unlike an LZ match, its cost does not depend on how far away the
previous occurrence was.  This is the main structural advantage over zstd.

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

### II.5 Account type templates

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
over a ~100 GiB snapshot.  A cheaper adaptive variant: **XOR each record
against the previous record of the same group** and zero-run code the result —
this captures "constant field" and "same mint as the last one" without any
preamble, at the cost of a serial dependency inside a group.

Known schemas worth hardcoding (they are protocol-stable and huge):
`SPL Token account (165)`, `SPL Mint (82)`, `Token-2022 + extensions`,
`stake account (200)`, `vote account (3762)`, `AddressLookupTable`,
`nonce account (80)`.

### II.6 BPF programs / ELF

Executable accounts and `*Loader*`-owned data accounts are ELF files: a
standard header, a small set of section names, and a lot of structure that a
generic LZ does handle well.  They are also a small share of bytes.  Plan:
route "opaque" groups to a **generic LZ backend** (zstd with a trained
dictionary, or our own LZ), rather than trying to schema-model them.  A zstd
dictionary trained on a few hundred BPF ELF headers is a 20 line experiment.

### II.7 Chunking

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

### II.8 Entropy coder

rANS with static per-chunk (or per-preamble) tables, interleaved 4-8 ways for
ILP, exactly like the existing `fd_zle` work.  Static tables allow SIMD decode;
adaptive models do not.  The ANS work in `ans_network/` and the `fd_zle*`
variants in `src/flamenco/accdb/` are directly reusable.

---

### II.9 Summary: what we expect to win, and where

| component | of the snapshot | zstd -1 | proposed | mechanism |
|---|---|---|---|---|
| header, non pubkey | 149.3 GiB raw | 2.5 GiB | ~1.1 GiB | ANS + conditional models |
| pubkey | 35.1 GiB raw | 35.1 GiB | 31.2 GiB | sorted set coding |
| data: zeros | 72.2% of data | included below | ~2 GiB | ZERO32 token |
| data: key references | 20.5% of data | ~14 B each (est) | ~2.5 B each | free dictionary |
| data: residual | 27.0% of data | included below | ~30 GiB | templates + ANS |
| **total** | **475.2 GiB raw** | **112.6 GiB** | **~69 GiB** | |

The header is nearly solved by zstd already; the **two real levers are the
global key dictionary and the per type templates**, and the third lever is
throughput, where a token/ANS design should beat LZ match finding outright.

---

### II.10 Sketch of the wire format

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

Per chunk, one buffer per stream, each entropy coded on its own:

| stream | content | expected cost |
|---|---|---|
| `owner` | ANS symbol: builtin id / dict index / escape | 0.38 B/acct |
| `len` | ANS symbol conditioned on owner: modal / table / escape varint | 0.11 B/acct |
| `lam` | ANS symbol: rent exempt min / zero / group modal / escape varint | 0.21 B/acct |
| `rent` | ANS binary conditioned on `lamports==0` | ~0 |
| `exec` | ANS binary conditioned on owner | ~0 |
| `slot` | delta vs the chunk's base slot | ~2.3 B/acct if pubkey sorted |
| `tok` | per account data: `ZERO_RUN(n) / KEYREF / LITERAL(n)`, ANS coded in the context of (group, offset) | |
| `ref` | key reference: rank in the dictionary, ANS coded | 1.7-3.8 B/ref, 1.87 refs/acct |
| `lit` | residual literal bytes | the remaining 27% of data |

Decoding a chunk is: decode the scalar streams (pure ANS, no dependencies),
then walk the token stream writing zeros, dictionary copies and literals into
the account buffer.  Both phases are table driven and branch predictable.

**Dictionary bootstrap.**  The key dictionary is the snapshot's own pubkey
array, so decode is two phase: (1) decode `pubkeys` (parallel across chunks,
independent of everything else), (2) decode the chunks, resolving `ref` against
the now complete array.  No dictionary is transmitted and no chunk depends on
another chunk's *data*.

### II.11 Why this should also be faster than zstd

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

---

## Part III — Plan

Ordered so that every phase ends in a number that either justifies or kills
the next phase.

### Status

Phase 0 is done and Phase 1 step 1 is answered, sampled *and* over the whole
snapshot.  The next thing to do is the Phase 1 entropy oracle, whose only real
job is to price the 86.1 GiB residual.

### Phase 0 — instrumentation  *(done)*

- `snapscan` — streams an uncompressed snapshot tar on stdin, parses tar +
  appendvecs, collects global statistics, and writes two sample corpora:
  - `rand.corpus`: 1/128 of accounts chosen by pubkey hash (unbiased, good for
    per-account statistics),
  - `av.corpus`: every 256th appendvec in full (preserves the on-disk ordering
    and locality, good for block-compression experiments).
- `corpstat` — structural analysis of a corpus: conditional entropies of the
  metadata fields, per `(owner,len)` field cardinality profiles, repeated 32
  byte window coverage, pubkey back-reference coverage.
- `corpsplit` + `bench_baseline.sh` — struct-of-arrays transform and the
  zstd baseline table, with and without clustering the records by
  `(owner, data_len, pubkey)`.

Results: Part I.

### Phase 1 — decide the model before writing a codec

Everything here is measurement, no wire format.

1. **Is the pubkey set its own dictionary?**  *Answered, and the gate is
   cleared*: matching the sampled account data against the fingerprints of all
   1.178 billion account pubkeys gives **20.47% of data bytes covered, 1.87
   references per account, reference entropy 13.47 bits**
   (`out/refscan_rand2g.txt`).  The free-dictionary design is the centrepiece.
   The unsampled pass (`snapscan -R`, `out/full_stats_ref.json`) confirms it
   over all 1.178 billion accounts: **2,204,113,696 references, 20.41% of the
   data (65.7 GiB raw), 203,132,064 distinct keys, 14.96 bits per reference =
   3.8 GiB entropy coded**.  It also answers the dictionary question: 203M
   distinct referenced keys is far too many to transmit, so the dictionary is
   the snapshot's own pubkey array and the skew is exploited by ANS coding the
   rank.
2. **Entropy oracle.**  A tool that prices a corpus under successively richer
   models without implementing a codec.  Each model reports bits/account and
   total bytes, against `zstd -1/-3/-19` on identical input:
   - `M0` zstd on the raw records (baseline),
   - `M1` field split, ANS metadata with the conditional contexts from §I.2,
     data still zstd,
   - `M2` `M1` + data tokenized into `ZERO32 / KEYREF / LITERAL`, literals
     zstd'd, KEYREF into the free dictionary,
   - `M3` `M2` + per `(owner,data_len)` templates (constant offsets, typed
     fields), residual literals only,
   - `M4` `M3` + sorted account order + XOR against previous record of the
     same group.

   Reference points the oracle has to beat, on the 1/128 sample: account data
   604,863,382 B at `zstd -1` and 532,736,486 B at `zstd -1` after clustering
   (`out/bench_full*.txt`).
   *Gate*: `M2` must beat `zstd -19` on the data stream, and `M3` must beat it
   by a wide margin, or the schema modelling is not worth the complexity.
3. **Dictionary sizing sweep.**  Referenced-key frequency is very skewed
   (`corpstat` prints top-N coverage).  Sweep dictionary size (2^12 … 2^24)
   and code the index with an ANS-coded rank, measuring bytes/reference.
4. **Where does the residual live?**  The 27.0% of data that is neither zeros
   nor a key reference is the least understood term in the budget and the one
   the ~69 GiB projection is most sensitive to.  Break it down by owner group,
   check the BPF/ELF share (71,476 executable accounts), and decide whether
   the LZ fallback needs a trained dictionary (`zstd --train` on ELF headers
   is the quick test).
5. **Sanity check the extrapolation.**  The 1/128 sample reproduces the real
   archive size to within 2% (114.6 vs 112.6 GiB), so sample-level results can
   be trusted; keep re-checking this as models get richer, since a model that
   overfits the sample will break the identity.

### Phase 2 — codec prototype

Only if Phase 1 clears its gates.

- Wire format sketch (see §II.10): preamble (builtin table, owner
  dictionary, templates) + chunks (target 1-4 MiB input each) + footer index.
- Streams per chunk, each entropy coded on its own: `owner`, `len`, `lamports`,
  `rent`, `exec`, `token`, `keyref`, `literal`, plus the `pubkey` stream in
  sorted, front-coded form.
- rANS with static per-chunk tables, interleaved 4-8 ways, reusing the ANS and
  `fd_zle` work already in this tree (`src/flamenco/accdb/fd_zle*.c`,
  `ans_network/`).
- API shape the user asked for: iterative feed
  `pack_acct(enc, slot, pubkey, owner, lamports, rent_epoch, exec, data, len)`,
  chunk flush on size threshold.
- Deliverables: roundtrip test over the full corpus, a differential fuzzer
  against a reference implementation (the `fuzz_zle*_diff.c` pattern in
  `src/flamenco/accdb/` is the template), and `bench_pack` reporting
  MB/s encode/decode and ratio versus zstd at matched throughput.
- *Gate*: ≥20% smaller than `zstd -1` at ≥1.5 GB/s single core encode, and
  decode no slower than zstd's ~1.5 GB/s.

### Phase 3 — make it real

- Parallel decode across chunks (this is why chunks are independent); measure
  aggregate throughput on this 128 core box.
- Snapshot creation is a *stream* out of the accounts DB, but templates and
  dictionaries want a stats pass.  Options, in order of preference:
  1. free dictionary = the snapshot's own pubkey array, decoded in a first
     parallel pass over the metadata streams — no preamble, no second encode
     pass;
  2. templates learned from the *previous* snapshot (they change slowly) and
     shipped in the preamble, with a per-group escape when a template misses;
  3. genuine two-pass encode (acceptable for a background snapshot writer,
     not for a hot path).
- Integration points: `fd_snapmk_tile.c` / `fd_snapzp_tile.c` (creation),
  `fd_ssparse.c` / restore tiles (load).  A new format is Firedancer↔Firedancer
  only, so it needs a capability negotiation and a fallback to the Agave
  tar+zstd format.  Keep the Agave format as the interop path and make the new
  one the fast path between Firedancer peers.
- Open question to settle early: the snapshot manifest (~190 MiB raw here) is
  untouched by this work; decide whether it is also worth re-coding or just
  passed through zstd.

### Non goals / risks

- **Do not** break the Agave wire format for interop; this is an additional
  format, not a replacement.
- The account hash field is zero in current snapshots (accounts-lt-hash), so
  nothing needs to be spent on it — but verify that holds for the snapshot
  versions we care about before relying on it.
- Schema modelling tracks on-chain programs that can change at any time; every
  template must have a cheap escape and the decoder must never *require* a
  template to be correct, only to be a good prediction.
- Ratio numbers from `--sort` experiments assume we are free to reorder
  accounts.  Confirm that against whatever consumes the snapshot (the loader
  wants sorted order anyway; the Agave format does not provide it).

---

## Appendix — tools and reproduction

Everything above comes out of five standalone C tools in `sol_snap_pack/src/`
(no Firedancer dependencies, `make` builds them with clang-19):

| tool | what it does |
|---|---|
| `snapscan` | streams an uncompressed snapshot tar on stdin, parses tar + appendvecs, prints a JSON statistics report; can also write sample corpora, dump every pubkey, and run the reference scan |
| `corpstat` | conditional entropies of the metadata fields, per `(owner,len)` field cardinality profiles, repeated 32 byte window coverage |
| `corpsplit` | struct-of-arrays transform into per field streams, optionally clustering records by `(owner, data_len, pubkey)` first |
| `refscan` | matches account data against the snapshot's own pubkey set; reference coverage and the entropy of the reference distribution |
| `corpgroup` | extracts one `(owner,len)` group in raw / transposed / per-key-field layouts |
| `bench_baseline.sh`, `report.py` | zstd baseline table; JSON report renderer |

```sh
SNAP=/home/ubuntu/snapshot/snapshot-447858864-*.tar.zst    # 113 GiB, do not delete

# full statistics pass + corpora (~20 min at ~460 MB/s)
zstd -dc $SNAP | ./snapscan -A out/av.corpus -a 256 -s out/rand.corpus -r 7 \
                            -m out/manifest.bin > out/full_stats.json
python3 src/report.py out/full_stats.json > out/report_full.md

# every account pubkey (37.7 GB), then the reference scans
zstd -dc $SNAP | ./snapscan -k out/pubkeys.bin > /dev/null
zstd -dc $SNAP | ./snapscan -R out/pubkeys.bin > out/full_stats_ref.json   # ~1.5 h
./refscan out/pubkeys.bin out/rand.corpus 4 > out/refscan_rand.txt         # sampled

# structure and baselines
./corpstat out/rand.corpus 16 > out/corpstat_rand.txt
./corpsplit out/rand.corpus out/split        && src/bench_baseline.sh out/split
./corpsplit out/rand.corpus out/split_sorted --sort && src/bench_baseline.sh out/split_sorted
./corpgroup out/rand.corpus TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA 165 out/tok165
```

Corpus format (`SNAPCORP1`): 16 byte magic, then records of

```
u64 slot; u64 lamports; u64 rent_epoch; u64 data_len;
u8  pubkey[32]; u8 owner[32]; u8 executable; u8 pad[7];
u8  data[data_len];
```

`rand.corpus` is a pubkey-hash sample (unbiased per account, 1/128 = 9,204,364
accounts); `av.corpus` takes whole appendvecs and preserves the on-disk
ordering and locality.  The sampled corpus reproduces the real archive size to
within 2%, which is what licenses extrapolating sample results to the snapshot.

`out/` is gitignored, ~55 GB, and fully reproducible from the snapshot with the
commands above.
