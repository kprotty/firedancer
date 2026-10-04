# FINDINGS — mainnet snapshot, slot 447,858,864

Everything below is measured, not estimated, unless it says "estimated".
Source: `snapshot-447858864-AjtJj28AjLeUffcyydNZZfXHJer8EtDVH22Rt53K6Dtq.tar.zst`
(the symlink `../snapshot.tar.zst`).  Raw reports are in `out/`:
`full_stats_pass2.json` (full pass, cross-checked against an independent second
pass in `full_stats3.json` — they agree exactly), `report_full.md` (rendered),
`corpstat_rand.txt`, `refscan_rand.txt`, `bench_full.txt`,
`bench_full_sorted.txt`, `oracle_report5.txt` + `oracle5_m5report.txt` (the
M1-M5 oracle), `full_stats_tokfield.json` (the unsampled field scan).

## 1. Shape of the archive

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

## 2. Per field entropy of the header

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
relative to the data, but the oracle in `PLAN.md` should charge for it
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

## 3. Owner distribution

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

## 4. Account data: zeros

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

## 5. Account data: pubkey references — the main result

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

(The 52.37% here and the 72.2% in §4 are not in conflict: `refscan` only counts
a *32 byte aligned window that is entirely zero* as a zero window, and its
greedy scan advances 4 bytes at a time otherwise, so short or misaligned zero
runs land in "residual".  §4 counts every zero byte.  The gap between the two,
~20% of the data, is zeros that a 32 byte token cannot swallow — which is why
the codec needs a zero *run* token, not only a zero *window* token.)

A second, independent measurement — counting *any* 32 byte blob that occurs
more than once inside the sample, whether or not it is an account pubkey —
covers **28.0% of data bytes** (154 million distinct blobs), leaving only
**17.4%** that is neither zeros nor a repeated 32 byte blob.  The two
measurements overlap, but together they say the same thing: **account data is
zeros and 32 byte keys, with a small remainder.**

### The same measurement over the entire snapshot

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

## 6. Baselines to beat

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

### Where the 112.6 GiB actually goes

| part | today (zstd -1) | floor / target | mechanism |
|---|---|---|---|
| account data | 72.1 GiB | ~36 GiB | zero tokens + key references + templates |
| pubkeys | 35.1 GiB | 31.2 GiB | sorted set coding (hard floor) |
| other header fields | 2.5 GiB | ~1.1 GiB | ANS with the §2 contexts |
| manifest + framing | ~1 GiB | ~1 GiB | untouched |
| **total** | **112.6 GiB** | **~69 GiB** | **~1.6x smaller** |

The ~36 GiB data target now rests on measured numbers for two of its three
terms: zero runs cost ~2% of their bytes (~2 GiB), the 2,204,113,696 key
references cost **3.8 GiB** entropy coded (8.0 GiB if coded as flat indices),
and the remaining term is the 86.1 GiB of residual, assumed to compress about
as well as zstd manages today (~2.9x → ~30 GiB).  **The residual is now the
only soft number in the budget** and is exactly what the Phase 1 oracle in
`PLAN.md` has to pin down.

## 7. The entropy oracle: Phase 1 gate cleared

`oracle` (`src/oracle.c`) prices the corpus under M0-M4 from `PLAN.md` without
implementing a codec: it tokenizes data into `ZERO32/KEYREF/LITERAL`,
discovers per `(owner,data_len)` templates (constant bytes, pubkey-shaped 32B
fields checked against the same global dictionary as `refscan`, small-alphabet
u64 fields) for the top 1,000 groups by population (covering 90.7% of the
1/128 sample), and writes residual literal bytes to disk so their size comes
from a real `zstd -19` run rather than an assumption.

Run: `./oracle out/pubkeys.bin out/rand.corpus out/oracle 1000 16384` (84 s,
37.5 GiB peak RSS -- the 1.178B key dictionary dominates both). Reports:
`out/oracle_report.txt` (analytical costs), `out/oracle_literal_bench.txt`
(real zstd sizes of `m2_literal.bin` / `m3_literal.bin` / `m4_literal.bin`
via `bench_baseline.sh`).

Data-only cost (excludes the ~30 B/account metadata+pubkey term from §2,
which is unchanged across M1-M4), against the **zstd -19 baseline of
498,748,121 B** measured directly on the sample's `data.bin`
(`out/data_zstd19.txt`, itself the same 2,674,206,708 B the whole-sample
baseline in §6 uses):

| model | analytical (keyref H + decisions + templates) | + real zstd(literal) | total | vs zstd -19 | x128 |
|---|---:|---:|---:|---:|---:|
| zstd -19 baseline | -- | -- | 498,748,121 | 1.00x | 59.46 GiB |
| M2 (zero/keyref/literal, no schema) | 57.8 MiB | 233,996,278 | 291,772,816 | **0.585x** | **34.78 GiB** |
| M3 (+ per-group templates) | 73.2 MiB | 151,873,741 | 225,095,803 | **0.451x** | **26.83 GiB** |
| M4a (sorted, whole record XORed, no dictionary) | 22.7 MiB | 597,360,010 | 621,162,685 | 1.245x (**worse**) | 74.05 GiB |
| M4b (M3 + sorted + REPEAT, dictionary kept) | 69.5 MiB | 143,960,668 | 216,836,700 | **0.435x** | **25.85 GiB** |

**Gate cleared**: M2 beats zstd -19 outright (-41.5%) and M3 beats it by
more than 2x (-54.9%), exactly the "wide margin" `PLAN.md` asked for.
(M3's number here is optimistic in one specific way that only became visible
in step 6: the order-0 alphabet of a template slot is bounded by the number of
records in the *sample*.  Priced sample-independently M3 is 0.483x rather than
0.451x -- see §9, "the sample bound".)
Adding back the metadata+pubkey term (32.90 GiB extrapolated -- consistent
with §1/§2's ~31.2 GiB pubkey floor) and ~1 GiB of untouched
manifest/framing, M3's *measured* numbers put the **whole snapshot at
~60.7 GiB**, a bit better than the ~69 GiB estimate in `IDEAS.md`/`PLAN.md`
-- **~1.85x smaller** than the current 112.6 GiB archive, not 1.6x.

**Sorting is worth much less than expected, and this is the most
actionable result here.**  M4 was measured two ways.  M4a is the literal
reading of "XOR against the previous record and zero-run code the result":
it disables the dictionary for the whole record, and a pubkey field that
legitimately differs between adjacent records becomes high-entropy XOR
noise instead of a cheap reference -- 1.245x, *worse than plain zstd -19*.
Same lesson as the naive transposition trap in §5: **removing the
dictionary to gain a different structural win throws away more than it
captures.**

M4b does it properly -- keep M3's template and the dictionary, and add a
REPEAT token for any 32B window or pubkey field equal to the previous
same-group record's window at that offset.  It works, and it is
*unimpressive*: **216,836,700 B, only 3.7% better than M3.**  The mechanism
shows why:

| | M3 | M4b |
|---|---:|---:|
| template pubkey fields repeating | -- | **58.74%** (8,265,736 of 14,071,202) |
| keyref count | 18,328,412 | 10,000,984 |
| keyref entropy | 24.5 MiB | 22.3 MiB |
| decision alphabet | 14.0 MiB | 16.1 MiB |

Nearly 59% of templated pubkey fields *do* repeat between adjacent sorted
records, and converting them to REPEAT tokens removes 45% of all key
references -- yet the keyref entropy only falls 9%, because **the ANS coded
rank over the skewed dictionary was already pricing those exact keys at
almost nothing** (§5: top 16 keys are 40% of references).  REPEAT mostly
relabels references that were already cheap, and it costs 2.1 MiB of extra
decision alphabet to do it.

XOR delta coding of the u64 template fields is a clear *loss*: 58.3 MiB
delta-always vs 31.2 MiB raw-always, and per field `min(raw,delta)` picks
delta on only **3,640 of 23,844 u64 slots**.  Sorting by pubkey leaves
adjacent records uncorrelated in those fields, so XOR just doubles the
alphabet.  The oracle therefore charges the cheaper of the two per field
(31.1 MiB).

**Consequence for the design**: the free dictionary already captures the
redundancy that sorted order + delta coding would capture, so the decision
to sort should rest on the *pubkey stream* (the ~3.9 GiB sorted-set saving,
§3 of `IDEAS.md`), the `(owner,len)` metadata clustering (§6), and
seekability -- not on the data stream, which gains 3.7%.  Since KEYREF
resolves an index into the pubkey array and never needs that array to be
*sorted*, only *agreed*, a streaming encoder that sorts within a chunk
rather than globally keeps almost all of the win at bounded memory.

Per-group template summary confirms §5/§8's design: the top group,
`Tokenkeg` len 165 (4.80M records, 52.3% of the sample), resolves to 0
const + 88 u64 + 32 pubkey + 45 free bytes/record -- SPL token accounts
vary in nearly every byte the template can see, so the win there is almost
entirely the KEYREF mint/wallet fields, not constant-folding. Metaplex
(`metaqbxx...`) groups show the largest constant regions (up to 2,840
B/record on the `len=3228` group), matching the "mostly padding" intuition
for NFT metadata accounts.

## 8. What this implies for the design

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

## 9. Known layouts beat learned templates: the step 6 gate

`oracle`'s M5 declares the struct of each protocol-stable program instead of
guessing it from 8 byte aligned statistics, dispatching on `owner`, `data_len`
and -- for Token-2022 -- the `account_type` byte.  11 layouts are declared
(SPL Token account / mint / multisig, Token-2022 in both variants plus its TLV
extensions, stake, vote, address lookup table, durable nonce, Metaplex
metadata); they cover **6,829,200 of 9,204,364 sampled accounts (74.2%) and
1,308,086,746 of 2,674,206,708 data bytes (48.9%)**.  Everything else keeps
M3.  Run:

```sh
./oracle out/pubkeys.bin out/rand.corpus out/oracle5 1000 16384 > out/oracle_report5.txt
python3 src/m5report.py out/oracle5 19 > out/oracle5_m5report.txt
```

Each cost below is analytic bits (field entropies, reference entropy, decision
entropy) **plus a real `zstd -19`** over that layout's own literal stream:

| scope | M3 | M5 | |
|---|---:|---:|---:|
| layout covered records | 157,871,795 | **136,891,136** | **0.867x** |
| of which token + Token-2022 groups | 114,258,360 | **105,172,020** | **0.920x** |
| whole data stream (covered + fallback) | 211,090,893 | **190,110,234** | **0.901x** |

**Gate cleared**: on the token and Token-2022 groups the plan asked about,
M5 is 8.0% smaller than M3; over all covered records 13.3%; over the whole
data stream 9.9%.  Against the `zstd -19` baseline of 498,748,121 B the data
stream is now **0.381x = 22.66 GiB extrapolated**, which with §7's
metadata+pubkey term (32.90 GiB) and ~1 GiB of framing puts the whole snapshot
at **~56.6 GiB, 1.99x smaller** than the 112.6 GiB archive.  (As in §7, that
total omits the per-account `slot` field, which is free in appendvec order and
costs up to 2.6 GiB in pubkey order -- see §6.  Every whole-snapshot total in
this document is on that convention.)

(M3's 211,090,893 B here is *better* than the 225,095,803 B in §7 for the same
model: splitting one literal stream into thirteen homogeneous ones before
compressing it is worth 4.1% -- 145,625,983 vs 151,873,741 B -- which is a
small free result of its own, and both columns above get it.)

### Per layout

| layout | records | raw B/rec | M3 total | M5 total | |
|---|---:|---:|---:|---:|---:|
| tok-acct | 4,796,143 | 165 | 86,392,342 | 84,507,152 | 0.978x |
| tok-mint | 581,222 | 82 | 13,772,865 | **6,629,992** | **0.481x** |
| t22-ext | 894,656 | 197 | 13,805,302 | 13,625,280 | 0.987x |
| meta | 456,689 | 560 | 38,047,617 | **24,221,115** | **0.637x** |
| alt | 17,747 | 1,541 | 4,772,118 | 5,780,808 | 1.211x |
| nonce | 37,391 | 80 | 509,198 | 1,559,244 | 3.062x |
| stake | 11,153 | 200 | 274,258 | **147,813** | **0.539x** |
| t22-acct165 | 16,631 | 165 | 125,415 | 175,120 | 1.396x |
| t22-mint82 | 17,456 | 82 | 158,411 | 233,016 | 1.471x |

Every row where M3 "wins" is a sampling artifact, not a model result -- see
"the sample bound" below; with that corrected M5 wins everywhere except
`t22-acct165`.

### What a declared field sees that a learned template cannot

The offsets the template is blind to are exactly the ones that pay:

| field | offset | zero | keyref | escape |
|---|---:|---:|---:|---:|
| token account mint | 0 | 0.0% | **100.0%** | 0.0% |
| token account owner (wallet) | 32 | 0.0% | 60.4% | **39.6%** |
| token account delegate | 76 | 98.7% | 0.5% | 0.8% |
| token account close_authority | 133 | 89.5% | 10.5% | 0.0% |
| mint authority | 4 | 0.1% | **71.4%** | 28.5% |
| freeze authority | 50 | 40.4% | 36.4% | 23.2% |
| stake staker / withdrawer | 12 / 44 | 0.0% | 75.2% / 85.6% | 24.8% / 14.4% |
| stake lockup custodian | 92 | 62.1% | 33.0% | 4.9% |
| stake delegation voter | 124 | 1.4% | **98.3%** | 0.3% |
| nonce authority | 8 | 0.0% | 69.8% | 30.2% |
| metadata update_authority | 1 | 2.5% | 70.2% | 27.3% |
| metadata mint | 33 | 11.3% | **87.5%** | 1.3% |
| ALT addresses (823,796 of them) | 56+32k | 1.0% | **80.1%** | 18.9% |
| Token-2022 extension pubkeys | TLV | 36.1% | 61.0% | 2.9% |

Offsets 4, 12, 44, 50, 76, 92, 124, 133 and 1/33 are *not* multiples of 8, so
M3 could not have found any of them, and the `zero` column is the other half
of the point: a `COption::None` pubkey is 32 zero bytes, which M3 charged as a
reference to the all-zero key (`11111111111111111111111111111111`, which is
itself an account) rather than as one decision symbol.

The layouts also validate themselves: a misdeclared offset shows up
immediately as a dictionary hit rate near zero, and every pubkey field
declared above lands between 33% and 100% -- including the ones that depend on
getting a whole borsh/bincode struct right, such as the stake delegation's
voter at offset 124 (98.3%) and the lookup table authority at 22 (56.7%, with
a one byte bincode `Option` tag rather than the four byte `COption` tag the
token program uses).  Token-2022's 83 bytes of padding between the 82 byte
mint and the account_type byte are zero in 100.0% of 93,453 records.

Two honest limitations of the layouts as declared.  The `meta` layout is
applied to every `metaqbxx` account of 65 bytes or more without looking at the
account discriminant at offset 0, so Edition and MasterEdition accounts are
being decoded as Metadata; a real codec should dispatch on that byte as well.
And the `vote` layout declares only the fixed head (version, node pubkey,
authorized withdrawer, commission) and leaves the vote deque and epoch credits
behind it to the generic walker.  Both numbers above are therefore floors for
those two layouts rather than best cases.

The Token-2022 TLV region parses cleanly: **1,129,970 extension entries, 24
distinct types**, dominated by ImmutableOwner (795,732), MetadataPointer
(92,601), TokenMetadata (91,444), TransferFeeAmount (66,062) and
TransferHookAccount (42,981).  Only the pubkey carrying types need a
declaration; the rest are two cheap categorical symbols and a value.

Token account `amount` (offset 64) is worth **15.96 bits** as a varint
(significant byte count + payload), and zstd independently spends
~1.96 B/record on the same field under M3 -- the two models agree to within
2% on the one field where they can be compared directly, which is a useful
cross check on both.

### A layout must *compose* with the learned template, not replace it

The first version of M5 handed every byte a layout did not declare to the
generic ZERO/KEYREF/LITERAL walker.  That is a loss wherever the learned
template already knew something -- Metaplex metadata has up to 2,840 constant
bytes per record (§7) that no hand written layout is going to enumerate.
Composing them (declared fields first, the group's template for the rest, the
generic walker only for what neither claims) moved 188.5 MB of covered data
out of the generic walker (236.9 -> 48.4 MB) and improved the whole data
stream from 0.914x to 0.902x (both measured before the loader stream of §11
was split out), with `t22-ext` flipping from a 1.171x loss to a 0.987x win.
**The codec's dispatch order is: declared layout, then learned template, then
generic tokenizer.**

### The sample bound: M3's template pricing is optimistic, M5's is not

An order-0 alphabet measured on a 1/128 sample cannot cost more than
`log2(records in the group)` bits, so a field that is unique per record looks
cheap no matter how random it is.  **13,614 of M3's 23,844 u64 template slots
are near unique per record in the sample.**  M3 charges 16.7 MiB for them; as
varints (significant byte count + payload, which does not depend on the sample
size) they cost 89.8 MiB, and at 64 bits/slot 98.0 MiB.  So M3's u64 term is
31.2 MiB as measured and **104.3 MiB** priced sample-independently.

The durable nonce group is the clean demonstration: its 32 byte blockhash is
incompressible by construction, and M3 charges it **zero literal bytes** and
~13 bits per 8 byte slot, because 37,391 sampled records cannot show more than
37,391 distinct values per slot.  M5 declares it a 32 byte blob and pays 32
bytes.  That single artifact is the whole of M3's apparent 3.06x win on
`nonce`, and the same effect explains `stake`, `alt` and `t22-mint82`.

Re-pricing both models with every near unique numeric field charged as a
varint:

| scope | M3 | M5 | |
|---|---:|---:|---:|
| layout covered records | 185,390,332 | 143,635,648 | **0.775x** |
| token + Token-2022 groups | 127,980,334 | 109,766,419 | **0.858x** |
| whole data stream | 238,609,430 | 196,854,746 | **0.825x** |

M5's total moves by +3.5% under that correction and M3's by +13.0%: **the
declared layout does not just compress better, it makes the estimate stop
depending on how the corpus was sampled.**  Extrapolated, the data stream is
23.47 GiB rather than 22.66, and the snapshot ~57.4 GiB rather than ~56.6.

### What is left, and it is one field

After all of this, **the single largest line item in the best model is the
token account owner field's escapes**: 39.6% of 4.8M sampled token accounts
carry a wallet that is not itself an account in the snapshot, and those
escaped keys cost 55,647,775 B of zstd -19 output -- **62% of the entire
covered literal cost and 29% of the whole data stream** (~6.6 GiB
extrapolated).

That number is the one measurement in this document that the sampling
*cannot* support, and in the pessimistic direction: a wallet with 12 token
accounts appears in a 1/128 sample about once, so every escape looks unique
even when the full snapshot would let the codec escape the key once and
reference it 11 times afterwards.  §10 measures it without sampling.

## 10. The escape dictionary: the token owner field, unsampled

`snapscan -W` prices one declared field over the whole snapshot instead of a
sample: for every SPL Token account (165 bytes) and every Token-2022 account
(`account_type == 2`) it asks whether the mint at offset 0 and the owner at
offset 32 are keys the snapshot already carries, and for the owners that are
not, it counts how many distinct keys they are and how often each recurs.

```sh
zstd -dc $SNAP | ./snapscan -W out/pubkeys.bin > out/full_stats_tokfield.json
python3 src/tokfield.py out/full_stats_tokfield.json
```

**719,051,458 token accounts**, nothing sampled:

| field | zero | key reference | escape |
|---|---:|---:|---:|
| mint (offset 0) | 0.00% | **99.96%** | 0.04% |
| owner (offset 32) | 0.00% | 62.36% | **37.64%** |

The mint field is essentially free: a mint is always an account of its own, so
719 million of them cost a reference each and nothing more.  The 1/128 sample
predicted 100.0% and 60.4% for these two fields, so **the hit rate itself was
never the sampling problem** -- `refscan` always matched against the complete
key set.  What the sample could not see is what the 270,656,592 escaped owner
keys look like *together*:

| metric | value |
|---|---|
| escapes | 270,656,592 |
| distinct escaped keys | **127,933,600** (2.12 escapes per key) |
| seen exactly once | 110,162,001 (86.1% of distinct, 40.7% of escapes) |
| repeats (occurrences after the first) | 142,722,992 |
| entropy of the repeat distribution | 3,050,160,617.9 bits = **21.37 bits per repeat** |
| most referenced escaped key | **2,708,894 token accounts** |

So the codec should carry a **second dictionary of the keys it has already
escaped**, which costs nothing to transmit -- the key was going to be written
out once anyway -- and reference it afterwards:

| coding | cost over the snapshot | |
|---|---:|---:|
| 32 raw bytes per escape (M5 as measured in §9) | 8,661,010,944 B = **8.07 GiB** | 1.000x |
| escape once, then a flat 26.93 bit rank | 4,574,331,101 B = 4.26 GiB | 0.528x |
| escape once, then an entropy coded rank | 4,475,145,277 B = **4.17 GiB** | **0.517x** |

**The escape dictionary halves the field and saves 3.90 GiB of the archive.**
What is left is a floor, and a hard one: 127,933,600 distinct keys x 32 bytes
= **3.81 GiB of wallet identity that appears nowhere else in the snapshot**,
86% of it belonging to a wallet that owns exactly one token account.  No model
can compress that; it is information the snapshot genuinely carries.

Priced end to end, the owner field of all 719 million token accounts costs
**5.03 GiB** (0.08 GiB of decisions + 0.78 GiB for 448,386,091 references at
14.96 bits + 4.17 GiB of escapes) against **21.43 GiB raw** -- and 3.81 GiB of
those 5.03 are the floor above.

### What it does to the projection

First, a cross check the two measurements make possible.  The sample says the
same escapes are 60,776,736 B in `tok-acct` plus 6,691,646 B in `t22-ext`,
which extrapolates to **8.04 GiB of raw escaped keys**; the unsampled scan
measures **8.07 GiB**.  The 1/128 sample gets the *volume* of this field right
to 0.4% -- it only gets its *compressibility* wrong.

§9's M5 charges those bytes as 32 byte escapes and zstd recovers ~10% of them
(55,647,775 B of output for `tok-acct`'s 62,014,752, and `t22-ext`'s share of
its 6,536,961), which is ~7.2 GiB extrapolated.  Replacing that with the
measured 4.17 GiB takes the data stream from 22.66 GiB to **~19.7 GiB** and the
whole snapshot to **~53.6 GiB, 2.10x smaller than today's archive** (or ~56.1
GiB and 2.01x if the accounts are globally pubkey sorted and the `slot` stream
has to be paid for) -- and unlike everything else in §9, this part of the
estimate is not extrapolated from a sample at all.

The escape dictionary is also the one mechanism here that a streaming encoder
gets for free: it is built from what has already been emitted, so it needs no
preamble, no second pass and no agreement beyond the format itself.

## 11. Where the residual lives, with the layouts in place

Half the account data (51.1%, 1,366,119,962 B of the sample) belongs to groups
no layout covers, and it costs 53.2 MB of the 190.1 MB M5 total (28.0%).
`oracle` now attributes the fallback's bytes to the owner that produced them:

| owner | records | raw B | literal B | zero% | kref% | lit% |
|---|---:|---:|---:|---:|---:|---:|
| `BPFLoaderUpgradeab1e...` | 732 | 83,111,476 | **63,316,660** | 23.8% | 0.0% | 76.2% |
| `cndyAnrLdpjq1Ssp...` (candy machine v2) | 239 | 16,245,899 | 5,726,059 | 64.7% | 0.1% | 35.3% |
| `cmtDvXumGCrqC1Age...` (compressed NFT trees) | 92 | 7,548,416 | 4,527,808 | 40.0% | 0.0% | 60.0% |
| `cndy3Z4yapfJBmL3ShUp...` | 278 | 9,330,192 | 4,310,592 | 53.6% | 0.1% | 46.2% |
| `9xQeWvG816bUx9EPjHmaT...` (Serum/OpenBook) | 12,846 | 57,861,173 | 4,079,445 | 22.7% | 5.6% | 7.1% |
| `compr6CUsB5m2jS4Y...` | 3 | 4,130,272 | 3,571,872 | 12.6% | 0.9% | 86.5% |
| `CoREENxT6tW1HoK8ypY...` | 70,319 | 12,958,695 | 2,123,359 | 0.2% | 0.5% | 16.4% |
| `LBUZKhRxPF3XUpBCjp4Y...` (Meteora DLMM) | 7,612 | 55,246,136 | 1,837,896 | 2.4% | **29.6%** | 3.3% |

**The residual is program binaries.**  732 loader-owned accounts in the sample
-- 93,696 in the snapshot -- hold 83.1 MB of data and produce **54.5% of the
fallback's raw literal bytes**, and they are the one part of this workload
where a schema is the wrong tool and an LZ stage is the right one.  Measured as
its own stream (`out/oracle5/m5/fallback_elf.bin`), ELF compresses **9.6x** --
63,958,510 raw to 6,637,305 at `zstd -19` -- against 2.4x for the rest of the
fallback.  So it is more than half the fallback's raw literal but only 22.8% of
its compressed literal, **0.79 GiB extrapolated**, and splitting it out of the
general literal stream is worth 0.33 MB on the sample by itself (homogeneous
streams again, §9).

Two things follow for the format.  **Keep a real LZ fallback** -- this is what
it is for, and it should be the ELF stream's own compressor with its own
window, not a shared one.  And **`--train` is worth a test but not a
priority**: 9.6x on unclustered ELF suggests the intra-account redundancy is
already being found, and 0.79 GiB is the whole prize.

After the loaders, the fallback is NFT infrastructure (candy machines,
compressed-NFT merkle trees: ~14 MB of literal between them) and DEX order
books, where Serum's 57.9 MB of sampled data costs only 4.1 MB of literal
because it is 22.7% zeros and mostly template-priced -- the dead-market padding
of §3 behaving exactly as predicted.  Meteora's DLMM accounts stand out for a
different reason: **29.6% of their bytes are key references**, the highest of
any uncovered owner, which makes them the best candidate for the next declared
layout.

## 12. One pass costs nothing; segmenting the dictionary costs everything

Phase 2 has to work as a library, not only inside Firedancer, so the question
is whether the free-dictionary design survives a *streaming* encoder: one pass
over the accounts, no pre-built pubkey index, no global sort.  Two things were
measured over the whole snapshot (`snapscan -G`, unsampled, 1,175,713,268
accounts and all 2,204,113,696 references).

The model is a **unified incremental dictionary**.  Every 32 byte key gets a
rank the moment it first appears, whether it arrives as an account's own
pubkey or as a reference to a key the encoder has not met yet -- a reference
to an unseen key costs 32 raw bytes once and a rank thereafter.  The decoder
replays the same stream and assigns identical ranks, so nothing has to be
transmitted, agreed in advance, or sorted.  This is LZ77's backward-reference
discipline applied to keys instead of byte offsets.

### One pass is free

```
two pass, global index:   35.13 GiB keys + 3.84 GiB refs = 38.96 GiB
one pass, unbounded:      35.13 GiB keys + 3.84 GiB refs = 38.96 GiB
```

The two are not merely close, they are the same bytes, and the measurement
says why: **`ref_miss` = 121,739,890 and `own_dup` = 121,739,927.**  Every key
that a forward reference had to escape turns out to be an account, so its 32
bytes were always going to be transmitted; the escape just pays them earlier,
and that account's own record later codes its pubkey as a rank instead of 32
raw bytes.  10.3% of all accounts take that path.  Total 32 byte key
emissions: **1,178,617,687** -- the account count, to within 37.

So the prepass, the global pubkey→rank map and the global sort in §3 and
`PLAN.md` Phase 2 are all unnecessary.  What remains of the sorting argument
is only the sorted-set coding of the key stream itself: `log2(n) - 1.44` bits
per key, which is 3.94 GiB globally but **1.72 GiB from sorting inside a 16k
account chunk buffer** -- a 256 KB radix sort of data already in hand, not a
pass over anything.  Global sorting buys 2.2 GiB beyond that and is not worth
a pass.

### Segmenting the dictionary is catastrophic

If each segment resets the dictionary so that segments decode independently --
the obvious analogy to a zstd frame of blocks -- a hot key is re-emitted at 32
raw bytes in *every segment that references it*:

| segment | segments | ref hit rate | keys re-emitted | total | vs one pass |
|---|---:|---:|---:|---:|---:|
| 16 MiB | 30,092 | 57.65% | 913,105,733 | 64.59 GiB | +25.62 |
| 64 MiB | 7,535 | 60.49% | 848,466,668 | 62.77 GiB | +23.81 |
| 256 MiB | 1,885 | 63.87% | 773,213,802 | 60.66 GiB | +21.70 |
| 1 GiB | 472 | 68.25% | 676,110,976 | 57.94 GiB | +18.97 |
| 4 GiB | 118 | 73.62% | 556,190,393 | 54.57 GiB | +15.61 |
| 16 GiB | 30 | 79.49% | 421,683,890 | 50.80 GiB | +11.83 |
| 64 GiB | 8 | 85.24% | 279,960,115 | 46.82 GiB | +7.86 |
| unbounded | 1 | 94.48% | -37 | **38.96 GiB** | — |

The zstd analogy fails, and it fails in a specific way: a zstd frame boundary
drops matches that degrade to literals, but a dictionary boundary drops
references that degrade to **32 raw bytes**, and the keys that get dropped are
precisely the hot ones.

The escape side says the same thing independently.  Re-running §10's token
owner field under the same segmentation (`snapscan -W -G`, all 719 million
token accounts) reproduces §10 exactly at unbounded -- 142,722,992 reuses over
127,933,600 distinct keys -- and then collapses:

| segment | escape reuse | of §10's 3.90 GiB retained |
|---|---:|---:|
| 16 MiB | 5.44% | ~0.1 GiB |
| 1 GiB | 11.42% | 0.57 GiB |
| 64 GiB | 38.89% | 2.78 GiB |
| unbounded | 52.73% | 3.90 GiB |

At 1 GiB segments the two effects together cost **+22.3 GiB** against a
~53.6 GiB target.

### A preamble hot table does not rescue it

The keys that get re-emitted are by definition the frequently referenced ones,
so the obvious repair is to ship the top K keys once in the preamble (or
inherit them from the previous snapshot, since the head of this distribution
moves slowly) and let every segment resolve them without re-emitting.
Measured, with the hot table built from the 1/128 sample's reference
frequencies (`refscan -H`, 5,330,139 keys) and swept against two segment
sizes:

| segment | hot K | hot hits | escapes | table | total | vs unbounded |
|---|---:|---:|---:|---:|---:|---:|
| 16 MiB | 0 | 0 | 933,402,094 | 0.00 | 64.59 GiB | +25.62 |
| 16 MiB | 65,536 | 1,301,252,790 | 758,072,485 | 0.00 | 59.67 GiB | +20.71 |
| 16 MiB | 1,048,576 | 1,520,355,369 | 587,153,020 | 0.03 | 54.90 GiB | +15.94 |
| 16 MiB | 5,330,139 | 1,634,253,703 | 492,383,618 | 0.16 | 52.39 GiB | +13.42 |
| 1 GiB | 0 | 0 | 699,769,581 | 0.00 | 57.94 GiB | +18.97 |
| 1 GiB | 65,536 | 1,301,252,790 | 672,902,516 | 0.00 | 57.18 GiB | +18.22 |
| 1 GiB | 1,048,576 | 1,520,355,369 | 558,299,753 | 0.03 | 54.00 GiB | +15.04 |
| 1 GiB | 5,330,139 | 1,634,253,703 | 474,375,183 | 0.16 | 51.79 GiB | +12.83 |

The full table is 170 MB of preamble -- free against a 53 GiB archive -- and it
halves the 16 MiB penalty, then stops: **a +12.8 GiB floor remains.**  The
reason closes the loop with §5.  A hot table can only carry keys known to be
hot; the 5,330,139 it holds absorb 74.1% of all references, but the snapshot
has **203,132,064 distinct referenced keys**, so ~198 million sit in a tail no
preamble can carry -- §5 already priced that dictionary at 6.5 GB and rejected
it.  Segmentation punishes precisely that tail, because a tail key referenced
from two segments is emitted twice at 32 raw bytes.

Two secondary results:

- **The hot table flattens segment size.**  Without it, 16 MiB and 1 GiB
  segments differ by 6.65 GiB; with the full table, by 0.60 GiB.  If a future
  design does segment, table size matters far more than segment size.
- **The sample bias is well behaved.**  The K=1,048,576 table takes 69.0% of
  references unsampled against §5's sampled prediction of 75.1% -- the same
  ~9% optimism §5 measured for `H(reference)` (13.73 vs 14.96 bits), in the
  expected direction.  `out/ref_freq_global.bin` holds the exact unsampled
  (fingerprint, count) table for every referenced key, if a future sweep wants
  to remove the sampling entirely.

### What this means for the format

The dictionary must be global.  That does **not** cost parallel decode,
because chunks are already independently decodable *given the key array*:

- **Encode**: one pass, one global dictionary, no prepass, no global sort, no
  segments.  Chunk-local sorting of the 16k account buffer is kept for the
  1.72 GiB it is worth.
- **The key stream is its own stream** -- account pubkeys and escapes
  interleaved in rank order -- and extracting it does not require decoding any
  account data.
- **Decode is two phase**: read the key stream (essentially a memcpy,
  parallel by recorded chunk offsets), then decode every data chunk in
  parallel against the complete key array.

A "segment" is therefore not a dictionary reset but an index checkpoint into
the key stream.  The price, now structural rather than optional, is that the
decoder must materialise the whole key array (~35 GiB) before any data chunk
decodes, which constrains a streaming restore.

Run:

```sh
zstd -dc $SNAP | ./snapscan -R out/pubkeys.bin -G 16,64,256,1024,4096,16384,65536,0 \
  > out/full_stats_segsim.json
zstd -dc $SNAP | ./snapscan -W out/pubkeys.bin -G 16,64,256,1024,4096,16384,65536,0 \
  > out/full_stats_segwallet.json
python3 src/segreport.py out/full_stats_segsim.json

./refscan out/pubkeys.bin out/rand.corpus 4 -H out/hot_sample.bin > out/refscan_hot.txt
zstd -dc $SNAP | ./snapscan -R out/pubkeys.bin -P out/hot_sample.bin \
  -F out/ref_freq_global.bin \
  -G 16:0,16:65536,16:1048576,16:5330139,1024:0,1024:65536,1024:1048576,1024:5330139 \
  > out/full_stats_hotsweep.json
python3 src/hotreport.py out/full_stats_hotsweep.json
```
