# sol_snap_pack — plan

Beat `zstd -1` (Firedancer's snapshot setting) on ratio *and* throughput for
the account payload of a mainnet snapshot.  Ordered so that every phase ends
in a number that either justifies or kills the next phase.

---

## Status

Phase 0 is done and Phase 1 steps 1, 2, 6 and 7 have run and cleared their
gates; step 4 is answered for the part that matters.

- **Step 1** (is the pubkey set its own dictionary?)  Yes, measured over the
  whole snapshot: 2,204,113,696 references, 20.41% of account data, 14.96 bits
  each (`FINDINGS.md` §5).
- **Step 2** (entropy oracle).  M2 beats `zstd -19` on account data by 41.5%,
  M3 by 54.9%, M4b by 56.5%; sorting is worth only 3.7% on the data stream
  (`FINDINGS.md` §7).
- **Step 6** (known layouts).  11 hand written layouts cover 74.2% of accounts
  and 48.9% of account data and beat M3 by **8.0% on the token and Token-2022
  groups**, 13.3% over all covered records and **9.9% over the whole data
  stream** -- 0.381x `zstd -19`, ~56.6 GiB for the whole snapshot
  (`FINDINGS.md` §9).
- **Step 7** (the escape dictionary).  Measured over all 719,051,458 token
  accounts, not sampled: the mint field is a resident key 99.96% of the time
  and the owner field 62.36%, and the 270,656,592 escaped owner keys are only
  **127,933,600 distinct** ones.  Escaping each key once and referencing it
  afterwards costs 4.17 GiB instead of 8.07 GiB -- **3.90 GiB saved**, and it
  is free to a streaming encoder.  That takes the data stream to **~19.7 GiB
  and the whole snapshot to ~53.6 GiB, 2.10x smaller** than today's 112.6 GiB
  (~56.1 GiB and 2.01x if global pubkey sorting makes the `slot` stream a real
  cost)
  (`FINDINGS.md` §10).  The sample and the full scan agree on the *volume* of
  this field to 0.4%, so the correction is purely about its compressibility.
- **Step 4** (where the residual lives).  With the layouts in place the
  uncovered half of the data is **program binaries**: 93,696 loader-owned
  accounts produce 54.5% of the fallback's literal bytes.  ELF compresses 9.6x
  on its own stream, so it is 0.79 GiB of the archive and it wants an LZ stage,
  not a schema.  After that it is NFT infrastructure and DEX order books, and
  Meteora's DLMM accounts (29.6% key references) are the next layout worth
  writing (`FINDINGS.md` §11).

Four results from step 6 change the design, and all four are in `FINDINGS.md`
§9:

1. **Dispatch order is layout, then learned template, then generic
   tokenizer.**  Handing everything a layout does not declare to the generic
   walker throws away the template's constant folding (Metaplex metadata has
   up to 2,840 constant bytes per record); composing them was worth 1.2% of
   the whole data stream on its own.
2. **A `COption::None` pubkey is one decision symbol, not a reference to the
   all-zero key.**  36-99% of the optional pubkey fields are None.
3. **M3's learned template is optimistic by construction on a sample** --
   13,614 of its 23,844 u64 slots are near unique per record, so their order-0
   alphabet is bounded by the sample size and not by the field (a durable
   nonce's blockhash is charged *zero* bytes).  Priced sample-independently M3
   loses 12.9% and M5 only 3.5%: the declared layout is what makes the estimate
   stop depending on the sampling.
4. **One field is the largest single cost in the best model**: the token
   account owner (wallet) at offset 32 escapes the dictionary for 37.6% of all
   719 million token accounts, which is 62% of the covered literal cost.  Step
   7 priced it unsampled and it is half escape dictionary (3.90 GiB
   recoverable) and half floor (3.81 GiB of wallets that own exactly one token
   account and appear nowhere else).

Remaining before Phase 2:

- **Step 3**, reframed: how to *code* a reference rank, not how big to make the
  dictionary (see below).  This is the last open modelling question.
- The 3.81 GiB floor from step 7 (127.9 million wallet pubkeys that appear
  nowhere else) is worth one more look: a wallet that owns exactly one token
  account is 86% of that floor, and nothing in this design addresses it.
- Then Phase 2, with the vocabulary the measurements settled:
  `{zero run, key reference, escape (+ escape dictionary), repeat, literal}`
  over `{declared layout, learned template, generic tokenizer}`, literal
  streams split per account type with the loaders' ELF on its own.

## Phase 0 — instrumentation  *(done)*

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

Results: `FINDINGS.md`.

## Phase 1 — decide the model before writing a codec

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
2. **Entropy oracle.**  *Done -- gate cleared.*  `src/oracle.c` prices a
   corpus under M1-M4 without implementing a codec: metadata contexts from
   `FINDINGS.md` (M1), `ZERO32/KEYREF/LITERAL` tokenization against the full
   pubkey dictionary (M2), per `(owner,data_len)` templates for the top 1,000
   groups by population -- constant bytes, pubkey-shaped fields, small-alphabet
   u64 fields (M3), and sorted order + XOR vs. the previous same-group record
   (M4).  Literal/residual streams are written to disk and priced with a real
   `zstd -19`, not assumed.  Full results, run command and per-stream numbers:
   `FINDINGS.md` §7.

   *Gate* (**cleared**): on the 1/128 sample, against the 498,748,121 B
   `zstd -19` baseline measured on `out/split_full/data.bin` --
   M2 = 291,772,816 B (**-41.5%**), M3 = 225,095,803 B (**-54.9%, 2.2x**).
   M4 as measured is *worse* than the baseline (621,162,685 B) because the
   implementation drops the KEYREF dictionary for the whole record before
   XORing, not just the untemplated residual -- a real negative result, not a
   dead end; see the caveat in `FINDINGS.md` §7 before relying on it.
3. **How to code a reference, not how big to make the dictionary.**  The
   original question -- sweep dictionary size 2^12 … 2^24 -- was answered by
   step 1 and is now moot: 203,132,064 distinct keys are referenced, so the
   dictionary has to be the snapshot's own pubkey array and nothing smaller
   will do.  What is left is the *code* for a rank in a 1.178e9 entry array,
   where the distribution is extremely skewed (top 16 keys are 40% of
   references, `FINDINGS.md` §5) and the entropy is 14.96 bits.  The candidate
   is two level: an ANS symbol for each of the top K keys plus an escape to a
   raw `log2(n)` bit index.  Sweep K (2^8 … 2^24) against the *full snapshot*
   reference frequency distribution -- `snapscan -R` already builds it, it
   just has to dump the histogram rather than only its entropy -- and pick the
   K where the table stops paying for itself.
4. **Where does the residual live?**  *Answered for the part that matters*,
   `FINDINGS.md` §11: with the layouts in place, `oracle` attributes the
   fallback's bytes to the owner that produced them, and **program binaries
   dominate** -- 54.5% of the fallback's literal bytes come from 732
   loader-owned accounts in the sample (93,696 in the snapshot).  On its own
   stream ELF compresses 9.6x (vs 2.4x for the rest of the fallback), which is
   0.79 GiB of the archive.  Conclusion: keep a real LZ fallback and give the
   loaders their own stream; `zstd --train` is worth a test but the whole prize
   is 0.79 GiB, so it is not a priority.  Still open, smaller: the NFT
   infrastructure and DEX order book groups behind the loaders, of which
   Meteora DLMM (29.6% key references) is the best next layout.
5. **Sanity check the extrapolation.**  The 1/128 sample reproduces the real
   archive size to within 2% (114.6 vs 112.6 GiB), so sample-level results can
   be trusted; keep re-checking this as models get richer, since a model that
   overfits the sample will break the identity.
6. **Known layouts beat learned templates.**  *Done -- gate cleared*,
   `FINDINGS.md` §9.  `oracle`'s M5 declares the struct of each
   protocol-stable program (SPL Token account/mint/multisig, Token-2022 in
   both variants plus its TLV extensions, stake, vote, address lookup table,
   durable nonce, Metaplex metadata) and dispatches on `owner`, `data_len` and
   Token-2022's `account_type` byte, with no alignment or hit-rate restriction
   on a field's offset.  The offsets that pay are exactly the ones a learned
   template cannot see: 4 and 50 (mint and freeze authority), 76 and 133
   (delegate, close authority), 12/44/92/124 (stake), 1 and 33 (Metaplex).
   *Gate*: **cleared** -- 0.920x M3 on the token and Token-2022 groups, 0.867x
   over all covered records.  Run:

   ```sh
   ./oracle out/pubkeys.bin out/rand.corpus out/oracle5 1000 16384 > out/oracle_report5.txt
   python3 src/m5report.py out/oracle5 19 > out/oracle5_m5report.txt
   ```

7. **The escape dictionary.**  A key that appears in account data but is not
   itself an account has to be escaped: 32 raw bytes, incompressible.  In the
   1/128 sample that is 39.6% of token account owners and 62% of the covered
   literal cost -- but the sample is biased against it, because a wallet with
   twelve token accounts appears about once in a 1/128 sample, so every escape
   looks unique.  A codec can keep a second dictionary of the keys it has
   already escaped and reference them thereafter, which costs nothing extra to
   transmit.  `snapscan -W` prices the field over all 1.178 billion accounts:
   how often the owner field is resident, and for the rest, how many distinct
   keys they are and what their frequency distribution is worth.

   ```sh
   zstd -dc $SNAP | ./snapscan -W out/pubkeys.bin > out/full_stats_tokfield.json
   ```

   *Gate*: **cleared, and the answer is both** (`FINDINGS.md` §10).  The keys
   do repeat -- 270,656,592 escapes are 127,933,600 distinct keys, so the
   escape dictionary is worth 3.90 GiB -- *and* 86% of those keys belong to a
   wallet with exactly one token account, so 3.81 GiB of wallet identity is a
   floor.  ESCAPE-to-second-dictionary is in the vocabulary; the floor is not
   addressable by any model in this plan and the Phase 2 target is set with it
   included.

## Phase 2 — codec prototype

Only if Phase 1 clears its gates.

- Wire format sketch (see `IDEAS.md` §7): preamble (builtin table, owner
  dictionary, templates) + chunks (target 1-4 MiB input each) + footer index.
- Streams per chunk, each entropy coded on its own: `owner`, `len`, `lamports`,
  `rent`, `exec`, `token`, `keyref`, `escape`, `literal`, plus the `pubkey`
  stream in sorted, front-coded form.  An escaped 32 byte key goes into a
  chunk-local (or stream-local) dictionary and later occurrences reference it
  by rank -- worth 3.90 GiB on the token owner field alone (`FINDINGS.md` §10).
  Loader-owned accounts get their own literal stream, which is the one place a
  real LZ window earns its keep (`FINDINGS.md` §11).  Splitting the literal stream by account type
  before compressing it is worth 3.9% on its own (`FINDINGS.md` §9), so the
  per-layout split is part of the format, not an artifact of the oracle.
- Decoder dispatch, in order: **declared layout** for a known program, then
  the **learned template** for the group, then the **generic tokenizer**
  (`FINDINGS.md` §9).  Each field's decision alphabet is its own -- in
  particular a `COption::None` pubkey is one symbol, not a reference to the
  all-zero key.
- rANS with static per-chunk tables, interleaved 4-8 ways, reusing the ANS and
  `fd_zle` work already in this tree (`src/flamenco/accdb/fd_zle*.c`,
  `ans_network/`).
- API shape the user asked for: iterative feed
  `pack_acct(enc, slot, pubkey, owner, lamports, rent_epoch, exec, data, len)`,
  chunk flush on size threshold.
- **Bounded memory, no global sort.**  This has to work as a library on a
  machine that cannot hold the account set, so the encoder buffers one chunk
  (1-4 MiB, ~8-16k accounts), sorts *within* the chunk, and emits.  That is
  affordable to sort and keeps the `(owner,len)` clustering that collapses
  the metadata streams (§6).  It is sound because **KEYREF resolves an index
  into the pubkey array and never requires that array to be sorted, only
  agreed** -- order of appearance works as well as sorted order, so the
  20.4%-of-data dictionary win survives streaming intact.  What chunk-local
  sorting gives up is part of the sorted-set pubkey coding: front coding
  saves `log2(k)` rather than `log2(n)` bits per key, ~1.6 B/account instead
  of ~3.6 B, so ~2 GiB over the snapshot.  If the accounts DB can iterate in
  index order the chunks come out globally ordered anyway and nothing is
  lost -- **check this against the accdb iterator before committing to a
  format**.  The data stream itself barely cares either way: sorted order is
  worth 3.7% there (M4b vs M3, `FINDINGS.md` §7).
- Deliverables: roundtrip test over the full corpus, a differential fuzzer
  against a reference implementation (the `fuzz_zle*_diff.c` pattern in
  `src/flamenco/accdb/` is the template), and `bench_pack` reporting
  MB/s encode/decode and ratio versus zstd at matched throughput.
- *Gate*: ≥20% smaller than `zstd -1` at ≥1.5 GB/s single core encode, and
  decode no slower than zstd's ~1.5 GB/s.

## Phase 3 — make it real

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

## Non goals / risks

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
