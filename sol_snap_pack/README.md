# sol_snap_pack

Experiments towards a Solana snapshot specific account compressor that beats
`zstd -1` (Firedancer's `FD_BACKUP_ZSTD_LEVEL`) on ratio *and* speed.

Read in this order:

- `FINDINGS.md` — what the mainnet snapshot actually contains (measured).
- `IDEAS.md` — the codec design ideas and why the measurements support them.
- `PLAN.md` — phased plan, with the gate that has to be cleared at each step.

`SOL_SNAP_PACK.md` is all three stitched into one standalone document (with a
short summary up front and a reproduction appendix), for saving or sending
elsewhere.  It is a point-in-time copy: edit the three source docs and
re-stitch rather than editing both.

## Data

- Full mainnet snapshot: `../snapshot.tar.zst` (symlink to
  `/home/ubuntu/snapshot/snapshot-447858864-*.tar.zst`, 113 GiB, slot
  447858864).  **Do not delete — it took hours to download.**
- Extracted incremental snapshot used for quick iteration:
  `/home/ubuntu/snap/incr.tar.zst`.

## Tools (`src/`, build with `make`)

| tool | what it does |
|---|---|
| `snapscan` | streams an uncompressed snapshot tar on stdin, parses tar + appendvecs, prints a JSON statistics report, and can write sample corpora, a pubkey dump and the manifest |
| `corpstat` | structural analysis of a corpus: conditional entropies of the metadata fields, per `(owner,len)` field profiles, repeated 32 byte window coverage |
| `corpsplit` | struct-of-arrays transform of a corpus into per field streams, optionally clustering records by `(owner, data_len, pubkey)` |
| `refscan` | matches account data against the snapshot's own pubkey set; prints reference coverage and the entropy of the reference distribution |
| `corpgroup` | extracts one `(owner,len)` group's data in raw / transposed / per-key-field layouts, for layout experiments |
| `oracle` | prices a corpus under the M1-M5 models from `PLAN.md` (metadata, tokenized data, per-group templates, sorted+XOR, hand written layouts for known programs) without implementing a codec; writes residual literal streams for a real zstd comparison |
| `bench_baseline.sh` | zstd baseline table over a directory of streams (`src/`) |
| `m5report.py` | combines `oracle`'s per layout analytic costs with a real zstd run over the literal streams, so M3 and M5 are compared on the same records (`src/`) |
| `tokfield.py` | prices the SPL token owner field from a `snapscan -W` report, with and without a dictionary of already escaped keys (`src/`) |
| `report.py` | renders a `snapscan` JSON report as markdown (`src/`) |

### Typical runs

```sh
SNAP=/home/ubuntu/snapshot/snapshot-447858864-*.tar.zst

# full statistics pass + corpora (~20 min; add -p for a distinct pubkey count,
# which needs a table larger than 2^30 slots for this snapshot and ~110 GiB RSS)
zstd -dc $SNAP | ./snapscan -A out/av.corpus -a 256 -s out/rand.corpus -r 7 \
                            -m out/manifest.bin > out/full_stats.json
python3 src/report.py out/full_stats.json > out/report_full.md

# dump every account pubkey (for refscan)
zstd -dc $SNAP | ./snapscan -k out/pubkeys.bin > /dev/null

# structure of the sampled accounts
./corpstat out/rand.corpus 24 > out/corpstat.txt
./refscan  out/pubkeys.bin out/rand.corpus 4 > out/refscan.txt

# zstd baselines, natural order and clustered order
./corpsplit out/rand.corpus out/split && src/bench_baseline.sh out/split
./corpsplit out/rand.corpus out/split_sorted --sort && src/bench_baseline.sh out/split_sorted

# the same reference scan over the whole snapshot rather than a sample
zstd -dc $SNAP | ./snapscan -R out/pubkeys.bin > out/full_stats_ref.json

# price one declared field (the SPL token owner) over all 1.18e9 accounts,
# including how many of the escaped keys are distinct
zstd -dc $SNAP | ./snapscan -W out/pubkeys.bin > out/full_stats_tokfield.json
python3 src/tokfield.py out/full_stats_tokfield.json

# entropy oracle: price M1-M5 against zstd -19 on the same sample (~2 min,
# 39 GiB RSS -- the 1.178e9 key dictionary dominates)
./oracle out/pubkeys.bin out/rand.corpus out/oracle5 1000 16384 > out/oracle_report5.txt
src/bench_baseline.sh out/oracle5 "1 3 19"        # M2/M3/M4 residual streams
python3 src/m5report.py out/oracle5 19            # M3 vs M5, per layout

# one account type's layout, e.g. SPL token accounts
./corpgroup out/rand.corpus TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA 165 out/tok165
```

## Corpus format (`SNAPCORP1`)

16 byte magic, then records:

```
u64 slot; u64 lamports; u64 rent_epoch; u64 data_len;
u8  pubkey[32]; u8 owner[32]; u8 executable; u8 pad[7];
u8  data[data_len];
```

`rand.corpus` is a pubkey-hash sample (unbiased per account); `av.corpus` takes
whole appendvecs and therefore preserves the on-disk ordering and locality.

## Results already in `out/`

| file | what |
|---|---|
| `full_stats_pass2.json`, `report_full.md` | the full snapshot statistics pass (`full_stats3.json` is an independent repeat that agrees exactly) |
| `full_stats_ref.json` | the unsampled pubkey reference scan over all 1.18e9 accounts |
| `rand.corpus` (3.6 GB) | complete 1/128 sample, 9,204,364 accounts |
| `av.corpus` (1.8 GB) | whole-appendvec sample |
| `pubkeys.bin` (37.7 GB) | every account pubkey, for `refscan` / `snapscan -R` |
| `corpstat_rand.txt` | conditional entropies and field profiles |
| `refscan_rand.txt` | pubkey reference coverage against the full key set |
| `bench_full.txt`, `bench_full_sorted.txt` | zstd baselines, natural and clustered order |
| `tok165/` | the SPL token account group, raw / transposed / per field |
| `manifest.bin` (189 MB) | the snapshot manifest, untouched so far |
| `oracle_report5.txt`, `oracle5/`, `oracle5_m5report.txt` | the M1-M5 oracle run, its per layout literal streams and the M3 vs M5 table |
| `full_stats_tokfield.json`, `tokfield_report.txt` | the unsampled scan of the SPL token mint/owner fields and what it costs |

`out/` is gitignored; the whole directory is ~50 GB and is reproducible from
the snapshot with the commands above.
