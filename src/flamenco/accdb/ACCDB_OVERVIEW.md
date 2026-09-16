# accounts_db (`accdb`) in Firedancer

The code lives in `src/flamenco/accdb/`. It is a **fork-aware, on-disk account
store** with an in-memory index and a size-classed cache. One writer tile owns
it; executor and RPC tiles join it concurrently.

From `fd_accdb.h`: *"a fork aware database that can be queried to get the current
state of any account as-of a given fork, and update them."*

---

## 1. Storage layout

**Two workspaces + one file** (`src/app/firedancer/topology.c`, `setup_topo_accdb`):

- `accdb` wksp — the shmem control region: index, cache, fork tree, pools,
  metrics (`fd_accdb_shmem_private` in `fd_accdb_private.h`).
- `accdb_data` wksp / the backing **file** — the actual account bytes. Passed to
  tiles as well-known fds `FD_ACCDB_FD_RW` (123461) / `FD_ACCDB_FD_RO` (123460)
  so seccomp can pin syscalls to a fixed fd (`fd_accdb.h`).

**On-disk record** = 72-byte header + data (`fd_accdb_disk_meta_t` in
`fd_accdb_shmem.h`):

```
fd_accdb_disk_meta_t: pubkey[32], size (uint), generation (uint), owner[32]
```

`size` packs the executable flag + a 28-bit data length. Records are
**append-only revisions** — a write never edits an old record in place (except
the same-fork "overwrite" fast path); it appends a new one and the old bytes
become dead weight until compaction.

The file is divided into fixed-size **partitions** (`partition_sz`, e.g. several
GiB each, `partition_cnt` of them). A partition offset is packed as
`[13 bits partition idx | 51 bits byte offset]` (`fd_accdb_private.h`).

---

## 2. In-memory index

**Hash table with chaining:**

- `acc_map[chain_cnt]` — array of `uint` bucket heads, indexed by
  `fd_hash32(pubkey, seed) & (chain_cnt-1)`.
- `acc_pool` — pool of `fd_accdb_accmeta_t`, each a **single 64-byte cache
  line** (`fd_accdb_private.h`). One element = **one version of one account**:

```
key = { pubkey[32], generation }      // the fork-generation that wrote it
map.next                              // hash chain link
executable_size                       // packed: exec bit + cache_valid/claim/pd_write + 28-bit len
lamports
offset_fork                           // packed: 48-bit file offset | 16-bit fork_id
cache_idx                             // packed (size_class, line) into the cache, if resident
```

Multiple versions of the same pubkey (from different forks) sit on the same hash
chain. A lookup walks the chain and picks the newest version *visible on the
query fork*.

---

## 3. Forks / MVCC

`fork_pool` holds a tree of forks (`parent_id` / `child_id` / `sibling_id`,
`fd_accdb_private.h`). Each fork also has a **`descends_set`** — a bitset of all
ancestor fork ids.

**Visibility test** for a version with generation `g` and fork id `f`, queried
as-of fork `F` (`fd_accdb.h` header comment; `fd_accdb.c` `acquire_inner`):

1. **Fast path:** `g <= root_generation` → unconditionally visible (any rooted
   fork is an ancestor of everything live). No fork-slot lookup, which is what
   makes fork-slot recycling safe.
2. **Slow path:** `descends_set(F).test(f)` → is `f` an ancestor of `F`?

`generation` is a monotonic counter that is **never recycled**; `fork_id` slots
*are* recycled, so on-disk metadata keys on generation.

**Lifecycle operations:**

| API | What it does |
|---|---|
| `fd_accdb_attach_child(parent)` | new fork for a new block; runs inline on replay (T1) |
| `fd_accdb_advance_root(fork)` | commit/finalize a fork: drops competing sibling subtrees, tombstones older versions of accounts rewritten on the new root, frees the old root slot. Offloaded to T2. |
| `fd_accdb_purge(fork)` | rare — delete a fork + all descendants (leader equivocation). Offloaded to T2. |

---

## 4. The cache

`fd_accdb_cache.h`: **8 geometric size classes** (128 B, 512 B, 2 K, 8 K, 32 K,
128 K, 1 M, 10 M). Operator gives a total byte budget (`cache_footprint`);
`fd_accdb_cache_class_cnt()` splits it across classes using access-density
weights from mainnet replay, with per-class minimums.

- Each slot = 88-byte `fd_accdb_cache_line_t` header (key, `acc_idx`, `refcnt`,
  `owner[32]`, …) + data capacity.
- **Class 7 (10 MiB) slots double as writable staging buffers** — every writable
  acquire gets one, data is copied into a right-sized slot on commit.
- **Eviction:** per-class CLOCK / second-chance sweep (`clock_hand[c]`), a
  per-class Treiber-stack free list (`cache_free[c]`), and a background
  pre-eviction pass that keeps free slots above a low-water mark.
- **Reservation counters** (`cache_class_used[c]`) guarantee a worst-case
  transaction batch (or 5-txn bundle) can always execute fully in memory — see
  the derivation in `fd_accdb_cache.h`.
- The `accmeta` embeds `cache_idx` plus `cache_valid` / `cache_claim` bits;
  `cache_claim` is a short CAS lock protecting the accmeta↔cache-line binding
  against concurrent readers and the evictor (`fd_accdb.c`
  `evict_clear_acc_cache_ref`).

---

## 5. The access API (acquire / release)

This is how the executor reads and writes accounts (`fd_accdb.h`):

```c
fd_accdb_acquire(accdb, fork_id, n, pubkeys, writable[], out_accs);
// ... executor mutates out_accs[i].data / lamports / owner ...
// set out_accs[i].commit = 1 to persist, leave 0 to discard
fd_accdb_release(accdb, n, accs);
```

`acquire` (`fd_accdb_acquire_inner`):

1. For each pubkey, walk the hash chain, pick the newest visible version.
2. Reserve cache slots (with a `MAYBE_PROGRAMDATA` placeholder for the implicit
   programdata account, refunded by `acquire_b`).
3. Pin the cache line (`refcnt++`); on a **miss**, `cold_load_acc` reads the
   record from the file (`preadv2`) into a freshly allocated slot.
4. Writables also get a 10 MiB staging buffer so data can grow/shrink or be
   rolled back on transaction failure.

`release` with `commit=1`:

- **Same fork as the existing version** → in-place "overwrite": mutate the
  accmeta's size/lamports/offset, append new data bytes at the layer-0 write
  head.
- **New version** → allocate a new accmeta, append the record, CAS it onto the
  hash-chain head, and record it in a **txn** entry linked to the fork (so
  `advance_root` / `purge` can find and tombstone/undo it). See `fd_accdb.c`
  `release_inner`.

**Concurrency contract:** many concurrent readers of one (pubkey, fork) are
fine; a writable acquire must not overlap any other acquire of the same (pubkey,
fork); cross-fork acquires are always safe *provided the parent fork finished
all its writes before the child was attached* — which the replay scheduler
guarantees.

**Convenience wrappers:** `fd_accdb_read_one`, `write_one`, `exists`,
`lamports`, `probe_pd_this_fork`. The slot-boundary/system-account helpers used
by the runtime are in `src/flamenco/runtime/fd_accdb_svm.h`
(`fd_accdb_svm_open_rw`, `_credit`, `_write`, …), which also update the bank
LtHash and capitalization.

**Read-only joiners** (`fd_accdb_join_readonly` + `fd_accdb_read_one_nocache`):
for tiles that `mmap` the data file read-only (rpc, resolv, gui, snaprd/snapzp).
They never touch a cache line — on a cache hit they do an ABA try-read loop, on
a miss they `preadv2` from disk. They only write their own epoch slot.

---

## 6. Write heads, tiering, and compaction

Three compaction **layers** (Hot → Warm → Cold, `fd_accdb_shmem.h`):

- `whead[0]` — hot execution write head. Multi-writer, advanced by atomic
  fetch-add during acquire/release.
- `whead[1]`, `whead[2]` — compaction-owned, single-writer.

When a partition accumulates dead bytes past **~30%**
(`FD_ACCDB_COMPACTION_THRESHOLD_PCT`), `fd_accdb_shmem_bytes_freed` enqueues it
on that layer's `compaction_dlist`.

**The accdb tile (T2)** calls `fd_accdb_background()` in a loop
(`fd_accdb_tile.c`). Each call:

1. First services any pending `advance_root` / `purge` command from T1
   (`cmd_op` slot).
2. Otherwise does one step of compaction per layer (`background_compact` in
   `fd_accdb.c`): read the oldest queued partition, for each record check
   whether a live index entry still points at it; if so relocate it to the next
   layer's write head and CAS the accmeta's offset; if not, skip it as dead.

**Epoch-based safe reclamation:** a compacted partition isn't freed immediately
— it's tagged with the current `epoch` and pushed on `deferred_free_dlist`.
`joiner_epochs[i]` (and external RO-consumer fseqs passed via
`external_epoch_slots[]`) record the epoch each joiner observed entering its
critical section. A partition is released back to the pool only once every
joiner's observed epoch has advanced past its tag, guaranteeing no in-flight
read or `pwritev2` still references it (`fd_accdb_private.h`).

---

## 7. Threading model

From `fd_accdb.h` (`fd_accdb_background` header comment):

| Role | Tiles | Calls |
|---|---|---|
| **T1** | replay | `attach_child` (inline), `advance_root` / `purge` (offloaded), `acquire` / `release` |
| **T2** | `accdb` tile | only `fd_accdb_background` |
| **T3** | `execle`, `execrp`, resolv | `acquire` / `release` |
| RO | rpc, gui, snaprd, snapzp | `read_one_nocache` via a read-only join |

`acquire`/`release` run concurrently from T1 and any number of T3.

---

## 8. Snapshots

- **Loading:** the snapin tile calls `fd_accdb_snapshot_write_one` /
  `_write_batch` (batch of 8 with prefetch). `snapshot_load_begin/end` toggle
  tiering backfill for accounts that will never get a second write.
  `fd_accdb_reset` reinitializes for a re-load.
- **Creating:** `snapshot_sync` is a small state machine
  (`IDLE → START_FULL/START_INCR → RUNNING → DONE`) that tells the accdb tile to
  inhibit compaction while a snapshot is produced. A **writer barrier**
  (`fd_accdb_shmem_writer_barrier_capture/poll`) lets the producer wait for
  in-flight writes to land.
- **Deltas:** accdb separately tracks the set of pubkeys modified since the last
  full snapshot (`fd_accdb_delta`, a hash set) — needed to build incremental
  snapshots and tombstones, since fork info at the full-snapshot slot is
  discarded.

---

## 9. ZLE (the `ripatel/zle` branch's work)

`fd_zle.h`: a **zero-run-length compression** codec tuned on individual Solana
mainnet account data — ~10× faster than LZ4/Zstd at similar ratio. It is not yet
wired into the acquire/release path (no `fd_zle` references in `fd_accdb.c`); the
branch is still iterating on the encoder (`fd_zle2..5`, benches, fuzzers).
Intended use is compressing account data on disk / in snapshots.
