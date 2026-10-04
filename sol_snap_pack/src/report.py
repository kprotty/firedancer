#!/usr/bin/env python3
"""report.py stats.json -- turn a snapscan JSON report into markdown tables."""
import json, sys, math
sys.path.insert(0, __file__.rsplit('/',1)[0])
from b58 import b58

d = json.load(open(sys.argv[1]))
G = 1<<30
def gib(x): return x/G

acc  = d['accounts']
data = d['bytes_data']
hdr  = d['bytes_hdr']
pad  = d['bytes_pad']
tot  = hdr+data+pad

print("## Snapshot totals\n")
print(f"| metric | value |\n|---|---|")
print(f"| accounts stored | {acc:,} |")
if 'distinct_pubkeys' in d: print(f"| distinct pubkeys | {d['distinct_pubkeys']:,} |")
print(f"| appendvecs | {d['appendvecs']:,} |")
print(f"| slots | {d['slot_min']:,} .. {d['slot_max']:,} ({d['distinct_slots']:,} distinct) |")
print(f"| account headers | {gib(hdr):.1f} GiB ({hdr/acc:.0f} B/acct) |")
print(f"| account data | {gib(data):.1f} GiB ({data/acc:.0f} B/acct) |")
print(f"| alignment padding | {gib(pad):.2f} GiB |")
print(f"| other tar files (manifest etc) | {gib(d['bytes_other_files']):.2f} GiB |")
print(f"| uncompressed total | {gib(d['stream_bytes']):.1f} GiB |")
print(f"| distinct owners | {d['distinct_owners']:,} |")
print(f"| distinct data_len values | {d['distinct_data_len']:,} |")
print(f"| distinct lamports values | {d['distinct_lamports']:,} (table capped) |")
print(f"| executable accounts | {d['executable']:,} |")
print(f"| zero lamports (tombstones) | {d['lamports_zero']:,} ({100*d['lamports_zero']/acc:.1f}%) |")
print(f"| data_len == 0 | {d['data_len_zero']:,} ({100*d['data_len_zero']/acc:.1f}%) |")
print(f"| write_version == 0 | {d['write_version_zero']:,} ({100*d['write_version_zero']/acc:.1f}%) |")
print(f"| account hash == 0 | {d['hash_zero']:,} ({100*d['hash_zero']/acc:.1f}%) |")
print(f"| rent_epoch == u64::MAX | {d['rent_epoch_max']:,} ({100*d['rent_epoch_max']/acc:.1f}%) |")
print(f"| rent_epoch == 0 | {d['rent_epoch_zero']:,} ({100*d['rent_epoch_zero']/acc:.1f}%) |")
print(f"| rent_epoch other | {d['rent_epoch_other']:,} ({100*d['rent_epoch_other']/acc:.4f}%) |")

print("\n## Account data: zeros\n")
z = d['data_zero_bytes']
print(f"| metric | value |\n|---|---|")
print(f"| zero bytes | {gib(z):.1f} GiB = {100*z/data:.1f}% of data |")
print(f"| trailing zero bytes | {gib(d['trailing_zero_bytes']):.1f} GiB = {100*d['trailing_zero_bytes']/data:.1f}% |")
print(f"| all-zero accounts | {d['acct_all_zero_data']:,} |")
print(f"| accounts with no trailing zero | {d['acct_no_trailing_zero']:,} |")
print("\n| zero runs >= | runs | bytes | % of data |\n|---|---|---|---|")
for r in d['zero_runs']:
    print(f"| {r['min_len']} | {r['runs']:,} | {gib(r['bytes']):.1f} GiB | {100*r['bytes']/data:.1f}% |")

bh = d['byte_hist']; bt = sum(bh)
if bt:
    H = -sum((c/bt)*math.log2(c/bt) for c in bh if c)
    nz = sum(bh[1:]); Hnz = -sum((c/nz)*math.log2(c/nz) for c in bh[1:] if c) if nz else 0
    print(f"\norder-0 byte entropy of account data: **{H:.2f} bits/byte** "
          f"(sampled over {bt/2**30:.2f} GiB); excluding zero bytes: {Hnz:.2f} bits/byte "
          f"over the {100*nz/bt:.1f}% of sampled bytes that are non-zero.")

print("\n## Top owners\n")
print("| owner | accounts | % | data | zero% | empty | len range |\n|---|---|---|---|---|---|---|")
for o in d['top_owners'][:25]:
    zf = 100*o['zero_bytes']/o['data_bytes'] if o['data_bytes'] else 0
    print(f"| `{b58(o['owner'])}` | {o['cnt']:,} | {100*o['cnt']/acc:.1f}% | {gib(o['data_bytes']):.2f} GiB | {zf:.0f}% | {o['empty']:,} | {o['dlen_min']}..{o['dlen_max']} |")

cum = 0
print("\nowner concentration: ", end="")
for i,o in enumerate(d['top_owners']):
    cum += o['cnt']
    if i+1 in (1,4,16,64): print(f"top{i+1}={100*cum/acc:.1f}% ", end="")
print()

print("\n## Top data lengths\n")
print("| data_len | accounts | % |\n|---|---|---|")
for e in d['top_data_len'][:25]:
    print(f"| {e['len']:,} | {e['cnt']:,} | {100*e['cnt']/acc:.1f}% |")
cum = sum(e['cnt'] for e in d['top_data_len'][:16])
print(f"\ntop 16 lengths cover {100*cum/acc:.1f}% of accounts")

print("\n## Top lamports values\n")
print("| lamports | accounts | rent exempt min for len |\n|---|---|---|")
for e in d['top_lamports'][:20]:
    v = e['v']
    L = v//6960 - 128 if v and v % 6960 == 0 else None
    print(f"| {v:,} | {e['cnt']:,} | {('len='+str(L)) if L is not None and L>=0 else '-'} |")

print("\n## Top (owner, data_len) groups\n")
print("| owner | len | accounts | zero% | order-0 columnar cost over first N bytes |\n|---|---|---|---|---|")
for g in d['top_groups'][:30]:
    cb = g.get('col_bytes_per_rec'); pfx = g.get('col_prefix')
    s = f"{cb:.1f} B of {pfx} B" if cb else "-"
    print(f"| `{b58(g['owner'])}` | {g['len']:,} | {g['cnt']:,} | {100*g['zero_frac']:.0f}% | {s} |")
print(f"\ndistinct (owner,len) groups: {d['distinct_owner_len_groups']:,}")
