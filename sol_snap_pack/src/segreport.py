#!/usr/bin/env python3
"""Price the one pass unified dictionary from snapscan -G output.

Every distinct key costs 32 raw bytes the first time a segment sees it --
whether it arrives as an account's own pubkey or as a reference to a key the
segment has not met yet.  Both land in the same rank space, so an escape is
not an extra cost: it is the same 32 bytes, paid earlier.  What segmentation
actually costs is re-emitting a key in every segment that references it."""
import json, math, sys

GIB = 1024.0**3
d = json.load(open(sys.argv[1]))
H_GLOBAL = d.get("ref_entropy_bits") or 14.96
REFS     = d["ref_hits"]
NACCT    = d["accounts"]

def price(r):
    ins  = r["own_new"] + r["ref_miss"]          # 32 byte key emissions
    h    = min(H_GLOBAL, math.log2(max(r["peak_dict"], 2)))
    ranks = r["ref_hit"] + r["own_dup"]          # coded as a dictionary rank
    return ins, ins*32.0/GIB, ranks*h/8.0/GIB, h

# two pass baseline: every key once in the pubkey stream, every ref at H
base = NACCT*32.0/GIB + REFS*H_GLOBAL/8.0/GIB

print(f"accounts {NACCT:,}   references {REFS:,}   global H {H_GLOBAL:.2f} bits")
print(f"two pass, global index: {NACCT*32.0/GIB:.2f} keys + "
      f"{REFS*H_GLOBAL/8.0/GIB:.2f} refs = {base:.2f} GiB\n")
hdr = ("  segment    segs    hit%   32B emissions   re-emitted     keys"
       "     refs     total   vs 2-pass")
print(hdr); print("  " + "-"*(len(hdr)-2))
for r in sorted(d["segsim"], key=lambda x: x["seg_mib"] if x["seg_mib"] else 1<<62):
    ins, kg, rg, h = price(r)
    tot  = kg + rg
    name = f"{r['seg_mib']} MiB" if r["seg_mib"] else "unbounded"
    print(f"  {name:>9} {r['segments']:7,} {100*r['ref_hit']/(r['ref_hit']+r['ref_miss']):6.2f}% "
          f"{ins:15,} {ins-NACCT:12,} {kg:8.2f} {rg:8.2f} {tot:9.2f} GiB {tot-base:+9.2f}")

u = next(r for r in d["segsim"] if r["seg_mib"] == 0)
print(f"\n  unbounded: {u['own_dup']:,} accounts ({100*u['own_dup']/NACCT:.1f}%) had their pubkey")
print(f"  already defined by an earlier reference -- it is a rank in the record,")
print(f"  not 32 bytes, so the escape and the pubkey stream entry are the same bytes.")
