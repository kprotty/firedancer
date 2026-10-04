#!/usr/bin/env python3
"""Price a preamble hot table against segment size (snapscan -G N:K output).

A key in the hot table resolves in every segment without being re-emitted, so
the table attacks exactly what segmentation costs.  The table itself is K*32
bytes, charged once -- or free if the encoder inherits it from the previous
snapshot, since the head of the reference distribution moves slowly."""
import json, sys

GIB = 1024.0**3
d = json.load(open(sys.argv[1]))
H     = d.get("ref_entropy_bits") or 14.96
REFS  = d["ref_hits"]
NACCT = d["accounts"]
base  = NACCT*32.0/GIB + REFS*H/8.0/GIB          # one pass, unbounded dictionary

print(f"accounts {NACCT:,}   references {REFS:,}   H {H:.2f} bits")
print(f"one pass, unbounded dictionary (no segments): {base:.2f} GiB\n")
hdr = ("  segment    hot K   hot hits   escapes   re-emitted   table    keys"
       "    refs    total  vs unbounded")
print(hdr); print("  " + "-"*(len(hdr)-2))
for r in sorted(d["segsim"], key=lambda x: (x["seg_mib"] or 1<<62, x["hot_k"])):
    ins   = r["own_new"] + r["ref_miss"]
    tbl   = r["hot_k"]*32.0/GIB
    keys  = ins*32.0/GIB
    refs  = (r["ref_hit"] + r["own_dup"])*H/8.0/GIB
    tot   = keys + refs + tbl
    name  = f"{r['seg_mib']} MiB" if r["seg_mib"] else "unbounded"
    print(f"  {name:>9} {r['hot_k']:>8,} {r['hot_hit']:>10,} {r['ref_miss']:>9,} "
          f"{ins-NACCT:>12,} {tbl:7.2f} {keys:7.2f} {refs:7.2f} {tot:8.2f} {tot-base:+8.2f}")
print("\n  (table charged in full; inherited from the previous snapshot it is free)")
