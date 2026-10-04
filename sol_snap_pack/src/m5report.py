#!/usr/bin/env python3
"""m5report.py <oracle outdir> [zstd level] -- combine the per layout analytic
costs from m5_layouts.csv with a real zstd run over the literal streams, so
that M3 and M5 (PLAN.md step 6) are compared on exactly the same records."""
import csv, subprocess, sys, os

d = sys.argv[1]
lvl = sys.argv[2] if len(sys.argv) > 2 else "19"

def zstd(path):
    if not os.path.exists(path): return 0
    if os.path.getsize(path) == 0: return 0
    p = subprocess.run(["zstd","-q","-%s"%lvl,"-c","--long=27","-T1",path],
                       stdout=subprocess.PIPE, check=True)
    return len(p.stdout)

BASE = 498748121   # zstd -19 on the same sample's data.bin (FINDINGS §6/§7)

rows = list(csv.DictReader(open(os.path.join(d,"m5_layouts.csv"))))

# the literal streams only have to be compressed once
z3 = {r["layout"]: zstd(os.path.join(d,"m3lay",r["layout"]+".bin")) for r in rows}
z5 = {r["layout"]: zstd(os.path.join(d,"m5",  r["layout"]+".bin")) for r in rows}

def table(bits3, bits5, title):
    print("\n%s (zstd -%s on the literal streams)" % (title,lvl))
    print("%-12s %10s %13s | %13s %13s %13s | %13s %13s %13s | %7s" % (
          "layout","records","raw B",
          "M3 analytic","M3 zstd","M3 total",
          "M5 analytic","M5 zstd","M5 total","M5/M3"))
    tot = [0,0,0,0,0,0]; fb = [0,0,0,0,0,0]
    for r in rows:
        n, raw = int(r["records"]), int(r["raw_bytes"])
        m3a, m5a = int(round(int(r[bits3])/8)), int(round(int(r[bits5])/8))
        m3z, m5z = z3[r["layout"]], z5[r["layout"]]
        m3t, m5t = m3a+m3z, m5a+m5z
        ratio = "%6.3fx"%(m5t/m3t) if m3t else "     --"
        print("%-12s %10d %13d | %13d %13d %13d | %13d %13d %13d | %7s" % (
              r["layout"],n,raw,m3a,m3z,m3t,m5a,m5z,m5t,ratio))
        if r["layout"].startswith("fallback"):
            for i,v in enumerate((n,raw,m5a,m5z,m5a,m5z)): fb[i]+=v
        else:
            for i,v in enumerate((n,raw,m3a,m3z,m5a,m5z)): tot[i]+=v
    n,raw,m3a,m3z,m5a,m5z = tot
    m3t, m5t = m3a+m3z, m5a+m5z
    print("%-12s %10d %13d | %13d %13d %13d | %13d %13d %13d | %6.3fx" % (
          "COVERED",n,raw,m3a,m3z,m3t,m5a,m5z,m5t,m5t/m3t))
    print("covered data: %.2f B/record under M3, %.2f B/record under M5 (raw %.1f B/record)"
          % (m3t/n, m5t/n, raw/n))
    # the records without a layout are priced by the same model in both columns
    gn,graw = n+fb[0], raw+fb[1]
    g3, g5  = m3t+fb[2]+fb[3], m5t+fb[4]+fb[5]
    print("%-12s %10d %13d | %13s %13s %13d | %13s %13s %13d | %6.3fx" % (
          "GLOBAL",gn,graw,"","",g3,"","",g5,g5/g3))
    print("global data stream: %.3fx the zstd -19 baseline of %d B, x128 = %.2f GiB"
          % (g5/BASE, BASE, g5*128/(1<<30)))

table("m3_bits","m5_bits","as measured")
if "m3_bits_rob" in rows[0]:
    table("m3_bits_rob","m5_bits_rob",
          "with near unique numeric fields charged as varints (sample independent)")
