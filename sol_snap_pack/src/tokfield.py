#!/usr/bin/env python3
"""tokfield.py <snapscan -W json> -- what the SPL token owner field costs over
the whole snapshot, with and without a second dictionary holding the keys the
codec has already escaped."""
import json, math, sys

j = json.load(open(sys.argv[1]))
N  = j["tokfield_accounts"]
mz, mh, me = (j["tokfield_mint_%s"%k] for k in ("zero","hit","escape"))
oz, oh, oe = (j["tokfield_owner_%s"%k] for k in ("zero","hit","escape"))
D  = j["tokfield_owner_escape_distinct"]
S  = j["tokfield_owner_escape_singletons"]
top= j["tokfield_owner_escape_max_cnt"]
H  = j["tokfield_owner_escape_entropy_bits"]
GiB = 1<<30

def pct(a,b): return 100.0*a/b if b else 0.0
print("token accounts (SPL Token 165 B + Token-2022 account_type 2): %d"%N)
print("  mint  @0 : zero %.2f%%  keyref %.2f%%  escape %.2f%%"%(pct(mz,N),pct(mh,N),pct(me,N)))
print("  owner @32: zero %.2f%%  keyref %.2f%%  escape %.2f%%"%(pct(oz,N),pct(oh,N),pct(oe,N)))
print("\nescaped owner keys (not accounts in this snapshot):")
print("  escapes            %d"%oe)
print("  distinct keys      %d  (%.2f escapes per key)"%(D,oe/D if D else 0))
print("  seen exactly once  %d (%.1f%% of distinct, %.1f%% of escapes)"%(S,pct(S,D),pct(S,oe)))
print("  most popular key   %d token accounts"%top)
print("  frequency entropy  %.1f bits total = %.2f bits per escape"%(H,H/oe if oe else 0))

# Each escaped key is transmitted in full once and referenced by rank after
# that.  The rank is charged a flat log2(distinct) bits, which is an upper
# bound: entropy coding it against the observed skew can only help, and the
# aggregate entropy printed above cannot be used for it directly because it
# also counts each key's first occurrence, which already paid 32 bytes.
R    = j.get("tokfield_owner_escape_repeats",0)
Hrep = j.get("tokfield_owner_escape_repeat_entropy_bits",0.0)
raw   = oe*32
flat  = D*32 + (oe-D)*math.log2(D)/8 if D else 0
ans   = D*32 + Hrep/8
print("\nwhat the field's escapes cost over the snapshot:")
print("  32 raw bytes each (no escape dictionary)   %15.0f B = %6.2f GiB"%(raw,raw/GiB))
print("  escape once + flat %.2f bit rank          %15.0f B = %6.2f GiB  (%.3fx)"%(math.log2(D),flat,flat/GiB,flat/raw))
print("  escape once + entropy coded rank           %15.0f B = %6.2f GiB  (%.3fx)"%(ans,ans/GiB,ans/raw))
print("    %d repeats at %.2f bits each; the %d first occurrences are the floor (%.2f GiB)"
      % (R, Hrep/R if R else 0, D, D*32/GiB))
print("  -> saving %.2f GiB of the snapshot"%((raw-ans)/GiB))
hit_bits = 14.96   # H(reference) over the whole snapshot, FINDINGS §5
dec = 0.0
for c in (oz,oh,oe):
    if c: dec -= c*math.log2(c/N)
tot = dec/8 + oh*hit_bits/8 + ans
print("\nowner field, whole snapshot, all three cases together:")
print("  decisions %.2f GiB + %d keyrefs at %.2f bits %.2f GiB + escapes %.2f GiB = %.2f GiB"
      % (dec/8/GiB, oh, hit_bits, oh*hit_bits/8/GiB, ans/GiB, tot/GiB))
print("  raw: %.2f GiB (%d accounts x 32 B)"%(N*32/GiB,N))
