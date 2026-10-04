#!/usr/bin/env python3
"""Independent check of the fp32 reference forward pass in transformer.c.

Runs ./ans_transformer --dump, which prints the token ids, every
binarization site's bit vector in forward order, and the logits for a
tiny config, then recomputes all of it here from the same seed.  No
code is shared: the weights come out of the same splitmix64 stream
(exactly representable in fp32, so both sides see identical weights)
and everything after that is plain Python.

Bits must match exactly.  Logits are fp32 in C and double here, so
they are compared to a relative tolerance.
"""

import math, subprocess, sys

M64 = (1 << 64) - 1

class Rng:
    def __init__(self, s): self.s = s
    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)
    def w(self, lo, hi):
        return lo + (hi - lo) * ((self.next() >> 40) / 16777216.0)

def pad64(n): return (n + 63) & ~63

def softplus2(w):
    return w + math.log2(1.0 + 2.0 ** -w) if w > 0 else math.log2(1.0 + 2.0 ** w)

class Lin:
    def __init__(self, rng, n_out, n_in):
        self.n_out, self.n_in, self.s_in = n_out, n_in, pad64(n_in)
        self.w = [[rng.w(-2.0, 2.0) for _ in range(n_in)] for _ in range(n_out)]
        pad = (self.s_in - n_in) * softplus2(0.0)          # padded features are w=0
        self.b = [sum(softplus2(v) for v in row) + pad for row in self.w]
    def __call__(self, bits):
        return [sum(w for w, x in zip(row, bits) if x) - b for row, b in zip(self.w, self.b)]

def layernorm(v):
    n = len(v)
    m = sum(v) / n
    s = sum((x - m) ** 2 for x in v) / n
    inv = 1.0 / math.sqrt(s + 1e-5)
    return [(x - m) * inv for x in v]

def main():
    d, nh, F, nl, S, V = 16, 2, 32, 2, 6, 16
    dh = d // nh
    rng = Rng(12345)
    emb = [[rng.w(-1.0, 1.0) for _ in range(d)] for _ in range(V)]
    pos = [[rng.w(-1.0, 1.0) for _ in range(d)] for _ in range(S)]
    layers = []
    for _ in range(nl):
        layers.append(dict(zip("q k v o f1 f2".split(),
            [Lin(rng, d, d), Lin(rng, d, d), Lin(rng, d, d),
             Lin(rng, d, d), Lin(rng, F, d), Lin(rng, d, F)])))
    wout = Lin(rng, V, d)
    tok = [rng.next() % V for _ in range(S)]

    bits_out, sites = [], []
    def binarize(vals):                    # tau is the mean over the sequence
        n = len(vals[0])
        tau = [sum(v[i] for v in vals) / len(vals) for i in range(n)]
        bs = [[1 if v[i] >= tau[i] else 0 for i in range(n)] for v in vals]
        bits_out.extend("".join(str(b) for b in row) for row in bs)
        return bs

    res = [[emb[tok[t]][i] + pos[t][i] for i in range(d)] for t in range(S)]
    for L in layers:
        b = binarize([layernorm(res[t]) for t in range(S)])
        q = [L["q"](b[t]) for t in range(S)]
        k = [L["k"](b[t]) for t in range(S)]
        v = [L["v"](b[t]) for t in range(S)]
        for t in range(S):
            for h in range(nh):
                q[t][h*dh:(h+1)*dh] = layernorm(q[t][h*dh:(h+1)*dh])
                k[t][h*dh:(h+1)*dh] = layernorm(k[t][h*dh:(h+1)*dh])
        ctx = [[0.0]*d for _ in range(S)]
        for h in range(nh):
            for t in range(S):
                sc = [sum(a*c for a, c in zip(q[t][h*dh:(h+1)*dh], k[s][h*dh:(h+1)*dh]))
                      / math.sqrt(dh) for s in range(t+1)]
                mx = max(sc)
                e = [math.exp(x - mx) for x in sc]
                tot = sum(e)
                for s in range(t+1):
                    for i in range(dh):
                        ctx[t][h*dh+i] += (e[s]/tot) * v[s][h*dh+i]
        bo = binarize(ctx)
        for t in range(S):
            o = L["o"](bo[t])
            for i in range(d): res[t][i] += o[i]
        bf = binarize([layernorm(res[t]) for t in range(S)])
        ff = [L["f1"](bf[t]) for t in range(S)]
        bh = binarize(ff)
        for t in range(S):
            o = L["f2"](bh[t])
            for i in range(d): res[t][i] += o[i]
    bl = binarize([layernorm(res[t]) for t in range(S)])
    logit = [wout(bl[t]) for t in range(S)]

    out = subprocess.run(["./ans_transformer", "--dump"], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    c_tok = [int(x) for x in out[0].split()[1:]]
    c_bits = [l for l in out if l and l[0] in "01"]
    c_logit = [[float(x) for x in l.split()[1:]] for l in out if l.startswith("logit")]

    ok = True
    if c_tok != tok:
        print("tokens differ", c_tok, tok); ok = False
    if len(c_bits) != len(bits_out):
        print("site count differs: C %d, py %d" % (len(c_bits), len(bits_out))); ok = False
    bad = sum(1 for a, b in zip(c_bits, bits_out) for x, y in zip(a, b) if x != y)
    n = sum(len(b) for b in bits_out)
    print("binarization sites: %d rows, %d bits, %d differ" % (len(bits_out), n, bad))
    ok &= bad == 0
    emax = 0.0
    rms = math.sqrt(sum(x*x for row in logit for x in row) / (S*V))
    for a, b in zip(c_logit, logit):
        for x, y in zip(a, b):
            emax = max(emax, abs(x - y))
    print("logits: max |C - py| %.3e, rms %.3f, ratio %.2e" % (emax, rms, emax/rms))
    ok &= emax/rms < 1e-4
    print("fp32 reference forward pass:", "matches" if ok else "DIFFERS")
    return 0 if ok else 1

sys.exit(main())
