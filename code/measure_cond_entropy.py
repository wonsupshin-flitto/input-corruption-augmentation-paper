"""Measure higher-order conditional entropy and n-gram repetition rate (paper appendix).

Bigram MI places the census corpus (harmed) and the system-log corpus (helped) within 0.02 of
each other. This script checks whether higher-order conditional entropy H(X_t | ctx_k)
separates them.

Method: sample N tokens from the token .bin. Since vocab < 65536, pack n-grams 16 bits at a
time (uint64, exact for n<=4).
  H_joint(n) = n-gram joint entropy (bits). H_cond(k) = H_joint(k+1) - H_joint(k).
  rep_k = 1 - distinct/total (n-gram repetition rate).

Usage:
  python measure_cond_entropy.py --bins data_bin_census data_bin_loghub data_bin_code data_bin_sqlctx \
      --sample-tokens 2000000
"""
import argparse, gzip, json, os
import numpy as np


def load(path, n):
    if os.path.isdir(path):
        path = os.path.join(path, "train.bin")
    a = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(n, len(a))
    return np.asarray(a[:n], dtype=np.uint64)


def pack(x, n):
    T = len(x) - n + 1
    p = np.zeros(T, dtype=np.uint64)
    for i in range(n):
        p |= (x[i:i+T] << np.uint64(16 * i))
    return p


def entropy_bits(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def joint_H(x, n):
    p = pack(x, n)
    _, c = np.unique(p, return_counts=True)
    return entropy_bits(c), 1.0 - len(c) / len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", nargs="+", required=True)
    ap.add_argument("--sample-tokens", type=int, default=2_000_000)
    args = ap.parse_args()
    rows = {}
    for b in args.bins:
        x = load(b, args.sample_tokens)
        Hj, rep = {}, {}
        for n in range(1, 5):  # joint entropy n=1..4 (16bit*4=64bit exact)
            Hj[n], rep[n] = joint_H(x, n)
        # conditional entropy H(X_t|ctx_k) = Hj(k+1)-Hj(k)
        Hc = {k: round(Hj[k + 1] - Hj[k], 4) for k in range(1, 4)}  # k=1..3
        MI1 = round(Hj[1] - Hc[1], 4)  # bigram MI
        rows[os.path.basename(b)] = {
            "tokens": len(x), "H1": round(Hj[1], 4),
            "Hcond1": Hc[1], "Hcond2": Hc[2], "Hcond3": Hc[3],
            "MI1/H1": round(MI1 / Hj[1], 4),
            "rep2": round(rep[2], 4), "rep3": round(rep[3], 4), "rep4": round(rep[4], 4),
        }
    # table
    keys = ["tokens", "H1", "Hcond1", "Hcond2", "Hcond3", "MI1/H1", "rep2", "rep3", "rep4"]
    w = max(len(k) for k in rows)
    print(f"{'corpus':<{w}}" + "".join(f"{k:>10}" for k in keys))
    for b in rows:
        print(f"{b:<{w}}" + "".join(f"{rows[b][k]:>10}" for k in keys))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
