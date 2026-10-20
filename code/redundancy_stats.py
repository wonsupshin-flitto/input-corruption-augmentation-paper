"""Model-independent redundancy metric, measured directly on token .bin files.

Puts synthetic Markov sources and real data
(web/transcripts) on the **same yardstick** for the redundancy axis. Compression ratio
(circular, since LM = compression) is reported for reference only; the primary metrics
are entropy and n-gram repetition.

Metrics (all computed on the token stream):
  - H1        : unigram entropy (bits/token)
  - Hrate     : bigram conditional entropy H(X_t|X_{t-1}), approx. entropy rate (lower = more redundant)
  - rep_ngram : n-gram repetition rate = 1 - distinct/total (higher = more redundant)
  - ttr       : type-token ratio (distinct unigram / total)
  - gzip_ratio: reference only (circularity caveat), byte compression ratio

Usage:
  python redundancy_stats.py --bins data_bin_koweb data_bin_code data_bin_mk_t02 \
      --sample-tokens 2000000 --ngram 4
"""
import argparse, gzip, json, os
import numpy as np


def load_sample(path, n):
    if os.path.isdir(path):
        path = os.path.join(path, "train.bin")
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(n, len(arr))
    return np.asarray(arr[:n], dtype=np.int64), path


def entropy_bits(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stats(x, ngram):
    T = len(x)
    # unigram
    uni = np.bincount(x)
    H1 = entropy_bits(uni)
    ttr = int((uni > 0).sum()) / T
    # bigram conditional entropy = H(x_{t-1},x_t) - H(x_{t-1})
    a, b = x[:-1], x[1:]
    V = int(x.max()) + 1
    big = a.astype(np.int64) * V + b
    _, bc = np.unique(big, return_counts=True)
    Hjoint = entropy_bits(bc)
    Hrate = Hjoint - H1
    # n-gram repetition rate
    if T > ngram:
        win = np.lib.stride_tricks.sliding_window_view(x, ngram)
        total = win.shape[0]
        distinct = np.unique(win, axis=0).shape[0]
        rep = 1.0 - distinct / total
    else:
        rep = float("nan")
    # gzip, for reference
    gz = len(gzip.compress(x.astype(np.uint16).tobytes(), 6)) / (x.nbytes)
    return {"tokens": T, "vocab_seen": V, "H1_bits": round(H1, 4),
            "Hrate_bits": round(Hrate, 4), f"rep_{ngram}gram": round(float(rep), 4),
            "ttr": round(ttr, 5), "gzip_ratio": round(gz, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", nargs="+", required=True, help=".bin file or directory containing train.bin")
    ap.add_argument("--sample-tokens", type=int, default=2_000_000)
    ap.add_argument("--ngram", type=int, default=4)
    args = ap.parse_args()

    rows = {}
    for b in args.bins:
        x, resolved = load_sample(b, args.sample_tokens)
        rows[b] = stats(x, args.ngram)
        rows[b]["path"] = resolved
    # Print table (sorted by redundancy, high to low = ascending Hrate)
    keys = ["tokens", "H1_bits", "Hrate_bits", f"rep_{args.ngram}gram", "ttr", "gzip_ratio"]
    name_w = max(len(b) for b in rows)
    print(f"{'corpus':<{name_w}}" + "".join(f"{k:>14}" for k in keys))
    for b in sorted(rows, key=lambda k: rows[k]["Hrate_bits"]):
        print(f"{b:<{name_w}}" + "".join(f"{rows[b][k]:>14}" for k in keys))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
