"""Synthetic Markov source: a toy corpus whose redundancy is controlled by a single knob.

The controlled source of the paper: a fixed, unique corpus whose entropy rate is set by
T (temperature) and known in closed form, which train.py repeats over many epochs.

Mechanism: order-1 Markov chain. Transition matrix P[i] = softmax(logits_i / T).
  - T -> 0: peaked (near-deterministic) -> low entropy -> high redundancy.
  - T -> inf: uniform -> high entropy (log V) -> low redundancy.
P is fixed by a fixed seed, so it is reproducible. train/val are separate samples (same
chain, held-out).

Output: out-dir/train.bin, val.bin (uint16), meta.json.
train.py compatibility: at training time, use --vocab-size (V+1) so aug mask-id=V has an
unused row reserved. E.g. V=1024 -> --vocab-size 1025.

Usage:
  python markov_source.py --vocab-size 1024 --temperature 0.7 \
      --n-tokens 7383561 --val-tokens 500000 --seed 0 --out-dir data_bin_mk_t07
"""
import argparse, json, os
import numpy as np


def build_transition(vocab, temperature, rng):
    """Each row = softmax(N(0,1) logits / T). Lower T = more peaked = higher redundancy."""
    logits = rng.standard_normal((vocab, vocab)).astype(np.float64)
    z = logits / max(1e-6, temperature)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p


def stationary(P, iters=2000, tol=1e-12):
    """Approximate the stationary distribution pi via power iteration."""
    V = P.shape[0]
    pi = np.full(V, 1.0 / V)
    for _ in range(iters):
        nxt = pi @ P
        nxt /= nxt.sum()
        if np.abs(nxt - pi).sum() < tol:
            pi = nxt
            break
        pi = nxt
    return pi


def entropy_rate_bits(P, pi):
    """order-1 Markov entropy rate H = sum_i pi_i * H(P[i]) (bits/token)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rowH = -np.where(P > 0, P * np.log2(P), 0.0).sum(axis=1)
    return float((pi * rowH).sum())


def sample_chain(P, n, rng, start=None):
    """Sample a length-n sequence from P (vectorized batch CDF sampling)."""
    V = P.shape[0]
    cdf = np.cumsum(P, axis=1)
    out = np.empty(n, dtype=np.uint16)
    s = int(rng.integers(V)) if start is None else start
    # Draw uniforms in chunks and search each step (stays memory-stable even for large n)
    CH = 1_000_000
    idx = 0
    while idx < n:
        m = min(CH, n - idx)
        u = rng.random(m)
        for j in range(m):
            r = cdf[s]
            nxt = int(np.searchsorted(r, u[j]))
            if nxt >= V:
                nxt = V - 1
            out[idx + j] = nxt
            s = nxt
        idx += m
    return out, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--temperature", type=float, required=True, help="repetition knob (lower = more repetitive)")
    ap.add_argument("--n-tokens", type=int, required=True, help="number of unique train tokens")
    ap.add_argument("--val-tokens", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    P = build_transition(args.vocab_size, args.temperature, rng)
    pi = stationary(P)
    Hrate = entropy_rate_bits(P, pi)          # bits/token (theoretical redundancy metric)
    H_uniform = np.log2(args.vocab_size)
    redundancy = 1.0 - Hrate / H_uniform      # 0 (no redundancy) to 1 (fully redundant)

    train, last = sample_chain(P, args.n_tokens, rng)
    val, _ = sample_chain(P, args.val_tokens, rng, start=last)
    train.tofile(os.path.join(args.out_dir, "train.bin"))
    val.tofile(os.path.join(args.out_dir, "val.bin"))

    meta = {
        "source": "markov_order1", "vocab_size": args.vocab_size,
        "temperature": args.temperature, "seed": args.seed,
        "train_tokens": int(args.n_tokens), "val_tokens": int(args.val_tokens),
        "entropy_rate_bits": round(Hrate, 4), "entropy_uniform_bits": round(H_uniform, 4),
        "redundancy_index": round(redundancy, 4),
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
