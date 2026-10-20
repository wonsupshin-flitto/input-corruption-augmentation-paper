"""Shrinkage weight of random-token replacement on a corpus (paper appendix C.3).

No training. Counts unigram and bigram frequencies in train.bin.

Replacing each input token with a uniform draw over the model vocabulary at rate p gives, for an
observed token c, the posterior that it is clean,
    alpha(c) = (1-p) pi(c) / [(1-p) pi(c) + p q(c)],   q(c) = 1 / noise_vocab,
and the optimal prediction alpha P(.|c) + (1-alpha) pi. Reported per corpus:
  - detect : probability that a noise draw is a token absent from the corpus
  - w      : shrinkage weight on clean contexts, sum_c pi(c) (1 - alpha(c))
  - MI     : bigram mutual information (nats)
  - bias   : w * MI

Usage:
  python shrinkage_weight.py --bin data_bin_koweb/train.bin --vocab 64000 --noise-vocab 64000
  python shrinkage_weight.py --bin data_bin_mk_t05/train.bin --vocab 1024 --noise-vocab 1025
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="train.bin (uint16 tokens)")
    ap.add_argument("--vocab", type=int, required=True, help="number of token ids that can occur in the corpus")
    ap.add_argument("--noise-vocab", type=int, required=True, help="size of the uniform replacement vocabulary")
    ap.add_argument("--p", type=float, default=0.15, help="replacement rate")
    args = ap.parse_args()

    a = np.fromfile(args.bin, dtype=np.uint16).astype(np.int64)
    V, p = args.vocab, args.p
    cnt = np.bincount(a, minlength=V).astype(float)
    pi = cnt / cnt.sum()
    alpha = (1 - p) * pi / ((1 - p) * pi + p / args.noise_vocab)
    types = int((pi > 0).sum())
    u, c = np.unique(a[:-1] * V + a[1:], return_counts=True)
    pab = c / c.sum()
    mi = float((pab * np.log(pab / (pi[u // V] * pi[u % V]))).sum())
    w = float((pi * (1 - alpha)).sum())
    print(json.dumps(dict(tokens=int(a.size), types=types, detect=1 - types / args.noise_vocab,
                          w=w, MI=mi, bias=w * mi), indent=2))


if __name__ == "__main__":
    main()
