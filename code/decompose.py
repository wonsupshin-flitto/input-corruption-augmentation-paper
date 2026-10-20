"""Known-entropy loss decomposition (paper Section 5 and the decomposition table in the appendix).

On the controlled Markov source the true entropy rate H is known in closed form
(printed by markov_source.py into meta.json). This lets validation loss be split
into interpretable terms:

    L_final = H + APPROX + EST + MEM
      APPROX = L_abund - H        (approximation: abundant-data loss above the entropy rate)
      EST    = L_min - L_abund    (estimation: minimum loss of the data-constrained run above it)
      MEM    = L_final - L_min    (memorization term: the loss rise after the minimum)

The three terms are operational: each is a difference between measured losses.

Inputs are the per-step validation curves (metrics.jsonl, one {"iter","val_loss"} per line)
of (a) the data-constrained run and (b) an abundant-data run of the same capacity on the
same source, plus the source's meta.json for H.

Usage:
  python decompose.py \
      --constrained runs/mk_t05_none_s1337/metrics.jsonl \
      --abundant   runs/mkab_t05_none_s1337/metrics.jsonl \
      --meta       data_bin_mk_t05/meta.json
"""
import argparse, json, math


def val_losses(metrics_path):
    xs = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "val_loss" in rec:
                xs.append(float(rec["val_loss"]))
    if not xs:
        raise RuntimeError(f"no val_loss in {metrics_path}")
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constrained", required=True, help="metrics.jsonl of the data-constrained run")
    ap.add_argument("--abundant", required=True, help="metrics.jsonl of the abundant-data run (same capacity, same source)")
    ap.add_argument("--meta", required=True, help="source meta.json (holds entropy_rate_bits)")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    # markov_source.py records the entropy rate in bits; validation loss is cross-entropy in nats.
    H = float(meta["entropy_rate_bits"]) * math.log(2.0)

    c = val_losses(args.constrained)
    L_min = min(c)          # constrained run's best (minimum) validation loss
    L_final = c[-1]           # constrained run's loss at the end of training
    L_abund = min(val_losses(args.abundant))  # abundant run converges without a memorization rise

    approx = L_abund - H
    est = L_min - L_abund
    mem = L_final - L_min

    print(f"H       = {H:.4f} nats  (entropy rate, lower bound on loss)")
    print(f"L_abund = {L_abund:.4f}   L_min = {L_min:.4f}   L_final = {L_final:.4f}")
    print(f"APPROX  = {approx:+.4f}")
    print(f"EST     = {est:+.4f}")
    print(f"MEM     = {mem:+.4f}")
    print(f"check   L_final = H + APPROX + EST + MEM = {H + approx + est + mem:.4f}  (should equal L_final)")


if __name__ == "__main__":
    main()
