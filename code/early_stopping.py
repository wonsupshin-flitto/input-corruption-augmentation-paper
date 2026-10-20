"""Simulate early stopping on recorded validation curves (paper appendix C.4).

For each run, early stopping halts once validation loss has not improved for `patience`
evaluations and keeps the best checkpoint seen so far. The script compares that loss with the
minimum of the full curve and reports how many runs it misses and by how much.

Usage:
  python early_stopping.py --patience 1 3 5 --runs runs/*/metrics.jsonl
"""
import argparse, json


def curve(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            # skip the evaluation before the first update (iter 0)
            if r.get("val_loss") is not None and r.get("iter", 1) != 0:
                out.append(float(r["val_loss"]))
    return out


def early_stop(losses, patience):
    best, since = float("inf"), 0
    for v in losses:
        if v < best:
            best, since = v, 0
        else:
            since += 1
            if since >= patience:
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="metrics.jsonl files")
    ap.add_argument("--patience", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()
    curves = [curve(p) for p in args.runs]
    curves = [c for c in curves if c]
    for k in args.patience:
        gaps = [early_stop(c, k) - min(c) for c in curves]
        missed = [g for g in gaps if g > args.tol]
        print(f"patience {k}: misses the minimum on {len(missed)} of {len(curves)} runs, "
              f"by up to {max(missed) if missed else 0.0:.3f} nats")


if __name__ == "__main__":
    main()
