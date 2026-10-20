"""System-log corpus (loghub) -> token .bin, one of the structured corpora in the paper.

Streams bolu61/loghub_2 (Apache and other system logs) and encodes it with tok_kej_64k, EOS per
line (raw-text gzip 0.063, bigram MI/H1 0.717).
Budget matched to the 7.4M-token corpora (train 7.38M / val 0.5M), same spec as build_code_bin.py
(no dedup).

Usage:
  python build_loghub_bin.py --tokenizer tok_kej_64k.model --out-dir data_bin_loghub \
      --target-tokens 7383561 --val-tokens 500000
"""
import argparse, gzip, json, os
import numpy as np
import sentencepiece as spm
from datasets import load_dataset

DATASET, FIELD = "bolu61/loghub_2", "text"


def entropy_bits(counts):
    p = counts / counts.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-tokens", type=int, default=7_383_561)
    ap.add_argument("--val-tokens", type=int, default=500_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos = sp.eos_id()
    total = args.target_tokens + args.val_tokens
    ds = load_dataset(DATASET, split="train", streaming=True)
    buf, raw_parts, raw_len = [], [], 0
    for ex in ds:
        t = ex.get(FIELD)
        if not t:
            continue
        buf.extend(sp.encode(t)); buf.append(eos)
        if raw_len < 3_000_000:
            b = t.encode("utf-8"); raw_parts.append(b); raw_len += len(b)
        if len(buf) >= total:
            break
    if len(buf) < total:
        raise RuntimeError(f"only {len(buf)} tokens < {total} needed")
    arr = np.array(buf[:total], dtype=np.uint16)
    val, train = arr[:args.val_tokens], arr[args.val_tokens:]
    train.tofile(os.path.join(args.out_dir, "train.bin"))
    val.tofile(os.path.join(args.out_dir, "val.bin"))

    # Redundancy metrics of the built corpus (re-measured, for documentation)
    raw = b"".join(raw_parts)[:3_000_000]
    gz = round(len(gzip.compress(raw, 6)) / len(raw), 4)
    x = arr.astype(np.int64)
    uni = np.bincount(x); H1 = entropy_bits(uni)
    a, b2 = x[:-1], x[1:]; V = int(x.max()) + 1
    _, bc = np.unique(a * V + b2, return_counts=True)
    Hrate = entropy_bits(bc) - H1; MI = H1 - Hrate
    meta = {"source": DATASET, "field": FIELD, "tokenizer": os.path.basename(args.tokenizer),
            "train_tokens": int(len(train)), "val_tokens": int(len(val)),
            "raw_gzip": gz, "H1": round(H1, 3), "MI/H1": round(MI / H1, 3)}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
