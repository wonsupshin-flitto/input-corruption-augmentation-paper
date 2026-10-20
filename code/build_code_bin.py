"""Code corpus (The Stack) -> token .bin, one of the structured corpora in the paper.

Streams code text from HF datasets and encodes it with tok_kej_64k, EOS per file.
Budget matched to the 7.4M-token corpora (train 7.38M / val 0.5M).

Usage:
  python build_code_bin.py --tokenizer tok_kej_64k.model --out-dir data_bin_code \
      --target-tokens 7383561 --val-tokens 500000
"""
import argparse, json, os
import numpy as np
import sentencepiece as spm
from datasets import load_dataset

# Streamable ungated candidates (tried in order)
CANDIDATES = [
    ("codeparrot/codeparrot-clean-valid", "content"),
    ("code_search_net", "whole_func_string"),
    ("bigcode/the-stack-smol", "content"),
]


def stream_texts(need):
    for name, field in CANDIDATES:
        try:
            cfg = "python" if name == "code_search_net" else None
            ds = load_dataset(name, cfg, split="train", streaming=True) if cfg else \
                 load_dataset(name, split="train", streaming=True)
            got = 0
            for ex in ds:
                t = ex.get(field)
                if t:
                    yield t; got += 1
                if got >= need:
                    return
            if got:
                return
        except Exception as e:
            print(f"skip {name}: {e}")
    raise RuntimeError("no code dataset available")


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
    buf = []
    for text in stream_texts(need=200000):
        buf.extend(sp.encode(text)); buf.append(eos)
        if len(buf) >= total:
            break
    if len(buf) < total:
        raise RuntimeError(f"only {len(buf)} tokens < {total} needed")
    arr = np.array(buf[:total], dtype=np.uint16)
    val, train = arr[:args.val_tokens], arr[args.val_tokens:]
    train.tofile(os.path.join(args.out_dir, "train.bin"))
    val.tofile(os.path.join(args.out_dir, "val.bin"))
    meta = {"source": "code", "tokenizer": os.path.basename(args.tokenizer),
            "train_tokens": int(len(train)), "val_tokens": int(len(val))}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
