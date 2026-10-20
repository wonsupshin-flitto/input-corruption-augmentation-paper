"""HF text dataset -> token .bin (Korean/Japanese web, chess, and the two JSON corpora via --json-dump).

A generalization of build_code_bin. Streams and encodes any HF text dataset via
--dataset/--config/--field. Budget matched (train 7.38M/val 0.5M), tok_kej_64k, EOS per document.

Usage:
  python build_hf_corpus.py --tokenizer tok_kej_64k.model --out-dir data_bin_wiki \
      --dataset wikimedia/wikipedia --config 20231101.en --field text \
      --target-tokens 7383561 --val-tokens 500000
"""
import argparse, json, os
import numpy as np
import sentencepiece as spm
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None, help="e.g. the-stack-smol data/{lang}")
    ap.add_argument("--field", default="text")
    ap.add_argument("--json-dump", action="store_true", help="json.dumps each full example -> highly templated (high-MI) text")
    ap.add_argument("--target-tokens", type=int, default=7_383_561)
    ap.add_argument("--val-tokens", type=int, default=500_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos = sp.eos_id()
    total = args.target_tokens + args.val_tokens
    kw = {"split": "train", "streaming": True}
    if args.config:
        kw["name"] = args.config
    if args.data_dir:
        kw["data_dir"] = args.data_dir
    ds = load_dataset(args.dataset, **kw)
    buf = []
    for ex in ds:
        t = json.dumps(ex, ensure_ascii=False, default=str) if args.json_dump else ex.get(args.field)
        if not t:
            continue
        buf.extend(sp.encode(t)); buf.append(eos)
        if len(buf) >= total:
            break
    if len(buf) < total:
        raise RuntimeError(f"only {len(buf)} tokens < {total} needed")
    arr = np.array(buf[:total], dtype=np.uint16)
    val, train = arr[:args.val_tokens], arr[args.val_tokens:]
    train.tofile(os.path.join(args.out_dir, "train.bin"))
    val.tofile(os.path.join(args.out_dir, "val.bin"))
    meta = {"source": args.dataset, "config": args.config, "field": args.field,
            "train_tokens": int(len(train)), "val_tokens": int(len(val))}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
