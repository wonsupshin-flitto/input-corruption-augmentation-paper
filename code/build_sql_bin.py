"""SQL corpus (b-mc2/sql-create-context) -> token .bin, the two SQL forms in the paper.

Each example holds a CREATE TABLE context and a SELECT query. The context-only form (SQL-ctx)
and the combined form (SQL-comb) are built from the raw text, without a JSON wrapper.
The paper labels both forms program-generated, since the CREATE TABLE contexts were generated
from Spider and WikiSQL queries. EOS at the document level.

Usage:
  python build_sql_bin.py --tokenizer tok_kej_64k.model --out-dir data_bin_sql \
      --target-tokens 4960000 --val-tokens 300000
"""
import argparse, json, os
import numpy as np
import sentencepiece as spm
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", default="combined", choices=["combined", "context", "answer"],
                    help="combined=DDL+query(MI0.62), context=CREATE TABLE only(MI0.684), answer=SELECT only(MI0.565)")
    ap.add_argument("--target-tokens", type=int, default=4_960_000)
    ap.add_argument("--val-tokens", type=int, default=300_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos = sp.eos_id()
    total = args.target_tokens + args.val_tokens
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    buf = []
    for ex in ds:
        c = ex.get("context") or ""
        a = ex.get("answer") or ""
        if args.mode == "context":
            t = c.strip()
        elif args.mode == "answer":
            t = a.strip()
        else:
            t = (c + "\n" + a).strip()  # raw SQL concatenation (schema + query)
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
    meta = {"source": "b-mc2/sql-create-context", "mode": args.mode,
            "train_tokens": int(len(train)), "val_tokens": int(len(val))}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
