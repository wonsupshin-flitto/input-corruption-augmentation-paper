#!/usr/bin/env python3
"""YODAS (open YouTube speech corpus) transcripts -> token .bin build (for reproducibility).

YODAS (espnet/yodas) is HF-native and open (no gate), and for each language config,
`data/{config}/text/{shard}.txt` can be downloaded directly without audio.
Text line format: `<utt_id> <transcript>` (utt_id is stripped; only the transcript is used).

Procedure:
  1. Download text shards in order until the budget is reached (hf_hub_download cache).
  2. Strip utt_id, normalize whitespace, apply a minimum-length filter, exact dedup, shuffle (seed).
  3. Encode with sentencepiece, EOS between documents, stop at a document boundary once the
     token budget is reached.
  4. val = the first --val-tokens tokens, the rest is train -> train.bin/val.bin/meta.json.

Usage:
  HF_TOKEN=... python build_yodas_transcript.py \
    --config ko000 --tokenizer tok_kej_64k.model --out-dir data_bin_yodasko \
    --budget-tokens 7883561 --val-tokens 500000 --seed 42
"""
import argparse, json, os, random
import numpy as np
import sentencepiece as spm
from huggingface_hub import hf_hub_download


def iter_shard_lines(repo, config, max_shards, token):
    """Download text shards in order starting from 0, yielding transcript lines with utt_id stripped."""
    for i in range(max_shards):
        fname = f"data/{config}/text/{i:08d}.txt"
        try:
            path = hf_hub_download(repo, fname, repo_type="dataset", token=token)
        except Exception:
            break  # no more shards
        with open(path, errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if " " not in line:
                    continue
                _utt, txt = line.split(" ", 1)
                txt = txt.strip()
                if len(txt) >= 2:
                    yield txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="espnet/yodas")
    ap.add_argument("--config", required=True, help="e.g. ko000, ja000")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget-tokens", type=int, default=7_883_561,
                    help="total token budget (matches the 7.4M-token corpora); val+train combined")
    ap.add_argument("--val-tokens", type=int, default=500_000)
    ap.add_argument("--max-shards", type=int, default=120, help="download cap (early stop when budget reached)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-out", default="", help="also save the cleaned raw text (optional)")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Collect lines (dedup): gather up to a generous multiple of the budget, then shuffle and truncate.
    #    Since we don't know the average tokens per line, target ~3x the budget in characters to avoid over-collecting.
    seen, lines = set(), []
    for txt in iter_shard_lines(args.repo, args.config, args.max_shards, token):
        if txt in seen:
            continue
        seen.add(txt); lines.append(txt)
        # Assuming at least 1 token per line, 4x the budget in lines is enough
        if len(lines) >= args.budget_tokens * 4:
            break
    random.seed(args.seed)
    random.shuffle(lines)
    print(f"collected {len(lines)} dedup lines from {args.config}")

    # 2) Tokenize until the budget is reached (stop at a document boundary)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos = sp.eos_id()

    ftrain = open(os.path.join(args.out_dir, "train.bin"), "wb")
    fval = open(os.path.join(args.out_dir, "val.bin"), "wb")
    n_train = 0; n_val = 0; n_lines = 0; total = 0
    buf = []

    def flush(force=False):
        nonlocal n_train, n_val
        if not buf or (len(buf) < 1_000_000 and not force):
            return
        arr = np.array(buf, dtype=np.uint16); buf.clear()
        if n_val < args.val_tokens:
            take = min(args.val_tokens - n_val, len(arr))
            arr[:take].tofile(fval); n_val += take
            arr[take:].tofile(ftrain); n_train += len(arr) - take
        else:
            arr.tofile(ftrain); n_train += len(arr)

    raw_kept = [] if args.raw_out else None
    CHUNK = 50000
    for i in range(0, len(lines), CHUNK):
        batch = lines[i:i + CHUNK]
        for j, enc in enumerate(sp.encode(batch)):
            buf.extend(enc); buf.append(eos)
            total += len(enc) + 1; n_lines += 1
            if raw_kept is not None:
                raw_kept.append(batch[j])
            if total >= args.budget_tokens:
                break
        flush()
        if total >= args.budget_tokens:
            break
    flush(force=True)
    ftrain.close(); fval.close()

    if raw_kept is not None:
        with open(args.raw_out, "w") as f:
            f.write("\n".join(raw_kept) + "\n")

    meta = {"source": f"{args.repo}:{args.config}", "tokenizer": os.path.basename(args.tokenizer),
            "vocab_size": sp.vocab_size(), "eos_id": eos,
            "total_tokens": n_train + n_val, "train_tokens": n_train,
            "val_tokens": n_val, "lines": n_lines, "seed": args.seed}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
