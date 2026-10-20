"""Redundancy survey: raw-text gzip distribution across many corpora.

Measures how redundant real text is across many public corpora (paper Figure 5 and appendix),
to see at what redundancy the one harmed corpus lies relative to them.

Streams each entry of the manifest (JSON list), computes raw_gzip + MI/H1, appends results to JSONL.
Skips slow entries via a per-item timeout (SIGALRM). Failures are logged and skipped, execution continues.

Usage:
  python redundancy_survey.py --tokenizer tok_kej_64k.model --manifest survey_manifest.json \
      --out survey_results.jsonl --raw-bytes 3000000 --sample-tokens 1000000 --timeout 200
"""
import argparse, gzip, json, os, signal
import numpy as np
import sentencepiece as spm
from datasets import load_dataset


class Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise Timeout()


def entropy_bits(counts):
    p = counts / counts.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def scan_one(sp, eos, entry, raw_need, tok_need):
    kw = {"split": entry.get("split", "train"), "streaming": True}
    if entry.get("config"):
        kw["name"] = entry["config"]
    kw.update(entry.get("load_kwargs", {}))
    ds = load_dataset(entry["dataset"], **kw)
    field = entry.get("field", "text")
    jd = entry.get("json_dump", False)
    raw_parts, raw_len, tok = [], 0, []
    for ex in ds:
        t = json.dumps(ex, ensure_ascii=False, default=str) if jd else ex.get(field)
        if not t:
            continue
        if isinstance(t, list):
            t = " ".join(map(str, t))
        if raw_len < raw_need:
            b = t.encode("utf-8"); raw_parts.append(b); raw_len += len(b)
        if len(tok) < tok_need:
            tok.extend(sp.encode(t)); tok.append(eos)
        if raw_len >= raw_need and len(tok) >= tok_need:
            break
    if raw_len < 100000 or len(tok) < 50000:
        return {"error": f"insufficient raw={raw_len} tok={len(tok)}"}
    raw = b"".join(raw_parts)[:raw_need]
    gz = round(len(gzip.compress(raw, 6)) / len(raw), 4)
    x = np.array(tok[:tok_need], dtype=np.int64)
    uni = np.bincount(x); H1 = entropy_bits(uni)
    a, b = x[:-1], x[1:]; V = int(x.max()) + 1
    _, bc = np.unique(a * V + b, return_counts=True)
    Hrate = entropy_bits(bc) - H1; MI = H1 - Hrate
    return {"raw_gzip": gz, "raw_bytes": len(raw), "tokens": len(x),
            "H1": round(H1, 3), "MI/H1": round(MI / H1, 3),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-bytes", type=int, default=3_000_000)
    ap.add_argument("--sample-tokens", type=int, default=1_000_000)
    ap.add_argument("--timeout", type=int, default=200)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos = sp.eos_id()
    manifest = json.load(open(args.manifest))
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r.get("dataset"), r.get("config"), r.get("field")))
            except Exception:
                pass
    signal.signal(signal.SIGALRM, _alarm)
    fout = open(args.out, "a")
    for i, entry in enumerate(manifest):
        key = (entry["dataset"], entry.get("config"), entry.get("field", "text"))
        if key in done:
            print(f"[{i+1}/{len(manifest)}] SKIP done {key}"); continue
        row = {"dataset": entry["dataset"], "config": entry.get("config"),
               "field": entry.get("field", "text"), "category": entry.get("category")}
        try:
            signal.alarm(args.timeout)
            row.update(scan_one(sp, eos, entry, args.raw_bytes, args.sample_tokens))
            signal.alarm(0)
        except Timeout:
            row["error"] = f"timeout>{args.timeout}s"
        except Exception as e:
            row["error"] = repr(e)[:120]
        finally:
            signal.alarm(0)
        fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
        tag = row.get("error") or f"gzip={row.get('raw_gzip')} MI/H1={row.get('MI/H1')}"
        print(f"[{i+1}/{len(manifest)}] {entry.get('category')} {entry['dataset']} {entry.get('config') or ''} -> {tag}", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
