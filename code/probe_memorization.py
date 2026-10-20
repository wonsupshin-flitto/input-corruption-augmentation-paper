"""Memorization and shrinkage probes on the Markov source (paper appendix C.1).

Inference only. For one trained checkpoint and its Markov source this measures, on clean input:
  - loss        : mean cross-entropy (nats) on training and on validation windows
  - top1        : share of positions (context >= 1024 tokens) where the argmax is the next token
  - extract50   : share of 50-token spans after >= 1024 tokens of context that greedy decoding
                  reproduces exactly (all 50 argmax predictions correct), i.e. the
                  k-extractable rate of Carlini et al. (2023)
  - mia_auc     : AUC of separating training windows from validation windows by window loss
  - KL(Q||P_true), KL(Q||pi): the model's next-token distribution, averaged over validation
                  positions by the preceding token, against the true transition and the
                  stationary marginal, weighted by pi

Windows: 20 batches x 8 windows x 2048 tokens per split, drawn with a fixed seed.

Usage:
  python probe_memorization.py --ckpt runs/mk_t05_replace/ckpt.pt \
      --data-dir data_bin_mk_t05 --temperature 0.5
"""
import argparse, json
import numpy as np
import torch
from model import GPT, GPTConfig
from markov_source import build_transition, stationary

BLOCK, BATCH, NB = 2048, 8, 20
P0, SEG, NSEG = 1024, 50, 20          # minimum context, span length, spans per window
KL_SEQS = 90                          # validation sequences used to estimate Q(.|s)


def load_model(path, dev):
    blob = torch.load(path, map_location="cpu")
    m = GPT(GPTConfig(**blob["cfg"])); m.load_state_dict(blob["model"]); m.eval()
    return m.to(dev)


@torch.no_grad()
def window_metrics(m, data, dev, seed=0):
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    g = torch.Generator().manual_seed(seed)
    win_loss, top1_hit, top1_n, ext_hit, ext_n = [], 0, 0, 0, 0
    for _ in range(NB):
        ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=g)
        x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix]).to(dev)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix]).to(dev)
        with ctx:
            logits, _ = m(x)
        logits = logits.float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        hit = logits.argmax(-1).eq(y)
        win_loss += nll.mean(1).tolist()
        top1_hit += hit[:, P0:].sum().item(); top1_n += hit[:, P0:].numel()
        segs = hit[:, P0:P0 + SEG * NSEG].reshape(BATCH, NSEG, SEG).all(-1)
        ext_hit += segs.sum().item(); ext_n += segs.numel()
    return dict(loss=float(np.mean(win_loss)), top1=top1_hit / top1_n,
                extract50=ext_hit / ext_n), win_loss


def auc(member, nonmember):
    a, b = np.asarray(member)[:, None], np.asarray(nonmember)[None, :]
    return float((a < b).mean() + 0.5 * (a == b).mean())


@torch.no_grad()
def model_Q(m, val, V, dev):
    n = (len(val) // BLOCK) * BLOCK
    seqs = val[:n].astype(np.int64).reshape(-1, BLOCK)[:KL_SEQS]
    accum = np.zeros((V, V)); cnt = np.zeros(V)
    for s in seqs:
        logits, _ = m(torch.from_numpy(s[None]).to(dev))
        q = torch.softmax(logits[0, :, :V].float(), dim=-1).cpu().numpy()
        np.add.at(accum, s[:-1], q[:-1]); np.add.at(cnt, s[:-1], 1.0)
    nz = cnt > 0
    Q = np.zeros((V, V)); Q[nz] = accum[nz] / cnt[nz][:, None]
    return Q, cnt


def rowkl(A, B):
    return np.where(A > 0, A * np.log((A + 1e-12) / (B + 1e-12)), 0.0).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="Markov source dir with train.bin, val.bin")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--states", type=int, default=1024, help="Markov states (model vocab is states+1)")
    ap.add_argument("--source-seed", type=int, default=0, help="seed passed to markov_source.py")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = np.memmap(f"{args.data_dir}/train.bin", dtype=np.uint16, mode="r")
    va = np.memmap(f"{args.data_dir}/val.bin", dtype=np.uint16, mode="r")
    m = load_model(args.ckpt, dev)
    rt, lt = window_metrics(m, tr, dev)
    rv, lv = window_metrics(m, va, dev)

    V = args.states
    P = build_transition(V, args.temperature, np.random.default_rng(args.source_seed))
    pi = stationary(P)
    Q, cnt = model_Q(m, va, V, dev)
    w = pi * (cnt > 0); w = w / w.sum()
    res = dict(train=rt, val=rv, mia_auc=auc(lt, lv),
               kl_true=float((w * rowkl(Q, P)).sum()),
               kl_marginal=float((w * rowkl(Q, np.tile(pi, (V, 1)))).sum()))
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
