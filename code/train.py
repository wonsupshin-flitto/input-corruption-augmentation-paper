"""Custom training loop (single GPU / DDP via torchrun).

- Data: .bin(uint16) memmap, random-offset batches (nanoGPT pattern)
- AdamW + warmup, then a constant or cosine learning rate, grad clip, grad accum, bf16 autocast
- --aug replace: random-token replacement of the input at rate --aug-p, targets unchanged
- writes per-eval validation loss to out-dir/metrics.jsonl and the latest checkpoint to out-dir/ckpt.pt

See README.md for the exact commands used in the paper.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from model import GPT, GPTConfig


def get_batch(data, block, batch, device):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(it, warmup, max_iters, lr, min_lr, schedule="cosine"):
    # warmup common to all schedules
    if it < warmup:
        return lr * (it + 1) / warmup
    if schedule == "constant":
        return lr
    # cosine (default)
    if it >= max_iters:
        return min_lr
    r = (it - warmup) / max(1, (max_iters - warmup))
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="ckpt")
    # model
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument("--n-layer", type=int, default=24)
    ap.add_argument("--n-embd", type=int, default=1024)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--n-kv-head", type=int, default=4)
    ap.add_argument("--ffn-hidden", type=int, default=2816)
    ap.add_argument("--block-size", type=int, default=2048)
    # optim
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337, help="seed for training randomness (init+batch order). data is unaffected.")
    # Training-time augmentation. Default none keeps the baseline behavior unchanged.
    ap.add_argument("--aug", default="none", choices=["none", "replace"],
                    help="none / random-token replacement")
    ap.add_argument("--aug-p", type=float, default=0.15, help="input replacement rate")
    args = ap.parse_args()

    # DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local}"; torch.cuda.set_device(device)
        master = rank == 0
    else:
        world = 1; master = True
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(args.seed + (int(os.environ.get("RANK", 0))))
    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device.startswith("cuda") else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_path = os.path.join(args.data_dir, "val.bin")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r") if os.path.exists(val_path) and os.path.getsize(val_path) > 0 else None

    cfg = GPTConfig(vocab_size=args.vocab_size, n_layer=args.n_layer, n_embd=args.n_embd,
                    n_head=args.n_head, n_kv_head=args.n_kv_head, ffn_hidden=args.ffn_hidden,
                    max_seq_len=args.block_size, grad_checkpoint=args.grad_checkpoint)
    model = GPT(cfg).to(device)
    if master:
        print(f"params {model.num_params()/1e6:.1f}M | device {device} | dtype {dtype} | world {world} | schedule {args.schedule}")
    if args.compile:
        model = torch.compile(model)
    raw = model
    if ddp:
        model = DDP(model, device_ids=[local]); raw = model.module

    optim = torch.optim.AdamW(raw.parameters(), lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)

    @torch.no_grad()
    def est_val():
        if val_data is None:
            return None
        model.eval(); losses = []
        for _ in range(20):
            x, y = get_batch(val_data, args.block_size, args.batch, device)
            with ctx:
                _, l = model(x, y)
            losses.append(l.item())
        model.train()
        return sum(losses) / len(losses)

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    t0 = time.time()
    for it in range(args.max_iters + 1):
        lr = lr_at(it, args.warmup, args.max_iters, args.lr, args.min_lr, args.schedule)
        for g in optim.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 and master:
            vl = est_val()
            print(f"[{it}] val_loss {vl}")
            # append and flush to out-dir so progress can be followed during training
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "metrics.jsonl"), "a") as mf:
                mf.write(json.dumps({"iter": it, "val_loss": vl, "aug": args.aug}) + "\n")
                mf.flush()
            if it > 0:
                blob = {"model": raw.state_dict(), "cfg": cfg.__dict__, "iter": it}
                torch.save(blob, os.path.join(args.out_dir, "ckpt.pt"))  # latest

        for micro in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch, device)
            if args.aug == "replace" and args.aug_p > 0:   # replace some input tokens with random tokens, target stays original
                m = torch.rand(x.shape, device=device) < args.aug_p
                x = torch.where(m, torch.randint(0, args.vocab_size, x.shape, device=device, dtype=x.dtype), x)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            if ddp:
                model.require_backward_grad_sync = (micro == args.grad_accum - 1)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip)
        optim.step(); optim.zero_grad(set_to_none=True)

        if it % args.log_interval == 0 and master:
            dt = time.time() - t0; t0 = time.time()
            print(f"[{it}] train_loss {loss.item()*args.grad_accum:.4f} lr {lr:.2e} {dt/args.log_interval*1000:.0f}ms/it")

    if ddp:
        import torch.distributed as dist; dist.destroy_process_group()


if __name__ == "__main__":
    main()
