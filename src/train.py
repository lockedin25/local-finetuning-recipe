"""Module 5/6: LoRA fine-tune with mlx-lm's trainer, but our own data encoding (src/encode.py).

Usage:
  uv run python src/train.py --iters 20  --limit 400  --adapter-path adapters/smoke   # quick smoke test
  uv run python src/train.py                                                         # full run from config.yaml
The adapter folder works directly with:  mlx_lm.chat --model <model> --adapter-path <folder>
"""

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

sys.path.insert(0, str(Path(__file__).parent))
from encode import encode_example

ROOT = Path(__file__).resolve().parent.parent
T = yaml.safe_load(Path(os.environ.get("LOCAL_FINETUNING_CONFIG", str(ROOT / "config.yaml"))).read_text())["train"]


def reply_only_loss(model, batch, lengths):
    """mlx-lm's default_loss masks `steps <= length`, which also scores one padding token
    past the end of every sequence (~45 nats each with Gemma). We use `steps < length`:
    exactly the reply tokens + end-of-turn marker, nothing beyond."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def load_split(name, tok, limit, seed, data_dir="data/final"):
    rows = [json.loads(l) for l in (ROOT / data_dir / f"{name}.jsonl").open(encoding="utf-8")]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    t0 = time.time()
    data = [encode_example(tok, r["messages"], T["template_kwargs"]) for r in rows]
    print(f"{name}: {len(data):,} examples encoded in {time.time() - t0:.0f}s, "
          f"{sum(len(x[0]) for x in data):,} tokens, {sum(len(x[0]) - x[1] for x in data):,} trained (reply) tokens")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=T["model"], help="base model repo (default: config.yaml train.model)")
    ap.add_argument("--iters", type=int, default=T["iters"])
    ap.add_argument("--limit", type=int, default=0, help="use a random subset of train (smoke tests)")
    ap.add_argument("--adapter-path", default=T["adapter_path"])
    ap.add_argument("--data", default="data/final", help="folder with train/valid jsonl")
    ap.add_argument("--resume", default=None, help="adapters.safetensors to continue training from")
    ap.add_argument("--lr", type=float, default=T["learning_rate"])
    ap.add_argument("--val-examples", type=int, default=max(200, T["batch_size"] * T["val_batches"]))
    ap.add_argument("--steps-per-eval", type=int, default=T["steps_per_eval"])
    ap.add_argument("--save-every", type=int, default=T["save_every"])
    ap.add_argument("--cosine", action="store_true",
                    help="100-step linear warmup, then cosine decay to lr/10 (for long runs)")
    ap.add_argument("--thinking", action="store_true",
                    help="train WITH thinking: enable_thinking=True, loss covers <think>reasoning</think> + reply")
    ap.add_argument("--max-seq", type=int, default=T["max_seq_length"], help="truncate longer sequences (v3 max is 2,412)")
    args = ap.parse_args()
    T["max_seq_length"] = args.max_seq
    if args.thinking:
        T["template_kwargs"] = {"enable_thinking": True}

    mx.random.seed(T["seed"])
    model, tok = load(args.model)
    model.freeze()
    n_layers = len(model.layers) if T["num_layers"] == -1 else T["num_layers"]
    linear_to_lora_layers(model, n_layers, T["lora_parameters"])
    print_trainable_parameters(model)
    if args.resume:
        model.load_weights(str(ROOT / args.resume), strict=False)
        print(f"resumed adapter weights from {args.resume}")

    train_set = load_split("train", tok, args.limit, T["seed"], args.data)
    valid_set = load_split("valid", tok, args.val_examples, T["seed"], args.data)

    out = ROOT / args.adapter_path
    out.mkdir(parents=True, exist_ok=True)
    # Same config keys mlx-lm writes, so mlx_lm.chat / fuse can load the adapter
    (out / "adapter_config.json").write_text(json.dumps({
        "model": args.model, "fine_tune_type": "lora", "num_layers": n_layers,
        "lora_parameters": T["lora_parameters"], "template_kwargs": T["template_kwargs"],
        "learning_rate": args.lr, "cosine": args.cosine, "batch_size": T["batch_size"], "iters": args.iters,
        "data": args.data, "resumed_from": args.resume,
    }, indent=2))

    targs = TrainingArgs(
        batch_size=T["batch_size"], iters=args.iters, val_batches=-1,
        steps_per_report=T["steps_per_report"], steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every, adapter_file=str(out / "adapters.safetensors"),
        max_seq_length=T["max_seq_length"], grad_checkpoint=T["grad_checkpoint"],
        grad_accumulation_steps=T["grad_accumulation_steps"],
    )
    lr = args.lr
    if args.cosine:
        warm = max(1, min(100, args.iters // 10))
        lr = optim.join_schedules(
            [optim.linear_schedule(args.lr / 10, args.lr, warm),
             optim.cosine_decay(args.lr, args.iters - warm, args.lr / 10)], [warm])
    t0 = time.time()
    train(model=model, optimizer=optim.Adam(learning_rate=lr),
          train_dataset=train_set, val_dataset=valid_set, args=targs, loss=reply_only_loss)
    print(f"done in {(time.time() - t0) / 60:.1f} min | peak memory {mx.get_peak_memory() / 1e9:.1f} GB | adapter: {out}")


if __name__ == "__main__":
    main()
