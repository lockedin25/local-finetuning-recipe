"""Runs ON THE GPU VM. LoRA fine-tune of LiquidAI/LFM3-1.7B (Hugging Face weights) with the SAME masking as the Mac
pipeline (encode.py): the prompt is masked, only "<thought></think>reply" (think) or "reply" (nothink) is trained.

  python gpu_train.py --variant think   --out runs/think   [--steps 20 --limit 400]   # smoke test
  python gpu_train.py --variant nothink --out runs/nothink
Batch 16 with length grouping, LR 4e-5 cosine (warmup 50), LoRA r16 scale 20 on all linear layers, 1 epoch.
At the end: validation loss, test BPC (reply only, no-think format, 500 fixed test examples), merged bf16 model in
<out>/merged. Prints counts/metrics only, never chat text.
"""

import os
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from encode import encode_example

H = Path(os.environ.get("LOCAL_FINETUNING_GPU_HOME", str(Path(__file__).resolve().parent)))
BASE = "LiquidAI/LFM3-1.7B"


THOUGHT_PREFIX = "\n\nWhat I'm thinking right now: "


def load_rows(variant, split):
    if variant == "thinker":
        # thinker: same examples as "think", but ONLY the thought is trained (the reply is removed)
        rows = []
        for l in (H / "data" / "think" / f"{split}.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            r["messages"][-1]["content"] = ""
            rows.append(r)
        return rows
    if variant != "texter":
        return [json.loads(l) for l in (H / "data" / variant / f"{split}.jsonl").open(encoding="utf-8")]
    # texter: same examples as "think", but the thought is GIVEN in the system prompt; only the reply is trained
    rows = []
    for l in (H / "data" / "think" / f"{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        thought = r["messages"][-1].pop("reasoning_content", "") or ""
        if thought:
            r["messages"][0]["content"] += THOUGHT_PREFIX + thought
        rows.append(r)
    return rows


def encode_rows(tok, rows, kw, max_seq):
    out = []
    for r in rows:
        ids, off = encode_example(tok, r["messages"], kw)
        ids = ids[:max_seq]
        labels = [-100] * off + ids[off:]
        out.append((ids, labels[:len(ids)]))
    return out


def batches(data, bs, seed, shuffle=True):
    idx = sorted(range(len(data)), key=lambda i: len(data[i][0]))          # length grouping
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        random.Random(seed).shuffle(chunks)
    return chunks


def collate(data, chunk, pad_id, device):
    L = max(len(data[i][0]) for i in chunk)
    ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
    lab = torch.full((len(chunk), L), -100, dtype=torch.long)
    att = torch.zeros((len(chunk), L), dtype=torch.long)
    for j, i in enumerate(chunk):
        x, y = data[i]
        ids[j, :len(x)] = torch.tensor(x)
        lab[j, :len(y)] = torch.tensor(y)
        att[j, :len(x)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def loss_fn(model, ids, lab, att, reduction="mean"):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    keep = tgt != -100                     # only trained positions go to float32 (saves ~20 GB at batch 16)
    return F.cross_entropy(logits[keep].float(), tgt[keep], reduction=reduction)


@torch.no_grad()
def evaluate(model, data, bs, pad_id, device):
    model.eval()
    tot, n = 0.0, 0
    for chunk in batches(data, bs, 0, shuffle=False):
        ids, lab, att = collate(data, chunk, pad_id, device)
        tot += loss_fn(model, ids, lab, att, "sum").item()
        n += (lab[:, 1:] != -100).sum().item()
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def test_bpc(model, tok, rows, device):
    """Bits per character of the real reply text, no-think format, same 500-example sample for every model."""
    model.eval()
    sample = random.Random(7).sample(rows, min(500, len(rows)))
    nll, chars = 0.0, 0
    for r in sample:
        msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in r["messages"]]
        ids, off = encode_example(tok, msgs, {"enable_thinking": False})
        reply = msgs[-1]["content"]
        x = torch.tensor([ids], device=device)
        logits = model(input_ids=x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None])[:, 0]
        k = len(tok(reply, add_special_tokens=False)["input_ids"])
        nll -= lp[off - 1: off - 1 + k].sum().item()
        chars += len(reply)
    model.train()
    return nll / chars / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["think", "nothink", "texter", "thinker"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="stop after N steps (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="use N random train examples (smoke test)")
    ap.add_argument("--max-seq", type=int, default=2560)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-only", action="store_true", help="only report the untrained base model's test BPC")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    kw = {"enable_thinking": a.variant in ("think", "thinker")}

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev)
    if a.base_only:
        print(f"[base] test BPC {test_bpc(model, tok, load_rows(a.variant, 'test'), dev):.3f}", flush=True)
        return
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank * a.scale, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_rows = load_rows(a.variant, "train")
    if a.limit:
        train_rows = random.Random(a.seed).sample(train_rows, a.limit)
    train = encode_rows(tok, train_rows, kw, a.max_seq)
    valid = encode_rows(tok, load_rows(a.variant, "valid"), kw, a.max_seq)
    # masking check (counts only): think targets must contain </think>, nothink targets must not contain <think>
    sample = random.Random(1).sample(train, min(200, len(train)))
    tgt_txt = [tok.decode([t for t in y if t != -100]) for _, y in sample]
    ok = sum(("</think>" in t) == (a.variant in ("think", "thinker")) and "<think>" not in t for t in tgt_txt)
    print(f"[{a.variant}] train {len(train):,} valid {len(valid):,} | trainable params {trainable / 1e6:.1f}M | "
          f"masking check {ok}/{len(sample)} | trained tokens {sum(sum(1 for t in y if t != -100) for _, y in train):,}", flush=True)

    chunks = [c for e in range(a.epochs) for c in batches(train, a.bs, a.seed + e)]
    total = min(len(chunks), a.steps) if a.steps else len(chunks)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, min(50, max(1, total // 10)), total)
    hist = {"val": [], "train": []}
    v0 = evaluate(model, valid, a.bs, pad_id, dev)
    hist["val"].append((0, v0))
    print(f"step 0 val {v0:.3f}", flush=True)
    t0, toks, run = time.time(), 0, []
    for step, chunk in enumerate(chunks[:total], 1):
        ids, lab, att = collate(train, chunk, pad_id, dev)
        loss = loss_fn(model, ids, lab, att)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run.append(loss.item())
        toks += att.sum().item()
        if step % 20 == 0 or step == total:
            print(f"step {step}/{total} loss {sum(run) / len(run):.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{toks / (time.time() - t0):,.0f} tok/s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
            hist["train"].append((step, sum(run) / len(run)))
            run = []
        if step % a.eval_every == 0 or step == total:
            v = evaluate(model, valid, a.bs, pad_id, dev)
            hist["val"].append((step, v))
            print(f"step {step} val {v:.3f}", flush=True)
    model.save_pretrained(out / "adapter")
    bpc = test_bpc(model, tok, load_rows(a.variant, "test"), dev) if a.variant != "thinker" else float("nan")
    print(f"[{a.variant}] test BPC {bpc:.3f} | {time.time() - t0:.0f}s", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    snap = Path(snapshot_download(BASE))
    for f in snap.glob("*.py"):
        shutil.copy(f, out / "merged" / f.name)
    json.dump({"variant": a.variant, "bs": a.bs, "lr": a.lr, "rank": a.rank, "scale": a.scale, "steps": total,
               "val": hist["val"], "train": hist["train"], "test_bpc": bpc}, open(out / "metrics.json", "w"), indent=1)
    print(f"[{a.variant}] DONE merged model -> {out / 'merged'}", flush=True)


if __name__ == "__main__":
    main()
