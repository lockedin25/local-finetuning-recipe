"""Module 7: evaluate a base model or a trained adapter on the TEST set (sessions from 2026-09-18 on,
never seen in training). Prints statistics only, never chat text.

  1. Bits per character (BPC) of the real replies, on the SAME 500 test examples as bakeoff.py,
     so it compares directly with the untrained baselines (Gemma-4 2.114, LFM3 1.858).
  2. Style: generate one reply for each of 100 test contexts and compare with the real replies:
     length, double-texting, emoji, formal/assistant words, Devanagari, empty or overlong replies.
  3. Memorisation: share of generated replies (>= 25 chars) that appear verbatim in train.

  zsh src/lfm3.sh src/evaluate.py --model LiquidAI/LFM3-1.7B-MLX-bf16 --adapter adapters/lfm3-best
  zsh src/lfm3.sh src/evaluate.py --model LiquidAI/LFM3-1.7B-MLX-bf16            # untrained base
Results are appended to eval_results.json under --name.
"""
import argparse
import json
import os
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

sys.path.insert(0, str(Path(__file__).parent))
from bakeoff import reply_token_count  # noqa: E402
from encode import encode_example  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / "eval_results.json"
KW = {"enable_thinking": False}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]")
DEVANAGARI = re.compile("[ऀ-ॿ]")
# words that signal "assistant voice" rather than texting
FORMAL = re.compile(r"\b(aap|aapka|aapki|aapko|kripya|dhanyavaad|dhanyawad|assist|assistant|happy to help|"
                    r"certainly|however|furthermore|i understand|feel free|as an ai)\b", re.I)


def style(replies):
    n = len(replies)
    texts = [[l for l in r.split("\n") if l.strip()] for r in replies]
    return {
        "n": n,
        "median_chars": statistics.median(len(r) for r in replies),
        "mean_chars": round(sum(len(r) for r in replies) / n, 1),
        "texts_per_turn": round(sum(max(len(t), 1) for t in texts) / n, 2),
        "double_text_pct": round(100 * sum(len(t) > 1 for t in texts) / n, 1),
        "emoji_pct": round(100 * sum(bool(EMOJI.search(r)) for r in replies) / n, 1),
        "formal_pct": round(100 * sum(bool(FORMAL.search(r)) for r in replies) / n, 1),
        "devanagari_pct": round(100 * sum(bool(DEVANAGARI.search(r)) for r in replies) / n, 1),
        "empty_pct": round(100 * sum(not r.strip() for r in replies) / n, 1),
        "over_300_chars_pct": round(100 * sum(len(r) > 300 for r in replies) / n, 1),
    }


def bpc(model, tok, examples):
    nll, chars = 0.0, 0
    for e in examples:
        msgs, reply = e["messages"], e["messages"][-1]["content"]
        toks, off = encode_example(tok, msgs, KW)
        k = reply_token_count(tok, toks, off, reply)
        x = mx.array(toks)[None]
        logits = model(x[:, :-1]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        lp = mx.take_along_axis(logp, x[:, 1:, None], axis=-1)[0, :, 0]
        nll -= lp[off - 1: off - 1 + k].sum().item()
        chars += len(reply)
    return nll / chars / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()
    name = a.name or (a.adapter or a.model).split("/")[-1]

    rows = [json.loads(l) for l in (Path(os.environ.get("LOCAL_FINETUNING_DATA", str(ROOT / "data"))) / "final/test.jsonl").open(encoding="utf-8")]
    model, tok = load(a.model, adapter_path=str(ROOT / a.adapter) if a.adapter else None)
    res = {"model": a.model, "adapter": a.adapter}

    t0 = time.time()
    res["test_bpc"] = bpc(model, tok, random.Random(7).sample(rows, 500))
    print(f"[{name}] test BPC (500 replies): {res['test_bpc']:.3f}   ({time.time() - t0:.0f}s)", flush=True)

    gen_rows = random.Random(11).sample(rows, a.n_gen)
    sampler = make_sampler(temp=a.temp, top_p=0.9)
    procs = make_logits_processors(repetition_penalty=1.1)
    outs, t0 = [], time.time()
    for r in gen_rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True, **KW)
        outs.append(generate(model, tok, prompt=prompt, max_tokens=150, sampler=sampler, logits_processors=procs).strip())
    real = [r["messages"][-1]["content"] for r in gen_rows]
    res["style_generated"], res["style_real"] = style(outs), style(real)
    print(f"[{name}] generated {a.n_gen} replies in {time.time() - t0:.0f}s", flush=True)

    train_replies = {json.loads(l)["messages"][-1]["content"].strip()
                     for l in (Path(os.environ.get("LOCAL_FINETUNING_DATA", str(ROOT / "data"))) / "final/train.jsonl").open(encoding="utf-8")}
    long_outs = [o for o in outs if len(o) >= 25]
    res["memorised_pct"] = round(100 * sum(o in train_replies for o in long_outs) / max(len(long_outs), 1), 1)

    g, rl = res["style_generated"], res["style_real"]
    print(f"\n{'metric':<20}{'real (you)':>12}{name[:18]:>20}")
    for k in g:
        if k != "n":
            print(f"{k:<20}{rl[k]:>12}{g[k]:>20}")
    print(f"{'memorised (>=25ch)':<20}{'':>12}{str(res['memorised_pct']) + '%':>20}")

    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres[name] = res
    OUT.write_text(json.dumps(allres, indent=2))


if __name__ == "__main__":
    main()
