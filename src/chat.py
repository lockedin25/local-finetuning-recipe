"""Chat with the clone (or the untrained base model) in your terminal.

  uv run python src/chat.py --adapter adapters/run1      # the clone
  uv run python src/chat.py                               # untrained Gemma-4, for comparison
  zsh src/lfm3.sh src/chat.py --adapter adapters/lfm3-run1 --ctx-turns 12   # LFM3 clone
You type as her; the model replies as Alex. Commands: /reset (new conversation), /quit.

Matches training exactly: same system prompt, thinking OFF, last N turns as context.
Multi-line replies are shown as separate texts, like WhatsApp double-texting.
"""

import os
import argparse
import re
import json
import time
from pathlib import Path

import yaml
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

ROOT = Path(__file__).resolve().parent.parent
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F1E6-\U0001F1FF\uFE0F\u200D]")


def clean(reply, no_emoji):
    """Drop sticker/photo placeholders (WhatsApp exports them as empty <media>); optionally strip emoji."""
    lines = [l for l in reply.split("\n") if l.strip() and l.strip() not in ("<media>", "<location>")]
    text = "\n".join(l.replace("<media>", "").strip() for l in lines)
    if no_emoji:
        text = "\n".join(EMOJI.sub("", l).strip() for l in text.split("\n"))
    return "\n".join(l for l in text.split("\n") if l.strip())
CFG = yaml.safe_load(Path(os.environ.get("LOCAL_FINETUNING_CONFIG", str(ROOT / "config.yaml"))).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter folder, e.g. adapters/run1")
    ap.add_argument("--model", default=None, help="base model (default: the adapter's own model, else config.yaml)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="1.0 = off. >1 penalises words already in the chat, which pushes replies OFF topic")
    ap.add_argument("--ctx-turns", type=int, default=12, help="previous turns kept as context (training used up to 11)")
    ap.add_argument("--no-emoji", action="store_true", help="strip emoji from replies")
    ap.add_argument("--think", action="store_true", help="let the model reason before replying (LFM3/Gemma/Qwen)")
    ap.add_argument("--show-thoughts", action="store_true", help="with --think: print the reasoning too")
    a = ap.parse_args()

    repo = a.model or CFG["train"]["model"]
    if a.adapter and not a.model:
        repo = json.loads((ROOT / a.adapter / "adapter_config.json").read_text())["model"]
    model, tok = load(repo, adapter_path=str(ROOT / a.adapter) if a.adapter else None)
    kw = dict(CFG["train"]["template_kwargs"])
    if a.think:
        kw["enable_thinking"] = True
    system = {"role": "system", "content": CFG["build"]["system_prompt"]}
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)
    procs = make_logits_processors(repetition_penalty=a.rep_penalty)
    who = f"clone ({a.adapter}, base {repo})" if a.adapter else f"UNTRAINED {repo}"
    print(f"\nChatting with {who}. You are her. /reset = new conversation, /quit = exit.\n")

    turns = []
    while True:
        try:
            msg = input("her > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "/quit":
            break
        if msg == "/reset":
            turns = []
            print("-- new conversation --\n")
            continue
        if not msg:
            continue
        turns.append({"role": "user", "content": msg})
        ctx = turns[-a.ctx_turns:]
        while ctx and ctx[0]["role"] != "user":
            ctx = ctx[1:]
        prompt = tok.apply_chat_template([system] + ctx, add_generation_prompt=True, **kw)
        t0 = time.time()
        for _ in range(4):  # regenerate if the reply was only a sticker/photo placeholder
            raw = generate(model, tok, prompt=prompt, max_tokens=1500 if a.think else 200,
                           sampler=sampler, logits_processors=procs).strip()
            reply = clean(raw.split("</think>")[-1], a.no_emoji)
            if reply:
                break
        reply = raw if "</think>" in raw else reply
        if "</think>" in reply:  # keep only the text after the reasoning
            thought, reply = reply.split("</think>", 1)
            reply = clean(reply, a.no_emoji)
            if a.show_thoughts:
                print(f"  (thought {time.time() - t0:.0f}s: {thought.replace('<think>', '').strip()[:600]})")
        turns.append({"role": "assistant", "content": reply})
        for line in [l for l in reply.split("\n") if l.strip()]:
            print(f"  me > {line}")
            time.sleep(0.4)
        print()


if __name__ == "__main__":
    main()
