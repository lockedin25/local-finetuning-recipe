"""Chat with the v3 clones on the Mac, set up exactly like training (system prompt + memory card + long context).

  zsh src/lfm3.sh src/chat_v3.py --model A                      # A: plain clone (no thinking)
  zsh src/lfm3.sh src/chat_v3.py --model B --mode think         # B: thinks, then replies (thought shown)
  zsh src/lfm3.sh src/chat_v3.py --model C --mode pipeline --thinker D   # D thinks, C texts (or --thinker B)
You type as Sam; the model replies as Alex. Commands: /reset, /quit, /card (show the memory card).
Models are looked up in ~/LocalData/hinglish-clone/models/<name>/mlx (A=A_nothink, B=B_think, C=C_texter, D=D_thinker).
Tip: you can type media the way the training data shows it, e.g. [sticker: kiss | her giving me a kiss | love, affection]
"""

import os
import argparse
import json
import time
from pathlib import Path

import yaml
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
L = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))
NAMES = {"A": "A_nothink", "B": "B_think", "C": "C_texter", "D": "D_thinker"}
SYSTEM = yaml.safe_load(Path(os.environ.get("LOCAL_FINETUNING_CONFIG", str(ROOT / "config.yaml"))).read_text())["build"]["system_prompt"]


def model_path(x):
    p = Path(x).expanduser()
    return str(p) if p.exists() else str(L / "models" / NAMES.get(x, x) / "mlx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="A, B, C, D or a path to an MLX model")
    ap.add_argument("--mode", choices=["plain", "think", "pipeline", "jarvis"], default="plain")
    ap.add_argument("--thinker", default="D", help="pipeline mode: the model that writes the thought (D or B)")
    ap.add_argument("--jarvis-model", default="deepseek-v4.1-flash-1017")
    ap.add_argument("--jarvis-effort", choices=["low", "high", "max"], default="high")
    ap.add_argument("--jarvis-max-tokens", type=int, default=4096,
                    help="Jarvis output budget, including provider thinking tokens")
    ap.add_argument("--card", default="latest", help="memory card month (YYYY-MM) or 'latest' or 'none'")
    ap.add_argument("--ctx-turns", type=int, default=40, help="previous turns kept (training used up to 40)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--hide-thoughts", action="store_true")
    ap.add_argument("--personal-style", action="store_true", help="Experimental personal texting and speaker-grounding instructions")
    a = ap.parse_args()

    card = ""
    if a.card != "none":
        cards = json.loads((L / "api_out/cards_clean.json").read_text())
        card = cards[sorted(cards)[-1] if a.card == "latest" else a.card]
    system = SYSTEM + ("\n\nWhat I know right now:\n" + card if card else "")
    if a.personal_style:
        system += (
            '\n\nStyle: write short, natural Hinglish replies. Follow the latest message, preserve factual details, avoid invented memories, and ask a clear question when context is missing.'











        )
    model, tok = load(model_path(a.model), tokenizer_config={"trust_remote_code": True})
    thinker = tok_t = None
    if a.mode == "pipeline":
        thinker, tok_t = load(model_path(a.thinker), tokenizer_config={"trust_remote_code": True})
    if a.mode == "jarvis":
        from jarvis_planner import JarvisError, plan_reply
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)
    think_sampler = make_sampler(temp=0.3, top_p=0.9)
    print(f"\nChatting with {a.model} ({a.mode}{', thinker ' + a.thinker if a.mode == 'pipeline' else ''}), "
          f"card {a.card}. You are Sam. /reset /card /quit\n")
    if a.mode == "jarvis":
        print(f"Planner: Jarvis {a.jarvis_model}, thinking {a.jarvis_effort}. "
              "Conversation context and memory are sent to Jarvis; intimate media descriptions "
              "are replaced by bare tags. Final replies are generated locally.\n")

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
        if msg == "/card":
            print(f"\n{card or '(no card)'}\n")
            continue
        if not msg:
            continue
        turns.append({"role": "user", "content": msg})
        ctx = turns[-a.ctx_turns:]
        while ctx and ctx[0]["role"] != "user":
            ctx = ctx[1:]
        t0 = time.time()
        thought = ""
        if a.mode in ("pipeline", "jarvis"):
            if a.mode == "jarvis":
                print("  Planning with Jarvis...", flush=True)
                try:
                    thought = plan_reply(ctx, card, model=a.jarvis_model,
                                         effort=a.jarvis_effort, max_tokens=a.jarvis_max_tokens)
                except JarvisError as error:
                    turns.pop()
                    print(f"  {error}\n  Message was not added to the conversation.\n")
                    continue
            else:
                p = tok_t.apply_chat_template([{"role": "system", "content": system}] + ctx,
                                              add_generation_prompt=True, enable_thinking=True)
                thought = generate(thinker, tok_t, prompt=p, max_tokens=80, sampler=think_sampler).split("</think>")[0].strip()
            sys2 = system + ("\n\nWhat I'm thinking right now: " + thought if thought else "")
            p = tok.apply_chat_template([{"role": "system", "content": sys2}] + ctx,
                                        add_generation_prompt=True, enable_thinking=False)
            reply = generate(model, tok, prompt=p, max_tokens=200, sampler=sampler).strip()
        else:
            p = tok.apply_chat_template([{"role": "system", "content": system}] + ctx,
                                        add_generation_prompt=True, enable_thinking=(a.mode == "think"))
            raw = generate(model, tok, prompt=p, max_tokens=300 if a.mode == "think" else 200, sampler=sampler).strip()
            if "</think>" in raw:
                thought, reply = raw.split("</think>", 1)
                thought, reply = thought.replace("<think>", "").strip(), reply.strip()
            elif a.mode == "think":
                turns.pop()
                print("  Thinking did not finish within the token budget. No reply was added; retry the message.\n")
                continue
            else:
                reply = raw
        if thought and not a.hide_thoughts:
            label = "response plan" if a.mode == "jarvis" else "thinking"
            print(f"  ({label} {time.time() - t0:.1f}s: {thought})")
        turns.append({"role": "assistant", "content": reply})
        for line in [l for l in reply.split("\n") if l.strip()]:
            print(f"  me  > {line}")
            time.sleep(0.4)
        print()


if __name__ == "__main__":
    main()
