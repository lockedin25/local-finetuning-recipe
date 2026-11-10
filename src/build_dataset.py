"""Module 3: clean.jsonl -> data/final/{train,valid,test}.jsonl (mlx-lm chat format)

One example per Alex reply:
  {"messages": [system, (context turns...), assistant=<his reply>], "meta": {...}}
mlx-lm's mask_prompt trains only on the LAST message, so every example ends on his reply.
mlx-lm truncates over-long examples from the END, so we enforce the token budget here:
his reply is always kept whole; the oldest context is trimmed instead.
Prints statistics only, never message text.
"""

import os
import json
import random
import statistics as st
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(Path(os.environ.get("LOCAL_FINETUNING_CONFIG", str(ROOT / "config.yaml"))).read_text())
B = CFG["build"]

SYSTEM = B["system_prompt"]
ROLE = {"her": "user", "me": "assistant"}


def load_messages():
    msgs = [json.loads(l) for l in (Path(os.environ.get("LOCAL_FINETUNING_DATA", str(ROOT / "data"))) / "interim/clean.jsonl").open(encoding="utf-8")]
    msgs.sort(key=lambda m: m["ts"])
    for m in msgs:
        m["dt"] = datetime.fromisoformat(m["ts"])
    return msgs


def sessionize(msgs):
    """Split into sessions at gaps >= gap_hours, then merge same-sender runs into turns."""
    sessions, cur = [], []
    for m in msgs:
        if cur and (m["dt"] - cur[-1]["dt"]).total_seconds() >= B["gap_hours"] * 3600:
            sessions.append(cur)
            cur = []
        cur.append(m)
    if cur:
        sessions.append(cur)

    out = []
    for s in sessions:
        turns = []
        for m in s:
            if turns and turns[-1]["sender"] == m["sender"]:
                turns[-1]["text"] += "\n" + m["text"]
                turns[-1]["n_msgs"] += 1
            else:
                turns.append({"sender": m["sender"], "text": m["text"], "ts": m["ts"], "n_msgs": 1})
        out.append(turns)
    return out


def is_filler(text):
    t = text.strip().lower()
    return (t in B["filler_words"]) or (len(t) <= 3 and not any(c.isalnum() for c in t)) or t == "<media>"


def build_examples(sessions, tok):
    tlen = lambda s: len(tok.encode(s, add_special_tokens=False))
    sys_tokens = tlen(SYSTEM) + B["turn_overhead"]
    examples, stats = [], Counter()
    for si, turns in enumerate(sessions):
        tok_counts = [tlen(t["text"]) + B["turn_overhead"] for t in turns]
        for i, t in enumerate(turns):
            if t["sender"] != "me":
                continue
            stats["your turns"] += 1
            if i == 0:
                stats["skipped: you opened the session (no message from her to reply to)"] += 1
                continue
            reply_budget = B["max_seq_tokens"] - sys_tokens - tok_counts[i]
            if reply_budget < tok_counts[i - 1]:
                stats["skipped: reply too long to fit with any context"] += 1
                continue
            # Walk backwards adding context turns while within budget
            ctx_budget = min(B["max_context_tokens"], reply_budget)
            j, used = i, 0
            while j > 0 and (i - (j - 1)) <= B["max_context_turns"] and used + tok_counts[j - 1] <= ctx_budget:
                j -= 1
                used += tok_counts[j]
            if j == i:  # even her last turn alone exceeds the context cap: keep it anyway if it fits overall
                j, used = i - 1, tok_counts[i - 1]
            # Chat templates expect the conversation to start with the user
            while turns[j]["sender"] != "her":
                used -= tok_counts[j]
                j += 1
            ctx = turns[j:i]
            messages = [{"role": "system", "content": SYSTEM}]
            messages += [{"role": ROLE[c["sender"]], "content": c["text"]} for c in ctx]
            messages.append({"role": "assistant", "content": t["text"]})
            examples.append({
                "messages": messages,
                "meta": {"session": si, "ts": t["ts"], "ctx_turns": len(ctx),
                         "reply_msgs": t["n_msgs"], "filler": is_filler(t["text"])},
            })
    return examples, stats


def split(examples, sessions):
    test_from = B["test_from"]
    rng = random.Random(B["seed"])
    early = sorted({e["meta"]["session"] for e in examples if e["meta"]["ts"] < test_from})
    valid_sessions = set(rng.sample(early, max(1, int(len(early) * B["valid_frac"]))))
    out = {"train": [], "valid": [], "test": []}
    for e in examples:
        s = e["meta"]["session"]
        if sessions[s][0]["ts"] >= test_from:
            out["test"].append(e)
        elif s in valid_sessions:
            out["valid"].append(e)
        else:
            out["train"].append(e)
    return out


def downsample_filler(train, max_share, seed):
    """Keep filler replies (ok/haha/...) at most `max_share` of training examples."""
    filler = [e for e in train if e["meta"]["filler"]]
    real = [e for e in train if not e["meta"]["filler"]]
    cap = int(max_share / (1 - max_share) * len(real))
    if len(filler) <= cap:
        return train, 0
    keep = random.Random(seed).sample(filler, cap)
    kept = sorted(real + keep, key=lambda e: e["meta"]["ts"])
    return kept, len(filler) - cap


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-turns", type=int, default=B["max_context_turns"])
    ap.add_argument("--ctx-tokens", type=int, default=B["max_context_tokens"])
    ap.add_argument("--out", default="data/final")
    a = ap.parse_args()
    B["max_context_turns"], B["max_context_tokens"] = a.ctx_turns, a.ctx_tokens
    print(f"context: up to {a.ctx_turns} turns / {a.ctx_tokens} tokens -> {a.out}")
    tok = AutoTokenizer.from_pretrained(B["budget_tokenizer"])
    msgs = load_messages()
    sessions = sessionize(msgs)
    examples, stats = build_examples(sessions, tok)
    splits = split(examples, sessions)

    filler_share = st.mean(e["meta"]["filler"] for e in splits["train"])
    splits["train"], removed = downsample_filler(splits["train"], B["max_filler_share"], B["seed"])

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        p = out_dir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        p.chmod(0o600)

    # ---------- report (numbers only) ----------
    turn_lens = [len(t) for t in sessions]
    print(f"sessions: {len(sessions):,}  (gap >= {B['gap_hours']}h)   turns per session: median {st.median(turn_lens)}, max {max(turn_lens)}")
    print(f"your turns: {stats['your turns']:,}")
    for k, v in stats.items():
        if k.startswith("skipped"):
            print(f"  {k}: {v:,}")
    print(f"examples built: {len(examples):,}")
    print(f"filler share of your replies (train, before cap): {filler_share:.1%}  -> removed {removed:,} to cap at {B['max_filler_share']:.0%}")

    # Exact token counts with the real chat template (what training will see)
    for name, rows in splits.items():
        lens = [len(tok.apply_chat_template(e["messages"], tokenize=True, return_dict=False)) for e in rows]
        ctx = [e["meta"]["ctx_turns"] for e in rows]
        rng = (rows[0]["meta"]["ts"][:10], rows[-1]["meta"]["ts"][:10]) if rows else ("-", "-")
        print(f"{name:5}: {len(rows):6,} examples | {rng[0]} -> {rng[1]} | tokens total {sum(lens):>10,}  "
              f"mean {st.mean(lens):5.0f}  p99 {sorted(lens)[int(0.99*(len(lens)-1))]:5}  max {max(lens):5} | ctx turns mean {st.mean(ctx):.1f}")
    over = sum(1 for e in examples if len(tok.apply_chat_template(e["messages"], tokenize=True, return_dict=False)) > B["max_seq_tokens"])
    print(f"examples over max_seq_tokens ({B['max_seq_tokens']}): {over}")


if __name__ == "__main__":
    main()
