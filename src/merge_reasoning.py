"""Build dataset v3 (two variants) from v2 + memory cards + reasoning. Local only; prints counts.

  python3 -I src/merge_reasoning.py
Writes ~/LocalData/hinglish-clone/v3/{think,nothink}/{train,valid,test}.jsonl
  * scope: train/valid replies from 2025-11-08 (last 12 months); test unchanged (all test examples)
  * system prompt = base prompt + "What I know right now:" + cleaned card from the END OF THE PREVIOUS MONTH
  * think:   final assistant message gets "reasoning_content" (Claude's hand-written ones win, else DeepSeek PASS;
             EMPTY -> "" which renders as an empty think block). Replies whose reasoning FAILED are dropped
             from BOTH variants so the comparison stays fair.
  * nothink: identical examples, no reasoning (template renders an empty think block)
  * test:    card in the system prompt, no reasoning, identical in both variants
Uses the full local v2 data (intimate descriptions kept, they never left the Mac).
"""

import os
import json
import re
from collections import Counter
from pathlib import Path

L = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))
V2 = L / "v2/final"
OUT = L / "v3"
CUTOFF = "2025-11-08"
EXPLICIT = re.compile(r"\b(boobs?|tits|nipples?|pussy|dick|cock|cum|wet|finger(ed|ing)?|suck(ing)?|blowjob|orgasm)\b", re.I)


def ngrams(s, n=5):
    w = re.findall(r"\w+", s.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def prev_month(m):
    y, mo = map(int, m.split("-"))
    return f"{y - 1}-12" if mo == 1 else f"{y}-{mo - 1:02d}"


def main():
    clean = json.load(open(L / "api_out/cards_clean.json"))
    raw = json.load(open(L / "api_out/cards.json"))
    card = lambda m: clean.get(m) or raw.get(m, {}).get("card", "")
    ds = {json.loads(l)["id"]: json.loads(l) for l in open(L / "api_out/reasoning.jsonl")}
    gold = {}
    for f in sorted((L / "v2/reasoning/out/claude").glob("chunk_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            gold[r["id"]] = r["reasoning"]

    stats = Counter()
    for variant in ("think", "nothink"):
        (OUT / variant).mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        rows = {"think": [], "nothink": []}
        for l in open(V2 / f"{split}.jsonl", encoding="utf-8"):
            e = json.loads(l)
            rid, ts = e["meta"]["rid"], e["meta"]["ts"]
            if split != "test" and ts < CUTOFF:
                continue
            c = card(prev_month(ts[:7]))
            sys_msg = e["messages"][0]["content"] + ("\n\nWhat I know right now:\n" + c if c else "")
            msgs = [{"role": "system", "content": sys_msg}] + e["messages"][1:]
            if split == "test":
                for v in rows:
                    rows[v].append({"messages": msgs, "meta": e["meta"]})
                stats["test"] += 1
                continue
            if rid in gold:
                reasoning, src = gold[rid], "claude"
            elif rid in ds and ds[rid]["verdict"] in ("PASS", "EMPTY"):
                reasoning, src = ds[rid]["reasoning"], "deepseek"
                if reasoning and (EXPLICIT.search(reasoning) or ngrams(reasoning) & ngrams(e["messages"][-1]["content"])):
                    stats[f"{split}: dropped (explicit word or copies reply)"] += 1
                    continue
            else:
                stats[f"{split}: dropped (reasoning failed or missing)"] += 1
                continue
            stats[f"{split}: {src} {'empty' if not reasoning else 'reasoning'}"] += 1
            think_msgs = [dict(m) for m in msgs]
            think_msgs[-1]["reasoning_content"] = reasoning
            rows["think"].append({"messages": think_msgs, "meta": {**e["meta"], "reasoning_src": src}})
            rows["nothink"].append({"messages": msgs, "meta": e["meta"]})
        for v, rs in rows.items():
            with open(OUT / v / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"{split}: examples"] = len(rows["think"])
    for k in sorted(stats):
        print(f"{k}: {stats[k]:,}")
    words = [len(json.loads(l)["messages"][0]["content"].split()) for l in open(OUT / "think/train.jsonl")]
    print(f"system prompt words: median {sorted(words)[len(words) // 2]}, max {max(words)}")


if __name__ == "__main__":
    main()
