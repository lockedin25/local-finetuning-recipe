"""Meaning-based evaluation: DeepSeek (Jarvis API, approved) judges each setup's reply BLIND against the conversation
and Alex's real reply. Prints scores only.
  python3 -I src/judge_eval.py   (reads ~/LocalData/hinglish-clone/eval/generations.jsonl)
Scores per reply (1-5): sense (responds to what she said, uses context), intent (same intent as the real reply),
style (sounds like Alex), plus which candidate is best overall.
"""

import os
import json
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jarvis_reason as J  # noqa: E402

L = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))
SETUPS = ["A", "B", "BC", "DC"]
PROMPT = """You evaluate candidate WhatsApp replies written by models imitating Alex, replying to his girlfriend Sam
(Hinglish chat). You get the recent conversation, Alex's REAL reply, and candidate replies labelled 1 to 4 in random
order. Judge MEANING, not exact wording.
For each candidate give three scores from 1 (bad) to 5 (excellent):
- sense: does it respond to what she just said and fit the conversation and context?
- intent: does it have the same intent as the real reply (e.g. reassure, tease, plan, apologise, hold ground, flirt)?
- style: does it sound like Alex's texting (tone, Hinglish, length)?
Then name the single best candidate overall.
Answer ONLY with JSON: {"1": {"sense": n, "intent": n, "style": n}, "2": {...}, "3": {...}, "4": {...}, "best": "k"}"""


def main():
    gens = [json.loads(l) for l in open(L / "eval/generations.jsonl", encoding="utf-8")]
    ex = {}
    for l in open(L / "v3/nothink/test.jsonl", encoding="utf-8"):
        e = json.loads(l)
        ex[e["meta"]["rid"]] = e
    rng = random.Random(5)
    convs, orders = [], []
    for g in gens:
        ctx = ex[g["id"]]["messages"][1:-1][-12:]
        conv = "\n".join(f"{'Sam' if m['role'] == 'user' else 'Alex'}: {m['content']}" for m in ctx)
        order = SETUPS[:]
        rng.shuffle(order)
        cands = "\n".join(f"Candidate {i + 1}: {g[s] or '(empty)'}" for i, s in enumerate(order))
        convs.append([{"role": "system", "content": PROMPT},
                      {"role": "user", "content": f"CONVERSATION:\n{conv}\n\nREAL REPLY: {g['real']}\n\n{cands}"}])
        orders.append(order)
    llm = J.LLM(J.MODEL)
    outs = llm.chat(convs, J.SamplingParams(temperature=0.0, max_tokens=300), use_tqdm=False)
    scores = defaultdict(lambda: defaultdict(list))
    best, bad = Counter(), 0
    for o, order in zip(outs, orders):
        t = o.outputs[0].text
        m = re.search(r"\{.*\}", t, re.S)
        try:
            d = json.loads(m.group(0))
            for i, s in enumerate(order):
                for k in ("sense", "intent", "style"):
                    scores[s][k].append(float(d[str(i + 1)][k]))
            best[order[int(d["best"]) - 1]] += 1
        except Exception:
            bad += 1
    n = len(gens) - bad
    print(f"judged {n} conversations ({bad} unparseable) | cost ${J.cost():.3f}")
    print(f"{'setup':6}{'sense':>8}{'intent':>8}{'style':>8}{'best %':>9}")
    for s in SETUPS:
        print(f"{s:6}" + "".join(f"{st.mean(scores[s][k]):8.2f}" for k in ("sense", "intent", "style"))
              + f"{100 * best[s] / max(n, 1):8.0f}%")
    json.dump({s: {k: st.mean(v) for k, v in scores[s].items()} | {"best_pct": best[s] / max(n, 1)} for s in SETUPS},
              open(L / "eval/judge_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
