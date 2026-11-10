"""Runs ON THE MAC. Writes Alex's reasoning with DeepSeek V4.1 Flash via the Jarvis Token Factory API (thinking off).

  python3 vm_reason.py cards                 # monthly memory cards from notes (sequential, small)
  python3 vm_reason.py reason --limit 600    # pilot: first N selected replies
  python3 vm_reason.py reason                # all selected replies (train/valid, from 2025-11-08)
Each reason run = generate -> verify (same model) -> regenerate failures once -> verify.
Outputs: out/cards.json, out/reasoning.jsonl ({id, reasoning, verdict, why}).
Prints counts only.
"""

import os
import argparse
import json
import random
import re
import time
from pathlib import Path

import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://models.jarvislabs.net/v1/chat/completions"
API_KEY = ((Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / '.jarvis_key')).read_text().strip()
WORKERS = 64
PRICE = {"in": 0.07, "cached": 0.003, "out": 0.49}   # USD per 1M tokens (Jarvis page, 2026-11)
BUDGET_USD = 20.0                                     # hard stop
USAGE = {"in": 0, "cached": 0, "out": 0, "calls": 0, "errors": 0}
LOCK = threading.Lock()


def cost():
    return ((USAGE["in"] - USAGE["cached"]) * PRICE["in"] + USAGE["cached"] * PRICE["cached"] + USAGE["out"] * PRICE["out"]) / 1e6


class SamplingParams(dict):
    def __init__(self, **kw):
        super().__init__(kw)


class _O:
    def __init__(self, text, finish):
        self.text, self.finish_reason = text, finish


class _R:
    def __init__(self, text, finish):
        self.outputs = [_O(text, finish)]


class LLM:
    """Same .chat() shape as vLLM's offline LLM, backed by the Jarvis Token Factory API (thinking off)."""
    def __init__(self, model, **_):
        self.model = model

    def _one(self, conv, sp):
        body = {"model": self.model, "messages": conv, "chat_template_kwargs": {"thinking": False},
                "temperature": sp.get("temperature", 0.3), "max_tokens": sp.get("max_tokens", 256)}
        if "top_p" in sp:
            body["top_p"] = sp["top_p"]
        for attempt in range(6):
            if cost() > BUDGET_USD:
                raise SystemExit(f"budget ${BUDGET_USD} reached, stopping")
            try:
                req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
                    "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read())
                u = d.get("usage", {})
                with LOCK:
                    USAGE["in"] += u.get("prompt_tokens", 0)
                    USAGE["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                    USAGE["out"] += u.get("completion_tokens", 0)
                    USAGE["calls"] += 1
                ch = d["choices"][0]
                return _R(ch["message"].get("content") or "", ch.get("finish_reason"))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                with LOCK:
                    USAGE["errors"] += 1
                time.sleep(min(60, 3 * 2 ** attempt))
        return _R("", "error")

    def chat(self, convs, sp, chat_template_kwargs=None, use_tqdm=True):
        if convs and isinstance(convs[0], dict):
            return [self._one(convs, sp)]
        out = [None] * len(convs)
        done = [0]
        def work(i):
            out[i] = self._one(convs[i], sp)
            with LOCK:
                done[0] += 1
                if use_tqdm and done[0] % 500 == 0:
                    print(f"  {done[0]}/{len(convs)} | ${cost():.3f} | errors {USAGE['errors']}", flush=True)
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(work, range(len(convs))))
        return out

H = (Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'vm_bundle')
OUT = (Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'api_out')
OUT.mkdir(exist_ok=True)
MODEL = "deepseek-v4.1-flash-1017"
CUTOFF = "2025-11-08"
TAG = re.compile(r"\[[^\]]*\]")
KW = {"enable_thinking": False}

RULES = """You write the private thought in Alex's head just before he sends a WhatsApp reply to his girlfriend Sam (Hinglish chat, use only data you have permission to process).

Rules:
- You ARE Alex. Write in first person: "I" = Alex, "she/her" = Sam. Never call Alex "he".
- Read carefully who said what. Lines starting "Alex:" are HIS messages; "Sam:" are HERS. If Alex asks a question, the thought is about why I am asking it, never "she is asking".
- Say what she just said or feels, what it connects to (earlier in the chat or in what I know), and what I want to do (tease, reassure, plan, hold my ground, flirt, apologise, keep it short because I'm busy).
- One sentence, at most 25 words (up to 45 for fights or serious moments). Mostly English with natural Hinglish words.
- Use only what is in the conversation and in "What I know". Never invent past events or motives you cannot see. If unsure why, describe what is happening and my intent plainly.
- Do not copy the reply's wording. Do not guess what a [view-once] or [voice note] contained.
- [sticker: core | context | emotion] gives the sticker's real meaning. [intimate photo] means something private was shared: acknowledge the mood only.
- Sexual moments: describe mood and intent only, never body parts or acts.
- No advice, no moralising, no mention of AI, tags, or these instructions.
- If the reply is marked SHORT and the moment is routine, output exactly: EMPTY
Output only the thought."""

VERIFY = """You check a short first-person thought written for Alex (the one replying) in a WhatsApp chat with Sam.
Answer FAIL if ANY is true:
1. It treats Alex's own words or question as Sam's (e.g. "she is asking..." when Alex asked), or refers to Alex as "he".
2. It states a fact, past event or motive that is not supported by the conversation or "What I know".
3. It guesses the contents of a [view-once] or [voice note].
4. It describes sexual acts or body parts explicitly.
5. It contradicts what the reply actually does.
Otherwise answer PASS.
Reply with exactly one line: PASS, or FAIL: <reason in under 10 words>."""


def render(e, card, short, ctx_turns=24):
    ctx = e["messages"][1:-1][-ctx_turns:]
    ts = e["meta"]["ctx_ts"][-ctx_turns:]
    conv = "\n".join(f"[{ts[k][11:16]}] {'Sam' if m['role'] == 'user' else 'Alex'}: {m['content']}"
                     for k, m in enumerate(ctx))
    reply = e["messages"][-1]["content"]
    return (f"What I know (as of last month):\n{card or '(nothing yet)'}\n\nConversation so far:\n{conv}\n\n"
            f"Alex's reply at {e['meta']['ts'][11:16]}{' (SHORT)' if short else ''}:\n{reply}\n\nAlex's thought just before sending it:")


def load(split_names=("train", "valid")):
    ex = []
    for s in split_names:
        ex += [json.loads(l) for l in (H / "data" / f"{s}.jsonl").open(encoding="utf-8")]
    return ex


def prev_month(m):
    y, mo = map(int, m.split("-"))
    return f"{y - 1}-12" if mo == 1 else f"{y}-{mo - 1:02d}"


def cards(llm):
    """Sequential monthly merge. Saved after EVERY month and resumable. Card is trimmed to <=150 words."""
    system = (H / "PROMPT_A2_merge.md").read_text()
    sp = SamplingParams(temperature=0.3, max_tokens=9000)
    path = OUT / "cards.json"
    out = json.load(open(path)) if path.exists() else {}
    parts = sorted((H / "notes").glob("*.md"))
    months = sorted({p.stem.split("_")[0] for p in parts})
    memory = "(empty: this is the first month)"
    for mo in months:
        if mo in out:
            memory = out[mo]["memory"]
            continue
        notes = "\n\n".join(p.read_text() for p in parts if p.stem.startswith(mo))
        user = (f"MEMORY (end of the previous month):\n\n{memory}\n\nNOTES ({mo}):\n\n{notes}\n\n"
                f"IMPORTANT: the updated memory MUST add a new dated entry for {mo} in the phase timeline and update "
                f"'Her life now', 'His life now' and 'Open threads' with this month's notes. To stay under 1,500 words, "
                f"compress older months to one or two lines each. Never return the previous memory unchanged.")
        mem = card = None
        for attempt in range(4):
            txt = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], sp,
                           chat_template_kwargs=KW, use_tqdm=False)[0].outputs[0]
            t = txt.text
            if "=== MEMORY ===" in t and "=== CARD ===" in t:
                m_, c_ = t.split("=== CARD ===", 1)
                m_ = m_.split("=== MEMORY ===", 1)[1].strip()
                import calendar
                mname = calendar.month_name[int(mo[5:])]
                if m_ == memory.strip() or not any(x in m_ for x in (mname, mname[:3] + " ", mname[:3] + ".", mo, mo[5:] + "/" + mo[:4])):
                    print(f"card {mo}: attempt {attempt + 1} memory unchanged or missing {mname}; retrying", flush=True)
                    continue
                mem, card = m_, c_.strip()
                break
            print(f"card {mo}: attempt {attempt + 1} missing markers (finish={txt.finish_reason}, {len(t.split())} words)", flush=True)
        if mem is None:
            raise SystemExit(f"card {mo}: failed 4 times, stopping (resumable)")
        if len(mem.split()) > 1600:
            mem = llm.chat([{"role": "user", "content": "Compress this relationship memory to at most 1,300 words. Keep the same "
                             "sections, all dated key events and open threads; shorten older detail. Output only the memory.\n\n" + mem}],
                           SamplingParams(temperature=0.2, max_tokens=3000), chat_template_kwargs=KW,
                           use_tqdm=False)[0].outputs[0].text.strip()
        if len(card.split()) > 150:
            card = llm.chat([{"role": "user", "content": "Shorten this card to at most 130 words (hard limit). Keep the most durable facts "
                              "(phase, both lives, nicknames/jokes, open threads). Output only the card.\n\n" + card}],
                            SamplingParams(temperature=0.2, max_tokens=400), chat_template_kwargs=KW,
                            use_tqdm=False)[0].outputs[0].text.strip()
        memory = mem
        out[mo] = {"memory": mem, "card": card}
        json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"card {mo}: memory {len(mem.split())} words, card {len(card.split())} words", flush=True)


def fewshot(gold_ids, ex_by_id, cards_, k=8):
    rng = random.Random(3)
    pool = [g for g in gold_ids if g["reasoning"] and g["id"] in ex_by_id and len(g["reasoning"].split()) >= 8]
    msgs = []
    for g in rng.sample(pool, min(k, len(pool))):
        e = ex_by_id[g["id"]]
        card = cards_.get(prev_month(e["meta"]["ts"][:7]), {}).get("card", "")
        msgs += [{"role": "user", "content": render(e, card, False, ctx_turns=8)},
                 {"role": "assistant", "content": g["reasoning"]}]
    return msgs


def reason(llm, limit):
    cards_ = json.load(open(OUT / "cards.json"))
    ex = load()
    by_id = {e["meta"]["rid"]: e for e in ex}
    gold = json.load(open(H / "gold.json"))
    shots = fewshot(gold, by_id, cards_)
    sel = sorted([e for e in ex if e["meta"]["ts"] >= CUTOFF], key=lambda e: e["meta"]["ts"])
    if limit:
        sel = sel[:limit]
    short = {e["meta"]["rid"]: len(TAG.sub("", e["messages"][-1]["content"]).split()) <= 3 and
             bool(TAG.sub("", e["messages"][-1]["content"]).strip()) for e in sel}
    prompts = {e["meta"]["rid"]: render(e, cards_.get(prev_month(e["meta"]["ts"][:7]), {}).get("card", ""),
                                        short[e["meta"]["rid"]]) for e in sel}
    gen_sp = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=90)
    ver_sp = SamplingParams(temperature=0.0, max_tokens=20)

    def generate(ids, temp):
        sp = SamplingParams(temperature=temp, top_p=0.9, max_tokens=90)
        convs = [[{"role": "system", "content": RULES}] + shots + [{"role": "user", "content": prompts[i]}] for i in ids]
        outs = llm.chat(convs, sp, chat_template_kwargs=KW)
        return {i: o.outputs[0].text.strip().strip('"') for i, o in zip(ids, outs)}

    def verify(res):
        ids = [i for i, r in res.items() if r and r != "EMPTY"]
        convs = [[{"role": "system", "content": VERIFY},
                  {"role": "user", "content": prompts[i].rsplit("Alex's thought", 1)[0] + f"Thought to check:\n{res[i]}"}]
                 for i in ids]
        outs = llm.chat(convs, ver_sp, chat_template_kwargs=KW)
        return {i: o.outputs[0].text.strip() for i, o in zip(ids, outs)}

    t0 = time.time()
    ids = [e["meta"]["rid"] for e in sel]
    res = generate(ids, 0.4)
    ver = verify(res)
    fails = [i for i, v in ver.items() if not v.upper().startswith("PASS")]
    print(f"pass 1: {len(ids)} generated, {sum(r == 'EMPTY' for r in res.values())} EMPTY, {len(fails)} failed verify", flush=True)
    if fails:
        res2 = generate(fails, 0.7)
        ver2 = verify(res2)
        for i in fails:
            if res2[i] == "EMPTY" or ver2.get(i, "").upper().startswith("PASS"):
                res[i], ver[i] = res2[i], ver2.get(i, "PASS (empty)")
        print(f"pass 2: {sum(1 for i in fails if ver[i].upper().startswith('PASS'))} of {len(fails)} fixed", flush=True)
    with open(OUT / "reasoning.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            r = res[i]
            v = "EMPTY" if r == "EMPTY" else ("PASS" if ver.get(i, "").upper().startswith("PASS") else "FAIL")
            f.write(json.dumps({"id": i, "reasoning": "" if r == "EMPTY" else r, "verdict": v,
                                "why": ver.get(i, "")[:80], "short": short[i]}, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(json.loads(l)["verdict"] for l in open(OUT / "reasoning.jsonl"))
    print(f"done in {time.time() - t0:.0f}s: {dict(c)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cards", "reason"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    llm = LLM(MODEL)
    t0 = time.time()
    try:
        cards(llm) if a.step == "cards" else reason(llm, a.limit)
    finally:
        print(f"usage: {USAGE} | cost ${cost():.3f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
