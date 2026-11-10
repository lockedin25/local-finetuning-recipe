"""Dataset v2: the with-media export, every attachment replaced by a text tag, long context.

  uv run python -I src/build_v2.py            (all outputs go to ~/LocalData/hinglish-clone/v2, outside iCloud)

What changes vs v1:
  * new export (with media); captions on photos/videos kept (v1 dropped them)
  * photo / video / gif  -> [photo: <local LFM3-VL description>] etc. (intimate ones: [intimate photo: ...])
  * stickers             -> [sticker: core | context | emotion] from Alex's labels, else the model's guess
  * voice notes, docs, contacts, view-once, unsaved media -> short tags
  * '<This message was edited>' stripped after multi-line merge
  * long context: up to 40 turns / 2,048 tokens (v1: 12 / 512), budgeted with the LFM3 tokenizer
Outputs:
  v2/final/{train,valid,test}.jsonl          mlx-lm chat format (same shape as v1)
  v2/reasoning/  chunks for an outside model to write the <think> reasoning (see README there)
Prints counts only.
"""

import os
import argparse
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from clean_redact import is_paste, redact  # noqa: E402
from parse_whatsapp import EDITED, INVISIBLE, PREFIX, SENDER, parse_dt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(Path(os.environ.get("LOCAL_FINETUNING_CONFIG", str(ROOT / "config.yaml"))).read_text())
B = CFG["build"]
HOME = Path.home()
EXPORT = Path(os.environ.get("WHATSAPP_EXPORT_DIR", str(Path.home() / "Downloads/whatsapp-export")))
LOCAL = (Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))))
OUT = LOCAL / "v2"
TOKENIZER = "LiquidAI/LFM3-1.7B-MLX-bf16"

SYSTEM = B["system_prompt"]
ROLE = {"her": "user", "me": "assistant"}
NAME = {"her": "Sam", "me": "Alex"}
ATT = re.compile(r"^(?P<file>\S.*?\.(?P<ext>\w{2,5})) \(file attached\)$")
DELETED = re.compile(r"^(This message was deleted|You deleted this message)\.?$")
OMITTED = re.compile(r"^<[A-Za-z ]+ omitted>$")
LOCATION = re.compile(r"^location: https?://", re.I)


# ---------------------------------------------------------------- 1. parse (captions kept)
def parse(path):
    msgs, stats = [], Counter()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = INVISIBLE.sub("", raw)
        m = PREFIX.match(line)
        if not m:
            if msgs:
                msgs[-1]["lines"].append(line)
                stats["continuation lines (incl. captions)"] += 1
            continue
        s = SENDER.match(m["rest"])
        if not s or s["sender"] not in CFG["senders"]:
            stats["system lines"] += 1
            continue
        msgs.append({"ts": parse_dt(m["date"], m["time"]).isoformat(), "sender": CFG["senders"][s["sender"]],
                     "lines": [s["text"]]})
    return msgs, stats


# ---------------------------------------------------------------- 2. media -> tags
def load_media():
    desc = {}
    for l in (LOCAL / "media_descriptions.jsonl").open():
        r = json.loads(l)
        desc[r["file"]] = r
    labels = json.loads((LOCAL / "sticker_labels_clean.json").read_text())
    by_file = {v["file"]: v["label"] for v in labels.values()}
    by_sha = {k: v["label"] for k, v in labels.items()}
    return desc, by_file, by_sha


def media_tag(fname, ext, desc, lab_file, lab_sha, stats):
    ext = ext.lower()
    r = desc.get(fname)
    if fname.startswith("STK-"):
        lab = lab_file.get(fname) or (r and lab_sha.get(r["sha1"]))
        if lab:
            stats["sticker: your label"] += 1
            return f"[sticker: {lab}]"
        if r and r.get("desc"):
            stats["sticker: model guess"] += 1
            return f"[sticker: {r['desc'].rstrip('.')}]"
        stats["sticker: animated, unlabelled"] += 1
        return "[animated sticker]"
    if r and r.get("desc"):
        kind = r["kind"]
        if kind == "photo":
            name = "intimate photo" if r["category"] == "intimate" else "photo"
        elif kind == "gif":
            name = "gif"
        else:
            name = "intimate video" if r["category"] == "intimate" else "video"
        stats[f"{name}: described"] += 1
        return f"[{name}: {r['desc'].rstrip('.')}]"
    if ext == "opus" or ext == "mp3":
        stats["voice note"] += 1
        return "[voice note]"
    if ext == "vcf":
        stats["contact card"] += 1
        return "[contact card]"
    if ext in ("jpg", "jpeg", "png", "webp"):
        stats["image: no description"] += 1
        return "[photo]"
    if ext == "mp4":
        stats["video: no description"] += 1
        return "[video]"
    stats["document"] += 1
    return f"[document: .{ext}]"


def to_text(m, desc, lab_file, lab_sha, stats):
    """One message -> (kind, text). Kinds: text, media, deleted, paste."""
    first, rest = m["lines"][0], m["lines"][1:]
    if DELETED.match(first) and not rest:
        return "deleted", ""
    if not first.strip() and not rest:
        stats["blank -> view-once"] += 1
        return "media", "[view-once]"
    if OMITTED.match(first):
        stats["<Media omitted> -> media not saved"] += 1
        return "media", "[media not saved]"
    if LOCATION.match(first):
        stats["location"] += 1
        return "media", "[location]"
    a = ATT.match(first)
    if a:
        tag = media_tag(a["file"], a["ext"], desc, lab_file, lab_sha, stats)
        caption = "\n".join(rest).strip()
        caption = EDITED.sub("", caption).strip()
        if caption:
            stats["captions kept"] += 1
            return "media", redact(f"{tag}\n{caption}", stats)
        return "media", redact(tag, stats)
    text = "\n".join(m["lines"])
    if EDITED.search(text):
        text = EDITED.sub("", text)
        stats["edited marker stripped"] += 1
    if is_paste({"ts": m["ts"], "text": text}):
        stats["pasted AI text -> tag"] += 1
        return "paste", "[pasted text]"
    return "text", redact(text, stats)


# ---------------------------------------------------------------- 3. sessions, turns, examples
def sessionize(msgs):
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


FILLER = set(B["filler_words"]) - {"<media>"}


def is_filler(text):
    t = text.strip().lower()
    return t in FILLER or (len(t) <= 3 and not any(c.isalnum() for c in t))


def build_examples(sessions, tlen, ctx_turns, ctx_tokens, max_seq):
    sys_tokens = tlen(SYSTEM) + B["turn_overhead"]
    examples, stats = [], Counter()
    for si, turns in enumerate(sessions):
        tc = [tlen(t["text"]) + B["turn_overhead"] for t in turns]
        for i, t in enumerate(turns):
            if t["sender"] != "me":
                continue
            stats["your turns"] += 1
            if i == 0:
                stats["skipped: you opened the session"] += 1
                continue
            budget = max_seq - sys_tokens - tc[i]
            if budget < tc[i - 1]:
                stats["skipped: reply too long to fit with any context"] += 1
                continue
            cap = min(ctx_tokens, budget)
            j, used = i, 0
            while j > 0 and (i - (j - 1)) <= ctx_turns and used + tc[j - 1] <= cap:
                j -= 1
                used += tc[j]
            if j == i:
                j, used = i - 1, tc[i - 1]
            while turns[j]["sender"] != "her":
                used -= tc[j]
                j += 1
            ctx = turns[j:i]
            msgs = [{"role": "system", "content": SYSTEM}]
            msgs += [{"role": ROLE[c["sender"]], "content": c["text"]} for c in ctx]
            msgs.append({"role": "assistant", "content": t["text"]})
            examples.append({"messages": msgs,
                             "meta": {"session": si, "turn": i, "rid": f"R-{si:04d}-{i:03d}", "ts": t["ts"], "ctx_turns": len(ctx),
                                      "reply_msgs": t["n_msgs"],
                                      "filler": is_filler(t["text"]), "ctx_ts": [c["ts"] for c in ctx]}})
    return examples, stats


def split(examples, sessions):
    rng = random.Random(B["seed"])
    early = sorted({e["meta"]["session"] for e in examples if e["meta"]["ts"] < B["test_from"]})
    valid = set(rng.sample(early, max(1, int(len(early) * B["valid_frac"]))))
    out = {"train": [], "valid": [], "test": []}
    for e in examples:
        s = e["meta"]["session"]
        out["test" if sessions[s][0]["ts"] >= B["test_from"] else "valid" if s in valid else "train"].append(e)
    return out


def cap_filler(train, share, seed):
    filler = [e for e in train if e["meta"]["filler"]]
    real = [e for e in train if not e["meta"]["filler"]]
    cap = int(share / (1 - share) * len(real))
    if len(filler) <= cap:
        return train, 0
    keep = random.Random(seed).sample(filler, cap)
    return sorted(real + keep, key=lambda e: e["meta"]["ts"]), len(filler) - cap


# ---------------------------------------------------------------- 4. reasoning hand-off
INTIMATE = re.compile(r"\[(intimate (?:photo|video)): [^\]]*\]")


def reasoning_record(e, idx, split_name, redact_intimate):
    ctx = e["messages"][1:-1]
    ts = e["meta"]["ctx_ts"]
    clean = (lambda s: INTIMATE.sub(r"[\1]", s)) if redact_intimate else (lambda s: s)
    return {
        "id": f"{split_name}-{idx:05d}",
        "reply_time": e["meta"]["ts"],
        "conversation": [{"time": ts[k][11:16], "from": "Sam" if c["role"] == "user" else "Alex",
                          "text": clean(c["content"])} for k, c in enumerate(ctx)],
        "Alex_reply": clean(e["messages"][-1]["content"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-turns", type=int, default=40)
    ap.add_argument("--ctx-tokens", type=int, default=2048)
    ap.add_argument("--max-seq", type=int, default=3072)
    ap.add_argument("--chunk", type=int, default=500, help="examples per reasoning chunk file")
    ap.add_argument("--keep-intimate-in-reasoning", action="store_true",
                    help="default: intimate photo/video descriptions become a bare [intimate photo] tag in the "
                         "reasoning chunks (they go to an outside model); training data keeps them in full")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tk = Tokenizer.from_file(hf_hub_download(TOKENIZER, "tokenizer.json"))
    tlen = lambda s: len(tk.encode(s, add_special_tokens=False).ids)

    raw = next(EXPORT.glob("*.txt"))
    msgs, pstats = parse(raw)
    desc, lab_file, lab_sha = load_media()
    stats, kinds, kept = Counter(), Counter(), []
    for m in msgs:
        kind, text = to_text(m, desc, lab_file, lab_sha, stats)
        kinds[kind] += 1
        if kind == "deleted":
            continue
        m["text"], m["dt"] = text, datetime.fromisoformat(m["ts"])
        kept.append(m)
    kept.sort(key=lambda m: m["ts"])

    print(f"messages parsed: {len(msgs):,} | kinds {dict(kinds)} | parse {dict(pstats)}")
    print("augmentation + redaction:", {k: v for k, v in sorted(stats.items(), key=lambda x: -x[1]) if v})

    sessions = sessionize(kept)
    examples, bstats = build_examples(sessions, tlen, a.ctx_turns, a.ctx_tokens, a.max_seq)
    splits = split(examples, sessions)
    fshare = st.mean(e["meta"]["filler"] for e in splits["train"])
    splits["train"], removed = cap_filler(splits["train"], B["max_filler_share"], B["seed"])

    final = OUT / "final"
    final.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (final / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"sessions {len(sessions):,} | your turns {bstats['your turns']:,} | "
          + " | ".join(f"{k} {v:,}" for k, v in bstats.items() if k.startswith("skipped")))
    print(f"examples {len(examples):,} | filler share {fshare:.1%}, removed {removed:,} (cap {B['max_filler_share']:.0%})")
    for name, rows in splits.items():
        lens = [sum(tlen(m["content"]) + B["turn_overhead"] for m in e["messages"]) for e in rows]
        ctx = [e["meta"]["ctx_turns"] for e in rows]
        tagged = sum("[" in e["messages"][-1]["content"] for e in rows)
        print(f"  {name:5}: {len(rows):6,} | {rows[0]['meta']['ts'][:10]} -> {rows[-1]['meta']['ts'][:10]} | "
              f"tokens ~{sum(lens):,} mean {st.mean(lens):.0f} p99 {sorted(lens)[int(.99 * (len(lens) - 1))]} "
              f"max {max(lens)} | ctx turns mean {st.mean(ctx):.1f} | replies with a media tag {tagged:,}")



if __name__ == "__main__":
    main()
