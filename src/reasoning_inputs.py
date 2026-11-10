"""Build the inputs for writing Alex's reasoning (two passes, see v2/reasoning/PLAN.md).

  uv run python -I src/reasoning_inputs.py

Writes to ~/LocalData/hinglish-clone/v2/reasoning/:
  months/YYYY-MM.md    every session of that month, readable transcript (pass A: memory). Includes the
                       test period, so the memory stays current; those months get no reasoning.
  chunks/chunk_NNN.md  consecutive train/valid sessions with an ID on each of Alex's turns (pass B)
  chunks/index.json    chunk -> month, memory snapshot to use, IDs (full / short-optional)
Intimate photo/video descriptions become a bare tag here (these files go to an outside model).
Prints counts only.
"""
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_v2 as V  # noqa: E402

R = V.OUT / "reasoning"
NAME = {"her": "Sam", "me": "Alex"}
TAG = re.compile(r"\[[^\]]*\]")
INTIMATE = re.compile(r"\[(intimate (?:photo|video)): [^\]]*\]")
TARGET_IDS = 350           # reasoning IDs per chunk
PREV_WORDS = 2500          # "earlier this month" context given read-only with each chunk


def sessions_with_messages(kept):
    """Same split as build_v2.sessionize, but keeps each message (for per-message times)."""
    sessions, cur = [], []
    for m in kept:
        if cur and (m["dt"] - cur[-1]["dt"]).total_seconds() >= V.B["gap_hours"] * 3600:
            sessions.append(cur)
            cur = []
        cur.append(m)
    if cur:
        sessions.append(cur)
    out = []
    for s in sessions:
        turns = []
        for m in s:
            if turns and turns[-1][0]["sender"] == m["sender"]:
                turns[-1].append(m)
            else:
                turns.append([m])
        out.append(turns)
    return out


def render_session(si, turns, prev_end, ids):
    start, end = turns[0][0]["dt"], turns[-1][-1]["dt"]
    gap = "first session" if prev_end is None else f"previous session ended {fmt_gap(start - prev_end)} earlier"
    lines = [f"## Session {si:04d} · {start:%a %Y-%m-%d} · {start:%H:%M} to {end:%H:%M} · {gap}"]
    for ti, turn in enumerate(turns):
        for k, m in enumerate(turn):
            text = INTIMATE.sub(r"[\1]", m["text"])
            label = NAME[m["sender"]]
            if k == 0 and (si, ti) in ids:
                label += " {" + ids[(si, ti)] + "}"
            lines.append(f"[{m['dt']:%H:%M}] {label}: {text}")
    return "\n".join(lines)


def fmt_gap(d):
    h = d.total_seconds() / 3600
    return f"{h:.0f} h" if h < 48 else f"{h / 24:.0f} days"


def main():
    raw = next(V.EXPORT.glob("*.txt"))
    msgs, _ = V.parse(raw)
    desc, lab_file, lab_sha = V.load_media()
    from collections import Counter
    stats, kept = Counter(), []
    for m in msgs:
        kind, text = V.to_text(m, desc, lab_file, lab_sha, stats)
        if kind == "deleted":
            continue
        m["text"], m["dt"] = text, datetime.fromisoformat(m["ts"])
        kept.append(m)
    kept.sort(key=lambda m: m["ts"])
    sessions = sessions_with_messages(kept)

    # which turns get an ID: every train/valid example; short text replies are optional
    ids, short = {}, set()
    for split in ("train", "valid"):
        for l in (V.OUT / "final" / f"{split}.jsonl").open(encoding="utf-8"):
            e = json.loads(l)
            mt = e["meta"]
            text = TAG.sub("", e["messages"][-1]["content"]).strip()
            sid = mt["rid"] + (" short" if text and len(text.split()) <= 3 else "")
            ids[(mt["session"], mt["turn"])] = sid
            if sid.endswith("short"):
                short.add(mt["rid"])
    # sanity: the IDs must point at Alex turns in these sessions
    bad = sum(1 for (s, t) in ids if s >= len(sessions) or t >= len(sessions[s]) or sessions[s][t][0]["sender"] != "me")
    assert bad == 0, f"{bad} IDs do not line up with sessions"

    (R / "months").mkdir(parents=True, exist_ok=True)
    (R / "chunks").mkdir(parents=True, exist_ok=True)
    for old in list((R / "months").glob("*.md")) + list((R / "chunks").glob("*")):
        old.unlink()

    by_month = defaultdict(list)
    prev_end = None
    rendered = {}
    for si, turns in enumerate(sessions):
        rendered[si] = render_session(si, turns, prev_end, ids)
        prev_end = turns[-1][-1]["dt"]
        by_month[f"{turns[0][0]['dt']:%Y-%m}"].append(si)

    months = sorted(by_month)
    for mo in months:
        test = mo >= V.B["test_from"][:7]
        head = f"# {mo}" + (" (from 18 Sep: test period, memory only)" if test else "")
        (R / "months" / f"{mo}.md").write_text(head + "\n\n" + "\n\n".join(rendered[s] for s in by_month[mo]) + "\n",
                                               encoding="utf-8")

    # pass B chunks: train/valid sessions only, never crossing a month
    index, n = [], 0
    for mi, mo in enumerate(months):
        sess = [s for s in by_month[mo] if any((s, t) in ids for t in range(len(sessions[s])))]
        cur, cur_ids = [], 0
        groups = []
        for s in sess:
            k = sum(1 for t in range(len(sessions[s])) if (s, t) in ids)
            if cur and cur_ids + k > TARGET_IDS:
                groups.append(cur)
                cur, cur_ids = [], 0
            cur.append(s)
            cur_ids += k
        if cur:
            groups.append(cur)
        for g in groups:
            earlier = [s for s in by_month[mo] if s < g[0]]
            prev_txt, words = [], 0
            for s in reversed(earlier):
                w = len(rendered[s].split())
                if words + w > PREV_WORDS:
                    break
                prev_txt.insert(0, re.sub(r" \{R-[^}]+\}", "", rendered[s]))
                words += w
            chunk_ids = [ids[(s, t)] for s in g for t in range(len(sessions[s])) if (s, t) in ids]
            memo = f"after_{months[mi - 1]}" if mi > 0 else "none (start of the relationship chat)"
            body = [f"# Chunk {n:03d} · {mo} · sessions {g[0]:04d} to {g[-1]:04d} · {len(chunk_ids)} IDs",
                    f"Memory to use: {memo}"]
            if prev_txt:
                body += ["", "## Earlier this month (read only, no IDs)", "", "\n\n".join(prev_txt)]
            body += ["", "## Write reasoning for this part", "", "\n\n".join(rendered[s] for s in g)]
            (R / "chunks" / f"chunk_{n:03d}.md").write_text("\n".join(body) + "\n", encoding="utf-8")
            index.append({"chunk": n, "month": mo, "memory": memo, "sessions": [g[0], g[-1]],
                          "ids": [i.split()[0] for i in chunk_ids], "short": [i.split()[0] for i in chunk_ids if i.endswith("short")]})
            n += 1
    (R / "chunks" / "index.json").write_text(json.dumps(index, indent=1))

    mw = {mo: len((R / "months" / f"{mo}.md").read_text(encoding="utf-8").split()) for mo in months}
    cw = [len((R / "chunks" / f"chunk_{c['chunk']:03d}.md").read_text(encoding="utf-8").split()) for c in index]
    print(f"months: {len(months)} ({months[0]} to {months[-1]}), words per month min {min(mw.values()):,} "
          f"median {sorted(mw.values())[len(mw) // 2]:,} max {max(mw.values()):,}, total {sum(mw.values()):,}")
    print(f"chunks: {len(index)} | IDs {len(ids):,} (short, optional {len(short):,}) | words per chunk median "
          f"{sorted(cw)[len(cw) // 2]:,} max {max(cw):,} | total words sent in pass B {sum(cw):,}")


if __name__ == "__main__":
    main()
