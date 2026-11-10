"""Module 1: WhatsApp (Android export) -> data/interim/messages.jsonl

One JSON object per message: {"ts", "sender", "text", "kind"}.
Nothing is dropped here; every message gets a `kind` label so Module 2
can decide what to filter. Prints counts only, never message text.
"""

import os
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(Path(os.environ.get("LOCAL_FINETUNING_CONFIG", str(ROOT / "config.yaml"))).read_text())

# Invisible direction marks WhatsApp sprinkles into lines (U+200E, U+200F, U+202A-E, U+2066-9)
INVISIBLE = re.compile(r"[‎‏‪-‮⁦-⁩]")
# "08/12/2024, 19:34 - Sender: text"   (the sender part is optional for system lines)
PREFIX = re.compile(r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}), (?P<time>\d{1,2}:\d{2}) - (?P<rest>.*)$")
SENDER = re.compile(r"^(?P<sender>[^:]{1,60}): (?P<text>.*)$", re.S)

# Placeholders WhatsApp writes instead of real content (export "without media")
KINDS = [
    ("media",   re.compile(r"^<[A-Za-z ]+ omitted>$")),   # <Media omitted>, <Video note omitted>, ...
    ("media",   re.compile(r"\.(vcf|jpg|jpeg|png|webp|mp4|opus|pdf|docx?) \(file attached\)$", re.I)),
    ("deleted", re.compile(r"^(This message was deleted|You deleted this message)\.?$")),
    ("viewonce", re.compile(r"^null$")),
    ("poll",    re.compile(r"^POLL:")),
    ("location", re.compile(r"^location: https://maps\.google", re.I)),
]
EDITED = re.compile(r"\s*<This message was edited>\s*$")


def parse_dt(date: str, time: str) -> datetime:
    d, m, y = (int(x) for x in date.split("/"))
    if CFG["date_order"] == "MDY":
        d, m = m, d
    if y < 100:
        y += 2000
    hh, mm = (int(x) for x in time.split(":"))
    return datetime(y, m, d, hh, mm)


def classify(text: str) -> str:
    for kind, pat in KINDS:
        if pat.search(text):
            return kind
    return "text"


def parse(path: Path):
    msgs, stats = [], Counter()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = INVISIBLE.sub("", raw)
        m = PREFIX.match(line)
        if not m:
            # No timestamp: continuation of the previous message (multi-line text)
            if msgs and msgs[-1]["kind"] == "text":
                msgs[-1]["text"] += "\n" + line
                stats["continuation lines merged"] += 1
            else:
                stats["orphan lines (no message to attach to)"] += 1
            continue
        ts = parse_dt(m["date"], m["time"])
        s = SENDER.match(m["rest"])
        if not s or s["sender"] not in CFG["senders"]:
            stats["system lines (encryption notice, name changes...)"] += 1
            continue
        text, kind = s["text"], classify(s["text"])
        msgs.append({"ts": ts.isoformat(), "sender": CFG["senders"][s["sender"]], "text": text, "kind": kind})
    # Strip the edited marker AFTER multi-line messages are merged: on a multi-line message WhatsApp puts
    # the marker at the end of the LAST line, which the per-line check used to miss (532 leaked into run 1).
    for msg in msgs:
        if msg["kind"] == "text" and EDITED.search(msg["text"]):
            msg["text"] = EDITED.sub("", msg["text"])
            stats["edited marker stripped"] += 1
    return msgs, stats


def main():
    msgs, stats = parse(ROOT / CFG["raw_chat"])
    out = ROOT / CFG["messages_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for msg in msgs:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    out.chmod(0o600)

    ts = sorted(m["ts"] for m in msgs)
    print(f"messages parsed: {len(msgs):,}   range {ts[0][:10]} -> {ts[-1][:10]}")
    print("by sender:", dict(Counter(m["sender"] for m in msgs)))
    print("by kind:  ", dict(Counter(m["kind"] for m in msgs)))
    multi = sum(1 for m in msgs if "\n" in m["text"])
    print(f"multi-line messages: {multi:,}")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
    print("wrote private messages output")


if __name__ == "__main__":
    main()
