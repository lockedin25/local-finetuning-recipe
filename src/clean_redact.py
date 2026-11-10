"""Module 2: messages.jsonl -> clean.jsonl with sensitive strings replaced by tags.

Keeps conversation structure (media becomes <media>), drops deleted messages,
and redacts links, emails, UPI IDs, phones, long numbers, OTPs, PANs and passwords.
Prints counts only, never message text.
"""

import os
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("LOCAL_FINETUNING_DATA", str(ROOT / "data"))) / "interim/messages.jsonl"
OUT = Path(os.environ.get("LOCAL_FINETUNING_DATA", str(ROOT / "data"))) / "interim/clean.jsonl"

UPI_HANDLES = r"(?:ok(?:axis|sbi|hdfcbank|icici)|ybl|ibl|axl|apl|paytm|upi|okbizaxis|waicici|yapl|jupiteraxis|fbl|kotak|hdfcbank|icici|sbi|axisbank|freecharge|airtel|naviaxis|superyes|pthdfc|ptsbi|ptyes|ptaxis)"

# (tag, pattern) applied IN ORDER. Earlier patterns protect their matches from later ones.
RULES = [
    ("LINK",   re.compile(r"(?:https?://|www\.)\S+", re.I)),
    ("EMAIL",  re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("UPI",    re.compile(rf"\b[\w.-]{{2,}}@{UPI_HANDLES}\b", re.I)),
    ("PHONE",  re.compile(r"(?<![\w])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),          # Indian mobile
    ("PHONE",  re.compile(r"(?<![\w])\+(?!91)\d{1,3}[\s-]?\d(?:[\s-]?\d){6,12}(?!\d)")),          # international
    ("NUM",    re.compile(r"(?<!\d)\d(?:[\s-]?\d){11,18}(?!\d)")),                                # cards, accounts, Aadhaar
    ("ID",     re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),                                            # PAN
    ("HANDLE", re.compile(r"(?<![\w@<])@[A-Za-z0-9_.]{3,30}")),                                   # @instagram handles
    ("NUM",    re.compile(r"(?<![\d.])\d{7,}(?![\d%]|\.\d)")),  # any other 7+ digit run, except decimals/percentages
]
OTP_CONTEXT = re.compile(r"\b(otp|code|pin|cvv|verification|verify)\b", re.I)
OTP = re.compile(r"(?<!\d)\d{4,8}(?!\d)")
SECRET = re.compile(r"\b(password|passcode|pwd|wifi pass(?:word)?|pin)\b(\s*(?:is|hai|:|-|=)\s*)(\S+)", re.I)


def redact(text: str, counts: Counter) -> str:
    for tag, pat in RULES:
        text, n = pat.subn(f"<{tag}>", text)
        counts[tag] += n
    text, n = SECRET.subn(lambda m: f"{m[1]}{m[2]}<SECRET>", text)
    counts["SECRET"] += n
    if OTP_CONTEXT.search(text):
        text, n = OTP.subn("<OTP>", text)
        counts["OTP"] += n
    return text


import yaml

KEEP_LONG = set(yaml.safe_load(Path(os.environ.get("LOCAL_FINETUNING_CONFIG", str(ROOT / "config.yaml"))).read_text()).get("keep_long") or [])
MARKDOWN = re.compile(r"\*\*[^*\n]+\*\*|^#{1,4} ", re.M)
LIST_ITEM = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+\S", re.M)


def is_paste(m: dict) -> bool:
    """Obvious pasted AI output: markdown or document-style list formatting.
    Length alone is NOT a signal: Alex's long plain-prose messages are genuine (emotional) and stay."""
    t = m["text"]
    if m["ts"] in KEEP_LONG:
        return False
    return bool(MARKDOWN.search(t)) or (len(LIST_ITEM.findall(t)) >= 3 and t.count("\n") >= 6)


def main():
    counts, kept, dropped, touched = Counter(), Counter(), Counter(), 0
    with SRC.open(encoding="utf-8") as f, OUT.open("w", encoding="utf-8") as out:
        for line in f:
            m = json.loads(line)
            if m["kind"] == "deleted":
                dropped["deleted"] += 1
                continue
            if m["kind"] == "media":
                m["text"] = "<media>"
            elif m["kind"] == "location":
                m["text"] = "<location>"
            elif is_paste(m):
                m["kind"], m["text"] = "paste", "<pasted text>"
                dropped[f"paste ({m['sender']}) -> <pasted text>"] += 1
            else:
                before = m["text"]
                m["text"] = redact(before, counts)
                touched += m["text"] != before
            kept[m["kind"]] += 1
            out.write(json.dumps(m, ensure_ascii=False) + "\n")
    OUT.chmod(0o600)

    print(f"kept: {sum(kept.values()):,} {dict(kept)}   dropped: {dict(dropped)}")
    print(f"messages with >=1 redaction: {touched:,}")
    print("redactions by type:", {k: v for k, v in counts.most_common() if v})

    # Residual scan on the OUTPUT: anything suspicious left? (counts only)
    resid = Counter()
    for line in OUT.open(encoding="utf-8"):
        t = json.loads(line)["text"]
        resid["digit runs >= 7"] += len(re.findall(r"\d{7,}", t))
        resid["'@' left"] += t.count("@")
        resid["'http' left"] += t.lower().count("http")
    print("residual scan:", dict(resid))


if __name__ == "__main__":
    main()
