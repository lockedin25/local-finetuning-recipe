"""Run the reasoning passes (see ~/LocalData/hinglish-clone/v2/reasoning/PLAN.md). Standard library only.

  a1  month notes, all months in parallel          python3 -I src/reasoning_run.py a1 --backend claude --workers 6
  a2  merge notes into dated memory + card          python3 -I src/reasoning_run.py a2 --backend claude
  b   reasoning for chunks, in parallel             python3 -I src/reasoning_run.py b --backend openrouter --chunks 117,118 --tag pilot_deepseek
                                                    python3 -I src/reasoning_run.py b --backend claude --from-month 2025-09 --workers 6

Backends
  claude      headless `claude -p` on Alex's subscription, no tools, no MCP, run from an empty folder
  openrouter  DeepSeek V4 Pro, ONLY zero-data-retention providers that do not collect data
              (key file: ~/LocalData/hinglish-clone/.openrouter_key)
Everything is resumable: finished outputs are skipped. Prints progress counts only, never chat text.
"""

import os
import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

L = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))
R = L / "v2/reasoning"
PART_WORDS = 45000          # months longer than this are split into parts for pass A1
LOCK = threading.Lock()


def say(msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ backends
def call_claude(system, user, model, timeout=1800):
    with tempfile.TemporaryDirectory() as empty:
        cmd = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
               "--no-session-persistence", "--output-format", "text", "--system-prompt", system]
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=empty, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"claude exit {p.returncode}: {p.stderr.strip()[:200]}")
    return p.stdout


def call_openrouter(system, user, model, timeout=1800):
    key = (L / ".openrouter_key").read_text().strip()
    body = {"model": model, "temperature": 0.4, "max_tokens": 32000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # privacy: only providers with zero data retention that do not collect or train on prompts
            "provider": {"zdr": True, "data_collection": "deny", "allow_fallbacks": True}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-Title": "local-finetune-reasoning"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"openrouter HTTP {e.code} (response body withheld for privacy)") from None
    if "choices" not in d:
        raise RuntimeError("openrouter response has no choices (response body withheld for privacy)")
    u = d.get("usage", {})
    with LOCK:
        USAGE["in"] += u.get("prompt_tokens", 0)
        USAGE["out"] += u.get("completion_tokens", 0)
        USAGE["cost"] += float(u.get("cost", 0) or 0)
    return d["choices"][0]["message"]["content"] or ""


USAGE = {"in": 0, "out": 0, "cost": 0.0}


def call(a, system, user):
    return call_claude(system, user, a.model) if a.backend == "claude" else call_openrouter(system, user, a.model)


# ------------------------------------------------------------------ pass A1: month notes
def month_units():
    units = []
    for f in sorted((R / "months").glob("*.md")):
        text = f.read_text(encoding="utf-8")
        head, *sessions = re.split(r"\n(?=## Session )", text)
        parts, cur, words = [], [], 0
        for s in sessions:
            w = len(s.split())
            if cur and words + w > PART_WORDS:
                parts.append(cur)
                cur, words = [], 0
            cur.append(s)
            words += w
        if cur:
            parts.append(cur)
        for k, p in enumerate(parts, 1):
            name = f"{f.stem}_p{k}"
            label = f"{head.strip()} (part {k} of {len(parts)})" if len(parts) > 1 else head.strip()
            units.append((name, f.stem, label + "\n\n" + "\n\n".join(p)))
    return units


def run_a1(a):
    system = (R / "PROMPT_A1_month.md").read_text()
    out = R / "notes"
    out.mkdir(exist_ok=True)
    units = [u for u in month_units() if not (out / f"{u[0]}.md").exists()]
    say(f"A1: {len(units)} month parts to write ({a.backend}, {a.workers} at a time)")

    def one(u):
        name, _, text = u
        for attempt in range(3):
            try:
                notes = call(a, system, text).strip()
                if len(notes.split()) < 80:
                    raise RuntimeError(f"too short ({len(notes.split())} words), possibly a refusal")
                (out / f"{name}.md").write_text(notes + "\n", encoding="utf-8")
                return name, len(notes.split()), None
            except Exception as e:
                err = str(e)[:160]
                time.sleep(10 * (attempt + 1))
        return name, 0, err

    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, u) for u in units]):
            name, w, err = f.result()
            say(f"A1 {name}: " + (f"done, {w} words" if not err else f"FAILED: {err}"))


# ------------------------------------------------------------------ pass A2: merge into dated memory
def run_a2(a):
    system = (R / "PROMPT_A2_merge.md").read_text()
    mem_dir = R / "memory"
    mem_dir.mkdir(exist_ok=True)
    units = month_units()
    missing = [u[0] for u in units if not (R / "notes" / f"{u[0]}.md").exists()]
    if missing:
        sys.exit(f"A2: notes missing for {missing}; run a1 first")
    prev = ""
    for i, (name, month, _) in enumerate(units):
        f = mem_dir / f"after_{name}.md"
        if f.exists():
            prev = f.read_text(encoding="utf-8")
            nxt = units[i + 1][1] if i + 1 < len(units) else None
            if nxt != month:
                (mem_dir / f"after_{month}.md").write_text(prev, encoding="utf-8")
            continue
        notes = (R / "notes" / f"{name}.md").read_text(encoding="utf-8")
        memory_only = prev.split("=== CARD ===")[0].replace("=== MEMORY ===", "").strip() if prev else "(empty: this is the first month)"
        user = f"MEMORY (end of the previous month):\n\n{memory_only}\n\nNOTES ({name}):\n\n{notes}"
        for attempt in range(3):
            try:
                txt = call(a, system, user).strip()
                if "=== MEMORY ===" not in txt or "=== CARD ===" not in txt:
                    raise RuntimeError("missing === MEMORY === / === CARD === markers")
                txt = txt[txt.index("=== MEMORY ==="):]
                f.write_text(txt + "\n", encoding="utf-8")
                mem, card = txt.split("=== CARD ===")
                say(f"A2 {name}: memory {len(mem.split())} words, card {len(card.split())} words")
                break
            except Exception as e:
                say(f"A2 {name}: attempt {attempt + 1} failed: {str(e)[:160]}")
                time.sleep(10)
        else:
            sys.exit(f"A2 stopped at {name}")
        prev = f.read_text(encoding="utf-8")
        nxt = units[i + 1][1] if i + 1 < len(units) else None
        if nxt != month:  # last part of the month: this is the month's snapshot
            (mem_dir / f"after_{month}.md").write_text(prev, encoding="utf-8")


# ------------------------------------------------------------------ pass B: reasoning
LINE = re.compile(r'^\s*\{.*"id"\s*:\s*"(R-\d{4}-\d{3})".*\}\s*,?\s*$')


def parse_out(txt, expected):
    got = {}
    for line in txt.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        try:
            r = json.loads(line.strip().rstrip(","))
        except json.JSONDecodeError:
            continue
        if r.get("id") in expected and isinstance(r.get("reasoning"), str):
            got[r["id"]] = r["reasoning"].strip()
    return got


def run_b(a):
    system = (R / "PROMPT_B_reasoning.md").read_text()
    index = json.loads((R / "chunks/index.json").read_text())
    if a.chunks:
        want = {int(x) for x in a.chunks.split(",")}
        index = [c for c in index if c["chunk"] in want]
    else:
        index = [c for c in index if c["month"] >= a.from_month]
    out = R / "out" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    def existing(c):
        path = out / f"chunk_{c['chunk']:03d}.jsonl"
        if not path.exists():
            return {}
        return parse_out(path.read_text(encoding="utf-8"), set(c["ids"]))

    todo = [c for c in index if set(existing(c)) != set(c["ids"])]
    say(f"B: {len(todo)} chunks to write -> out/{a.tag} ({a.backend} {a.model}, {a.workers} at a time)")

    def one(c):
        name = f"chunk_{c['chunk']:03d}"
        memo = c["memory"]
        if memo.startswith("after_"):
            mf = R / "memory" / f"{memo}.md"
            if not mf.exists():
                return name, 0, 0, f"memory {memo} missing (run a2)"
            memory = mf.read_text(encoding="utf-8").split("=== CARD ===")[0].replace("=== MEMORY ===", "").strip()
        else:
            memory = "(empty: this is the first month of the chat)"
        chunk = (R / "chunks" / f"{name}.md").read_text(encoding="utf-8")
        expected = set(c["ids"])
        got, err = existing(c), None
        for attempt in range(3):
            need = expected - set(got)
            ask = "" if len(need) == len(expected) else (
                "\n\nONLY write the lines for these IDs (the others are done): " + ", ".join(sorted(need)))
            try:
                txt = call(a, system, f"MEMORY:\n\n{memory}\n\nCHUNK:\n\n{chunk}{ask}")
                got.update(parse_out(txt, need))
                if not expected - set(got):
                    break
                err = f"{len(expected - set(got))} IDs missing"
            except Exception as e:
                err = str(e)[:160]
                time.sleep(10 * (attempt + 1))
        missing = sorted(expected - set(got))
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for i in c["ids"]:
                if i in got:
                    f.write(json.dumps({"id": i, "reasoning": got[i]}, ensure_ascii=False) + "\n")
        missing_path = out / f"{name}.missing.json"
        if missing:
            missing_path.write_text(json.dumps(missing))
        else:
            missing_path.unlink(missing_ok=True)
        return name, len(got), len(missing), (err if missing else None)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(one, c) for c in todo]):
            name, ok, miss, err = f.result()
            say(f"B {name}: {ok} written, {miss} missing" + (f" ({err})" if err else ""))
    if a.backend == "openrouter":
        say(f"B usage: {USAGE['in']:,} tokens in, {USAGE['out']:,} out, ${USAGE['cost']:.3f}, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_", choices=["a1", "a2", "b"])
    ap.add_argument("--backend", choices=["claude", "openrouter"], required=True)
    ap.add_argument("--model", default=None, help="claude: opus/sonnet; openrouter: deepseek/deepseek-v4-pro")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunks", default=None, help="comma-separated chunk numbers (pass b)")
    ap.add_argument("--from-month", default="2025-09", help="pass b: chunks from this month on")
    ap.add_argument("--tag", default=None, help="pass b: output folder name under out/")
    a = ap.parse_args()
    a.model = a.model or ("opus" if a.backend == "claude" else "deepseek/deepseek-v4-pro")
    a.tag = a.tag or a.backend
    {"a1": run_a1, "a2": run_a2, "b": run_b}[a.pass_](a)


if __name__ == "__main__":
    main()
