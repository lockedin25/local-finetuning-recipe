"""Check reasoning outputs and build the blind pilot comparison page. Prints counts only.

  python3 -I src/reasoning_review.py check pilot_deepseek
  python3 -I src/reasoning_review.py compare pilot_deepseek pilot_claude --n 60
      -> ~/LocalData/hinglish-clone/v2/reasoning/pilot_compare.html  (local file; writers shown as A/B)
         the A/B key is saved to pilot_key.json; your votes download as pilot_votes.json
"""

import os
import argparse
import html
import json
import random
import re
import statistics as st
from pathlib import Path

L = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))
R = L / "v2/reasoning"
TAG = re.compile(r"\[[^\]]*\]")
BANNED = re.compile(r"\b(AI|dataset|chunk|memory file|the memory|transcript|tags?|ID|assistant|user|model)\b")
EXPLICIT = re.compile(r"\b(sex|sexual|naked|nude|boobs?|breasts?|dick|penis|vagina|pussy|cum|orgasm|fuck\w*|horny)\b", re.I)


def load_examples():
    ex = {}
    for split in ("train", "valid"):
        for l in (L / "v2/final" / f"{split}.jsonl").open(encoding="utf-8"):
            e = json.loads(l)
            ex[e["meta"]["rid"]] = e
    return ex


def load_out(tag):
    rows = {}
    for f in sorted((R / "out" / tag).glob("chunk_*.jsonl")):
        for l in f.open(encoding="utf-8"):
            r = json.loads(l)
            rows[r["id"]] = r["reasoning"]
    return rows


def ngrams(s, n=4):
    w = re.findall(r"\w+", s.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def check(tag):
    rows, ex = load_out(tag), load_examples()
    index = {i: c for c in json.loads((R / "chunks/index.json").read_text()) for i in c["ids"]}
    short = {i for c in json.loads((R / "chunks/index.json").read_text()) for i in c["short"]}
    chunks = {index[i]["chunk"] for i in rows}
    expected = sum(len(c["ids"]) for c in json.loads((R / "chunks/index.json").read_text()) if c["chunk"] in chunks)
    full = {i: r for i, r in rows.items() if i not in short}
    sh = {i: r for i, r in rows.items() if i in short}
    words = [len(r.split()) for r in full.values() if r]
    copy = [i for i, r in rows.items() if r and ngrams(r) & ngrams(ex[i]["messages"][-1]["content"])]
    print(f"[{tag}] chunks {len(chunks)} | IDs answered {len(rows):,} of {expected:,}")
    print(f"  full replies: {len(full):,}, empty {sum(not r for r in full.values()):,} | words median "
          f"{st.median(words) if words else 0} p95 {sorted(words)[int(.95 * (len(words) - 1))] if words else 0} "
          f"max {max(words) if words else 0} | over 50 words {sum(w > 50 for w in words)}")
    print(f"  short replies: {len(sh):,}, written {sum(bool(r) for r in sh.values()):,} "
          f"({100 * sum(bool(r) for r in sh.values()) / max(len(sh), 1):.0f}%)")
    print(f"  copies 4+ words of its reply: {len(copy)} | meta words (AI/tag/chunk/memory...): "
          f"{sum(bool(BANNED.search(r)) for r in rows.values())} | explicit words: "
          f"{sum(bool(EXPLICIT.search(r)) for r in rows.values())}")


def compare(tag_a, tag_b, n, seed=7):
    a, b, ex = load_out(tag_a), load_out(tag_b), load_examples()
    short = {i for c in json.loads((R / "chunks/index.json").read_text()) for i in c["short"]}
    both = sorted(i for i in set(a) & set(b) if (a[i] or b[i]))
    rng = random.Random(seed)
    pick = sorted(rng.sample([i for i in both if i not in short], min(n - n // 6, len(both))) +
                  rng.sample([i for i in both if i in short], min(n // 6, len([i for i in both if i in short]))))
    key, cards = {}, []
    for k, i in enumerate(pick, 1):
        flip = rng.random() < 0.5
        left, right = (b[i], a[i]) if flip else (a[i], b[i])
        key[i] = {"A": tag_b if flip else tag_a, "B": tag_a if flip else tag_b}
        e = ex[i]
        ctx = e["messages"][1:-1][-8:]
        convo = "".join(f'<div class="{ "v" if m["role"] == "user" else "s"}"><b>{"Sam" if m["role"] == "user" else "Alex"}:</b> '
                        f'{html.escape(m["content"]).replace(chr(10), "<br>")}</div>' for m in ctx)
        reply = html.escape(e["messages"][-1]["content"]).replace("\n", "<br>")
        cards.append(f'''<div class="card" data-id="{i}"><div class="h">#{k} · {i} · {e["meta"]["ts"][:16].replace("T", " ")}{" · short reply" if i in short else ""}</div>
<div class="convo">{convo}<div class="s r"><b>Alex (real reply):</b> {reply}</div></div>
<div class="opts"><label><input type="radio" name="{i}" value="A"><span class="t">A</span> {html.escape(left) or "<i>(empty)</i>"}</label>
<label><input type="radio" name="{i}" value="B"><span class="t">B</span> {html.escape(right) or "<i>(empty)</i>"}</label>
<label class="tie"><input type="radio" name="{i}" value="tie"> both fine</label><label class="tie"><input type="radio" name="{i}" value="neither"> both wrong</label></div></div>''')
    (R / "pilot_key.json").write_text(json.dumps(key, indent=1))
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Reasoning pilot</title><style>
body{{font-family:-apple-system,sans-serif;max-width:900px;margin:20px auto;padding:0 16px;background:#fafafa;color:#222}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px;margin:14px 0}} .h{{font-size:12px;color:#888;margin-bottom:8px}}
.convo div{{padding:3px 0;font-size:14px}} .v b{{color:#b0306a}} .s b{{color:#2a5db0}} .r{{background:#eef4ff;padding:6px!important;border-radius:6px;margin-top:4px}}
.opts{{margin-top:10px;display:flex;flex-direction:column;gap:6px}} .opts label{{display:block;padding:8px;border:1px solid #e3e3e3;border-radius:8px;cursor:pointer}}
.t{{display:inline-block;width:22px;font-weight:700}} .tie{{font-size:13px;color:#666;border:none!important;padding:2px 8px!important}}
.bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:2}} button{{padding:8px 14px;font-size:15px}}</style></head><body>
<div class="bar"><b>{len(pick)} replies.</b> For each: which reasoning better explains why you replied that way, using what came before?
<button onclick="save()">Save votes</button> <span id="st"></span></div>{"".join(cards)}
<script>const K="pilot_votes_v1";let v={{}};try{{v=JSON.parse(localStorage.getItem(K)||"{{}}")}}catch(e){{}}
document.querySelectorAll("input[type=radio]").forEach(r=>{{if(v[r.name]===r.value)r.checked=true;r.onchange=()=>{{v[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(v))}}catch(e){{}}}}}});
function save(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(v,null,1)],{{type:"application/json"}}));
a.download="pilot_votes.json";a.click();document.getElementById("st").textContent="saved "+Object.keys(v).length+" votes to Downloads";}}</script></body></html>'''
    p = R / "pilot_compare.html"
    p.write_text(page, encoding="utf-8")
    print(f"compare page: {p} ({len(pick)} replies, {sum(i in short for i in pick)} of them short)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "compare"])
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "check":
        for t in a.tags:
            check(t)
    else:
        compare(a.tags[0], a.tags[1], a.n)


if __name__ == "__main__":
    main()
