"""Local review page: N random reasonings (default from 2026) next to their conversation and the real reply.
Alex marks right / partly / wrong; votes download as reasoning_votes.json. Prints counts only.
  python3 -I src/review_reasoning_page.py --n 30 --from 2026-01
"""

import os
import argparse
import html
import json
import random
from pathlib import Path

L = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--from", dest="frm", default="2026-01")
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    rs = {json.loads(l)["id"]: json.loads(l) for l in open(L / "api_out/reasoning.jsonl")}
    ex = {}
    for s in ("train", "valid"):
        for l in open(L / "v2/final" / f"{s}.jsonl", encoding="utf-8"):
            e = json.loads(l)
            ex[e["meta"]["rid"]] = e
    pool = [i for i, r in rs.items() if r["verdict"] == "PASS" and r["reasoning"] and i in ex and ex[i]["meta"]["ts"] >= a.frm]
    pick = sorted(random.Random(a.seed).sample(pool, min(a.n, len(pool))), key=lambda i: ex[i]["meta"]["ts"])
    cards = []
    for k, i in enumerate(pick, 1):
        e = ex[i]
        ctx = e["messages"][1:-1][-12:]
        conv = "".join(f'<div class="{"v" if m["role"] == "user" else "s"}"><b>{"Sam" if m["role"] == "user" else "Alex"}:</b> '
                       f'{html.escape(m["content"]).replace(chr(10), "<br>")}</div>' for m in ctx)
        cards.append(f'''<div class="card"><div class="h">#{k} · {e["meta"]["ts"][:16].replace("T", " ")}</div>{conv}
<div class="s r"><b>Alex (real reply):</b> {html.escape(e["messages"][-1]["content"]).replace(chr(10), "<br>")}</div>
<div class="t"><b>Thought written for you:</b> {html.escape(rs[i]["reasoning"])}</div>
<div class="o">{"".join(f'<label><input type="radio" name="{i}" value="{v}"> {v}</label>' for v in ("right", "partly", "wrong"))}</div></div>''')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Reasoning check</title><style>
body{{font-family:-apple-system,sans-serif;max-width:860px;margin:20px auto;padding:0 16px;background:#fafafa;color:#222}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px;margin:14px 0}} .h{{font-size:12px;color:#888}}
.v b{{color:#b0306a}} .s b{{color:#2a5db0}} .r{{background:#eef4ff;padding:6px;border-radius:6px;margin-top:6px}}
.t{{background:#fff7e6;padding:8px;border-radius:6px;margin-top:8px}} .o{{margin-top:8px;display:flex;gap:18px}}
.bar{{position:sticky;top:0;background:#fafafa;padding:8px 0}} button{{padding:8px 14px;font-size:15px}}</style></head><body>
<div class="bar"><b>{len(pick)} random replies from {a.frm} on.</b> Is the thought what was really going on in your head?
<button onclick="save()">Save votes</button> <span id="st"></span></div>{"".join(cards)}
<script>function save(){{const v={{}};document.querySelectorAll("input:checked").forEach(r=>v[r.name]=r.value);
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(v,null,1)],{{type:"application/json"}}));
a.download="reasoning_votes.json";a.click();document.getElementById("st").textContent="saved "+Object.keys(v).length+" votes";}}</script></body></html>'''
    p = L / "reasoning_review.html"
    p.write_text(page, encoding="utf-8")
    p.chmod(0o600)
    print(f"review page: {p} ({len(pick)} replies)")


if __name__ == "__main__":
    main()
