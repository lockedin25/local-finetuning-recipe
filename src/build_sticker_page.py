"""Build a LOCAL labelling page for the most-sent stickers (opened from disk, never published).

For each sticker (ranked by how often it was sent): the sticker itself, who sends it, the vision model's
guess, and the messages that most often come right before it, so Alex can write what it MEANS
between them (e.g. 'kiss', 'sorry baby', 'goodnight'). Labels are saved with a button that downloads
sticker_labels.json (and kept in the browser while editing).

Prints counts only.
  python3 -I src/build_sticker_page.py --export "<export folder>" --desc <media_descriptions.jsonl> --out <page.html> [--top 100]
"""
import argparse
import collections
import html
import json
import re
import zipfile
from pathlib import Path

LINE = re.compile(r"^\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}[^-]*- ([^:\n]+): ?(.*)$")
STK = re.compile(r"(STK-\S+\.(?:webp|was)) \(file attached\)")
ME = "Alex"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()
    exp = Path(a.export).expanduser()
    chat = next(exp.glob("*.txt"))
    rows = {json.loads(l)["file"]: json.loads(l) for l in Path(a.desc).expanduser().open()}

    msgs = []
    for raw in chat.read_text(encoding="utf-8").splitlines():
        m = LINE.match(raw)
        if m:
            msgs.append((m.group(1), m.group(2)))

    # group sticker files by content hash (same sticker can arrive under different file names)
    key = lambda f: rows[f]["sha1"] if f in rows else f
    sends, who, before, files = collections.Counter(), collections.defaultdict(collections.Counter), \
        collections.defaultdict(collections.Counter), {}
    for i, (sender, text) in enumerate(msgs):
        m = STK.search(text)
        if not m:
            continue
        f = m.group(1)
        k = key(f)
        files.setdefault(k, f)
        sends[k] += 1
        who[k]["me" if sender == ME else "her"] += 1
        for j in range(i - 1, max(i - 6, -1), -1):           # nearest earlier text from the other person
            s2, t2 = msgs[j]
            if s2 != sender and t2.strip() and "(file attached)" not in t2:
                before[k][t2.strip().lower()[:60]] += 1
                break

    ranked = sends.most_common(a.top)
    total = sum(sends.values())
    cards = []
    for rank, (k, n) in enumerate(ranked, 1):
        f = files[k]
        guess = rows.get(f, {}).get("desc", "")
        if f.endswith(".was"):
            anim = zipfile.ZipFile(exp / f).read("animation/animation.json").decode("utf-8")
            media = f'<div class="lot" data-anim="{html.escape(anim)}"></div>'
            guess = guess or "(animated sticker)"
        else:
            media = f'<img src="file://{html.escape(str(exp / f))}">'
        ctx = "".join(f"<li>{html.escape(t)} <span>×{c}</span></li>" for t, c in before[k].most_common(3))
        cards.append(f'''<div class="card" data-key="{k}" data-file="{html.escape(f)}">
  <div class="m">{media}</div>
  <div class="b"><div class="h">#{rank} · sent {n}× (you {who[k]["me"]}, her {who[k]["her"]})</div>
  <div class="g">model's guess: {html.escape(guess)}</div>
  <div class="c">usually right after:<ul>{ctx or "<li>-</li>"}</ul></div>
  <input placeholder="what it means between you (e.g. kiss, sorry baby, goodnight)"></div></div>''')

    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Sticker labels</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
<style>body{{font-family:-apple-system,sans-serif;max-width:900px;margin:20px auto;padding:0 16px;background:#fafafa}}
.card{{display:flex;gap:16px;background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px;margin:10px 0}}
.m img,.lot{{width:140px;height:140px;object-fit:contain}} .b{{flex:1}} .h{{font-weight:600}} .g{{color:#666;margin:4px 0}}
.c{{font-size:13px;color:#444}} .c span{{color:#999}} ul{{margin:4px 0 8px 18px;padding:0}}
input{{width:100%;padding:8px;font-size:15px;box-sizing:border-box}} .bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:2}}
button{{padding:8px 14px;font-size:15px}}</style></head><body>
<div class="bar"><b>Top {len(ranked)} stickers = {100 * sum(n for _, n in ranked) // total}% of all {total} sticker sends.</b>
Leave blank to keep the model's guess. <button onclick="save()">Save labels</button> <span id="st"></span></div>
{"".join(cards)}
<script>
const K='sticker_labels_v1';let saved={{}};try{{saved=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
document.querySelectorAll('.card').forEach(c=>{{const i=c.querySelector('input');i.value=saved[c.dataset.key]||'';
 i.oninput=()=>{{saved[c.dataset.key]=i.value;try{{localStorage.setItem(K,JSON.stringify(saved))}}catch(e){{}}}}}});
document.querySelectorAll('.lot').forEach(d=>{{try{{lottie.loadAnimation({{container:d,renderer:'svg',loop:true,autoplay:true,animationData:JSON.parse(d.dataset.anim)}})}}catch(e){{d.textContent='(animated sticker)'}}}});
function save(){{const out={{}};document.querySelectorAll('.card').forEach(c=>{{const v=c.querySelector('input').value.trim();if(v)out[c.dataset.key]={{file:c.dataset.file,label:v}}}});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}}));
 a.download='sticker_labels.json';a.click();document.getElementById('st').textContent='saved '+Object.keys(out).length+' labels to Downloads';}}
</script></body></html>'''
    Path(a.out).expanduser().write_text(page, encoding="utf-8")
    print(f"{len(ranked)} stickers on the page, covering {100 * sum(n for _, n in ranked) / total:.0f}% of {total} sends; "
          f"animated: {sum(files[k].endswith('.was') for k, _ in ranked)}")


if __name__ == "__main__":
    main()
