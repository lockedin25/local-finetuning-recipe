"""Build a private, offline review page. Prints counts only; never document contents."""

import os
from datetime import datetime
from html import escape
from pathlib import Path
import re

R = (Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'v2/reasoning')


def build():
    notes = sorted((R / 'notes').glob('*.md'))
    memories = sorted(p for p in (R / 'memory').glob('after_*.md') if re.fullmatch(r'after_\d{4}-\d{2}', p.stem))
    parts = list((R / 'memory').glob('after_*_p*.md'))
    months = sorted(p.stem for p in (R / 'months').glob('*.md'))
    sections = []
    links = []
    for month in months:
        files = [p for p in notes if p.stem.startswith(month + '_p')]
        snapshot = R / 'memory' / ('after_' + month + '.md')
        card = ''
        memory = ''
        if snapshot.exists():
            text = snapshot.read_text(encoding='utf-8')
            memory, sep, card = text.partition('=== CARD ===')
            memory = memory.replace('=== MEMORY ===', '').strip()
            card = card.strip() if sep else ''
        previews = []
        full = []
        for p in files:
            text = p.read_text(encoding='utf-8')
            # Deterministic excerpts, not a new model summary or a factual verification.
            bullets = [line.strip() for line in text.splitlines() if re.match(r'^\s*(?:[-*] |\d+[.)] )', line)]
            if not bullets:
                bullets = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#')]
            preview = '\n'.join(bullets[:5])
            previews.append('<h4>' + escape(p.stem) + '</h4><pre>' + escape(preview) + '</pre>')
            full.append('<details><summary>Full notes: ' + escape(p.stem) + '</summary><pre>' + escape(text) + '</pre></details>')
        links.append(f'<a href="#m{month}">{month}</a>')
        body = '<h3>Memory card</h3><pre>' + escape(card) + '</pre>' if card else '<p class="muted">Memory card not ready yet.</p>'
        if memory:
            body += '<details><summary>Cumulative memory</summary><pre>' + escape(memory) + '</pre></details>'
        body += '<h3>Note excerpts</h3>' + (''.join(previews) or '<p class="muted">Notes not ready yet.</p>')
        body += ''.join(full)
        sections.append(f'<section id="m{month}"><h2>{month}</h2><p class="muted">{len(files)} note parts available</p>{body}</section>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Private memory review</title><style>
body{font-family:system-ui,sans-serif;background:#f4f5f7;color:#20242b;max-width:940px;margin:32px auto;padding:0 20px;line-height:1.6}
section,header{background:white;padding:24px;border:1px solid #ddd;border-radius:12px;margin-bottom:20px}
h1,h2,h3{line-height:1.25}h3{margin-top:26px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f7f8fa;padding:16px;border-radius:8px}
nav{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}a{color:#2658a0}.muted{color:#656d79}summary{cursor:pointer;font-weight:600;padding:12px 0}
</style></head><body><header><h1>Private memory review</h1>
<p>This page stays on your Mac and loads no external resources.</p>
'''
    page += f'<p><b>{len(notes)} / 37 note parts</b> · <b>{len(parts)} / 37 memory merges</b> · <b>{len(memories)} / {len(months)} monthly snapshots</b></p>'
    page += '<p class="muted">Built ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '. This is a snapshot; regenerate it to include newer outputs.</p>'
    page += '<p>Cards summarize cumulative context at the end of each month. Note excerpts show the first five bullet lines from each part; expand full notes to read the rest. These are generated outputs for your review and may contain mistakes.</p></header><nav>' + ''.join(links) + '</nav>' + ''.join(sections) + '</body></html>'
    target = R / 'memory_summary.html'
    target.write_text(page, encoding='utf-8')
    target.chmod(0o600)
    print(f'Summary ready: {len(notes)} note parts, {len(parts)} memory merges, {len(memories)} monthly snapshots.')
    print(f'Local page: {target}')


if __name__ == '__main__':
    build()
