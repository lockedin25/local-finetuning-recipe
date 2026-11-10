"""Build the local reading page: uv run --with markdown python essay/build.py."""
from pathlib import Path
import html
import re
import markdown

ROOT = Path(__file__).resolve().parent
text = (ROOT / 'draft.md').read_text()
body = markdown.markdown(text, extensions=['tables', 'toc'])
headings = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
nav = ''.join(f'<a href="#{key}">{label}</a>' for key, label in headings)
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" /></p>', r'<figure><img alt="\1" src="\2" loading="lazy"><figcaption>\1</figcaption></figure>', body)
title = html.escape(text.splitlines()[0].removeprefix('# '))
page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · Essay draft</title>
<style>
:root{color-scheme:light;--paper:#faf9f6;--ink:#292824;--muted:#736f66;--line:#dfdcd4;--accent:#285b61}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;background:var(--paper);color:var(--ink);font:19px/1.75 Georgia,serif}
a{color:var(--accent);text-underline-offset:4px}a:focus-visible{outline:2px solid var(--accent);outline-offset:5px}
header{max-width:1160px;margin:auto;padding:32px 32px 23px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;font:12px/1.5 system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase}
header a{text-decoration:none}header span{color:var(--muted)}.layout{max-width:1160px;margin:60px auto;display:grid;grid-template-columns:210px minmax(0,760px);gap:70px;padding:0 32px}
aside{align-self:start;position:sticky;top:30px;font:12px/1.6 system-ui,sans-serif}aside p{text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 18px}aside nav{display:flex;flex-direction:column;gap:11px}aside a{color:var(--muted);text-decoration:none}aside a:hover{color:var(--accent)}main{min-width:0}
.note{font:13px/1.6 system-ui,sans-serif;padding:13px 17px;border-left:2px solid var(--accent);background:#efeee8;margin-bottom:35px;color:#56564e}.note a{white-space:nowrap}
h1{font-size:clamp(36px,5vw,55px);line-height:1.12;letter-spacing:-.035em;font-weight:400;margin:0 0 35px}h2{font-size:29px;line-height:1.25;letter-spacing:-.02em;font-weight:400;margin:60px 0 22px}p{margin:0 0 24px}figure{margin:38px -20px}figure img{display:block;width:100%;height:auto;background:white;border:1px solid var(--line)}figcaption{font:12px/1.5 system-ui,sans-serif;color:var(--muted);margin-top:9px}
table{width:100%;border-collapse:collapse;font:15px/1.6 system-ui,sans-serif;margin:30px 0}th{text-align:left;font-weight:600}th,td{padding:12px 10px;border-bottom:1px solid var(--line)}footer{margin-top:65px;border-top:1px solid var(--line);padding:22px 0;color:var(--muted);font:12px/1.8 system-ui,sans-serif}.skip{position:absolute;left:-1000px}.skip:focus{left:20px;top:10px;background:white;padding:10px}.progress{position:fixed;top:0;left:0;height:2px;background:var(--accent);width:0;z-index:2}
@media(max-width:1000px){.layout{grid-template-columns:160px minmax(0,1fr);gap:35px}body{font-size:18px}}
@media(max-width:720px){header{padding:23px 22px}.layout{display:block;margin:35px auto;padding:0 22px}aside{position:static;margin-bottom:35px}aside nav{display:none}aside p{margin:0}h2{margin-top:46px;font-size:26px}figure{margin:30px 0}body{font-size:18px}table{font-size:13px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body><a class="skip" href="#essay">Skip to essay</a><div class="progress" aria-hidden="true"></div>
<header><a href="index.html">Notes on a small model</a><span>November 2026 · Working draft</span></header>
<div class="layout"><aside aria-label="Essay contents"><p>Inside the experiment</p><nav>__NAV__</nav></aside>
<main id="essay"><div class="note">Reading version · <a href="draft.md">Markdown source</a></div>
__BODY__
<footer>Experiment notes, 6–9 November 2026. Figures use saved training records and aggregate results.<br>Writing reference: <a href="https://datavorous.com/writing/ohev/">Flirting with decision models</a>. <a href="research/research_notes.md">Research and evidence gaps</a>.</footer></main></div>
<script>const bar=document.querySelector('.progress');function update(){const available=document.documentElement.scrollHeight-innerHeight;bar.style.width=(available>0?100*scrollY/available:0)+'%'}addEventListener('scroll',update,{passive:true});addEventListener('resize',update);update();</script>
</body></html>'''
(ROOT / 'index.html').write_text(page.replace('__TITLE__', title).replace('__NAV__', nav).replace('__BODY__', body))
print(f'Built reading page with {len(headings)} sections.')
