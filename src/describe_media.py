"""Describe the photos and stickers inside a WhatsApp 'Export chat -> Include media' zip with a LOCAL
vision model (LFM3-VL-2B on MLX). Nothing leaves the Mac: images are read from the zip straight into
memory (never unpacked to disk) and the descriptions go to one JSONL file.

Prints COUNTS ONLY. Never prints a description (Claude reads this terminal, and that would leave the Mac).

  uv run --no-project --python 3.12 --with "mlx-vlm>=0.7.11" python -I src/describe_media.py \
      --zip "~/Downloads/WhatsApp Chat with X.zip" --out <non-iCloud path>/media_descriptions.jsonl [--limit 10]

Output rows: {"file": "IMG-...jpg", "kind": "photo"|"sticker", "sha1": ..., "category": ..., "desc": ...}
Stickers are de-duplicated by content hash (the same sticker is sent many times; described once).
Safe to stop and restart: files already in --out are skipped.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

MODEL = "LiquidAI/LFM3-VL-2B-MLX-bf16"
CATEGORIES = ["selfie", "intimate", "people", "food", "place", "screenshot", "meme", "object", "document", "other"]
CAT_HINT = "document = receipts, bills, tickets, forms; screenshot = phone/computer screens"

CAT_DEFS = (
    "selfie = a person photographing themselves, clothed; "
    "intimate = ONLY if there is nudity, underwear/lingerie, or sexual content; ordinary clothed photos are NOT intimate; "
    "people = clothed people, not a selfie; document = receipts, bills, tickets, forms; "
    "screenshot = phone/computer screens; meme = jokes, text images; food; place; object; other"
)
PHOTO_PROMPT = (
    "Look carefully at this photo and answer in exactly two lines.\n"
    f"Line 1: one category from: {', '.join(CATEGORIES)}. Definitions: {CAT_DEFS}.\n"
    "Line 2: a plain, factual description in at most 30 words: who/what is in it, what they are doing or "
    "wearing, the setting, and any visible text. If there is nudity, say so plainly."
)
STICKER_PROMPT = (
    "This is a WhatsApp sticker. In at most 12 words, say what it shows and the feeling or message it "
    "conveys, as 'what it shows: message'. Example: 'cartoon bear blowing a kiss: kiss / love you'."
)
GIF_PROMPT = (
    "These are 3 frames (left to right: start, middle, end) from a short looping GIF sent in a WhatsApp chat. "
    "In at most 15 words, say what it shows and the feeling or message it conveys, as 'what it shows: message'. "
    "Example: 'cat slowly yawning and stretching: sleepy / chilling'."
)
VIDEO_PROMPT = (
    "These are 3 frames (left to right: start, middle, end) from a video. Answer in exactly two lines.\n"
    f"Line 1: one category from: {', '.join(CATEGORIES)}. Definitions: {CAT_DEFS}.\n"
    "Line 2: a plain, factual description in at most 30 words of what happens in the video, who is in it and "
    "the setting. Describe nudity or intimacy plainly if present."
)
VID_RE = re.compile(r"(?:^|/)VID-.*\.mp4$", re.I)
PHOTO_RE = re.compile(r"(?:^|/)IMG-.*\.(jpe?g|png|webp|heic)$", re.I)
STICKER_RE = re.compile(r"(?:^|/)STK-.*\.webp$", re.I)


def shrink(img, side=640):
    img = img.convert("RGB")
    img.thumbnail((side, side))
    return img


def video_frames(raw):
    """(has_audio, seconds, 3-frame strip image). The video goes to a temp file (ffmpeg needs to seek) that is
    deleted straight away."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as t:
        t.write(raw)
        t.flush()
        info = json.loads(subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
                                          "-show_format", t.name], capture_output=True, text=True).stdout or "{}")
        audio = any(st.get("codec_type") == "audio" for st in info.get("streams", []))
        dur = float(info.get("format", {}).get("duration", 0) or 0)
        frames = []
        for frac in (0.1, 0.5, 0.9):
            png = subprocess.run(["ffmpeg", "-v", "quiet", "-ss", f"{dur * frac:.2f}", "-i", t.name, "-frames:v", "1",
                                  "-f", "image2pipe", "-vcodec", "png", "pipe:1"], capture_output=True).stdout
            if png:
                frames.append(shrink(Image.open(io.BytesIO(png)), 360))
    if not frames:
        raise ValueError("no frames")
    strip = Image.new("RGB", (sum(f.width for f in frames) + 8 * (len(frames) - 1), max(f.height for f in frames)))
    x = 0
    for f in frames:
        strip.paste(f, (x, 0))
        x += f.width + 8
    return audio, dur, strip


def kind_of(n):
    return "sticker" if STICKER_RE.search(n) else "photo" if PHOTO_RE.search(n) else "video" if VID_RE.search(n) else None


def two_lines(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    first = re.sub(r"^line ?1:?\s*", "", lines[0].lower()) if lines else ""
    cat = next((c for c in CATEGORIES if c in first), "other")
    desc = re.sub(r"^line ?2:?\s*", "", " ".join(lines[1:]) if len(lines) > 1 else text, flags=re.I)
    return cat, desc[:300]


def one_line(text, n):
    t = text.splitlines()[0] if text else ""
    t = re.sub(r"^(these are \d+ frames from a gif showing|this (gif|sticker) shows|what it shows:)\s*", "", t, flags=re.I)
    return re.sub(r"\.?\s*message:\s*", ": ", t, flags=re.I).strip()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new descriptions (for a test run)")
    ap.add_argument("--files", nargs="*", default=None, help="only these file names (e.g. IMG-20261106-WA0000.jpg)")
    ap.add_argument("--only", choices=["photo", "sticker", "gif", "video", "all"], default="all",
                    help="gif = silent videos <= 15 s (WhatsApp sends GIFs as mp4); video = the rest")
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    r = str(out.resolve())
    if "Mobile Documents" in r or r.startswith(str(Path.home() / "Desktop")) or r.startswith(str(Path.home() / "Documents")):
        raise SystemExit("refusing: --out is inside an iCloud-synced folder (Desktop/Documents/iCloud Drive)")
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = [json.loads(l) for l in out.open()] if out.exists() else []
    done = {x["file"] for x in prev}
    seen_stk = {x["sha1"]: x for x in prev if x["kind"] == "sticker"}

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(MODEL)
    cfg = model.config

    def ask(prompt_text, img, max_tokens=90):
        prompt = apply_chat_template(processor, cfg, prompt_text, num_images=1)
        return generate(model, processor, prompt, image=[img], max_tokens=max_tokens, temperature=0.1,
                        repetition_penalty=1.05, verbose=False).text.strip()

    zf = zipfile.ZipFile(Path(a.zip).expanduser())
    want = {"photo": {"photo"}, "sticker": {"sticker"}, "gif": {"video"}, "video": {"video"},
            "all": {"photo", "sticker", "video"}}[a.only]
    names = [n for n in zf.namelist() if kind_of(n) in want and (a.files is None or Path(n).name in a.files)]
    todo = [n for n in names if Path(n).name not in done]
    kinds = Counter(kind_of(n) for n in names)
    print(f"zip has {dict(kinds)}; {len(done)} already in output, {len(todo)} to go", flush=True)

    cats, new, reused, failed, skipped, t0 = Counter(), 0, 0, 0, 0, time.time()
    with out.open("a") as f:
        for n in todo:
            raw = zf.read(n)
            kind = kind_of(n)
            sha = hashlib.sha1(raw).hexdigest()
            row = {"file": Path(n).name, "kind": kind, "sha1": sha}
            if kind == "sticker" and sha in seen_stk:
                row.update(category="sticker", desc=seen_stk[sha]["desc"], dup=True)
                reused += 1
            else:
                try:
                    if kind == "photo":
                        row["category"], row["desc"] = two_lines(ask(PHOTO_PROMPT, shrink(Image.open(io.BytesIO(raw)))))
                    elif kind == "sticker":
                        row.update(category="sticker", desc=one_line(ask(STICKER_PROMPT, shrink(Image.open(io.BytesIO(raw)))), 120))
                        seen_stk[sha] = row
                    else:
                        audio, dur, strip = video_frames(raw)
                        is_gif = not audio and dur <= 15
                        row.update(kind="gif" if is_gif else "video", seconds=round(dur, 1))
                        if (a.only == "gif" and not is_gif) or (a.only == "video" and is_gif):
                            skipped += 1
                            continue
                        if is_gif:
                            row.update(category="gif", desc=one_line(ask(GIF_PROMPT, strip), 150))
                        else:
                            row["category"], row["desc"] = two_lines(ask(VIDEO_PROMPT, strip))
                except Exception as e:
                    row.update(category="error", desc="", error=type(e).__name__)
                    failed += 1
                new += 1
                cats[f"{row['kind']}/{row['category']}"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (new + reused) % 50 == 0:
                rate = (time.time() - t0) / max(new, 1)
                print(f"  {new + reused}/{len(todo)}  ({rate:.1f}s per new item, "
                      f"~{rate * (len(todo) - new - reused) / 60:.0f} min left)", flush=True)
            if a.limit and new >= a.limit:
                break

    print(f"done: {new} described, {reused} repeat stickers reused, {failed} failed, {skipped} skipped (other "
          f"video type), {time.time() - t0:.0f}s", flush=True)
    print("categories:", dict(cats.most_common()), flush=True)


if __name__ == "__main__":
    main()
