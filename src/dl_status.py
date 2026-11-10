"""Live download tracker for the bake-off models.

Usage:  uv run python src/dl_status.py          (refreshes every 5 s, Ctrl+C to quit)
        uv run python src/dl_status.py --once   (print once and exit)
Reads only file sizes on disk; expected sizes come from the Hugging Face API (cached).
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUB = Path.home() / ".cache/huggingface/hub"
SIZES = ROOT / "data/.model_sizes.json"
REPOS = [
    "mlx-community/gemma-4-12B-it-4bit",
    # "mlx-community/Qwen3.5-9B-4bit",          # fallback 1
    # "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",  # fallback 2
    # "mlx-community/sarvam-1-4bit",
]


def expected_sizes():
    if SIZES.exists():
        return json.loads(SIZES.read_text())
    from huggingface_hub import HfApi
    api, out = HfApi(), {}
    for r in REPOS:
        info = api.model_info(r, files_metadata=True)
        out[r] = sum(s.size or 0 for s in info.siblings)
    SIZES.parent.mkdir(exist_ok=True)
    SIZES.write_text(json.dumps(out))
    return out


def on_disk(repo):
    d = HUB / ("models--" + repo.replace("/", "--")) / "blobs"
    if not d.exists():
        return 0, False
    total, partial = 0, False
    for f in d.iterdir():
        total += f.stat().st_size
        partial |= f.name.endswith(".incomplete")
    return total, partial


def fmt(b):
    return f"{b / 1e9:5.2f} GB"


def main():
    exp = expected_sizes()
    prev, prev_t = None, None
    while True:
        now = {r: on_disk(r) for r in REPOS}
        t = time.time()
        total_now = sum(v[0] for v in now.values())
        speed = (total_now - prev) / (t - prev_t) if prev is not None else 0
        if "--once" not in sys.argv:
            os.system("clear")
        print(f"Model downloads   {time.strftime('%H:%M:%S')}   speed {speed / 1e6:5.2f} MB/s\n")
        remaining = 0
        for r in REPOS:
            got, partial = now[r]
            e = exp.get(r, 0) or 1
            pct = min(100, 100 * got / e)
            bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
            state = "downloading" if partial else ("done" if pct >= 99.5 else ("waiting" if got < 50e6 else "partial"))
            remaining += max(0, e - got)
            print(f"  {r.split('/')[1]:34} {bar} {pct:5.1f}%  {fmt(got)} / {fmt(e)}  {state}")
        eta = f"{remaining / speed / 60:5.0f} min" if speed > 1e4 else "   -   "
        print(f"\n  remaining {fmt(remaining)}   ETA at current speed: {eta}")
        log = ROOT / "downloads.log"
        if log.exists():
            print("\n  log: " + "\n       ".join(log.read_text().strip().splitlines()[-3:]))
        if "--once" in sys.argv:
            return
        prev, prev_t = total_now, t
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
