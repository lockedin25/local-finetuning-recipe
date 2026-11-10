
import os
import sys, re
from pathlib import Path
R = (Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'v2/reasoning')
n = int(sys.argv[1]); start = int(sys.argv[2]) if len(sys.argv) > 2 else 0; count = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
t = (R / "chunks" / f"chunk_{n:03d}.md").read_text()
main = t[t.index("## Write reasoning for this part"):]
def short(m):
    inner = m.group(0)[1:-1]
    if inner.startswith("sticker:"):
        return "[stk:" + inner.split("|")[0][8:].strip() + "]"
    if ":" in inner:
        k, d = inner.split(":", 1); w = d.split()
        return f"[{k}: {' '.join(w[:9])}{'…' if len(w) > 9 else ''}]"
    return m.group(0)
lines = [re.sub(r"\[[^\]]*\]", short, l).replace("Sam:", "V:").replace("Alex", "S") for l in main.splitlines()[1:] if l.strip()]
out = lines[start:start + count]
print("\n".join(out)); print(f"--- lines {start}-{start+len(out)} of {len(lines)}", file=sys.stderr)
