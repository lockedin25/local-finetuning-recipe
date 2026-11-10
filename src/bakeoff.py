"""Module 4: score candidate base models on how well they ALREADY predict Alex's replies.

Metric: bits per character (BPC) of his reply given the context, on 500 test examples.
Per-character (not per-token) so models with different tokenizers are compared fairly.
Lower = the model is less surprised by how he texts = better starting point.
Runs one model at a time, writes results to bakeoff_results.json. No text is printed.
"""
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parent))
from encode import encode_example

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bakeoff_results.json"
N = 500

CANDIDATES = [
    # name, repo, chat-template kwargs (None = no chat template: plain-text format)
    ("Gemma-4-12B", "mlx-community/gemma-4-12B-it-4bit", {"enable_thinking": False}),
    ("Qwen3.5-9B", "mlx-community/Qwen3.5-9B-4bit", {"enable_thinking": False}),
    ("Llama-3.1-8B", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", {}),
    ("LFM3-1.7B", "LiquidAI/LFM3-1.7B-MLX-bf16", {"enable_thinking": False}),   # run via src/lfm3.sh
    ("Sarvam-1-2.5B (base)", "mlx-community/sarvam-1-4bit", None),
]


def encode_plain(tok, messages):
    """For base models without a chat template: 'Her: ... / Alex: ...' transcript."""
    name = {"user": "Her", "assistant": "Alex"}
    head = messages[0]["content"] + "\n\n"
    head += "".join(f"{name[m['role']]}: {m['content']}\n" for m in messages[1:-1]) + "Alex:"
    p = tok.encode(head)
    t = tok.encode(head + " " + messages[-1]["content"] + "\n")
    d = next((i for i, (a, b) in enumerate(zip(t, p)) if a != b), len(p))
    return p + t[d:], len(p)


def reply_token_count(tok, toks, off, reply):
    """Number of target tokens that make up the reply itself (excluding the end marker)."""
    target = toks[off:]
    for k in range(len(target), 0, -1):
        if tok.decode(target[:k]).strip() == reply.strip():
            return k
    return len(target)  # fallback: count everything


def score(name, repo, kw, examples):
    t0 = time.time()
    model, tok = load(repo)
    load_s = time.time() - t0
    mx.reset_peak_memory()
    nll, chars, ntok, t1 = 0.0, 0, 0, time.time()
    total_tokens = 0
    for e in examples:
        msgs, reply = e["messages"], e["messages"][-1]["content"]
        toks, off = encode_plain(tok, msgs) if kw is None else encode_example(tok, msgs, kw)
        k = reply_token_count(tok, toks, off, reply)
        x = mx.array(toks)[None]
        logits = model(x[:, :-1]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = x[:, 1:]
        tok_lp = mx.take_along_axis(logp, tgt[..., None], axis=-1)[0, :, 0]
        # targets for reply tokens are positions off .. off+k-1 in toks -> index off-1 .. off+k-2 in tok_lp
        nll -= tok_lp[off - 1: off - 1 + k].sum().item()
        chars += len(reply)
        ntok += k
        total_tokens += len(toks)
    secs = time.time() - t1
    res = {
        "repo": repo,
        "bits_per_char": nll / chars / math.log(2),
        "per_token_ppl": math.exp(nll / ntok),
        "tokens_per_char": ntok / chars,
        "tokens_per_sec_forward": total_tokens / secs,
        "peak_mem_gb": mx.get_peak_memory() / 1e9,
        "load_s": load_s,
        "eval_s": secs,
    }
    del model, tok
    gc.collect()
    mx.clear_cache()
    return res


def main():
    rows = [json.loads(l) for l in (ROOT / "data/final/test.jsonl").open(encoding="utf-8")]
    examples = random.Random(7).sample(rows, N)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    only = set(sys.argv[1:])
    for name, repo, kw in CANDIDATES:
        if (only and name not in only) or name in results:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {name}: loading (downloads on first use)...", flush=True)
        try:
            results[name] = score(name, repo, kw, examples)
            r = results[name]
            print(f"   BPC {r['bits_per_char']:.3f} | ppl/token {r['per_token_ppl']:.1f} | tok/char {r['tokens_per_char']:.3f} | "
                  f"{r['tokens_per_sec_forward']:.0f} tok/s | peak {r['peak_mem_gb']:.1f} GB", flush=True)
        except Exception as ex:
            results[name] = {"repo": repo, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
            print(f"   FAILED: {results[name]['error']}", flush=True)
        OUT.write_text(json.dumps(results, indent=2))
    print("done")


if __name__ == "__main__":
    main()
