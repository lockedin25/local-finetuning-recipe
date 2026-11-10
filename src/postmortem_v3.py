"""Offline v3 audit. Private outputs stay in LocalData; stdout contains statistics only."""

import os
import argparse
import copy
import gc
import html
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from encode import encode_example

LOCAL = Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))
OUT = LOCAL / "postmortem_v3"
BASE = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-MLX-bf16/snapshots/52041852d5b5c59bfe120c075d2744ab97959d91"
CLONE_SYSTEM = "You are Alex, texting your girlfriend on WhatsApp. Reply exactly the way he does."
EXPLICIT_SYSTEM = "You are Alex. The user is Sam, your girlfriend. Assistant messages are yours; user messages are hers. Reply to her latest message naturally and attentively, using the facts in the conversation. Never swap who did what."
GENERIC_SYSTEM = "You are a helpful conversational assistant. Answer the user's latest message and remember the facts in this conversation."
NAMES = {"base": BASE, "A": LOCAL / "models/A_nothink/mlx", "B": LOCAL / "models/B_think/mlx", "C": LOCAL / "models/C_texter/mlx"}


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    p.chmod(0o600)


def structural():
    tok = AutoTokenizer.from_pretrained(str(LOCAL / "models/A_nothink/merged"), local_files_only=True, trust_remote_code=True)
    summary = {}
    rows = [json.loads(line) for line in (LOCAL / "v3/think/train.jsonl").open()]
    reply_words = [len(r["messages"][-1]["content"].split()) for r in rows]
    thought_words = [len((r["messages"][-1].get("reasoning_content") or "").split()) for r in rows]
    summary["data"] = {"train_rows": len(rows), "median_reply_words": statistics.median(reply_words),
                       "reply_le3_words": sum(n <= 3 for n in reply_words),
                       "median_thought_words": statistics.median(thought_words)}
    sample = random.Random(7).sample(rows, 500)
    nt = sum(len(tok(r["messages"][-1].get("reasoning_content") or "", add_special_tokens=False)["input_ids"]) for r in sample)
    nr = sum(len(tok(r["messages"][-1]["content"], add_special_tokens=False)["input_ids"]) for r in sample)
    summary["data"]["sample_thought_token_fraction"] = nt / (nt + nr)
    # Check every training example for corruption, loss masking, and target truncation.
    for variant in ["nothink", "think", "texter", "thinker"]:
        counts = Counter()
        lengths = []
        for row in rows:
            msgs = copy.deepcopy(row["messages"])
            if variant == "nothink":
                msgs[-1].pop("reasoning_content", None)
            if variant == "texter":
                thought = msgs[-1].pop("reasoning_content", "") or ""
                if thought:
                    msgs[0]["content"] += "\n\nWhat I'm thinking right now: " + thought
            if variant == "thinker":
                msgs[-1]["content"] = ""
            kw = {"enable_thinking": variant in ("think", "thinker")}
            full = tok.apply_chat_template(msgs, return_dict=False, **kw)
            prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, return_dict=False, **kw)
            ids, off = encode_example(tok, msgs, kw)
            # Ignore only the trailing whitespace that encode_example intentionally removes.
            counts["full_prefix_mismatch"] += full[:len(prompt)] != prompt
            counts["encoded_differs_from_full_prefix"] += ids != full[:len(ids)]
            counts["empty_target"] += off >= len(ids)
            counts["target_fully_cut_at_2560"] += off >= 2560
            counts["target_partially_cut_at_2560"] += off < 2560 < len(ids)
            counts["target_missing_eos"] += tok.eos_token_id not in ids[off:]
            counts["bad_roles"] += [m["role"] for m in msgs[1:]] != ["user" if i % 2 == 0 else "assistant" for i in range(len(msgs) - 1)]
            lengths.append(len(ids))
        summary[variant] = dict(counts) | {"examples": len(rows), "max_tokens": max(lengths), "median_tokens": statistics.median(lengths)}
        save("structural.json", summary)
        print("STRUCTURAL", variant, json.dumps(summary[variant]), flush=True)
    for split in ["train", "valid", "test"]:
        rs = [json.loads(line) for line in (LOCAL / "v3/think" / (split + ".jsonl")).open()]
        summary["data"][split + "_sessions"] = len({r["meta"]["session"] for r in rs})
    save("structural.json", summary)


CASES = [{'id': 'weekday', 'turns': [('user', 'Our workshop is on Thursday. Please remember the day.'), ('assistant', 'Thursday, got it.'), ('user', 'What day is the workshop?')], 'requires': ['thursday']}, {'id': 'budget', 'turns': [('user', 'The notebook costs 40 and the pen costs 10. What is the total?')], 'requires': ['50']}, {'id': 'clarify', 'turns': [('user', 'Can you move it to the other one?')], 'requires': []}, {'id': 'hinglish', 'turns': [('user', 'kal library chalein?')], 'requires': []}]











def record(model, tok, setup, case, system, think=False, temperature=0.0, extra_plan=""):
    msgs = [{"role": "system", "content": system + ("\n\nWhat I'm thinking right now: " + extra_plan if extra_plan else "")}]
    msgs += [{"role": role, "content": text} for role, text in case["turns"]]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
    mx.random.seed(7)
    t = time.monotonic()
    raw = generate(model, tok, prompt=ids, max_tokens=240 if think else 120,
                   sampler=make_sampler(temp=temperature, top_p=0.9))
    reply = raw.split("</think>")[-1].strip()
    lower = reply.lower()
    requirements = case.get("required", [])
    passed = all(term in lower for term in requirements)
    if case.get("one_of"):
        passed = any(term in lower for term in case["one_of"])
    objective = bool(requirements or case.get("one_of"))
    return {"setup": setup, "case": case["id"], "system": system, "turns": case["turns"],
            "output": raw, "reply": reply, "empty": not reply, "objective": objective,
            "pass": passed if objective else None, "seconds": round(time.monotonic() - t, 2),
            "output_tokens": len(tok.encode(raw)), "thought_closed": "</think>" in raw if think else None}


def render_review(results):
    blocks = ["<!doctype html><meta charset='utf-8'><title>Local v3 post-mortem review</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;background:#f4f4f4;padding:15px}article{border-top:1px solid #ccc;margin-top:30px}</style><h1>Controlled synthetic conversations</h1><p>All prompts are synthetic. These are diagnostics, not a warmth or personality score.</p>"]
    for r in results:
        text = "\n".join(f"{role}: {content}" for role, content in r["turns"])
        blocks.append(f"<article><h2>{html.escape(r['setup'])}: {html.escape(r['case'])}</h2><pre>{html.escape(text)}</pre><h3>Output</h3><pre>{html.escape(r['output'])}</pre><p>Objective check: {r['pass']}; seconds: {r['seconds']}</p></article>")
    p = OUT / "review.html"
    p.write_text("\n".join(blocks))
    p.chmod(0o600)


def benchmark(selected):
    results = []
    for name in selected:
        model, tok = load(str(NAMES[name]), tokenizer_config={"trust_remote_code": True})
        setups = [("generic", GENERIC_SYSTEM, False, 0.0), ("clone", CLONE_SYSTEM, False, 0.0),
                  ("explicit_roles", EXPLICIT_SYSTEM, False, 0.0)]
        if name in ["base", "B"]:
            setups += [("clone_think", CLONE_SYSTEM, True, 0.0)]
        for label, system, think, temp in setups:
            group = []
            for case in CASES:
                r = record(model, tok, name + ":" + label, case, system, think, temp)
                results.append(r)
                group.append(r)
                save("conversations.json", results)
            summary = {"objective_pass": sum(r["pass"] is True for r in group),
                       "objective_total": sum(r["objective"] for r in group),
                       "empty": sum(r["empty"] for r in group),
                       "median_words": statistics.median(len(r["reply"].split()) for r in group),
                       "unclosed_thoughts": sum(r["thought_closed"] is False for r in group)}
            print("BENCH", name, label, json.dumps(summary), flush=True)
        del model, tok
        gc.collect()
        mx.clear_cache()
    render_review(results)
    aggregate = {}
    for setup in sorted({r["setup"] for r in results}):
        group = [r for r in results if r["setup"] == setup]
        aggregate[setup] = {"objective_pass": sum(r["pass"] is True for r in group),
                            "objective_total": sum(r["objective"] for r in group),
                            "empty": sum(r["empty"] for r in group),
                            "median_words": statistics.median(len(r["reply"].split()) for r in group),
                            "unclosed_thoughts": sum(r["thought_closed"] is False for r in group)}
    save("benchmark_summary.json", aggregate)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["structural", "benchmark"])
    ap.add_argument("--models", nargs="+", default=["base", "A", "B", "C"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    structural() if args.phase == "structural" else benchmark(args.models)
