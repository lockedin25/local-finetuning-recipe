"""Runs ON THE GPU VM. For N fixed test conversations, every setup writes its reply (batched generation):
  A  plain (no thinking)          B  thinks, then replies
  BC B writes the thought, C texts   DC D writes the thought, C texts
Output: out_eval/generations.jsonl  ({id, real, A, B, B_thought, BC, BC_thought, DC, DC_thought}). Prints counts only.
  python gen_eval.py --n 300
"""

import os
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = Path(os.environ.get("LOCAL_FINETUNING_GPU_HOME", str(Path(__file__).resolve().parent)))
R = H / "runs"
THOUGHT = "\n\nWhat I'm thinking right now: "


def load(name):
    p = R / name / "merged"
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    return m, tok


@torch.no_grad()
def gen(model, tok, prompts, max_new, stop_think=False, bs=16):
    """prompts: list of token-id lists. Returns decoded continuations (cut at </think> if stop_think)."""
    end_think = tok.convert_tokens_to_ids("</think>")
    eos = [tok.eos_token_id] + ([end_think] if stop_think else [])
    out = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        L = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, L - len(p):] = torch.tensor(p)
            att[j, L - len(p):] = 1
        g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=max_new, do_sample=True,
                           temperature=0.7, top_p=0.9, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        for row in g[:, L:].tolist():
            out.append(tok.decode([t for t in row if t not in (end_think,)], skip_special_tokens=True).strip())
    return out


def prompt(tok, msgs, think):
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think, return_dict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    torch.manual_seed(0)
    rows = [json.loads(l) for l in (H / "data/nothink/test.jsonl").open(encoding="utf-8")]
    sample = random.Random(13).sample(rows, a.n)
    ctxs = [r["messages"][:-1] for r in sample]
    res = [{"id": r["meta"]["rid"], "real": r["messages"][-1]["content"]} for r in sample]

    mA, tA = load("A_nothink")
    for r, t in zip(res, gen(mA, tA, [prompt(tA, c, False) for c in ctxs], 120)):
        r["A"] = t
    del mA
    print("A done", flush=True)

    mB, tB = load("B_think")
    # B's own reply: re-generate with the thought in place so the reply follows it
    thoughts_B = gen(mB, tB, [prompt(tB, c, True) for c in ctxs], 60, stop_think=True)
    replies_B = gen(mB, tB, [prompt(tB, c, True) + tB(t + "</think>", add_special_tokens=False)["input_ids"]
                             for c, t in zip(ctxs, thoughts_B)], 120)
    for r, t, rep in zip(res, thoughts_B, replies_B):
        r["B_thought"], r["B"] = t, rep
    print("B done", flush=True)

    mD, tD = load("D_thinker")
    thoughts_D = gen(mD, tD, [prompt(tD, c, True) for c in ctxs], 60, stop_think=True)
    del mD
    print("D thoughts done", flush=True)

    mC, tC = load("C_texter")
    for key, ths in (("BC", thoughts_B), ("DC", thoughts_D)):
        ps = []
        for c, t in zip(ctxs, ths):
            m2 = [dict(m) for m in c]
            if t:
                m2[0]["content"] += THOUGHT + t
            ps.append(prompt(tC, m2, False))
        for r, t, rep in zip(res, ths, gen(mC, tC, ps, 120)):
            r[key + "_thought"], r[key] = t, rep
    print("C done", flush=True)

    (H / "out_eval").mkdir(exist_ok=True)
    with open(H / "out_eval/generations.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    empty = {k: sum(not r[k].strip() for r in res) for k in ("A", "B", "BC", "DC")}
    print(f"GEN_DONE {len(res)} conversations | empty replies {empty}", flush=True)


if __name__ == "__main__":
    main()
