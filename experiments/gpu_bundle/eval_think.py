"""Fair test score for the THINK model: it writes its own thought (greedy, <=60 tokens), then the real reply is
scored given that thought. Same 500 test examples as gpu_train.test_bpc. Prints numbers only."""

import os
import json, math, random, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
H = Path(os.environ.get("LOCAL_FINETUNING_GPU_HOME", str(Path(__file__).resolve().parent)))
m_path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(m_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(m_path, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
rows = [json.loads(l) for l in (H / "data/think/test.jsonl").open(encoding="utf-8")]
sample = random.Random(7).sample(rows, 500)
nll, chars, think_words, empty = 0.0, 0, [], 0
with torch.no_grad():
    for r in sample:
        msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in r["messages"]]
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, enable_thinking=True, return_dict=False)
        x = torch.tensor([prompt], device="cuda")
        end_id = tok.convert_tokens_to_ids("</think>")
        g = model.generate(x, max_new_tokens=60, do_sample=False, eos_token_id=end_id, pad_token_id=tok.eos_token_id)[0, len(prompt):].tolist()
        thought = tok.decode([t for t in g if t != end_id], skip_special_tokens=True).strip()
        think_words.append(len(thought.split())); empty += not thought
        reply = msgs[-1]["content"]
        full = prompt + tok(thought + "</think>" + reply, add_special_tokens=False)["input_ids"]
        k = len(tok(reply, add_special_tokens=False)["input_ids"])
        ids = torch.tensor([full], device="cuda")
        lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1).gather(1, ids[0, 1:, None])[:, 0]
        nll -= lp[len(full) - 1 - k: len(full) - 1].sum().item()
        chars += len(reply)
print(f"think-mode test BPC {nll / chars / math.log(2):.3f} | thoughts: median {sorted(think_words)[250]} words, empty {empty}/500")
