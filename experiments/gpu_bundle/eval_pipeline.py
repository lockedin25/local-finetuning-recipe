"""Thinker + texter pipeline test: the THINK model writes a thought (greedy, <=60 tokens), the TEXTER gets it in its
system prompt and the real reply is scored. Same 500 test examples as before. Prints numbers only."""

import os
import json, math, random, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from encode import encode_example
H = Path(os.environ.get("LOCAL_FINETUNING_GPU_HOME", str(Path(__file__).resolve().parent)))
thinker_p, texter_p = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(thinker_p, trust_remote_code=True)
load = lambda p: AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
thinker, texter = load(thinker_p), load(texter_p)
rows = [json.loads(l) for l in (H / "data/think/test.jsonl").open(encoding="utf-8")]
sample = random.Random(7).sample(rows, 500)
end_id = tok.convert_tokens_to_ids("</think>")
nll = nll0 = 0.0; chars = 0
with torch.no_grad():
    for r in sample:
        msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in r["messages"]]
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, enable_thinking=True, return_dict=False)
        g = thinker.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=60, do_sample=False,
                             eos_token_id=end_id, pad_token_id=tok.eos_token_id)[0, len(prompt):].tolist()
        thought = tok.decode([t for t in g if t != end_id], skip_special_tokens=True).strip()
        reply = msgs[-1]["content"]
        k = len(tok(reply, add_special_tokens=False)["input_ids"])
        for with_thought in (True, False):
            m2 = [dict(m) for m in msgs]
            if with_thought and thought:
                m2[0]["content"] += "\n\nWhat I'm thinking right now: " + thought
            ids, off = encode_example(tok, m2, {"enable_thinking": False})
            x = torch.tensor([ids], device="cuda")
            lp = torch.log_softmax(texter(input_ids=x).logits[0, :-1].float(), -1).gather(1, x[0, 1:, None])[:, 0]
            v = -lp[off - 1: off - 1 + k].sum().item()
            if with_thought: nll += v
            else: nll0 += v
        chars += len(reply)
b = lambda x: x / chars / math.log(2)
print(f"pipeline test BPC (thinker thought -> texter) {b(nll):.3f} | texter WITHOUT a thought {b(nll0):.3f}")
