
import os
import json,os
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
p=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'models/B_think/mlx')
m,t=load(str(p))
prompts=['kal library chalein?', 'Please reply in one short sentence.', 'What is 40 plus 10?']
rows=[]
for i,prompt in enumerate(prompts):
 mx.random.seed(42)
 s=t.apply_chat_template([{'role':'system','content':'You are a friendly conversational assistant. Respond thoughtfully and naturally. Ask a relevant follow-up when useful.'},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
 out=generate(m,t,prompt=s,max_tokens=256,sampler=make_sampler(temp=0.7,top_p=0.9),verbose=False)
 rows.append({'case':i,'output':out,'tokens':len(t.encode(out))})
 print(json.dumps({'case':i,'tokens':rows[-1]['tokens'],'closed_think':'</think>' in out}),flush=True)
p=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'weight_edit_audit');p.mkdir(exist_ok=True)
(p/'synthetic_baseline.json').write_text(json.dumps(rows,indent=2))
