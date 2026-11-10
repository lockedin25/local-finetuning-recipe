import os,json,gc,shutil,html
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
root=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))); out=root/'weight_edit_audit';out.mkdir(exist_ok=True)
b=root/'models/B_think/mlx'
base=next((Path.home()/'.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-MLX-bf16/snapshots').iterdir())
cases=[('weekday', 'Our workshop is on Thursday. What day is the workshop?', 'thursday'), ('budget', 'A notebook costs 40 and a pen costs 10. What is the total?', '50'), ('color', 'The folder is blue. What color is the folder?', 'blue')]
# Synthetic content only; outputs remain local in case the fine-tune recalls private material.
results=[]
for alpha in [1.0,0.0,0.25,0.5]:
 m,t=load(str(b if alpha else base),tokenizer_config={'trust_remote_code':True})
 if alpha not in (0.,1.):
  bw=mx.load(str(b/'model.safetensors')); ow={}
  for f in base.glob('*.safetensors'): ow.update(mx.load(str(f)))
  ow=m.sanitize(ow)
  assert bw.keys()==ow.keys(), 'Tensor names differ; cannot interpolate safely'
  edits=[]
  for k,v in bw.items():
   assert v.shape==ow[k].shape
   edits.append((k,(ow[k].astype(mx.float32)*(1-alpha)+v.astype(mx.float32)*alpha).astype(v.dtype)))
  m.load_weights(edits);mx.eval(m.parameters());del bw,ow,edits;gc.collect();mx.clear_cache()
 rows=[]
 for i,(name,prompt,expected) in enumerate(cases):
  mx.random.seed(42)
  p=t.apply_chat_template([{'role':'system','content':'You are a friendly conversational assistant. Respond thoughtfully and naturally. Ask a relevant follow-up when useful.'},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
  raw=generate(m,t,prompt=p,max_tokens=300,sampler=make_sampler(temp=.7,top_p=.9),verbose=False)
  reply=raw.split('</think>',1)[-1].strip();words=reply.lower().split(); grams=[tuple(words[j:j+4]) for j in range(max(0,len(words)-3))]
  repeated= bool(grams) and len(set(grams))/len(grams)<.6
  rows.append(dict(case=name,output=raw,tokens=len(t.encode(raw)),closed='</think>' in raw,repeated=repeated,expected=expected,match=(expected in reply.lower()) if expected else None))
 record={'alpha':alpha,'rows':rows}; results.append(record)
 (out/'interpolation_results.json').write_text(json.dumps(results,indent=2))
 print(json.dumps({'alpha':alpha,'closed':sum(r['closed'] for r in rows),'repetition_flags':sum(r['repeated'] for r in rows),'keyword_checks':sum(r['match'] is True for r in rows),'max_tokens':max(r['tokens'] for r in rows)}),flush=True)
 if alpha==.25:
  candidate=root/'models/B_weight_edit_025/mlx';candidate.mkdir(parents=True,exist_ok=True)
  from mlx.utils import tree_flatten
  mx.save_safetensors(str(candidate/'model.safetensors'),dict(tree_flatten(m.parameters())))
  for f in b.iterdir():
   if f.is_file() and f.suffix in ('.json','.jinja','.py') and f.name!='model.safetensors.index.json':shutil.copyfile(f,candidate/f.name)
  (candidate.parent/'EDIT.json').write_text(json.dumps({'formula':'W_candidate = W_base + 0.25 * (W_B - W_base)','original':str(b),'base':str(base),'status':'experimental; not promoted; semantic review pending'},indent=2))
 del m,t;gc.collect();mx.clear_cache()
page=['<meta charset="utf-8"><title>Model B weight edit review</title><h1>Synthetic comparison</h1><p>Alpha is the retained fraction of B fine-tuning. Keyword checks are not semantic quality scores.</p>']
for x in results:
 page.append(f'<h2>Alpha {x["alpha"]}</h2>')
 for r in x['rows']:page.append('<h3>'+html.escape(r['case'])+'</h3><pre style="white-space:pre-wrap">'+html.escape(r['output'])+'</pre>')
(out/'review.html').write_text('\n'.join(page))
