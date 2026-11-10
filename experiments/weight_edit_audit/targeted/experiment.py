import os,json,gc,shutil,re,html,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten
R=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone")))); O=R/'weight_edit_audit/targeted';O.mkdir(exist_ok=True)
B=R/'models/B_think/mlx'; BASE=next((Path.home()/'.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B-MLX-bf16/snapshots').iterdir())
# Entirely synthetic. Positive and negative continuations distinguish grounding, ownership and intent.
dev=[('The folder is blue. What color is it?', 'blue', 'red'), ('What is forty plus ten?', 'fifty', 'ninety'), ('The workshop is on Thursday. What day is it?', 'Thursday', 'Monday')]
held=[('The box is circular. What shape is it?', 'circular', 'square'), ('What is two plus three?', 'five', 'nine'), ('The meeting begins at ten. When does it begin?', 'ten', 'eight')]
system='Write the next casual Hinglish text from Alex to Sam. Respond to her actual message, keep speakers distinct, and do not invent events or activities. Keep it short and natural.'
m,t=load(str(B),tokenizer_config={'trust_remote_code':True})
bw=mx.load(str(B/'model.safetensors'));ow={}
for f in BASE.glob('*.safetensors'):ow.update(mx.load(str(f)))
ow=m.sanitize(ow);assert bw.keys()==ow.keys()
for k in bw:assert bw[k].shape==ow[k].shape

def chosen(k,kind):
 return (kind=='attention' and '.self_attn.' in k) or (kind=='mlp' and '.feed_forward.' in k) or (kind=='conv' and '.conv.' in k)
def setvariant(kind):
 vals=[]
 for k,v in bw.items():
  if kind=='base' or chosen(k,kind):v=ow[k]
  elif kind=='blend025':v=(ow[k].astype(mx.float32)*.75+v.astype(mx.float32)*.25).astype(v.dtype)
  vals.append((k,v))
 m.load_weights(vals);mx.eval(m.parameters())
def score(prompt,answer):
 ids=t.apply_chat_template([{'role':'system','content':system},{'role':'user','content':prompt}],add_generation_prompt=True,enable_thinking=False)
 ans=t.encode(answer,add_special_tokens=False);x=mx.array([ids+ans]);logits=m(x[:,:-1]).astype(mx.float32)
 lp=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
 labels=x[:,1:];v=mx.take_along_axis(lp,labels[:,:,None],axis=-1).squeeze(-1)
 return float(mx.mean(v[:,len(ids)-1:]).item())
results=[]
for variant in ['B','blend025','base','attention','mlp','conv']:
 setvariant(variant);rows=[]
 for i,(q,pos,neg) in enumerate(dev):
  a=score(q,pos);b=score(q,neg);rows.append({'case':i,'margin':a-b})
 rec={'variant':variant,'dev_mean_margin':sum(r['margin'] for r in rows)/len(rows),'dev_preferences':sum(r['margin']>0 for r in rows),'dev':rows};results.append(rec)
 (O/'scores.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v for k,v in rec.items() if k!='dev'}),flush=True)
best=max([r for r in results if r['variant'] in ['attention','mlp','conv']],key=lambda r:r['dev_mean_margin'])['variant']
for variant in ['B','blend025',best]:
 setvariant(variant);rows=[]
 for i,(q,pos,neg) in enumerate(held):rows.append({'case':i,'margin':score(q,pos)-score(q,neg)})
 rec=next(r for r in results if r['variant']==variant);rec['heldout']=rows;rec['heldout_mean_margin']=sum(r['margin'] for r in rows)/len(rows);rec['heldout_preferences']=sum(r['margin']>0 for r in rows)
 print(json.dumps({'variant':variant,'heldout_mean_margin':rec['heldout_mean_margin'],'heldout_preferences':rec['heldout_preferences']}),flush=True)
(O/'scores.json').write_text(json.dumps(results,indent=2))
setvariant(best)
# Save as an experiment, never replace an existing model or launcher.
D=R/'models'/('B_targeted_'+best)/'mlx';D.mkdir(parents=True,exist_ok=True)
mx.save_safetensors(str(D/'model.safetensors'),dict(tree_flatten(m.parameters())))
for f in B.iterdir():
 if f.is_file() and f.suffix in ('.json','.jinja','.py') and f.name!='model.safetensors.index.json':shutil.copyfile(f,D/f.name)
keys=[k for k in bw if chosen(k,best)]
meta={'edit':'Restore original base weights for selected module family; retain original B elsewhere','family':best,'tensor_names':keys,'selected_parameters':sum(bw[k].size for k in keys),'status':'experimental; likelihood proxy is not conversation-quality validation','checkpoint':str(D)}
(O/'edit.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'saved':str(D),'family':best,'tensors':len(keys),'parameters':meta['selected_parameters']}),flush=True)
# Actual generation in the intended thinking mode. Outputs stay local.
page=['<meta charset="utf-8"><h1>Targeted weight edit: held-out conversations</h1><p>These outputs need human review; probability scores do not prove coherence.</p>'];gens=[]
for variant in ['B',best]:
 setvariant(variant);page.append('<h2>'+variant+'</h2>')
 for i,(q,_,_) in enumerate(held):
  mx.random.seed(81)
  ids=t.apply_chat_template([{'role':'system','content':system},{'role':'user','content':q}],add_generation_prompt=True,enable_thinking=True)
  raw=generate(m,t,prompt=ids,max_tokens=240,sampler=make_sampler(temp=.7,top_p=.9),verbose=False)
  gens.append({'variant':variant,'case':i,'output':raw,'closed_think':'</think>' in raw,'tokens':len(t.encode(raw))})
  page.append('<h3>'+html.escape(q)+'</h3><pre style="white-space:pre-wrap">'+html.escape(raw)+'</pre>')
 (O/'generations.json').write_text(json.dumps(gens,indent=2));print(json.dumps({'generation_variant':variant,'completed':sum(r['closed_think'] for r in gens if r['variant']==variant),'cases':len(held)}),flush=True)
(O/'review.html').write_text('\n'.join(page))
