import os,json,gc,html,re
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
root=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))));out=root/'weight_edit_audit'
# New synthetic multi-turn cases, not used to choose interpolation fractions.
cases=[('weekday', [('user', 'Our workshop is on Thursday. What day is the workshop?'), ('assistant', 'I will keep that in mind.'), ('user', 'Our workshop is on Thursday. What day is the workshop?')], 'thursday'), ('budget', [('user', 'A notebook costs 40 and a pen costs 10. What is the total?'), ('assistant', 'I will keep that in mind.'), ('user', 'A notebook costs 40 and a pen costs 10. What is the total?')], '50'), ('color', [('user', 'The folder is blue. What color is the folder?'), ('assistant', 'I will keep that in mind.'), ('user', 'The folder is blue. What color is the folder?')], 'blue')]
results=[]
for name in ['B_think','B_weight_edit_025']:
 m,t=load(str(root/'models'/name/'mlx'),tokenizer_config={'trust_remote_code':True})
 rows=[]
 for key,history,expected in cases:
  mx.random.seed(123)
  messages=[{'role':'system','content':'You are a friendly conversational assistant. Respond thoughtfully and naturally. Ask a relevant follow-up when useful.'}]+[{'role':r,'content':c} for r,c in history]
  p=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
  raw=generate(m,t,prompt=p,max_tokens=300,sampler=make_sampler(temp=.7,top_p=.9),verbose=False)
  closed='</think>' in raw;reply=raw.split('</think>',1)[-1].strip()
  match=closed and reply.casefold().strip(' .!\n')==expected.casefold()
  rows.append(dict(case=key,output=raw,expected=expected,exact_match=match,closed=closed,tokens=len(t.encode(raw))))
 results.append(dict(model=name,rows=rows))
 print(json.dumps({'model':name,'exact_matches':sum(r['exact_match'] for r in rows),'closed':sum(r['closed'] for r in rows),'cases':len(rows)}),flush=True)
 del m,t;gc.collect();mx.clear_cache()
(out/'holdout_results.json').write_text(json.dumps(results,indent=2))
page=['<meta charset="utf-8"><title>Held-out speaker and context checks</title><h1>Held-out synthetic checks</h1><p>Exact-match tests check constrained answers, not natural conversation quality.</p>']
for result in results:
 page.append('<h2>'+result['model']+'</h2>')
 for row in result['rows']:page.append('<h3>'+row['case']+'</h3><p>Expected: '+html.escape(row['expected'])+'</p><pre style="white-space:pre-wrap">'+html.escape(row['output'])+'</pre>')
(out/'holdout_review.html').write_text('\n'.join(page))
