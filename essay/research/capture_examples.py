"""Bounded offline synthetic chats. No private data, memory cards or API calls."""

import os
import gc,json,os,sys,hashlib
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
OUT=Path(__file__).resolve().parent
MODELS=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'models')
SYSTEM="You are Alex, texting your girlfriend on WhatsApp. Reply exactly the way he does."
CASES={'speaker': ['I am preparing a workshop. You are reviewing a spreadsheet. Who is preparing the workshop?', 'The workshop is on Thursday. What day is it?', 'The spreadsheet folder is blue. What color is the folder?'], 'memory': ['The club meets at the library at six. Remember the time.', 'Correction: the club now meets at seven. What is the updated time?', 'Where does the club meet?'], 'conversation': ['The puzzle club needs a name. Suggest one short name.', 'Make it about geometric shapes.', 'Use only two words.', 'Now suggest a different two-word name.']}



results=[]
configs=[('A_nothink',False),('B_think',True),('C_texter',False),('B_weight_edit_025',True)]
if '--c-only' in sys.argv:
 results=[r for r in json.loads((OUT/'synthetic_examples.json').read_text()) if r['model']!='C_texter']
 configs=[('C_texter',False)]
for name,think in configs:
 model,tok=load(str(MODELS/name/'mlx'),tokenizer_config={'trust_remote_code':True})
 for case,prompts in CASES.items():
  turns=[]
  for i,prompt in enumerate(prompts):
   turns.append({'role':'user','content':prompt});system=SYSTEM;plan=''
   if name=='C_texter':
    plan='Answer the latest fictional question briefly, using only facts already supplied. Ask for clarification when needed.'
    system+='\n\nWhat I\'m thinking right now: '+plan
   ids=tok.apply_chat_template([{'role':'system','content':system}]+turns,add_generation_prompt=True,enable_thinking=think)
   mx.random.seed(7)
   raw=generate(model,tok,prompt=ids,max_tokens=300 if think else 160,sampler=make_sampler(temp=0.7,top_p=0.9)).strip()
   complete=(not think) or '</think>' in raw
   reply=raw.split('</think>',1)[-1].strip() if complete else ''
   results.append({'model':name,'case':case,'turn':i+1,'input':prompt,'raw':raw,'reply':reply,'complete':complete,'supplied_plan':plan,'system':system,'seed':7,'temperature':0.7,'max_tokens':300 if think else 160,'new_synthetic_demonstration':True})
   if complete:turns.append({'role':'assistant','content':reply})
   else:turns.pop()
   (OUT/'synthetic_examples.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
  print(name,case,'finished',flush=True)
 del model,tok;gc.collect();mx.clear_cache()
print('Saved all',len(results),'attempts; no private inputs used.',flush=True)
