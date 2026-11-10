
import os
from pathlib import Path
import json,html,textwrap,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
O=Path(__file__).resolve().parent
rows=json.loads((O/'synthetic_examples.json').read_text())
for r in rows:
 p=(Path(os.environ.get("LOCAL_FINETUNING_HOME", str(Path.home() / "LocalData/hinglish-clone"))) / 'models')/r['model']/'mlx'
 r['checkpoint']={'local_name':r['model'],'config_sha256':hashlib.sha256((p/'config.json').read_bytes()).hexdigest(),'weight_files':[{'name':f.name,'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in sorted(p.glob('*.safetensors'))]}
 r['prompt_source']='Synthetic; no private conversation or memory card.'
 if r['model']=='C_texter':r['planner']='Hand-written synthetic plan; not DeepSeek.'
(O/'synthetic_examples.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
style='''body{font:16px/1.6 system-ui;max-width:900px;margin:40px auto;padding:20px;background:#faf9f6;color:#292824}article{border-top:1px solid #ddd;margin-top:32px;padding-top:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px;border:1px solid #ddd}.meta{color:#666;font-size:13px}a{color:#285b61}'''
parts=[f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All synthetic chat demonstrations</title><style>{style}</style><h1>All synthetic chat demonstrations</h1><p>Fresh dummy conversations against saved local checkpoints. No private messages or memory cards. Fictional situations generated locally for each configuration. One fixed seed (7), temperature 0.7, top-p 0.9. The thinking checkpoints have a 300-token budget; other checkpoints have 160. This is an illustrative probe, not a measured failure rate or a historical training comparison.</p><p>C receives hand-written plans using only facts available at that turn. These are not DeepSeek outputs. All outputs in the local demonstration file are included. Supplied plans must use only available context.</p><p><a href="../index.html">Back to essay</a> · <a href="synthetic_examples.json">Complete output and settings</a></p>']
for name in ['A_nothink','B_think','C_texter','B_weight_edit_025']:
 for case in ['speaker','memory','conversation']:
  group=[r for r in rows if r['model']==name and r['case']==case]
  parts.append(f'<article><h2>{name} · {case}</h2>')
  for r in group:
   parts.append('<p><strong>Synthetic user</strong></p><pre>'+html.escape(r['input'])+'</pre>')
   if r['supplied_plan']:parts.append('<details><summary>Hand-written plan supplied to C</summary><pre>'+html.escape(r['supplied_plan'])+'</pre></details>')
   parts.append('<p><strong>Actual model reply</strong></p><pre>'+html.escape(r['reply'] if r['complete'] else '[No final reply: thinking budget exhausted.]')+'</pre>')
   parts.append('<details><summary>Raw generated output</summary><pre>'+html.escape(r['raw'])+'</pre></details>')
  parts.append('</article>')
parts.append('</html>');(O/'synthetic_review.html').write_text('\n'.join(parts))
def render(name,case,limit,title,filename):
 group=sorted([r for r in rows if r['model']==name and r['case']==case and r['turn']<=limit],key=lambda x:x['turn'])
 bubbles=[]
 for r in group:
  bubbles.extend([('Synthetic user',r['input']),('Actual model reply',r['reply'])])
 wrap=lambda t:[part for line in t.splitlines() for part in (textwrap.wrap(line,70) or [''])] or ['[Empty reply]']
 total=1.5+sum(.8+.27*len(wrap(t)) for _,t in bubbles)
 fig,ax=plt.subplots(figsize=(8.5,total));ax.set(xlim=(0,8.5),ylim=(0,total));ax.axis('off');fig.patch.set_facecolor('#faf9f6')
 ax.text(.35,total-.45,title,fontsize=14,weight='bold',color='#254947');ax.text(.35,total-.8,name+' · New synthetic demonstration · seed 7',fontsize=9,color='#666')
 y=total-1.2
 for role,text in bubbles:
  lines=wrap(text);h=.6+.27*len(lines)
  ax.add_patch(FancyBboxPatch((.35,y-h),7.8,h,boxstyle='round,pad=.08,rounding_size=.09',facecolor='#e5eeea' if role=='Actual model reply' else '#eeeae3',edgecolor='none'))
  ax.text(.55,y-.18,role,fontsize=9,weight='bold',color='#59655d',va='top');ax.text(.55,y-.48,'\n'.join(lines),fontsize=11,va='top',linespacing=1.45,color='#292824');y-=h+.2
 ax.text(.35,.16,'Rendered transcript excerpt. Inputs are invented; model replies are unedited. No private memory.',fontsize=8,color='#666')
 fig.savefig(O/filename,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
render('A_nothink','speaker',2,'The model gets the speaker wrong','example_speaker.png')
render('B_think','memory',3,'The correction only partly survives','example_memory.png')
render('B_weight_edit_025','speaker',1,'The blended model invents background','example_blend.png')
render('C_texter','memory',3,'A supplied plan still becomes fragmented texts','example_texter.png')
print('Rendered four transcript images and full 40-turn review page.')
