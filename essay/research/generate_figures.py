import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
o=Path(__file__).resolve().parent
data=json.loads((o/'figure_data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,ax=plt.subplots(figsize=(7,4))
x=[1,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,9455]
y=[4.870,3.742,3.626,3.509,3.462,3.429,3.374,3.339,3.327,3.306,3.298,3.293]
ax.plot(x,y,marker='o',color='#215c78');ax.set(xlabel='Training step',ylabel='Validation loss (nats/token)',title='First LFM run: validation loss kept improving')
fig.text(.12,.01,'Source: PROGRESS.md • 373 minutes • This does not measure conversational quality.',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(o/'initial_lfm_loss.png');plt.close(fig)
e=data['evaluation']
fig,axs=plt.subplots(1,2,figsize=(9,4))
labels=['Real replies','Base LFM','Fine-tuned LFM']
for ax,key,title in zip(axs,['median_chars','double_text_pct'],['Median reply length (characters)','Turns with multiple texts (%)']):
 vals=[e['lfm3-best']['style_real'][key],e['lfm3-base']['style_generated'][key],e['lfm3-best']['style_generated'][key]]
 ax.bar(labels,vals,color=['#888888','#d58b50','#215c78']);ax.set_title(title,fontsize=11);ax.tick_params(axis='x',labelsize=9)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:g}',ha='center')
 ax.set_ylim(0,max(vals)*1.2)
fig.suptitle('The first adapter learned visible parts of the texting style')
fig.text(.08,.015,'100 test contexts • Style proxies; these do not establish coherence or indistinguishability.',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(o/'style_comparison.png');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(9,6))
for ax,name,title in zip(axs.flat,['A_nothink','B_think','C_texter','D_thinker'],['A: reply only','B: thought + reply','C: reply given a thought','D: thought only']):
 d=data['variants'][name];x,y=zip(*d['val']);ax.plot(x,y,marker='o',color='#215c78');ax.set(title=title,xlabel='Step',ylabel='Validation loss')
fig.suptitle('Four variants, four different prediction targets')
fig.text(.08,.01,'Losses across panels are not directly comparable: targets and conditioning differ.',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(o/'four_variant_losses.png')
print('Saved three figures.')
