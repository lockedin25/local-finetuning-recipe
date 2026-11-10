from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
O=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':180})
def box(ax,x,y,w,h,text,color='#edf2ef'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.07',facecolor=color,edgecolor='#90a59d'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
def arrow(ax,a,b,label=''):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#375a59','lw':1.5})
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=8,color='#52605b')
fig,ax=plt.subplots(figsize=(10,6));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
ax.text(.1,5.7,'The memory setup I was trying to build',fontsize=17,color='#254947')
box(ax,.2,4,2.4,1,'Chats from month M');box(ax,3.6,4,2.5,1,'Month-level notes');arrow(ax,(2.65,4.5),(3.55,4.5))
box(ax,.2,2,2.4,1,'Rolling memory\nthrough month M−1');box(ax,3.6,2,2.5,1,'Sequential merge\nUpdate what is known');arrow(ax,(2.65,2.5),(3.55,2.5));arrow(ax,(4.85,3.95),(4.85,3.05))
box(ax,7.1,2,2.6,1,'Memory card after M\nTarget: ≤150 words');arrow(ax,(6.15,2.5),(7.05,2.5))
box(ax,.2,.2,3,1,'Recent conversation window\nUp to 40 turns / 2,048 tokens');box(ax,5.3,.2,4.4,1,'Prompt for a reply in month M+1\nRecent context + previous-month card',color='#eee8dd');arrow(ax,(3.25,.7),(5.25,.7));arrow(ax,(8.4,1.95),(8.4,1.25))
fig.text(.035,.015,'Recent messages carry immediate context. The rolling card carries compressed background, including relationship facts.\nThis is a schematic of the attempted pipeline, not a measured model of human memory.',fontsize=8,color='#5d655f')
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(O/'memory_architecture.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.6));ax.set(xlim=(-.12,1.12),ylim=(-.8,1));ax.axis('off')
ax.text(-.1,.85,'What the 25% weight blend actually did',fontsize=17,color='#254947')
ax.plot([0,1],[.3,.3],color='#a1aaa3',lw=3);ax.scatter([0,.25,1],[.3]*3,s=[100,160,100],c=['#777','#285b61','#c88b55'],zorder=3)
for x,t in [(0,'Base model\nW_base'),(.25,'Saved blend\n25% of the change'),(1,'Fine-tuned B\nW_B')]:ax.text(x,.15,t,ha='center',va='top',fontsize=10)
ax.annotate('',xy=(.25,.52),xytext=(0,.52),arrowprops={'arrowstyle':'->','color':'#285b61','lw':2})
ax.text(.5,-.42,'W_edit = W_base + 0.25 × (W_B − W_base)',ha='center',fontsize=14)
ax.text(.5,-.67,'A schematic path between checkpoints. Distances do not represent measured behavior.',ha='center',fontsize=9,color='#666')
fig.tight_layout();fig.savefig(O/'weight_blend.png',bbox_inches='tight')
print('Saved memory and weight-blend diagrams.')
