"""Group-meeting figure: persisted current report and documented Phase17 trajectory."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
r=json.loads((ROOT/'output/subtree_local_reflection/discovery100_27b_strict_preserve5/report.json').read_text())
current=[r['initial']['strict_accuracy']*100]+[e['system_after']['strict_accuracy']*100 for e in r['epochs']]
# Source: docs/Evolving Structured Rubrics Implementation Plan.md, section 16.2.
historical=[64,62,65,66,65,65]
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B4C3D1','text.color':'#193B5E','axes.labelcolor':'#193B5E','xtick.color':'#526A80','ytick.color':'#526A80','pdf.fonttype':42})
blue='#075B97'; orange='#CE7350'
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,6.8),gridspec_kw={'width_ratios':[1.65,1]},facecolor='white')
fig.subplots_adjust(left=.065,right=.98,top=.80,bottom=.32,wspace=.27)
for a in (ax,bx):a.set_facecolor('#F5F8FB');a.grid(axis='y',color='#DCE5EC',lw=.8);a.set_axisbelow(True)
x=np.arange(6)
ax.plot(x,current,'o-',color=blue,lw=2.8,ms=7,label='当前：逐例子树反思＋Global Arbiter')
ax.plot(x,historical,'s--',color=orange,lw=2.4,ms=6,label='历史参考：Split＋Refine＋显式递归投票')
for vals,c,dy in [(current,blue,0.65),(historical,orange,-1.15)]:
 for i,v in enumerate(vals):ax.text(i,v+(1.0 if i==0 and c==orange else dy),f'{v:.0f}',ha='center',color=c,fontweight='bold',fontsize=14)
ax.set(xticks=x,xticklabels=['起点','E1','E2','E3','E4','E5 / Final'],ylim=(60,74),yticks=[60,64,68,72],ylabel='Strict ACC (%)')
ax.set_title('(a) Discovery100：系统判断的演化轨迹',loc='left',fontsize=16,pad=16,fontweight='bold')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,fontsize=12)
datasets=['Discovery100','Dev150','VLRB']
before=[r['initial']['strict_accuracy']*100,r['dev']['initial']['strict_accuracy']*100,r['vlrb']['official']['initial']['strict_accuracy']*100]
after=[r['final']['strict_accuracy']*100,r['dev']['final']['strict_accuracy']*100,r['vlrb']['official']['final']['strict_accuracy']*100]
delta=np.array(after)-before
bx.bar(np.arange(3),delta,color=[blue,orange,orange],width=.53,zorder=3)
bx.axhline(0,color='#73879A',lw=1)
for i,v in enumerate(delta):bx.text(i,v+(.3 if v>=0 else -.35),f'{v:+.2f} pp',ha='center',va='bottom' if v>=0 else 'top',fontweight='bold',color=blue if v>=0 else orange)
bx.set(xticks=np.arange(3),xticklabels=datasets,ylim=(-3,8),yticks=[-2,0,2,4,6,8],ylabel='Final − Init（百分点）')
bx.tick_params(axis='x',labelsize=11)
bx.set_title('(b) 当前方法：Final 相对 Init 的变化',loc='left',fontsize=16,pad=16,fontweight='bold')
fig.suptitle('本次运行：演化集提升，但验证与测试未获益',fontsize=21,fontweight='bold',y=.97)
fig.text(.065,.11,'起点不同：当前为 Init Split（已有 children）；历史为五个初始 root。',fontsize=11,color='#526A80')
fig.text(.065,.075,'Manager：当前 Qwen3.5-27B；历史 Qwen3.5-397B-A17B。两条曲线是描述性对照，不是单变量消融。',fontsize=11,color='#526A80')
fig.text(.065,.04,'Worker 均为 Qwen3-VL-8B；VLRB 使用官方 K=3 汇总。仅展示已有观测，未对缺失的逐轮测试结果插值。',fontsize=10,color='#526A80')
for ext in ['png','pdf','svg']:fig.savefig(HERE/f'evolution_comparison.{ext}',dpi=300,bbox_inches='tight')
json.dump({'current_discovery':current,'historical_discovery':historical,'current_initial':before,'current_final':after,'datasets':datasets},open(HERE/'evolution_comparison_data.json','w'),indent=2)

# Chart-only assets for PowerPoint; slide prose and legend remain editable.
ax.set_title('', loc='left')
bx.set_title('', loc='left')
ax.get_legend().remove()
fig.canvas.draw()
for chart, name in [(ax, 'discovery_curve'), (bx, 'transfer_bars')]:
    box = chart.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025, 1.045)
    fig.savefig(HERE / f'{name}.png', dpi=300, bbox_inches=box, facecolor='white')
