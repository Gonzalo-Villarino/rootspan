"""Reproduce the development comparison figure; no model fitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parent
rows=json.loads((D/'expanded_fields_one_seven/comparison.json').read_text())['rows']
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.6));y=np.arange(3)
old=np.array([r['V1_absolute_T_error_um'] for r in rows]);new=np.array([r['expanded_absolute_T_error_um'] for r in rows])
ax.barh(y-.18,old,.34,label='Current V1 detector',color='#506779')
ax.barh(y+.18,new,.34,label='New 11-field candidate',color='#bf7455')
ax.invert_yaxis();ax.set_yticks(y,[f'Root {r["root"]}' for r in rows]);ax.set_xlim(0,9500)
ax.set_xlabel('Distance between model T and your manual T (µm) — smaller is better')
for ys,vals in [(y-.18,old),(y+.18,new)]:
    for yy,v in zip(ys,vals):ax.text(v+100,yy,f'{v:,.0f}',va='center',fontsize=11)
ax.legend(loc='upper right')
fig.suptitle('Slide 1_7: the candidate did not improve consistently',fontsize=16,fontweight='bold',x=.08,ha='left')
fig.text(.08,.015,'Same automatic paths; three development roots, not an independent final test.',fontsize=10,color='#555555')
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(D/'TRANSITION_ERROR_COMPARISON.png',dpi=170)
