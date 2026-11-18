from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY/'data/redistribution_20261112'
FIG = STUDY/'figures'
os.environ.setdefault('MPLCONFIGDIR', str(STUDY/'work/redistribution_20261112/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HEADS = [-1.5, -.5, -.15]
COLORS = ['#B55D22', '#268577', '#285CA5']
ORDERS = ['Sand over loam', 'Loam over sand']


def setup():
    font_manager.fontManager.addfont('/System/Library/Fonts/Helvetica.ttc')
    resolved = font_manager.findfont('Helvetica', fallback_to_default=False)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Helvetica'],
        'mathtext.fontset':'custom', 'mathtext.rm':'Helvetica', 'mathtext.it':'Helvetica:italic',
        'mathtext.bf':'Helvetica:bold', 'mathtext.sf':'Helvetica', 'mathtext.fallback':None,
        'font.size':9, 'axes.labelsize':9, 'legend.fontsize':8,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.spines.top':True,
        'axes.spines.right':True, 'axes.linewidth':.7, 'pdf.fonttype':42,
        'lines.linewidth':1.6, 'savefig.facecolor':'white'})
    (DATA/'figure_font.json').write_text(json.dumps({'resolved_font':resolved})+'\n')


def finish(fig, axes, name):
    for letter, ax in zip('abcdefghijklmnop', np.ravel(axes)):
        ax.text(.025,.975,f'({letter})',ha='left',va='top',transform=ax.transAxes,fontsize=10)
    for ax in fig.axes:
        ax.tick_params(direction='out', length=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
    for ext in ('pdf','png','svg'):
        fig.savefig(FIG/f'{name}.{ext}',dpi=220)
    plt.close(fig)


def depth_support():
    d=pd.read_csv(DATA/'hydraulic_cases.csv');d=d[d.population=='depth']
    if len(d):
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.0),layout='constrained')
        for order in (0,1):
            for head,color in zip(HEADS,COLORS):
                g=d[(d.order==order)&(d.initial_head_m==head)].groupby('interface_m')
                avg=g[['initial_storage_mm','storage_72h_mm','base_drainage_24_72_mm']].mean()
                axes[0,order].plot(avg.index,avg.initial_storage_mm,'--',color=color,alpha=.65)
                axes[0,order].plot(avg.index,avg.storage_72h_mm,'o-',color=color,ms=4)
                axes[1,order].plot(avg.index,avg.base_drainage_24_72_mm,'o-',color=color,ms=4,label=f'$h_i={head:g}$ m')
                limits=g.base_drainage_24_72_mm.quantile([.1,.9]).unstack()
                axes[1,order].fill_between(avg.index,limits[.1],limits[.9],color=color,alpha=.12)
            axes[0,order].text(.97,.96,ORDERS[order],ha='right',va='top',transform=axes[0,order].transAxes)
            for ax in axes[:,order]:
                ax.set(xlim=(.2,.8),xticks=[.25,.5,.75],xlabel='Interface depth (m)');ax.margins(y=.22)
            axes[1,order].legend(loc='center right',frameon=False,fontsize=7)
        axes[0,0].set_ylabel('Water storage (mm)')
        axes[1,0].set_ylabel('Drainage from 24 to 72 h (mm)')
        axes[0,0].legend(handles=[Line2D([],[],color='.3',ls='--',label='Before infiltration'),Line2D([],[],color='.3',marker='o',label='72 h')],loc='lower left',frameon=False,fontsize=7)
        finish(fig,axes,'redistribution_depth_support')


if __name__=='__main__':
    setup()
    depth_support()
