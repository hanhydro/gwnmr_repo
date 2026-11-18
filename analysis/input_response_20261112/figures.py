from pathlib import Path
import os,json
import numpy as np
import pandas as pd
STUDY=Path(__file__).resolve().parents[2];DATA=STUDY/'data/input_response_20261112';FIG=STUDY/'figures'
os.environ.setdefault('MPLCONFIGDIR',str(STUDY/'work/input_response_20261112/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm
from matplotlib import patheffects
from matplotlib.ticker import NullFormatter

TEAL='#007E87';PURPLE='#8B4095';UPPER='#BE6422';LOWER='#2767A1'
SELECTED=[(0,.25),(1,.75)];CONTOURS=[.01,.03,.06];T2_MIN=.5

def setup():
    font_manager.fontManager.addfont('/System/Library/Fonts/Helvetica.ttc')
    resolved=font_manager.findfont('Helvetica',fallback_to_default=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica'],
        'mathtext.fontset':'custom','mathtext.rm':'Helvetica','mathtext.it':'Helvetica:italic',
        'mathtext.bf':'Helvetica:bold','mathtext.sf':'Helvetica','mathtext.fallback':None,
        'font.size':9,'axes.labelsize':9,'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.7,'pdf.fonttype':42,
        'lines.linewidth':1.7,'savefig.facecolor':'white'})
    (DATA/'figure_font.json').write_text(json.dumps(dict(resolved_font=resolved))+'\n')

def finish(fig,axes,name):
    for letter,ax in zip('abcdefghijklmnop',np.ravel(axes)):
        ax.text(.025,.975,f'({letter})',ha='left',va='top',transform=ax.transAxes,fontsize=10,
            color='black',path_effects=[patheffects.withStroke(linewidth=2.4,foreground='white')]
            if name=='input_response_histories' and letter in 'ef' else None)
    for ax in fig.axes:
        ax.tick_params(direction='out',length=3)
        for spine in ax.spines.values():spine.set_visible(True)
    for ext in ['pdf','png','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=220)
    plt.close(fig)

def hydraulics():
    d=pd.read_csv(DATA/'matched_histories.csv');h=pd.read_csv(DATA/'hydraulic_histories.csv')
    rates=pd.read_csv(DATA/'selected_interval_fluxes.csv')
    fig,axes=plt.subplots(4,2,figsize=(7.2,9.0),layout='constrained')
    heat=[]
    for j,(order,depth) in enumerate(SELECTED):
        c=d[(d.order==order)&(d.interface_m==depth)&(d.initial_head_m==-.5)&(d.input_low_mm==0)]
        for inp,color in [(14.4,TEAL),(28.8,PURPLE)]:
            x=c[c.input_high_mm==inp].sort_values('time_h')
            for key,ls,label in [('additional_upper_mm','--','Upper'),('additional_lower_mm','-','Lower')]:
                axes[0,j].plot(x.time_h,x[key],color=color,ls=ls,label=f'{inp:g} mm, {label.lower()}')
            g=rates[(rates.order==order)&(rates.infiltration_mm==inp)].set_index('time_h')
            zero=rates[(rates.order==order)&(rates.infiltration_mm==0)].set_index('time_h')
            for key,ls in [('transfer_flux_mm_h','--'),('basal_flux_mm_h','-')]:
                v=g[key]-zero[key];axes[1,j].plot(v.index,v,color=color,ls=ls)
        b=h[(h.order==order)&(h.interface_m==depth)&(h.initial_head_m==-.5)]
        for inp,color,style in [(0,'.45',':'),(14.4,TEAL,'-'),(28.8,PURPLE,'-')]:
            x=b[b.infiltration_mm==inp].sort_values('time_h')
            axes[3,j].plot(x.time_h,x.base_head_m,color=color,ls=style,label=f'{inp:g} mm')
        files=[DATA/f'hydraulics/half_hour/mechanism_o{order}_d{0 if order==0 else 2}_h1_r{r}.npz' for r in ['144','288']]
        lo,hi=[np.load(f) for f in files];times=(lo['time_s']-7200)/3600
        diff=(hi['theta_node']-lo['theta_node']).T
        im=axes[2,j].pcolormesh(times,lo['z_m'],diff,shading='auto',cmap='Blues',
            norm=PowerNorm(.4,vmin=0,vmax=.18),rasterized=True)
        lines=axes[2,j].contour(times,lo['z_m'],diff,levels=CONTOURS,colors='#08306B',linewidths=.6)
        heat.append((im,lines));axes[2,j].axhline(depth,color='#262B2F',ls=(0,(1.2,1.6)),lw=1)
        axes[2,j].set_ylim(1,0)
        for ax in axes[:,j]:ax.set(xlim=(0,72),xticks=[0,24,48,72])
        axes[0,j].set_ylim(-.5,33)
        axes[1,j].set_ylim(-.1,4.2 if j==0 else .9)
        axes[3,j].set_ylim(-.65,-.15)
        axes[0,j].text(.98,.97,f'{["Sand over loam","Loam over sand"][order]}\nInterface {depth:.2f} m',
            ha='right',va='top',transform=axes[0,j].transAxes,fontsize=8)
        axes[3,j].set_xlabel('Time since infiltration ended (h)')
    axes[0,0].set_ylabel('Additional layer storage (mm)')
    axes[1,0].set_ylabel('Additional flux (mm h$^{-1}$)')
    axes[2,0].set_ylabel('Depth (m)')
    axes[3,0].set_ylabel('Pressure head at base (m)')
    axes[0,0].legend(loc='upper right',bbox_to_anchor=(.99,.79),frameon=False,ncol=2,fontsize=6.7)
    axes[1,0].legend(handles=[Line2D([],[],color='.3',ls='--',label='Across interface'),
        Line2D([],[],color='.3',ls='-',label='Through base')],loc='upper right',frameon=False)
    axes[3,0].legend(loc='upper right',bbox_to_anchor=(1,.88),frameon=False,ncol=3)
    bar=fig.colorbar(heat[1][0],ax=list(axes[2,:]),fraction=.03,pad=.02,ticks=[0,.01,.03,.06,.12,.18],
        label='Added water content\n(28.8 minus 14.4 mm, m$^3$ m$^{-3}$)')
    bar.ax.set_yticklabels(['0','0.01','0.03','0.06','0.12','0.18']);bar.add_lines(heat[1][1])
    finish(fig,axes,'input_response_histories')

def nmr():
    d=pd.read_csv(DATA/'pore_water_distributions.csv.gz');c=pd.read_csv(DATA/'signal_input_contrasts.csv')
    c=c[(c.initial_head_m==-.5)&(c.input_high_mm==28.8)&(c.input_low_mm==14.4)]
    fig,axes=plt.subplots(3,2,figsize=(7.2,7.0),layout='constrained')
    t2=np.sort(d.T2_s.unique());step=np.diff(np.log10(t2)).mean()
    edges=10**np.r_[np.log10(t2)-step/2,np.log10(t2[-1])+step/2]
    for j,(order,depth) in enumerate(SELECTED):
        g=d[(d.order==order)&(d.interface_m==depth)&(d.initial_head_m==-.5)]
        for layer,color in [('upper',UPPER),('lower',LOWER)]:
            for t,ls in [(24,'-'),(72,'--')]:
                p=g[(g.layer==layer)&(g.time_h==t)].pivot(index='T2_s',columns='infiltration_mm',values='capillary_water_mm')
                amount=np.histogram(p.index,bins=edges,weights=p[28.8]-p[14.4])[0]
                assert abs(amount.sum()-(p[28.8]-p[14.4]).sum())<1e-10
                assert abs(amount[edges[1:]<=T2_MIN]).sum()<.01
                axes[0,j].stairs(amount,edges,color=color,ls=ls,label=f'{layer.capitalize()}, {t} h')
        x=c[(c.order==order)&(c.interface_m==depth)&np.isclose(c.q_ams,21.24462820428606)].sort_values('time_h')
        for layer,color in [('upper',UPPER),('lower',LOWER)]:
            axes[1,j].plot(x.time_h,x[layer+'_projection_nV'],color=color,marker='o',ms=3,label=layer.capitalize())
        axes[1,j].plot(x.time_h,x.complex_difference_magnitude_nV,color='.2',ls=':',label='Combined change')
        for t,color,ls in [(0,'.55',':'),(24,TEAL,'-'),(72,PURPLE,'--')]:
            x=c[(c.order==order)&(c.interface_m==depth)&(c.time_h==t)].sort_values('q_ams')
            axes[2,j].plot(x.q_ams,x.magnitude_difference_nV,color=color,ls=ls,label=f'{t} h')
        axes[0,j].set(xscale='log',xlim=(T2_MIN,edges[-1]),xlabel='Intrinsic relaxation time (s)')
        axes[0,j].set_xticks([.5,1,2],['0.5','1','2']);axes[0,j].xaxis.set_minor_formatter(NullFormatter())
        axes[0,j].margins(y=.25)
        axes[1,j].set(xlim=(0,72),xticks=[0,24,48,72],ylim=(-.002,.07),xlabel='Time since infiltration ended (h)')
        axes[2,j].set(xscale='log',xlim=(.05,300),xlabel='Pulse moment (A ms)')
        axes[0,j].text(.98,.97,f'{["Sand over loam","Loam over sand"][order]}\nInterface {depth:.2f} m',ha='right',va='top',transform=axes[0,j].transAxes,fontsize=8)
        for ax in axes[:,j]:ax.axhline(0,color='.7',lw=.6,zorder=-2)
    axes[0,0].set_ylabel('Additional capillary water\nper $T_2$ interval (mm)')
    axes[1,0].set_ylabel('Projected layer-voltage change (nV)')
    axes[2,0].set_ylabel('Full-FID magnitude difference (nV)')
    axes[0,0].legend(loc='upper left',bbox_to_anchor=(.03,.83),frameon=False,fontsize=7)
    axes[1,0].legend(loc='upper right',frameon=False,fontsize=7)
    axes[2,0].legend(loc='upper left',bbox_to_anchor=(.03,.88),frameon=False,fontsize=7)
    finish(fig,axes,'input_response_pores_signals')

if __name__=='__main__':setup();hydraulics();nmr()
