from pathlib import Path
import numpy as np
import pandas as pd
from figures import STUDY,setup,finish
import matplotlib.pyplot as plt
def main():
    setup();root=STUDY/'data/physical_recovery_20260929'
    table=pd.read_csv(root/'matched_capillary_water_control.csv')
    values=np.load(root/'matched_capillary_water_control.npz')
    cdf=pd.read_csv(STUDY/'data/figure_revision_20261022/filled_pore_cumulative_volume.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    for soil,color in [('sand','#BE6422'),('loam','#2767A1')]:
        row=table[(table.soil==soil)&(table.assignment=='common_rho')&np.isclose(table.capillary_water,.1)].iloc[0]
        a=cdf[cdf.soil==soil];curve=values['normalized_intrinsic_decay'][row.name]
        axes[0].semilogx(a.radius_um,a.filled_water_fraction,color=color,label=f'{soil.capitalize()}, $h={row.equivalent_pressure_head_m:.3f}$ m')
        axes[1].plot(values['times_s']*1000,curve/curve[0],color=color,label=soil.capitalize())
    axes[0].set(xlim=(.01,1000),ylim=(0,1.12),xlabel='Filled-pore radius (µm)',ylabel='Cumulative capillary-water fraction')
    axes[1].set(ylim=(0,1.12),xlabel='Intrinsic decay time (ms)',ylabel='Normalized intrinsic signal')
    axes[0].legend(loc='upper left',bbox_to_anchor=(.01,.9),frameon=False,fontsize=7.5)
    axes[1].legend(loc='lower right',frameon=False,fontsize=7.5)
    finish(fig,axes,'redistribution_pore_support')
if __name__=='__main__':main()
