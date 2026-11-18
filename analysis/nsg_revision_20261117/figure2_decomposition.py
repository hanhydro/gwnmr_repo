from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'input_response_20261112'))
from figures import STUDY, DATA, setup, finish
import matplotlib.pyplot as plt

MATH = STUDY / 'data/mathematical_generalization_20261116'
OUT = STUDY / 'data/nsg_revision_20261117'
NAME = 'input_response_configurations_decomposition'
TERM_COLORS = {'conductivity': '#BE6422', 'head': '#2767A1', 'sum': '#1A1A1A'}


def cumulative_terms():
    d = pd.read_csv(MATH / 'basal_state_divided_differences.csv')
    rows = []
    for order, g in d.groupby('order'):
        g = g.sort_values('time_h')
        t = g.time_h.to_numpy()
        out = {'order': order, 'time_h': t}
        for key, col in [('conductivity', 'conductivity_secant_term_mm_h'),
                         ('head', 'head_increment_term_mm_h')]:
            y = g[col].to_numpy()
            out[key + '_mm'] = np.concatenate([[0.0], np.cumsum(np.diff(t) * (y[1:] + y[:-1]) / 2)])
        out['sum_mm'] = out['conductivity_mm'] + out['head_mm']
        rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True)


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    cum = cumulative_terms()
    cum.to_csv(OUT / 'decomposition_cumulative.csv', index=False)
    ref = pd.read_csv(MATH / 'basal_state_integrals.csv').set_index('order')
    for order in (0, 1):
        end = cum[cum.order == order].iloc[-1]
        assert abs(end.conductivity_mm - ref.loc[order, 'conductivity_secant_term_mm']) < 1e-6
        assert abs(end.head_mm - ref.loc[order, 'head_increment_term_mm']) < 1e-6
    peaks = pd.read_csv(MATH / 'storage_peak_balance.csv')

    d = pd.read_csv(DATA / 'input_response_by_configuration.csv')
    fig, axes = plt.subplots(3, 2, figsize=(5.4, 6.7), layout='constrained')
    colors = ['#BE6422', '#2767A1', '#598B36']
    for order in range(2):
        for j, head in enumerate([-.5, -.15]):
            ax = axes[order, j]
            for depth, c, m in zip([.25, .5, .75], colors, ['o', 's', '^']):
                r = d[(d.order == order) & (d.initial_head_m == head) & (d.interface_m == depth)].iloc[0]
                ax.plot([0, 14.4, 28.8], [0, r.first_increment_mm, r.first_increment_mm + r.second_increment_mm],
                        marker=m, ms=4, color=c, label=f'{depth:.2f} m')
            ax.set(xticks=[0, 14.4, 28.8], xlim=(-1, 30))
            ax.text(.97, .96, f'$h_i={head:g}$ m', ha='right', va='top', transform=ax.transAxes)
            ax.set_ylim((-.5, 15) if j == 0 else (-1, 32))
            if order == 1:
                ax.set_xlabel('Two-hour input (mm)')
            if j == 0:
                ax.set_ylabel(['Sand over loam', 'Loam over sand'][order] + '\nAdditional drainage (mm)')
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(.03, .88), frameon=False, title='Interface')

    for j, order in enumerate((0, 1)):
        ax = axes[2, j]
        c = cum[cum.order == order]
        for t in peaks[peaks.order == order].peak_time_h:
            ax.axvline(t, color='.55', ls=(0, (1.2, 1.6)), lw=1)
        ax.axhline(0, color='.55', lw=.6)
        ax.plot(c.time_h, c.conductivity_mm, color=TERM_COLORS['conductivity'], ls='--', label='Conductivity slope')
        ax.plot(c.time_h, c.head_mm, color=TERM_COLORS['head'], ls='-.', label='Head increments')
        ax.plot(c.time_h, c.sum_mm, color=TERM_COLORS['sum'], label='Sum')
        ax.set(xlim=(0, 72), xticks=[0, 24, 48, 72], ylim=(-7.5, 16.5))
        ax.set_xlabel('Time since infiltration ended (h)')
    axes[2, 0].set_ylabel('Integrated term (mm)')
    axes[2, 0].legend(loc='upper left', bbox_to_anchor=(.03, .88), frameon=False)
    finish(fig, axes, NAME)


if __name__ == '__main__':
    main()
