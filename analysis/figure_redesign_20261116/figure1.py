from pathlib import Path
import json
import os
import sys

STUDY = Path(__file__).resolve().parents[2]
WORK = STUDY / 'work/figure_redesign_20261116'
WORK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(WORK / 'matplotlib'))
sys.path.insert(0, str(STUDY / 'analysis'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from physics import SIGMA, WATER_DENSITY, GRAVITY

SAND_P = (0.045, 0.430, 13.0, 2.6, 8e-5)
LOAM_P = (0.078, 0.430, 3.6, 1.6, 1e-5)
HEADS = (0.15, 0.50, 1.50)
INTERFACES = (0.25, 0.50, 0.75)
SELECTED = {(0, 0.25), (1, 0.75)}
Q_SHOWN = (4.68, 9.97, 21.24, 45.27, 96.45)

SAND_FILL, LOAM_FILL = '#EBD7A8', '#B88D63'
SAND_LINE, LOAM_LINE = '#C99A2E', '#5C3B1E'
WATER, INK, MUTED = '#1F6FB2', '#262B2F', '#8C9499'
CAPILLARY = 2 * SIGMA / (WATER_DENSITY * GRAVITY) * 1e6


def setup():
    font_manager.fontManager.addfont('/System/Library/Fonts/Helvetica.ttc')
    oblique = WORK / 'fonts/Helvetica-Oblique.ttf'
    if not oblique.exists():
        from fontTools.ttLib import TTCollection
        oblique.parent.mkdir(parents=True, exist_ok=True)
        TTCollection('/System/Library/Fonts/Helvetica.ttc').fonts[2].save(str(oblique))
    font_manager.fontManager.addfont(str(oblique))
    font = font_manager.findfont('Helvetica', fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Helvetica', 'mathtext.it': 'Helvetica:italic',
        'mathtext.bf': 'Helvetica:bold', 'mathtext.sf': 'Helvetica', 'mathtext.fallback': None,
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7, 'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
        'xtick.minor.width': .5, 'ytick.minor.width': .5, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    return font


def van_genuchten(suction, theta_r, theta_s, alpha, n, ks, ell=.5):
    m = 1 - 1 / n
    se = (1 + (alpha * suction) ** n) ** (-m)
    k = ks * se ** ell * (1 - (1 - se ** (1 / m)) ** m) ** 2
    return theta_r + (theta_s - theta_r) * se, k * 3.6e6


def label(ax, letter, x=.03, y=.975):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, ha='left', va='top', fontsize=9, color='black')


def arrow(ax, start, end, color=WATER, width=1.3, scale=8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale, lw=width,
                                 color=color, shrinkA=0, shrinkB=0, zorder=5))


def design(ax):
    ax.set(xlim=(-.02, 1.0), ylim=(1.235, -.215))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#7D858A'); spine.set_linewidth(.65)
    width, gap, split, x0 = .088, .03, .13, .215
    starts = []
    for order in range(2):
        for j, depth in enumerate(INTERFACES[::-1]):
            starts.append((order, depth, x0 + order * (3 * width + 2 * gap + split) + j * (width + gap)))
    for order, depth, x in starts:
        chosen = (order, depth) in SELECTED
        upper, lower = (SAND_FILL, LOAM_FILL) if order == 0 else (LOAM_FILL, SAND_FILL)
        alpha = 1 if chosen else .6
        ax.add_patch(Rectangle((x, 0), width, depth, facecolor=upper, alpha=alpha, lw=0))
        ax.add_patch(Rectangle((x, depth), width, 1 - depth, facecolor=lower, alpha=alpha, lw=0))
        edge = INK if chosen else '#A3AAAF'
        ax.add_patch(Rectangle((x, 0), width, 1, fill=False, edgecolor=edge, lw=1.15 if chosen else .55, zorder=4))
        ax.plot([x, x + width], [depth, depth], color=edge, lw=.7 if chosen else .55, zorder=4)
        if chosen:
            mid = x + width / 2
            arrow(ax, (mid, -.16), (mid, -.015))
            arrow(ax, (mid, depth - .06), (mid, depth + .06), width=1.5)
            arrow(ax, (mid, 1.015), (mid, 1.155))
            side, ha = (x + width + .016, 'left') if order == 0 else (x - .016, 'right')
            ax.text(side, depth, r'$M$', ha=ha, va='center', fontsize=8.5, color=INK)
            ax.text(mid, depth / 2, r'$S_U$', ha='center', va='center', fontsize=7.5, color=INK, zorder=6)
            ax.text(mid, (1 + depth) / 2 + (.03 if order == 1 else 0), r'$S_L$', ha='center', va='center',
                    fontsize=7.5, color=INK, zorder=6)
    centre = (starts[2][2] + width + starts[3][2]) / 2
    ax.text(centre, -.09, r'$I$', ha='center', va='center', fontsize=8.5, color=INK)
    ax.text(centre, 1.085, r'$D$', ha='center', va='center', fontsize=8.5, color=INK)
    axis_x = .15
    ax.plot([axis_x, axis_x], [0, 1], color=INK, lw=.7)
    for depth in [0, .25, .5, .75, 1]:
        ax.plot([axis_x - .016, axis_x], [depth, depth], color=INK, lw=.7)
        ax.text(axis_x - .027, depth, f'{depth:g}', ha='right', va='center', fontsize=7.5, color=INK)
    ax.text(.04, .5, 'Depth (m)', rotation=90, ha='center', va='center', fontsize=8, color=INK)
    for x, color, name in [(starts[0][2], SAND_FILL, 'Sand'), (starts[4][2], LOAM_FILL, 'Loam')]:
        ax.add_patch(Rectangle((x, 1.12), .04, .07, facecolor=color, edgecolor=INK, lw=.45))
        ax.text(x + .055, 1.157, name, ha='left', va='center', fontsize=7.5, color=INK)
    label(ax, 'a', .025, .985)


def hydraulic_functions(ax_theta, ax_k):
    suction = np.geomspace(.01, 10, 400)
    for params, color, name in [(SAND_P, SAND_LINE, 'Sand'), (LOAM_P, LOAM_LINE, 'Loam')]:
        theta, k = van_genuchten(suction, *params)
        ax_theta.plot(suction, theta, color=color, lw=1.6, label=name)
        ax_k.plot(suction, k, color=color, lw=1.6)
        theta_i, k_i = van_genuchten(np.array(HEADS), *params)
        ax_theta.plot(HEADS, theta_i, 'o', ms=3.6, mfc='white', mec=color, mew=1.1, zorder=5)
        ax_k.plot(HEADS, k_i, 'o', ms=3.6, mfc='white', mec=color, mew=1.1, zorder=5)
    for ax in (ax_theta, ax_k):
        for h in HEADS:
            ax.axvline(h, color='#5F676C' if h == .5 else '#B9BFC3', lw=.8, ls=(0, (3, 2)), zorder=0)
        ax.set_xscale('log'); ax.set_xlim(.01, 10)
    ax_theta.set(ylim=(0, .53), yticks=[0, .1, .2, .3, .4, .5])
    ax_theta.set_ylabel(r'$\theta$ (m$^3$ m$^{-3}$)')
    ax_theta.tick_params(labelbottom=False)
    ax_k.set_yscale('log'); ax_k.set_ylim(1e-6, 3e4)
    ax_k.set_yticks([1e-6, 1e-4, 1e-2, 1, 1e2])
    ax_k.set_yticklabels([r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', r'$10^{0}$', r'$10^{2}$'])
    ax_k.set_xticks([.01, .1, 1, 10]); ax_k.set_xticklabels(['0.01', '0.1', '1', '10'])
    ax_k.set_xlabel(r'Suction, $-h$ (m)')
    ax_k.set_ylabel(r'$K$ (mm h$^{-1}$)')
    inverse = lambda x: CAPILLARY / np.maximum(x, 1e-12)
    top = ax_theta.secondary_xaxis('top', functions=(inverse, inverse))
    top.set_xticks([1, 10, 100, 1000]); top.set_xticklabels(['1', '10', '100', '1000'])
    top.set_xlabel('Largest filled pore radius (µm)', labelpad=3)
    top.tick_params(direction='out', length=3, width=.7)
    ax_theta.legend(loc='lower left', frameon=False, handlelength=1.6, borderaxespad=.3)
    label(ax_theta, 'b'); label(ax_k, 'c')


def sensitivity(ax):
    obs = np.load(STUDY / 'data/observation_kernel_full_100.npz')
    k = int(np.argmin(abs(obs['t2_s'] - .5)))
    kernel = np.abs(obs['kernel_nV'][:, :, k]) / obs['dz_m']
    z = obs['z_m']; q = obs['q_ams']
    ax.axhspan(1, 2.2, color='#ECEFF1', lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=.7, zorder=1)
    ax.axhline(1, color='#5F676C', lw=.8, ls=(0, (4, 2)), zorder=1)
    for depth in (.25, .75):
        ax.axhline(depth, color='#9AA1A6', lw=.7, ls=':', zorder=1)
    colors = plt.cm.viridis(np.linspace(0, .8, len(Q_SHOWN)))
    chosen = []
    for target, color in zip(Q_SHOWN, colors):
        i = int(np.argmin(abs(q - target))); chosen.append(float(q[i]))
        bold = abs(q[i] - 21.24) < .01
        ax.plot(kernel[i], z, color=color, lw=2.1 if bold else 1.1, zorder=4 if bold else 3,
                label=f'{q[i]:.2f}' if bold else f'{q[i]:.2g}')
    ax.set(ylim=(2, -.22), xlim=(0, 7.2), xticks=[0, 2, 4, 6], yticks=[0, .5, 1, 1.5, 2])
    ax.set_xlabel(r'Kernel magnitude, $|K|$ (nV m$^{-1}$)')
    ax.set_ylabel('Depth (m)')
    ax.legend(title=r'$q$ (A ms)', loc='lower right', frameon=False, handlelength=1.6,
              title_fontsize=7, borderaxespad=.3, labelspacing=.3)
    label(ax, 'd')
    return dict(t2_s=float(obs['t2_s'][k]), pulse_moments_ams=chosen)


def main():
    setup()
    fig = plt.figure(figsize=(7.2, 3.15))
    ax_a = fig.add_axes([.008, .145, .37, .725])
    ax_b = fig.add_axes([.455, .555, .245, .315])
    ax_c = fig.add_axes([.455, .145, .245, .405])
    ax_d = fig.add_axes([.785, .145, .207, .725])
    design(ax_a)
    hydraulic_functions(ax_b, ax_c)
    sensitivity(ax_d)
    for ax in (ax_b, ax_c, ax_d):
        for spine in ax.spines.values():
            spine.set_visible(True)
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(STUDY / 'figures' / ('study_conceptual_overview.' + extension), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
