"""One plotting style for every figure and video: fonts, colours per model, saving.

Palette: the validated categorical palette of the dataviz reference (light mode, checked with its
validate_palette.js: all hard checks pass; aqua and magenta are below 3:1 contrast, so every
figure carries a legend and distinct line styles).
"""
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'figures'
RES = ROOT / 'results'

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
COLORS = {
    'ours': '#2a78d6',          # slow-prior line fit
    'predictive': '#eb6834',    # Salles predictive (velocity + offset)
    'velocity': '#1baf7a',      # Salles velocity estimation
    'nonpredictive': '#6250d6', # last echo
    'ours_noprior': '#e87ba4',  # ablation, lambda = 0
    'fixed': '#008300',         # last echo + fixed offset
    'data': INK,                # measured head angle
    'target': '#8a8984',
    'thin': '#2a78d6', 'fat': '#eb6834',
}
LABELS = {
    'nonpredictive': 'Non-predictive (last echo)',
    'fixed': 'Fixed head angle',
    'velocity': 'Velocity estimation',
    'predictive': 'Predictive (Salles)',
    'ours': 'Slow-prior line fit (ours)',
    'ours_noprior': 'Line fit, no prior',
}
STYLES = {'nonpredictive': ':', 'fixed': (0, (5, 2)), 'velocity': '-.', 'predictive': '--',
          'ours': '-', 'ours_noprior': (0, (1, 1.5))}


def setup():
    """Apply the shared rcParams (white background, recessive axes, readable labels)."""
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 2.0, 'legend.frameon': False,
        'legend.fontsize': 9, 'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
    })


def save(fig, name):
    """Save a figure as 300 dpi PNG and as PDF in figures/."""
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
