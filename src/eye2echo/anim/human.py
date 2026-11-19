"""Human simulation: a rhombus moves to the right; the Weiss model guesses its motion.

The guess is the model's real output: v* from Eq. 1 on the rhombus image at the given contrast, with the
same lambda as the human experiments. The real shape moves 1 unit per unit time to the right; the blue
"guess" shape moves with v*, from the same start. Used by the video (render) and by scripts/live_human.py.
"""
import numpy as np
from matplotlib.patches import Polygon
from .. import weiss, stimuli as st, style
from .common import figure, writer, card, hold, VID, FPS, BIG, MID, SMALL, BLUE

START, LENGTH = (250.0, 600.0), 1100.0             # path of the real shape (display units, y downward)
SIDE = 170.0
PASS_S = 7.0                                      # seconds for one crossing


def guess(angle_deg, contrast):
    """Model guess v* for a rhombus moving at (1, 0) px/frame (Weiss Eq. 1; A and b scale with contrast^2)."""
    lam = weiss.ref_lambda(st.rhombus(st.THIN_DEG, 1.0))
    A, b = weiss.sums(st.rhombus(angle_deg, 1.0))
    return weiss.estimate_from_sums(contrast**2 * A, contrast**2 * b, lam)


def corners(angle_deg, cx, cy):
    """Rhombus vertices: one side pair at FIXED_SIDE_DEG, the other at angle_deg (as stimuli.rhombus)."""
    u1 = np.array([np.cos(np.radians(st.FIXED_SIDE_DEG)), np.sin(np.radians(st.FIXED_SIDE_DEG))])
    u2 = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    return np.array([[cx, cy] + SIDE / 2 * (a * u1 + b * u2) for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def setup_axes(fig):
    ax = fig.add_axes([0.03, 0.13, 0.94, 0.72])
    return ax


def draw(ax, angle_deg, contrast, v, u):
    """Scene at progress u in [0, 1]: real shape, guess shape, both paths, labels."""
    ax.clear()
    ax.set_facecolor('#808080')
    ax.set_xlim(0, 1800); ax.set_ylim(780, 0); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    x0, y0 = START
    real = np.array([x0 + u * LENGTH, y0])
    seen = np.array([x0, y0]) + u * LENGTH * np.asarray(v)
    ax.plot([x0, real[0]], [y0, y0], color='black', lw=3, ls='--')
    ax.plot([x0, seen[0]], [y0, seen[1]], color=BLUE, lw=4)
    shade = 0.5 + 0.5 * contrast
    ax.add_patch(Polygon(corners(angle_deg, *real), closed=True, fc=(shade,) * 3, ec="black", lw=3, ls="--"))
    ax.add_patch(Polygon(corners(angle_deg, *seen), closed=True, fc=BLUE, alpha=0.35, ec=BLUE, lw=4))
    ax.text(real[0], y0 + SIDE * 0.9, 'real shape', ha='center', fontsize=SMALL, color='white', weight='bold')
    ax.text(seen[0], seen[1] - SIDE * 0.8, "computer's guess", ha='center', fontsize=SMALL, color='#cfe0ff',
            weight='bold')


def describe(v):
    """Plain-words description of a guess v relative to the true motion (1, 0)."""
    off = np.degrees(np.arctan2(v[1], v[0]))
    slower = 100 * (1 - np.hypot(*v))
    where = 'same direction' if abs(off) < 3 else f'{abs(off):.0f}° {"downward" if off > 0 else "upward"}'
    return f'Guess: {where}, {slower:.0f}% slower than real'


def scene(fig, w, title, angle_deg, contrast, takeaway):
    """One crossing (PASS_S s). The guess in words shows mid-way, the takeaway over the last 40%, then 2 s hold."""
    fig.clear()
    ax = setup_axes(fig)
    fig.text(0.03, 0.9, title, fontsize=BIG, weight='bold')
    fig.text(0.97, 0.9, f'brightness of the shape: {contrast:.0%}', fontsize=MID, ha='right')
    v = guess(angle_deg, contrast)
    status = fig.text(0.03, 0.06, '', fontsize=MID)
    n = int(PASS_S * FPS)
    for i in range(n + 1):
        draw(ax, angle_deg, contrast, v, i / n)
        status.set_text('' if i < n // 6 else describe(v) if i < 0.6 * n else takeaway)
        w.grab_frame()
    hold(w, 2)


def render():
    """videos/human_rhombus.mp4: about 75 s, one idea per scene."""
    import json
    h = json.loads((style.RES / 'numbers/human.json').read_text())
    fig = figure()
    w = writer()
    VID.mkdir(exist_ok=True)
    with w.saving(fig, str(VID / 'human_rhombus.mp4'), dpi=100):
        card(fig, w, 'How does the brain guess motion?',
             'A computer model of human vision (Weiss et al., 2002)\n'
             'watches a moving shape and guesses its direction and speed.\n\n'
             'Dashed: the real shape.   Blue: the model\'s guess.')
        scene(fig, w, '1. Thin shape, clearly visible', st.THIN_DEG, 1.0,
              'Clear picture: the guess follows the real shape.')
        scene(fig, w, '2. Same thin shape, but faint', st.THIN_DEG, 0.1,
              'Faint picture: the guess is slower and slants upward.')
        card(fig, w, 'Why?',
             'The model assumes that things usually move slowly.\n'
             'A faint picture is weak evidence, so that belief wins.\n'
             'A thin shape\'s edges say little about up-down motion,\n'
             'so the guess drifts at a slant. People see the same illusion.')
        scene(fig, w, '3. Fat shape, equally faint', st.FAT_DEG, 0.1,
              'Fat shape: much smaller mistake, because its edges point in different directions.')
        card(fig, w, 'Result (direction mistake of the guess)',
             f'Thin shape, clear:  {h["thinHighErr"]:.0f}°\n'
             f'Thin shape, faint:  {h["thinLowErr"]:.0f}°\n'
             f'Fat shape, faint:  {h["fatLowErr"]:.0f}°\n\n'
             'Thin and faint looks most wrong, as in the human experiments.')
