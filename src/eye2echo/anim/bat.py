"""Bat simulation and real-data videos: a top view of the Salles et al. (2020) setup.

A target slides along a straight track 300 mm in front of the bat. Each returned echo is drawn as an
orange dot on the track where the echo says the target was. Two guesses of where the target is now:
the last echo (purple) and our rule, a line through the last 5 echoes continued to now (blue).
Everything is shown SLOW times slower than real. Used by the videos and by scripts/live_bat.py.
"""
import json
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from .. import bat_models as bm, bat_sim as bs, style
from .common import figure, writer, card, hold, VID, FPS, BIG, MID, SMALL, BLUE, PURPLE, ORANGE

SLOW = 6                  # video seconds per second of trial time
FADE = 300.0              # ms an echo dot stays visible
NAMES = {'nonpredictive': 'last echo', 'ours': 'our rule (line through 5 echoes)',
         'predictive': 'prediction + fixed lead', 'head': 'real bat heads (average)'}
COLS = {'nonpredictive': PURPLE, 'ours': BLUE, 'predictive': BLUE, 'head': 'black'}


def simulate(track, wall, noise_mm=10.0, seed=0):
    """One simulated trial: echoes (return time, reported position) and each guess at every ms."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(track), dtype=float)
    h, s_hit, r = bm.echoes(bs.call_times(rng, 20.0, t[-1], cv=0.77), t, track)
    keep = ~((s_hit >= bs.OCCLUDER[0]) & (s_hit <= bs.OCCLUDER[1])) if wall else np.ones(len(h), bool)
    x = s_hit[keep] + rng.normal(0, noise_mm, keep.sum())
    est = bm.aim_positions(t, h[keep], x, r[keep], k=5, lam=noise_mm**2 / 1.0**2)
    return dict(track=track, wall=wall, echo_r=r[keep], echo_x=x,
                guesses={n: est[n] for n in ('nonpredictive', 'ours')})


def miss_deg(pos, target_pos):
    """How far (deg) the bat would point from the target if it aimed at pos."""
    return abs(np.degrees(bm.angle(pos) - bm.angle(target_pos)))


def ray(theta_deg, length):
    th = np.radians(theta_deg)
    return [1000, 1000 + length * np.sin(th)], [0, length * np.cos(th)]


def draw(ax, t, s, wall, echo_r, echo_x, aims):
    """Top view at trial time t (ms). aims: {name: (angle_deg, track position or None)}."""
    ax.clear()
    ax.set_xlim(0, 1700); ax.set_ylim(-110, 400); ax.set_aspect('equal'); ax.axis('off')
    ax.plot([0, 1700], [300, 300], color='#c9c8c3', lw=10, solid_capstyle='butt', zorder=1)
    ax.text(10, 335, 'track', fontsize=SMALL, color=style.INK2)
    hidden = wall and bs.OCCLUDER[0] <= s <= bs.OCCLUDER[1]
    if wall:
        ax.add_patch(Rectangle((bs.OCCLUDER[0], 268), bs.OCCLUDER[1] - bs.OCCLUDER[0], 64, color='#5f5e5a', zorder=4))
        ax.text(np.mean(bs.OCCLUDER), 348, 'wall: no echoes', ha='center', fontsize=SMALL, color=style.INK)
    new = (echo_r <= t) & (echo_r > t - FADE)
    for r, x in zip(echo_r[new], echo_x[new]):
        ax.add_patch(Circle((x, 300), 11, color=ORANGE, alpha=1 - (t - r) / FADE, zorder=3))
    for name, (ang, pos) in aims.items():
        if not np.isfinite(ang):
            continue
        x, y = ray(ang, 300 / np.cos(np.radians(ang)))      # from the bat to the track
        ax.plot(x, y, color=COLS[name], lw=7 if name == 'head' else 3.5, alpha=1 if name == 'head' else 0.85,
                zorder=6 if name == 'head' else 5)
        if pos is not None:
            ax.add_patch(Circle((pos, 300), 30, fill=False, ec=COLS[name], lw=5, zorder=7))
    ax.add_patch(Circle((s, 300), 20, fc='white' if hidden else 'black', ec='black', lw=3,
                        ls='--' if hidden else '-', zorder=8))
    ax.add_patch(Polygon([[965, -45], [1035, -45], [1000, 22]], color=style.INK, zorder=9))
    ax.text(1000, -100, 'bat', ha='center', fontsize=MID)


def legend(fig, names, anchor=(0.98, 0.12), wall=True):
    handles = [Line2D([], [], marker='o', ls='', ms=16, color='black', label='target (white = hidden)' if wall else 'target'),
               Line2D([], [], marker='o', ls='', ms=12, color=ORANGE, label='echo: where the target was')]
    handles += [Line2D([], [], color=COLS[n], lw=7 if n == 'head' else 4, label=NAMES[n]) for n in names]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=anchor, fontsize=SMALL, frameon=False)


def play(fig, w, title, takeaway, sim, t0, t1, readout, pause_at=None, pause_text='', slow=SLOW):
    """Animate one trial from t0 to t1 ms, slowed `slow` times; optionally freeze at pause_at with a note."""
    fig.clear()
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.72])
    fig.text(0.03, 0.92, title, fontsize=BIG, weight='bold')
    legend(fig, list(sim['aims']), wall=sim['wall'])
    status = fig.text(0.03, 0.04, '', fontsize=MID)
    step = 1000 / FPS / slow
    times = np.arange(t0, t1, step)
    late = 0.7 * len(times)                         # show the takeaway over the last 30%
    for k, t in enumerate(times):
        i = int(t)
        aims = {n: (a[i], p[i] if p is not None else None) for n, (a, p) in sim['aims'].items()}
        draw(ax, t, sim['track'][i], sim['wall'], sim['echo_r'], sim['echo_x'], aims)
        if k % 6 == 0:
            status.set_text(takeaway if k > late else readout(i))
        w.grab_frame()
        if pause_at is not None and t <= pause_at < t + step:
            note = fig.text(0.03, 0.84, pause_text, fontsize=MID, weight='bold', color=style.INK,
                            bbox=dict(fc='#fff3c4', ec='none', pad=12))
            hold(w, 2.5)
            note.remove()
    hold(w, 2)


def sim_aims(sim):
    return {n: (np.degrees(bm.angle(p)), p) for n, p in sim['guesses'].items()}


def sim_readout(sim):
    def text(i):
        m = {n: miss_deg(p[i], sim['track'][i]) for n, p in sim['guesses'].items()}
        if not all(np.isfinite(list(m.values()))):
            return ''
        return f'How far off each guess points:   last echo {m["nonpredictive"]:.0f}°     our rule {m["ours"]:.0f}°'
    return text


def time_at(track, pos):
    return float(np.argmax(track >= pos))


def render_sim():
    """videos/bat_simulation.mp4: steady target, hidden target, hidden slow-down (about 100 s)."""
    s = json.loads((style.RES / 'numbers/sim.json').read_text())
    std = bs.smooth(bs.track_data()['standard'])
    slow = bs.change_profile(0.19)
    fig = figure()
    w = writer()
    with w.saving(fig, str(VID / 'bat_simulation.mp4'), dpi=100):
        card(fig, w, 'How does a bat follow a target in the dark?',
             'It clicks about 20 times a second.\n'
             'Each echo (orange dot) shows where the target was a moment ago.\n\n'
             'Purple guess: point at the last echo.\n'
             'Blue guess (our rule): continue a line through the last 5 echoes.\n\n'
             'Simulation of the Salles et al. (2020) setup, slowed down 6 times.')
        scenes = [
            ('1. Target moving steadily', std, False, 1450, None, '',
             'Both stay close, but "last echo" always lags a little behind the target.'),
            ('2. Target goes behind a wall', std, True, 1450, time_at(std, bs.OCCLUDER[1]),
             'Target comes out. Which guess was right?',
             'Behind the wall, "last echo" got stuck. Our rule kept moving and was right.'),
            ('3. Target slows down behind the wall', slow, True, 760, time_at(slow, bs.OCCLUDER[1]),
             'It slowed down while hidden. Our rule did not know and ran ahead.',
             'Prediction can be wrong: if the target changes speed while hidden.'),
        ]
        for k, (title, track, wall, end, pause, note, takeaway) in enumerate(scenes):
            sim = simulate(track, wall, seed=40 + k)
            sim['aims'] = sim_aims(sim)
            play(fig, w, title, takeaway, sim, time_at(track, 150), time_at(track, end), sim_readout(sim), pause, note)
        card(fig, w, f'Result over {s["nTrialsPerCell"]} simulated runs per situation',
             'Average miss, in degrees (lower is better):\n\n'
             f'Steady target:  our rule {s["steadyRmsOurs"]:.1f}°,  last echo {s["steadyRmsNonpred"]:.1f}°\n'
             f'Coming out from behind a long wall:  our rule {s["occl360ExitOurs"]:.0f}°,  last echo {s["occl360ExitNonpred"]:.0f}°\n'
             f'Slowed down while hidden:  our rule {s["hiddenSlowExitOurs"]:.0f}°,  last echo {s["hiddenSlowExitNonpred"]:.0f}°')


def render_real():
    """videos/bat_real_data.mp4: recorded head direction of real bats vs the two guesses (about 50 s)."""
    r = json.loads((style.RES / 'numbers/real.json').read_text())
    tr = pd.read_csv(style.RES / 'bat_real/mean_traces_standard.csv')
    std = bs.track_data()['standard']
    import pickle
    trial = next(t for t in pickle.loads((style.ROOT / 'data/external/trials_cache.pkl').read_bytes())
                 if t['bat'] == 0 and t['condition'] == 'B-L')
    calls = np.sort(trial['calls'][(trial['calls'] > 0) & (trial['calls'] < len(std))])
    _, s_hit, echo_r = bm.echoes(calls, np.arange(len(std), dtype=float), std)
    ms = np.arange(len(std))
    col = lambda c: np.interp(ms, tr.t_ms.values, tr[c].values)
    head, target = col('head_deg'), col('target_deg')
    base = dict(track=std, wall=False, echo_r=echo_r, echo_x=s_hit)

    def lead_text(i):
        d = head[i] - target[i]
        return f'The heads point {abs(d):.0f}° {"ahead of" if d > 0 else "behind"} the target'

    def match_text(i):
        return (f'Angle between the real heads and each guess:   last echo {abs(col_np[i] - head[i]):.0f}°'
                f'     prediction + lead {abs(col_p[i] - head[i]):.0f}°')
    col_np, col_p = col('nonpredictive_deg'), col('predictive_deg')
    cross = time_at(std, 1000)                    # target straight in front of the bat
    t0, t1 = cross - 660, cross + 70              # window tested in the paper and in real_data.py
    fig = figure()
    w = writer()
    with w.saving(fig, str(VID / 'bat_real_data.mp4'), dpi=100):
        card(fig, w, 'Real bats',
             f'{r["nBats"]} bats were filmed turning their heads to follow the target\n'
             f'(data published by Salles, Diebold & Moss, 2020; {r["nBL"]} runs).\n\n'
             'Black line: where the bats\' heads pointed, on average.\nSlowed down 12 times.')
        play(fig, w, 'Where do real bats point their heads?', 'The heads point ahead of the target: they predict.',
             dict(base, aims={'head': (head, None)}), t0, t1, lead_text, slow=12)
        play(fig, w, 'Real heads vs two guesses made from the bats\' real clicks', 'The heads follow the prediction, not the last echo.',
             dict(base, aims={'head': (head, None), 'nonpredictive': (col_np, None), 'predictive': (col_p, None)}),
             t0, t1, match_text, slow=12)
        card(fig, w, 'Result',
             f'In the 0.7 s before the target passes in front of the bat,\n'
             f'the heads point on average {r["leadMeanDeg"]:.0f}° ahead of the target.\n\n'
             f'Mismatch with the real heads (lower is better):\n'
             f'prediction + lead {r["rssPredictive"]:.2f},  last echo {r["rssNonpredictive"]:.1f}\n\n'
             'Adding the "things move slowly" belief did not make the match better.')
