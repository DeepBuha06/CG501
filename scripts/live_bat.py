"""Live bat simulation: a target slides along the track; two guesses of where it is, from echoes.

    python scripts/live_bat.py

Tick "wall" to hide part of the track, tick "slows down" to make the target slow down behind the wall,
set the echo noise and how slow to play, then press "Run". Every run is a new seeded trial (the seed
is shown), so any run can be repeated.
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, Slider
from eye2echo import bat_sim as bs
from eye2echo.anim import bat
from eye2echo.anim.common import MID

fig = plt.figure(figsize=(14, 8.5))
ax = fig.add_axes([0.01, 0.28, 0.70, 0.66])
status = fig.text(0.03, 0.22, 'Press Run.', fontsize=MID * 0.8)
bat.legend(fig, ['nonpredictive', 'ours'], anchor=(0.995, 0.29))
opts = CheckButtons(fig.add_axes([0.02, 0.03, 0.25, 0.14]), ['wall', 'slows down behind wall'], [True, False])
noise = Slider(fig.add_axes([0.45, 0.12, 0.28, 0.04]), 'echo noise (mm)', 2, 80, valinit=10, valstep=2)
slow = Slider(fig.add_axes([0.45, 0.05, 0.28, 0.04]), 'slow motion (x)', 2, 12, valinit=6, valstep=1)
run = Button(fig.add_axes([0.80, 0.05, 0.15, 0.10]), 'Run')
state = dict(sim=None, t=0.0, t1=0.0, seed=0)
WIDGETS = (opts, noise, slow, run)

for w in WIDGETS:                     # readable control labels
    for t in w.ax.texts + getattr(w, 'labels', []) + [getattr(w, 'label', None), getattr(w, 'valtext', None)]:
        if t is not None:
            t.set_fontsize(16)


def start(_):
    wall, slows = opts.get_status()
    track = bs.change_profile(0.19) if slows else bs.smooth(bs.track_data()['standard'])
    state['seed'] += 1
    sim = bat.simulate(track, wall or slows, noise_mm=noise.val, seed=state['seed'])
    sim['aims'] = bat.sim_aims(sim)
    state.update(sim=sim, t=bat.time_at(track, 150), t1=bat.time_at(track, 850 if slows else 1450))


def frame(_):
    sim = state['sim']
    if sim is None:
        return
    state['t'] = min(state['t'] + 1000 / 30 / slow.val, state['t1'])
    i = int(state['t'])
    aims = {n: (a[i], p[i]) for n, (a, p) in sim['aims'].items()}
    bat.draw(ax, state['t'], sim['track'][i], sim['wall'], sim['echo_r'], sim['echo_x'], aims)
    text = bat.sim_readout(sim)(i)
    status.set_text(f'{text}      (run {state["seed"]})' if text else f'(run {state["seed"]})')


run.on_clicked(start)
anim = FuncAnimation(fig, frame, interval=33, cache_frame_data=False)
plt.show()
