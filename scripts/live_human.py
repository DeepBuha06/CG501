"""Live human simulation: run the Weiss model on a moving rhombus and watch its guess.

    python scripts/live_human.py

Set the brightness and the shape, then press "Run". The real shape (dashed) crosses once from left to
right; the blue shape moves the way the model thinks it moves (Weiss et al. 2002, Eq. 1, computed now).
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider
from eye2echo import stimuli as st
from eye2echo.anim import human
from eye2echo.anim.common import MID

STEP = 1 / (human.PASS_S * 30)      # progress per frame at 30 frames per second

fig = plt.figure(figsize=(14, 8.5))
ax = fig.add_axes([0.03, 0.30, 0.94, 0.62])
status = fig.text(0.03, 0.24, '', fontsize=MID * 0.8)
bright = Slider(fig.add_axes([0.20, 0.13, 0.45, 0.04]), 'brightness (%)', 5, 100, valinit=100, valstep=5)
shape = RadioButtons(fig.add_axes([0.72, 0.04, 0.12, 0.15]), ('thin', 'fat'))
run = Button(fig.add_axes([0.87, 0.07, 0.10, 0.08]), 'Run')
state = dict(u=0.0, running=False)
WIDGETS = (bright, shape, run)

for w in WIDGETS:                     # readable control labels
    for t in w.ax.texts + getattr(w, 'labels', []) + [getattr(w, 'label', None), getattr(w, 'valtext', None)]:
        if t is not None:
            t.set_fontsize(16)


def current():
    angle = st.THIN_DEG if shape.value_selected == 'thin' else st.FAT_DEG
    c = bright.val / 100
    return angle, c, human.guess(angle, c)


def start(_):
    state.update(u=0.0, running=True)


def frame(_):
    angle, c, v = current()
    if state['running']:
        state['u'] = min(state['u'] + STEP, 1.0)
        state['running'] = state['u'] < 1.0
    human.draw(ax, angle, c, v, state['u'])
    status.set_text(human.describe(v))


run.on_clicked(start)
anim = FuncAnimation(fig, frame, interval=33, cache_frame_data=False)
plt.show()
