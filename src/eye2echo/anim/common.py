"""Shared video helpers: 1920x1080 figures, text cards, H.264 MP4 writer, GIF preview.

Text sizes are in points at 100 dpi (1 pt = 1.39 px), so the smallest text (SMALL = 22 pt) is about
30 px tall on a 1080p screen.
"""
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from .. import style

FPS = 30
VID = style.ROOT / 'videos'
BIG, MID, SMALL = 40, 30, 22
BLUE, PURPLE, ORANGE = style.COLORS['ours'], style.COLORS['nonpredictive'], style.COLORS['predictive']


def figure():
    """A white 1920x1080 figure."""
    style.setup()
    plt.rcParams.update({'font.size': SMALL})
    return plt.figure(figsize=(19.2, 10.8), dpi=100)


def writer():
    return FFMpegWriter(fps=FPS, codec='libx264', bitrate=6000,
                        extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '20'])


def card(fig, w, title, body):
    """Text card that builds up line by line, like a presenter clicking through.

    The title shows for 1 s, then each line appears and stays for 0.6 s + 4 words per second
    (blank lines add a short gap); the full card is held 1.5 s at the end.
    """
    fig.clear()
    fig.text(0.08, 0.72, title, fontsize=BIG + 6, weight='bold', color=style.INK, va='bottom')
    hold(w, 1)
    y = 0.64
    for line in body.split('\n'):
        if line.strip():
            fig.text(0.08, y, line, fontsize=MID, color=style.INK, va='top')
            hold(w, 0.6 + len(line.split()) / 4)
        y -= 0.065 if line.strip() else 0.035
    hold(w, 1.5)


def hold(w, seconds):
    for _ in range(int(seconds * FPS)):
        w.grab_frame()


def gif_preview(name, start=0.0, seconds=6.0, width=640):
    """Short GIF preview of videos/<name>.mp4 (two-pass palette for clean colours)."""
    src, dst = VID / f'{name}.mp4', VID / f'{name}.gif'
    vf = f'fps=12,scale={width}:-1:flags=lanczos'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(start), '-t', str(seconds), '-i', str(src),
                    '-vf', f'{vf},split[a][b];[a]palettegen[p];[b][p]paletteuse', str(dst)], check=True)


def use_agg():
    """Render off-screen (videos); the live demos keep the interactive backend."""
    matplotlib.use('Agg')
