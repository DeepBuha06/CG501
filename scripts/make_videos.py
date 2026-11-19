"""Render the three videos (1920x1080, 30 fps, H.264) and short GIF previews of each.

Each video replays the real simulations (seeded) or the real data; scripts/live_human.py and
scripts/live_bat.py show the same scenes live, with controls.
"""
from eye2echo.anim import common
common.use_agg()
from eye2echo.anim import human, bat

if __name__ == '__main__':
    human.render()
    bat.render_sim()
    bat.render_real()
    for name, start in (('human_rhombus', 25), ('bat_simulation', 45), ('bat_real_data', 40)):
        common.gif_preview(name, start)
