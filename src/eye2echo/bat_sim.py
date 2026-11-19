"""Simulated bat trials on the Salles et al. (2020) setup: target profiles, calls, echoes, occluder, metrics.

Track profiles are ported from the authors' track_data/track_data.m (position in mm at 1 ms steps).
"""
import numpy as np
from . import bat_models as bm

OCCLUDER = (450.0, 630.0)    # hidden track segment, mm (plotOccluder.m: -55 to -37 cm from the centre)
WINDOW = (470.0, 1060.0)     # tracking window, mm (Salles Methods: 47 to 106 cm)


def _profile(accel, n):
    a = np.zeros(n)
    for (i0, i1), val in accel:      # MATLAB 1-based inclusive index ranges
        a[i0 - 1:i1] = val
    track = np.repeat(np.cumsum(np.cumsum(a)), 10)
    return track / track.max() * 1575


def track_data():
    """Standard, velocity-change fast, velocity-change slow profiles of track_data.m (mm, 1 ms steps)."""
    std = _profile([((1, 25), 1), ((176, 200), -1)], 200)
    n = 60
    fast = _profile([((1, 25), 0.34285), ((100 - n + 1, 100), -0.34285 * 25 / n), ((126, 150), 0.34285),
                     ((246, 270), -0.34285)], 270)
    slow = _profile([((1, 25), 0.34285), ((100 - n + 1, 100), -0.34285 * 25 / n), ((136, 160), 0.183),
                     ((356, 380), -0.183)], 380)
    return {'standard': std, 'fast': fast, 'slow': slow}


def change_profile(v2, v1=0.75, x_change=500.0, ramp_ms=250.0, length=1600.0):
    """Generic profile for the velocity-change sweep: accelerate to v1 (mm/ms), switch to v2 at x_change.

    The 250 ms ramp to the cruising speed copies the standard profile (Table 1: 250 ms to maximum).
    """
    t = np.arange(0, 6000.0)
    v = np.where(t < ramp_ms, v1 * t / ramp_ms, v1)
    s = np.cumsum(v)
    i = np.searchsorted(s, x_change)
    s[i:] = s[i] + np.cumsum(np.full(len(t) - i, v2))
    s = np.minimum(s, length)
    return s[:np.searchsorted(s, length) + 1]


def call_times(rng, rate_hz, t_end, cv):
    """Call schedule with gamma-distributed intervals (mean 1000/rate ms, coefficient of variation cv)."""
    mean = 1000.0 / rate_hz
    k = 1 / cv**2
    iv = rng.gamma(k, mean / k, int(t_end / mean * 2) + 10)
    c = rng.uniform(0, mean) + np.cumsum(iv)
    return c[c < t_end]


def angle_error_deg(est_pos, true_pos, offset_deg=0.0):
    """Aim angle minus true target angle, degrees."""
    return np.degrees(bm.angle(est_pos) - bm.angle(true_pos)) + offset_deg


def metrics(t, s, err, tol=5.0, occluder=OCCLUDER):
    """Summary metrics of one trial's angle error (deg).

    rms: RMS error inside the tracking window; off: ms with |err| > tol inside the window;
    exit: |err| when the target leaves the occluder; recovery: ms after the exit until |err| < tol
    for 50 ms in a row (NaN if never within the trial).
    """
    win = (s >= WINDOW[0]) & (s <= WINDOW[1])
    e = err[win]
    out = dict(rms=np.sqrt(np.nanmean(e**2)), off=np.sum(~(np.abs(e) <= tol)))
    if occluder:
        i_exit = np.searchsorted(s, occluder[1])
        out['exit'] = abs(err[i_exit])
        good = np.abs(err[i_exit:]) < tol
        run = np.convolve(good, np.ones(50), 'valid') == 50
        out['recovery'] = float(np.argmax(run)) if run.any() else np.nan
    return out


def smooth(track, width=10):
    """10 ms moving average of a track_data.m profile, which is a 10 ms staircase; used in simulation
    so that the true target moves continuously (the real-data analysis keeps the staircase)."""
    padded = np.concatenate([np.full(width, track[0]), track, np.full(width, track[-1])])
    return np.convolve(padded, np.ones(width) / width, 'same')[width:-width]
