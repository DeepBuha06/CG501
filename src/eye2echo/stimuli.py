"""Moving stimuli for the Weiss et al. (2002) replication: rhombus, grating, plaid, texture.

Every function returns three frames (t-1, t, t+1) of shape (3, size, size), mean luminance 0.5,
so they can go straight into weiss.estimate. Velocity v = (vx, vy) is in pixels per frame.
"""
import numpy as np
from scipy.special import erf

FIXED_SIDE_DEG = 60.0   # orientation of the rhombus's first pair of sides (our definition, see DECISIONS)
THIN_DEG, FAT_DEG = 40.0, 10.0


def _grid(size):
    """Pixel coordinates centred on the image."""
    c = (size - 1) / 2
    y, x = np.mgrid[0:size, 0:size].astype(float)
    return x - c, y - c


def rhombus(angle_deg, contrast, v=(1.0, 0.0), size=128, side=48.0, blur=1.5):
    """Rhombus with one pair of sides at FIXED_SIDE_DEG and the other at angle_deg (the "rhombus angle").

    Angle 0 gives a fat rhombus with horizontal top and bottom; as the angle rises toward 60 deg the
    rhombus gets thinner (interior angle 60 - angle_deg). Edges are soft: each pair of sides is an
    erf step of width `blur` pixels. Weiss et al. Fig. 2 and Fig. 4.
    """
    u1 = np.radians(FIXED_SIDE_DEG)
    u2 = np.radians(angle_deg)
    M = np.array([[np.cos(u1), np.cos(u2)], [np.sin(u1), np.sin(u2)]])
    Minv = np.linalg.inv(M)
    sin_between = abs(np.sin(u1 - u2))
    x, y = _grid(size)
    frames = []
    for t in (-1, 0, 1):
        px, py = x - v[0] * t, y - v[1] * t
        a = Minv[0, 0] * px + Minv[0, 1] * py
        b = Minv[1, 0] * px + Minv[1, 1] * py
        da = (side / 2 - np.abs(a)) * sin_between   # distance to the sides parallel to u2
        db = (side / 2 - np.abs(b)) * sin_between   # distance to the sides parallel to u1
        m = 0.25 * (1 + erf(da / blur)) * (1 + erf(db / blur))
        frames.append(0.5 * (1 + contrast * m))
    return np.array(frames)


def grating(contrast, normal_deg=0.0, speed=1.0, wavelength=16.0, size=128):
    """Drifting sine grating; it moves at `speed` along its normal (normal_deg from +x)."""
    th = np.radians(normal_deg)
    x, y = _grid(size)
    k = 2 * np.pi / wavelength
    proj = x * np.cos(th) + y * np.sin(th)
    return np.array([0.5 * (1 + contrast * np.sin(k * (proj - speed * t))) for t in (-1, 0, 1)])


def plaid(c1, c2, n1_deg, n2_deg, v=(1.0, 0.0), wavelength=16.0, size=128):
    """Sum of two gratings that both move with pattern velocity v.

    Grating i has its normal at ni_deg and therefore normal speed v . n_i. Type I: the two normals lie on
    opposite sides of v; type II: on the same side (Weiss et al. Fig. 5d, ref. 17).
    """
    frames = 0.5 * np.ones((3, size, size))
    for c, nd in ((c1, n1_deg), (c2, n2_deg)):
        th = np.radians(nd)
        normal_speed = v[0] * np.cos(th) + v[1] * np.sin(th)
        frames += grating(c, nd, normal_speed, wavelength, size) - 0.5
    return frames


def texture(v=(1.0, 0.0), size=128, seed=0, smooth=2.0):
    """Smoothed random texture moved by v with Fourier shifts (2-D information everywhere)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    base = np.fft.fft2(noise) * np.exp(-2 * (np.pi * smooth) ** 2 * (fx ** 2 + fy ** 2))
    frames = []
    for t in (-1, 0, 1):
        shift = np.exp(-2j * np.pi * (fx * v[0] * t + fy * v[1] * t))
        img = np.real(np.fft.ifft2(base * shift))
        frames.append(0.5 + 0.2 * img / img.std())
    return np.array(frames)
