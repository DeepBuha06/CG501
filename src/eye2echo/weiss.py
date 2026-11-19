"""Bayesian velocity estimator of Weiss, Simoncelli & Adelson (2002), Nat. Neurosci. 5:598.

Coordinates: x = column index, y = row index, t = frame index. Velocities are in pixels per frame.
"""
import numpy as np


def derivatives(frames):
    """Ix, Iy, It at the middle frame by central differences over three frames.

    frames: array (3, H, W). Ix and Iy are taken on the middle frame (one pixel border dropped),
    It = (I[t+1] - I[t-1]) / 2. These are the derivatives in Weiss et al. Eq. 4.
    """
    f0, f1, f2 = frames
    ix = (f1[1:-1, 2:] - f1[1:-1, :-2]) / 2
    iy = (f1[2:, 1:-1] - f1[:-2, 1:-1]) / 2
    it = (f2[1:-1, 1:-1] - f0[1:-1, 1:-1]) / 2
    return ix, iy, it


def sums(frames):
    """A (2x2) and b (2,) of Weiss Eq. 1, summed over the whole image."""
    ix, iy, it = derivatives(frames)
    A = np.array([[np.sum(ix * ix), np.sum(ix * iy)],
                  [np.sum(ix * iy), np.sum(iy * iy)]])
    b = np.array([np.sum(ix * it), np.sum(iy * it)])
    return A, b


def estimate(frames, lam):
    """Posterior mean velocity v* = -(A + lam I)^-1 b (Weiss Eq. 1), lam = sigma^2 / sigma_p^2."""
    A, b = sums(frames)
    return -np.linalg.solve(A + lam * np.eye(2), b)


def estimate_from_sums(A, b, lam):
    """Eq. 1 from precomputed sums; lets a sweep over lam reuse one derivative pass."""
    return -np.linalg.solve(A + lam * np.eye(2), b)


def direction_deg(v):
    """Direction of v in degrees, measured from +x; positive means toward +y (down the rows)."""
    return np.degrees(np.arctan2(v[1], v[0]))


KAPPA = 0.002   # one prior strength for all human experiments, relative to stimulus energy (DECISIONS)


def ref_lambda(frames_contrast1, kappa=KAPPA):
    """lambda = kappa * trace(A) of the stimulus at contrast 1.

    Weiss et al. set the free parameter per experiment; we instead scale one kappa by each stimulus
    family's gradient energy, so a full-field grating and a small rhombus get comparable priors.
    """
    return kappa * np.trace(sums(frames_contrast1)[0])
