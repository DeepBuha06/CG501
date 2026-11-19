"""Head-aim models for a bat tracking a target on the Salles et al. (2020) track.

Units: positions in mm along the track, times in ms, velocities in mm/ms (= m/s), angles in rad.
Geometry (Salles Methods; createGetAllBats.m): bat 300 mm from the track, target in front of the bat
at track position 1000 mm, target angle atan((s - 1000) / 300), sound speed 346 mm/ms.

Every model sees echoes as (hit time h_j, perceived position x_j, return time r_j) and, at time t,
uses only echoes with r_j <= t. Models 1-3 follow Salles Methods; model 4 is our slow-prior line fit.
"""
import numpy as np

D_TRACK, CENTER, V_SOUND = 300.0, 1000.0, 346.0


def angle(s):
    """Target (or aim) angle of track position s, rad (Salles Methods: atan(s / d_track))."""
    return np.arctan((np.asarray(s) - CENTER) / D_TRACK)


def travel_time(s):
    """One-way sound travel time bat -> target at track position s, ms (createGetAllBats.m line 25)."""
    return np.sqrt((np.asarray(s) - CENTER) ** 2 + D_TRACK ** 2) / V_SOUND


def echoes(call_times, track_t, track_s):
    """Hit time h, true position at the hit, and return time r of each call.

    As in createGetAllBats.m, the call reaches the target after the travel time at the emission
    position, h = C + y(s(C)); the echo is back at r = h + y(s(h)).
    """
    C = np.sort(np.asarray(call_times, float))
    h = C + travel_time(np.interp(C, track_t, track_s))
    s_hit = np.interp(h, track_t, track_s)
    return h, s_hit, h + travel_time(s_hit)


# ---- the link: Weiss 1-D estimator and its echo form ----

def weiss_1d(ix, it, lam):
    """1-D Weiss Eq. 1: v* = -sum(Ix It) / (sum(Ix^2) + lambda)."""
    ix, it = np.asarray(ix), np.asarray(it)
    return -np.sum(ix * it) / (np.sum(ix * ix) + lam)


def echo_slope(h, x, lam):
    """Echo form: v* = sum(tau d) / (sum(tau^2) + lambda), tau = h - mean(h), d = x - mean(x).

    This is the MAP slope of x = a + v tau + noise(sigma_e) with a flat prior on a and a slow prior
    v ~ N(0, sigma_p^2), lambda = sigma_e^2 / sigma_p^2 (derivation: report, Eq. 2).
    It equals weiss_1d(tau, -d, lambda).
    """
    tau, d = h - np.mean(h), x - np.mean(x)
    return np.sum(tau * d) / (np.sum(tau * tau) + lam)


def predict_sd(h, t, sigma_e, lam):
    """Posterior SD of the predicted position a + v (t - mean(h)): sigma_e^2 (1/n + (t-hbar)^2 / (S + lambda))."""
    tau = h - np.mean(h)
    return sigma_e * np.sqrt(1 / len(h) + (t - np.mean(h)) ** 2 / (np.sum(tau * tau) + lam))


# ---- per-echo estimates (computed once per echo, then held until the next echo returns) ----

def window_fits(h, x, k, lam):
    """For each echo j: mean time, mean position, ridge slope over echoes j-k+1..j (ours, Eq. above)."""
    hb, xb, v = np.empty(len(h)), np.empty(len(h)), np.zeros(len(h))
    for j in range(len(h)):
        hh, xx = h[max(0, j - k + 1):j + 1], x[max(0, j - k + 1):j + 1]
        hb[j], xb[j] = hh.mean(), xx.mean()
        if len(hh) > 1:
            v[j] = echo_slope(hh, xx, lam)
    return hb, xb, v


def pair_velocity(h, x, k):
    """Salles velocity estimate: mean of the last k-1 two-echo velocities (createGetAllBats.m, usePings)."""
    pv = np.diff(x) / np.diff(h)
    v = np.zeros(len(h))
    for j in range(1, len(h)):
        v[j] = pv[max(0, j - k + 1):j].mean()
    return v


def aim_positions(t, h, x, r, k=5, lam=0.0):
    """Estimated target position at times t for the position-based models (dict of arrays, NaN before 1st echo).

    nonpredictive: position of the last returned echo (Salles, after Masters et al.).
    velocity: last echo + Salles velocity * (t - h_J).
    ours: ridge line through the last k echoes, read at t; ours_noprior: same with lambda = 0.
    """
    if len(h) == 0:
        return {m: np.full(len(t), np.nan) for m in ('nonpredictive', 'velocity', 'ours', 'ours_noprior')}
    J = np.searchsorted(r, t, side='right') - 1
    ok = J >= 0
    Jc = np.clip(J, 0, None)
    out = {}
    nan = lambda a: np.where(ok, a, np.nan)
    out['nonpredictive'] = nan(x[Jc])
    out['velocity'] = nan(x[Jc] + pair_velocity(h, x, k)[Jc] * (t - h[Jc]))
    for name, l in (('ours', lam), ('ours_noprior', 0.0)):
        hb, xb, v = window_fits(h, x, k, l)
        out[name] = nan(xb[Jc] + v[Jc] * (t - hb[Jc]))
    return out
