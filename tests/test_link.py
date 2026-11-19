"""Checks: echo-form estimator = 1-D Weiss form; posterior band calibration."""
import numpy as np
from eye2echo import bat_models as bm


def test_echo_form_equals_weiss_1d():
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = np.sort(rng.uniform(0, 200, 6))
        x = rng.normal(0, 50, 6) + 0.75 * h
        lam = rng.uniform(0, 5000)
        tau, d = h - h.mean(), x - x.mean()
        assert np.isclose(bm.echo_slope(h, x, lam), bm.weiss_1d(tau, -d, lam))


def test_no_prior_is_least_squares_slope():
    rng = np.random.default_rng(1)
    h = np.sort(rng.uniform(0, 100, 8))
    x = 3 + 0.75 * h + rng.normal(0, 5, 8)
    assert np.isclose(bm.echo_slope(h, x, 0.0), np.polyfit(h, x, 1)[0])


def test_band_covers_95_percent_when_world_matches_prior():
    rng = np.random.default_rng(2)
    sigma_e, sigma_p, n, t_pred = 20.0, 1.0, 5, 250.0     # mm, mm/ms, echoes, ms (prediction 150 ms ahead)
    lam = sigma_e**2 / sigma_p**2
    h = np.arange(n) * 25.0
    inside = 0
    trials = 20000
    for _ in range(trials):
        a, v = rng.uniform(-500, 500), rng.normal(0, sigma_p)
        x = a + v * (h - h.mean()) + rng.normal(0, sigma_e, n)
        pred = x.mean() + bm.echo_slope(h, x, lam) * (t_pred - h.mean())
        truth = a + v * (t_pred - h.mean())
        inside += abs(pred - truth) <= 1.96 * bm.predict_sd(h, t_pred, sigma_e, lam)
    assert abs(inside / trials - 0.95) < 0.01


def test_echo_timing_matches_geometry():
    # target straight in front: 300 mm away, 300/346 ms one way
    assert np.isclose(bm.travel_time(1000.0), 300 / 346)
    t = np.arange(0, 2000.0)
    h, s, r = bm.echoes([100.0], t, np.full_like(t, 1000.0))
    assert np.isclose(h[0], 100 + 300 / 346) and np.isclose(r[0], 100 + 600 / 346)
