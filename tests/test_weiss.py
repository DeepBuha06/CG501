"""Known-answer checks for the Weiss et al. (2002) estimator (Eq. 1)."""
import numpy as np
from eye2echo import weiss, stimuli as st

LAM0 = weiss.ref_lambda(st.rhombus(st.THIN_DEG, 1.0))   # experiment value, 100-fold range below


def test_no_prior_recovers_texture_velocity():
    v = np.array([0.6, -0.3])
    est = weiss.estimate(st.texture(v, smooth=3.0), lam=0.0)
    assert np.allclose(est, v, atol=0.02)


def test_zero_contrast_gives_zero():
    assert np.allclose(weiss.estimate(st.rhombus(40, 0.0), LAM0), 0.0)


def test_huge_prior_gives_zero():
    assert np.linalg.norm(weiss.estimate(st.rhombus(40, 1.0), 1e9)) < 1e-6


def test_grating_gives_only_normal_velocity():
    # grating with normal at 30 deg; lambda -> 0+ picks the slowest velocity on the constraint line
    est = weiss.estimate(st.grating(1.0, normal_deg=30.0, speed=1.0), lam=1e-9)
    # 0.5 deg tolerance: central differences scale Ix and Iy of an oblique sinusoid slightly differently
    assert np.isclose(weiss.direction_deg(est), 30.0, atol=0.5)
    assert np.isclose(np.linalg.norm(est), 1.0, atol=0.03)


def test_grating_matches_closed_form():
    frames = st.grating(0.5, speed=1.0)
    A, b = weiss.sums(frames)
    S = A[0, 0]
    v0 = -b[0] / S        # 1-D speed at lambda = 0 (along the grating normal, here +x)
    for lam in (0.1, 1.0, 10.0, 100.0):
        assert np.isclose(weiss.estimate_from_sums(A, b, lam)[0], v0 * S / (S + lam), rtol=1e-9)


def test_thin_never_more_veridical_than_fat_and_low_never_more_than_high():
    sums = {(a, c): weiss.sums(st.rhombus(a, c)) for a in (st.THIN_DEG, st.FAT_DEG) for c in (1.0, 0.1)}
    for lam in LAM0 * np.logspace(-1, 1, 21):
        err = {k: abs(weiss.direction_deg(weiss.estimate_from_sums(A, b, lam))) for k, (A, b) in sums.items()}
        for c in (1.0, 0.1):
            assert err[(st.THIN_DEG, c)] >= err[(st.FAT_DEG, c)]
        for a in (st.THIN_DEG, st.FAT_DEG):
            assert err[(a, 0.1)] >= err[(a, 1.0)]
