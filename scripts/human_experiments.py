"""Replication of Weiss et al. (2002) Figs. 4d, 5a, 5b, 5d, 5e and the lambda fit to Fig. 4d.

Outputs: results/human/*.csv, results/numbers/human.json, figures fig01-fig05.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from eye2echo import weiss, stimuli as st, style
from eye2echo.stats import bootstrap_ci, save_numbers

OUT = style.RES / 'human'
CONTRASTS = np.logspace(-2, 0, 25)
ANGLES = np.arange(2, 46, 1.0)
FACTORS = (0.1, 1.0, 10.0)        # lambda x0.1, x1, x10 (Weiss Fig. 4d dotted lines)


def direction_error(A, b, lam):
    """Unsigned direction of v* from the true (horizontal) motion, degrees."""
    return abs(weiss.direction_deg(weiss.estimate_from_sums(A, b, lam)))


def rhombus_vs_contrast(lam):
    """Direction error of thin and fat rhombuses as contrast falls (Weiss Fig. 2, 4a-c)."""
    rows = []
    for shape, ang in (('thin', st.THIN_DEG), ('fat', st.FAT_DEG)):
        A1, b1 = weiss.sums(st.rhombus(ang, 1.0))
        for c in CONTRASTS:     # A and b scale with contrast^2, so reuse the contrast-1 sums
            for f in FACTORS:
                rows.append(dict(shape=shape, contrast=c, factor=f, error_deg=direction_error(c**2 * A1, c**2 * b1, f * lam)))
    return pd.DataFrame(rows)


def rhombus_angle_sums():
    """Sums A, b at contrast 1 for every rhombus angle (reused by the sweep and the fit)."""
    return {a: weiss.sums(st.rhombus(a, 1.0)) for a in ANGLES}


def model_curve(sums, angles, lam_eff):
    return np.array([direction_error(*sums[a], lam_eff) for a in angles])


def fit_lambda(sums, angles, directions):
    """Least-squares fit of log10(lambda_eff) to perceived direction vs rhombus angle."""
    loss = lambda lg: np.sum((model_curve(sums, angles, 10**lg) - directions) ** 2)
    grid = np.linspace(-4, 2, 61)
    start = grid[np.argmin([loss(g) for g in grid])]
    res = minimize_scalar(loss, bounds=(start - 0.2, start + 0.2), method='bounded')
    return 10**res.x, res.fun


def grating_speed(lam):
    """Perceived speed of a low-contrast grating relative to one at max contrast (Weiss Fig. 5a)."""
    A1, b1 = weiss.sums(st.grating(1.0))
    speed = lambda c: weiss.estimate_from_sums(c**2 * A1, c**2 * b1, lam)[0]
    rows = []
    for cmax in (0.4, 0.7):
        for lr in np.linspace(-2, 0, 21):
            rows.append(dict(cmax=cmax, log10_ratio=lr, relative_speed=speed(cmax * 10**lr) / speed(cmax)))
    return pd.DataFrame(rows)


def plaid_dir(c1, c2, n1, n2, lam):
    return weiss.direction_deg(weiss.estimate(st.plaid(c1, c2, n1, n2), lam))


def va_dir(n1, n2):
    """Vector average of the two normal velocities for pattern velocity (1, 0)."""
    vs = [np.cos(np.radians(n)) * np.array([np.cos(np.radians(n)), np.sin(np.radians(n))]) for n in (n1, n2)]
    v = np.mean(vs, axis=0)
    return np.degrees(np.arctan2(v[1], v[0]))


TYPE1, TYPE2 = (-11.15, 11.15), (50.0, 72.3)     # 22.3 deg separation, as in Weiss Fig. 5d


def plaids(lam):
    """Type I vs II (5d), direction vs separation (5e), bias vs contrast ratio (5b)."""
    d = [dict(plaid=t, contrast=c, direction=plaid_dir(c, c, *n, lam), va=va_dir(*n))
         for t, n in (('I', TYPE1), ('II', TYPE2)) for c in (0.5, 0.05)]
    seps = np.arange(4, 52, 2.0)
    e = [dict(separation=s, low=plaid_dir(0.05, 0.05, 85 - s, 85, lam), high=plaid_dir(0.5, 0.5, 85 - s, 85, lam),
              va=va_dir(85 - s, 85)) for s in seps]
    b = []
    for total in (0.05, 0.4):
        for l2 in np.linspace(0, 4, 17):
            r = 2**l2
            c1, c2 = total * r / (1 + r), total / (1 + r)
            b.append(dict(total=total, log2_ratio=l2, bias=plaid_dir(c1, c2, 45, -45, lam)))
    return pd.DataFrame(d), pd.DataFrame(e), pd.DataFrame(b)


def parameter_recovery(sums, angles, seed=1):
    """Simulate Fig. 4d data from known lambda + 2 deg Gaussian noise; refit; 30 repeats per value."""
    rng = np.random.default_rng(seed)
    rows = []
    for true in (0.3, 1.0, 3.0, 10.0):
        clean = model_curve(sums, angles, true)
        for rep in range(30):
            rec, _ = fit_lambda(sums, angles, clean + rng.normal(0, 2.0, len(angles)))
            rows.append(dict(true=true, recovered=rec, rep=rep))
    return pd.DataFrame(rows)


def main():
    style.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    lam_rh = weiss.ref_lambda(st.rhombus(st.THIN_DEG, 1.0))
    lam_gr = weiss.ref_lambda(st.grating(1.0))
    lam_pl = weiss.ref_lambda(st.plaid(1.0, 1.0, *TYPE2))

    rc = rhombus_vs_contrast(lam_rh)
    rc.to_csv(OUT / 'rhombus_contrast.csv', index=False)
    sums = rhombus_angle_sums()
    digit = pd.read_csv(style.ROOT / 'data/digitized/weiss_fig4d.csv')
    ang_d, dir_d = digit.rhombus_angle_deg.round().values, digit.perceived_direction_deg.values
    lam_fit, rss = fit_lambda(sums, ang_d, dir_d)
    rng = np.random.default_rng(2)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(ang_d), len(ang_d))
        boots.append(fit_lambda(sums, ang_d[i], dir_d[i])[0])
    ci = np.percentile(boots, [2.5, 97.5])
    rec = parameter_recovery(sums, ang_d)
    rec.to_csv(OUT / 'parameter_recovery.csv', index=False)
    curves = pd.DataFrame({'angle': ANGLES, **{f'x{f:g}': model_curve(sums, ANGLES, f * lam_fit) for f in FACTORS}})
    curves.to_csv(OUT / 'rhombus_angle_fit.csv', index=False)
    gr = grating_speed(lam_gr)
    gr.to_csv(OUT / 'grating.csv', index=False)
    p5d, p5e, p5b = plaids(lam_pl)
    p5d.to_csv(OUT / 'plaid_types.csv', index=False)
    p5e.to_csv(OUT / 'plaid_separation.csv', index=False)
    p5b.to_csv(OUT / 'plaid_bias.csv', index=False)

    pred = model_curve(sums, ang_d, lam_fit)
    low = rc[(rc.factor == 1) & np.isclose(rc.contrast, 0.1)]
    high = rc[(rc.factor == 1) & np.isclose(rc.contrast, 1.0)]
    g = lambda df, s: float(df[df['shape'] == s].error_deg.iloc[0])
    rec_med = rec.groupby('true').recovered.median()
    nums = dict(
        kappa=weiss.KAPPA, lamRhombus=lam_rh, lamGrating=lam_gr, lamPlaid=lam_pl,
        thinLowErr=g(low, 'thin'), fatLowErr=g(low, 'fat'), thinHighErr=g(high, 'thin'), fatHighErr=g(high, 'fat'),
        lamFit=lam_fit, lamFitLo=ci[0], lamFitHi=ci[1], fitRSS=rss, fitRMSE=np.sqrt(rss / len(ang_d)),
        fitMaxResid=np.max(np.abs(pred - dir_d)), fitResidAtThirty=pred[-1] - dir_d[-1], nDigitized=len(ang_d),
        recoveryMaxRelErr=float(np.max(np.abs(rec_med.values / rec_med.index.values - 1))),
        gratingRelSpeedFortyAtTenth=float(gr[(gr.cmax == 0.4) & np.isclose(gr.log10_ratio, -1)].relative_speed.iloc[0]),
        gratingRelSpeedSeventyAtTenth=float(gr[(gr.cmax == 0.7) & np.isclose(gr.log10_ratio, -1)].relative_speed.iloc[0]),
        plaidTwoLowDir=float(p5d[(p5d.plaid == 'II') & (p5d.contrast == 0.05)].direction.iloc[0]),
        plaidTwoHighDir=float(p5d[(p5d.plaid == 'II') & (p5d.contrast == 0.5)].direction.iloc[0]),
        plaidOneLowDir=float(abs(p5d[(p5d.plaid == 'I') & (p5d.contrast == 0.05)].direction.iloc[0])),
        plaidTwoVA=float(p5d[p5d.plaid == 'II'].va.iloc[0]),
        plaidBiasLowMax=float(p5b[p5b.total == 0.05].bias.max()), plaidBiasHighMax=float(p5b[p5b.total == 0.4].bias.max()),
    )
    save_numbers('human', nums)
    (OUT / 'fit.json').write_text(json.dumps({k: nums[k] for k in nums if k.startswith(('lamFit', 'fit'))}, indent=1))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for shape in ('thin', 'fat'):
        for f, ls in zip(FACTORS, (':', '-', ':')):
            d = rc[(rc['shape'] == shape) & (rc.factor == f)]
            ax.plot(d.contrast, d.error_deg, ls, color=style.COLORS[shape], lw=2 if f == 1 else 1.2,
                    label=f'{shape} rhombus ({shape == "thin" and st.THIN_DEG or st.FAT_DEG:.0f} deg)' if f == 1 else None)
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xlabel('Contrast (log scale, falling to the right)'); ax.set_ylabel('Direction error (deg)')
    ax.set_title('Faint thin rhombuses look diagonal; fat ones much less')
    ax.legend(loc='upper left'); ax.text(0.98, 0.04, 'dotted: lambda x0.1 and x10', transform=ax.transAxes,
                                         ha='right', color=style.INK2, fontsize=8)
    style.save(fig, 'fig01_rhombus_contrast')

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for f, ls in zip(FACTORS, (':', '-', ':')):
        ax.plot(curves.angle, curves[f'x{f:g}'], ls, color=style.COLORS['ours'], lw=2 if f == 1 else 1.2,
                label=f'model, fitted lambda' if f == 1 else ('model, lambda x0.1 / x10' if f == 0.1 else None))
    ax.plot(ang_d, dir_d, 'o', mfc='white', mec=style.INK, ms=7, mew=1.5, label='human (digitized, 1 subject)')
    ax.set_xlim(0, 45); ax.set_ylim(50, 0)
    ax.set_xlabel('Rhombus angle (deg)'); ax.set_ylabel('Perceived direction (deg)')
    ax.set_title('Fit to Weiss Fig. 4d: right trend, knee too soft')
    ax.legend(loc='lower left')
    style.save(fig, 'fig02_rhombus_angle_fit')

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for cmax, ls in ((0.7, '-'), (0.4, '--')):
        d = gr[gr.cmax == cmax]
        ax.plot(d.log10_ratio, d.relative_speed, ls, color=style.COLORS['ours'], label=f'max contrast {cmax:.0%}')
    ax.set_xlabel('log10 contrast ratio (low / high)'); ax.set_ylabel('Relative perceived speed')
    ax.set_title('Fainter gratings look slower (Weiss Fig. 5a)')
    ax.legend(loc='lower right')
    style.save(fig, 'fig03_grating')

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axs[0]
    x = np.arange(2)
    for i, (c, col) in enumerate(((0.5, style.COLORS['fat']), (0.05, style.COLORS['thin']))):
        vals = [p5d[(p5d.plaid == t) & (p5d.contrast == c)].direction.iloc[0] for t in ('I', 'II')]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=col, label=f'contrast {c:.0%}')
    ax.axhline(0, color=style.INK, lw=1); ax.axhline(p5d[p5d.plaid == 'II'].va.iloc[0], color=style.INK2, ls=':', lw=1)
    ax.text(1.45, p5d[p5d.plaid == 'II'].va.iloc[0], 'VA (type II)', va='bottom', ha='right', fontsize=8, color=style.INK2)
    ax.text(0, 1.5, 'both about 0 deg', ha='center', fontsize=8, color=style.INK2)
    ax.set_ylim(0, 68)
    ax.set_xticks(x, ['type I', 'type II']); ax.set_ylabel('Direction from veridical (deg)')
    ax.set_title('a  Faint type II plaids drift to VA'); ax.legend(loc='upper left')
    ax = axs[1]
    ax.plot(p5e.separation, p5e.low, color=style.COLORS['thin'], label='contrast 5%')
    ax.plot(p5e.separation, p5e.high, '--', color=style.COLORS['fat'], label='contrast 50%')
    ax.plot(p5e.separation, p5e.va, ':', color=style.INK2, lw=1.2, label='VA')
    ax.axhline(0, color=style.INK, lw=1)
    ax.set_xlabel('Component separation (deg)'); ax.set_ylabel('Direction from IOC (deg)')
    ax.set_title('b  VA to IOC as separation grows'); ax.legend(loc='upper right')
    ax = axs[2]
    for total, ls, col in ((0.05, '-', style.COLORS['thin']), (0.4, '--', style.COLORS['fat'])):
        d = p5b[p5b.total == total]
        ax.plot(d.log2_ratio, d.bias, ls, color=col, label=f'total contrast {total:.0%}')
    ax.set_xlabel('log2 contrast ratio'); ax.set_ylabel('Bias to stronger grating (deg)')
    ax.set_title('c  Bias shrinks at higher contrast'); ax.legend(loc='upper left')
    fig.tight_layout()
    style.save(fig, 'fig04_plaids')

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for t, d in rec.groupby('true'):
        ax.plot(np.full(len(d), t) * np.exp(np.random.default_rng(3).normal(0, 0.04, len(d))), d.recovered, 'o',
                ms=3, color=style.COLORS['ours'], alpha=0.5)
    lims = [0.2, 15]
    ax.plot(lims, lims, color=style.INK2, lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('True lambda'); ax.set_ylabel('Recovered lambda')
    ax.set_title('Parameter recovery (2 deg noise, 30 runs)')
    style.save(fig, 'fig05_param_recovery')
    print(json.dumps(nums, indent=1))


if __name__ == '__main__':
    main()
