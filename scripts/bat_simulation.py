"""Simulated head aim on the Salles et al. (2020) setup; sweeps, statistics, hypotheses H2-H4.

Every cell runs N_TRIALS seeded trials; all models see the same echoes in a trial, so comparisons are
paired (Wilcoxon signed-rank, rank-biserial r, Holm correction over every test in this script).
Outputs: results/bat_sim/*.csv, results/numbers/sim.json, figures fig06-fig08.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eye2echo import bat_sim as bs, bat_models as bm, style
from eye2echo.stats import bootstrap_ci, wilcoxon, holm, save_numbers

OUT = style.RES / 'bat_sim'
N_TRIALS = 500
N_CURVES = 200        # trials averaged for the error-over-time curves (fig06)
BASE = dict(rate_hz=20.0, noise_mm=10.0, k=5, sigma_p=1.0)   # 20 Hz: median real call interval 51 ms
CV = 0.77                                                     # real call-interval CV (scripts/real_data.py)
OFFSET = 6.24                                                 # Salles predictive model, deg
TOL = 5.0                                                     # off-target tolerance, deg
MODELS = ['nonpredictive', 'velocity', 'predictive', 'ours', 'ours_noprior']


def simulate(track, seed, n=N_TRIALS, occluder=bs.OCCLUDER, event=None, curves=False, **kw):
    """Run n trials; return per-trial metrics (long table) and, optionally, the mean error curve per model."""
    p = {**BASE, **kw}
    rng = np.random.default_rng(seed)
    rows, errs = [], {m: [] for m in MODELS}
    for i in range(n):
        t = np.arange(len(track), dtype=float)
        h, s_hit, r = bm.echoes(bs.call_times(rng, p['rate_hz'], t[-1], cv=CV), t, track)
        keep = ~((s_hit >= occluder[0]) & (s_hit <= occluder[1])) if occluder else np.ones(len(h), bool)
        x = s_hit[keep] + rng.normal(0, p['noise_mm'], keep.sum())
        est = bm.aim_positions(t, h[keep], x, r[keep], k=p['k'], lam=p['noise_mm']**2 / p['sigma_p']**2)
        est['predictive'] = est['velocity']
        for m in MODELS:
            e = bs.angle_error_deg(est[m], track, OFFSET if m == 'predictive' else 0.0)
            rows.append(dict(trial=i, model=m, **bs.metrics(t, track, e, TOL, event or occluder)))
            if curves:
                errs[m].append(e)
    df = pd.DataFrame(rows)
    return df, ({m: np.nanmean(errs[m], 0) for m in MODELS} if curves else None)


def summarize(df, cell):
    """Mean and bootstrap 95% CI of each metric per model."""
    out = []
    for m, d in df.groupby('model'):
        row = dict(cell, model=m)
        for col in ('rms', 'off', 'exit', 'recovery'):
            if col in d:
                v = d[col].dropna().values
                lo, hi = bootstrap_ci(v, n_boot=1000, seed=0) if len(v) > 1 else (np.nan, np.nan)
                row.update({col: v.mean() if len(v) else np.nan, col + '_lo': lo, col + '_hi': hi,
                            col + '_n': len(v)})
        out.append(row)
    return out


def paired(df, a, b, metric):
    """Wilcoxon signed-rank on the same trials, models a vs b; returns W, p, r (r > 0: a larger)."""
    x = df[df.model == a].set_index('trial')[metric]
    y = df[df.model == b].set_index('trial')[metric]
    ok = x.notna() & y.notna()
    return wilcoxon(x[ok].values, y[ok].values)


def coverage(seed=5, trials=20000):
    """Share of prediction errors inside the 95% posterior band when the world matches the prior."""
    rng = np.random.default_rng(seed)
    sigma_e, sigma_p = BASE['noise_mm'], BASE['sigma_p']
    lam, h = sigma_e**2 / sigma_p**2, np.arange(5) * 50.0
    hits = 0
    for _ in range(trials):
        a, v = rng.uniform(-500, 500), rng.normal(0, sigma_p)
        x = a + v * (h - h.mean()) + rng.normal(0, sigma_e, 5)
        T = h[-1] + 200.0
        err = x.mean() + bm.echo_slope(h, x, lam) * (T - h.mean()) - (a + v * (T - h.mean()))
        hits += abs(err) <= 1.96 * bm.predict_sd(h, T, sigma_e, lam)
    return hits / trials


def main():
    style.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    std = bs.smooth(bs.track_data()['standard'])
    summary, tests, seed = [], [], 100

    def cell(name, value, track, occluder=bs.OCCLUDER, event=None, **kw):
        nonlocal seed
        seed += 1
        df, _ = simulate(track, seed, occluder=occluder, event=event, **kw)
        c = dict(sweep=name, value=value)
        summary.extend(summarize(df, c))
        for a, b, metric in (('ours', 'nonpredictive', 'rms'), ('ours', 'ours_noprior', 'rms'),
                             ('ours', 'velocity', 'rms'), ('ours', 'nonpredictive', 'exit')):
            if metric in df and df[metric].notna().any():
                W, p, r = paired(df, a, b, metric)
                tests.append(dict(c, a=a, b=b, metric=metric, W=W, p=p, r=r))
        return df

    cell('steady', 0, std, occluder=None)                             # H2: no occluder, steady motion
    for L in (0, 90, 180, 270, 360):                                  # occlusion length (mm)
        cell('occlusion_mm', L, std, occluder=(450.0, 450.0 + L) if L else None, event=(450.0, 450.0 + L) if L else None)
    for noise in (2, 5, 10, 20, 40, 80):
        cell('noise_mm', noise, std, noise_mm=float(noise))
    for rate in (5, 10, 20, 40):
        cell('rate_hz', rate, std, rate_hz=float(rate))
    for v2 in (0.19, 0.375, 0.75, 1.5, 3.0):
        prof = bs.change_profile(v2)
        cell('change_hidden', v2, prof)                               # change at 500 mm, behind the occluder
        cell('change_visible', v2, prof, occluder=None, event=(500.0, 500.0))

    S = pd.DataFrame(summary)
    S.to_csv(OUT / 'sweeps_summary.csv', index=False)
    Tt = pd.DataFrame(tests)
    Tt['p_holm'] = holm(Tt.p)
    Tt.to_csv(OUT / 'sweeps_wilcoxon.csv', index=False)

    # example curves for the figure: steady, occluded, hidden fast change
    curves = {}
    for key, track, occ in (('steady', std, None), ('occluded', std, bs.OCCLUDER), ('change', bs.change_profile(3.0), bs.OCCLUDER)):
        _, curves[key] = simulate(track, 7, n=N_CURVES, occluder=occ, curves=True)
        curves[key]['track'] = track
    # ---- numbers and hypotheses ----
    g = lambda sw, v, m, col: float(S[(S.sweep == sw) & np.isclose(S.value, v) & (S.model == m)][col].iloc[0])
    t = lambda sw, v, a, b, metric: Tt[(Tt.sweep == sw) & np.isclose(Tt.value, v) & (Tt.a == a) & (Tt.b == b) & (Tt.metric == metric)].iloc[0]
    nums = dict(nTrialsPerCell=N_TRIALS, nTrialsCurves=N_CURVES, baseRateHz=BASE['rate_hz'], baseNoiseMm=BASE['noise_mm'], tolDeg=TOL,
                sigmaP=BASE['sigma_p'], nCells=int(len(S) / len(MODELS)), nTests=len(Tt), coverage=coverage())
    h2 = t('steady', 0, 'ours', 'nonpredictive', 'rms')
    nums.update(steadyRmsOurs=g('steady', 0, 'ours', 'rms'), steadyRmsNonpred=g('steady', 0, 'nonpredictive', 'rms'),
                steadyRmsOursLo=g('steady', 0, 'ours', 'rms_lo'), steadyRmsOursHi=g('steady', 0, 'ours', 'rms_hi'),
                steadyRmsNonpredLo=g('steady', 0, 'nonpredictive', 'rms_lo'), steadyRmsNonpredHi=g('steady', 0, 'nonpredictive', 'rms_hi'),
                steadyOffOurs=g('steady', 0, 'ours', 'off'), steadyOffNonpred=g('steady', 0, 'nonpredictive', 'off'),
                hTwoP=float(h2.p_holm), hTwoR=float(h2.r))
    for v2, tag in ((3.0, 'Fast'), (0.19, 'Slow')):
        h3 = t('change_hidden', v2, 'ours', 'nonpredictive', 'exit')
        nums.update({f'hidden{tag}ExitOurs': g('change_hidden', v2, 'ours', 'exit'),
                     f'hidden{tag}ExitNonpred': g('change_hidden', v2, 'nonpredictive', 'exit'),
                     f'hidden{tag}RecOurs': g('change_hidden', v2, 'ours', 'recovery'),
                     f'hidden{tag}RecNonpred': g('change_hidden', v2, 'nonpredictive', 'recovery'),
                     f'hThree{tag}P': float(h3.p_holm), f'hThree{tag}R': float(h3.r)})
    nums.update(stdExitOurs=g('change_hidden', 0.75, 'ours', 'exit'), stdExitNonpred=g('change_hidden', 0.75, 'nonpredictive', 'exit'))
    h4 = Tt[(Tt.sweep == 'noise_mm') & (Tt.a == 'ours') & (Tt.b == 'ours_noprior') & (Tt.metric == 'rms')].sort_values('value')
    helps = h4[(h4.p_holm < 0.05) & (h4.r < 0)].value.tolist()
    hurts = h4[(h4.p_holm < 0.05) & (h4.r > 0)].value.tolist()
    for _, row in h4.iterrows():
        v = int(row.value)
        nums[f'noise{v}Gain'] = g('noise_mm', v, 'ours_noprior', 'rms') - g('noise_mm', v, 'ours', 'rms')
        nums[f'noise{v}R'] = float(row.r)
    nums.update(priorHelpsAt=', '.join(f'{v:g}' for v in helps) or 'none', priorHurtsAt=', '.join(f'{v:g}' for v in hurts) or 'none',
                priorHelpsMin=min(helps) if helps else -1)
    for L in (90, 180, 270, 360):
        nums[f'occl{L}ExitOurs'] = g('occlusion_mm', L, 'ours', 'exit')
        nums[f'occl{L}ExitNonpred'] = g('occlusion_mm', L, 'nonpredictive', 'exit')
    for rate in (5, 40):
        nums[f'rate{rate}RmsOurs'] = g('rate_hz', rate, 'ours', 'rms')
        nums[f'rate{rate}RmsNonpred'] = g('rate_hz', rate, 'nonpredictive', 'rms')
    save_numbers('sim', nums)

    # ---- figures ----
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    titles = {'steady': 'a  Steady motion, no occluder', 'occluded': 'b  Occluded (450-630 mm)',
              'change': 'c  Speed 0.75 -> 3 m/s behind occluder'}
    for ax, key in zip(axs, ('steady', 'occluded', 'change')):
        c = curves[key]
        x = c['track']
        win = (x >= 300) & (x <= 1300)
        if key != 'steady':
            ax.axvspan(450, 630, color='#ecebe7', lw=0)
        for m in ('nonpredictive', 'velocity', 'predictive', 'ours'):
            ax.plot(x[win], c[m][win], linestyle=style.STYLES[m], color=style.COLORS[m], label=style.LABELS[m])
        ax.axhline(0, color=style.INK, lw=1)
        ax.set_xlabel('Target position on track (mm)'); ax.set_title(titles[key])
    axs[0].set_ylabel('Mean aim error (deg, + = ahead)')
    axs[0].set_ylim(-25, 15)
    axs[0].legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    style.save(fig, 'fig06_sim_error_over_time')

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    panels = (('occlusion_mm', 'exit', 'Occluder length (mm)', 'a  Error when the target reappears', 'Error at exit (deg)'),
              ('noise_mm', 'rms', 'Echo position noise (mm)', 'b  Echo noise', 'RMS error (deg)'),
              ('rate_hz', 'rms', 'Call rate (Hz)', 'c  Call rate', 'RMS error (deg)'),
              ('change_hidden', 'exit', 'Speed after the change (m/s)', 'd  Speed change behind the occluder', 'Error at exit (deg)'))
    for ax, (sw, col, xl, title, yl) in zip(axs.ravel(), panels):
        d = S[S.sweep == sw]
        for m in ('nonpredictive', 'velocity', 'ours', 'ours_noprior'):
            dm = d[d.model == m].sort_values('value')
            ax.errorbar(dm.value, dm[col], yerr=[dm[col] - dm[col + '_lo'], dm[col + '_hi'] - dm[col]],
                        linestyle=style.STYLES[m], color=style.COLORS[m], marker='o', ms=4, capsize=2, label=style.LABELS[m])
        if sw in ('noise_mm', 'rate_hz', 'change_hidden'):
            ax.set_xscale('log'); ax.set_xticks(sorted(d.value.unique()), [f'{v:g}' for v in sorted(d.value.unique())])
            ax.minorticks_off()
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    axs[0, 0].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    style.save(fig, 'fig07_sim_sweeps')

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    d = S[S.sweep == 'noise_mm']
    gain = d[d.model == 'ours_noprior'].set_index('value').rms - d[d.model == 'ours'].set_index('value').rms
    ax.plot(gain.index, gain.values, 'o-', color=style.COLORS['ours'])
    ax.axhline(0, color=style.INK, lw=1)
    ax.set_xscale('log'); ax.set_xticks(gain.index, [f'{v:g}' for v in gain.index]); ax.minorticks_off()
    ax.set_xlabel('Echo position noise (mm)'); ax.set_ylabel('RMS error saved by the prior (deg)')
    ax.set_title('The slow prior pays off only with noisy echoes')
    style.save(fig, 'fig08_prior_gain')
    print(json.dumps(nums, indent=1))


if __name__ == '__main__':
    main()
