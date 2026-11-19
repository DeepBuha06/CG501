"""The real bat data of Salles et al. (2020), loaded with eye2echo.bat_data.

1. Behaviour (Salles Fig. 1A/1B, 3A): mean head angle vs target angle, 95% CI bands, per-bin t-tests
   of the lead with Bonferroni correction (73 bins of 10 ms, -660 to +70 ms around the crossing).
2. Models rebuilt per trial from the real call times; RSS as in calcFVU.m, compared with SI Table S1.
3. Our slow-prior line fit added; free parameters chosen by leave-one-bat-out cross-validation.
Outputs: results/bat_real/*, results/numbers/real.json, figures fig09-fig11.
"""
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from eye2echo import bat_data as bd, bat_sim as bs, bat_models as bm, style
from eye2echo.stats import wilcoxon, holm, save_numbers

OUT = style.RES / 'bat_real'
CACHE = style.ROOT / 'data/external/trials_cache.pkl'
KS = range(2, 11)
LAMS = [0.0, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5]     # ms^2
WIN = (65, 130)                                     # 1-based 10 ms bins = 640-1290 ms (47-106 cm)
PAPER = {'nonpredictive': 3.1061, 'fixed': 0.9858, 'velocity': 0.8866, 'predictive': 0.1458}
# before/after-change windows of statistics_miro.m (lines 265-298, 368-398), 1-based bins
CHANGE_WIN = {'B-MF': ((58, 161), (162, 215)), 'O-MF': ((58, 178), (179, 215)),
              'B-MS': ((58, 180), (181, 235)), 'O-MS': ((58, 190), (191, 235))}


def load():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    trials = bd.load_trials()
    CACHE.write_bytes(pickle.dumps(trials))
    return trials


def angles(trials, track, n, k, lam, occluded):
    return {m: bm.angle(p) for m, p in bd.model_positions(trials, track, n, k=k, lam=lam, occluded=occluded).items()}


def trial_rss(model, head, shift, lo, hi):
    """Per-trial RSS over the window (rad^2), for paired tests."""
    return np.nansum((model[:, lo - 1:hi] + shift - head[:, lo - 1:hi]) ** 2, axis=1)


def fit_all(A, H, lo, hi):
    """Best parameters of each model on one set of trials. A[(k, lam)] = model angle dict."""
    a0 = A[(5, 0.0)]
    best = {'nonpredictive': dict(k=None, lam=None, shift=0.0),
            'fixed': dict(k=None, lam=None, shift=bd.best_shift(a0['nonpredictive'], H, lo, hi))}
    rv = {k: bd.rss(A[(k, 0.0)]['velocity'], H, lo, hi) for k in KS}
    best['velocity'] = dict(k=min(rv, key=rv.get), lam=None, shift=0.0)
    cand = {}
    for k in KS:
        sh = bd.best_shift(A[(k, 0.0)]['velocity'], H, lo, hi)
        cand[k] = (bd.rss(A[(k, 0.0)]['velocity'], H, lo, hi, sh), sh)
    kb = min(cand, key=lambda k: cand[k][0])
    best['predictive'] = dict(k=kb, lam=None, shift=cand[kb][1])
    for name, lams in (('ours', LAMS), ('ours_noprior', [0.0])):
        cand = {}
        for k in KS:
            for lam in lams:
                m = A[(k, lam)]['ours']
                sh = bd.best_shift(m, H, lo, hi)
                cand[(k, lam)] = (bd.rss(m, H, lo, hi, sh), sh)
        kb, lb = min(cand, key=lambda p: cand[p][0])
        best[name] = dict(k=kb, lam=lb, shift=cand[(kb, lb)][1])
    return best


def model_angle(A, name, p):
    key = (p['k'] or 5, p['lam'] or 0.0)
    return A[key]['nonpredictive' if name in ('nonpredictive', 'fixed') else 'velocity' if name in ('velocity', 'predictive') else 'ours']


def main():
    style.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    trials = [t for t in load() if t['bat'] in bd.SELECT_BATS]
    tracks = bs.track_data()
    std = tracks['standard']
    t_cross = float(np.argmax(std > 1000))
    by = lambda c: [t for t in trials if t['condition'] == c]
    counts = pd.DataFrame([(t['name'], t['condition']) for t in trials], columns=['bat', 'condition']).value_counts().unstack(fill_value=0)
    counts.to_csv(OUT / 'trial_counts.csv')
    nums = dict(nBats=len(bd.SELECT_BATS), nTrialsAll=len(trials), nBL=len(by('B-L')), nOL=len(by('O-L')),
                nCatch=len(by('Catch')), nBMF=len(by('B-MF')), nOMF=len(by('O-MF')), nBMS=len(by('B-MS')), nOMS=len(by('O-MS')))

    # call statistics in the tracking window (used as the simulation's baseline call schedule)
    inside = [np.sort(t['calls'][(t['calls'] > 640) & (t['calls'] < 1290)]) for t in by('B-L')]
    iv = np.concatenate([np.diff(c) for c in inside])
    nums.update(callRateMedianHz=float(np.median([len(c) / 0.65 for c in inside])), callIntervalMedianMs=float(np.median(iv)),
                callIntervalCV=float(iv.std() / iv.mean()))

    # ---- 1. behaviour ----
    tgt = bm.angle(std[::10])
    t_rel = np.arange(200) * 10 - t_cross
    beh = {}
    for c in ('B-L', 'Catch', 'O-L'):
        H = bd.head_matrix(by(c), 200)
        m, se = np.nanmean(H, 0), stats.sem(H, 0, nan_policy='omit')
        beh[c] = (np.degrees(m), np.degrees(1.96 * se))
    H = bd.head_matrix(by('B-L'), 200)
    bins = np.where((t_rel >= -660) & (t_rel < 70))[0]
    lead = H[:, bins] - tgt[bins]
    tt = stats.ttest_1samp(lead, 0, nan_policy='omit')
    p_bonf = np.minimum(tt.pvalue * len(bins), 1)
    nums.update(leadBins=len(bins), leadMeanDeg=float(np.degrees(np.nanmean(lead))),
                leadSigBins=int(np.sum((p_bonf < 0.05) & (tt.statistic > 0))), leadMinT=float(tt.statistic.min()),
                leadMaxT=float(tt.statistic.max()), leadMaxPBonf=float(p_bonf.max()),
                leadFirstSigMs=float(t_rel[bins][np.argmax((p_bonf < 0.05) & (tt.statistic > 0))]))
    whole = np.nanmean(lead, axis=1)
    tw = stats.ttest_1samp(whole, 0)
    nums.update(leadTrialT=float(tw.statistic), leadTrialP=float(tw.pvalue), leadTrialDf=int(len(whole) - 1),
                leadTrialR=float(np.sqrt(tw.statistic**2 / (tw.statistic**2 + len(whole) - 1))))
    pd.DataFrame({'t_rel_ms': t_rel[bins], 't': tt.statistic, 'p_bonf': p_bonf}).to_csv(OUT / 'lead_tests.csv', index=False)

    # ---- 2. models on standard trials ----
    BL, OL = by('B-L'), by('O-L')
    H_BL, H_OL = bd.head_matrix(BL, 200), bd.head_matrix(OL, 200)
    A_BL = {(k, l): angles(BL, std, 200, k, l, False) for k in KS for l in LAMS}
    A_OL = {(k, l): angles(OL, std, 200, k, l, True) for k in KS for l in LAMS}
    best = fit_all(A_BL, H_BL, *WIN)
    order = ['nonpredictive', 'fixed', 'velocity', 'predictive', 'ours_noprior', 'ours']
    rows = []
    for name in order:
        p = best[name]
        rows.append(dict(model=name, k=p['k'], lam=p['lam'], shift_deg=np.degrees(p['shift']),
                         rss_BL=bd.rss(model_angle(A_BL, name, p), H_BL, *WIN, p['shift']),
                         rss_OL=bd.rss(model_angle(A_OL, name, p), H_OL, *WIN, p['shift']), rss_paper=PAPER.get(name)))
    fit = pd.DataFrame(rows)
    fit.to_csv(OUT / 'model_fit_standard.csv', index=False)
    f = fit.set_index('model')
    for name in order:
        key = name.replace('_', '').capitalize()
        nums[f'rss{key}'] = float(f.loc[name, 'rss_BL'])
        nums[f'rssOcc{key}'] = float(f.loc[name, 'rss_OL'])
        nums[f'shift{key}'] = float(f.loc[name, 'shift_deg'])
    nums.update(kPredictive=int(best['predictive']['k']), kVelocity=int(best['velocity']['k']), kOurs=int(best['ours']['k']),
                lamOurs=float(best['ours']['lam']))

    # ---- 3. leave-one-bat-out cross-validation ----
    cv_rows, per_trial = [], {m: np.full(len(BL), np.nan) for m in order}
    bat_of = np.array([t['bat'] for t in BL])
    for b in bd.SELECT_BATS:
        tr_i, te_i = np.where(bat_of != b)[0], np.where(bat_of == b)[0]
        sub = lambda A, idx: {key: {m: v[idx] for m, v in d.items()} for key, d in A.items()}
        p_tr = fit_all(sub(A_BL, tr_i), H_BL[tr_i], *WIN)
        A_te = sub(A_BL, te_i)
        for name in order:
            p = p_tr[name]
            m = model_angle(A_te, name, p)
            cv_rows.append(dict(bat=bd.BATS[b][0] + bd.BATS[b][1], model=name, k=p['k'], lam=p['lam'],
                                shift_deg=np.degrees(p['shift']), rss=bd.rss(m, H_BL[te_i], *WIN, p['shift'])))
            per_trial[name][te_i] = trial_rss(m, H_BL[te_i], p['shift'], *WIN)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT / 'lobo_cv.csv', index=False)
    cvsum = cv.groupby('model').rss.sum()
    for name in order:
        nums['cv' + name.replace('_', '').capitalize()] = float(cvsum[name])
    lam_choices = cv[cv.model == 'ours'].lam.tolist()
    nums['cvOursLamChoices'] = ', '.join(f'{l:g}' for l in lam_choices)
    nums['cvOursLamZeroCount'] = int(sum(l == 0 for l in lam_choices))

    pairs = [('predictive', 'nonpredictive'), ('predictive', 'velocity'), ('predictive', 'fixed'),
             ('ours', 'predictive'), ('ours', 'ours_noprior')]
    tests = []
    for a, b in pairs:
        W, p, r = wilcoxon(per_trial[a], per_trial[b])
        tests.append(dict(a=a, b=b, median_a=np.median(per_trial[a]), median_b=np.median(per_trial[b]), W=W, p=p, r=r))
    tests = pd.DataFrame(tests)
    tests['p_holm'] = holm(tests.p)
    tests.to_csv(OUT / 'cv_wilcoxon.csv', index=False)
    for _, row in tests.iterrows():
        key = row.a.replace('_', '').capitalize() + 'Vs' + row.b.replace('_', '').capitalize()
        nums['p' + key], nums['r' + key] = float(row.p_holm), float(row.r)

    # ---- 4. velocity-change conditions (parameters from the standard fit, as in the paper) ----
    chg = []
    for cond, (w1, w2) in CHANGE_WIN.items():
        track = tracks['fast' if 'MF' in cond else 'slow']
        n = len(track) // 10
        T = by(cond)
        H = bd.head_matrix(T, n)
        occ = cond.startswith('O')
        A5 = angles(T, track, n, best['predictive']['k'], 0.0, occ)
        Ao = angles(T, track, n, best['ours']['k'], best['ours']['lam'], occ)
        for seg, (lo, hi) in (('before', w1), ('after', w2)):
            chg.append(dict(condition=cond, segment=seg,
                            nonpredictive=bd.rss(A5['nonpredictive'], H, lo, hi),
                            predictive=bd.rss(A5['velocity'], H, lo, hi, best['predictive']['shift']),
                            ours=bd.rss(Ao['ours'], H, lo, hi, best['ours']['shift'])))
    chg = pd.DataFrame(chg)
    chg.to_csv(OUT / 'velocity_change_rss.csv', index=False)
    nums['changeOursBeatsPredCount'] = int((chg.ours < chg.predictive).sum())
    nums['changeNonpredBeatsPredAfterOccl'] = int(((chg.segment == 'after') & chg.condition.str.startswith('O') &
                                                   (chg.nonpredictive < chg.predictive)).sum())
    for _, row in chg.iterrows():
        for m in ('nonpredictive', 'predictive', 'ours'):
            nums[f"chg{row.condition.replace('-', '')}{row.segment.capitalize()}{m.capitalize()}"] = float(row[m])
    save_numbers('real', nums)
    # reference values quoted from the paper (Salles et al. 2020, Results and SI Tables S1-S2), for comparison only
    save_numbers('paper', dict(rssNonpredictive=3.1061, rssFixed=0.9858, shiftFixed=10.2695, rssVelocity=0.8866,
                               rssPredictive=0.1458, shiftPredictive=6.241, kPredictive=5, kVelocity=2,
                               nBL=97, nOL=62, leadT=4.06, leadDf=97, leadR=0.3811,
                               rssOccNonpredictive=4.4234, rssOccPredictive=0.4137,
                               chgOMFAfterNonpredictive=0.047, chgOMFAfterPredictive=3.3818,
                               chgOMSAfterNonpredictive=0.2852, chgOMSAfterPredictive=0.4733))

    # mean traces for the video
    traces = pd.DataFrame({'t_ms': np.arange(200) * 10.0, 'head_deg': beh['B-L'][0], 'target_deg': np.degrees(tgt)})
    for name in ('nonpredictive', 'predictive', 'ours'):
        p = best[name]
        traces[name + '_deg'] = np.degrees(np.nanmean(model_angle(A_BL, name, p), 0) + p['shift'])
    traces.to_csv(OUT / 'mean_traces_standard.csv', index=False)
    (OUT / 'best_params.json').write_text(json.dumps({k: {kk: (float(vv) if vv is not None else None) for kk, vv in v.items()}
                                                      for k, v in best.items()}, indent=1))

    # ---- figures ----
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axs[0]
    occ = (np.argmax(std > 450) - t_cross, np.argmax(std > 630) - t_cross)
    ax.axvspan(*occ, color='#ecebe7', lw=0)
    ax.text(np.mean(occ), -62, 'occluder\n(O-L)', ha='center', fontsize=8, color=style.INK2)
    ax.plot(t_rel, np.degrees(tgt), '--', color=style.INK, lw=1.2, label='target')
    for c, col, lab in (('B-L', style.COLORS['ours'], f'standard (n={nums["nBL"]})'),
                        ('O-L', style.COLORS['predictive'], f'occluded (n={nums["nOL"]})'),
                        ('Catch', style.COLORS['nonpredictive'], f'catch (n={nums["nCatch"]})')):
        m, ci = beh[c]
        ax.fill_between(t_rel, m - ci, m + ci, color=col, alpha=0.18, lw=0)
        ax.plot(t_rel, m, color=col, label=lab)
    ax.set_xlim(-800, 300); ax.set_ylim(-75, 45)
    ax.set_xlabel('Time from target crossing (ms)'); ax.set_ylabel('Angle (deg)')
    ax.set_title('a  Head aim vs target (mean, 95% CI)'); ax.legend(loc='upper left')
    ax = axs[1]
    m = np.degrees(np.nanmean(H_BL - tgt, 0)); ci = np.degrees(1.96 * stats.sem(H_BL - tgt, 0, nan_policy='omit'))
    ax.fill_between(t_rel, m - ci, m + ci, color=style.COLORS['ours'], alpha=0.18, lw=0)
    ax.plot(t_rel, m, color=style.COLORS['ours'], label='head - target, standard')
    sig = bins[(p_bonf < 0.05) & (tt.statistic > 0)]
    ax.plot(t_rel[sig], np.full(len(sig), -8), 's', ms=3, color=style.INK, label='lead, p < 0.05 (Bonferroni, 73 bins)')
    ax.axhline(0, color=style.INK, lw=1)
    ax.set_xlim(-800, 300); ax.set_ylim(-12, 30)
    ax.set_xlabel('Time from target crossing (ms)'); ax.set_ylabel('Head minus target (deg)')
    ax.set_title('b  The head leads the target'); ax.legend(loc='upper left')
    fig.tight_layout()
    style.save(fig, 'fig09_real_tracking')

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axs[0]
    for name in ('nonpredictive', 'fixed', 'velocity', 'predictive', 'ours'):
        p = best[name]
        off = np.degrees(np.nanmean(model_angle(A_BL, name, p) + p['shift'] - H_BL, 0))
        ax.plot(t_rel, off, linestyle=style.STYLES[name], color=style.COLORS[name], label=style.LABELS[name])
    ax.axvspan(t_rel[WIN[0] - 1], t_rel[WIN[1] - 1], color='#f3f2ef', lw=0, zorder=0)
    ax.axhline(0, color=style.INK, lw=1)
    ax.set_xlim(-700, 150); ax.set_ylim(-30, 15)
    ax.set_xlabel('Time from target crossing (ms)'); ax.set_ylabel('Model minus head (deg)')
    ax.text(-690, 12, 'predictive and ours overlap (best lambda = 0)', fontsize=8, color=style.INK2)
    ax.set_title('a  Offset of each model from the bat (standard trials)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8)
    ax = axs[1]
    names = ['nonpredictive', 'fixed', 'velocity', 'predictive', 'ours']
    y = np.arange(len(names))
    ax.barh(y + 0.2, [f.loc[n, 'rss_BL'] for n in names], 0.38, color=style.COLORS['ours'], label='this replication')
    ax.barh(y - 0.2, [PAPER.get(n, np.nan) for n in names], 0.38, color='#b9b8b2', label='Salles et al. SI Table S1')
    for i, n in enumerate(names):
        ax.text(f.loc[n, 'rss_BL'] + 0.05, i + 0.2, f'{f.loc[n, "rss_BL"]:.2f}', va='center', fontsize=8)
    ax.set_yticks(y, [style.LABELS[n].replace(' (ours)', '\n(ours)') for n in names], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel('RSS (rad$^2$), lower = better fit')
    ax.set_title('b  Goodness of fit'); ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    style.save(fig, 'fig10_real_models')

    fig, ax = plt.subplots(figsize=(6, 3.6))
    batnames = cv.bat.unique()
    names = ['nonpredictive', 'fixed', 'velocity', 'predictive', 'ours_noprior', 'ours']
    w = 0.13
    for i, n in enumerate(names):
        d = cv[cv.model == n].set_index('bat').loc[batnames]
        ax.bar(np.arange(len(batnames)) + (i - 2.5) * w, d.rss, w * 0.92, color=style.COLORS[n], label=style.LABELS[n])
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(batnames)), [f'held-out {b}' for b in batnames])
    ax.set_ylabel('Held-out RSS (rad$^2$, log)')
    ax.set_title('Leave-one-bat-out: each bat scored with parameters from the other three')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    style.save(fig, 'fig11_lobo_cv')
    print(json.dumps(nums, indent=1))
    print(fit.round(4).to_string())
    print(tests.round(5).to_string())
    print(chg.round(3).to_string())


if __name__ == '__main__':
    main()
