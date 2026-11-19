"""Load the head-tracking trials of Salles, Diebold & Moss (2020) in Python.

Source: https://github.com/angiesalles/batTargetPrediction (MATLAB code and data by A. Salles et al.;
no licence file, so the data are read locally and not redistributed). This module re-implements the
trial selection of the authors' miroTracking.m (sheets, quality labels, bad-trial exclusions, ear
correction per bat), the head angle from the two ear markers, and the bat selection and 10 ms time
base of statistics_miro.m / createGetAllBats.m. Line references are to those files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2] / 'data/external/battargetprediction'
DATA = REPO / 'data'

BATS = [('B', '24'), ('G', '38'), ('W', '80'), ('DB', '8'), ('W', '28')]       # miroTracking.m line 3
CORRECTION = [-1, 0, -8, 0, 0]                                                  # ear correction per bat (line 4)
_D2 = ['20180912', '20180913', '20180914', '20180916', '20180917', 'EMPTYDAY', '20180919', '20180920',
       '20180921', '20180923', '20180924', '20180925']
_D4 = ['20190429', '20190430', '20190501', '20190502', '20190503', '20190505', '20190507', '20190508',
       '20190509', '20190511', '20190512', '20190513']
DATES = [['20180309', '20180311', '20180312', '20180313', '20180315', '20180316'], _D2, _D2, _D4,
         _D4[:-1] + ['20190514', '20190515']]                                   # lines 5-9
CONDITIONS = ['B-L', 'O-L', 'Catch', 'B-MF', 'B-MS', 'O-MF', 'O-MS', 'O-stop', 'B-stop']   # line 10
SELECT_BATS = [0, 1, 2, 4]          # statistics_miro.m line 5: selectBats = [1:3, 5] (DB8 not used)
BAD = {'Wrong distance', "Don't Use", 'Not good', 'VG; mic was covered', 'DO NOT USE', 'Bad',
       'Ok (not focused)', 'Useless', 'N', 'N ', 'M'}                           # line 54
BAD2 = {'not much calling', 'wrong distance, no audio', 'It was 1m DO NOT USE', 'Camera triggered twice',
        "Tether not attached, don't use", "Video didn't save"}                  # line 57
BIN_MS = 10.0                        # one head-angle sample per 10 ms (statistics_miro.m uses columns 1:200 = 2 s)


def _isnan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _sheet(bat, day):
    """Raw cell grid of one day's trial sheet (miroTracking.m lines 17-46)."""
    if bat == 0:
        xls = pd.ExcelFile(DATA / 'Blue24 Data.xlsx')
        return pd.read_excel(xls, xls.sheet_names[day + 1], header=None).values
    return pd.read_excel(DATA / 'SmallFlightRoom' / f'{DATES[bat][day]}_PreyTracking.xlsx',
                         sheet_name=BATS[bat][1], header=None).values


def _csv_path(bat, day, trial):
    """Path of the 3-D ear-marker file of one trial (lines 36-45, 82-87)."""
    date = DATES[bat][day]
    if bat == 0:
        folder = DATA / 'Miro tracking' / date
    elif bat in (1, 2):
        folder = DATA / 'green38andwhite80' / date / ('W38' if bat == 1 and day == 1 else ''.join(BATS[bat]))
    else:
        folder = DATA / date / ('B8' if bat == 3 else ''.join(BATS[bat]))
    return folder / f'trial_{int(trial)}_xyzpts.csv'


def _excluded(bat, day, idx):
    """Hand-listed exclusions of miroTracking.m lines 73-81 (1-based day and row as in MATLAB)."""
    d, i = day + 1, idx
    if bat == 0 and ((d == 2 and (i < 5 or i in (27, 30))) or (d == 5 and i == 3) or (d == 6 and i in (26, 29, 32))):
        return True
    return (bat == 3 and d == 5 and i > 23) or (bat == 4 and d == 1)


def head_angle(csv):
    """Head angle (rad) per sample from the two ear markers: atan((z1 - z2 + corr) / (x1 - x2)) (line 90)."""
    xyz = pd.read_csv(csv).values
    return xyz[:, 2] - xyz[:, 5], xyz[:, 0] - xyz[:, 3]


def load_trials():
    """All usable trials: list of dicts (bat, day, row, condition, head angle in rad at 10 ms, call times in ms)."""
    meta = sio.loadmat(DATA / 'shouts.mat')['metastore']
    qual, typ, trial_col = (5, 1, 0)
    trials = []
    for bat in range(len(BATS)):
        for day in range(len(DATES[bat])):
            if bat > 0 and day == 5:                 # 'EMPTYDAY' and its counterparts (line 30)
                continue
            raw = _sheet(bat, day)
            if bat > 0:
                qual, typ, trial_col = (8, 3, 2)
            for idx in range(1, raw.shape[0]):       # idx = MATLAB row number below the header
                row = list(raw[idx]) + [np.nan] * 12
                if all(_isnan(v) for v in row[:6]):
                    continue
                q, q2 = row[qual], row[qual + 1]
                if isinstance(q2, str) and q2.startswith('Tether not attached, do'):
                    q2 = "Tether not attached, don't use"
                if q in BAD or q2 in BAD2:
                    continue
                if bat > 0 and _isnan(row[qual - 1]):  # 'Video done' empty
                    continue
                calls = meta[bat, day, idx - 1] if idx - 1 < meta.shape[2] else np.empty((0, 1))
                if calls.size == 0:
                    continue
                cond = str(row[typ]).replace('B-Catch', 'Catch').replace('Stop', 'stop')
                if bat == 0 and day == 0 and len(str(row[typ])) < 3:
                    cond = 'B-' + cond
                if cond not in CONDITIONS or _excluded(bat, day, idx):
                    continue
                dz, dx = head_angle(_csv_path(bat, day, row[trial_col]))
                trials.append(dict(bat=bat, name=''.join(BATS[bat]), day=day, row=idx, condition=cond,
                                   head=np.arctan((dz + CORRECTION[bat]) / dx), calls=calls.ravel().astype(float)))
    return trials


def head_matrix(trials, n=200):
    """Trials x n matrix of head angles (rad); NaN where a trial is shorter or untracked."""
    M = np.full((len(trials), n), np.nan)
    for i, tr in enumerate(trials):
        m = min(n, len(tr['head']))
        M[i, :m] = tr['head'][:m]
    return M


def occluded_calls(track):
    """Call-time window (ms) removed in occluded trials: target between 450 and 630 mm (statistics_miro.m
    lines 128-130, 235 with plotOccluder.m limits)."""
    return float(np.argmax(track > 450) + 1), float(np.argmax(track > 630) + 1)


def model_positions(trials, track, n_bins, k=5, lam=0.0, occluded=False):
    """Per-trial model position estimates at t = 0, 10, ... ms, from the trial's real call times.

    Mirrors createGetAllBats.m: calls with 0 < C < length of the track are used; in occluded trials the
    calls emitted while the target is behind the occluder are dropped. Returns dict model -> trials x bins.
    """
    from . import bat_models as bm
    t_track = np.arange(len(track), dtype=float)
    t = np.arange(n_bins) * BIN_MS
    lo, hi = occluded_calls(track)
    out = {}
    for i, tr in enumerate(trials):
        C = tr['calls'][(tr['calls'] > 0) & (tr['calls'] < len(track))]
        if occluded:
            C = C[(C < lo) | (C > hi)]
        h, s_hit, r = bm.echoes(C, t_track, track)
        for name, pos in bm.aim_positions(t, h, s_hit, r, k=k, lam=lam).items():
            out.setdefault(name, np.full((len(trials), n_bins), np.nan))[i] = pos
    return out


def rss(model_angle, head, lo, hi, shift=0.0):
    """Salles goodness of fit (calcFVUsuper.m / calcFVU.m): sum over bins lo..hi (1-based, inclusive) of
    the squared trial-mean offset (model - head + shift), rad^2."""
    off = np.nanmean(model_angle[:, lo - 1:hi] - head[:, lo - 1:hi], axis=0) + shift
    return np.nansum(off**2)


def best_shift(model_angle, head, lo, hi, bounds=(0.0, np.radians(20))):
    """Least-squares shift (rad) of the trial-mean offset, limited to 0-20 deg as in findModelParams.m."""
    off = np.nanmean(model_angle[:, lo - 1:hi] - head[:, lo - 1:hi], axis=0)
    return float(np.clip(-np.nanmean(off), *bounds))
