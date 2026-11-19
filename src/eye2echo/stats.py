"""Statistics used in the report, plus the helper that records every reported number.

Paired model comparisons follow Salles et al. (2020, Methods): Wilcoxon signed-rank test with the
rank-biserial correlation as effect size. Multiple comparisons: Holm-Bonferroni.
"""
import json
import numpy as np
from scipy import stats
from .style import RES


def bootstrap_ci(x, n_boot=2000, seed=0, stat=np.mean):
    """Percentile bootstrap 95% interval of stat(x)."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return np.percentile(boots, [2.5, 97.5])


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation r = (R+ - R-) / (R+ + R-) of the differences x - y.

    Positive r means x tends to be larger than y.
    """
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))
    rp, rm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return (rp - rm) / (rp + rm)


def wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank test on paired samples; returns (W, p, r)."""
    res = stats.wilcoxon(x, y)
    return res.statistic, res.pvalue, rank_biserial(x, y)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def save_numbers(group, values):
    """Write one group of reported numbers to results/numbers/<group>.json (merged later)."""
    out = RES / 'numbers'
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in values.items()}
    (out / f'{group}.json').write_text(json.dumps(clean, indent=1))
