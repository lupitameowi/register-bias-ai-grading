"""stats_utils.py: statistics used in both phases."""

import math
import warnings

import numpy as np
from scipy import stats


def qwk(a, b, categories=(1, 2, 3, 4, 5)):
    """Quadratic weighted kappa: 1 = perfect agreement, 0 = chance, <0 = worse than chance.
    The standard agreement statistic in automated-scoring research."""
    a = [int(round(x)) for x in a]
    b = [int(round(x)) for x in b]
    cats = list(categories)
    k, idx, n = len(cats), {c: i for i, c in enumerate(cats)}, len(a)
    if n == 0:
        return float("nan")
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[idx[x], idx[y]] += 1
    expected = np.outer(observed.sum(1), observed.sum(0)) / n
    weights = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    denom = (weights * expected).sum()
    return float("nan") if denom == 0 else float(1 - (weights * observed).sum() / denom)


def paired_summary(casual, formal):
    """Paired comparison, casual minus formal. Reports the mean difference with a
    95% CI, a paired t-test, a Wilcoxon signed-rank check (scores are ordinal),
    and Cohen's dz."""
    x, y = np.asarray(casual, float), np.asarray(formal, float)
    d = x - y
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    out = {"n": n, "mean_casual": float(x.mean()), "mean_formal": float(y.mean()),
           "mean_diff": mean, "ci_low": float("nan"), "ci_high": float("nan"),
           "t": float("nan"), "p_t": float("nan"), "p_wilcoxon": float("nan"), "dz": float("nan")}
    if n > 1 and sd > 0:
        se = sd / math.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        out.update(ci_low=mean - tcrit * se, ci_high=mean + tcrit * se, dz=mean / sd)
        t = stats.ttest_rel(x, y)
        out.update(t=float(t.statistic), p_t=float(t.pvalue))
    if n > 1 and np.any(d != 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out["p_wilcoxon"] = float(stats.wilcoxon(x, y).pvalue)
            except ValueError:
                pass
    return out


def friedman(*conditions):
    """Friedman test across 3+ repeated conditions (same texts in each)."""
    try:
        res = stats.friedmanchisquare(*conditions)
        return {"chi2": float(res.statistic), "p": float(res.pvalue)}
    except ValueError:
        return {"chi2": float("nan"), "p": float("nan")}


def spearman(a, b):
    """Rank correlation. Returns nan if either input is constant (no variation to
    correlate), which happens if the grader gives every text the same score."""
    if len(set(a)) < 2 or len(set(b)) < 2:
        return {"rho": float("nan"), "p": float("nan")}
    res = stats.spearmanr(a, b)
    return {"rho": float(res.statistic), "p": float(res.pvalue)}


def fmt(x, digits=2):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "<.001" if p < 0.001 else f"{p:.3f}"
