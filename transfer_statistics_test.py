"""Statistical comparison of paired DGCDR and LightGCN user scores."""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.stats import t, ttest_ind, ttest_rel


@dataclass(frozen=True)
class PairedTTestResult:
    n: int
    mean_delta: Optional[float]
    statistic: Optional[float]
    degrees_of_freedom: Optional[int]
    pvalue: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    reason: Optional[str] = None


@dataclass(frozen=True)
class WelchTTestResult:
    n_high: int
    n_low: int
    mean_high: Optional[float]
    mean_low: Optional[float]
    std_high: Optional[float]
    std_low: Optional[float]
    difference: Optional[float]
    statistic: Optional[float] = None
    degrees_of_freedom: Optional[float] = None
    pvalue: Optional[float] = None
    reason: Optional[str] = None


def welch_transfer_test(high_delta, low_delta):
    """Two-sided independent Welch test of user gains; effect = high minus low."""
    groups = []
    for values, name in [(high_delta, 'source alto'), (low_delta, 'source basso')]:
        scores = np.asarray(values, dtype=np.float64)
        if scores.ndim != 1 or not np.isfinite(scores).all() or (
                (scores < -1) | (scores > 1)).any():
            raise ValueError('I Delta di %s devono essere un vettore finito in [-1, 1]' % name)
        groups.append(scores)
    high, low = groups
    nh, nl = len(high), len(low)
    mh, ml = float(high.mean()) if nh else None, float(low.mean()) if nl else None
    sh = float(high.std(ddof=1)) if nh >= 2 else None
    sl = float(low.std(ddof=1)) if nl >= 2 else None
    base = WelchTTestResult(nh, nl, mh, ml, sh, sl,
                           mh - ml if mh is not None and ml is not None else None)
    if nh < 2 or nl < 2:
        return replace(base, reason='servono almeno due utenti in ciascun gruppo')
    vh, vl = sh ** 2 / nh, sl ** 2 / nl
    if vh + vl == 0:
        return replace(base, reason='la varianza delle differenze è zero in entrambi i gruppi')
    result = ttest_ind(high, low, equal_var=False, nan_policy='raise')
    # Compute Welch-Satterthwaite df explicitly for compatibility with old SciPy.
    weight = vh / (vh + vl)
    df = 1 / (weight ** 2 / (nh - 1) + (1 - weight) ** 2 / (nl - 1))
    if not np.isfinite([result.statistic, result.pvalue, df]).all():
        return replace(base, reason='il test ha prodotto valori numerici non finiti')
    return replace(base, statistic=float(result.statistic), degrees_of_freedom=float(df),
                   pvalue=float(result.pvalue))


def validate_ndcg_values(values, name):
    """Return finite NDCG scores in [0, 1] without dropping missing values."""
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} contiene valori NDCG non numerici') from exc
    if scores.ndim != 1:
        raise ValueError(f'{name} deve contenere un valore per osservazione')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f'{name} contiene valori NDCG mancanti o fuori da [0, 1]')
    return scores


def holm_adjust(pvalues):
    """Holm-adjust valid p-values, returning them in their original order.

    Callers can reserve an untestable comparison's place in the family with 1,
    while keeping its actual p-value undefined in the reported results.
    """
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (
            (values < 0) | (values > 1)).any():
        raise ValueError('I p-value devono essere un vettore finito in [0, 1]')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def global_paired_ttest(dgcdr, lightgcn):
    """Two-sided paired t-test of per-user NDCG, including zero-zero pairs."""
    dgcdr = validate_ndcg_values(dgcdr, 'DGCDR')
    lightgcn = validate_ndcg_values(lightgcn, 'LightGCN')
    if dgcdr.shape != lightgcn.shape:
        raise ValueError('DGCDR e LightGCN devono avere una coppia per ogni utente')

    n = len(dgcdr)
    delta = dgcdr - lightgcn
    mean_delta = float(delta.mean()) if n else None
    if n < 2:
        return PairedTTestResult(n, mean_delta, None, None, None, None, None,
                                 'servono almeno due utenti')
    if np.all(delta == delta[0]):
        return PairedTTestResult(n, mean_delta, None, None, None, None, None,
                                 'la varianza delle differenze è zero')

    # The default paired test is two-sided. Compute its interval explicitly so
    # older SciPy Ttest_relResult objects need no confidence_interval() or df.
    result = ttest_rel(dgcdr, lightgcn, nan_policy='raise')
    degrees_of_freedom = n - 1
    standard_error = float(delta.std(ddof=1) / np.sqrt(n))
    half_width = float(t.ppf(0.975, degrees_of_freedom) * standard_error)
    ci_low, ci_high = mean_delta - half_width, mean_delta + half_width
    estimates = np.array([result.statistic, result.pvalue, ci_low, ci_high])
    if not np.isfinite(estimates).all():
        return PairedTTestResult(n, mean_delta, None, None, None, None, None,
                                 'il test ha prodotto valori numerici non finiti')
    return PairedTTestResult(n, mean_delta, float(result.statistic), degrees_of_freedom,
                             float(result.pvalue), ci_low, ci_high)
