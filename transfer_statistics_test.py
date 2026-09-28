"""Statistical comparison of paired DGCDR and LightGCN user scores."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import ttest_rel


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

    result = ttest_rel(dgcdr, lightgcn, alternative='two-sided', nan_policy='raise')
    ci = result.confidence_interval(confidence_level=0.95)
    estimates = np.array([result.statistic, result.pvalue, ci.low, ci.high])
    if not np.isfinite(estimates).all():
        return PairedTTestResult(n, mean_delta, None, None, None, None, None,
                                 'il test ha prodotto valori numerici non finiti')
    return PairedTTestResult(n, mean_delta, float(result.statistic), int(result.df),
                             float(result.pvalue), float(ci.low), float(ci.high))
