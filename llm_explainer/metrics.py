"""
Evaluation metrics module: ROUGE (1, 2, L) and BLEU calculation.
Self-contained, mathematically rigorous implementation with graceful edge-case handling.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    # Tokenize words, removing punctuation
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a list of tokens."""
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def compute_rouge_n(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> Dict[str, float]:
    """
    Compute ROUGE-N (Precision, Recall, F1).
    Overlap is calculated using minimum counts of matching n-grams.
    """
    cand_ngrams = get_ngrams(candidate_tokens, n)
    ref_ngrams = get_ngrams(reference_tokens, n)

    cand_len = max(0, len(candidate_tokens) - n + 1)
    ref_len = max(0, len(reference_tokens) - n + 1)

    if cand_len == 0 or ref_len == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = 0
    for ngram, count in cand_ngrams.items():
        if ngram in ref_ngrams:
            overlap += min(count, ref_ngrams[ngram])

    prec = overlap / cand_len if cand_len > 0 else 0.0
    rec = overlap / ref_len if ref_len > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": prec, "recall": rec, "f1": f1}


def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the length of Longest Common Subsequence (LCS) using dynamic programming."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    # Memory optimized DP with 2 rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)

    return curr[n]


def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> Dict[str, float]:
    """Compute ROUGE-L based on Longest Common Subsequence (LCS)."""
    cand_len = len(candidate_tokens)
    ref_len = len(reference_tokens)

    if cand_len == 0 or ref_len == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(candidate_tokens, reference_tokens)
    prec = lcs / cand_len
    rec = lcs / ref_len
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": prec, "recall": rec, "f1": f1}


def compute_bleu(
    candidate_tokens: List[str], reference_tokens: List[str], max_n: int = 4, weights: Tuple[float, ...] = None
) -> float:
    """
    Compute sentence-level BLEU score with smoothing (Chen and Cherry, 2014, Method 1).
    Smoothing adds a small epsilon when higher order n-gram precision is 0.
    """
    c_len = len(candidate_tokens)
    r_len = len(reference_tokens)

    if c_len == 0 or r_len == 0:
        return 0.0

    # Brevity Penalty (BP)
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(r_len) / float(c_len))

    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)

    p_ns = []
    smooth_applied = False

    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate_tokens, n)
        ref_ngrams = get_ngrams(reference_tokens, n)
        tot_cand_ngrams = max(0, c_len - n + 1)

        if tot_cand_ngrams == 0:
            p_ns.append(0.0)
            continue

        overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items() if ngram in ref_ngrams)

        if overlap > 0:
            p_ns.append(overlap / tot_cand_ngrams)
        else:
            # Smoothing Method 1: replace 0 with small value epsilon
            smooth_applied = True
            p_ns.append(0.1 / tot_cand_ngrams)

    # Compute weighted geometric mean
    log_sum = 0.0
    for w, p in zip(weights, p_ns):
        if p <= 0:
            return 0.0
        log_sum += w * math.log(p)

    bleu = bp * math.exp(log_sum)
    return float(bleu)


def evaluate_explanation_vs_review(candidate_text: str, reference_text: str) -> Dict[str, float]:
    """
    Evaluate candidate explanation against ground-truth user review.
    Returns:
        bleu: float
        rouge1_f1: float
        rouge1_p: float
        rouge1_r: float
        rouge2_f1: float
        rouge2_p: float
        rouge2_r: float
        rougeL_f1: float
        rougeL_p: float
        rougeL_r: float
    """
    cand_tokens = tokenize(candidate_text)
    ref_tokens = tokenize(reference_text)

    r1 = compute_rouge_n(cand_tokens, ref_tokens, n=1)
    r2 = compute_rouge_n(cand_tokens, ref_tokens, n=2)
    rl = compute_rouge_l(cand_tokens, ref_tokens)
    bleu = compute_bleu(cand_tokens, ref_tokens)

    return {
        "bleu": round(bleu, 4),
        "rouge1_f1": round(r1["f1"], 4),
        "rouge1_p": round(r1["precision"], 4),
        "rouge1_r": round(r1["recall"], 4),
        "rouge2_f1": round(r2["f1"], 4),
        "rouge2_p": round(r2["precision"], 4),
        "rouge2_r": round(r2["recall"], 4),
        "rougeL_f1": round(rl["f1"], 4),
        "rougeL_p": round(rl["precision"], 4),
        "rougeL_r": round(rl["recall"], 4),
    }
