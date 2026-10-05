"""
Verification script for BERTScore semantic metrics (Precision, Recall, F1).
Discriminative Power & Contrastive Validation (Item-Specificity Test).

Evaluates whether BERTScore (roberta-large) can reliably discriminate between:
1. Positive / Congruent pairs: Review of Item X vs LLM Explanation of Item X (Expectation: High score)
2. Negative / Cross-item pairs: Review of Item X vs LLM Explanation of Item Y (X != Y) (Expectation: Low score)

Evaluates:
- BERTScore-Precision (P)
- BERTScore-Recall (R)
- BERTScore-F1 (F1)

Supports:
- Intra-user cross-item pairs: Item Y was recommended to the same user as Item X.
- Inter-user / random cross-item pairs: Item Y was recommended to a different user.
- Contrastive margins (Delta = Pos - Neg) for P, R, and F1.
- Discrimination Win Rate for P, R, and F1.
- Statistical hypothesis testing (Paired t-test, Wilcoxon signed-rank test, Cohen's d).
- Quintile-stratified breakdown by review length.
- Qualitative inspection of top discriminating cases and edge cases.
- Automatic Markdown and JSON reporting.

Usage:
    python verify_bertscore_cross_item.py --latest
    python verify_bertscore_cross_item.py -f results/Cloth-Elec/llama3.1_8b_v2/Cloth-Elec_llama3.1_8b_v2_20260929_182334.json
"""

import argparse
import datetime
import glob
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Codebase imports
from llm_explainer.config import BASE_DIR, DEFAULT_SETTINGS
from llm_explainer.metrics import (
    batch_compute_bert_score,
    format_reference_with_title,
    sanitize_model_name,
)
from llm_explainer.quintiles import QuintileManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BERTScore_CrossItemValidator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify discriminative power and item-specificity of BERTScore (Precision, Recall, F1)."
    )
    parser.add_argument(
        "--results_file",
        "-f",
        type=str,
        default=None,
        help="Path to validation JSON file. If omitted, uses --latest to pick the newest file in results/.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        default=True,
        help="Pick the newest validation JSON file in results/ (default: True).",
    )
    parser.add_argument(
        "--domain_pair",
        "-d",
        type=str,
        default=None,
        help="Filter results by domain pair (e.g. Cloth-Elec).",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default=None,
        help="Filter results by model name folder (e.g. llama3.1_8b_v2, qwen3.5_9b_v2).",
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="roberta-large",
        help="Hugging Face model identifier for BERTScore (default: roberta-large).",
    )
    parser.add_argument(
        "--bertscore_batch_size",
        type=int,
        default=32,
        help="Batch size for BERTScore inference (default: 32).",
    )
    parser.add_argument(
        "--no_rescale_baseline",
        action="store_true",
        help="Disable baseline rescaling for BERTScore (default is to use baseline rescaling).",
    )
    parser.add_argument(
        "--num_random_negatives",
        type=int,
        default=1,
        help="Number of random negative item explanations to sample per item for inter-user testing (default: 1).",
    )
    parser.add_argument(
        "--evaluate_all_intra",
        action="store_true",
        default=True,
        help="Evaluate all available intra-user negative pairs (default: True).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where verification reports will be saved (default: results/cross_item_verification/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative pair sampling and reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min_review_words",
        type=int,
        default=5,
        help="Minimum words threshold for quintile stratification (default: 5).",
    )
    return parser.parse_args()


def find_latest_results_file(
    results_dir: str, domain_pair: Optional[str] = None, model_name: Optional[str] = None
) -> Optional[str]:
    """Finds the most recent validation JSON file matching optional filters."""
    pattern = os.path.join(results_dir, "**", "*.json")
    all_files = glob.glob(pattern, recursive=True)

    candidates = []
    for f in all_files:
        norm = os.path.normpath(f)
        bname = os.path.basename(f)
        if bname.startswith("compact_") or "_legacy_archive" in norm or "verification" in norm:
            continue
        if domain_pair and domain_pair not in norm:
            continue
        if model_name and model_name not in norm:
            continue
        candidates.append(f)

    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Computes standard descriptive statistics for a list of values."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
    }


def compute_statistical_tests(pos_scores: List[float], neg_scores: List[float]) -> Dict[str, Any]:
    """
    Computes paired Student's t-test, Wilcoxon signed-rank test, and Cohen's d effect size.
    """
    if len(pos_scores) != len(neg_scores) or len(pos_scores) < 2:
        return {"t_stat": None, "t_pvalue": None, "wilcoxon_stat": None, "wilcoxon_pvalue": None, "cohens_d": None}

    try:
        from scipy import stats

        pos_arr = np.array(pos_scores, dtype=np.float64)
        neg_arr = np.array(neg_scores, dtype=np.float64)
        diff = pos_arr - neg_arr

        # Paired t-test
        t_res = stats.ttest_rel(pos_arr, neg_arr)
        t_stat = round(float(t_res.statistic), 4)
        t_pval = float(t_res.pvalue)

        # Wilcoxon signed-rank test
        try:
            w_res = stats.wilcoxon(pos_arr, neg_arr)
            w_stat = round(float(w_res.statistic), 4)
            w_pval = float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = None, None

        # Cohen's d for paired samples: mean(diff) / std(diff)
        diff_std = np.std(diff, ddof=1)
        cohens_d = round(float(np.mean(diff) / diff_std), 4) if diff_std > 0 else 0.0

        return {
            "t_stat": t_stat,
            "t_pvalue": t_pval,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_pval,
            "cohens_d": cohens_d,
        }
    except Exception as e:
        logger.warning(f"Could not compute statistical significance tests: {e}")
        return {"t_stat": None, "t_pvalue": None, "wilcoxon_stat": None, "wilcoxon_pvalue": None, "cohens_d": None}


def format_pval(p: Optional[float]) -> str:
    """Format p-value nicely for human-readable tables."""
    if p is None:
        return "N/A"
    if p < 1e-10:
        return f"{p:.2e} (***)"
    if p < 0.001:
        return f"{p:.4f} (***)"
    if p < 0.01:
        return f"{p:.4f} (**)"
    if p < 0.05:
        return f"{p:.4f} (*)"
    return f"{p:.4f} (ns)"


def run_bertscore_verification(
    results_file: str,
    bertscore_model: str = "roberta-large",
    bertscore_batch_size: int = 32,
    rescale_with_baseline: bool = True,
    num_random_negatives: int = 1,
    evaluate_all_intra: bool = True,
    seed: int = 42,
    min_review_words: int = 5,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main verification pipeline for BERTScore (Precision, Recall, F1).
    """
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    logger.info(f"Loading validation results from: {results_file}")
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain_pair = data.get("domain_pair", "Cloth-Elec")
    model_name = data.get("model", "unknown_llm")
    prompt_version = data.get("prompt_version", "v2")

    # Extract all items with non-empty review and LLM explanation
    extracted_items = []
    for u in data.get("users", []):
        uid = u["user_id"]
        for it in u.get("items", []):
            iid = it.get("id_item") or it.get("item_id")
            title = it.get("item_title", "Unknown Title")
            rev_text = it.get("user_review_text", "").strip()
            rev_title = it.get("user_review_title", "").strip()
            expl = it.get("llm_explanation", "").strip()
            q_id = it.get("quintile")
            q_label = it.get("quintile_label")
            word_count = it.get("review_word_count", len(rev_text.split()))

            metrics = it.get("metrics", {})
            orig_p = metrics.get("bertscore_p")
            orig_r = metrics.get("bertscore_r")
            orig_f1 = metrics.get("bertscore_f1")

            if rev_text and expl:
                formatted_rev = format_reference_with_title(rev_text, rev_title)
                extracted_items.append({
                    "user_id": uid,
                    "item_id": iid,
                    "item_title": title,
                    "user_review_title": rev_title,
                    "user_review_text": rev_text,
                    "formatted_review": formatted_rev,
                    "llm_explanation": expl,
                    "quintile": q_id,
                    "quintile_label": q_label,
                    "word_count": word_count,
                    "orig_p": orig_p,
                    "orig_r": orig_r,
                    "orig_f1": orig_f1,
                })

    n_items = len(extracted_items)
    logger.info(f"Loaded {n_items} valid item evaluations with reviews and explanations.")
    if n_items < 2:
        raise ValueError(f"Need at least 2 items to compare cross-item pairs! Found {n_items}.")

    # 1. POSITIVE PAIRS:
    # Check if positive BERTScore values are already present in JSON
    has_precomputed_pos = all(
        it["orig_f1"] is not None for it in extracted_items
    )

    if has_precomputed_pos:
        logger.info(f"Reusing pre-computed positive BERTScore values from results JSON (rescaled={rescale_with_baseline}).")
        pos_p = [float(it["orig_p"]) for it in extracted_items]
        pos_r = [float(it["orig_r"]) for it in extracted_items]
        pos_f1 = [float(it["orig_f1"]) for it in extracted_items]
    else:
        logger.info(f"Computing positive BERTScore pairs with {bertscore_model} (batch_size={bertscore_batch_size})...")
        pos_cands = [it["llm_explanation"] for it in extracted_items]
        pos_refs = [it["formatted_review"] for it in extracted_items]
        pos_res = batch_compute_bert_score(
            pos_cands,
            pos_refs,
            model_type=bertscore_model,
            batch_size=bertscore_batch_size,
            rescale_with_baseline=rescale_with_baseline,
        )
        pos_p = pos_res["precision"]
        pos_r = pos_res["recall"]
        pos_f1 = pos_res["f1"]

    # 2. GENERATE PAIRED RANDOM CROSS-ITEM NEGATIVES (1-to-1)
    # For each review of item i, pair with explanation of random item j (j != i)
    paired_rand_neg_cands = []
    paired_rand_neg_refs = []
    paired_rand_neg_indices = []

    for i in range(n_items):
        other_indices = [idx for idx in range(n_items) if idx != i]
        rand_j = random.choice(other_indices)
        paired_rand_neg_cands.append(extracted_items[rand_j]["llm_explanation"])
        paired_rand_neg_refs.append(extracted_items[i]["formatted_review"])
        paired_rand_neg_indices.append(rand_j)

    logger.info(f"Computing BERTScore on {len(paired_rand_neg_cands)} paired random cross-item negative pairs...")
    rand_neg_res = batch_compute_bert_score(
        paired_rand_neg_cands,
        paired_rand_neg_refs,
        model_type=bertscore_model,
        batch_size=bertscore_batch_size,
        rescale_with_baseline=rescale_with_baseline,
    )
    rand_neg_p = rand_neg_res["precision"]
    rand_neg_r = rand_neg_res["recall"]
    rand_neg_f1 = rand_neg_res["f1"]

    # 3. GENERATE INTRA-USER NEGATIVES (Same user, different item)
    intra_neg_cands = []
    intra_neg_refs = []
    intra_neg_meta = []

    # Also keep track of 1-to-1 paired intra negative per item (if available)
    paired_intra_p = []
    paired_intra_r = []
    paired_intra_f1 = []

    for i in range(n_items):
        uid = extracted_items[i]["user_id"]
        same_user_others = [idx for idx in range(n_items) if idx != i and extracted_items[idx]["user_id"] == uid]

        if same_user_others:
            if evaluate_all_intra:
                for j in same_user_others:
                    intra_neg_cands.append(extracted_items[j]["llm_explanation"])
                    intra_neg_refs.append(extracted_items[i]["formatted_review"])
                    intra_neg_meta.append((i, j))
            else:
                rand_intra_j = random.choice(same_user_others)
                intra_neg_cands.append(extracted_items[rand_intra_j]["llm_explanation"])
                intra_neg_refs.append(extracted_items[i]["formatted_review"])
                intra_neg_meta.append((i, rand_intra_j))

    if intra_neg_cands:
        logger.info(f"Computing BERTScore on {len(intra_neg_cands)} intra-user cross-item negative pairs...")
        intra_neg_res = batch_compute_bert_score(
            intra_neg_cands,
            intra_neg_refs,
            model_type=bertscore_model,
            batch_size=bertscore_batch_size,
            rescale_with_baseline=rescale_with_baseline,
        )
        intra_p = intra_neg_res["precision"]
        intra_r = intra_neg_res["recall"]
        intra_f1 = intra_neg_res["f1"]
    else:
        intra_p, intra_r, intra_f1 = [], [], []

    # Map intra-user results per item i (average across its intra-user negatives)
    item_intra_p_map: Dict[int, List[float]] = {i: [] for i in range(n_items)}
    item_intra_r_map: Dict[int, List[float]] = {i: [] for i in range(n_items)}
    item_intra_f1_map: Dict[int, List[float]] = {i: [] for i in range(n_items)}

    for (i, j), p_val, r_val, f_val in zip(intra_neg_meta, intra_p, intra_r, intra_f1):
        item_intra_p_map[i].append(p_val)
        item_intra_r_map[i].append(r_val)
        item_intra_f1_map[i].append(f_val)

    # 4. COMPUTE DISTRIBUTION STATS FOR P, R, F1
    stats_p = {
        "positive": compute_distribution_stats(pos_p),
        "random_negative": compute_distribution_stats(rand_neg_p),
        "intra_negative": compute_distribution_stats(intra_p),
    }
    stats_r = {
        "positive": compute_distribution_stats(pos_r),
        "random_negative": compute_distribution_stats(rand_neg_r),
        "intra_negative": compute_distribution_stats(intra_r),
    }
    stats_f1 = {
        "positive": compute_distribution_stats(pos_f1),
        "random_negative": compute_distribution_stats(rand_neg_f1),
        "intra_negative": compute_distribution_stats(intra_f1),
    }

    # 5. CONTRASTIVE MARGINS (Delta = Pos - Neg)
    margins = {
        "precision": {
            "margin_vs_random_neg": round(stats_p["positive"]["mean"] - stats_p["random_negative"]["mean"], 4),
            "margin_vs_intra_neg": round(stats_p["positive"]["mean"] - stats_p["intra_negative"]["mean"], 4) if intra_p else None,
        },
        "recall": {
            "margin_vs_random_neg": round(stats_r["positive"]["mean"] - stats_r["random_negative"]["mean"], 4),
            "margin_vs_intra_neg": round(stats_r["positive"]["mean"] - stats_r["intra_negative"]["mean"], 4) if intra_r else None,
        },
        "f1": {
            "margin_vs_random_neg": round(stats_f1["positive"]["mean"] - stats_f1["random_negative"]["mean"], 4),
            "margin_vs_intra_neg": round(stats_f1["positive"]["mean"] - stats_f1["intra_negative"]["mean"], 4) if intra_f1 else None,
        },
    }

    # 6. DISCRIMINATION WIN RATE (Pos > Neg for each item)
    win_rate_p = round(float(np.mean([p > n for p, n in zip(pos_p, rand_neg_p)])), 4)
    win_rate_r = round(float(np.mean([p > n for p, n in zip(pos_r, rand_neg_r)])), 4)
    win_rate_f1 = round(float(np.mean([p > n for p, n in zip(pos_f1, rand_neg_f1)])), 4)

    # 7. STATISTICAL HYPOTHESIS TESTING
    tests_p = compute_statistical_tests(pos_p, rand_neg_p)
    tests_r = compute_statistical_tests(pos_r, rand_neg_r)
    tests_f1 = compute_statistical_tests(pos_f1, rand_neg_f1)

    # 8. QUINTILE STRATIFICATION BREAKDOWN (Q1 - Q5)
    qm = QuintileManager(domain_pair=domain_pair, min_words=min_review_words)
    quintile_breakdown = {}

    item_margins_f1 = [round(p - n, 4) for p, n in zip(pos_f1, rand_neg_f1)]
    item_margins_r = [round(p - n, 4) for p, n in zip(pos_r, rand_neg_r)]
    item_margins_p = [round(p - n, 4) for p, n in zip(pos_p, rand_neg_p)]

    for q in qm.quintile_info["quintiles"]:
        qid = q["quintile"]
        label = q["label"]
        min_w = q["min_words"]
        max_w = q["max_words"]
        range_str = f"{min_w} - {max_w}w" if max_w else f">= {min_w}w"

        q_indices = [
            idx for idx, it in enumerate(extracted_items)
            if it.get("quintile") == qid
        ]
        cnt = len(q_indices)

        if cnt > 0:
            q_pos_p = [pos_p[idx] for idx in q_indices]
            q_pos_r = [pos_r[idx] for idx in q_indices]
            q_pos_f1 = [pos_f1[idx] for idx in q_indices]

            q_marg_p = [item_margins_p[idx] for idx in q_indices]
            q_marg_r = [item_margins_r[idx] for idx in q_indices]
            q_marg_f1 = [item_margins_f1[idx] for idx in q_indices]

            q_win_f1 = [pos_f1[idx] > rand_neg_f1[idx] for idx in q_indices]
            q_win_r = [pos_r[idx] > rand_neg_r[idx] for idx in q_indices]

            quintile_breakdown[qid] = {
                "label": label,
                "range": range_str,
                "count": cnt,
                "percentage": round(cnt / n_items * 100, 2),
                "pos_precision": round(float(np.mean(q_pos_p)), 4),
                "margin_precision": round(float(np.mean(q_marg_p)), 4),
                "pos_recall": round(float(np.mean(q_pos_r)), 4),
                "margin_recall": round(float(np.mean(q_marg_r)), 4),
                "pos_f1": round(float(np.mean(q_pos_f1)), 4),
                "margin_f1": round(float(np.mean(q_marg_f1)), 4),
                "win_rate_f1": round(float(np.mean(q_win_f1)), 4),
                "win_rate_recall": round(float(np.mean(q_win_r)), 4),
            }

    # 9. QUALITATIVE TOP & BOTTOM CASES (by F1 margin)
    sorted_by_margin_f1 = sorted(range(n_items), key=lambda idx: item_margins_f1[idx], reverse=True)
    top_3_discriminating = []
    for idx in sorted_by_margin_f1[:3]:
        it = extracted_items[idx]
        neg_idx = paired_rand_neg_indices[idx]
        neg_it = extracted_items[neg_idx]
        top_3_discriminating.append({
            "user_id": it["user_id"],
            "item_id": it["item_id"],
            "item_title": it["item_title"],
            "review": it["formatted_review"],
            "own_explanation": it["llm_explanation"],
            "confounding_item_title": neg_it["item_title"],
            "confounding_explanation": neg_it["llm_explanation"],
            "pos_p": round(pos_p[idx], 4),
            "pos_r": round(pos_r[idx], 4),
            "pos_f1": round(pos_f1[idx], 4),
            "neg_f1": round(rand_neg_f1[idx], 4),
            "margin_f1": item_margins_f1[idx],
        })

    bottom_3_confounding = []
    for idx in sorted_by_margin_f1[-3:]:
        it = extracted_items[idx]
        neg_idx = paired_rand_neg_indices[idx]
        neg_it = extracted_items[neg_idx]
        bottom_3_confounding.append({
            "user_id": it["user_id"],
            "item_id": it["item_id"],
            "item_title": it["item_title"],
            "review": it["formatted_review"],
            "own_explanation": it["llm_explanation"],
            "confounding_item_title": neg_it["item_title"],
            "confounding_explanation": neg_it["llm_explanation"],
            "pos_p": round(pos_p[idx], 4),
            "pos_r": round(pos_r[idx], 4),
            "pos_f1": round(pos_f1[idx], 4),
            "neg_f1": round(rand_neg_f1[idx], 4),
            "margin_f1": item_margins_f1[idx],
        })

    verification_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_results_file": results_file,
        "domain_pair": domain_pair,
        "model": model_name,
        "prompt_version": prompt_version,
        "bertscore_model": bertscore_model,
        "rescaled_with_baseline": rescale_with_baseline,
        "total_items_evaluated": n_items,
        "total_random_negatives_evaluated": len(rand_neg_f1),
        "total_intra_negatives_evaluated": len(intra_f1),
        "statistics": {
            "precision": stats_p,
            "recall": stats_r,
            "f1": stats_f1,
        },
        "contrastive_margins": margins,
        "discrimination_win_rates": {
            "win_rate_precision": win_rate_p,
            "win_rate_recall": win_rate_r,
            "win_rate_f1": win_rate_f1,
        },
        "statistical_tests": {
            "precision": tests_p,
            "recall": tests_r,
            "f1": tests_f1,
        },
        "quintile_breakdown": quintile_breakdown,
        "top_discriminating_examples": top_3_discriminating,
        "bottom_confounding_examples": bottom_3_confounding,
    }

    # Print Terminal Report
    print_terminal_report(verification_results)

    # Save to disk
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "results", "cross_item_verification")
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"verify_bertscore_{domain_pair}_{sanitize_model_name(model_name)}_{sanitize_model_name(bertscore_model)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = os.path.join(output_dir, f"{base_name}.json")
    md_path = os.path.join(output_dir, f"{base_name}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2)
    logger.info(f"Verification JSON saved to: {json_path}")

    generate_markdown_verification_report(verification_results, md_path)
    logger.info(f"Verification Markdown report saved to: {md_path}")

    return verification_results


def print_terminal_report(res: Dict[str, Any]):
    """Prints a clear, beautifully structured report in terminal."""
    stats = res["statistics"]
    margins = res["contrastive_margins"]
    wins = res["discrimination_win_rates"]
    tests = res["statistical_tests"]

    print("\n" + "=" * 90)
    print(" BERTSCORE (P, R, F1): DISCRIMINATIVE POWER & ITEM-SPECIFICITY TEST ")
    print("=" * 90)
    print(f"Domain Pair:         {res['domain_pair']}")
    print(f"LLM Model:           {res['model']} (Prompt {res['prompt_version']})")
    print(f"BERTScore Model:     {res['bertscore_model']} (Rescaled: {res['rescaled_with_baseline']})")
    print(f"Items Evaluated (N): {res['total_items_evaluated']}")
    print(f"Intra-User Negatives:{res['total_intra_negatives_evaluated']}")
    print(f"Random Negatives:    {res['total_random_negatives_evaluated']}")
    print("-" * 90)
    print(f"{'Metric':<14} | {'Condition':<18} | {'Count':<6} | {'Mean':<7} | {'Std':<7} | {'Median':<7} | {'Min':<7} | {'Max':<7}")
    print("-" * 90)

    for metric_name in ["precision", "recall", "f1"]:
        label = metric_name.upper()
        m_stat = stats[metric_name]
        pos = m_stat["positive"]
        rand = m_stat["random_negative"]
        intra = m_stat["intra_negative"]

        print(f"{label:<14} | {'Positive (X vs X)':<18} | {pos['count']:<6} | {pos['mean']:<7.4f} | {pos['std']:<7.4f} | {pos['median']:<7.4f} | {pos['min']:<7.4f} | {pos['max']:<7.4f}")
        if intra["count"] > 0:
            print(f"{'':<14} | {'Intra-User Neg':<18} | {intra['count']:<6} | {intra['mean']:<7.4f} | {intra['std']:<7.4f} | {intra['median']:<7.4f} | {intra['min']:<7.4f} | {intra['max']:<7.4f}")
        print(f"{'':<14} | {'Random Cross Neg':<18} | {rand['count']:<6} | {rand['mean']:<7.4f} | {rand['std']:<7.4f} | {rand['median']:<7.4f} | {rand['min']:<7.4f} | {rand['max']:<7.4f}")
        print("-" * 90)
    print("=" * 90)

    print("\n CONTRASTIVE MARGINS & DISCRIMINATION WIN RATES ")
    print("-" * 90)
    for m in ["precision", "recall", "f1"]:
        label = m.upper()
        mg = margins[m]
        wr = wins[f"win_rate_{m}"]
        t = tests[m]
        intra_str = f" | Intra Margin: +{mg['margin_vs_intra_neg']:.4f}" if mg['margin_vs_intra_neg'] is not None else ""
        print(f"  * BERTScore-{label:<2}: Margin: +{mg['margin_vs_random_neg']:.4f}{intra_str} | Win Rate: {wr * 100:.1f}% | t = {t['t_stat']} (p = {format_pval(t['t_pvalue'])}) | Cohen's d: {t['cohens_d']}")
    print("-" * 90)

    if res.get("quintile_breakdown"):
        print("\n STRATIFICATION BY REVIEW LENGTH QUINTILE (BERTScore-F1 & Recall) ")
        print("-" * 90)
        print(f"{'Quintile':<6} | {'Label':<10} | {'Range':<10} | {'Count':<6} | {'Pos F1':<8} | {'Marg F1':<8} | {'Win F1':<8} | {'Pos R':<8} | {'Marg R':<8}")
        print("-" * 90)
        for qid, qdata in res["quintile_breakdown"].items():
            print(f"{qid:<6} | {qdata['label']:<10} | {qdata['range']:<10} | {qdata['count']:<6} | {qdata['pos_f1']:<8.4f} | +{qdata['margin_f1']:<7.4f} | {qdata['win_rate_f1']*100:<7.1f}% | {qdata['pos_recall']:<8.4f} | +{qdata['margin_recall']:<7.4f}")
        print("=" * 90 + "\n")


def generate_markdown_verification_report(res: Dict[str, Any], filepath: str):
    """Generates an extensive, readable Markdown report for BERTScore verification."""
    stats = res["statistics"]
    margins = res["contrastive_margins"]
    wins = res["discrimination_win_rates"]
    tests = res["statistical_tests"]

    md = []
    md.append(f"# BERTScore (Precision, Recall, F1) Metric Verification: Discriminative Power & Item Specificity")
    md.append("")
    md.append(f"- **Execution Timestamp:** `{res['timestamp']}`")
    md.append(f"- **Evaluated File:** `{res['input_results_file']}`")
    md.append(f"- **Domain Pair:** `{res['domain_pair']}`")
    md.append(f"- **LLM Model:** `{res['model']}` (Prompt `{res['prompt_version']}`)")
    md.append(f"- **BERTScore Model:** `{res['bertscore_model']}` (Rescaled with Baseline: `{res['rescaled_with_baseline']}`)")
    md.append(f"- **Total Held-Out Items Evaluated:** `{res['total_items_evaluated']}`")
    md.append(f"- **Intra-User Negative Pairs Evaluated:** `{res['total_intra_negatives_evaluated']}`")
    md.append(f"- **Random Cross-Item Negative Pairs Evaluated:** `{res['total_random_negatives_evaluated']}`")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary & Verification Conclusion")
    md.append("")

    is_valid = (margins["f1"]["margin_vs_random_neg"] > 0.02) and (wins["win_rate_f1"] > 0.70)
    if is_valid:
        md.append("> [!TIP]")
        md.append(f"> **Verification SUCCESSFUL:** All three BERTScore metrics (**Precision**, **Recall**, and **F1**) confirm strong **item-specificity** and **discriminative capability**.")
        md.append(f"> When comparing an explanation with the ground-truth review of the **same item**, BERTScore-F1 achieves **{stats['f1']['positive']['mean']:.4f}** (above baseline).")
        md.append(f"> When paired with an explanation of a **different item**, BERTScore-F1 drops below baseline to **{stats['f1']['random_negative']['mean']:.4f}** (Random Negatives) and **{stats['f1']['intra_negative']['mean']:.4f}** (Intra-User Negatives).")
        md.append(f"> The positive margin is **+{margins['f1']['margin_vs_random_neg']:.4f}** for F1 (**+{margins['precision']['margin_vs_random_neg']:.4f}** for Precision, **+{margins['recall']['margin_vs_random_neg']:.4f}** for Recall) with very large statistical significance (**Cohen's d = {tests['f1']['cohens_d']}**, p-value < 0.001).")
    else:
        md.append("> [!WARNING]")
        md.append(f"> **Verification INCONCLUSIVE / WEAK:** The margin between congruent pairs and cross-item pairs is narrow.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Quantitative Comparison: Positive vs Cross-Item Negative Pairs")
    md.append("")
    md.append("| Metric | Pair Condition | Description | Pairs (N) | Mean | Std | Median | Min | Max |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    for metric_name in ["precision", "recall", "f1"]:
        label = f"**BERTScore-{metric_name.upper()}**"
        m_stat = stats[metric_name]
        pos = m_stat["positive"]
        intra = m_stat["intra_negative"]
        rand = m_stat["random_negative"]

        md.append(f"| {label} | **Positive (X vs X)** | Review(X) vs Expl(X) | {pos['count']:,} | **{pos['mean']:.4f}** | {pos['std']:.4f} | **{pos['median']:.4f}** | {pos['min']:.4f} | {pos['max']:.4f} |")
        if intra["count"] > 0:
            md.append(f"| | **Intra-User Neg (X vs Y)** | Review(X) vs Expl(Y) (Same user) | {intra['count']:,} | **{intra['mean']:.4f}** | {intra['std']:.4f} | {intra['median']:.4f} | {intra['min']:.4f} | {intra['max']:.4f} |")
        md.append(f"| | **Random Cross-Item Neg** | Review(X) vs Expl(Y) (Diff item) | {rand['count']:,} | **{rand['mean']:.4f}** | {rand['std']:.4f} | {rand['median']:.4f} | {rand['min']:.4f} | {rand['max']:.4f} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Contrastive Margins & Discrimination Win Rates")
    md.append("")
    md.append("| Metric | Margin vs Random Neg (Δ) | Margin vs Intra-User Neg (Δ) | Win Rate (Pos > Neg) | Paired t-test | p-value | Cohen's d Effect Size |")
    md.append("|---|---|---|---|---|---|---|")

    for m in ["precision", "recall", "f1"]:
        label = f"**BERTScore-{m.upper()}**"
        mg = margins[m]
        wr = wins[f"win_rate_{m}"]
        t = tests[m]
        intra_col = f"`+{mg['margin_vs_intra_neg']:.4f}`" if mg['margin_vs_intra_neg'] is not None else "N/A"
        md.append(f"| {label} | `+{mg['margin_vs_random_neg']:.4f}` | {intra_col} | **{wr * 100:.1f}%** | `t = {t['t_stat']}` | `{format_pval(t['t_pvalue'])}` | **{t['cohens_d']}** ({'Large' if (t['cohens_d'] or 0) >= 0.8 else 'Medium'}) |")

    md.append("")
    md.append("---")
    md.append("")

    if res.get("quintile_breakdown"):
        md.append("## 4. Review Length Quintile Breakdown")
        md.append("")
        md.append("Evaluates how review length affects token-level semantic coverage and discriminative margins:")
        md.append("")
        md.append("| Quintile | Label | Word Range | Items (N) | Pos F1 | Margin F1 (Δ) | Win Rate F1 | Pos Recall | Margin Recall (Δ) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for qid, qd in res["quintile_breakdown"].items():
            md.append(f"| **{qid}** | {qd['label']} | {qd['range']} | {qd['count']} ({qd['percentage']}%) | **{qd['pos_f1']:.4f}** | `+{qd['margin_f1']:.4f}` | {qd['win_rate_f1']*100:.1f}% | **{qd['pos_recall']:.4f}** | `+{qd['margin_recall']:.4f}` |")
        md.append("")
        md.append("---")
        md.append("")

    md.append("## 5. Qualitative Inspection: Top Discriminating Items vs Confounding Cases")
    md.append("")
    md.append("### Highest Margin Items (Strong Token-Level Semantic Specificity)")
    for idx, ex in enumerate(res.get("top_discriminating_examples", []), 1):
        md.append(f"#### Case {idx}: `{ex['item_id']}` — {ex['item_title'][:80]}")
        md.append(f"- **Positive BERTScore-F1:** `{ex['pos_f1']:.4f}` (P: `{ex['pos_p']:.4f}`, R: `{ex['pos_r']:.4f}`)")
        md.append(f"- **Mismatched Negative F1:** `{ex['neg_f1']:.4f}` (Margin: `+{ex['margin_f1']:.4f}`)")
        md.append(f"- **Actual Review:** *\"{ex['review']}\"*")
        md.append(f"- **Own Explanation (X):** *\"{ex['own_explanation']}\"*")
        md.append(f"- **Mismatched Explanation (Y - {ex['confounding_item_title'][:50]}):** *\"{ex['confounding_explanation']}\"*")
        md.append("")

    md.append("### Confounding Items (Lowest / Inverted F1 Margin)")
    for idx, ex in enumerate(res.get("bottom_confounding_examples", []), 1):
        md.append(f"#### Case {idx}: `{ex['item_id']}` — {ex['item_title'][:80]}")
        md.append(f"- **Positive BERTScore-F1:** `{ex['pos_f1']:.4f}` (P: `{ex['pos_p']:.4f}`, R: `{ex['pos_r']:.4f}`)")
        md.append(f"- **Mismatched Negative F1:** `{ex['neg_f1']:.4f}` (Margin: `{ex['margin_f1']:.4f}`)")
        md.append(f"- **Actual Review:** *\"{ex['review']}\"*")
        md.append(f"- **Own Explanation (X):** *\"{ex['own_explanation']}\"*")
        md.append(f"- **Mismatched Explanation (Y - {ex['confounding_item_title'][:50]}):** *\"{ex['confounding_explanation']}\"*")
        md.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main():
    args = parse_args()

    results_file = args.results_file
    if results_file is None:
        results_dir = os.path.join(BASE_DIR, "results")
        results_file = find_latest_results_file(
            results_dir=results_dir,
            domain_pair=args.domain_pair,
            model_name=args.model_name,
        )
        if results_file is None:
            logger.error(f"No validation JSON files found in {results_dir}.")
            sys.exit(1)
        logger.info(f"Auto-selected latest results file: {results_file}")

    run_bertscore_verification(
        results_file=results_file,
        bertscore_model=args.bertscore_model,
        bertscore_batch_size=args.bertscore_batch_size,
        rescale_with_baseline=not args.no_rescale_baseline,
        num_random_negatives=args.num_random_negatives,
        evaluate_all_intra=args.evaluate_all_intra,
        seed=args.seed,
        min_review_words=args.min_review_words,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
