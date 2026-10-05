"""
Verification script for Sentence-BERT semantic similarity metric ("sbert_similarity").
Discriminative Power & Contrastive Validation (Item-Specificity Test).

Evaluates whether the metric can reliably discriminate between:
1. Positive / Congruent pairs: Review of Item X vs LLM Explanation of Item X (Expectation: High similarity)
2. Negative / Cross-item pairs: Review of Item X vs LLM Explanation of Item Y (X != Y) (Expectation: Low similarity)

Supports:
- Intra-user cross-item pairs: Item Y was recommended to the same user as Item X (challenging test).
- Inter-user cross-item pairs: Item Y was recommended to a different user.
- Global cross-item pairs: All off-diagonal elements in the N x N similarity matrix.
- Contrastive margins, discriminative win rate, retrieval ranking (Hit@1, Hit@5, MRR).
- Statistical hypothesis testing (Paired t-test, Wilcoxon signed-rank test, Cohen's d).
- Quintile-stratified breakdown by review length.
- Qualitative inspection of top discriminating cases and edge cases.
- Automatic Markdown and JSON reporting.

Usage:
    python verify_sbert_cross_item.py --latest
    python verify_sbert_cross_item.py -f results/Cloth-Elec/llama3.1_8b_v2/Cloth-Elec_llama3.1_8b_v2_20260929_182334.json
    python verify_sbert_cross_item.py --latest --sbert_model BAAI/bge-large-en-v1.5
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
from llm_explainer.config import BASE_DIR, DEFAULT_SETTINGS, DOMAIN_CONFIGS
from llm_explainer.metrics import (
    format_reference_with_title,
    get_sbert_model,
    sanitize_model_name,
    validate_and_resolve_sbert_model,
)
from llm_explainer.quintiles import QuintileManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SBERT_CrossItemValidator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify discriminative power and item-specificity of sbert_similarity."
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
        "--sbert_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Sentence-BERT model to use (default: BAAI/bge-large-en-v1.5).",
    )
    parser.add_argument(
        "--sbert_batch_size",
        type=int,
        default=32,
        help="Batch size for Sentence-BERT embedding encoding (default: 32).",
    )
    parser.add_argument(
        "--sbert_max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for Sentence-BERT model (default: 512).",
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
    Requires len(pos_scores) == len(neg_scores).
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


def run_verification(
    results_file: str,
    sbert_model_name: Optional[str] = None,
    sbert_batch_size: int = 32,
    sbert_max_seq_length: int = 512,
    seed: int = 42,
    min_review_words: int = 5,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main verification pipeline.
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

    # Determine SBERT model: prefer CLI argument, otherwise read from JSON file
    if sbert_model_name is None:
        sbert_model_name = data.get("sbert_model", "BAAI/bge-large-en-v1.5")

    sbert_model_name = validate_and_resolve_sbert_model(sbert_model_name)
    logger.info(f"Using Sentence-BERT model: {sbert_model_name}")

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
                    "original_sbert": it.get("metrics", {}).get("sbert_similarity"),
                })

    n_items = len(extracted_items)
    logger.info(f"Loaded {n_items} valid item evaluations with reviews and explanations across users.")
    if n_items < 2:
        raise ValueError(f"Need at least 2 items to compare cross-item pairs! Found {n_items}.")

    # Load SBERT Model
    logger.info(f"Loading Sentence-BERT model '{sbert_model_name}'...")
    model = get_sbert_model(model_name=sbert_model_name, max_seq_length=sbert_max_seq_length)

    # Encode all explanations and formatted reviews in batch
    candidate_texts = [it["llm_explanation"] for it in extracted_items]
    reference_texts = [it["formatted_review"] for it in extracted_items]

    logger.info(f"Batch-encoding {n_items} candidate explanations (batch_size={sbert_batch_size})...")
    cand_embs = model.encode(
        candidate_texts,
        batch_size=sbert_batch_size,
        normalize_embeddings=True,
        show_progress_bar=n_items > 20,
    )

    logger.info(f"Batch-encoding {n_items} reference reviews (batch_size={sbert_batch_size})...")
    ref_embs = model.encode(
        reference_texts,
        batch_size=sbert_batch_size,
        normalize_embeddings=True,
        show_progress_bar=n_items > 20,
    )

    # Compute full N x N Cosine Similarity Matrix:
    # M[i, j] = similarity between candidate explanation of Item i and review of Item j
    logger.info("Computing full N x N pairwise similarity matrix...")
    M = np.dot(cand_embs, ref_embs.T)
    M = np.clip(M, -1.0, 1.0)

    # 1. POSITIVE PAIRS: M[i, i] (Explanation of Item i vs Review of Item i)
    pos_sims = np.diag(M).tolist()

    # 2. ALL NEGATIVE PAIRS: M[i, j] for i != j
    mask_off_diag = ~np.eye(n_items, dtype=bool)
    all_neg_sims = M[mask_off_diag].tolist()

    # 3. INTRA-USER NEGATIVES vs INTER-USER NEGATIVES
    user_ids = [it["user_id"] for it in extracted_items]
    intra_user_neg_sims = []
    inter_user_neg_sims = []

    # Map each item to a 1-to-1 random negative item for paired tests
    paired_random_neg_sims = []
    paired_intra_neg_sims = []

    for i in range(n_items):
        item_intra_negs = []
        item_inter_negs = []
        for j in range(n_items):
            if i == j:
                continue
            sim_val = float(M[i, j])
            if user_ids[i] == user_ids[j]:
                item_intra_negs.append(sim_val)
                intra_user_neg_sims.append(sim_val)
            else:
                item_inter_negs.append(sim_val)
                inter_user_neg_sims.append(sim_val)

        # 1-to-1 paired random negative: pick a random j != i
        other_indices = [idx for idx in range(n_items) if idx != i]
        rand_j = random.choice(other_indices)
        paired_random_neg_sims.append(float(M[i, rand_j]))

        if item_intra_negs:
            paired_intra_neg_sims.append(float(np.mean(item_intra_negs)))
        else:
            paired_intra_neg_sims.append(None)

    # 4. RANKING & RETRIEVAL METRICS
    # For each review j, where does the true explanation j rank among all candidate explanations?
    # Or conversely, for each explanation i, where does true review i rank among all reviews?
    # Standard: Given review i, rank explanations [0..N-1]. Does explanation i score highest?
    # Column j corresponds to review j against all explanations: M[:, j]
    rev_retrieval_ranks = []
    for j in range(n_items):
        col = M[:, j]
        true_score = col[j]
        # rank is 1 + number of other explanations with score strictly higher
        rank = int(np.sum(col > true_score)) + 1
        rev_retrieval_ranks.append(rank)

    rev_retrieval_ranks = np.array(rev_retrieval_ranks)
    hit_1 = round(float(np.mean(rev_retrieval_ranks == 1)), 4)
    hit_3 = round(float(np.mean(rev_retrieval_ranks <= 3)), 4)
    hit_5 = round(float(np.mean(rev_retrieval_ranks <= 5)), 4)
    hit_10 = round(float(np.mean(rev_retrieval_ranks <= 10)), 4)
    mrr = round(float(np.mean(1.0 / rev_retrieval_ranks)), 4)

    # 5. DISCRIMINATIVE WIN RATE
    # Percentage of items where Positive Similarity > Average Cross-Item Similarity for that item
    item_mean_neg_sims = []
    item_win_flags = []
    item_margins = []

    for i in range(n_items):
        row = M[i, :]
        negs = np.delete(row, i)
        avg_neg = float(np.mean(negs))
        item_mean_neg_sims.append(avg_neg)
        pos_s = pos_sims[i]
        item_margins.append(round(pos_s - avg_neg, 4))
        item_win_flags.append(pos_s > avg_neg)

    win_rate = round(float(np.mean(item_win_flags)), 4)

    # Pairwise win rate against 1-to-1 random negative
    pairwise_rand_wins = [p > n for p, n in zip(pos_sims, paired_random_neg_sims)]
    pairwise_win_rate = round(float(np.mean(pairwise_rand_wins)), 4)

    # 6. DESCRIPTIVE STATISTICS
    pos_stats = compute_distribution_stats(pos_sims)
    all_neg_stats = compute_distribution_stats(all_neg_sims)
    intra_neg_stats = compute_distribution_stats(intra_user_neg_sims)
    inter_neg_stats = compute_distribution_stats(inter_user_neg_sims)

    # Contrastive Margins
    margin_all = round(pos_stats["mean"] - all_neg_stats["mean"], 4)
    margin_intra = round(pos_stats["mean"] - intra_neg_stats["mean"], 4) if intra_neg_stats["count"] > 0 else None
    margin_inter = round(pos_stats["mean"] - inter_neg_stats["mean"], 4)

    rel_diff_all = round((margin_all / all_neg_stats["mean"]) * 100, 2) if all_neg_stats["mean"] != 0 else 0.0
    rel_diff_intra = round((margin_intra / intra_neg_stats["mean"]) * 100, 2) if (margin_intra is not None and intra_neg_stats["mean"] != 0) else None

    # 7. HYPOTHESIS TESTING (Positive vs Random Negatives)
    tests_rand = compute_statistical_tests(pos_sims, paired_random_neg_sims)
    tests_mean = compute_statistical_tests(pos_sims, item_mean_neg_sims)

    # 8. QUINTILE STRATIFICATION BREAKDOWN
    qm = QuintileManager(domain_pair=domain_pair, min_words=min_review_words)
    quintile_breakdown = {}

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
            q_pos = [pos_sims[idx] for idx in q_indices]
            q_margins = [item_margins[idx] for idx in q_indices]
            q_wins = [item_win_flags[idx] for idx in q_indices]
            q_ranks = rev_retrieval_ranks[q_indices]

            quintile_breakdown[qid] = {
                "label": label,
                "range": range_str,
                "count": cnt,
                "percentage": round(cnt / n_items * 100, 2),
                "avg_positive_sbert": round(float(np.mean(q_pos)), 4),
                "avg_margin": round(float(np.mean(q_margins)), 4),
                "win_rate": round(float(np.mean(q_wins)), 4),
                "hit_1": round(float(np.mean(q_ranks == 1)), 4),
                "mrr": round(float(np.mean(1.0 / q_ranks)), 4),
            }

    # 9. QUALITATIVE TOP & BOTTOM CASES
    sorted_by_margin = sorted(range(n_items), key=lambda idx: item_margins[idx], reverse=True)
    top_3_discriminating = []
    for idx in sorted_by_margin[:3]:
        it = extracted_items[idx]
        top_3_discriminating.append({
            "user_id": it["user_id"],
            "item_id": it["item_id"],
            "item_title": it["item_title"],
            "review": it["formatted_review"],
            "explanation": it["llm_explanation"],
            "positive_similarity": round(pos_sims[idx], 4),
            "mean_negative_similarity": round(item_mean_neg_sims[idx], 4),
            "margin": item_margins[idx],
        })

    bottom_3_confounding = []
    for idx in sorted_by_margin[-3:]:
        it = extracted_items[idx]
        # Find which item explanation was most erroneously similar to this review
        col = M[:, idx]
        err_idx = int(np.argmax([s if i != idx else -999 for i, s in enumerate(col)]))
        err_it = extracted_items[err_idx]

        bottom_3_confounding.append({
            "user_id": it["user_id"],
            "item_id": it["item_id"],
            "item_title": it["item_title"],
            "review": it["formatted_review"],
            "own_explanation": it["llm_explanation"],
            "positive_similarity": round(pos_sims[idx], 4),
            "confounding_item_id": err_it["item_id"],
            "confounding_item_title": err_it["item_title"],
            "confounding_explanation": err_it["llm_explanation"],
            "confounding_similarity": round(float(col[err_idx]), 4),
            "margin": item_margins[idx],
        })

    # Prepare summary results dictionary
    verification_results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_results_file": results_file,
        "domain_pair": domain_pair,
        "model": model_name,
        "prompt_version": prompt_version,
        "sbert_model": sbert_model_name,
        "total_items_evaluated": n_items,
        "total_negative_pairs": len(all_neg_sims),
        "total_intra_user_negative_pairs": len(intra_user_neg_sims),
        "statistics": {
            "positive_pairs": pos_stats,
            "all_negative_pairs": all_neg_stats,
            "intra_user_negative_pairs": intra_neg_stats,
            "inter_user_negative_pairs": inter_neg_stats,
        },
        "contrastive_margins": {
            "margin_vs_all_negatives": margin_all,
            "relative_increase_vs_all_pct": rel_diff_all,
            "margin_vs_intra_user_negatives": margin_intra,
            "relative_increase_vs_intra_pct": rel_diff_intra,
            "margin_vs_inter_user_negatives": margin_inter,
        },
        "discrimination_metrics": {
            "win_rate_vs_mean_negatives": win_rate,
            "win_rate_pairwise_random": pairwise_win_rate,
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "hit_at_5": hit_5,
            "hit_at_10": hit_10,
            "mrr": mrr,
        },
        "statistical_tests": {
            "vs_random_negative": tests_rand,
            "vs_mean_negative": tests_mean,
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

    base_name = f"verify_{domain_pair}_{sanitize_model_name(model_name)}_{sanitize_model_name(sbert_model_name)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    discrim = res["discrimination_metrics"]
    tests = res["statistical_tests"]["vs_mean_negative"]

    print("\n" + "=" * 90)
    print(" SBERT SIMILARITY: DISCRIMINATIVE POWER & ITEM-SPECIFICITY TEST ")
    print("=" * 90)
    print(f"Domain Pair:         {res['domain_pair']}")
    print(f"LLM Model:           {res['model']} (Prompt {res['prompt_version']})")
    print(f"Sentence-BERT Model: {res['sbert_model']}")
    print(f"Items Evaluated (N): {res['total_items_evaluated']}")
    print(f"Total Neg Pairs:     {res['total_negative_pairs']:,} (Intra-user: {res['total_intra_user_negative_pairs']:,})")
    print("-" * 90)
    print(f"{'Pair Type':<32} | {'Count':<7} | {'Mean':<7} | {'Std':<7} | {'Median':<7} | {'Min':<7} | {'Max':<7}")
    print("-" * 90)

    p = stats["positive_pairs"]
    print(f"{'1. Positive (Expl X vs Rev X)':<32} | {p['count']:<7} | {p['mean']:<7.4f} | {p['std']:<7.4f} | {p['median']:<7.4f} | {p['min']:<7.4f} | {p['max']:<7.4f}")

    intra = stats["intra_user_negative_pairs"]
    if intra["count"] > 0:
        print(f"{'2. Intra-User Neg (Same User, Y!=X)':<32} | {intra['count']:<7} | {intra['mean']:<7.4f} | {intra['std']:<7.4f} | {intra['median']:<7.4f} | {intra['min']:<7.4f} | {intra['max']:<7.4f}")

    inter = stats["inter_user_negative_pairs"]
    print(f"{'3. Inter-User Neg (Diff User, Y!=X)':<32} | {inter['count']:<7} | {inter['mean']:<7.4f} | {inter['std']:<7.4f} | {inter['median']:<7.4f} | {inter['min']:<7.4f} | {inter['max']:<7.4f}")

    all_neg = stats["all_negative_pairs"]
    print(f"{'4. Global Negatives (All Y != X)':<32} | {all_neg['count']:<7} | {all_neg['mean']:<7.4f} | {all_neg['std']:<7.4f} | {all_neg['median']:<7.4f} | {all_neg['min']:<7.4f} | {all_neg['max']:<7.4f}")
    print("=" * 90)

    print("\n CONTRASTIVE MARGIN & DISCRIMINATION ACCURACY ")
    print("-" * 90)
    print(f"  * Margin vs Global Negatives:    +{margins['margin_vs_all_negatives']:.4f}  (+{margins['relative_increase_vs_all_pct']:.1f}% relative difference)")
    if margins["margin_vs_intra_user_negatives"] is not None:
        print(f"  * Margin vs Intra-User Negatives: +{margins['margin_vs_intra_user_negatives']:.4f}  (+{margins['relative_increase_vs_intra_pct']:.1f}% relative difference)")
    print(f"  * Discrimination Win Rate:       {discrim['win_rate_vs_mean_negatives'] * 100:.1f}% (Pos > Mean Neg)")
    print(f"  * Retrieval Top-1 (Hit@1):       {discrim['hit_at_1'] * 100:.1f}%")
    print(f"  * Retrieval Top-5 (Hit@5):       {discrim['hit_at_5'] * 100:.1f}%")
    print(f"  * Mean Reciprocal Rank (MRR):    {discrim['mrr']:.4f}")
    print("-" * 90)

    print("\n STATISTICAL HYPOTHESIS TESTING (Paired Contrastive Differences) ")
    print("-" * 90)
    print(f"  * Paired t-test t-statistic:     {tests['t_stat']} (p-value: {format_pval(tests['t_pvalue'])})")
    print(f"  * Wilcoxon signed-rank stat:     {tests['wilcoxon_stat']} (p-value: {format_pval(tests['wilcoxon_pvalue'])})")
    print(f"  * Cohen's d Effect Size:         {tests['cohens_d']} ({'Large' if (tests['cohens_d'] or 0) >= 0.8 else 'Medium' if (tests['cohens_d'] or 0) >= 0.5 else 'Small'})")
    print("=" * 90)

    if res.get("quintile_breakdown"):
        print("\n STRATIFICATION BY REVIEW LENGTH QUINTILE ")
        print("-" * 90)
        print(f"{'Quintile':<6} | {'Label':<10} | {'Range':<10} | {'Count':<6} | {'Pos SBERT':<10} | {'Margin':<10} | {'Win Rate':<10} | {'Hit@1':<8} | {'MRR':<6}")
        print("-" * 90)
        for qid, qdata in res["quintile_breakdown"].items():
            print(f"{qid:<6} | {qdata['label']:<10} | {qdata['range']:<10} | {qdata['count']:<6} | {qdata['avg_positive_sbert']:<10.4f} | +{qdata['avg_margin']:<9.4f} | {qdata['win_rate']*100:<9.1f}% | {qdata['hit_1']*100:<7.1f}% | {qdata['mrr']:<6.4f}")
        print("=" * 90 + "\n")


def generate_markdown_verification_report(res: Dict[str, Any], filepath: str):
    """Generates an extensive, readable Markdown report."""
    stats = res["statistics"]
    margins = res["contrastive_margins"]
    discrim = res["discrimination_metrics"]
    tests = res["statistical_tests"]["vs_mean_negative"]

    md = []
    md.append(f"# SBERT Similarity Metric Verification: Discriminative Power & Item Specificity")
    md.append("")
    md.append(f"- **Execution Timestamp:** `{res['timestamp']}`")
    md.append(f"- **Evaluated File:** `{res['input_results_file']}`")
    md.append(f"- **Domain Pair:** `{res['domain_pair']}`")
    md.append(f"- **LLM Model:** `{res['model']}` (Prompt `{res['prompt_version']}`)")
    md.append(f"- **Sentence-BERT Model:** `{res['sbert_model']}`")
    md.append(f"- **Total Held-Out Items Evaluated:** `{res['total_items_evaluated']}`")
    md.append(f"- **Cross-Item Negative Pairs Tested:** `{res['total_negative_pairs']:,}`")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary & Verification Conclusion")
    md.append("")

    is_valid = (margins["margin_vs_all_negatives"] > 0.05) and (discrim["win_rate_vs_mean_negatives"] > 0.70)
    if is_valid:
        md.append("> [!TIP]")
        md.append(f"> **Verification SUCCESSFUL:** The `sbert_similarity` metric demonstrates **robust discriminative power**.")
        md.append(f"> Comparing a user review with the LLM explanation for the **same item** yields a mean similarity of **{stats['positive_pairs']['mean']:.4f}**, whereas pairing the same review with explanations of **different items** drops similarity to **{stats['all_negative_pairs']['mean']:.4f}**.")
        md.append(f"> The positive margin is **+{margins['margin_vs_all_negatives']:.4f}** (+{margins['relative_increase_vs_all_pct']:.1f}% relative difference), with a statistically significant effect size (**Cohen's d = {tests['cohens_d']}**, p-value < 0.001).")
    else:
        md.append("> [!WARNING]")
        md.append(f"> **Verification INCONCLUSIVE / WEAK:** The margin between congruent pairs and cross-item pairs is narrow (+{margins['margin_vs_all_negatives']:.4f}).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Quantitative Comparison: Positive vs Cross-Item Negative Pairs")
    md.append("")
    md.append("| Pair Condition | Description | Pairs (N) | Mean Sim | Std | Median | Min | Max |")
    md.append("|---|---|---|---|---|---|---|---|")
    p = stats["positive_pairs"]
    md.append(f"| **Positive Pairs (X vs X)** | Review(X) vs Expl(X) | {p['count']:,} | **{p['mean']:.4f}** | {p['std']:.4f} | **{p['median']:.4f}** | {p['min']:.4f} | {p['max']:.4f} |")
    intra = stats["intra_user_negative_pairs"]
    if intra["count"] > 0:
        md.append(f"| **Intra-User Negative (X vs Y)** | Review(X) vs Expl(Y) (Same user, Y != X) | {intra['count']:,} | **{intra['mean']:.4f}** | {intra['std']:.4f} | {intra['median']:.4f} | {intra['min']:.4f} | {intra['max']:.4f} |")
    inter = stats["inter_user_negative_pairs"]
    md.append(f"| **Inter-User Negative (X vs Y)** | Review(X) vs Expl(Y) (Diff user, Y != X) | {inter['count']:,} | **{inter['mean']:.4f}** | {inter['std']:.4f} | {inter['median']:.4f} | {inter['min']:.4f} | {inter['max']:.4f} |")
    all_n = stats["all_negative_pairs"]
    md.append(f"| **Global Negatives (All X != Y)** | Review(X) vs Expl(Y) (All off-diagonal pairs) | {all_n['count']:,} | **{all_n['mean']:.4f}** | {all_n['std']:.4f} | {all_n['median']:.4f} | {all_n['min']:.4f} | {all_n['max']:.4f} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Contrastive Margins & Discrimination Power")
    md.append("")
    md.append("| Metric | Value | Interpretation |")
    md.append("|---|---|---|")
    md.append(f"| **Contrastive Margin (vs All Negatives)** | `+{margins['margin_vs_all_negatives']:.4f}` | Difference in average similarity between congruent and random mismatched pairs |")
    md.append(f"| **Relative Margin Increase** | `+{margins['relative_increase_vs_all_pct']:.1f}%` | Percentage increase of congruent alignment over cross-item baseline |")
    if margins['margin_vs_intra_user_negatives'] is not None:
        md.append(f"| **Margin vs Intra-User Negatives** | `+{margins['margin_vs_intra_user_negatives']:.4f}` | Strict test: holds user preference constant while varying recommended item |")
        md.append(f"| **Relative Increase vs Intra-User** | `+{margins['relative_increase_vs_intra_pct']:.1f}%` | Percentage increase over same-user cross-item baseline |")
    md.append(f"| **Discrimination Win Rate** | `{discrim['win_rate_vs_mean_negatives'] * 100:.1f}%` | Percentage of items where Positive Similarity > Average Cross-Item Similarity |")
    md.append(f"| **Top-1 Retrieval Accuracy (Hit@1)** | `{discrim['hit_at_1'] * 100:.1f}%` | Fraction of reviews where the true explanation scored highest across all {res['total_items_evaluated']} items |")
    md.append(f"| **Top-5 Retrieval Accuracy (Hit@5)** | `{discrim['hit_at_5'] * 100:.1f}%` | Fraction of reviews where the true explanation is in the top 5 candidates |")
    md.append(f"| **Mean Reciprocal Rank (MRR)** | `{discrim['mrr']:.4f}` | Average reciprocal rank of the true explanation across all candidates |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Hypothesis Testing & Effect Size")
    md.append("")
    md.append("| Test | Statistic | p-value | Significance | Effect Size (Cohen's d) |")
    md.append("|---|---|---|---|---|")
    md.append(f"| **Paired Student's t-test** | `t = {tests['t_stat']}` | `{format_pval(tests['t_pvalue'])}` | {'Statistically Significant (p < 0.001)' if (tests['t_pvalue'] or 1.0) < 0.001 else 'Not Significant'} | **{tests['cohens_d']}** ({'Large effect (d >= 0.8)' if (tests['cohens_d'] or 0) >= 0.8 else 'Medium effect'}) |")
    md.append(f"| **Wilcoxon Signed-Rank Test** | `W = {tests['wilcoxon_stat']}` | `{format_pval(tests['wilcoxon_pvalue'])}` | {'Statistically Significant (p < 0.001)' if (tests['wilcoxon_pvalue'] or 1.0) < 0.001 else 'Not Significant'} | — |")
    md.append("")
    md.append("---")
    md.append("")

    if res.get("quintile_breakdown"):
        md.append("## 5. Review Length Quintile Breakdown")
        md.append("")
        md.append("Stratification helps understand whether SBERT's discriminative ability varies depending on user review length (from short Micro reviews to long detailed Macro reviews):")
        md.append("")
        md.append("| Quintile | Label | Word Range | Items (N) | Pos SBERT | Margin (Δ) | Win Rate | Hit@1 | MRR |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for qid, qd in res["quintile_breakdown"].items():
            md.append(f"| **{qid}** | {qd['label']} | {qd['range']} | {qd['count']} ({qd['percentage']}%) | **{qd['avg_positive_sbert']:.4f}** | `+{qd['avg_margin']:.4f}` | {qd['win_rate']*100:.1f}% | {qd['hit_1']*100:.1f}% | **{qd['mrr']:.4f}** |")
        md.append("")
        md.append("---")
        md.append("")

    md.append("## 6. Qualitative Inspection: Top Discriminating Items vs Confounding Cases")
    md.append("")
    md.append("### Highest Margin Items (Strong Item Specificity)")
    for idx, ex in enumerate(res.get("top_discriminating_examples", []), 1):
        md.append(f"#### Case {idx}: `{ex['item_id']}` — {ex['item_title'][:80]}")
        md.append(f"- **Positive Similarity S(X, X):** `{ex['positive_similarity']:.4f}`")
        md.append(f"- **Mean Negative Similarity S(X, Y):** `{ex['mean_negative_similarity']:.4f}` (Margin: `+{ex['margin']:.4f}`)")
        md.append(f"- **Actual Review:** *\"{ex['review']}\"*")
        md.append(f"- **LLM Explanation:** *\"{ex['explanation']}\"*")
        md.append("")

    md.append("### Challenging / Confounding Items (S(X, X) <= S(X, Y))")
    for idx, ex in enumerate(res.get("bottom_confounding_examples", []), 1):
        md.append(f"#### Case {idx}: `{ex['item_id']}` — {ex['item_title'][:80]}")
        md.append(f"- **Positive Similarity S(X, X):** `{ex['positive_similarity']:.4f}`")
        md.append(f"- **Confounding Negative Item:** `{ex['confounding_item_id']}` — {ex['confounding_item_title'][:70]}")
        md.append(f"- **Confounding Negative Similarity:** `{ex['confounding_similarity']:.4f}` (Margin: `{ex['margin']:.4f}`)")
        md.append(f"- **Actual Review of X:** *\"{ex['review']}\"*")
        md.append(f"- **LLM Explanation of X:** *\"{ex['own_explanation']}\"*")
        md.append(f"- **Confounding Explanation of Y:** *\"{ex['confounding_explanation']}\"*")
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

    run_verification(
        results_file=results_file,
        sbert_model_name=args.sbert_model,
        sbert_batch_size=args.sbert_batch_size,
        sbert_max_seq_length=args.sbert_max_seq_length,
        seed=args.seed,
        min_review_words=args.min_review_words,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
