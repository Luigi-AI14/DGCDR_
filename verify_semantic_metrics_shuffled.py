#!/usr/bin/env python3
"""
Verification and Contrastive Benchmark Script for Semantic Metrics:
Sentence-BERT (cosine similarity) and BERTScore (Precision, Recall, F1).

Evaluates whether continuous semantic metrics reliably discriminate between:
1. Matched / Congruent pairs: Review of Item X vs LLM Explanation of Item X (Expectation: High score)
2. Shuffled / Negative baseline pairs: Review of Item X vs LLM Explanation of Item Y (X != Y, User Z != User X)

Methodology:
- Strictly aligned with `llm_explainer/judge/shuffle_for_judge.py` using Hungarian bipartite
  matching derangement (`inter_user`, seed 42) to guarantee 100% dataset identity between
  LLM Judge validation and Semantic Metrics validation.
- Computes ONLY continuous semantic metric values:
  * Sentence-BERT Cosine Similarity (`sbert_similarity`)
  * BERTScore Precision (`bertscore_precision` / `P`)
  * BERTScore Recall (`bertscore_recall` / `R`)
  * BERTScore F1 (`bertscore_f1` / `F1`)
- Compares matched vs shuffled distributions (Means, Stds, Medians, Contrastive Margins Delta,
  Paired t-test, Wilcoxon signed-rank test, Cohen's d effect size, and length quintile stratification).
- Excludes win-rates, Hit@k retrieval ranking, and syntactic metrics (BLEU/ROUGE).

Usage:
    # 1. Evaluate using an existing shuffled dataset generated for LLM Judge:
    python verify_semantic_metrics_shuffled.py --shuffled_file results/Cloth-Elec/shuffled/Cloth-Elec_llama3.1_8b_shuffled_inter_user_seed42_20261008_172009.json

    # 2. Evaluate from congruent results (auto-detects or creates matching shuffled dataset):
    python verify_semantic_metrics_shuffled.py --results_file results/Cloth-Elec/llama3.1_8b_v2/Cloth-Elec_llama3.1_8b_v2_20260930_164711.json

    # 3. Pick the latest congruent results in results/:
    python verify_semantic_metrics_shuffled.py --latest
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

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from scipy import stats
except ImportError:
    stats = None

# Base directory points to the repository root (DGCDR_)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Codebase imports
from llm_explainer.config import DEFAULT_SETTINGS
from llm_explainer.judge.shuffle_for_judge import (
    compute_inter_user_derangement,
    create_shuffled_dataset,
)
from llm_explainer.metrics import (
    batch_compute_bert_score,
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
logger = logging.getLogger("SemanticMetricsShuffledValidator")

DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "results_semantic_shuffled")


# ==============================================================================
# 1. UTILITY & FILE FINDING FUNCTIONS
# ==============================================================================

def find_latest_results_file(
    results_dir: str = DEFAULT_RESULTS_DIR,
    domain_pair: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """Finds the most recent congruent results JSON file matching optional filters."""
    pattern = os.path.join(results_dir, "**", "*.json")
    all_files = glob.glob(pattern, recursive=True)

    candidates = []
    for f in all_files:
        norm = os.path.normpath(f)
        bname = os.path.basename(f)
        if (
            bname.startswith("compact_")
            or "_legacy_archive" in norm
            or "verification" in norm
            or "shuffled" in norm
            or "benchmark" in norm
        ):
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


def find_matching_shuffled_file(
    source_results_path: str,
    domain_pair: str,
    seed: int = 42,
    mode: str = "inter_user",
) -> Optional[str]:
    """
    Looks in results/<domain_pair>/shuffled/ for a shuffled file generated
    from the given source results path with matching seed and mode.
    """
    shuffled_dir = os.path.join(DEFAULT_RESULTS_DIR, domain_pair, "shuffled")
    if not os.path.exists(shuffled_dir):
        return None

    pattern = os.path.join(shuffled_dir, "*.json")
    all_shuffled = glob.glob(pattern)

    norm_source = os.path.normpath(os.path.abspath(source_results_path))

    for shuf_path in all_shuffled:
        try:
            with open(shuf_path, "r", encoding="utf-8") as f:
                header = json.load(f)
            s_info = header.get("shuffle_info", {})
            orig_src = s_info.get("source_file")
            if orig_src and os.path.normpath(os.path.abspath(orig_src)) == norm_source:
                if s_info.get("seed") == seed and s_info.get("shuffle_mode") == mode:
                    logger.info(f"Found existing matching shuffled dataset: {shuf_path}")
                    return shuf_path
        except Exception:
            continue

    return None


def format_pval(p: Optional[float]) -> str:
    """Nicely formats p-values with statistical significance stars."""
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


# ==============================================================================
# 2. STATISTICAL FUNCTIONS
# ==============================================================================

def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Computes standard descriptive statistics for a distribution of float values."""
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


def compute_statistical_tests(matched_scores: List[float], shuffled_scores: List[float]) -> Dict[str, Any]:
    """
    Computes paired Student's t-test, Wilcoxon signed-rank test, and Cohen's d effect size.
    Requires len(matched_scores) == len(shuffled_scores).
    """
    if len(matched_scores) != len(shuffled_scores) or len(matched_scores) < 2 or stats is None:
        return {
            "t_stat": None,
            "t_pvalue": None,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "cohens_d": None,
        }

    try:
        m_arr = np.array(matched_scores, dtype=np.float64)
        s_arr = np.array(shuffled_scores, dtype=np.float64)
        diff = m_arr - s_arr

        # Paired Student's t-test
        t_res = stats.ttest_rel(m_arr, s_arr)
        t_stat = round(float(t_res.statistic), 4)
        t_pval = float(t_res.pvalue)

        # Wilcoxon signed-rank test
        try:
            w_res = stats.wilcoxon(m_arr, s_arr)
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
        logger.warning(f"Error computing statistical tests: {e}")
        return {
            "t_stat": None,
            "t_pvalue": None,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "cohens_d": None,
        }


# ==============================================================================
# 3. DATA EXTRACTION & SHUFFLE PAIR BUILDER
# ==============================================================================

def load_or_create_shuffled_pairs(
    shuffled_file: Optional[str] = None,
    results_file: Optional[str] = None,
    domain_pair: str = "Cloth-Elec",
    seed: int = 42,
    min_review_words: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Extracts matched (congruent) and shuffled pairs from a shuffled file or congruent results.
    Guarantees 100% identity with the LLM Judge baseline shuffle.
    Returns:
        (extracted_items, metadata, resolved_shuffled_filepath)
    """
    resolved_shuffled_path = None
    source_congruent_path = None

    if shuffled_file and os.path.exists(shuffled_file):
        resolved_shuffled_path = os.path.abspath(shuffled_file)
        logger.info(f"Loading existing shuffled dataset: {resolved_shuffled_path}")
        with open(resolved_shuffled_path, "r", encoding="utf-8") as f:
            shuffled_data = json.load(f)
    else:
        if not results_file or not os.path.exists(results_file):
            raise FileNotFoundError(f"Neither valid shuffled_file nor results_file provided. results_file={results_file}")

        source_congruent_path = os.path.abspath(results_file)
        # Check if matching shuffled file already exists in results/<domain_pair>/shuffled/
        existing_shuf = find_matching_shuffled_file(
            source_results_path=source_congruent_path,
            domain_pair=domain_pair,
            seed=seed,
            mode="inter_user",
        )
        if existing_shuf:
            resolved_shuffled_path = existing_shuf
            logger.info(f"Using existing matching shuffled dataset for LLM Judge alignment: {resolved_shuffled_path}")
            with open(resolved_shuffled_path, "r", encoding="utf-8") as f:
                shuffled_data = json.load(f)
        else:
            logger.info(
                f"No matching shuffled dataset found. Generating new baseline dataset "
                f"via Hungarian matching (seed={seed}, mode='inter_user')..."
            )
            created_json, _, _ = create_shuffled_dataset(
                source_results_path=source_congruent_path,
                mode="inter_user",
                seed=seed,
                min_review_words=min_review_words,
            )
            resolved_shuffled_path = created_json
            with open(resolved_shuffled_path, "r", encoding="utf-8") as f:
                shuffled_data = json.load(f)

    # Metadata extraction
    domain_pair = shuffled_data.get("domain_pair", domain_pair)
    model_name = shuffled_data.get("model", "unknown_model")
    users = shuffled_data.get("users", [])

    # Extract all items containing review, matched explanation, and shuffled explanation
    extracted_items: List[Dict[str, Any]] = []

    for u_idx, u in enumerate(users):
        uid = u.get("user_id", "")
        for it_idx, it in enumerate(u.get("items", [])):
            rev_text = it.get("user_review_text", "").strip()
            rev_title = it.get("user_review_title", "").strip()
            iid = it.get("id_item") or it.get("item_id", "")
            item_title = it.get("item_title", "").strip()
            q_id = it.get("quintile")
            q_label = it.get("quintile_label")
            word_count = it.get("review_word_count", len(rev_text.split()))

            # Shuffled candidate is stored in 'llm_explanation'
            shuffled_expl = it.get("llm_explanation", "").strip()

            # Matched candidate is stored in 'original_llm_explanation'
            matched_expl = it.get("original_llm_explanation", "").strip()

            # Fallback if original_llm_explanation was not saved: check source congruent file
            if not matched_expl and not it.get("is_shuffled", False):
                # This file is itself a congruent results file
                matched_expl = shuffled_expl
                shuffled_expl = ""

            # Check pre-computed metrics for matched explanation
            metrics = it.get("metrics", {})
            orig_sbert = metrics.get("sbert_similarity")
            orig_bp = metrics.get("bertscore_p")
            orig_br = metrics.get("bertscore_r")
            orig_bf1 = metrics.get("bertscore_f1")

            if rev_text and (matched_expl or shuffled_expl):
                formatted_rev = format_reference_with_title(rev_text, rev_title)
                extracted_items.append({
                    "user_id": uid,
                    "item_id": iid,
                    "item_title": item_title,
                    "user_review_title": rev_title,
                    "user_review_text": rev_text,
                    "formatted_review": formatted_rev,
                    "matched_explanation": matched_expl,
                    "shuffled_explanation": shuffled_expl,
                    "shuffled_from_item_id": it.get("shuffled_from_item_id"),
                    "shuffled_from_item_title": it.get("shuffled_from_item_title"),
                    "shuffled_from_user_id": it.get("shuffled_from_user_id"),
                    "quintile": q_id,
                    "quintile_label": q_label,
                    "word_count": word_count,
                    "precomputed_matched_sbert": orig_sbert,
                    "precomputed_matched_bp": orig_bp,
                    "precomputed_matched_br": orig_br,
                    "precomputed_matched_bf1": orig_bf1,
                })

    # If the loaded file was a congruent file, generate the derangement mapping now
    if extracted_items and not extracted_items[0]["shuffled_explanation"]:
        logger.info(f"Computing Hungarian inter_user derangement for {len(extracted_items)} items...")
        flat_dicts = [
            {"user_id": it["user_id"], "item_id": it["item_id"], "llm_explanation": it["matched_explanation"]}
            for it in extracted_items
        ]
        assignment = compute_inter_user_derangement(flat_dicts, seed=seed)
        for i, j in enumerate(assignment):
            extracted_items[i]["shuffled_explanation"] = extracted_items[j]["matched_explanation"]
            extracted_items[i]["shuffled_from_item_id"] = extracted_items[j]["item_id"]
            extracted_items[i]["shuffled_from_item_title"] = extracted_items[j]["item_title"]
            extracted_items[i]["shuffled_from_user_id"] = extracted_items[j]["user_id"]

    metadata = {
        "domain_pair": domain_pair,
        "model_name": model_name,
        "shuffled_file": resolved_shuffled_path,
        "source_file": source_congruent_path or shuffled_data.get("shuffle_info", {}).get("source_file"),
        "seed": seed,
        "total_items": len(extracted_items),
        "created_at": datetime.datetime.now().isoformat(),
    }
    return extracted_items, metadata, resolved_shuffled_path


# ==============================================================================
# 4. METRIC COMPUTATION PIPELINE
# ==============================================================================

def run_semantic_verification(
    extracted_items: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    sbert_model_name: str = "BAAI/bge-large-en-v1.5",
    sbert_max_seq_length: int = 512,
    bertscore_model: str = "roberta-large",
    bertscore_rescale: bool = True,
    batch_size: int = 8,
    force_recompute: bool = False,
    output_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, str]:
    """
    Computes Sentence-BERT and BERTScore (P, R, F1) for both matched and shuffled pairs,
    runs contrastive comparisons, statistical hypothesis tests, quintile stratifications,
    and exports JSON & Markdown reports to output_dir.
    """
    n_items = len(extracted_items)
    if n_items < 2:
        raise ValueError(f"Need at least 2 items to run contrastive verification! Found {n_items}.")

    domain_pair = metadata.get("domain_pair", "Cloth-Elec")
    model_name = metadata.get("model_name", "unknown_model")

    # --------------------------------------------------------------------------
    # A. SENTENCE-BERT COMPUTATION
    # --------------------------------------------------------------------------
    sbert_resolved_name = validate_and_resolve_sbert_model(sbert_model_name)
    logger.info(f"Loading Sentence-BERT model: '{sbert_resolved_name}'...")
    sbert_model = get_sbert_model(model_name=sbert_resolved_name, max_seq_length=sbert_max_seq_length)

    ref_texts = [it["formatted_review"] for it in extracted_items]
    matched_cands = [it["matched_explanation"] for it in extracted_items]
    shuffled_cands = [it["shuffled_explanation"] for it in extracted_items]

    # Check if pre-computed matched SBERT can be reused
    has_precomputed_sbert = (
        not force_recompute
        and all(it["precomputed_matched_sbert"] is not None for it in extracted_items)
    )

    logger.info(f"Encoding {n_items} reference reviews (batch_size={batch_size})...")
    ref_embs = sbert_model.encode(
        ref_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=n_items > 20,
    )

    if has_precomputed_sbert:
        logger.info("Reusing pre-computed matched Sentence-BERT scores from source dataset.")
        matched_sbert = [float(it["precomputed_matched_sbert"]) for it in extracted_items]
    else:
        logger.info(f"Encoding {n_items} matched candidate explanations (batch_size={batch_size})...")
        m_embs = sbert_model.encode(
            matched_cands,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=n_items > 20,
        )
        matched_sbert = np.sum(m_embs * ref_embs, axis=1)
        matched_sbert = [round(float(np.clip(s, -1.0, 1.0)), 4) for s in matched_sbert]

    logger.info(f"Encoding {n_items} shuffled candidate explanations (batch_size={batch_size})...")
    s_embs = sbert_model.encode(
        shuffled_cands,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=n_items > 20,
    )
    shuffled_sbert = np.sum(s_embs * ref_embs, axis=1)
    shuffled_sbert = [round(float(np.clip(s, -1.0, 1.0)), 4) for s in shuffled_sbert]

    # Free SBERT GPU memory
    del ref_embs, s_embs
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # --------------------------------------------------------------------------
    # B. BERTSCORE COMPUTATION (P, R, F1)
    # --------------------------------------------------------------------------
    has_precomputed_bs = (
        not force_recompute
        and all(
            it["precomputed_matched_bp"] is not None
            and it["precomputed_matched_br"] is not None
            and it["precomputed_matched_bf1"] is not None
            for it in extracted_items
        )
    )

    if has_precomputed_bs:
        logger.info("Reusing pre-computed matched BERTScore values from source dataset.")
        matched_bp = [float(it["precomputed_matched_bp"]) for it in extracted_items]
        matched_br = [float(it["precomputed_matched_br"]) for it in extracted_items]
        matched_bf1 = [float(it["precomputed_matched_bf1"]) for it in extracted_items]
    else:
        logger.info(
            f"Computing matched BERTScore ({bertscore_model}, rescale={bertscore_rescale}) "
            f"for {n_items} pairs (batch_size={batch_size})..."
        )
        matched_bs_res = batch_compute_bert_score(
            candidate_texts=matched_cands,
            reference_texts=ref_texts,
            model_type=bertscore_model,
            batch_size=batch_size,
            rescale_with_baseline=bertscore_rescale,
        )
        matched_bp = matched_bs_res["precision"]
        matched_br = matched_bs_res["recall"]
        matched_bf1 = matched_bs_res["f1"]

    logger.info(
        f"Computing shuffled BERTScore ({bertscore_model}, rescale={bertscore_rescale}) "
        f"for {n_items} pairs (batch_size={batch_size})..."
    )
    shuffled_bs_res = batch_compute_bert_score(
        candidate_texts=shuffled_cands,
        reference_texts=ref_texts,
        model_type=bertscore_model,
        batch_size=batch_size,
        rescale_with_baseline=bertscore_rescale,
    )
    shuffled_bp = shuffled_bs_res["precision"]
    shuffled_br = shuffled_bs_res["recall"]
    shuffled_bf1 = shuffled_bs_res["f1"]

    # Free BERTScore GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # --------------------------------------------------------------------------
    # C. CONTRASTIVE ANALYSIS & COMPARISON FOR EACH METRIC
    # --------------------------------------------------------------------------
    metrics_map = {
        "sbert_similarity": {
            "name": "Sentence-BERT Similarity",
            "matched": matched_sbert,
            "shuffled": shuffled_sbert,
        },
        "bertscore_precision": {
            "name": "BERTScore Precision",
            "matched": matched_bp,
            "shuffled": shuffled_bp,
        },
        "bertscore_recall": {
            "name": "BERTScore Recall",
            "matched": matched_br,
            "shuffled": shuffled_br,
        },
        "bertscore_f1": {
            "name": "BERTScore F1",
            "matched": matched_bf1,
            "shuffled": shuffled_bf1,
        },
    }

    comparison_results: Dict[str, Any] = {}

    for m_key, m_info in metrics_map.items():
        m_vals = m_info["matched"]
        s_vals = m_info["shuffled"]

        m_stats = compute_distribution_stats(m_vals)
        s_stats = compute_distribution_stats(s_vals)

        margin = round(m_stats["mean"] - s_stats["mean"], 4)
        rel_diff = round((margin / s_stats["mean"]) * 100, 2) if s_stats["mean"] != 0 else 0.0

        tests = compute_statistical_tests(m_vals, s_vals)

        comparison_results[m_key] = {
            "display_name": m_info["name"],
            "matched_stats": m_stats,
            "shuffled_stats": s_stats,
            "contrastive_margin_delta": margin,
            "relative_delta_pct": rel_diff,
            "statistical_tests": tests,
        }

    # --------------------------------------------------------------------------
    # D. QUINTILE STRATIFICATION BREAKDOWN (Q1-Q5)
    # --------------------------------------------------------------------------
    qm = QuintileManager(domain_pair=domain_pair, min_words=5)
    quintile_breakdown: Dict[str, Dict[str, Any]] = {}

    for q in qm.quintile_info["quintiles"]:
        qid = q["quintile"]
        label = q["label"]
        min_w = q["min_words"]
        max_w = q["max_words"]
        range_str = f"{min_w} - {max_w}w" if max_w else f">= {min_w}w"

        q_indices = [idx for idx, it in enumerate(extracted_items) if it.get("quintile") == qid]
        cnt = len(q_indices)

        if cnt > 0:
            quintile_breakdown[qid] = {
                "label": label,
                "range_words": range_str,
                "count": cnt,
                "metrics": {},
            }

            for m_key in metrics_map.keys():
                q_m_vals = [metrics_map[m_key]["matched"][idx] for idx in q_indices]
                q_s_vals = [metrics_map[m_key]["shuffled"][idx] for idx in q_indices]

                q_m_mean = round(float(np.mean(q_m_vals)), 4)
                q_s_mean = round(float(np.mean(q_s_vals)), 4)
                q_margin = round(q_m_mean - q_s_mean, 4)
                q_rel = round((q_margin / q_s_mean) * 100, 2) if q_s_mean != 0 else 0.0

                quintile_breakdown[qid]["metrics"][m_key] = {
                    "matched_mean": q_m_mean,
                    "shuffled_mean": q_s_mean,
                    "margin_delta": q_margin,
                    "relative_delta_pct": q_rel,
                }

    # --------------------------------------------------------------------------
    # E. COMPOSE OUTPUT REPORTS (JSON & MARKDOWN)
    # --------------------------------------------------------------------------
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model = sanitize_model_name(model_name)

    if output_dir is None:
        output_dir = os.path.join(DEFAULT_OUTPUT_BASE_DIR, domain_pair)
    os.makedirs(output_dir, exist_ok=True)

    json_report_path = os.path.join(
        output_dir, f"semantic_benchmark_{domain_pair}_{clean_model}_{now_str}.json"
    )
    md_report_path = os.path.join(
        output_dir, f"semantic_benchmark_{domain_pair}_{clean_model}_{now_str}.md"
    )

    # Attach item-level evaluation details
    item_evaluations = []
    for idx, it in enumerate(extracted_items):
        item_evaluations.append({
            "user_id": it["user_id"],
            "item_id": it["item_id"],
            "item_title": it["item_title"],
            "quintile": it["quintile"],
            "quintile_label": it["quintile_label"],
            "review_word_count": it["word_count"],
            "shuffled_from_item_id": it["shuffled_from_item_id"],
            "shuffled_from_item_title": it["shuffled_from_item_title"],
            "shuffled_from_user_id": it["shuffled_from_user_id"],
            "matched_scores": {
                "sbert": matched_sbert[idx],
                "bertscore_p": matched_bp[idx],
                "bertscore_r": matched_br[idx],
                "bertscore_f1": matched_bf1[idx],
            },
            "shuffled_scores": {
                "sbert": shuffled_sbert[idx],
                "bertscore_p": shuffled_bp[idx],
                "bertscore_r": shuffled_br[idx],
                "bertscore_f1": shuffled_bf1[idx],
            },
            "deltas": {
                "sbert": round(matched_sbert[idx] - shuffled_sbert[idx], 4),
                "bertscore_p": round(matched_bp[idx] - shuffled_bp[idx], 4),
                "bertscore_r": round(matched_br[idx] - shuffled_br[idx], 4),
                "bertscore_f1": round(matched_bf1[idx] - shuffled_bf1[idx], 4),
            },
        })

    report_dict: Dict[str, Any] = {
        "metadata": {
            "domain_pair": domain_pair,
            "model_name": model_name,
            "total_items_evaluated": n_items,
            "sbert_model": sbert_resolved_name,
            "bertscore_model": bertscore_model,
            "bertscore_rescaled": bertscore_rescale,
            "shuffled_source_file": metadata.get("shuffled_file"),
            "congruent_source_file": metadata.get("source_file"),
            "random_seed": metadata.get("seed"),
            "timestamp": datetime.datetime.now().isoformat(),
        },
        "metrics_comparison": comparison_results,
        "quintile_breakdown": quintile_breakdown,
        "items": item_evaluations,
    }

    # Save JSON Report
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved full JSON benchmark report to: {json_report_path}")

    # Build Markdown Report
    md_lines = [
        f"# Semantic Metrics Discriminative Benchmark: {domain_pair} ({model_name})\n",
        "## Contrastive Validation: Matched vs. Shuffled Baseline (Hungarian Inter-User)",
        f"- **Evaluated Pairs**: {n_items} items",
        f"- **Domain Pair**: `{domain_pair}`",
        f"- **LLM Evaluated**: `{model_name}`",
        f"- **SBERT Model**: `{sbert_resolved_name}`",
        f"- **BERTScore Model**: `{bertscore_model}` (Rescaled baseline: `{bertscore_rescale}`)",
        f"- **Shuffled Source**: `{os.path.basename(metadata.get('shuffled_file', 'generated_on_the_fly'))}`",
        f"- **Timestamp**: `{datetime.datetime.now().isoformat()}`\n",
        "> [!IMPORTANT]",
        "> The shuffled baseline pairs follow the exact same Hungarian derangement methodology (`inter_user`) ",
        "> as `llm_explainer/judge/shuffle_for_judge.py`, ensuring identical test conditions between LLM Judge ",
        "> and continuous semantic metric validation.\n",
        "### 1. Key Semantic Discriminative Metrics",
        "| Metric | Matched (Congruent) | Shuffled (Baseline) | Contrastive Margin ($\\Delta$) | Rel $\\Delta$ (%) | Paired t-test | Wilcoxon | Cohen's d |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for m_key, m_data in comparison_results.items():
        name = m_data["display_name"]
        m_s = m_data["matched_stats"]
        s_s = m_data["shuffled_stats"]
        margin = m_data["contrastive_margin_delta"]
        rel = m_data["relative_delta_pct"]
        t = m_data["statistical_tests"]
        t_str = f"t={t['t_stat']}, {format_pval(t['t_pvalue'])}"
        w_str = f"W={t['wilcoxon_stat']}, {format_pval(t['wilcoxon_pvalue'])}"
        d_str = f"d={t['cohens_d']}"

        md_lines.append(
            f"| **{name}** | **{m_s['mean']}** &plusmn; {m_s['std']} | **{s_s['mean']}** &plusmn; {s_s['std']} | **{margin:+.4f}** | **{rel:+.1f}%** | {t_str} | {w_str} | {d_str} |"
        )

    md_lines.extend([
        "\n### 2. Score Distribution Descriptive Statistics",
        "| Metric | Condition | Mean | Std | Median | Min | Max | Q25 | Q75 |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for m_key, m_data in comparison_results.items():
        name = m_data["display_name"]
        ms = m_data["matched_stats"]
        ss = m_data["shuffled_stats"]
        md_lines.append(f"| **{name}** | Matched | {ms['mean']} | {ms['std']} | {ms['median']} | {ms['min']} | {ms['max']} | {ms['q25']} | {ms['q75']} |")
        md_lines.append(f"| | Shuffled | {ss['mean']} | {ss['std']} | {ss['median']} | {ss['min']} | {ss['max']} | {ss['q25']} | {ss['q75']} |")

    md_lines.extend([
        "\n### 3. Stratification by Review Length Quintiles (Q1-Q5)",
        "| Quintile | Label | Words Range | Items | Metric | Matched Mean | Shuffled Mean | Margin ($\\Delta$) | Rel $\\Delta$ (%) |",
        "|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|",
    ])

    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        if qid in quintile_breakdown:
            q_info = quintile_breakdown[qid]
            for idx_m, (m_key, m_name) in enumerate([
                ("sbert_similarity", "Sentence-BERT"),
                ("bertscore_precision", "BERTScore-P"),
                ("bertscore_recall", "BERTScore-R"),
                ("bertscore_f1", "BERTScore-F1"),
            ]):
                qm_data = q_info["metrics"][m_key]
                prefix = f"| **{qid}** | {q_info['label']} | {q_info['range_words']} | {q_info['count']} " if idx_m == 0 else "| | | | "
                md_lines.append(
                    f"{prefix}| {m_name} | {qm_data['matched_mean']} | {qm_data['shuffled_mean']} | **{qm_data['margin_delta']:+.4f}** | {qm_data['relative_delta_pct']:+.1f}% |"
                )

    # Example Qualitative Contrasts (Top items)
    md_lines.extend([
        "\n### 4. Qualitative Contrast Examples (First 3 Items)\n"
    ])
    for k in range(min(3, n_items)):
        it = extracted_items[k]
        md_lines.extend([
            f"#### Example #{k+1} (User: `{it['user_id']}`)",
            f"- **Target Product**: {it['item_title']} (`{it['item_id']}`)",
            f"- **Ground-Truth Review** ({it['quintile']} {it['quintile_label']}, {it['word_count']}w):",
            f"  > *\"{it['user_review_text']}\"*",
            f"- **Matched Explanation**:",
            f"  > *\"{it['matched_explanation']}\"*",
            f"  - **Scores**: SBERT = `{matched_sbert[k]}`, P = `{matched_bp[k]}`, R = `{matched_br[k]}`, F1 = `{matched_bf1[k]}`",
            f"- **Shuffled Mismatched Explanation** (From `{it['shuffled_from_item_title']}`, ID: `{it['shuffled_from_item_id']}`, User: `{it['shuffled_from_user_id']}`):",
            f"  > *\"{it['shuffled_explanation']}\"*",
            f"  - **Scores**: SBERT = `{shuffled_sbert[k]}`, P = `{shuffled_bp[k]}`, R = `{shuffled_br[k]}`, F1 = `{shuffled_bf1[k]}`",
            f"  - **Margin Drop ($\\Delta$)**: $\\Delta_{{SBERT}}$ = `{round(matched_sbert[k] - shuffled_sbert[k], 4):+}`, $\\Delta_{{F1}}$ = `{round(matched_bf1[k] - shuffled_bf1[k], 4):+}`\n",
        ])

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved Markdown report to: {md_report_path}")

    # --------------------------------------------------------------------------
    # F. PRINT CONSOLE SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      BENCHMARK METRICHE SEMANTICHE: MATCHED vs. SHUFFLED BASELINE       ")
    print("=" * 80)
    print(f"Item Valutati:             {n_items}")
    print(f"Modello LLM:               {model_name}")
    print(f"Sentence-BERT:             {sbert_resolved_name}")
    print(f"BERTScore:                 {bertscore_model} (Rescaled: {bertscore_rescale})")
    print(f"File Shuffled:             {metadata.get('shuffled_file')}")
    print("-" * 80)
    print(f"{'Metrica':<24} | {'Matched Mean':<14} | {'Shuffled Mean':<14} | {'Delta':<10} | {'Rel %':<8} | {'p-value':<14}")
    print("-" * 80)
    for m_key, m_data in comparison_results.items():
        name = m_data["display_name"]
        m_mean = m_data["matched_stats"]["mean"]
        m_std = m_data["matched_stats"]["std"]
        s_mean = m_data["shuffled_stats"]["mean"]
        s_std = m_data["shuffled_stats"]["std"]
        margin = m_data["contrastive_margin_delta"]
        rel = m_data["relative_delta_pct"]
        pval = format_pval(m_data["statistical_tests"]["t_pvalue"])
        print(f"{name:<24} | {m_mean:>6.4f} ± {m_std:<5.4f} | {s_mean:>6.4f} ± {s_std:<5.4f} | {margin:>+8.4f} | {rel:>+6.1f}% | {pval:<14}")
    print("-" * 80)
    print(f"Report salvati in:\n  - JSON: {json_report_path}\n  - MD:   {md_report_path}")
    print("=" * 80 + "\n")

    return report_dict, json_report_path, md_report_path


# ==============================================================================
# 5. CLI PARSER & MAIN ENTRY POINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and benchmark Sentence-BERT and BERTScore metrics (Matched vs Shuffled baseline)."
    )
    parser.add_argument(
        "--results_file",
        "-f",
        type=str,
        default=None,
        help="Path to congruent validation JSON file in results/. If omitted and no --shuffled_file, picks latest.",
    )
    parser.add_argument(
        "--shuffled_file",
        "-s",
        type=str,
        default=None,
        help="Path to an existing shuffled baseline JSON file (from shuffle_for_judge.py).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        default=False,
        help="Pick the newest congruent validation JSON file in results/.",
    )
    parser.add_argument(
        "--domain_pair",
        "-d",
        type=str,
        default=None,
        help="Domain pair filter (e.g. Cloth-Elec).",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default=None,
        help="Model filter (e.g. llama3.1_8b_v2).",
    )
    parser.add_argument(
        "--sbert_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Sentence-BERT model identifier (default: BAAI/bge-large-en-v1.5).",
    )
    parser.add_argument(
        "--sbert_max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for Sentence-BERT (default: 512).",
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="roberta-large",
        help="BERTScore model identifier (default: roberta-large).",
    )
    parser.add_argument(
        "--bertscore_no_rescale",
        action="store_true",
        default=False,
        help="Disable baseline rescaling for BERTScore (default: baseline rescaling enabled).",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="Batch size for GPU embedding inference (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Hungarian matching derangement (default: 42, aligned with LLM Judge).",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        default=False,
        help="Force recomputing matched metrics even if already present in input JSON.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for JSON and Markdown benchmark reports (default: results_semantic_shuffled/<domain_pair>/).",
    )
    parser.add_argument(
        "--min_review_words",
        type=int,
        default=5,
        help="Minimum words filter threshold for reviews (default: 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine source file
    source_results = args.results_file
    shuffled_file = args.shuffled_file

    if not shuffled_file and (args.latest or not source_results):
        logger.info("Looking for newest congruent results in results/...")
        latest_file = find_latest_results_file(
            domain_pair=args.domain_pair,
            model_name=args.model_name,
        )
        if latest_file:
            source_results = latest_file
            logger.info(f"Selected newest results file: {source_results}")
        else:
            logger.error("No valid results file found in results/.")
            sys.exit(1)

    domain_pair = args.domain_pair or "Cloth-Elec"

    # Load / create pairs
    extracted_items, metadata, resolved_shuf_path = load_or_create_shuffled_pairs(
        shuffled_file=shuffled_file,
        results_file=source_results,
        domain_pair=domain_pair,
        seed=args.seed,
        min_review_words=args.min_review_words,
    )

    # Output directory: default to results_semantic_shuffled/<domain_pair>/
    domain_pair = metadata.get("domain_pair", domain_pair)
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_BASE_DIR, domain_pair)

    # Run verification pipeline
    run_semantic_verification(
        extracted_items=extracted_items,
        metadata=metadata,
        sbert_model_name=args.sbert_model,
        sbert_max_seq_length=args.sbert_max_seq_length,
        bertscore_model=args.bertscore_model,
        bertscore_rescale=not args.bertscore_no_rescale,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
