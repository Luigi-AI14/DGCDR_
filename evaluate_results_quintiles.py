"""
Standalone evaluation CLI script to compute Sentence-BERT semantic similarity
and stratify metrics by Review Length Quintiles (Scenario A, min_words >= 5)
on existing or new validation result files in results/.

Usage:
    python evaluate_results_quintiles.py
    python evaluate_results_quintiles.py --results_file results/validation_Cloth-Elec_50users_20260924_123808.json
    python evaluate_results_quintiles.py --latest
    python evaluate_results_quintiles.py --domain_pair Cloth-Elec --min_words 5
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np

from llm_explainer.config import BASE_DIR, DEFAULT_SETTINGS
from llm_explainer.metrics import (
    evaluate_explanation_vs_review,
    format_reference_with_title,
    get_sbert_model,
)
from llm_explainer.quintiles import QuintileManager, count_words

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("QuintileEvaluator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Sentence-BERT similarity and quintile stratification on validation JSON results."
    )
    parser.add_argument(
        "--results_file",
        "-f",
        type=str,
        default=None,
        help="Path to validation JSON file. If omitted, uses the latest file in results/.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Explicitly pick the newest file in results/ directory.",
    )
    parser.add_argument(
        "--domain_pair",
        "-d",
        type=str,
        default="Cloth-Elec",
        help="Domain pair (default: Cloth-Elec).",
    )
    parser.add_argument(
        "--sbert_model",
        "-m",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-BERT model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--min_words",
        "-w",
        type=int,
        default=5,
        help="Minimum words threshold for quintiles (default: 5).",
    )
    parser.add_argument(
        "--use_title_in_quintiles",
        action="store_true",
        help="Whether to include review title in quintile word counting.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite input results file instead of saving to a new file.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Custom output JSON path.",
    )
    return parser.parse_args()


def find_latest_results_file(results_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(results_dir, "validation_*.json"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_compact_titles(domain_pair: str, cache_dir: str) -> Dict[tuple, str]:
    """Loads review titles from compact cache to enrich existing results."""
    pkl_path = os.path.join(cache_dir, f"compact_{domain_pair}.pkl")
    json_path = os.path.join(cache_dir, f"compact_{domain_pair}.json")
    title_map = {}

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            tgt_revs = data.get("target_reviews", {})
            for k, v in tgt_revs.items():
                title_map[k] = v.get("title", "")
            return title_map
        except Exception as e:
            logger.warning(f"Could not load pkl cache: {e}")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tgt_revs = data.get("target_reviews", {})
            for k_str, v in tgt_revs.items():
                parts = k_str.split(":::")
                if len(parts) == 2:
                    title_map[(parts[0], parts[1])] = v.get("title", "")
            return title_map
        except Exception as e:
            logger.warning(f"Could not load json cache: {e}")

    return title_map


def evaluate_results(
    results_path: str,
    domain_pair: str,
    sbert_model_name: str = "all-MiniLM-L6-v2",
    min_words: int = 5,
    use_title_in_quintiles: bool = False,
    output_path: Optional[str] = None,
    in_place: bool = False,
):
    logger.info(f"Loading results from: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actual_pair = data.get("domain_pair", domain_pair)
    logger.info(f"Domain pair: {actual_pair}")

    # Load titles from compact cache if not already in items
    cache_dir = os.path.join(BASE_DIR, "cache")
    title_map = load_compact_titles(actual_pair, cache_dir)
    logger.info(f"Loaded {len(title_map):,} review metadata entries from cache.")

    # Initialize Quintile stratification manager
    qm = QuintileManager(
        domain_pair=actual_pair,
        min_words=min_words,
        min_rating=data.get("rating_threshold", 4.0),
        use_title=use_title_in_quintiles,
    )

    # Initialize SBERT model
    logger.info(f"Loading Sentence-BERT model: {sbert_model_name}...")
    sbert_model = get_sbert_model(model_name=sbert_model_name)

    all_evaluated_items = []
    global_bleu = []
    global_r1 = []
    global_r2 = []
    global_rl = []
    global_sbert = []

    # Process all users and items
    for u in data.get("users", []):
        uid = u["user_id"]
        user_bleu = []
        user_r1 = []
        user_r2 = []
        user_rl = []
        user_sbert = []

        for item in u.get("items", []):
            iid = item.get("id_item") or item.get("item_id")
            title = item.get("item_title", "")
            rev_text = item.get("user_review_text", "")
            rev_title = item.get("user_review_title")
            if not rev_title:
                rev_title = title_map.get((uid, iid), "")
            item["user_review_title"] = rev_title

            expl = item.get("llm_explanation", "")

            # Quintile classification
            q_id, q_label, word_count = qm.classify_text(
                rev_text,
                title=rev_title if use_title_in_quintiles else None,
            )
            is_filtered = (q_id is None)

            item["review_word_count"] = word_count
            item["quintile"] = q_id
            item["quintile_label"] = q_label
            item["is_filtered_ultrashort"] = is_filtered

            # Ensure metrics dictionary exists and has SBERT
            metrics = item.get("metrics", {})
            if "sbert_similarity" not in metrics or metrics["sbert_similarity"] == 0.0:
                full_rev = format_reference_with_title(rev_text, rev_title)
                eval_out = evaluate_explanation_vs_review(
                    candidate_text=expl,
                    reference_text=rev_text,
                    reference_title=rev_title,
                    sbert_model=sbert_model,
                )
                metrics["sbert_similarity"] = eval_out["sbert_similarity"]
                # Keep BLEU and ROUGE intact if present
                for k in ["bleu", "rouge1_f1", "rouge1_p", "rouge1_r", "rouge2_f1", "rouge2_p", "rouge2_r", "rougeL_f1", "rougeL_p", "rougeL_r"]:
                    if k not in metrics:
                        metrics[k] = eval_out[k]
                item["metrics"] = metrics

            b = metrics.get("bleu", 0.0)
            r1 = metrics.get("rouge1_f1", 0.0)
            r2 = metrics.get("rouge2_f1", 0.0)
            rl = metrics.get("rougeL_f1", 0.0)
            sb = metrics.get("sbert_similarity", 0.0)

            user_bleu.append(b)
            user_r1.append(r1)
            user_r2.append(r2)
            user_rl.append(rl)
            user_sbert.append(sb)

            global_bleu.append(b)
            global_r1.append(r1)
            global_r2.append(r2)
            global_rl.append(rl)
            global_sbert.append(sb)

            all_evaluated_items.append(item)

        # Update user averages
        u["user_averages"] = {
            "avg_bleu": round(float(np.mean(user_bleu)), 4) if user_bleu else 0.0,
            "avg_rouge1_f1": round(float(np.mean(user_r1)), 4) if user_r1 else 0.0,
            "avg_rouge2_f1": round(float(np.mean(user_r2)), 4) if user_r2 else 0.0,
            "avg_rougeL_f1": round(float(np.mean(user_rl)), 4) if user_rl else 0.0,
            "avg_sbert_similarity": round(float(np.mean(user_sbert)), 4) if user_sbert else 0.0,
        }

    # Global averages
    macro_bleu = round(float(np.mean(global_bleu)), 4) if global_bleu else 0.0
    macro_r1 = round(float(np.mean(global_r1)), 4) if global_r1 else 0.0
    macro_r2 = round(float(np.mean(global_r2)), 4) if global_r2 else 0.0
    macro_rl = round(float(np.mean(global_rl)), 4) if global_rl else 0.0
    macro_sbert = round(float(np.mean(global_sbert)), 4) if global_sbert else 0.0

    data["global_averages"] = {
        "macro_avg_bleu": macro_bleu,
        "macro_avg_rouge1_f1": macro_r1,
        "macro_avg_rouge2_f1": macro_r2,
        "macro_avg_rougeL_f1": macro_rl,
        "macro_avg_sbert_similarity": macro_sbert,
    }

    # Quintile breakdown (ignoring < min_words)
    valid_items = [it for it in all_evaluated_items if not it["is_filtered_ultrashort"]]
    filtered_items = [it for it in all_evaluated_items if it["is_filtered_ultrashort"]]

    quintile_breakdown = {}
    for q in qm.quintile_info["quintiles"]:
        qid = q["quintile"]
        label = q["label"]
        min_w = q["min_words"]
        max_w = q["max_words"]
        range_str = f"{min_w} - {max_w}w" if max_w else f">= {min_w}w"

        q_items = [it for it in valid_items if it["quintile"] == qid]
        cnt = len(q_items)
        pct = round(cnt / len(valid_items) * 100, 2) if valid_items else 0.0

        if cnt > 0:
            avg_w = round(float(np.mean([it["review_word_count"] for it in q_items])), 1)
            q_bleu = round(float(np.mean([it["metrics"]["bleu"] for it in q_items])), 4)
            q_r1 = round(float(np.mean([it["metrics"]["rouge1_f1"] for it in q_items])), 4)
            q_r2 = round(float(np.mean([it["metrics"]["rouge2_f1"] for it in q_items])), 4)
            q_rl = round(float(np.mean([it["metrics"]["rougeL_f1"] for it in q_items])), 4)
            q_sbert = round(float(np.mean([it["metrics"]["sbert_similarity"] for it in q_items])), 4)
        else:
            avg_w = 0.0
            q_bleu = q_r1 = q_r2 = q_rl = q_sbert = 0.0

        quintile_breakdown[qid] = {
            "label": label,
            "range_words": range_str,
            "min_words": min_w,
            "max_words": max_w,
            "count": cnt,
            "percentage_of_valid": pct,
            "avg_words": avg_w,
            "avg_bleu": q_bleu,
            "avg_rouge1_f1": q_r1,
            "avg_rouge2_f1": q_r2,
            "avg_rougeL_f1": q_rl,
            "avg_sbert_similarity": q_sbert,
        }

    macro_valid_bleu = round(float(np.mean([it["metrics"]["bleu"] for it in valid_items])), 4) if valid_items else 0.0
    macro_valid_r1 = round(float(np.mean([it["metrics"]["rouge1_f1"] for it in valid_items])), 4) if valid_items else 0.0
    macro_valid_r2 = round(float(np.mean([it["metrics"]["rouge2_f1"] for it in valid_items])), 4) if valid_items else 0.0
    macro_valid_rl = round(float(np.mean([it["metrics"]["rougeL_f1"] for it in valid_items])), 4) if valid_items else 0.0
    macro_valid_sbert = round(float(np.mean([it["metrics"]["sbert_similarity"] for it in valid_items])), 4) if valid_items else 0.0

    quintile_metrics = {
        "min_words_filter": min_words,
        "total_items_evaluated": len(all_evaluated_items),
        "valid_items_evaluated": len(valid_items),
        "filtered_ultrashort_items": len(filtered_items),
        "macro_avg_valid_items": {
            "macro_avg_bleu": macro_valid_bleu,
            "macro_avg_rouge1_f1": macro_valid_r1,
            "macro_avg_rouge2_f1": macro_valid_r2,
            "macro_avg_rougeL_f1": macro_valid_rl,
            "macro_avg_sbert_similarity": macro_valid_sbert,
        },
        "by_quintile": quintile_breakdown,
    }

    data["sbert_model"] = sbert_model_name
    data["min_review_words"] = min_words
    data["use_title_in_quintiles"] = use_title_in_quintiles
    data["quintile_metrics"] = quintile_metrics

    # Save output
    if in_place:
        save_target = results_path
    elif output_path:
        save_target = output_path
    else:
        base, ext = os.path.splitext(results_path)
        save_target = f"{base}_quintiles{ext}"

    with open(save_target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print Report
    print("\n" + "=" * 95)
    print(" VALIDATION RESULTS SUMMARY")
    print("=" * 95)
    print(f"Results File:                    {results_path}")
    print(f"Domain Pair:                     {actual_pair}")
    print(f"Total Users Evaluated:           {len(data.get('users', []))}")
    print(f"Total Held-Out Items Evaluated:  {len(all_evaluated_items)}")
    print(f"Macro Average BLEU:              {macro_bleu:.4f}")
    print(f"Macro Average ROUGE-1 (F1):      {macro_r1:.4f}")
    print(f"Macro Average ROUGE-2 (F1):      {macro_r2:.4f}")
    print(f"Macro Average ROUGE-L (F1):      {macro_rl:.4f}")
    print(f"Macro Average SBERT Sim:         {macro_sbert:.4f}")
    print(f"Saved Enriched Results:          {save_target}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print(f" EVALUATION METRICS STRATIFIED BY REVIEW QUINTILES (Min Words >= {min_words})")
    print("=" * 95)
    print(f"{'Quintile':<10} {'Label':<12} {'Range':<15} {'Count':<8} {'Avg Words':<11} {'BLEU':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'SBERT Sim':<10}")
    print("-" * 95)
    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        qdata = quintile_breakdown[qid]
        print(
            f"{qid:<10} {qdata['label']:<12} {qdata['range_words']:<15} {qdata['count']:<8} "
            f"{qdata['avg_words']:<11.1f} {qdata['avg_bleu']:<10.4f} {qdata['avg_rouge1_f1']:<10.4f} "
            f"{qdata['avg_rouge2_f1']:<10.4f} {qdata['avg_rougeL_f1']:<10.4f} {qdata['avg_sbert_similarity']:<10.4f}"
        )
    print("-" * 95)
    print(
        f"{'Overall (Valid >= ' + str(min_words) + 'w)':<39} {len(valid_items):<8} "
        f"{'-':<11} {macro_valid_bleu:<10.4f} {macro_valid_r1:<10.4f} {macro_valid_r2:<10.4f} {macro_valid_rl:<10.4f} {macro_valid_sbert:<10.4f}"
    )
    print(
        f"{'Filtered Ultra-Short (< ' + str(min_words) + 'w)':<39} {len(filtered_items):<8} "
        f"(Excluded from quintile evaluation)"
    )
    print("=" * 95 + "\n")


def main():
    args = parse_args()
    results_dir = os.path.join(BASE_DIR, "results")

    if args.results_file:
        target_file = args.results_file
    elif args.latest or not args.results_file:
        target_file = find_latest_results_file(results_dir)
        if not target_file:
            logger.error(f"No validation JSON files found in {results_dir}!")
            sys.exit(1)
        logger.info(f"Targeting latest result file: {target_file}")

    if not os.path.exists(target_file):
        logger.error(f"File not found: {target_file}")
        sys.exit(1)

    evaluate_results(
        results_path=target_file,
        domain_pair=args.domain_pair,
        sbert_model_name=args.sbert_model,
        min_words=args.min_words,
        use_title_in_quintiles=args.use_title_in_quintiles,
        output_path=args.output_file,
        in_place=args.in_place,
    )


if __name__ == "__main__":
    main()
