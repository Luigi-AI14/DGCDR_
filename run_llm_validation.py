"""
Main execution script for LLM explanation generation and validation on DGCDR recommendations.

Pipeline steps:
1. Filter dataset keeping rating >= 4.0 and non-empty reviews (check_review_null.py criteria).
2. Extract user history (100% source, 80% target) enriched with item titles, review titles and texts.
3. Extract target held-out items (remaining 20% target test set).
4. Compose personalized English prompt per user, save to file (saved_prompts/prompt_<user_id>.txt).
5. Generate explanations using Qwen 3.5 9B via Ollama.
6. Compute syntactic metrics (BLEU, ROUGE-1, ROUGE-2, ROUGE-L) comparing explanation vs actual user review.
7. Compute user-level and global average metrics.
8. Export structured JSON result file.
"""

import argparse
import datetime
import json
import logging
import os
import sys

from recbole.utils import init_seed

from llm_explainer.config import DEFAULT_SETTINGS, DOMAIN_CONFIGS
from llm_explainer.data_extractor import DataExtractor
from llm_explainer.metrics import evaluate_explanation_vs_review, get_sbert_model
from llm_explainer.ollama_client import OllamaClient
from llm_explainer.prompt_builder import build_user_prompt
from llm_explainer.quintiles import QuintileManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DGCDR_LLM_Validator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and validate LLM explanations for DGCDR recommendations."
    )
    parser.add_argument(
        "--num_users",
        "-n",
        type=int,
        default=DEFAULT_SETTINGS["num_users"],
        help="Number of users to validate (default: 5).",
    )
    parser.add_argument(
        "--domain_pair",
        "-d",
        type=str,
        default=DEFAULT_SETTINGS["domain_pair"],
        choices=list(DOMAIN_CONFIGS.keys()),
        help="Domain pair (default: Cloth-Elec).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=DEFAULT_SETTINGS["seed"],
        help="Random seed for exact RecBole split reproducibility (default: 42).",
    )
    parser.add_argument(
        "--rating_threshold",
        "-r",
        type=float,
        default=DEFAULT_SETTINGS["rating_threshold"],
        help="Minimum rating threshold to include interactions (default: 4.0).",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=DEFAULT_SETTINGS["temperature"],
        help="LLM sampling temperature (default: 0.0 for deterministic reproducibility).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_SETTINGS["model_name"],
        help="Ollama model name (default: qwen3.5:9b).",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default=DEFAULT_SETTINGS["ollama_url"],
        help="Ollama server endpoint URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default=DEFAULT_SETTINGS["prompts_dir"],
        help="Directory to save generated prompt files (default: saved_prompts).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_SETTINGS["output_dir"],
        help="Directory to save validation JSON results (default: results).",
    )
    parser.add_argument(
        "--sbert_model",
        type=str,
        default=DEFAULT_SETTINGS.get("sbert_model", "all-MiniLM-L6-v2"),
        help="Sentence-BERT model name for semantic similarity (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--min_review_words",
        type=int,
        default=DEFAULT_SETTINGS.get("min_review_words", 5),
        help="Minimum words threshold to filter ultra-short reviews in quintiles (default: 5).",
    )
    parser.add_argument(
        "--use_title_in_quintiles",
        action="store_true",
        help="Whether to include review title in quintile word count classification.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform data extraction and prompt generation with mock LLM explanations (dry run).",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    init_seed(args.seed, reproducibility=True)

    print("=" * 80)
    print(" DGCDR EXPLANATION & VALIDATION PIPELINE (Qwen 3.5 9B)")
    print("=" * 80)
    print(f"Domain Pair:         {args.domain_pair}")
    print(f"Number of Users:     {args.num_users}")
    print(f"Seed:                {args.seed}")
    print(f"Temperature:         {args.temperature}")
    print(f"Rating Threshold:    >= {args.rating_threshold}")
    print(f"LLM Model:           {args.model} ({'DRY RUN / MOCK' if args.dry_run else 'Ollama Local API'})")
    print(f"Sentence-BERT Model: {args.sbert_model}")
    print(f"Quintile Min Words:  >= {args.min_review_words} words (excluding < {args.min_review_words})")
    print(f"Use Title in Q-Word: {args.use_title_in_quintiles}")
    print(f"Prompts Directory:   {args.prompts_dir}")
    print(f"Output Directory:    {args.output_dir}")
    print("=" * 80)

    os.makedirs(args.prompts_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Ollama client
    ollama_client = OllamaClient(
        base_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
    )
    if not args.dry_run:
        logger.info(f"Checking Ollama server connectivity at {args.ollama_url}...")
        if not ollama_client.check_health():
            logger.warning(
                f"Could not connect to Ollama at {args.ollama_url}. "
                "Ensure Ollama is running ('ollama serve')."
            )

    # 2. Initialize Quintile stratification manager (Scenario A)
    logger.info(f"Initializing QuintileManager for {args.domain_pair} (min_words={args.min_review_words})...")
    qm = QuintileManager(
        domain_pair=args.domain_pair,
        min_words=args.min_review_words,
        min_rating=args.rating_threshold,
        use_title=args.use_title_in_quintiles,
    )

    # 3. Pre-load Sentence-BERT model for semantic evaluation
    logger.info(f"Loading Sentence-BERT model: {args.sbert_model}...")
    sbert_model = get_sbert_model(model_name=args.sbert_model)

    # 4. Extract Data & Splits
    extractor = DataExtractor(
        domain_pair=args.domain_pair,
        seed=args.seed,
        min_rating=args.rating_threshold,
    )

    users_data = extractor.get_users_for_validation(num_users=args.num_users)
    if not users_data:
        logger.error("No eligible users found matching all constraints!")
        sys.exit(1)

    # Prepare output filenames and dedicated run prompt directory
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_basename = f"validation_{args.domain_pair}_{len(users_data)}users_{run_timestamp}"
    output_filename = f"{output_basename}.json"
    output_filepath = os.path.join(args.output_dir, output_filename)

    run_prompts_dir = os.path.join(args.prompts_dir, f"prompt_{output_basename}")
    os.makedirs(run_prompts_dir, exist_ok=True)
    logger.info(f"Prompts for this run will be saved in: {run_prompts_dir}")

    all_user_results = []
    global_bleu_scores = []
    global_r1_scores = []
    global_r2_scores = []
    global_rl_scores = []
    global_sbert_scores = []
    all_evaluated_items = []

    # 5. Process each user
    for u_idx, user in enumerate(users_data, 1):
        uid = user["user_id"]
        logger.info(f"\n[{u_idx}/{len(users_data)}] Processing User: {uid}")
        logger.info(
            f"  History items: {len(user['source_history'])} source ({args.domain_pair.split('-')[0]}), "
            f"{len(user['target_history'])} target ({args.domain_pair.split('-')[1]})"
        )
        logger.info(f"  Held-out items to explain: {len(user['held_out_items'])}")

        # Build prompt & save to dedicated run directory
        recommended_for_prompt = [
            {"item_id": it["item_id"], "item_title": it["item_title"]}
            for it in user["held_out_items"]
        ]

        prompt_text, prompt_file = build_user_prompt(
            user_id=uid,
            source_domain_name=user["source_domain"],
            target_domain_name=user["target_domain"],
            source_history=user["source_history"],
            target_history=user["target_history"],
            recommended_items=recommended_for_prompt,
            prompts_dir=run_prompts_dir,
        )
        logger.info(f"  Prompt generated and saved to: {prompt_file}")

        # Call LLM
        logger.info(f"  Generating explanations with {args.model}...")
        try:
            llm_explanations = ollama_client.generate_explanations(
                prompt=prompt_text,
                mock=args.dry_run,
                expected_items=recommended_for_prompt,
            )
        except Exception as e:
            logger.error(f"  Error calling LLM for user {uid}: {e}")
            llm_explanations = []

        # Index explanations by item_id
        expl_map = {entry["item_id"]: entry.get("explanation", "") for entry in llm_explanations}

        # Calculate metrics for each held-out item
        user_items_evaluated = []
        user_bleu = []
        user_r1 = []
        user_r2 = []
        user_rl = []
        user_sbert = []

        for held_it in user["held_out_items"]:
            iid = held_it["item_id"]
            title = held_it["item_title"]
            ground_truth_text = held_it["ground_truth_review"]
            ground_truth_title = held_it.get("ground_truth_title", "")
            explanation = expl_map.get(iid, "")

            if not explanation:
                logger.warning(f"  No explanation returned by LLM for item {iid}!")

            # Classify review length quintile (Scenario A: ignore < min_review_words)
            q_id, q_label, word_count = qm.classify_text(
                ground_truth_text,
                title=ground_truth_title if args.use_title_in_quintiles else None,
            )
            is_filtered_ultrashort = (q_id is None)

            # Evaluate syntactic (BLEU, ROUGE) and semantic (Sentence-BERT with review title)
            metrics = evaluate_explanation_vs_review(
                candidate_text=explanation,
                reference_text=ground_truth_text,
                reference_title=ground_truth_title,
                sbert_model=sbert_model,
            )

            user_bleu.append(metrics["bleu"])
            user_r1.append(metrics["rouge1_f1"])
            user_r2.append(metrics["rouge2_f1"])
            user_rl.append(metrics["rougeL_f1"])
            user_sbert.append(metrics["sbert_similarity"])

            global_bleu_scores.append(metrics["bleu"])
            global_r1_scores.append(metrics["rouge1_f1"])
            global_r2_scores.append(metrics["rouge2_f1"])
            global_rl_scores.append(metrics["rougeL_f1"])
            global_sbert_scores.append(metrics["sbert_similarity"])

            item_record = {
                "id_item": iid,
                "item_title": title,
                "user_review_title": ground_truth_title,
                "user_review_text": ground_truth_text,
                "review_word_count": word_count,
                "quintile": q_id,
                "quintile_label": q_label,
                "is_filtered_ultrashort": is_filtered_ultrashort,
                "llm_explanation": explanation,
                "metrics": metrics,
            }
            user_items_evaluated.append(item_record)
            all_evaluated_items.append(item_record)

        # User averages
        avg_bleu = round(sum(user_bleu) / len(user_bleu), 4) if user_bleu else 0.0
        avg_r1 = round(sum(user_r1) / len(user_r1), 4) if user_r1 else 0.0
        avg_r2 = round(sum(user_r2) / len(user_r2), 4) if user_r2 else 0.0
        avg_rl = round(sum(user_rl) / len(user_rl), 4) if user_rl else 0.0
        avg_sbert = round(sum(user_sbert) / len(user_sbert), 4) if user_sbert else 0.0

        user_result = {
            "user_id": uid,
            "prompt_file": prompt_file,
            "items": user_items_evaluated,
            "user_averages": {
                "avg_bleu": avg_bleu,
                "avg_rouge1_f1": avg_r1,
                "avg_rouge2_f1": avg_r2,
                "avg_rougeL_f1": avg_rl,
                "avg_sbert_similarity": avg_sbert,
            },
        }
        all_user_results.append(user_result)

        logger.info(
            f"  User Averages -> BLEU: {avg_bleu:.4f} | ROUGE-1: {avg_r1:.4f} | "
            f"ROUGE-2: {avg_r2:.4f} | ROUGE-L: {avg_rl:.4f} | SBERT Sim: {avg_sbert:.4f}"
        )

    # 6. Global Averages
    macro_bleu = round(sum(global_bleu_scores) / len(global_bleu_scores), 4) if global_bleu_scores else 0.0
    macro_r1 = round(sum(global_r1_scores) / len(global_r1_scores), 4) if global_r1_scores else 0.0
    macro_r2 = round(sum(global_r2_scores) / len(global_r2_scores), 4) if global_r2_scores else 0.0
    macro_rl = round(sum(global_rl_scores) / len(global_rl_scores), 4) if global_rl_scores else 0.0
    macro_sbert = round(sum(global_sbert_scores) / len(global_sbert_scores), 4) if global_sbert_scores else 0.0

    # 7. Stratified Metrics by Review Quintiles (excluding < min_review_words)
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
            avg_w = round(sum(it["review_word_count"] for it in q_items) / cnt, 1)
            q_bleu = round(sum(it["metrics"]["bleu"] for it in q_items) / cnt, 4)
            q_r1 = round(sum(it["metrics"]["rouge1_f1"] for it in q_items) / cnt, 4)
            q_r2 = round(sum(it["metrics"]["rouge2_f1"] for it in q_items) / cnt, 4)
            q_rl = round(sum(it["metrics"]["rougeL_f1"] for it in q_items) / cnt, 4)
            q_sbert = round(sum(it["metrics"]["sbert_similarity"] for it in q_items) / cnt, 4)
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

    macro_valid_bleu = round(sum(it["metrics"]["bleu"] for it in valid_items) / len(valid_items), 4) if valid_items else 0.0
    macro_valid_r1 = round(sum(it["metrics"]["rouge1_f1"] for it in valid_items) / len(valid_items), 4) if valid_items else 0.0
    macro_valid_r2 = round(sum(it["metrics"]["rouge2_f1"] for it in valid_items) / len(valid_items), 4) if valid_items else 0.0
    macro_valid_rl = round(sum(it["metrics"]["rougeL_f1"] for it in valid_items) / len(valid_items), 4) if valid_items else 0.0
    macro_valid_sbert = round(sum(it["metrics"]["sbert_similarity"] for it in valid_items) / len(valid_items), 4) if valid_items else 0.0

    quintile_metrics = {
        "min_words_filter": args.min_review_words,
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

    final_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "domain_pair": args.domain_pair,
        "source_domain": users_data[0]["source_domain"],
        "target_domain": users_data[0]["target_domain"],
        "model": args.model,
        "sbert_model": args.sbert_model,
        "seed": args.seed,
        "temperature": args.temperature,
        "rating_threshold": args.rating_threshold,
        "min_review_words": args.min_review_words,
        "use_title_in_quintiles": args.use_title_in_quintiles,
        "num_users": len(users_data),
        "total_held_out_items_evaluated": len(global_bleu_scores),
        "prompts_dir": run_prompts_dir,
        "users": all_user_results,
        "global_averages": {
            "macro_avg_bleu": macro_bleu,
            "macro_avg_rouge1_f1": macro_r1,
            "macro_avg_rouge2_f1": macro_r2,
            "macro_avg_rougeL_f1": macro_rl,
            "macro_avg_sbert_similarity": macro_sbert,
        },
        "quintile_metrics": quintile_metrics,
    }

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print(" VALIDATION RESULTS SUMMARY")
    print("=" * 95)
    print(f"Total Users Evaluated:           {len(users_data)}")
    print(f"Total Held-Out Items Evaluated:  {len(global_bleu_scores)}")
    print(f"Macro Average BLEU:              {macro_bleu:.4f}")
    print(f"Macro Average ROUGE-1 (F1):      {macro_r1:.4f}")
    print(f"Macro Average ROUGE-2 (F1):      {macro_r2:.4f}")
    print(f"Macro Average ROUGE-L (F1):      {macro_rl:.4f}")
    print(f"Macro Average SBERT Similarity:  {macro_sbert:.4f}")
    print(f"Saved Results JSON:              {output_filepath}")
    print(f"Saved Prompts Directory:         {run_prompts_dir}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print(f" EVALUATION METRICS STRATIFIED BY REVIEW QUINTILES (Min Words >= {args.min_review_words})")
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
        f"{'Overall (Valid >= ' + str(args.min_review_words) + 'w)':<39} {len(valid_items):<8} "
        f"{'-':<11} {macro_valid_bleu:<10.4f} {macro_valid_r1:<10.4f} {macro_valid_r2:<10.4f} {macro_valid_rl:<10.4f} {macro_valid_sbert:<10.4f}"
    )
    print(
        f"{'Filtered Ultra-Short (< ' + str(args.min_review_words) + 'w)':<39} {len(filtered_items):<8} "
        f"(Excluded from quintile evaluation)"
    )
    print("=" * 95 + "\n")



if __name__ == "__main__":
    main()
