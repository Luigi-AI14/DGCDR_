#!/usr/bin/env python3
"""Review-alignment judge: separate system/instance prompts, strict JSON and nullable scores.

The historical G-Eval filenames and weighted fields are retained for compatibility.
Probability-weighted scoring is not computed by this JSON protocol.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llm_explainer.judge.judge_client import call_ollama_judge, judgment_counts

try:
    from llm_explainer.quintiles import QuintileManager, count_words
except ImportError:
    QuintileManager = None

    def count_words(text: str) -> int:
        """Counts alphanumeric words, ignoring punctuation."""
        if not text:
            return 0
        return len(re.findall(r"\b\w+\b", text))


PROMPTS_DIR = os.path.join(REPO_ROOT, "llm_explainer", "prompts")

METRICS_REGISTRY: Dict[str, Dict[str, str]] = {
    "alignment": {
        "id": "alignment",
        "display_name": "Alignment",
        "prompt_file": os.path.join(PROMPTS_DIR, "TEST_PROMPT_ALIGNMENT.txt"),
        "fallback_file": None,
    },
}

DEFAULT_QUINTILE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "Q1": {"label": "Micro", "range_words": "5 - 15w", "min_words": 5, "max_words": 15},
    "Q2": {"label": "Short", "range_words": "16 - 33w", "min_words": 16, "max_words": 33},
    "Q3": {"label": "Medium", "range_words": "34 - 61w", "min_words": 34, "max_words": 61},
    "Q4": {"label": "Detailed", "range_words": "62 - 123w", "min_words": 62, "max_words": 123},
    "Q5": {"label": "In-Depth", "range_words": ">= 124w", "min_words": 124, "max_words": None},
}


def load_text(file_path: str) -> str:
    """Reads a text file with automatic path resolution."""
    if not os.path.exists(file_path):
        candidate = os.path.join(REPO_ROOT, file_path)
        if os.path.exists(candidate):
            file_path = candidate
        else:
            candidate_prompt = os.path.join(PROMPTS_DIR, os.path.basename(file_path))
            if os.path.exists(candidate_prompt):
                file_path = candidate_prompt
            else:
                raise FileNotFoundError(f"File non trovato: {file_path}")
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return f.read().strip()


def load_metric_prompt(metric_info: Dict[str, str]) -> str:
    """Loads system prompt for a metric with fallback support."""
    primary = metric_info.get("prompt_file")
    fallback = metric_info.get("fallback_file")

    if primary and os.path.exists(primary):
        return load_text(primary)
    elif fallback and os.path.exists(fallback):
        return load_text(fallback)
    else:
        raise FileNotFoundError(f"Nessun file di prompt trovato per {metric_info['display_name']} (cercato in {primary})")


def resolve_quintile_schema(
    results_data: Dict[str, Any],
    domain_pair: str = "Cloth-Elec",
    min_words: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolves quintile definitions from results JSON metadata, QuintileManager, or defaults.
    """
    by_q = results_data.get("quintile_metrics", {}).get("by_quintile")
    if by_q and isinstance(by_q, dict) and len(by_q) == 5:
        schema = {}
        for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            if qid in by_q:
                qinfo = by_q[qid]
                schema[qid] = {
                    "label": qinfo.get("label", ""),
                    "range_words": qinfo.get("range_words", ""),
                    "min_words": qinfo.get("min_words", 0),
                    "max_words": qinfo.get("max_words"),
                }
        if len(schema) == 5:
            return schema

    if QuintileManager is not None:
        try:
            qm = QuintileManager(domain_pair=domain_pair, min_words=min_words)
            schema = {}
            for q in qm.quintile_info.get("quintiles", []):
                qid = q["quintile"]
                min_w = q["min_words"]
                max_w = q["max_words"]
                range_str = f"{min_w} - {max_w}w" if max_w else f">= {min_w}w"
                schema[qid] = {
                    "label": q["label"],
                    "range_words": range_str,
                    "min_words": min_w,
                    "max_words": max_w,
                }
            if len(schema) == 5:
                return schema
        except Exception:
            pass

    return DEFAULT_QUINTILE_SCHEMA


def classify_review_quintile(
    word_count: int,
    schema: Dict[str, Dict[str, Any]],
    min_words: int = 5,
) -> Tuple[Optional[str], str]:
    """
    Classifies word_count into quintile (Q1-Q5).
    If word_count < min_words, returns (None, 'Filtered_UltraShort').
    """
    if word_count < min_words:
        return None, "Filtered_UltraShort"

    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        if qid not in schema:
            continue
        low = schema[qid]["min_words"]
        high = schema[qid]["max_words"]
        if high is None:
            if word_count >= low:
                return qid, schema[qid]["label"]
        else:
            if low <= word_count <= high:
                return qid, schema[qid]["label"]

    return "Q5", schema.get("Q5", {}).get("label", "In-Depth")


def format_instance_prompt(
    template: str,
    user_review: str,
    explanation: str,
    metric_name: str,
    item_title: Optional[str] = None,
) -> str:
    """
    Formats the three input fields. Legacy item-title aliases remain supported.
    """
    content = template

    if "{metric_name}" in content:
        content = content.replace("{metric_name}", metric_name)

    format_kwargs = {
        "user_review": user_review,
        "explanation": explanation,
        "metric_name": metric_name,
        "item_title": item_title or "",
        "item_name": item_title or "",
        "product_name": item_title or "",
        "product_title": item_title or "",
    }

    return content.format(**format_kwargs)


def extract_users_from_results(
    results_path: str,
    max_users: Optional[int] = None,
    min_words: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts users and their held-out items from the results JSON file.
    Filters out ultra-short reviews (< min_words words).
    Classifies items into review length quintiles (Q1-Q5).
    Returns (extracted_users, stats_dict).
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"File risultati non trovato: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain_pair = data.get("domain_pair", "Cloth-Elec")
    quintile_schema = resolve_quintile_schema(data, domain_pair=domain_pair, min_words=min_words)

    all_users = data.get("users", [])
    selected_users = all_users[:max_users] if max_users is not None else all_users

    total_source_items = 0
    filtered_ultrashort_items = 0
    skipped_missing_input_items = 0
    valid_items_count = 0

    extracted = []
    for u in selected_users:
        user_id = u.get("user_id", "")
        user_items = []
        for it in u.get("items", []):
            total_source_items += 1
            item_title = it.get("item_title", "").strip()
            rev_title = it.get("user_review_title", "").strip()
            rev_text = it.get("user_review_text", "").strip()
            explanation = it.get("llm_explanation", "").strip()

            if not item_title or not rev_text or not explanation:
                skipped_missing_input_items += 1
                continue

            # Review word count
            word_count = it.get("review_word_count")
            if word_count is None:
                word_count = count_words(rev_text)

            # Filter ultra-short (< min_words)
            is_filtered = it.get("is_filtered_ultrashort")
            if is_filtered is None:
                is_filtered = (word_count < min_words)
            else:
                is_filtered = is_filtered or (word_count < min_words)

            if is_filtered:
                filtered_ultrashort_items += 1
                continue

            # Quintile classification
            q_id = it.get("quintile")
            q_label = it.get("quintile_label")
            if not q_id or not q_label or q_id not in quintile_schema:
                q_id, q_label = classify_review_quintile(word_count, quintile_schema, min_words=min_words)

            # Combina titolo e testo della recensione se non duplicati
            if rev_title and rev_title.lower() not in rev_text.lower():
                full_review = f"{rev_title}. {rev_text}"
            else:
                full_review = rev_text

            item_entry = {
                "item_id": it.get("id_item", ""),
                "item_title": item_title,
                "user_review": full_review,
                "review_word_count": word_count,
                "quintile": q_id,
                "quintile_label": q_label,
                "explanation": explanation,
            }
            if it.get("is_shuffled"):
                item_entry["is_shuffled"] = True
            if it.get("original_llm_explanation"):
                item_entry["original_llm_explanation"] = it["original_llm_explanation"]
            if it.get("shuffled_from_item_id"):
                item_entry["shuffled_from_item_id"] = it["shuffled_from_item_id"]
            if it.get("shuffled_from_item_title"):
                item_entry["shuffled_from_item_title"] = it["shuffled_from_item_title"]
            if it.get("shuffled_from_user_id"):
                item_entry["shuffled_from_user_id"] = it["shuffled_from_user_id"]
            user_items.append(item_entry)
            valid_items_count += 1

        if user_items:
            extracted.append({
                "user_id": user_id,
                "items": user_items,
            })

    stats = {
        "total_source_users": len(selected_users),
        "total_source_items": total_source_items,
        "valid_items_count": valid_items_count,
        "filtered_ultrashort_items": filtered_ultrashort_items,
        "skipped_missing_input_items": skipped_missing_input_items,
        "min_words_filter": min_words,
        "quintile_schema": quintile_schema,
        "is_shuffled": data.get("is_shuffled", False),
        "shuffle_info": data.get("shuffle_info", {}),
        "domain_pair": domain_pair,
    }

    return extracted, stats


def save_results_json(output_file: str, summary: Dict[str, Any], evaluated_users: List[Dict[str, Any]]) -> None:
    """
    Saves results to a JSON file.
    Contains global mean score, global mean weighted score, and quintile stratification.
    Includes judge reasons, response statuses and raw attempt diagnostics.
    """
    payload = {
        "global_mean_score": summary["global_mean_score"],
        "global_mean_weighted_score": summary["global_mean_weighted_score"],
        "total_users_evaluated": summary["total_users_evaluated"],
        "total_items_evaluated": summary["total_items_evaluated"],
        "total_items_in_source": summary.get("total_items_in_source", summary["total_items_evaluated"]),
        "filtered_ultrashort_items": summary.get("filtered_ultrashort_items", 0),
        "skipped_missing_input_items": summary.get("skipped_missing_input_items", 0),
        "min_review_words_filter": summary.get("min_review_words_filter", 5),
        "quintile_metrics": summary.get("quintile_metrics", {}),
        "metadata": {
            "metric_id": summary["metric_id"],
            "metric_name": summary["metric_name"],
            "model": summary["model"],
            "temperature": summary["temperature"],
            "timestamp": summary["timestamp"],
            "source_results": summary["source_results"],
            **summary["protocol_metadata"],
        },
        "judgment_counts": summary["judgment_counts"],
        "is_shuffled": summary.get("is_shuffled", False),
        "shuffle_info": summary.get("shuffle_info", {}),
        "users": evaluated_users,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_results_markdown(output_file: str, summary: Dict[str, Any], evaluated_users: List[Dict[str, Any]]) -> None:
    """
    Saves results to a Markdown (.md) file.
    Contains global mean score, global mean weighted score, quintile stratification table,
    and detailed items including reasons and assessment statuses.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    mean_s_str = f"{summary['global_mean_score']:.4f} / 5" if summary["global_mean_score"] is not None else "N/A"
    mean_w_str = f"{summary['global_mean_weighted_score']:.4f} / 5.0" if summary["global_mean_weighted_score"] is not None else "N/A"
    min_w = summary.get("min_review_words_filter", 5)
    counts = summary["judgment_counts"]

    lines = [
        f"# Review Alignment Evaluation Report: {summary['metric_name']}\n",
        "## Global Summary",
        f"- **Metric**: `{summary['metric_name']}` (`{summary['metric_id']}`)",
        f"- **Global Mean Score**: {mean_s_str}",
        f"- **Global Mean Weighted Score**: {mean_w_str}",
        f"- **Attempted Items**: {counts['attempted_count']}",
        f"- **Insufficient Evidence**: {counts['insufficient_evidence_count']}",
        f"- **Technical Failures**: {counts['failed_count']}",
        f"- **Score Coverage**: {counts['score_coverage_pct']}%",
        "- Weighted scores are not computed by this judge protocol.",
        f"- **Total Users Evaluated**: {summary['total_users_evaluated']}",
        f"- **Total Valid Items Evaluated**: {summary['total_items_evaluated']} (&ge; {min_w} words)",
        f"- **Filtered Ultra-Short Items**: {summary.get('filtered_ultrashort_items', 0)} (< {min_w} words excluded)",
        f"- **Missing Input Items**: {summary.get('skipped_missing_input_items', 0)}",
        f"- **Total Items in Source**: {summary.get('total_items_in_source', summary['total_items_evaluated'])}",
        f"- **Model**: `{summary['model']}`",
        f"- **Timestamp**: `{summary['timestamp']}`",
        f"- **Source File**: `{summary['source_results']}`\n",
    ]

    if summary.get("is_shuffled"):
        lines.append(
            "> ⚠️ **NEGATIVE BASELINE RUN (SHUFFLED PAIRS)**: This run evaluates randomized cross-item review-explanation pairs to establish the discriminative lower bound baseline for the LLM Judge.\n"
        )

    lines.append("---\n")

    # Tabella Stratificazione Quintili
    qm = summary.get("quintile_metrics", {})
    by_q = qm.get("by_quintile", {})
    if by_q:
        lines.append(f"## Stratified Metrics by Review Length Quintiles (Min Words &ge; {min_w})\n")
        lines.append("| Quintile | Label | Word Range | Attempted | % Attempted | Avg Words | Mean Score | Mean Weighted Score | Scored | Insufficient | Failed |")
        lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            if qid in by_q:
                q = by_q[qid]
                s_str = f"{q['avg_score']:.4f} / 5" if q.get("avg_score") is not None else "N/A"
                w_str = f"{q['avg_weighted_score']:.4f} / 5.0" if q.get("avg_weighted_score") is not None else "N/A"
                row = [
                    f"**{qid}**",
                    q.get("label", ""),
                    q.get("range_words", ""),
                    str(q.get("count", 0)),
                    f"{q.get('percentage_of_valid', 0.0):.1f}%",
                    f"{q.get('avg_words', 0.0):.1f}",
                    s_str,
                    w_str,
                    str(q["scored_count"]),
                    str(q["insufficient_evidence_count"]),
                    str(q["failed_count"]),
                ]
                lines.append("| " + " | ".join(row) + " |")

        # Riga di sintesi globale
        lines.append(
            f"| **Overall** | - | - | {counts['attempted_count']} | 100.0% | - | {mean_s_str} | {mean_w_str} | {counts['scored_count']} | {counts['insufficient_evidence_count']} | {counts['failed_count']} |"
        )
        lines.append("")
        filtered_cnt = summary.get("filtered_ultrashort_items", 0)
        if filtered_cnt > 0:
            lines.append(
                f"> **Note:** {filtered_cnt} ultra-short item review(s) (< {min_w} words) were filtered out to prevent uninformative evaluation skew.\n"
            )
        lines.append("---\n")

    lines.append("## Evaluated Users\n")

    for u_idx, u in enumerate(evaluated_users, 1):
        u_mean_s = f"{u['user_mean_score']:.4f} / 5" if u["user_mean_score"] is not None else "N/A"
        u_mean_w = f"{u['user_mean_weighted_score']:.4f} / 5.0" if u["user_mean_weighted_score"] is not None else "N/A"

        lines.append(f"### User {u_idx}: `{u['user_id']}`")
        lines.append(f"- **User Mean Score**: {u_mean_s}")
        lines.append(f"- **User Mean Weighted Score**: {u_mean_w}")
        uc = u["judgment_counts"]
        lines.append(f"- **Judgments**: {uc['scored_count']} scored, {uc['insufficient_evidence_count']} insufficient, {uc['failed_count']} failed")
        lines.append(f"- **Items Evaluated**: {u['items_count']}\n")
        lines.append("#### Items:")

        for it_idx, it in enumerate(u["items"], 1):
            w_score = f"{it['weighted_score']:.4f} / 5.0" if it.get("weighted_score") is not None else "N/A"
            q_label_str = f"`{it.get('quintile', '-')}` ({it.get('quintile_label', '-')}, {it.get('review_word_count', '-')} words)"
            lines.append(f"##### Item {it_idx} (ID: `{it.get('item_id', '')}`)")
            score_text = f"{it['score']} / 5" if it["score"] is not None else "N/A"
            lines.append(f"- **Score**: {score_text}")
            lines.append(f"- **Status**: {it['status']}")
            lines.append(f"- **Reason**: {it['reason'] or 'N/A'}")
            if it.get("error"):
                lines.append(f"- **Error**: {it['error']}")
            lines.append(f"- **Weighted Score**: {w_score}")
            lines.append(f"- **Quintile**: {q_label_str}")
            lines.append(f"- **Item Title**: {it['item_title']}")
            lines.append(f"- **Review**:\n  > {it['review']}")
            lines.append(f"- **Explanation**:\n  > {it['explanation']}\n")

        lines.append("---\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def evaluate_single_metric(
    metric_id: str,
    users_data: List[Dict[str, Any]],
    instance_template: str,
    stats: Dict[str, Any],
    args: Any,
    now_str: str,
) -> Tuple[str, str]:
    """
    Evaluates all users on a single metric, calculates averages and quintile breakdowns,
    and saves separate .json and .md files, including reasons and failures.
    Returns (json_path, md_path).
    """
    metric_info = METRICS_REGISTRY[metric_id]
    metric_display_name = metric_info["display_name"]
    system_prompt = load_text(args.system_prompt) if args.system_prompt else load_metric_prompt(metric_info)

    is_shuffled_data = stats.get("is_shuffled", False) or any(it.get("is_shuffled") for u in users_data for it in u.get("items", []))
    shuffled_tag = "_shuffled" if is_shuffled_data else ""

    clean_model_name = args.model.replace(":", "_").replace("/", "_")
    domain_pair = stats.get("domain_pair", "Cloth-Elec")
    json_filename = f"geval_{domain_pair}_{metric_id}_{clean_model_name}{shuffled_tag}_{now_str}.json"
    md_filename = f"geval_{domain_pair}_{metric_id}_{clean_model_name}{shuffled_tag}_{now_str}.md"
    json_output_path = os.path.join(args.output_dir, json_filename)
    md_output_path = os.path.join(args.output_dir, md_filename)

    total_valid_items = sum(len(u["items"]) for u in users_data)
    print(f"\n=================================================================")
    print(f"  AVVIO VALUTAZIONE METRICA: {metric_display_name.upper()} ({metric_id})")
    print(f"  Utenti validi: {len(users_data)} | Item validi da valutare: {total_valid_items}")
    print(f"  Item ultra-short filtrati (< {stats['min_words_filter']}w): {stats['filtered_ultrashort_items']}")
    print("  Output: JSON reason + alignment_score (1-5 or null); reasons saved")
    print(f"  Output JSON: {json_output_path}")
    print(f"  Output MD:   {md_output_path}")
    print(f"=================================================================\n")

    evaluated_users: List[Dict[str, Any]] = []
    global_scores: List[int] = []
    global_weighted: List[float] = []

    for u_idx, u in enumerate(users_data, 1):
        user_id = u["user_id"]
        print(f"[{metric_display_name}] User {u_idx}/{len(users_data)}: {user_id} ({len(u['items'])} items)")
        user_eval_items = []

        for it_idx, it in enumerate(u["items"], 1):
            instance_prompt = format_instance_prompt(
                template=instance_template,
                user_review=it["user_review"],
                explanation=it["explanation"],
                metric_name=metric_display_name,
                item_title=it["item_title"],
            )

            eval_result = call_ollama_judge(
                system_prompt=system_prompt,
                instance_prompt=instance_prompt,
                model=args.model,
                temperature=args.temperature,
                seed=args.seed,
                ollama_url=args.ollama_url,
                max_tokens=args.max_tokens,
                num_ctx=args.num_ctx,
                max_retries=args.max_retries,
            )

            if eval_result.get("error"):
                print(f"    Item #{it_idx} ({it['item_id']}) -> ERRORE: {eval_result['error']}")

            discrete = eval_result["discrete_score"]
            weighted = eval_result["weighted_score"]
            reasoning = eval_result.get("reason", "")

            print(
                f"    Item #{it_idx} ({it['item_id']} | {it['quintile']} {it['quintile_label']}) -> "
                f"Score: {discrete if discrete is not None else 'N/A'} | Status: {eval_result['status']}"
            )
            if reasoning:
                print(f"      [Reasoning]: {reasoning}")

            if discrete is not None:
                global_scores.append(discrete)
            if weighted is not None:
                global_weighted.append(weighted)

            # Preserve valid nulls and technical failures as distinct outcomes.
            eval_item_entry = {
                "item_id": it["item_id"],
                "item_title": it["item_title"],
                "review": it["user_review"],
                "review_word_count": it["review_word_count"],
                "quintile": it["quintile"],
                "quintile_label": it["quintile_label"],
                "explanation": it["explanation"],
                "score": discrete,
                "weighted_score": weighted,
                "weighted_score_status": eval_result["weighted_score_status"],
                "reason": reasoning,
                "status": eval_result["status"],
                "error": eval_result["error"],
                "response_text": eval_result["response_text"],
                "done_reason": eval_result["done_reason"],
                "attempts": eval_result["attempts"],
            }
            if it.get("is_shuffled"):
                eval_item_entry["is_shuffled"] = True
            if it.get("original_llm_explanation"):
                eval_item_entry["original_llm_explanation"] = it["original_llm_explanation"]
            if it.get("shuffled_from_item_id"):
                eval_item_entry["shuffled_from_item_id"] = it["shuffled_from_item_id"]
            if it.get("shuffled_from_item_title"):
                eval_item_entry["shuffled_from_item_title"] = it["shuffled_from_item_title"]
            if it.get("shuffled_from_user_id"):
                eval_item_entry["shuffled_from_user_id"] = it["shuffled_from_user_id"]
            user_eval_items.append(eval_item_entry)

        u_valid_scores = [x["score"] for x in user_eval_items if x["score"] is not None]
        u_valid_weighted = [x["weighted_score"] for x in user_eval_items if x["weighted_score"] is not None]

        u_mean_score = round(sum(u_valid_scores) / len(u_valid_scores), 4) if u_valid_scores else None
        u_mean_weighted = round(sum(u_valid_weighted) / len(u_valid_weighted), 4) if u_valid_weighted else None

        print(f"  -> Media Utente: Score={u_mean_score} | Weighted={u_mean_weighted}\n")

        evaluated_users.append({
            "user_id": user_id,
            "user_mean_score": u_mean_score,
            "user_mean_weighted_score": u_mean_weighted,
            "items_count": len(user_eval_items),
            "judgment_counts": judgment_counts(user_eval_items),
            "items": user_eval_items,
        })

    # Medie globali per questa metrica
    global_mean_score = round(sum(global_scores) / len(global_scores), 4) if global_scores else None
    global_mean_weighted = round(sum(global_weighted) / len(global_weighted), 4) if global_weighted else None

    # Calcolo Stratificazione per Quintili (Q1 - Q5)
    all_evaluated_items = [it for u in evaluated_users for it in u["items"]]
    quintile_schema = stats.get("quintile_schema", DEFAULT_QUINTILE_SCHEMA)
    by_quintile: Dict[str, Dict[str, Any]] = {}

    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        q_info = quintile_schema.get(qid, {})
        q_label = q_info.get("label", "")
        q_range = q_info.get("range_words", "")
        min_w = q_info.get("min_words", 0)
        max_w = q_info.get("max_words")

        q_items = [it for it in all_evaluated_items if it.get("quintile") == qid]
        cnt = len(q_items)
        pct = round(cnt / len(all_evaluated_items) * 100, 2) if all_evaluated_items else 0.0

        q_discrete = [it["score"] for it in q_items if it.get("score") is not None]
        q_weighted = [it["weighted_score"] for it in q_items if it.get("weighted_score") is not None]

        avg_w = round(sum(it["review_word_count"] for it in q_items) / cnt, 1) if cnt > 0 else 0.0
        avg_score = round(sum(q_discrete) / len(q_discrete), 4) if q_discrete else None
        avg_weighted = round(sum(q_weighted) / len(q_weighted), 4) if q_weighted else None
        score_dist = {str(s): q_discrete.count(s) for s in range(1, 6)}

        by_quintile[qid] = {
            "label": q_label,
            "range_words": q_range,
            "min_words": min_w,
            "max_words": max_w,
            "count": cnt,
            "percentage_of_valid": pct,
            "avg_words": avg_w,
            "avg_score": avg_score,
            "avg_weighted_score": avg_weighted,
            "score_distribution": score_dist,
            **judgment_counts(q_items),
        }

    quintile_metrics = {
        "min_words_filter": stats["min_words_filter"],
        "total_items_in_source": stats["total_source_items"],
        "valid_items_evaluated": len(global_scores),
        "judgment_counts": judgment_counts(all_evaluated_items),
        "filtered_ultrashort_items": stats["filtered_ultrashort_items"],
        "macro_avg_valid_score": global_mean_score,
        "macro_avg_valid_weighted_score": global_mean_weighted,
        "by_quintile": by_quintile,
    }

    summary = {
        "global_mean_score": global_mean_score,
        "global_mean_weighted_score": global_mean_weighted,
        "total_users_evaluated": len(evaluated_users),
        "total_items_evaluated": len(global_scores),
        "total_items_in_source": stats["total_source_items"],
        "filtered_ultrashort_items": stats["filtered_ultrashort_items"],
        "skipped_missing_input_items": stats.get("skipped_missing_input_items", 0),
        "min_review_words_filter": stats["min_words_filter"],
        "quintile_metrics": quintile_metrics,
        "metric_id": metric_id,
        "metric_name": metric_display_name,
        "model": args.model,
        "temperature": args.temperature,
        "timestamp": datetime.datetime.now().isoformat(),
        "source_results": args.results_file,
        "is_shuffled": stats.get("is_shuffled", False),
        "shuffle_info": stats.get("shuffle_info", {}),
        "judgment_counts": judgment_counts(all_evaluated_items),
        "protocol_metadata": {
            "schema_version": 2,
            "judge_protocol": "alignment_json_v1",
            "system_prompt_file": args.system_prompt or metric_info["prompt_file"],
            "instance_prompt_file": args.instance_prompt,
            "system_prompt": system_prompt,
            "instance_template": instance_template,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "instance_prompt_sha256": hashlib.sha256(instance_template.encode("utf-8")).hexdigest(),
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "num_ctx": args.num_ctx,
            "max_retries": args.max_retries,
            "think": False,
            "weighted_scoring": "not_computed",
            "aggregation": "item_mean_over_numeric_scores",
        },
    }

    save_results_json(json_output_path, summary, evaluated_users)
    save_results_markdown(md_output_path, summary, evaluated_users)

    print(f"--- RIEPILOGO {metric_display_name.upper()} ---")
    print(f"Media Globale Score:        {global_mean_score}/5" if global_mean_score is not None else "N/A")
    print(f"Media Globale Score Pesato: {global_mean_weighted}/5.0" if global_mean_weighted is not None else "N/A")
    print("Stratificazione Quintili:")
    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        q_res = by_quintile[qid]
        print(
            f"  [{qid} {q_res['label']:<8} ({q_res['range_words']})]: "
            f"Items={q_res['count']} ({q_res['percentage_of_valid']}%) | "
            f"Score={q_res['avg_score']} | Weighted={q_res['avg_weighted_score']}"
        )
    print(f"\nFile salvati con successo in:\n  - {json_output_path}\n  - {md_output_path}\n")

    return json_output_path, md_output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate review alignment with JSON reason and nullable score on Ollama")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="LLM judge model (default: llama3.1:8b)")
    parser.add_argument(
        "--metrics",
        "--metric",
        nargs="+",
        default=["alignment"],
        help="Metric to evaluate (default: alignment)",
    )
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override the system rubric file (default: active metric prompt)")
    parser.add_argument("--num_ctx", type=int, default=8192,
                        help="Ollama judge context size (default: 8192)")
    parser.add_argument("--max_retries", type=int, default=1,
                        help="Retries for technical failures; valid nulls are not retried")
    parser.add_argument("--top_logprobs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=os.path.join(PROMPTS_DIR, "TEST_PROMPT_INSTANCE.txt"),
        help="Path to instance prompt template",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/Cloth-Elec/llama3.1_8b_v2/Cloth-Elec_llama3.1_8b_v2_20260930_164711.json",
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--num_users",
        "--max_users",
        type=int,
        default=None,
        help="Number of users to evaluate (default: None, processes all users in results file)",
    )
    parser.add_argument(
        "--min_words",
        "--min_review_words",
        type=int,
        default=5,
        help="Minimum word count filter for user reviews (default: 5, excludes < 5 words)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens for JSON reason and score (default: 256)",
    )
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama URL")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save JSON and MD output files (default: results_judge/<domain_pair>)",
    )
    args = parser.parse_args()
    if args.max_tokens < 1 or args.num_ctx < 1 or args.max_retries not in (0, 1):
        parser.error("max_tokens/num_ctx must be positive; max_retries must be 0 or 1")
    if args.top_logprobs is not None:
        print("Note: --top_logprobs is ignored; JSON judge scoring is discrete.")

    # Normalizzazione percorsi file rispetto a REPO_ROOT se necessario
    if not os.path.exists(args.results_file):
        candidate_res = os.path.join(REPO_ROOT, args.results_file)
        if os.path.exists(candidate_res):
            args.results_file = candidate_res

    # Risoluzione metriche selezionate
    all_metric_keys = list(METRICS_REGISTRY.keys())
    if not args.metrics or "all" in [m.lower() for m in args.metrics]:
        selected_metrics = all_metric_keys
    else:
        selected_metrics = []
        for m in args.metrics:
            norm_m = m.lower().strip().replace("-", "_").replace(" ", "_")
            if norm_m in METRICS_REGISTRY:
                if norm_m not in selected_metrics:
                    selected_metrics.append(norm_m)
            else:
                print(f"[ERRORE] Metrica '{m}' non valida. Scegli tra: {all_metric_keys} o 'all'.")
                sys.exit(1)

    instance_template = load_text(args.instance_prompt)
    users_data, stats = extract_users_from_results(
        args.results_file,
        max_users=args.num_users,
        min_words=args.min_words,
    )

    if not users_data:
        print("Nessun utente valido estratto. Verifica i filtri o il file di input.")
        sys.exit(1)

    # Risoluzione cartella di output dinamica in base al domain_pair
    if args.output_dir is None:
        domain_pair = stats.get("domain_pair", "Cloth-Elec")
        args.output_dir = os.path.join(REPO_ROOT, "results_judge", domain_pair)
    elif not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(REPO_ROOT, args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=================================================================")
    print("           REVIEW ALIGNMENT JUDGE (JSON SCORE OR NULL)            ")
    print("=================================================================")
    print(f"Model:                {args.model}")
    print(f"Selected Metrics:     {selected_metrics} ({len(selected_metrics)} in totale)")
    print(f"Results Source File:  {args.results_file}")
    print(f"Total Users:          {stats['total_source_users']} (validi da valutare: {len(users_data)})")
    print(
        f"Total Items:          {stats['total_source_items']} "
        f"(validi: {stats['valid_items_count']}, filtrati ultra-short < {args.min_words}w: {stats['filtered_ultrashort_items']})"
    )
    print(f"Max Tokens per Item:  {args.max_tokens} (JSON reason + score)")
    print(f"Output Directory:     {args.output_dir}")
    print("=================================================================\n")

    saved_reports = []
    total_successful = 0
    for metric_id in selected_metrics:
        json_path, md_path = evaluate_single_metric(
            metric_id=metric_id,
            users_data=users_data,
            instance_template=instance_template,
            stats=stats,
            args=args,
            now_str=now_str,
        )
        saved_reports.append((metric_id, json_path, md_path))
        with open(json_path, "r", encoding="utf-8") as report_file:
            counts = json.load(report_file)["judgment_counts"]
        total_successful += counts["scored_count"] + counts["insufficient_evidence_count"]

    print("\n=================================================================")
    print("              TUTTE LE VALUTAZIONI SONO COMPLETATE!              ")
    print("=================================================================")
    for m_id, j_path, m_path in saved_reports:
        display_name = METRICS_REGISTRY[m_id]["display_name"]
        print(f"• {display_name}:")
        print(f"    JSON: {j_path}")
        print(f"    MD:   {m_path}")
    print("=================================================================")

    if not total_successful:
        sys.exit(1)


if __name__ == "__main__":
    main()
