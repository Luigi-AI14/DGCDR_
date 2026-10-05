"""
Markdown exporter for DGCDR LLM explanation and validation results.
Converts structured validation JSON reports into human-readable Markdown (.md) documents,
and provides utility functions to organize results by domain pair.
"""

import argparse
import datetime
import glob
import json
import logging
import os
import re
import shutil
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MarkdownExporter")


def _format_score(val: Any) -> str:
    """Format floating point score to 4 decimals or return string/N/A."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{float(val):.4f}"
    return str(val)


def sanitize_model_name(model_name: str) -> str:
    """Sanitize LLM model name for safe filesystem naming across OS platforms."""
    cleaned = model_name.replace(":", "_").replace("/", "_")
    return re.sub(r'[^a-zA-Z0-9._-]', '_', cleaned)


def _format_blockquote(text: str) -> str:
    """Formats a text block as a Markdown blockquote, ensuring each line has '>'."""
    if not text:
        return "> *(No text provided)*"
    lines = text.strip().split("\n")
    return "\n".join(f"> {line}" for line in lines)


def json_to_markdown(data: Dict[str, Any]) -> str:
    """
    Converts a DGCDR validation report dictionary into a comprehensive,
    cleanly formatted Markdown report.
    """
    domain_pair = data.get("domain_pair", "Unknown")
    source_domain = data.get("source_domain", "")
    target_domain = data.get("target_domain", "")
    model = data.get("model", "Unknown")
    prompt_version = data.get("prompt_version")
    sbert_model = data.get("sbert_model", "N/A")
    timestamp = data.get("timestamp", "N/A")
    seed = data.get("seed", "N/A")
    temp = data.get("temperature", "N/A")
    threshold = data.get("rating_threshold", "N/A")
    min_review_words = data.get("min_review_words", "N/A")
    users = data.get("users", [])
    num_users = data.get("num_users", len(users))
    total_held_out = data.get("total_held_out_items_evaluated", sum(len(u.get("items", [])) for u in users))
    prompts_dir = data.get("prompts_dir", "N/A")

    global_avg = data.get("global_averages", {})
    quintile_metrics = data.get("quintile_metrics", {})

    has_sbert = "macro_avg_sbert_similarity" in global_avg or any(
        "avg_sbert_similarity" in u.get("user_averages", {}) for u in users
    )
    has_bertscore = "macro_avg_bertscore_r" in global_avg or any(
        "avg_bertscore_r" in u.get("user_averages", {}) for u in users
    )
    bertscore_model = data.get("bertscore_model", "roberta-large")

    md = []
    # Title & Metadata
    md.append(f"# DGCDR Validation Report: {domain_pair}")
    md.append(f"**Model:** `{model}` | **Timestamp:** `{timestamp}` | **Users:** `{num_users}` | **Items Evaluated:** `{total_held_out}`\n")

    # Run Configuration Table
    md.append("## 1. Run Configuration")
    md.append("| Parameter | Value |")
    md.append("|:---|:---|")
    md.append(f"| **Domain Pair** | `{domain_pair}` ({source_domain} &rarr; {target_domain}) |")
    md.append(f"| **LLM Model** | `{model}` |")
    if prompt_version:
        md.append(f"| **Prompt Version** | `{prompt_version}` |")
    md.append(f"| **Sentence-BERT Model** | `{sbert_model}` |")
    rescale_baseline = data.get("bertscore_rescale_with_baseline", True)
    if has_bertscore and bertscore_model:
        rescale_str = " (Rescaled with Baseline)" if rescale_baseline else " (Raw)"
        md.append(f"| **BERTScore Model** | `{bertscore_model}`{rescale_str} |")
    md.append(f"| **Random Seed** | `{seed}` |")
    md.append(f"| **Sampling Temperature** | `{temp}` |")
    md.append(f"| **Rating Threshold** | `&ge; {threshold}` |")
    md.append(f"| **Quintile Min Words Filter** | `&ge; {min_review_words} words` |")
    md.append(f"| **Prompts Directory** | `{prompts_dir}` |")
    md.append("")

    # Global Performance Metrics
    md.append("## 2. Global Evaluation Metrics (Macro Averages)")
    md.append("| Metric | Score | Description |")
    md.append("|:---|:---:|:---|")
    md.append(f"| **BLEU** | `{_format_score(global_avg.get('macro_avg_bleu'))}` | Syntactic n-gram precision with smoothing |")
    md.append(f"| **ROUGE-1 (F1)** | `{_format_score(global_avg.get('macro_avg_rouge1_f1'))}` | Unigram lexical overlap |")
    md.append(f"| **ROUGE-2 (F1)** | `{_format_score(global_avg.get('macro_avg_rouge2_f1'))}` | Bigram lexical overlap |")
    md.append(f"| **ROUGE-L (F1)** | `{_format_score(global_avg.get('macro_avg_rougeL_f1'))}` | Longest common subsequence |")
    if has_sbert:
        md.append(f"| **SBERT Similarity** | `{_format_score(global_avg.get('macro_avg_sbert_similarity'))}` | Semantic cosine similarity (Sentence-BERT) |")
    if has_bertscore:
        rescale_note = " (rescaled with baseline)" if rescale_baseline else ""
        md.append(f"| **BERTScore (Recall)** | `{_format_score(global_avg.get('macro_avg_bertscore_r'))}` | Semantic token coverage of user review{rescale_note} |")
        md.append(f"| **BERTScore (F1)** | `{_format_score(global_avg.get('macro_avg_bertscore_f1'))}` | Harmonic mean of token-level semantic match{rescale_note} |")
        md.append(f"| **BERTScore (Precision)** | `{_format_score(global_avg.get('macro_avg_bertscore_p'))}` | Grounding of explanation tokens in user review{rescale_note} |")
    md.append("")

    # Stratified Quintile Metrics (if present)
    if quintile_metrics and "by_quintile" in quintile_metrics:
        md.append(f"## 3. Stratified Metrics by Review Length Quintiles (Min Words &ge; {min_review_words})")
        
        q_cols = ["Quintile", "Label", "Word Range", "Count", "% Valid", "Avg Words", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        if has_sbert:
            q_cols.append("SBERT Sim")
        if has_bertscore:
            q_cols.extend(["BERT-R", "BERT-F1"])
        
        md.append("| " + " | ".join(q_cols) + " |")
        md.append("|" + "|".join([":---" if i < 3 else ":---:" for i in range(len(q_cols))]) + "|")

        by_q = quintile_metrics.get("by_quintile", {})
        for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            if qid in by_q:
                q = by_q[qid]
                row = [
                    f"**{qid}**",
                    q.get("label", ""),
                    q.get("range_words", ""),
                    str(q.get("count", 0)),
                    f"{q.get('percentage_of_valid', 0.0):.1f}%",
                    f"{q.get('avg_words', 0.0):.1f}",
                    _format_score(q.get("avg_bleu")),
                    _format_score(q.get("avg_rouge1_f1")),
                    _format_score(q.get("avg_rouge2_f1")),
                    _format_score(q.get("avg_rougeL_f1")),
                ]
                if has_sbert:
                    row.append(_format_score(q.get("avg_sbert_similarity")))
                if has_bertscore:
                    row.append(_format_score(q.get("avg_bertscore_r")))
                    row.append(_format_score(q.get("avg_bertscore_f1")))
                md.append("| " + " | ".join(row) + " |")

        # Summary rows
        valid_items_cnt = quintile_metrics.get("valid_items_evaluated", total_held_out)
        filtered_cnt = quintile_metrics.get("filtered_ultrashort_items", 0)
        macro_valid = quintile_metrics.get("macro_avg_valid_items", {})
        if macro_valid:
            valid_row = [
                f"**Overall Valid (&ge; {min_review_words}w)**",
                "-",
                "-",
                str(valid_items_cnt),
                "100.0%",
                "-",
                _format_score(macro_valid.get("macro_avg_bleu")),
                _format_score(macro_valid.get("macro_avg_rouge1_f1")),
                _format_score(macro_valid.get("macro_avg_rouge2_f1")),
                _format_score(macro_valid.get("macro_avg_rougeL_f1")),
            ]
            if has_sbert:
                valid_row.append(_format_score(macro_valid.get("macro_avg_sbert_similarity")))
            if has_bertscore:
                valid_row.append(_format_score(macro_valid.get("macro_avg_bertscore_r")))
                valid_row.append(_format_score(macro_valid.get("macro_avg_bertscore_f1")))
            md.append("| " + " | ".join(valid_row) + " |")

        md.append("")
        if filtered_cnt > 0:
            md.append(f"> **Note:** {filtered_cnt} ultra-short item review(s) (< {min_review_words} words) were excluded from quintile stratification to prevent synthetic skew.\n")

    # User-Level Summary Table
    md.append("## 4. Per-User Summary")
    user_cols = ["#", "User ID", "Items", "Avg BLEU", "Avg ROUGE-1", "Avg ROUGE-2", "Avg ROUGE-L"]
    if has_sbert:
        user_cols.append("Avg SBERT Sim")
    if has_bertscore:
        user_cols.extend(["Avg BERT-R", "Avg BERT-F1"])
    md.append("| " + " | ".join(user_cols) + " |")
    md.append("|" + "|".join([":---:" if i == 0 or i == 2 else (":---" if i == 1 else ":---:") for i in range(len(user_cols))]) + "|")

    for idx, u in enumerate(users, 1):
        uid = u.get("user_id", "Unknown")
        u_avgs = u.get("user_averages", {})
        it_count = len(u.get("items", []))
        row = [
            str(idx),
            f"`{uid}`",
            str(it_count),
            _format_score(u_avgs.get("avg_bleu")),
            _format_score(u_avgs.get("avg_rouge1_f1")),
            _format_score(u_avgs.get("avg_rouge2_f1")),
            _format_score(u_avgs.get("avg_rougeL_f1")),
        ]
        if has_sbert:
            row.append(_format_score(u_avgs.get("avg_sbert_similarity")))
        if has_bertscore:
            row.append(_format_score(u_avgs.get("avg_bertscore_r")))
            row.append(_format_score(u_avgs.get("avg_bertscore_f1")))
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Detailed Per-User and Item Evaluations
    md.append("## 5. Detailed User & Item Evaluations")
    for idx, u in enumerate(users, 1):
        uid = u.get("user_id", "Unknown")
        prompt_f = u.get("prompt_file", "")
        u_avgs = u.get("user_averages", {})
        
        md.append(f"### User {idx}/{len(users)}: `{uid}`")
        if prompt_f:
            md.append(f"- **Prompt File:** `{prompt_f}`")
        
        avg_line = (
            f"- **User Averages:** BLEU: `{_format_score(u_avgs.get('avg_bleu'))}` | "
            f"ROUGE-1: `{_format_score(u_avgs.get('avg_rouge1_f1'))}` | "
            f"ROUGE-2: `{_format_score(u_avgs.get('avg_rouge2_f1'))}` | "
            f"ROUGE-L: `{_format_score(u_avgs.get('avg_rougeL_f1'))}`"
        )
        if has_sbert:
            avg_line += f" | SBERT: `{_format_score(u_avgs.get('avg_sbert_similarity'))}`"
        if has_bertscore:
            avg_line += f" | BERT-R: `{_format_score(u_avgs.get('avg_bertscore_r'))}` | BERT-F1: `{_format_score(u_avgs.get('avg_bertscore_f1'))}`"
        md.append(avg_line)
        md.append("")

        for item_idx, it in enumerate(u.get("items", []), 1):
            iid = it.get("id_item", "Unknown")
            title = it.get("item_title", "Untitled Item").strip()
            rev_title = it.get("user_review_title", "").strip()
            rev_text = it.get("user_review_text", "").strip()
            explanation = it.get("llm_explanation", "").strip()
            w_count = it.get("review_word_count", "N/A")
            qid = it.get("quintile") or "N/A"
            qlabel = it.get("quintile_label") or "N/A"
            m = it.get("metrics", {})

            md.append(f"#### Item {item_idx}: {title}")
            md.append(f"- **Item ID:** `{iid}` | **Quintile:** `{qid} ({qlabel})` | **Review Words:** `{w_count}`")
            
            it_metric_line = (
                f"- **Metrics:** BLEU: `{_format_score(m.get('bleu'))}` | "
                f"ROUGE-1 (F1): `{_format_score(m.get('rouge1_f1'))}` | "
                f"ROUGE-2 (F1): `{_format_score(m.get('rouge2_f1'))}` | "
                f"ROUGE-L (F1): `{_format_score(m.get('rougeL_f1'))}`"
            )
            if has_sbert and "sbert_similarity" in m:
                it_metric_line += f" | SBERT Sim: `{_format_score(m.get('sbert_similarity'))}`"
            if has_bertscore and "bertscore_r" in m:
                it_metric_line += f" | BERT-R: `{_format_score(m.get('bertscore_r'))}` | BERT-F1: `{_format_score(m.get('bertscore_f1'))}`"
            md.append(it_metric_line)
            md.append("")

            # Ground-truth review block
            md.append("**Actual User Review (Ground Truth):**")
            if rev_title:
                full_rev = f"**{rev_title}**\n\n{rev_text}"
            else:
                full_rev = rev_text
            md.append(_format_blockquote(full_rev))
            md.append("")

            # LLM explanation block
            md.append("**Generated LLM Explanation:**")
            md.append(_format_blockquote(explanation))
            md.append("")
            md.append("---")
            md.append("")

    return "\n".join(md)


def generate_markdown_report(report_data: Dict[str, Any], md_filepath: str) -> str:
    """Generates and writes a Markdown report from report data dictionary."""
    md_content = json_to_markdown(report_data)
    os.makedirs(os.path.dirname(os.path.abspath(md_filepath)), exist_ok=True)
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    return md_filepath


def export_json_to_markdown(json_path: str, md_path: Optional[str] = None) -> str:
    """Reads a JSON validation file and exports it to a companion Markdown file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if md_path is None:
        base, _ = os.path.splitext(json_path)
        md_path = f"{base}.md"

    return generate_markdown_report(data, md_path)


def reorganize_and_convert_legacy_results(
    results_dir: str = "results",
    move_files: bool = False,
) -> List[Dict[str, str]]:
    """
    Scans legacy validation JSON files in results_dir (e.g. results/validation_Cloth-Elec_50users_20260926_120411.json),
    reorganizes them into domain subfolders (e.g. results/Cloth-Elec/),
    renames them to: <source>-<target>_<model>_<timestamp>.json,
    and generates the corresponding Markdown (.md) reports.
    """
    migrated = []
    # Find all direct .json files in results_dir root
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    for src_json in json_files:
        filename = os.path.basename(src_json)
        # Skip non-validation files if any
        if filename.startswith("compact_"):
            continue

        try:
            with open(src_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {src_json}: {e}")
            continue

        domain_pair = data.get("domain_pair", "Unknown")
        model = data.get("model", "unknown_model")
        clean_model = sanitize_model_name(model)

        # Extract timestamp
        ts_match = re.search(r'(\d{8}_\d{6})', filename)
        if ts_match:
            timestamp = ts_match.group(1)
        else:
            iso_ts = data.get("timestamp", "")
            try:
                dt = datetime.datetime.fromisoformat(iso_ts)
                timestamp = dt.strftime("%Y%m%d_%H%M%S")
            except Exception:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine target directory: results/<domain_pair>/<clean_model>/
        target_folder = os.path.join(results_dir, domain_pair, clean_model)
        os.makedirs(target_folder, exist_ok=True)

        suffix = "_quintiles" if "_quintiles" in filename else ""
        new_basename = f"{domain_pair}_{clean_model}_{timestamp}{suffix}"
        dest_json = os.path.join(target_folder, f"{new_basename}.json")
        dest_md = os.path.join(target_folder, f"{new_basename}.md")

        if move_files:
            shutil.move(src_json, dest_json)
        else:
            shutil.copy2(src_json, dest_json)

        # Generate markdown companion report
        generate_markdown_report(data, dest_md)

        migrated.append({
            "original": src_json,
            "new_json": dest_json,
            "new_md": dest_md,
        })
        logger.info(f"Processed: {filename} -> {domain_pair}/{clean_model}/{new_basename}.json & .md")

    return migrated


def organize_domain_folders_by_model(results_dir: str = "results") -> int:
    """
    Checks domain folders (e.g. results/Cloth-Elec/) and moves any direct
    .json and .md files into model subfolders (e.g. results/Cloth-Elec/llama3.1_8b/).
    """
    moved_count = 0
    if not os.path.exists(results_dir):
        return 0

    for domain_entry in os.listdir(results_dir):
        domain_path = os.path.join(results_dir, domain_entry)
        if not os.path.isdir(domain_path) or domain_entry.startswith("_"):
            continue

        # Look for .json files directly in the domain folder
        json_files = glob.glob(os.path.join(domain_path, "*.json"))
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                model = data.get("model", "unknown_model")
                clean_model = sanitize_model_name(model)
                target_dir = os.path.join(domain_path, clean_model)
                os.makedirs(target_dir, exist_ok=True)

                base_name = os.path.basename(jf)
                target_json = os.path.join(target_dir, base_name)
                shutil.move(jf, target_json)

                # Move corresponding .md if it exists, or generate it
                md_path = os.path.splitext(jf)[0] + ".md"
                target_md = os.path.join(target_dir, os.path.basename(md_path))
                if os.path.exists(md_path):
                    shutil.move(md_path, target_md)
                else:
                    generate_markdown_report(data, target_md)

                moved_count += 1
                logger.info(f"Organized into model folder: {domain_entry}/{clean_model}/{base_name}")
            except Exception as e:
                logger.error(f"Error organizing {jf}: {e}")

    return moved_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert validation JSON reports to Markdown and/or reorganize by domain."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to results directory (default: results).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Convert a specific JSON file to Markdown.",
    )
    parser.add_argument(
        "--reorganize",
        action="store_true",
        help="Reorganize root results/*.json into domain subfolders with new naming and generate .md files.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying when reorganizing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.file:
        md_out = export_json_to_markdown(args.file)
        print(f"Generated Markdown: {md_out}")
    elif args.reorganize:
        res = reorganize_and_convert_legacy_results(results_dir=args.results_dir, move_files=args.move)
        print(f"Reorganized and converted {len(res)} file(s).")
    else:
        # Default action: scan all json files in results (including subdirectories) and generate/update .md
        all_jsons = glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True)
        count = 0
        for jf in all_jsons:
            if not os.path.basename(jf).startswith("compact_"):
                export_json_to_markdown(jf)
                count += 1
        print(f"Generated Markdown for {count} JSON file(s) in {args.results_dir}.")
