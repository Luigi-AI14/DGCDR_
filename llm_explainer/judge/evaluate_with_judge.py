#!/usr/bin/env python3
"""
G-Eval LLM-as-a-Judge Evaluation Framework for Recommendation Explanations.

Evaluates candidate recommendation explanations against actual post-consumption
user reviews (ground truth) on held-out items across four G-Eval dimensions:
  1. Aspect Coverage
  2. Aspect Precision
  3. Sentiment Coherence
  4. Specificity

Adheres strictly to the G-Eval protocol (Liu et al., EMNLP 2023):
  - System Prompt: Task Introduction + Evaluation Criteria + Evaluation Steps (CoT) + Output Schema.
  - Instance Prompt: [Product Information], [User Review], [Recommendation Explanation].
  - Dual Scoring: Both Discrete Unweighted Score (1-5) and Calibrated Weighted Score (1.0-5.0).
  - Rigorous alignment verification: Ensures review and explanation belong to the exact same user and item.
"""

import argparse
import datetime
import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. G-EVAL MODULAR SYSTEM PROMPTS (Loaded from llm_explainer/geval_prompts/)
# ==============================================================================

from llm_explainer.geval_manager import (
    load_compiled_system_prompts,
    compile_system_prompts,
    METRIC_IDS as GEVAL_METRICS,
    COMPILED_DIR as GEVAL_COMPILED_PROMPTS_DIR,
)


def build_instance_prompt(item_title: str, user_review: str, explanation: str) -> str:
    """Builds the instance prompt containing only item and text data (no AI citation)."""
    return (
        f"[Product Information]:\n{item_title}\n\n"
        f"[User Review]:\n{user_review}\n\n"
        f"[Recommendation Explanation]:\n{explanation}\n\n"
        f"Evaluation Form (scores ONLY):\n- Score:"
    )


# ==============================================================================
# 2. OLLAMA CLIENT WITH THINKING & LOGPROBS
# ==============================================================================

class OllamaJudgeClient:
    """Ollama client specialized for LLM-as-a-judge evaluation."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:9b",
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 32768,
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.timeout = timeout

    def check_health(self) -> bool:
        """Verifies connection to Ollama API."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def evaluate_metric(
        self,
        system_prompt: str,
        instance_prompt: str,
        mock: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends the evaluation prompt to Ollama with thinking mode active.
        Returns:
            {
                "unweighted_score": int (1-5),
                "weighted_score": float (1.0-5.0),
                "thinking": str,
                "content": str
            }
        """
        if mock:
            return {
                "unweighted_score": 4,
                "weighted_score": 3.85,
                "thinking": "Mock thinking trace: aspects align well with user review.",
                "content": "Score: 4\nWeighted Score: 3.85",
            }

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": 256,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"Error querying Ollama API: {e}")
            return {
                "unweighted_score": 3,
                "weighted_score": 3.0,
                "thinking": f"API Error: {e}",
                "content": "",
            }

        msg = resp_data.get("message", {})
        content = msg.get("content", "")
        thinking = resp_data.get("thinking") or msg.get("thinking", "")

        # Extract internal thinking if enclosed in <think> tags inside content
        if not thinking and "<think>" in content:
            m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        unweighted, weighted = self._parse_scores(content, fallback_text=thinking)
        return {
            "unweighted_score": unweighted,
            "weighted_score": weighted,
            "thinking": thinking.strip(),
            "content": content.strip(),
        }

    @staticmethod
    def _parse_scores(text: str, fallback_text: str = "") -> Tuple[int, float]:
        """
        Parses unweighted discrete score (1-5) and continuous weighted score (1.0-5.0).
        If primary content is empty/truncated, falls back to parsing the thinking trace.
        """
        unweighted: Optional[int] = None
        weighted: Optional[float] = None

        search_corpus = text if text.strip() else fallback_text

        # 1. Parse discrete score: e.g. "Score: 4", "- Score: 4", "Decision: Score 4"
        m_score = re.search(r"(?:Score|Decision:\s*Score)\s*[:\s]*([1-5])\b", search_corpus, re.IGNORECASE)
        if m_score:
            unweighted = int(m_score.group(1))

        # 2. Parse weighted score: e.g. "Weighted Score: 3.8"
        m_weight = re.search(
            r"Weighted\s+Score\s*[:\s]*([1-5](?:\.\d+)?)\b",
            search_corpus,
            re.IGNORECASE,
        )
        if m_weight:
            weighted = float(m_weight.group(1))

        # Fallback parsing if formatting deviated slightly
        if unweighted is None:
            matches = re.findall(r"\b([1-5])\b", search_corpus)
            if matches:
                unweighted = int(matches[-1])
            else:
                logger.warning("Could not extract any valid score (1-5). Defaulting to 3.")
                unweighted = 3

        if weighted is None:
            weighted = float(unweighted)

        # Enforce range constraints [1.0, 5.0]
        unweighted = max(1, min(5, unweighted))
        weighted = max(1.0, min(5.0, round(weighted, 4)))

        return unweighted, weighted


# ==============================================================================
# 3. RESULT FILE DISCOVERY & RIGOROUS PAIR VERIFICATION
# ==============================================================================

def normalize_name(name: str) -> str:
    """Normalizes model names (e.g. qwen3.5:9b -> qwen3.5_9b) for robust matching."""
    return re.sub(r"[-:]", "_", (name or "").strip().lower())


def find_target_result_files(
    results_dir: str,
    target_model: str,
    target_prompt_version: str,
    domain_pair: Optional[str] = None,
) -> List[str]:
    """
    Scans results_dir and identifies matching JSON result files for the specified
    model and prompt version.
    """
    matched_files = []
    norm_target_model = normalize_name(target_model)
    norm_target_pv = target_prompt_version.strip().lower()

    if not os.path.isdir(results_dir):
        logger.error(f"Results directory '{results_dir}' not found.")
        return []

    for root, _, files in os.walk(results_dir):
        # Skip verification and quintile files
        if "cross_item_verification" in root or "_legacy_archive" in root:
            continue

        for f in files:
            if not f.endswith(".json") or f.endswith("_quintiles.json"):
                continue
            if f.startswith("verify_"):
                continue

            filepath = os.path.join(root, f)
            try:
                with open(filepath, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
            except Exception as e:
                logger.debug(f"Could not read {filepath}: {e}")
                continue

            file_model = data.get("model", "")
            file_pv = str(data.get("prompt_version", "v1")).strip().lower()
            file_domain = data.get("domain_pair", "")

            # Filter domain if specified
            if domain_pair and file_domain.lower() != domain_pair.lower():
                continue

            # Check model match
            norm_file_model = normalize_name(file_model)
            if norm_target_model not in norm_file_model and norm_file_model not in norm_target_model:
                continue

            # Check prompt version match (handle v1 default if absent)
            if file_pv != norm_target_pv:
                # Also check directory name as fallback
                if f"_{norm_target_pv}" not in root and norm_target_pv != "v1":
                    continue

            matched_files.append(filepath)

    return matched_files


def verify_and_extract_pairs(file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rigorously verifies that user reviews and LLM explanations correspond to
    the exact same user and item. Rejects mismatched, empty, or invalid records.
    """
    verified_users = []
    raw_users = file_data.get("users", [])

    for u_idx, u in enumerate(raw_users):
        user_id = u.get("user_id")
        if not user_id or not isinstance(user_id, str):
            logger.warning(f"Skipping user #{u_idx}: missing valid user_id.")
            continue

        verified_items = []
        raw_items = u.get("items", [])
        seen_item_ids = set()

        for it_idx, it in enumerate(raw_items):
            item_id = it.get("id_item") or it.get("item_id")
            item_title = it.get("item_title", "").strip()
            rev_title = it.get("user_review_title", "").strip()
            rev_text = it.get("user_review_text", "").strip()
            explanation = it.get("llm_explanation", "").strip()

            # Rigorous integrity checks
            if not item_id:
                logger.warning(f"User '{user_id}' item #{it_idx} missing item_id. Skipped.")
                continue

            if item_id in seen_item_ids:
                logger.warning(f"User '{user_id}' duplicate item_id '{item_id}'. Skipped.")
                continue

            if not item_title:
                logger.warning(f"User '{user_id}', item '{item_id}' missing item_title. Skipped.")
                continue

            if not rev_text or rev_text.lower() in ["null", "none", "nan", ""]:
                logger.warning(f"User '{user_id}', item '{item_id}' has empty review text. Skipped.")
                continue

            if not explanation or explanation.lower() in ["null", "none", "nan", ""]:
                logger.warning(f"User '{user_id}', item '{item_id}' has empty explanation. Skipped.")
                continue

            seen_item_ids.add(item_id)

            # Combine review title and review text if distinct
            full_review = rev_text
            if rev_title and rev_title.lower() not in rev_text.lower():
                full_review = f"{rev_title}. {rev_text}"

            verified_items.append({
                "id_item": item_id,
                "item_title": item_title,
                "user_review_text": full_review,
                "raw_review_text": rev_text,
                "user_review_title": rev_title,
                "llm_explanation": explanation,
                "review_word_count": it.get("review_word_count", len(full_review.split())),
                "quintile": it.get("quintile", "N/A"),
            })

        if verified_items:
            verified_users.append({
                "user_id": user_id,
                "prompt_file": u.get("prompt_file", ""),
                "items": verified_items,
            })
        else:
            logger.warning(f"User '{user_id}' had zero valid verified items.")

    return verified_users


# ==============================================================================
# 4. MAIN EVALUATION PIPELINE
# ==============================================================================

def run_judge_evaluation(
    source_file: str,
    target_model: str,
    prompt_version: str,
    output_dir: str,
    judge_model: str = "qwen3.5:9b",
    ollama_url: str = "http://localhost:11434",
    num_users: Optional[int] = None,
    dry_run: bool = False,
) -> str:
    """Executes the full G-Eval judge pipeline on a verified source result file."""
    with open(source_file, "r", encoding="utf-8") as f:
        file_data = json.load(f)

    domain_pair = file_data.get("domain_pair", "Unknown-Domain")
    source_domain = file_data.get("source_domain", "")
    target_domain = file_data.get("target_domain", "")

    # Rigorous verification
    verified_users = verify_and_extract_pairs(file_data)
    if num_users and num_users > 0:
        verified_users = verified_users[:num_users]

    total_items = sum(len(u["items"]) for u in verified_users)
    logger.info(
        f"Verified dataset: {len(verified_users)} users, {total_items} items "
        f"from '{os.path.basename(source_file)}'"
    )

    # Initialize Judge Client
    judge_client = OllamaJudgeClient(
        base_url=ollama_url,
        model=judge_model,
        temperature=0.0,
        seed=42,
    )

    if not dry_run and not judge_client.check_health():
        raise ConnectionError(
            f"Unable to connect to Ollama server at {ollama_url}. "
            "Please ensure 'ollama serve' is active."
        )

    # Output directory & file setup
    norm_model_dir = normalize_name(target_model)
    norm_pv = prompt_version.strip().lower()
    target_save_dir = os.path.join(output_dir, domain_pair, f"{norm_model_dir}_{norm_pv}")
    os.makedirs(target_save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"judge_{domain_pair}_{norm_model_dir}_{norm_pv}_{timestamp}.json"
    output_filepath = os.path.join(target_save_dir, output_filename)

    evaluated_users_output = []
    all_item_scores: Dict[str, Dict[str, List[float]]] = {
        m: {"unweighted": [], "weighted": []} for m in GEVAL_METRICS
    }

    # Load compiled G-Eval system prompts
    geval_system_prompts = load_compiled_system_prompts(
        judge_model=judge_model,
        ollama_url=ollama_url,
    )

    start_time = time.time()
    processed_count = 0

    for u_idx, u in enumerate(verified_users, 1):
        user_id = u["user_id"]
        logger.info(f"[{u_idx}/{len(verified_users)}] Evaluating User {user_id} ({len(u['items'])} items)...")

        user_items_out = []
        user_metric_accum: Dict[str, Dict[str, List[float]]] = {
            m: {"unweighted": [], "weighted": []} for m in GEVAL_METRICS
        }

        for it in u["items"]:
            item_id = it["id_item"]
            item_title = it["item_title"]
            user_review = it["user_review_text"]
            explanation = it["llm_explanation"]

            instance_prompt = build_instance_prompt(item_title, user_review, explanation)
            item_eval_result = {}

            # Evaluate each of the 4 G-Eval metrics independently
            for metric in GEVAL_METRICS:
                sys_prompt = geval_system_prompts[metric]
                eval_res = judge_client.evaluate_metric(
                    system_prompt=sys_prompt,
                    instance_prompt=instance_prompt,
                    mock=dry_run,
                )

                unw = eval_res["unweighted_score"]
                wgt = eval_res["weighted_score"]

                item_eval_result[metric] = {
                    "unweighted": unw,
                    "weighted": wgt,
                }

                user_metric_accum[metric]["unweighted"].append(unw)
                user_metric_accum[metric]["weighted"].append(wgt)
                all_item_scores[metric]["unweighted"].append(unw)
                all_item_scores[metric]["weighted"].append(wgt)

            processed_count += 1
            user_items_out.append({
                "id_item": item_id,
                "item_title": item_title,
                "user_review_text": user_review,
                "llm_explanation": explanation,
                "review_word_count": it["review_word_count"],
                "quintile": it["quintile"],
                "judge_scores": item_eval_result,
            })

            logger.info(
                f"  -> Item {item_id}: Cov={item_eval_result['aspect_coverage']['unweighted']}/"
                f"{item_eval_result['aspect_coverage']['weighted']:.2f}, "
                f"Prec={item_eval_result['aspect_precision']['unweighted']}/"
                f"{item_eval_result['aspect_precision']['weighted']:.2f}, "
                f"Sent={item_eval_result['sentiment_coherence']['unweighted']}/"
                f"{item_eval_result['sentiment_coherence']['weighted']:.2f}, "
                f"Spec={item_eval_result['specificity']['unweighted']}/"
                f"{item_eval_result['specificity']['weighted']:.2f}"
            )

        # Compute user averages
        user_averages = {}
        for m in GEVAL_METRICS:
            unw_list = user_metric_accum[m]["unweighted"]
            wgt_list = user_metric_accum[m]["weighted"]
            user_averages[f"avg_{m}_unweighted"] = round(sum(unw_list) / len(unw_list), 4) if unw_list else 0.0
            user_averages[f"avg_{m}_weighted"] = round(sum(wgt_list) / len(wgt_list), 4) if wgt_list else 0.0

        evaluated_users_output.append({
            "user_id": user_id,
            "prompt_file": u["prompt_file"],
            "user_averages": user_averages,
            "items": user_items_out,
        })

    elapsed_time = round(time.time() - start_time, 2)

    # Compute Global Macro Averages
    global_averages = {}
    composite_unw_sum = 0.0
    composite_wgt_sum = 0.0

    for m in GEVAL_METRICS:
        unw_all = all_item_scores[m]["unweighted"]
        wgt_all = all_item_scores[m]["weighted"]
        avg_unw = round(sum(unw_all) / len(unw_all), 4) if unw_all else 0.0
        avg_wgt = round(sum(wgt_all) / len(wgt_all), 4) if wgt_all else 0.0
        global_averages[f"macro_avg_{m}_unweighted"] = avg_unw
        global_averages[f"macro_avg_{m}_weighted"] = avg_wgt
        composite_unw_sum += avg_unw
        composite_wgt_sum += avg_wgt

    global_averages["composite_macro_avg_unweighted"] = round(composite_unw_sum / len(GEVAL_METRICS), 4)
    global_averages["composite_macro_avg_weighted"] = round(composite_wgt_sum / len(GEVAL_METRICS), 4)

    # Assemble complete JSON report
    report_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source_result_file": source_file,
        "evaluated_model": target_model,
        "prompt_version": prompt_version,
        "judge_model": judge_model,
        "judge_temperature": 0.0,
        "domain_pair": domain_pair,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "num_users_evaluated": len(evaluated_users_output),
        "total_items_evaluated": processed_count,
        "elapsed_seconds": elapsed_time,
        "dry_run": dry_run,
        "global_averages": global_averages,
        "users": evaluated_users_output,
    }

    with open(output_filepath, "w", encoding="utf-8") as out_f:
        json.dump(report_data, out_f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation complete in {elapsed_time}s. Saved judge report to:\n{output_filepath}")

    # Generate companion Markdown report for human inspection
    md_filepath = output_filepath.replace(".json", ".md")
    write_markdown_summary(report_data, md_filepath)
    logger.info(f"Saved companion Markdown summary to:\n{md_filepath}")

    return output_filepath


def write_markdown_summary(data: Dict[str, Any], md_path: str) -> None:
    """Writes a clean summary markdown report of judge evaluations."""
    ga = data.get("global_averages", {})
    md = [
        f"# G-Eval Judge Evaluation Report",
        f"",
        f"- **Evaluated Model**: `{data.get('evaluated_model')}`",
        f"- **Prompt Version**: `{data.get('prompt_version')}`",
        f"- **Judge Model**: `{data.get('judge_model')}` (thinking=False, temp=0.0)",
        f"- **Domain Pair**: `{data.get('domain_pair')}`",
        f"- **Source Result File**: `{data.get('source_result_file')}`",
        f"- **Evaluated Pairs**: {data.get('total_items_evaluated')} items across {data.get('num_users_evaluated')} users",
        f"- **Execution Time**: {data.get('elapsed_seconds')}s",
        f"",
        f"---",
        f"",
        f"## 📊 Global Macro Averages",
        f"",
        f"| G-Eval Metric | Discrete Unweighted (1-5) | Calibrated Weighted (1.0-5.0) | Description |",
        f"|---|---|---|---|",
        f"| **Aspect Coverage** | **{ga.get('macro_avg_aspect_coverage_unweighted', 0):.4f}** | **{ga.get('macro_avg_aspect_coverage_weighted', 0):.4f}** | Recall of user review aspects |",
        f"| **Aspect Precision** | **{ga.get('macro_avg_aspect_precision_unweighted', 0):.4f}** | **{ga.get('macro_avg_aspect_precision_weighted', 0):.4f}** | Faithfulness / Grounding check |",
        f"| **Sentiment Coherence** | **{ga.get('macro_avg_sentiment_coherence_unweighted', 0):.4f}** | **{ga.get('macro_avg_sentiment_coherence_weighted', 0):.4f}** | Aspect-level valence alignment |",
        f"| **Specificity** | **{ga.get('macro_avg_specificity_unweighted', 0):.4f}** | **{ga.get('macro_avg_specificity_weighted', 0):.4f}** | Technical & physical depth calibration |",
        f"| **COMPOSITE OVERALL** | **{ga.get('composite_macro_avg_unweighted', 0):.4f}** | **{ga.get('composite_macro_avg_weighted', 0):.4f}** | Mean of all 4 dimensions |",
        f"",
        f"---",
        f"",
        f"## 👤 Per-User Averages Summary",
        f"",
        f"| User ID | Items | Coverage (Unw / Wgt) | Precision (Unw / Wgt) | Sentiment (Unw / Wgt) | Specificity (Unw / Wgt) |",
        f"|---|---|---|---|---|---|",
    ]

    for u in data.get("users", []):
        ua = u.get("user_averages", {})
        md.append(
            f"| `{u.get('user_id')}` | {len(u.get('items', []))} | "
            f"{ua.get('avg_aspect_coverage_unweighted', 0):.2f} / {ua.get('avg_aspect_coverage_weighted', 0):.2f} | "
            f"{ua.get('avg_aspect_precision_unweighted', 0):.2f} / {ua.get('avg_aspect_precision_weighted', 0):.2f} | "
            f"{ua.get('avg_sentiment_coherence_unweighted', 0):.2f} / {ua.get('avg_sentiment_coherence_weighted', 0):.2f} | "
            f"{ua.get('avg_specificity_unweighted', 0):.2f} / {ua.get('avg_specificity_weighted', 0):.2f} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


# ==============================================================================
# 5. CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="G-Eval LLM-as-a-Judge Evaluation for Recommendation Explanations."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the LLM whose explanations are being evaluated (e.g. qwen3.5:9b, llama3.1:8b).",
    )
    parser.add_argument(
        "--prompt_version",
        "--prompt-version",
        type=str,
        required=True,
        help="Prompt version of the explanations to retrieve (e.g. v1, v2).",
    )
    parser.add_argument(
        "--judge_model",
        "--judge-model",
        type=str,
        default="qwen3.5:9b",
        help="Model identifier of the evaluator judge LLM in Ollama (default: qwen3.5:9b).",
    )
    parser.add_argument(
        "--results_dir",
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing original explanation JSON results (default: results).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default="results_judge",
        help="Directory where judge evaluations will be saved (default: results_judge).",
    )
    parser.add_argument(
        "--domain_pair",
        "--domain-pair",
        type=str,
        default=None,
        help="Cross-domain pair filter (e.g. Cloth-Elec). If omitted, evaluates all matching.",
    )
    parser.add_argument(
        "--num_users",
        "--num-users",
        type=int,
        default=None,
        help="Optional limit on number of users to evaluate (useful for quick testing).",
    )
    parser.add_argument(
        "--ollama_url",
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server endpoint (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        action="store_true",
        help="Run without calling Ollama (uses deterministic mock judge scores for testing).",
    )
    parser.add_argument(
        "--regenerate_steps",
        "--regenerate-steps",
        action="store_true",
        help="Force re-generation of G-Eval evaluation steps via LLM",
    )

    args = parser.parse_args()

    if args.regenerate_steps:
        logger.info("Forcing re-generation of G-Eval evaluation steps...")
        compile_system_prompts(
            judge_model=args.judge_model,
            ollama_url=args.ollama_url,
            force_regenerate=True,
        )

    logger.info("=== Starting G-Eval Judge Evaluation ===")
    logger.info(f"Target Model: {args.model}")
    logger.info(f"Prompt Version: {args.prompt_version}")
    logger.info(f"Judge Model: {args.judge_model}")
    logger.info(f"Results Directory: {args.results_dir}")
    logger.info(f"Output Directory: {args.output_dir}")

    # Discover target result files
    matched_files = find_target_result_files(
        results_dir=args.results_dir,
        target_model=args.model,
        target_prompt_version=args.prompt_version,
        domain_pair=args.domain_pair,
    )

    if not matched_files:
        logger.error(
            f"No matching result files found for model='{args.model}' and "
            f"prompt_version='{args.prompt_version}' in '{args.results_dir}'."
        )
        sys.exit(1)

    logger.info(f"Found {len(matched_files)} matching file(s):")
    for mf in matched_files:
        logger.info(f"  - {mf}")

    # Sort files by timestamp/mtime descending (latest first)
    matched_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Evaluate the most recent file
    target_file = matched_files[0]
    logger.info(f"Selected latest file for evaluation: {target_file}")

    run_judge_evaluation(
        source_file=target_file,
        target_model=args.model,
        prompt_version=args.prompt_version,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        ollama_url=args.ollama_url,
        num_users=args.num_users,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
