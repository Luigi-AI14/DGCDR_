#!/usr/bin/env python3
"""
Dedicated Shuffle & Random Baseline Script for LLM-as-a-Judge Validation.

Purpose:
Evaluates the discriminative power and item-specificity of the LLM Judge (G-Eval Alignment)
by pairing actual user reviews of Item X with mismatched/randomized recommendation explanations
of Item Y (X != Y), establishing a rigorous negative baseline (lower bound).

If the Judge is valid and sensitive to semantic alignment:
- Congruent Score (Item X Review vs Item X Explanation) >> Shuffled Score (Item X Review vs Item Y Explanation)
- Shuffled score distribution should collapse toward 1 (Hallucination / Contradiction) and 2 (Generic mismatch)
- Contrastive margin (Delta = Congruent - Shuffled) should be strictly positive and statistically significant (p < 0.001)

Features:
1. Exact Derangement Permutation (1-to-1 bijection via Hungarian matching):
   - 'inter_user' (default): Cross-user, cross-item (u_i != u_j and p_i != p_j).
   - 'global': Cross-item across whole dataset (p_i != p_j and i != j).
   - 'intra_user': Permutation within the items of the same user (for multi-item users).
   - 'random_sampling': Random negative drawing with replacement.
2. Full Provenance Tracking:
   - Stores original explanation, source item ID, source item title, and source user ID.
   - Retains 100% schema compatibility with `test_geval_prompts.py`.
3. Integrated Pipeline Support:
   - Can generate the shuffled dataset only (`python shuffle_for_judge.py`).
   - Can immediately launch the Judge evaluation (`python shuffle_for_judge.py --run_judge --max_users 5`).
   - Can compare congruent vs shuffled evaluation reports (`python shuffle_for_judge.py --compare --auto`).

Usage:
  # 1. Generate shuffled baseline dataset (default: seed=42, mode=inter_user)
  python shuffle_for_judge.py

  # 2. Generate and immediately run LLM Judge test on 5 users
  python shuffle_for_judge.py --run_judge --max_users 5

  # 3. Generate and run full LLM Judge evaluation
  python shuffle_for_judge.py --run_judge

  # 4. Compare congruent vs shuffled evaluation reports
  python shuffle_for_judge.py --compare --auto
"""

import argparse
import datetime
import glob
import json
import logging
import math
import os
import random
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats
    from scipy.optimize import linear_sum_assignment
except ImportError:
    stats = None
    linear_sum_assignment = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LLMJudgeShuffler")

# Base directory points to the repository root (DGCDR_)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_JUDGE_RESULTS_DIR = os.path.join(BASE_DIR, "results_judge")


# ==============================================================================
# 1. UTILITY & FILE FINDING FUNCTIONS
# ==============================================================================

def find_latest_results_file(
    results_dir: str = DEFAULT_RESULTS_DIR,
    domain_pair: str = "Cloth-Elec",
    model_name: Optional[str] = None,
) -> Optional[str]:
    """Finds the most recent congruent results JSON file matching filters."""
    pattern = os.path.join(results_dir, "**", "*.json")
    all_files = glob.glob(pattern, recursive=True)

    candidates = []
    for f in all_files:
        norm = os.path.normpath(f)
        bname = os.path.basename(f)
        if bname.startswith("compact_") or "_legacy_archive" in norm or "verification" in norm or "shuffled" in norm:
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


def find_latest_eval_files(
    judge_dir: str = DEFAULT_JUDGE_RESULTS_DIR,
    domain_pair: str = "Cloth-Elec",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds the latest congruent and latest shuffled evaluation JSON files.
    Returns (latest_congruent_file, latest_shuffled_file).
    """
    pattern = os.path.join(judge_dir, domain_pair, "*.json")
    all_files = glob.glob(pattern)

    congruent_files = []
    shuffled_files = []

    for f in all_files:
        bname = os.path.basename(f)
        if "_shuffled_" in bname or "_shuffled" in bname:
            shuffled_files.append(f)
        else:
            congruent_files.append(f)

    latest_cong = None
    latest_shuf = None

    if congruent_files:
        congruent_files.sort(key=os.path.getmtime, reverse=True)
        latest_cong = congruent_files[0]

    if shuffled_files:
        shuffled_files.sort(key=os.path.getmtime, reverse=True)
        latest_shuf = shuffled_files[0]

    return latest_cong, latest_shuf


# ==============================================================================
# 2. CORE SHUFFLE & DERANGEMENT ALGORITHMS
# ==============================================================================

def compute_inter_user_derangement(
    flat_items: List[Dict[str, Any]],
    seed: int = 42,
) -> List[int]:
    """
    Finds an exact derangement (1-to-1 bijection) using bipartite matching
    such that for every item i:
      1. Permutation is collision-free: perm[i] != i
      2. Cross-item: item_id[perm[i]] != item_id[i]
      3. Inter-user: user_id[perm[i]] != user_id[i]
    """
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian bipartite matching derangement.")

    n = len(flat_items)
    user_ids = [it["user_id"] for it in flat_items]
    item_ids = [it["item_id"] for it in flat_items]

    # Verify feasibility: no single user can own > 50% of the items
    user_counts = Counter(user_ids)
    max_user_cnt = max(user_counts.values()) if user_counts else 0
    if max_user_cnt > n / 2:
        raise ValueError(
            f"Inter-user derangement impossible: user with most items owns {max_user_cnt}/{n} items (> 50%)."
        )

    rng = np.random.RandomState(seed)
    cost_matrix = rng.uniform(0.01, 1.0, size=(n, n))

    for i in range(n):
        for j in range(n):
            # Heavy penalty if same instance, same user, or same product ID
            if i == j or user_ids[i] == user_ids[j] or item_ids[i] == item_ids[j]:
                cost_matrix[i, j] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = list(col_ind)

    # Sanity check conflicts
    conflicts = sum(1 for i, j in enumerate(assignment) if cost_matrix[i, j] >= 1e5)
    if conflicts > 0:
        logger.warning(
            f"Hungarian matching found {conflicts} soft constraint violations. "
            "Falling back to global cross-item derangement."
        )
        return compute_global_derangement(flat_items, seed=seed)

    return assignment


def compute_global_derangement(
    flat_items: List[Dict[str, Any]],
    seed: int = 42,
) -> List[int]:
    """
    Finds a global derangement where perm[i] != i and item_id[perm[i]] != item_id[i].
    Allows same user only if strictly necessary.
    """
    n = len(flat_items)
    item_ids = [it["item_id"] for it in flat_items]

    rng = np.random.RandomState(seed)
    cost_matrix = rng.uniform(0.01, 1.0, size=(n, n))

    for i in range(n):
        for j in range(n):
            if i == j or item_ids[i] == item_ids[j]:
                cost_matrix[i, j] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(col_ind)


def compute_intra_user_derangement(
    flat_items: List[Dict[str, Any]],
    seed: int = 42,
    fallback_inter_user: bool = True,
) -> List[int]:
    """
    Shuffles explanations within each user's items for users with >= 2 items.
    For users with 1 item, falls back to inter-user partner if fallback_inter_user=True.
    """
    rng = random.Random(seed)
    n = len(flat_items)
    assignment = list(range(n))

    user_to_indices: Dict[str, List[int]] = {}
    for idx, it in enumerate(flat_items):
        uid = it["user_id"]
        user_to_indices.setdefault(uid, []).append(idx)

    single_item_indices = []

    for uid, indices in user_to_indices.items():
        if len(indices) == 1:
            single_item_indices.append(indices[0])
        elif len(indices) == 2:
            assignment[indices[0]] = indices[1]
            assignment[indices[1]] = indices[0]
        else:
            # Derange indices within user
            local_perm = indices.copy()
            while any(p == orig for p, orig in zip(local_perm, indices)):
                rng.shuffle(local_perm)
            for orig, shuffled in zip(indices, local_perm):
                assignment[orig] = shuffled

    # Handle single item users via inter-user cycle
    if single_item_indices and fallback_inter_user:
        s_count = len(single_item_indices)
        if s_count > 1:
            shuffled_singles = single_item_indices.copy()
            while any(p == orig for p, orig in zip(shuffled_singles, single_item_indices)):
                rng.shuffle(shuffled_singles)
            for orig, target in zip(single_item_indices, shuffled_singles):
                assignment[orig] = target
        elif s_count == 1:
            # Swap with any random item from a different user
            s_idx = single_item_indices[0]
            other_indices = [k for k in range(n) if flat_items[k]["user_id"] != flat_items[s_idx]["user_id"]]
            if other_indices:
                partner = rng.choice(other_indices)
                assignment[s_idx] = assignment[partner]
                assignment[partner] = s_idx

    return assignment


def compute_random_sampling(
    flat_items: List[Dict[str, Any]],
    seed: int = 42,
) -> List[int]:
    """
    Samples random negatives with replacement such that for each i:
    item_id[j] != item_id[i] and user_id[j] != user_id[i].
    """
    rng = random.Random(seed)
    n = len(flat_items)
    assignment = []

    for i in range(n):
        curr_uid = flat_items[i]["user_id"]
        curr_iid = flat_items[i]["item_id"]
        candidates = [
            j for j in range(n)
            if flat_items[j]["user_id"] != curr_uid and flat_items[j]["item_id"] != curr_iid
        ]
        if not candidates:
            candidates = [j for j in range(n) if j != i]
        assignment.append(rng.choice(candidates))

    return assignment


# ==============================================================================
# 3. DATASET GENERATOR & EXPORTER
# ==============================================================================

def create_shuffled_dataset(
    source_results_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    mode: str = "inter_user",
    seed: int = 42,
    min_review_words: int = 5,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Generates the randomized baseline dataset and companion Markdown info card.
    Returns (json_output_path, md_info_path, stats_dict).
    """
    if not os.path.exists(source_results_path):
        raise FileNotFoundError(f"Source results file not found: {source_results_path}")

    logger.info(f"Loading source dataset: {source_results_path}")
    with open(source_results_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    domain_pair = source_data.get("domain_pair", "Cloth-Elec")
    model_name = source_data.get("model", "llama3.1_8b")
    users = source_data.get("users", [])

    # Extract all flat items with valid explanation
    flat_items: List[Dict[str, Any]] = []
    for u_idx, u in enumerate(users):
        uid = u.get("user_id", "")
        for it_idx, it in enumerate(u.get("items", [])):
            expl = it.get("llm_explanation", "").strip()
            rev_text = it.get("user_review_text", "").strip()
            item_title = it.get("item_title", "").strip()
            iid = it.get("id_item") or it.get("item_id", "")

            if expl and rev_text:
                flat_items.append({
                    "user_index": u_idx,
                    "item_index": it_idx,
                    "user_id": uid,
                    "item_id": iid,
                    "item_title": item_title,
                    "review_text": rev_text,
                    "review_word_count": it.get("review_word_count", len(rev_text.split())),
                    "quintile": it.get("quintile"),
                    "quintile_label": it.get("quintile_label"),
                    "llm_explanation": expl,
                })

    n_items = len(flat_items)
    logger.info(f"Loaded {len(users)} users and {n_items} valid items with explanations and reviews.")
    if n_items < 2:
        raise ValueError(f"Need at least 2 items to perform shuffle! Found {n_items}.")

    # Compute assignment based on requested mode
    logger.info(f"Computing '{mode}' derangement mapping (seed={seed})...")
    if mode == "inter_user":
        assignment = compute_inter_user_derangement(flat_items, seed=seed)
    elif mode == "global":
        assignment = compute_global_derangement(flat_items, seed=seed)
    elif mode == "intra_user":
        assignment = compute_intra_user_derangement(flat_items, seed=seed)
    elif mode == "random_sampling":
        assignment = compute_random_sampling(flat_items, seed=seed)
    else:
        raise ValueError(f"Unknown shuffle mode: {mode}")

    # Verify derangement quality
    self_collisions = sum(1 for i, j in enumerate(assignment) if i == j)
    same_item_collisions = sum(1 for i, j in enumerate(assignment) if flat_items[i]["item_id"] == flat_items[j]["item_id"])
    same_user_collisions = sum(1 for i, j in enumerate(assignment) if flat_items[i]["user_id"] == flat_items[j]["user_id"])

    logger.info(
        f"Derangement Quality Check:\n"
        f"  - Total items: {n_items}\n"
        f"  - Self-instance collisions: {self_collisions} (0 expected)\n"
        f"  - Same-item collisions:     {same_item_collisions} (0 expected)\n"
        f"  - Same-user collisions:     {same_user_collisions} (0 expected for inter_user)"
    )

    # Deep copy source data to construct shuffled dataset
    shuffled_data = json.loads(json.dumps(source_data))
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Inject shuffled explanations and metadata into items
    for i, j in enumerate(assignment):
        u_idx = flat_items[i]["user_index"]
        it_idx = flat_items[i]["item_index"]

        source_partner = flat_items[j]
        target_item = shuffled_data["users"][u_idx]["items"][it_idx]

        target_item["original_llm_explanation"] = flat_items[i]["llm_explanation"]
        target_item["llm_explanation"] = source_partner["llm_explanation"]
        target_item["is_shuffled"] = True
        target_item["shuffled_from_item_id"] = source_partner["item_id"]
        target_item["shuffled_from_item_title"] = source_partner["item_title"]
        target_item["shuffled_from_user_id"] = source_partner["user_id"]
        target_item["shuffled_from_quintile"] = source_partner["quintile"]

    # Root metadata
    shuffled_data["is_shuffled"] = True
    shuffled_data["shuffle_info"] = {
        "is_shuffled": True,
        "shuffle_mode": mode,
        "seed": seed,
        "source_file": os.path.abspath(source_results_path),
        "created_at": datetime.datetime.now().isoformat(),
        "total_items_shuffled": n_items,
        "self_collisions": self_collisions,
        "same_item_collisions": same_item_collisions,
        "same_user_collisions": same_user_collisions,
        "min_review_words": min_review_words,
    }

    # Resolve output directory & path
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_RESULTS_DIR, domain_pair, "shuffled")
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        clean_model = model_name.replace(":", "_").replace("/", "_")
        filename = f"{domain_pair}_{clean_model}_shuffled_{mode}_seed{seed}_{now_str}.json"
        output_path = os.path.join(output_dir, filename)

    info_path = os.path.splitext(output_path)[0] + "_info.md"

    # Save JSON dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shuffled_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved shuffled dataset to: {output_path}")

    # Create companion Markdown info card
    sample_indices = list(range(min(3, n_items)))
    md_lines = [
        f"# Shuffled Dataset Card: {domain_pair} ({mode.upper()})\n",
        "## Overview",
        f"- **Purpose**: Negative baseline dataset for evaluating LLM Judge discriminative alignment.",
        f"- **Domain Pair**: `{domain_pair}`",
        f"- **Model Origin**: `{model_name}`",
        f"- **Shuffle Mode**: `{mode}`",
        f"- **Random Seed**: `{seed}`",
        f"- **Total Shuffled Items**: {n_items}",
        f"- **Self Collisions**: {self_collisions}",
        f"- **Same-Item Collisions**: {same_item_collisions}",
        f"- **Same-User Collisions**: {same_user_collisions}",
        f"- **Source File**: `{os.path.basename(source_results_path)}`",
        f"- **Generated At**: `{datetime.datetime.now().isoformat()}`\n",
        "---",
        "## Example Qualitative Mismatches (Negative Pairs):\n",
    ]

    for k in sample_indices:
        orig = flat_items[k]
        shuf = flat_items[assignment[k]]
        md_lines.extend([
            f"### Sample #{k+1} (User: `{orig['user_id']}`)",
            f"- **Target Product**: {orig['item_title']} (ID: `{orig['item_id']}`)",
            f"- **Actual User Review** ({orig['quintile']} {orig['quintile_label']}, {orig['review_word_count']}w):",
            f"  > *\"{orig['review_text']}\"*",
            f"- **Original Congruent Explanation** (SHOULD score ~3-5):",
            f"  > *\"{orig['llm_explanation']}\"*",
            f"- **Mismatched Shuffled Explanation** (from `{shuf['item_title']}`, ID: `{shuf['item_id']}`, User: `{shuf['user_id']}`) (SHOULD score 1-2):",
            f"  > *\"{shuf['llm_explanation']}\"*\n",
        ])

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved shuffled dataset summary to: {info_path}")

    stats_summary = {
        "output_path": output_path,
        "info_path": info_path,
        "total_items": n_items,
        "self_collisions": self_collisions,
        "same_item_collisions": same_item_collisions,
        "same_user_collisions": same_user_collisions,
    }
    return output_path, info_path, stats_summary


# ==============================================================================
# 4. RUN JUDGE ON SHUFFLED DATASET
# ==============================================================================

def run_judge_evaluation(
    shuffled_dataset_path: str,
    num_users: Optional[int] = None,
    model: str = "llama3.1:8b",
    temperature: float = 0.0,
    seed: int = 42,
    ollama_url: str = "http://localhost:11434",
    output_dir: str = "results_judge/Cloth-Elec",
) -> int:
    """Runs test_geval_prompts.py on the shuffled dataset using the active Python interpreter."""
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "llm_explainer", "judge", "test_geval_prompts.py"),
        "--results_file", shuffled_dataset_path,
        "--model", model,
        "--temperature", str(temperature),
        "--seed", str(seed),
        "--ollama_url", ollama_url,
        "--output_dir", output_dir,
    ]
    if num_users is not None:
        cmd.extend(["--num_users", str(num_users)])

    logger.info("Executing LLM Judge evaluation on shuffled baseline...")
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# ==============================================================================
# 5. CONTRASTIVE BENCHMARK & COMPARISON (CONGRUENT VS SHUFFLED)
# ==============================================================================

def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Computes descriptive stats for a list of values."""
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def compute_statistical_tests(pos_scores: List[float], neg_scores: List[float]) -> Dict[str, Any]:
    """Computes paired t-test, Wilcoxon signed-rank test, and Cohen's d effect size."""
    if len(pos_scores) != len(neg_scores) or len(pos_scores) < 2 or stats is None:
        return {"t_stat": None, "t_pvalue": None, "wilcoxon_stat": None, "wilcoxon_pvalue": None, "cohens_d": None}

    try:
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

        # Cohen's d
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
        return {"t_stat": None, "t_pvalue": None, "wilcoxon_stat": None, "wilcoxon_pvalue": None, "cohens_d": None}


def compare_congruent_vs_shuffled(
    congruent_path: str,
    shuffled_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compares congruent vs shuffled evaluation reports.
    Computes contrastive margins, discriminative win rate, score distributions,
    hypothesis tests, and quintile breakdown.
    """
    with open(congruent_path, "r", encoding="utf-8") as f:
        cong_data = json.load(f)
    with open(shuffled_path, "r", encoding="utf-8") as f:
        shuf_data = json.load(f)

    # Build lookup table for shuffled items: (user_id, item_id) -> item_data
    shuf_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for u in shuf_data.get("users", []):
        uid = u["user_id"]
        for it in u.get("items", []):
            iid = it.get("item_id") or it.get("id_item", "")
            shuf_map[(uid, iid)] = it

    matched_pairs = []
    for u in cong_data.get("users", []):
        uid = u["user_id"]
        for it in u.get("items", []):
            iid = it.get("item_id") or it.get("id_item", "")
            key = (uid, iid)
            if key in shuf_map:
                shuf_it = shuf_map[key]
                c_score = it.get("score")
                s_score = shuf_it.get("score")
                c_weight = it.get("weighted_score")
                s_weight = shuf_it.get("weighted_score")

                if c_score is not None and s_score is not None:
                    matched_pairs.append({
                        "user_id": uid,
                        "item_id": iid,
                        "item_title": it.get("item_title", ""),
                        "review": it.get("review") or it.get("user_review", ""),
                        "quintile": it.get("quintile", "Q3"),
                        "quintile_label": it.get("quintile_label", ""),
                        "words": it.get("review_word_count", 0),
                        "congruent_score": c_score,
                        "congruent_weighted": c_weight,
                        "shuffled_score": s_score,
                        "shuffled_weighted": s_weight,
                        "shuffled_from_title": shuf_it.get("shuffled_from_item_title", "Unknown"),
                    })

    n_matched = len(matched_pairs)
    logger.info(f"Found {n_matched} matched evaluation pairs between congruent and shuffled runs.")
    if n_matched == 0:
        raise ValueError("No common items found between the two evaluation files.")

    c_scores = [p["congruent_score"] for p in matched_pairs]
    s_scores = [p["shuffled_score"] for p in matched_pairs]
    c_weights = [p["congruent_weighted"] for p in matched_pairs if p["congruent_weighted"] is not None]
    s_weights = [p["shuffled_weighted"] for p in matched_pairs if p["shuffled_weighted"] is not None]

    # Metrics
    stats_c_disc = compute_distribution_stats(c_scores)
    stats_s_disc = compute_distribution_stats(s_scores)
    stats_c_wt = compute_distribution_stats(c_weights)
    stats_s_wt = compute_distribution_stats(s_weights)

    margin_disc = round(stats_c_disc["mean"] - stats_s_disc["mean"], 4)
    margin_wt = round(stats_c_wt["mean"] - stats_s_wt["mean"], 4)

    wins = sum(1 for c, s in zip(c_scores, s_scores) if c > s)
    ties = sum(1 for c, s in zip(c_scores, s_scores) if c == s)
    losses = sum(1 for c, s in zip(c_scores, s_scores) if c < s)

    win_rate = round(wins / n_matched * 100, 2)
    tie_rate = round(ties / n_matched * 100, 2)
    loss_rate = round(losses / n_matched * 100, 2)

    # Distributions
    dist_c = {s: c_scores.count(s) for s in range(1, 6)}
    dist_s = {s: s_scores.count(s) for s in range(1, 6)}
    dist_c_pct = {s: round(dist_c[s] / n_matched * 100, 1) for s in range(1, 6)}
    dist_s_pct = {s: round(dist_s[s] / n_matched * 100, 1) for s in range(1, 6)}

    # Hypothesis testing
    tests_disc = compute_statistical_tests(c_scores, s_scores)
    tests_wt = compute_statistical_tests(c_weights, s_weights)

    # Quintile breakdown
    by_quintile = {}
    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        q_pairs = [p for p in matched_pairs if p["quintile"] == qid]
        if q_pairs:
            q_c = [p["congruent_score"] for p in q_pairs]
            q_s = [p["shuffled_score"] for p in q_pairs]
            q_cw = [p["congruent_weighted"] for p in q_pairs if p["congruent_weighted"] is not None]
            q_sw = [p["shuffled_weighted"] for p in q_pairs if p["shuffled_weighted"] is not None]

            q_wins = sum(1 for c, s in zip(q_c, q_s) if c > s)
            by_quintile[qid] = {
                "count": len(q_pairs),
                "label": q_pairs[0]["quintile_label"],
                "avg_congruent": round(float(np.mean(q_c)), 4),
                "avg_shuffled": round(float(np.mean(q_s)), 4),
                "margin_discrete": round(float(np.mean(q_c) - np.mean(q_s)), 4),
                "avg_congruent_weighted": round(float(np.mean(q_cw)), 4) if q_cw else None,
                "avg_shuffled_weighted": round(float(np.mean(q_sw)), 4) if q_sw else None,
                "margin_weighted": round(float(np.mean(q_cw) - np.mean(q_sw)), 4) if q_cw and q_sw else None,
                "win_rate": round(q_wins / len(q_pairs) * 100, 2),
            }

    # Summary
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "congruent_file": congruent_path,
        "shuffled_file": shuffled_path,
        "matched_items": n_matched,
        "margins": {
            "discrete_margin": margin_disc,
            "weighted_margin": margin_wt,
        },
        "rates": {
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "loss_rate": loss_rate,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "discrete_stats": {
            "congruent": stats_c_disc,
            "shuffled": stats_s_disc,
        },
        "weighted_stats": {
            "congruent": stats_c_wt,
            "shuffled": stats_s_wt,
        },
        "score_distributions": {
            "congruent_counts": dist_c,
            "congruent_pct": dist_c_pct,
            "shuffled_counts": dist_s,
            "shuffled_pct": dist_s_pct,
        },
        "statistical_tests": {
            "discrete": tests_disc,
            "weighted": tests_wt,
        },
        "by_quintile": by_quintile,
    }

    # Save Markdown report
    if output_dir is None:
        output_dir = os.path.dirname(congruent_path)
    os.makedirs(output_dir, exist_ok=True)

    json_report_path = os.path.join(output_dir, f"benchmark_judge_congruent_vs_shuffled_{now_str}.json")
    md_report_path = os.path.join(output_dir, f"benchmark_judge_congruent_vs_shuffled_{now_str}.md")

    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # Format p-value helper
    def fmt_p(p):
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

    md_lines = [
        "# LLM-as-a-Judge Discriminative Benchmark Report\n",
        "## Contrastive Validation (Congruent vs. Shuffled Baseline)\n",
        f"- **Evaluated Pairs**: {n_matched} items",
        f"- **Congruent Source**: `{os.path.basename(congruent_path)}`",
        f"- **Shuffled Source**: `{os.path.basename(shuffled_path)}`",
        f"- **Timestamp**: `{datetime.datetime.now().isoformat()}`\n",
        "### 1. Key Discriminative Metrics",
        f"| Metric | Congruent (Real) | Shuffled (Baseline) | Contrastive Margin ($\Delta$) | Win Rate (Pos > Neg) |",
        f"|:---|:---:|:---:|:---:|:---:|",
        f"| **Discrete Score (1-5)** | **{stats_c_disc['mean']}** &plusmn; {stats_c_disc['std']} | **{stats_s_disc['mean']}** &plusmn; {stats_s_disc['std']} | **{margin_disc:+.4f}** | **{win_rate}%** |",
        f"| **Weighted Score (G-Eval)** | **{stats_c_wt['mean']}** &plusmn; {stats_c_wt['std']} | **{stats_s_wt['mean']}** &plusmn; {stats_s_wt['std']} | **{margin_wt:+.4f}** | - |\n",
        "### 2. Discrimination Outcome Breakdown",
        f"- **Discriminative Wins (Congruent > Shuffled)**: **{wins}** ({win_rate}%)",
        f"- **Ties (Congruent == Shuffled)**: **{ties}** ({tie_rate}%)",
        f"- **Inversions (Congruent < Shuffled)**: **{losses}** ({loss_rate}%)\n",
        "### 3. Score Distributions Comparison",
        "| Score Level | Congruent Count (%) | Shuffled Count (%) | Delta % (Cong - Shuf) | Theoretical Expectation |",
        "|:---:|:---:|:---:|:---:|:---|",
    ]

    for s in range(1, 6):
        d_pct = round(dist_c_pct[s] - dist_s_pct[s], 1)
        exp_txt = "Expected high on Shuffled" if s in [1, 2] else ("Expected dominant on Congruent" if s in [4, 5] else "Corroboration failure zone")
        md_lines.append(
            f"| **Score {s}** | {dist_c[s]} ({dist_c_pct[s]}%) | {dist_s[s]} ({dist_s_pct[s]}%) | {d_pct:+.1f}% | {exp_txt} |"
        )

    md_lines.extend([
        "\n### 4. Statistical Hypothesis Testing",
        f"- **Paired Student's t-test**: t = {tests_disc.get('t_stat')}, p-value = {fmt_p(tests_disc.get('t_pvalue'))}",
        f"- **Wilcoxon Signed-Rank Test**: W = {tests_disc.get('wilcoxon_stat')}, p-value = {fmt_p(tests_disc.get('wilcoxon_pvalue'))}",
        f"- **Cohen's d (Effect Size)**: d = {tests_disc.get('cohens_d')} ({'Large effect' if (tests_disc.get('cohens_d') or 0) > 0.8 else 'Moderate effect'})\n",
        "### 5. Stratification by Review Length Quintiles",
        "| Quintile | Label | Count | Congruent Mean | Shuffled Mean | Margin ($\Delta$) | Win Rate |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        if qid in by_quintile:
            q = by_quintile[qid]
            md_lines.append(
                f"| **{qid}** | {q['label']} | {q['count']} | {q['avg_congruent']} | {q['avg_shuffled']} | **{q['margin_discrete']:+.4f}** | {q['win_rate']}% |"
            )

    md_lines.append("\n---\n")

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 65)
    print("      BENCHMARK LLM JUDGE: CONGRUENT vs. SHUFFLED BASELINE       ")
    print("=" * 65)
    print(f"Items Evaluati:             {n_matched}")
    print(f"Media Congruent:            {stats_c_disc['mean']} / 5  (Weighted: {stats_c_wt['mean']})")
    print(f"Media Shuffled (Baseline):  {stats_s_disc['mean']} / 5  (Weighted: {stats_s_wt['mean']})")
    print(f"Margine Contrastivo Delta:  {margin_disc:+.4f}     (Weighted: {margin_wt:+.4f})")
    print(f"Discriminative Win Rate:    {win_rate}% ({wins}/{n_matched})")
    print(f"T-Test p-value:             {fmt_p(tests_disc.get('t_pvalue'))}")
    print(f"Cohen's d Effect Size:      {tests_disc.get('cohens_d')}")
    print("Distribuzione Punteggi:")
    for s in range(1, 6):
        print(f"  Score {s}: Congruent = {dist_c[s]:>3} ({dist_c_pct[s]:>4}%) | Shuffled = {dist_s[s]:>3} ({dist_s_pct[s]:>4}%)")
    print("-" * 65)
    print(f"Report salvati in:\n  - {json_report_path}\n  - {md_report_path}")
    print("=" * 65 + "\n")

    return report_dict


# ==============================================================================
# 6. CLI ENTRY POINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dedicated shuffle and baseline script for evaluating LLM-as-a-judge."
    )
    # Shuffle dataset creation options
    parser.add_argument(
        "--results_file",
        "-f",
        type=str,
        default="results/Cloth-Elec/llama3.1_8b_v2/Cloth-Elec_llama3.1_8b_v2_20260930_164711.json",
        help="Path to results JSON file to shuffle. If omitted and --latest is set, automatically finds newest.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically pick the latest results JSON file in results/.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inter_user", "global", "intra_user", "random_sampling"],
        default="inter_user",
        help="Shuffle mode: 'inter_user' (default, cross-user derangement), 'global', 'intra_user', 'random_sampling'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Custom output path for shuffled JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save shuffled dataset files (default: results/<domain_pair>/shuffled).",
    )
    parser.add_argument(
        "--min_words",
        "--min_review_words",
        type=int,
        default=5,
        help="Minimum words filter threshold for reviews (default: 5).",
    )

    # Judge execution options
    parser.add_argument(
        "--run_judge",
        action="store_true",
        help="Automatically launch LLM Judge (test_geval_prompts.py) on the generated shuffled dataset.",
    )
    parser.add_argument(
        "--num_users",
        "--max_users",
        type=int,
        default=None,
        help="Number of users to evaluate when running judge (e.g. 5 for quick test, default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="LLM Judge model (default: llama3.1:8b).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0.0).",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434).",
    )

    # Comparison options
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare congruent vs shuffled evaluation files and generate benchmark report.",
    )
    parser.add_argument(
        "--congruent_eval",
        type=str,
        default=None,
        help="Path to congruent evaluation JSON file in results_judge/.",
    )
    parser.add_argument(
        "--shuffled_eval",
        type=str,
        default=None,
        help="Path to shuffled evaluation JSON file in results_judge/.",
    )
    parser.add_argument(
        "--auto_compare",
        action="store_true",
        help="Automatically find latest congruent and shuffled evaluation files in results_judge/.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Mode 1: Benchmark comparison between evaluation runs
    if args.compare or args.auto_compare:
        cong_file = args.congruent_eval
        shuf_file = args.shuffled_eval

        if args.auto_compare or not (cong_file and shuf_file):
            logger.info("Auto-detecting latest evaluation runs in results_judge/Cloth-Elec/...")
            latest_c, latest_s = find_latest_eval_files()
            if not cong_file:
                cong_file = latest_c
            if not shuf_file:
                shuf_file = latest_s

        if not cong_file or not os.path.exists(cong_file):
            logger.error(f"Congruent evaluation file not found: {cong_file}")
            sys.exit(1)
        if not shuf_file or not os.path.exists(shuf_file):
            logger.error(f"Shuffled evaluation file not found: {shuf_file}")
            sys.exit(1)

        logger.info(f"Comparing:\n  Congruent: {cong_file}\n  Shuffled:  {shuf_file}")
        compare_congruent_vs_shuffled(cong_file, shuf_file)
        return

    # Mode 2: Generate shuffled baseline dataset
    source_file = args.results_file
    if args.latest:
        latest = find_latest_results_file()
        if latest:
            source_file = latest

    if not os.path.exists(source_file):
        logger.error(f"Source results file not found: {source_file}")
        sys.exit(1)

    print("=================================================================")
    print("      GENERAZIONE DATASET BASELINE SHUFFLED PER LLM JUDGE        ")
    print("=================================================================")
    print(f"File Sorgente:       {source_file}")
    print(f"Shuffle Mode:        {args.mode}")
    print(f"Seed:                {args.seed}")
    print(f"Esecuzione Judge:    {'SI' if args.run_judge else 'NO'}")
    if args.run_judge:
        print(f"Utenti da Valutare:  {args.num_users if args.num_users is not None else 'TUTTI'}")
        print(f"Modello Judge:       {args.model}")
    print("=================================================================\n")

    json_path, info_path, stats = create_shuffled_dataset(
        source_results_path=source_file,
        output_path=args.output_file,
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        min_review_words=args.min_words,
    )

    print("\n-----------------------------------------------------------------")
    print("Dataset Shuffled generato con successo!")
    print(f"  - JSON Dataset: {json_path}")
    print(f"  - Card Info MD: {info_path}")
    print(f"  - Totale Item:  {stats['total_items']}")
    print(f"  - Collisioni:   {stats['same_item_collisions']} item / {stats['same_user_collisions']} user")
    print("-----------------------------------------------------------------\n")

    # Mode 3: Optional immediate evaluation
    if args.run_judge:
        print("Avvio della valutazione LLM Judge sul dataset shuffled...")
        ret = run_judge_evaluation(
            shuffled_dataset_path=json_path,
            num_users=args.num_users,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
            ollama_url=args.ollama_url,
        )
        if ret != 0:
            logger.error("Valutazione LLM Judge terminata con errori.")
            sys.exit(ret)


if __name__ == "__main__":
    main()
