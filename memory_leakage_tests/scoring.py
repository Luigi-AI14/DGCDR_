"""Exact recovery, permutation controls, clustered intervals and concise reports."""

import csv
import io
import logging
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .common import digest, normalize, now, read_json, word_count, write_json
from .probes import FIELDS, tasks_for
from .runner import load_prepared, parse_answer, read_journal

LOG = logging.getLogger(__name__)
VALID = ("ok", "abstained")
FUZZY_CHAR_LIMIT = 10000


def score(task, record):
    result = {"task": task, "status": record["status"] if record else "missing",
              "answer": None, "primary_exact": False, "raw_exact": False,
              "record_exact": False, "field_exact": {}, "fuzzy_similarity": 0.0}
    if not record or record["status"] not in VALID:
        return result
    # Reparse the recorded raw output; do not trust a cached score/parsed answer.
    try:
        answer = parse_answer(record["raw_response"], task["probe"])
    except (ValueError, TypeError):
        result["status"] = "format_error"
        return result
    result["answer"] = answer
    if task["condition"] != "real":
        result["hallucinated"] = any(v is not None and (not isinstance(v, str) or bool(v.strip()))
                                     for v in answer.values())
        return result
    expected = task["expected"]
    for field in FIELDS[task["probe"]]:
        actual, truth = answer[field], expected[field]
        result["field_exact"][field] = (actual is not None and
            (actual == truth if field == "rating" else normalize(actual) == normalize(truth)))
    field = "item_title" if task["probe"] == "item" else "review_text"
    actual = answer[field]
    result["primary_exact"] = bool(normalize(actual)) and result["field_exact"][field]
    result["raw_exact"] = bool(actual) and actual == expected[field]
    result["record_exact"] = all(result["field_exact"].values())
    if isinstance(actual, str) and actual:
        result["fuzzy_similarity"] = (1.0 if result["primary_exact"] else
            SequenceMatcher(None, normalize(actual)[:FUZZY_CHAR_LIMIT],
                            normalize(expected[field])[:FUZZY_CHAR_LIMIT]).ratio())
    return result


def permutation_control(rows, iterations, seed):
    """Shuffle references, keeping outputs and their failures fixed (one-sided)."""
    field = "item_title" if rows[0]["task"]["probe"] == "item" else "review_text"
    predictions = [normalize(row["answer"].get(field)) if row["answer"] else "" for row in rows]
    references = [normalize(row["task"]["expected"][field]) for row in rows]
    observed = sum(row["primary_exact"] for row in rows)
    if len(rows) < 2:
        return {"shuffled_exact_rate": None, "permutation_p": None}
    if observed == 0 and not any(predictions):
        return {"shuffled_exact_rate": 0.0, "permutation_p": 1.0}
    rng = random.Random(seed)
    greater, null_sum = 0, 0
    shuffled = list(references)
    for _ in range(iterations):
        rng.shuffle(shuffled)
        matches = sum(bool(pred) and pred == ref for pred, ref in zip(predictions, shuffled))
        null_sum += matches
        greater += matches >= observed
    return {"shuffled_exact_rate": null_sum / (iterations * len(rows)),
            "permutation_p": (greater + 1) / (iterations + 1)}


def cluster_interval(rows, iterations, seed):
    """Percentile bootstrap: user clusters for reviews, unique items for items."""
    clusters = defaultdict(lambda: [0, 0])
    for row in rows:
        task = row["task"]
        key = task["user_id"] if task["probe"] == "review" else task["item_id"]
        clusters[key][0] += int(row["primary_exact"])
        clusters[key][1] += 1
    if len(clusters) < 2:
        return [None, None]
    values = list(clusters.values())
    rng = random.Random(seed)
    rates = []
    for _ in range(iterations):
        selected = rng.choices(values, k=len(values))
        rates.append(sum(c[0] for c in selected) / sum(c[1] for c in selected))
    rates.sort()
    return [rates[int(0.025 * (iterations - 1))], rates[int(0.975 * (iterations - 1))]]


def summarize(rows, controls, model, mode, cohort_seed, probe, config, text_counts,
              missing_item_titles, complete_users):
    # Cross-seed item overlap is allowed, but pooled coverage counts each item once.
    if cohort_seed == "ALL" and probe == "item":
        unique = {}
        for row in rows:
            unique.setdefault(row["task"]["item_id"], row)
        rows = list(unique.values())
    n = len(rows)
    matched = sum(row["primary_exact"] for row in rows)
    failures = sum(row["status"] not in VALID for row in rows)
    result = {"model": model, "mode": mode, "cohort_seed": cohort_seed, "probe": probe,
              "evidence_scope": "item_knowledge" if probe == "item" else "user_item_review",
              "users": complete_users, "unique_items": len({r["task"]["item_id"] for r in rows}),
              "scheduled": n, "matched": matched, "exact_rate": matched / n if n else None,
              "raw_exact": sum(row["raw_exact"] for row in rows),
              "complete_record_exact": sum(row["record_exact"] for row in rows),
              "failures_or_missing": failures,
              "abstentions": sum(row["status"] == "abstained" for row in rows),
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "missing_item_titles": missing_item_titles,
              "negative_controls": len(controls),
              "negative_control_failures": sum(row["status"] not in VALID for row in controls),
              "negative_control_hallucinations": sum(row.get("hallucinated", False) for row in controls),
              "fuzzy_similarity_mean": sum(row["fuzzy_similarity"] for row in rows) / n if n else None,
              "fuzzy_char_limit": FUZZY_CHAR_LIMIT}
    for field in FIELDS[probe]:
        eligible = [row for row in rows if field == "rating" or normalize(row["task"]["expected"][field])]
        result[field + "_scorable"] = len(eligible)
        result[field + "_exact"] = sum(row["field_exact"].get(field, False) for row in eligible)
    if probe == "review" and n:
        ratings = Counter(row["task"]["expected"]["rating"] for row in rows)
        result["majority_rating_baseline"] = max(ratings.values()) / n
    else:
        result["majority_rating_baseline"] = None
    if not n:
        result.update(shuffled_exact_rate=None, permutation_p=None, adjusted_p=None,
                      exact_rate_ci95=[None, None], strong_distinct_matches=0,
                      evidence_status="INCONCLUSIVE")
        return result
    seed = digest([model, mode, cohort_seed, probe])
    result.update(permutation_control(rows, config["permutations"], seed))
    result["exact_rate_ci95"] = cluster_interval(rows, config["bootstrap_samples"], seed)
    rule = config["evidence_rule"]
    if probe == "review":
        # Avoid treating common five-word reviews or duplicate texts as strong
        # identity-specific evidence. Uniqueness is checked across the full pool.
        strong = {normalize(row["task"]["expected"]["review_text"]) for row in rows
                  if row["record_exact"]
                  and word_count(row["task"]["expected"]["review_text"]) >= rule["min_unique_review_words"]
                  and text_counts[normalize(row["task"]["expected"]["review_text"])] == 1}
    else:
        strong = {row["task"]["item_id"] for row in rows if row["primary_exact"]}
    result["strong_distinct_matches"] = len(strong)
    # Only pooled hypotheses determine evidence labels. Seed rows are descriptive.
    comparisons = len(config["models"]) * len(config["modes"]) * 2
    p = result["permutation_p"]
    result["adjusted_p"] = min(1.0, p * comparisons) if p is not None else None
    if cohort_seed != "ALL":
        result["evidence_status"] = ("RECOVERY_DETECTED" if matched else
                                     "INCONCLUSIVE" if failures or not controls or any(r["status"] not in VALID for r in controls)
                                     else "NO_EVIDENCE_DETECTED")
    elif (result["adjusted_p"] is not None and result["adjusted_p"] <= rule["alpha"]
          and len(strong) >= rule["min_distinct_records"]):
        result["evidence_status"] = "EVIDENCE_DETECTED"
    elif matched:
        result["evidence_status"] = "RECOVERY_DETECTED_REVIEW_REQUIRED"
    elif failures or (probe == "item" and missing_item_titles) or not controls or any(row["status"] not in VALID for row in controls):
        result["evidence_status"] = "INCONCLUSIVE"
    else:
        result["evidence_status"] = "NO_EVIDENCE_DETECTED"
    return result


def _csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json_value(value) for key, value in row.items()})
    Path(path).write_text(stream.getvalue(), encoding="utf-8-sig")


def json_value(value):
    import json
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value


def report(output_dir, run_name="default"):
    dataset, manifest = load_prepared(output_dir)
    run_dir = Path(output_dir) / "runs" / run_name
    config = read_json(run_dir / "run_config.json")["config"]
    if config["manifest_sha256"] != digest(manifest):
        raise ValueError("Run refers to a different cohort manifest")
    journal = read_journal(run_dir / "responses.jsonl")
    config_sha = digest(config)
    text_counts = Counter(normalize(i["ground_truth_review"]) for u in dataset["users"] for i in u["held_out_items"])
    summaries, candidates = [], []
    groups, negative = defaultdict(list), defaultdict(list)
    expected_keys = set()
    for task in tasks_for(manifest, config["modes"]):
        for model in config["models"]:
            key = (model, task["task_id"])
            expected_keys.add(key)
            record = journal.get(key)
            if record and (record["config_sha256"] != config_sha or
                           record["request_sha256"] != digest(task["request"])):
                raise ValueError("Response configuration or prompt fingerprint mismatch")
            row = score(task, record)
            for seed in (task["cohort_seed"], "ALL"):
                group = (model, task["mode"], seed, task["probe"])
                (groups if task["condition"] == "real" else negative)[group].append(row)
            if task["condition"] == "real" and (row["primary_exact"] or row["fuzzy_similarity"] >= 0.8):
                candidates.append({"model": model, "mode": task["mode"], "cohort_seed": task["cohort_seed"],
                    "probe": task["probe"], "user_id": task["user_id"], "item_id": task["item_id"],
                    "normalized_exact": row["primary_exact"], "raw_exact": row["raw_exact"],
                    "complete_record_exact": row["record_exact"], "fuzzy_similarity": row["fuzzy_similarity"],
                    "expected": task["expected"], "generated": row["answer"], "manual_verdict": ""})
    if set(journal) - expected_keys:
        raise ValueError("Journal contains unexpected model/task IDs")
    for model in config["models"]:
        for mode in config["modes"]:
            for seed in ["ALL"] + manifest["cohort_seeds"]:
                cohorts = manifest["cohorts"] if seed == "ALL" else [c for c in manifest["cohorts"] if c["seed"] == seed]
                missing_titles = len({i["item_id"] for c in cohorts for u in c["users"]
                                      for i in u["held_out_items"] if not normalize(i.get("item_title"))
                                      or normalize(i.get("item_title")) == normalize("Item " + i["item_id"])})
                for probe in ("item", "review"):
                    group = (model, mode, seed, probe)
                    summaries.append(summarize(groups[group], negative[group], model, mode, seed,
                                               probe, config, text_counts, missing_titles,
                                               sum(len(c["users"]) for c in cohorts)))
    write_json(run_dir / "summary.json", {"created_at": now(), "manifest_sha256": digest(manifest),
               "config_sha256": config_sha, "rows": summaries})
    _csv(run_dir / "summary.csv", summaries)
    _csv(run_dir / "recovery_candidates.csv", candidates)
    lines = ["ID-ONLY MEMORY AUDIT", "Split seed: {} | Cohort seeds: {} | Users per cohort: {}".format(
             manifest["split_seed"], manifest["cohort_seeds"], manifest["num_users"]),
             "Exact = NFKC + casefold + collapsed whitespace; punctuation is preserved.",
             "Coverage denominator includes abstentions, failures and missing requests.",
             "ALL item rows count each item once; review uncertainty clusters by user.",
             "EVIDENCE_DETECTED flags ID recovery, not proven training membership or inflated judge scores.",
             "Item knowledge alone does not establish user-review leakage. Inspect recovery_candidates.csv.",
             "Synthetic controls test abstention on random IDs; they do not certify global nonexistence.",
             "", "POOLED RESULTS (five cohorts)",
             "Model            Mode       Probe   Exact / N       Shuffled    Failed    Status"]
    def percent(value):
        return "n/a" if value is None else "{:.2f}%".format(value * 100)
    for row in summaries:
        if row["cohort_seed"] == "ALL":
            lines.append("{:<16} {:<10} {:<7} {:>5}/{:<7} {:>9} {:>7}    {}".format(
                row["model"], row["mode"], row["probe"], row["matched"], row["scheduled"],
                percent(row["shuffled_exact_rate"]), row["failures_or_missing"], row["evidence_status"]))
            lines.append("  Coverage {} | 95% cluster CI {} to {} | adjusted p={} | strong matches={} | synthetic hallucinations={}/{} (failures={})".format(
                percent(row["exact_rate"]), percent(row["exact_rate_ci95"][0]), percent(row["exact_rate_ci95"][1]),
                "n/a" if row["adjusted_p"] is None else "{:.4g}".format(row["adjusted_p"]),
                row["strong_distinct_matches"], row["negative_control_hallucinations"], row["negative_controls"],
                row["negative_control_failures"]))
    lines += ["", "PER-SEED RESULTS", "Model            Mode       Seed    Probe   Exact / N       Coverage    Failed"]
    for row in summaries:
        if row["cohort_seed"] != "ALL":
            lines.append("{:<16} {:<10} {:<7} {:<7} {:>5}/{:<7} {:>9} {:>7}".format(
                row["model"], row["mode"], row["cohort_seed"], row["probe"], row["matched"],
                row["scheduled"], percent(row["exact_rate"]), row["failures_or_missing"]))
    lines += ["", "More detail: summary.csv / summary.json; matches: recovery_candidates.csv; raw audit: responses.jsonl.",
              "Fuzzy similarity is exploratory and is never used to declare evidence.",
              "No evidence detected means no recovery under these prompts; it does not prove absence of leakage."]
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Summary saved: %s", run_dir / "summary.txt")
    print("\n".join(lines[:lines.index("PER-SEED RESULTS") - 1]))
    return summaries
