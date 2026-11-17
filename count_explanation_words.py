"""Count explanation/review words, overall and by saved review-length quintile.

Usage: python count_explanation_words.py [--include-review-title]
Basic counts use only the standard library; --user-analysis needs NumPy.
Reviews are the held-out item reviews
saved in the reports, NOT the review histories provided to the LLM.
"""

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUP_FIELDS = ("domain_pair", "model", "prompt_version", "generation_mode")
QUINTILE_FIELDS = ("quintile", "quintile_label", "quintile_range_words",
                   "quintile_uses_review_title")


def count_words(text):
    """Count whitespace-separated words after removing HTML and decoding entities.

    Contractions and hyphenated words each count as one word, matching the
    whitespace-based explanation length limit in the generation prompts.
    """
    return len(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def pearson(xs, ys):
    """Return None when correlation is undefined (too few or constant values)."""
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator else None


def ranks(values):
    """Average ranks for ties, for Spearman correlation."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def load_rows(directory, include_title=False):
    rows, seen = [], set()
    duplicates = skipped = files = 0
    for path in sorted(directory.rglob("explanations_*.json")):
        # Shuffled baselines are not the original review/explanation pairing.
        if "shuffled" in path.as_posix().lower():
            continue
        with path.open(encoding="utf-8-sig") as stream:
            report = json.load(stream)
        if not isinstance(report.get("users"), list):
            raise ValueError("Missing users list in {}".format(path))
        files += 1
        metadata = {key: report.get(key, "unknown") for key in GROUP_FIELDS}
        seed = report.get("sampling_seed", report.get("seed", "unknown"))
        definitions = report.get("quintile_metrics", {}).get("by_quintile", {})
        for user in report["users"]:
            for item in user.get("items", []):
                explanation = item.get("llm_explanation") or ""
                review = item.get("user_review_text") or ""
                title = item.get("user_review_title") or ""
                quintile = item.get("quintile") or "unassigned"
                if item.get("is_filtered_ultrashort"):
                    quintile = "excluded"
                definition = definitions.get(quintile, {})
                quintile_metadata = dict(
                    quintile=quintile,
                    quintile_label=item.get("quintile_label") or "unknown",
                    quintile_range_words=definition.get("range_words") or "unknown",
                    quintile_uses_review_title=report.get("use_title_in_quintiles", "unknown"),
                )
                # Collapse exact repeated observations within the same run settings.
                # Different seeds and genuinely different generations are retained.
                key = tuple(metadata.values()) + (
                    seed, report.get("split_seed"), report.get("inference_seed"),
                    report.get("temperature"), user["user_id"], item["id_item"],
                    item.get("prompt_sha256"), explanation, review, title,
                ) + tuple(quintile_metadata.values())
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                review_words = count_words((title + " " if include_title else "") + review)
                explanation_words = count_words(explanation)
                if not review_words or not explanation_words:
                    skipped += 1
                rows.append(dict(
                    metadata, **quintile_metadata,
                    seed=seed, user_id=user["user_id"], item_id=item["id_item"],
                    saved_review_word_count=item.get("review_word_count"),
                    review_body_words=count_words(review), review_title_words=count_words(title),
                    review_words=review_words, explanation_words=explanation_words,
                    source_file=path.relative_to(directory).as_posix(),
                ))
    return rows, files, duplicates, skipped


def summarize(rows, group_fields=GROUP_FIELDS):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_fields)].append(row)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda entry: tuple(map(str, entry[0]))):
        pairs = [r for r in group if r["review_words"] and r["explanation_words"]]
        xs = [r["review_words"] for r in pairs]
        ys = [r["explanation_words"] for r in pairs]
        lengths = [r["explanation_words"] for r in group]
        summaries.append(dict(
            zip(group_fields, key),
            explanations=len(group), paired_observations=len(pairs),
            unique_users=len({r["user_id"] for r in group}),
            total_explanation_words=sum(lengths),
            mean_explanation_words=statistics.mean(lengths),
            median_explanation_words=statistics.median(lengths),
            std_explanation_words=statistics.stdev(lengths) if len(lengths) > 1 else None,
            min_explanation_words=min(lengths), max_explanation_words=max(lengths),
            mean_review_words=statistics.mean(xs) if xs else None,
            pearson_r=pearson(xs, ys), spearman_rho=pearson(ranks(xs), ranks(ys)),
        ))
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results_explanations")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "explanation_word_counts")
    parser.add_argument("--include-review-title", action="store_true",
                        help="Correlate explanation length with review title + body (default: body only).")
    parser.add_argument("--user-analysis", action="store_true",
                        help="Analyze user averages and Q5-only users with significance tests (NumPy).")
    parser.add_argument("--user-min-reviews", type=int, nargs="+", default=[1, 2],
                        help="Minimum distinct reviews per user; default: 1 plus a sensitivity run with 2.")
    parser.add_argument("--resamples", type=int, default=9999,
                        help="Permutations/bootstrap samples for each user test (default: 9999).")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if any(n < 1 for n in args.user_min_reviews) or args.resamples < 99:
        parser.error("Minimum reviews must be positive and resamples must be at least 99.")
    if args.user_analysis:
        try:
            from llm_explainer.tools.user_length_analysis import analyze_users, write_report
        except ImportError:
            parser.error("--user-analysis requires NumPy.")
    if not args.results_dir.is_dir():
        parser.error("Results directory does not exist: {}".format(args.results_dir))
    rows, files, duplicates, skipped = load_rows(args.results_dir, args.include_review_title)
    if not rows:
        parser.error("No explanation records found in {}".format(args.results_dir))
    summaries = summarize(rows)
    quintile_summaries = summarize(rows, GROUP_FIELDS + QUINTILE_FIELDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "word_counts.csv", rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "quintile_summary.csv", quintile_summaries)
    print("Read {} reports; retained {} explanations; omitted {} exact duplicates.".format(
        files, len(rows), duplicates))
    print("Total explanation words: {:,}; mean: {:.2f}".format(
        sum(r["explanation_words"] for r in rows),
        statistics.mean(r["explanation_words"] for r in rows)))
    print("Review length: {}; {} empty pairs excluded from correlations.".format(
        "title + body" if args.include_review_title else "body only", skipped))
    for summary in summaries:
        label = " / ".join(str(summary[k]) for k in GROUP_FIELDS)
        coefficients = ["N/A" if summary[k] is None else "{:.4f}".format(summary[k])
                        for k in ("pearson_r", "spearman_rho")]
        print("{}: n={}, mean={:.2f} words, Pearson={}, Spearman={}".format(
            label, summary["paired_observations"], summary["mean_explanation_words"],
            *coefficients))
    print("\nExplanation lengths by saved review-length quintile:")
    for summary in quintile_summaries:
        label = " / ".join(str(summary[k]) for k in GROUP_FIELDS)
        print("{} / {} ({}): n={}, mean={:.2f}, median={:.2f} words".format(
            label, summary["quintile"], summary["quintile_range_words"],
            summary["explanations"], summary["mean_explanation_words"],
            summary["median_explanation_words"]))
    print("Saved word_counts.csv, summary.csv and quintile_summary.csv in {}".format(
        args.output_dir.resolve()))
    print("Quintiles use the saved assignments and title settings; "
          "--include-review-title changes correlation counts, not quintile membership.")
    print("Item-level correlations are descriptive: users/items can repeat across seeds. "
          "Reviews are held out, not prompt history. Undefined statistics are blank in CSV.")
    if args.user_analysis:
        print("\nTesting user averages and Q5-only users...")
        users, correlations, comparisons = analyze_users(
            rows, args.user_min_reviews, args.resamples, args.random_seed)
        if not users:
            parser.error("No users with valid review/explanation pairs for user analysis.")
        write_csv(args.output_dir / "user_word_counts.csv", users)
        write_csv(args.output_dir / "user_correlations.csv", correlations)
        write_csv(args.output_dir / "q5_only_comparison.csv", comparisons)
        write_report(args.output_dir / "user_analysis.md", correlations, comparisons, args.include_review_title)
        for result in comparisons:
            label = " / ".join(str(result[k]) for k in GROUP_FIELDS)
            delta, pvalue = result["difference_vs_overall"], result["difference_p_holm"]
            print("{} / min reviews {}: Q5 users={}, difference vs overall={}, Holm p={}".format(
                label, result["min_reviews"], result["q5_only_users"],
                "N/A" if delta is None else "{:+.2f} words".format(delta),
                "N/A" if pvalue is None else "{:.4g}".format(pvalue)))
        print("Saved user_word_counts.csv, user_correlations.csv, q5_only_comparison.csv and user_analysis.md.")


if __name__ == "__main__":
    main()
