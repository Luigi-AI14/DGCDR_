"""Markdown presentation of existing judge summaries; no scoring or JSON changes."""

from typing import Any, Dict, List, Optional


def format_outcome_count(count: Optional[int], attempted: Optional[int]) -> str:
    """Keep exact counts visible even when a percentage is unavailable."""
    if count is None:
        return "N/A"
    percentage = f"{100 * count / attempted:.2f}%" if attempted else "N/A"
    return f"{count} ({percentage})"


def score_distribution_markdown(report: Dict[str, Any]) -> List[str]:
    """Render saved Q1-Q5 counts, preserving missing historical data as N/A."""
    by_q = report.get("quintile_metrics", {}).get("by_quintile", {})
    counts = report.get("judgment_counts", {})
    quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    lines = [
        "| Quintile | Attempted | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 | Null | Failed |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for qid in quintiles + ["Overall"]:
        if qid == "Overall":
            row_counts = counts
            distribution = {}
            for score in range(1, 6):
                values = [by_q.get(q, {}).get("score_distribution", {}).get(str(score))
                          for q in quintiles]
                distribution[str(score)] = sum(values) if all(v is not None for v in values) else None
        else:
            row_counts = by_q.get(qid, {})
            distribution = row_counts.get("score_distribution", {})

        attempted = row_counts.get("attempted_count")
        outcomes = [distribution.get(str(score)) for score in range(1, 6)]
        outcomes += [row_counts.get("insufficient_evidence_count"), row_counts.get("failed_count")]
        cells = [f"**{qid}**", str(attempted) if attempted is not None else "N/A"]
        cells.extend(format_outcome_count(count, attempted) for count in outcomes)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "Percentages use all attempted items in each row, including nulls and failures. "
        "Rounded percentages may not sum to exactly 100%.",
        "",
        "**Null** means a valid judge null (insufficient evidence). **Failed** means a technical failure. "
        "Filtered reviews and missing-input items are excluded from attempted items. "
        "N/A indicates an empty denominator or information unavailable in the source report; "
        "null counts are not inferred from historical scores without status information.",
        "",
    ])
    return lines
