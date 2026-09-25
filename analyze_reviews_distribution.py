"""
Analysis script for review word count distributions in DGCDR compact cache.
Evaluates Target and Source domain reviews, descriptive statistics, percentiles,
bucket distributions, and effect of Title + Text concatenation.

Usage:
    python analyze_reviews_distribution.py
    python analyze_reviews_distribution.py --domain_pair Cloth-Elec --min_rating 4.0 --min_words 5
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
import numpy as np


def count_words(text: str) -> int:
    """Counts alphanumeric words, ignoring punctuation."""
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def compute_bucket_stats(lengths, buckets):
    total = len(lengths)
    res = []
    for name, low, high in buckets:
        cnt = sum(1 for l in lengths if low <= l <= high)
        pct = (cnt / total * 100) if total > 0 else 0.0
        res.append((name, cnt, pct))
    return res


def analyze_domain(reviews_dict, domain_label, min_rating=4.0, min_words=1):
    print("\n" + "=" * 70)
    print(f" ANALYZING: {domain_label} (Rating >= {min_rating}, Words >= {min_words})")
    print("=" * 70)

    lengths_text = []
    lengths_full = []
    sample_reviews = defaultdict(list)

    for k, v in reviews_dict.items():
        rating = v.get("rating", 0)
        if min_rating is not None and rating < min_rating:
            continue

        txt = v.get("text", "")
        title = v.get("title", "")

        w_txt = count_words(txt)
        w_full = count_words((title + " " + txt).strip())

        if w_txt >= min_words:
            lengths_text.append(w_txt)
            lengths_full.append(w_full)
            if w_txt <= 10 and len(sample_reviews[w_txt]) < 3:
                sample_reviews[w_txt].append(txt.replace("\n", " ").strip())

    if not lengths_text:
        print("No reviews found matching the specified constraints.")
        return

    arr = np.array(lengths_text)
    arr_full = np.array(lengths_full)
    total = len(arr)

    print(f"Total evaluated reviews: {total:,}")
    print(f"--- Review Body Text Stats ---")
    print(f"  Min words:    {np.min(arr)}")
    print(f"  Max words:    {np.max(arr)}")
    print(f"  Mean words:   {np.mean(arr):.2f}")
    print(f"  Median words: {np.median(arr):.1f}")
    print(f"  Std Dev:      {np.std(arr):.2f}")
    print(f"  Percentiles:")
    for p in [5, 10, 20, 25, 33, 50, 66, 75, 80, 90, 95, 99]:
        print(f"    P{p:02d}: {np.percentile(arr, p):.1f} words")

    # 1. Standard granular buckets
    std_buckets = [
        ("Ultra-Short (1 - 4 words)", 1, 4),
        ("Short (5 - 9 words)", 5, 9),
        ("Medium (10 - 25 words)", 10, 25),
        ("Long (26 - 50 words)", 26, 50),
        ("Detailed (51 - 100 words)", 51, 100),
        ("Very Detailed (> 100 words)", 101, 1_000_000),
    ]

    print(f"\n--- Standard Granular Buckets ---")
    for name, cnt, pct in compute_bucket_stats(arr, std_buckets):
        print(f"  {name:30s}: {cnt:7,d} ({pct:5.2f}%)")

    # 2. Equal Tertiles (33.3% each) for reviews >= 5 words
    t33 = np.percentile(arr, 33.333)
    t66 = np.percentile(arr, 66.667)
    b1_hi = int(round(t33))
    b2_hi = int(round(t66))

    tertile_buckets = [
        (f"Tertile 1: Low ({min_words} - {b1_hi} words)", min_words, b1_hi),
        (f"Tertile 2: Medium ({b1_hi + 1} - {b2_hi} words)", b1_hi + 1, b2_hi),
        (f"Tertile 3: High (> {b2_hi} words)", b2_hi + 1, 1_000_000),
    ]
    print(f"\n--- Equal Tertiles (33% each) ---")
    for name, cnt, pct in compute_bucket_stats(arr, tertile_buckets):
        print(f"  {name:35s}: {cnt:7,d} ({pct:5.2f}%)")

    # 3. Semantic / Rounded 3-Bucket Proposal
    rounded_buckets = [
        (f"Bucket 1: Short / Synthetic ({min_words} - 25 words)", min_words, 25),
        ("Bucket 2: Medium / Contextual (26 - 70 words)", 26, 70),
        ("Bucket 3: Detailed / Analytical (> 70 words)", 71, 1_000_000),
    ]
    print(f"\n--- Proposal: Rounded Semantic 3-Buckets (~33% balanced) ---")
    for name, cnt, pct in compute_bucket_stats(arr, rounded_buckets):
        print(f"  {name:45s}: {cnt:7,d} ({pct:5.2f}%)")

    # 4. Effect of Title + Text
    print(f"\n--- Effect of Title + Text ---")
    print(f"  Mean words:   {np.mean(arr_full):.2f}")
    print(f"  Median words: {np.median(arr_full):.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze review word count distributions in DGCDR compact cache.")
    parser.add_argument("--domain_pair", type=str, default="Cloth-Elec", help="Domain pair name.")
    parser.add_argument("--min_rating", type=float, default=4.0, help="Minimum rating threshold.")
    parser.add_argument("--min_words", type=int, default=1, help="Minimum review words (set to 5 to filter 1-4).")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Cache directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = os.path.join(args.cache_dir, f"compact_{args.domain_pair}.json")

    if not os.path.exists(json_path):
        print(f"Error: Compact file not found at {json_path}")
        print("Run 'python preprocess_compact_data.py' first.")
        return

    print(f"Loading {json_path}...")
    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded compact cache in {time.time() - t0:.2f}s!")

    tgt_revs = data.get("target_reviews", {})
    src_revs = data.get("source_reviews", {})

    # 1. Target Domain (Electronics)
    analyze_domain(tgt_revs, f"Target Domain: {data.get('target_name', 'Target')}", args.min_rating, args.min_words)

    # 2. Source Domain (Clothing)
    analyze_domain(src_revs, f"Source Domain: {data.get('source_name', 'Source')}", args.min_rating, args.min_words)


if __name__ == "__main__":
    main()
