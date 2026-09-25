"""
CLI script to compute, display, and export Scenario A Quintile partitions
for any Amazon domain pair in DGCDR.

Usage:
    python compute_quintiles.py --domain_pair Cloth-Elec
    python compute_quintiles.py --domain_pair Cloth-Sport
    python compute_quintiles.py --all
    python compute_quintiles.py --domain_pair Cloth-Elec --use_title
"""

import argparse
import sys
import os

from llm_explainer.config import DOMAIN_CONFIGS
from llm_explainer.quintiles import QuintileManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and export statistical 5-Quintile partitions (Scenario A) for DGCDR datasets."
    )
    parser.add_argument(
        "--domain_pair",
        "-d",
        type=str,
        default="Cloth-Elec",
        choices=list(DOMAIN_CONFIGS.keys()) + ["all"],
        help="Domain pair name (default: Cloth-Elec, or 'all').",
    )
    parser.add_argument(
        "--min_words",
        "-w",
        type=int,
        default=5,
        help="Minimum words threshold to filter uninformative reviews (default: 5).",
    )
    parser.add_argument(
        "--min_rating",
        "-r",
        type=float,
        default=4.0,
        help="Minimum review rating threshold (default: 4.0).",
    )
    parser.add_argument(
        "--use_title",
        action="store_true",
        help="Whether to compute quintiles on concatenated 'title + text'.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputing even if cached quintile JSON exists.",
    )
    return parser.parse_args()


def process_pair(pair_name: str, min_words: int, min_rating: float, use_title: bool, recompute: bool):
    print("\n" + "=" * 80)
    print(f" COMPUTING QUINTILES: {pair_name}")
    print("=" * 80)

    try:
        qm = QuintileManager(
            domain_pair=pair_name,
            min_words=min_words,
            min_rating=min_rating,
            use_title=use_title,
        )
        if recompute:
            qm.compute_and_save_quintiles()

        print(qm.get_cutoffs_summary())
        print(f"Saved cutoffs schema to: {qm.cutoffs_file}\n")
    except Exception as e:
        print(f"Could not compute quintiles for {pair_name}: {e}")
        print("Ensure 'python preprocess_compact_data.py --domain_pair <pair>' has been executed.")


def main():
    args = parse_args()

    if args.domain_pair == "all":
        pairs = list(DOMAIN_CONFIGS.keys())
    else:
        pairs = [args.domain_pair]

    for p in pairs:
        process_pair(
            pair_name=p,
            min_words=args.min_words,
            min_rating=args.min_rating,
            use_title=args.use_title,
            recompute=args.recompute,
        )


if __name__ == "__main__":
    main()
