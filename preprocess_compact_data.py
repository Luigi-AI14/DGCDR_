"""
CLI script to preprocess compact metadata and reviews for DGCDR.
Filters 50GB+ Amazon files into lightweight, instant-loading compact caches in cache/.

Usage:
    python preprocess_compact_data.py --domain_pair Cloth-Elec
    python preprocess_compact_data.py --all
"""

import argparse
import sys
from llm_explainer.compact_builder import build_compact_dataset
from llm_explainer.config import DOMAIN_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and build compact metadata & reviews JSON/PKL for DGCDR."
    )
    parser.add_argument(
        "--domain_pair",
        type=str,
        default="Cloth-Elec",
        choices=list(DOMAIN_CONFIGS.keys()) + ["all"],
        help="Domain pair to preprocess (default: Cloth-Elec, or 'all').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.domain_pair == "all":
        pairs = list(DOMAIN_CONFIGS.keys())
    else:
        pairs = [args.domain_pair]

    for pair in pairs:
        print(f"\nProcessing {pair}...")
        build_compact_dataset(domain_pair=pair)

    print("\nAll requested compact datasets have been created successfully!")


if __name__ == "__main__":
    main()
