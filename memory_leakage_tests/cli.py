"""Command-line entry point for prepare, preview, run, and report."""

import argparse
import logging
import re
import sys
from pathlib import Path

from .data import prepare
from .runner import DEFAULT_MODELS, DEFAULT_MODES, preview, run
from .scoring import report


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    root = argparse.ArgumentParser(description="Audit ID-based item/review recovery on frozen DGCDR held-out cohorts.")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("prepare", "preview", "run", "report"):
        sub = commands.add_parser(command)
        sub.add_argument("--output-dir", type=Path, required=True,
                         help="Prepared audit directory; use a new one to change cohorts/split.")
        if command == "prepare":
            sub.add_argument("--domain-pair", choices=["Cloth-Elec", "Elec-Cloth", "Cloth-Sport", "Sport-Cloth"], default="Cloth-Elec")
            sub.add_argument("--split-seed", type=int, default=42)
            sub.add_argument("--cohort-seeds", nargs=5, type=int, default=[11, 22, 33, 44, 55])
            sub.add_argument("--num-users", type=positive, default=200, help="Users per cohort (five disjoint cohorts).")
            sub.add_argument("--few-shot-count", type=positive, default=3)
            sub.add_argument("--demo-seed", type=int, default=314159)
            sub.add_argument("--negative-controls", type=positive, default=3, help="Synthetic-ID controls per probe, seed and mode.")
            sub.add_argument("--min-rating", type=float, default=4.0)
            sub.add_argument("--min-review-words", type=positive, default=5)
            sub.add_argument("--cache-dir", type=Path, default=None)
            sub.add_argument("--dataset", type=Path, help="Import a frozen dataset.json instead of loading RecBole; its split/filters are authoritative.")
        if command in ("preview", "run"):
            sub.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
        if command in ("run", "report"):
            sub.add_argument("--run-name", default="default", help="Name within OUTPUT/runs; use a new name for changed settings.")
        if command == "run":
            sub.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
            sub.add_argument("--ollama-url", default="http://localhost:11434")
            sub.add_argument("--inference-seed", type=int, default=42)
            sub.add_argument("--num-ctx", type=positive, default=8192)
            sub.add_argument("--max-tokens", type=positive, default=4096)
            sub.add_argument("--timeout", type=positive, default=300)
            sub.add_argument("--retries", type=int, default=1, help="Transport retries only; no response repair.")
            sub.add_argument("--thinking", choices=["off", "on", "default"], default="off")
            sub.add_argument("--retry-errors", action="store_true", help="Retry prior failed tasks; successful/abstained tasks stay frozen.")
            sub.add_argument("--permutations", type=positive, default=499)
            sub.add_argument("--bootstrap-samples", type=positive, default=499)
    return root


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if hasattr(args, "run_name") and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        arg_parser.error("--run-name must be a simple directory name")
    if hasattr(args, "modes") and len(set(args.modes)) != len(args.modes):
        arg_parser.error("--modes must be distinct")
    try:
        if args.command == "prepare":
            if not 1 <= args.min_rating <= 5:
                arg_parser.error("--min-rating must be between 1 and 5")
            prepare(args.output_dir, args.cohort_seeds, args.num_users, args.few_shot_count,
                    args.demo_seed, args.negative_controls, args.dataset,
                    domain_pair=args.domain_pair, split_seed=args.split_seed,
                    min_rating=args.min_rating, min_words=args.min_review_words, cache_dir=args.cache_dir)
        elif args.command == "preview":
            result = preview(args.output_dir, args.modes)
            for key, count in result["requests_per_model"].items():
                print("{}: {}".format(key, count))
        elif args.command == "run":
            if args.retries < 0 or len(set(args.models)) != len(args.models):
                arg_parser.error("--retries must be nonnegative and --models must be distinct")
            if args.permutations < 2 or args.bootstrap_samples < 2:
                arg_parser.error("Use at least two permutations/bootstrap samples (499+ recommended)")
            run(args.output_dir, args.models, args.modes, args.ollama_url, args.inference_seed,
                args.num_ctx, args.max_tokens, args.timeout, args.retries, args.thinking,
                args.run_name, args.retry_errors, args.permutations, args.bootstrap_samples)
        else:
            report(args.output_dir, args.run_name)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        logging.error("%s", error)
        raise SystemExit(1)
