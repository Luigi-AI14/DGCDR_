import argparse
import subprocess
import sys
import time

DEFAULT_SEEDS = [2022, 2023, 42, 24, 1]


def main():
    parser = argparse.ArgumentParser(
        description="Automate execution of run_recbole_cdr.py across multiple random seeds."
    )
    parser.add_argument(
        "--seeds",
        "-s",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"List of random seeds to execute (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Optional model name to forward to run_recbole_cdr.py (e.g. DGCDR, BiTGCF, DCCDR, DTCDR)",
    )

    args, unknown_args = parser.parse_known_args()
    total_seeds = len(args.seeds)

    print("=" * 60)
    print("Multi-Seed Execution Runner for RecBole CDR")
    print(f"Seeds to execute ({total_seeds}): {args.seeds}")
    if args.model:
        print(f"Model override: {args.model}")
    if unknown_args:
        print(f"Additional arguments forwarded: {' '.join(unknown_args)}")
    print("=" * 60)

    start_total_time = time.time()

    for idx, seed in enumerate(args.seeds, start=1):
        print(f"\n[{idx}/{total_seeds}] >>> Starting run with seed: {seed} <<<\n")
        
        cmd = [sys.executable, "run_recbole_cdr.py", f"--seed={seed}"]
        if args.model:
            cmd.extend(["--model", args.model])
        if unknown_args:
            cmd.extend(unknown_args)

        run_start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
            run_elapsed = time.time() - run_start_time
            print(f"\n[OK] Completed seed {seed} in {run_elapsed:.2f}s ({run_elapsed/60:.2f}m)")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Execution failed for seed {seed} with return code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Process manually stopped by user.")
            sys.exit(1)

    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 60)
    print(f"All {total_seeds} runs finished successfully!")
    print(f"Total time elapsed: {total_elapsed:.2f}s ({total_elapsed/60:.2f}m)")
    print("=" * 60)


if __name__ == "__main__":
    main()
