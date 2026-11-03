#!/usr/bin/env python3
"""
Orchestration pipeline script for DGCDR LLM Explanations and LLM-as-a-Judge Evaluation.

This script executes a two-phase pipeline across multiple models for a specified domain pair:
  Phase 1 (Generation & Validation):
    Runs `run_llm_validation.py` for each of the 3 explanation generator models:
      - qwen3.5:9b
      - llama3.1:8b
      - gemma4:12b
    with mandatory flags:
      --num_users 200 --no_cache_llm --no_cache_sbert --sbert_model BAAI/bge-large-en-v1.5
  
  Phase 2 (LLM-as-a-Judge Evaluation):
    Runs `llm_explainer/judge/test_geval_prompts.py` for each generated JSON report
    using the judge model:
      - qwen3.8:27b
    Results are saved under `results_judge/<domain_pair>/seed<seed>/<generator>/`.
    (Gracefully handles missing local judge model without interrupting or discarding Phase 1 data).

Supported Domain Pairs:
  - Cloth-Elec (default, currently available)
  - Elec-Cloth, Sport-Cloth, Cloth-Sport

RecBole always uses split seed 42. --seed selects one frozen sampling cohort
(default 11); --all_seeds executes cohorts 11, 22, 33, 44, 55 sequentially.
"""

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llm_explainer.config import DOMAIN_CONFIGS, DEFAULT_SETTINGS, JUDGE_DEFAULTS, GENERATOR_MODELS
from llm_explainer.result_paths import EXPLANATIONS_ROOT, generator_folder, judge_directory
from llm_explainer.cohorts import COHORT_SEEDS, DEFAULT_SPLIT_SEED, DEFAULT_INFERENCE_SEED, default_manifest_path
from llm_explainer.history_budget import budget_settings
from llm_explainer.prompt_builder import (
    DEFAULT_PROMPT_VERSION, GENERATION_PROMPT_VERSIONS, add_generation_mode_arguments,
    normalize_generation_arguments, resolve_generation_mode, generation_prompt_metadata,
    add_prompt_saving_arguments,
)
from llm_explainer.markdown_exporter import sanitize_model_name
from llm_explainer.judge.prompt_selection import add_judge_prompt_arguments, resolve_judge_prompt

# Pre-configured fixed pipeline settings as requested
DEFAULT_DOMAIN_PAIR = DEFAULT_SETTINGS["domain_pair"]
DEFAULT_GENERATOR_MODELS = list(GENERATOR_MODELS)
DEFAULT_JUDGE_MODEL = JUDGE_DEFAULTS["model"]
DEFAULT_NUM_USERS = DEFAULT_SETTINGS["num_users"]
DEFAULT_SBERT_MODEL = DEFAULT_SETTINGS["sbert_model"]
DEFAULT_OLLAMA_URL = DEFAULT_SETTINGS["ollama_url"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-phase DGCDR explanation generation and G-Eval judge pipeline."
    )
    parser.add_argument(
        "--domain_pair",
        "-d",
        type=str,
        default=DEFAULT_DOMAIN_PAIR,
        choices=list(DOMAIN_CONFIGS.keys()),
        help=f"Domain pair to evaluate (default: {DEFAULT_DOMAIN_PAIR}).",
    )
    parser.add_argument(
        "--num_users",
        "-n",
        type=int,
        default=DEFAULT_NUM_USERS,
        help=f"Number of users per validation run (default: {DEFAULT_NUM_USERS}).",
    )
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", "-s", type=int, choices=COHORT_SEEDS, default=DEFAULT_SETTINGS["seed"],
                       help="Sampling cohort seed (default: 11); RecBole split always uses 42.")
    seeds.add_argument("--all_seeds", action="store_true",
                       help="Run all sampling cohorts: 11, 22, 33, 44, 55 (200 users each by default).")
    parser.add_argument(
        "--sbert_model",
        type=str,
        default=DEFAULT_SBERT_MODEL,
        help=f"Sentence-BERT model identifier (default: {DEFAULT_SBERT_MODEL}).",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Ollama model for G-Eval Judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama endpoint URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip Phase 1 (generation) and execute only Phase 2 (judge) on latest existing results files.",
    )
    add_judge_prompt_arguments(parser)
    parser.add_argument("--instance_prompt", default=None, help="Judge instance template override")
    parser.add_argument("--judge_max_tokens", type=int, default=JUDGE_DEFAULTS["max_tokens"])
    parser.add_argument("--judge_num_ctx", type=int, default=JUDGE_DEFAULTS["num_ctx"])
    parser.add_argument("--num_ctx", type=int, default=DEFAULT_SETTINGS["num_ctx"],
                        help="Generation context and cohort token-budget limit (default: 32768).")
    parser.add_argument("--judge_top_logprobs", type=int, default=JUDGE_DEFAULTS["top_logprobs"])
    parser.add_argument("--no_weighted_scoring", action="store_true", default=JUDGE_DEFAULTS["no_weighted_scoring"], help="Disable judge token-probability scoring")
    parser.add_argument("--judge_num_gpu", type=int, default=JUDGE_DEFAULTS["num_gpu"], help="Judge GPU layers; 0 requests CPU execution")
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        help="Skip Phase 2 (judge) and execute only Phase 1 (generation).",
    )
    parser.add_argument(
        "--prompt_version", "-pv", choices=GENERATION_PROMPT_VERSIONS,
        default=DEFAULT_SETTINGS["prompt_version"],
        help="Generation template: v3 (default) or legacy v2; examples selected by --generation_mode.",
    )
    add_generation_mode_arguments(parser, DEFAULT_SETTINGS["no_few_shot"])
    add_prompt_saving_arguments(parser, DEFAULT_SETTINGS["save_prompts"])
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Execute mock generation in validation script (dry run).",
    )
    args = normalize_generation_arguments(parser.parse_args())
    if args.num_users < 1:
        parser.error("--num_users must be positive")
    try:
        budget_settings(args.prompt_version, args.num_ctx)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def find_latest_results_file(
    domain_pair: str,
    model_name: str,
    base_results_dir: str = EXPLANATIONS_ROOT,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    no_few_shot: Optional[bool] = None,
    generation_mode: Optional[str] = None,
    split_seed: Optional[int] = None,
    sampling_seed: Optional[int] = None,
    num_users: Optional[int] = None,
) -> Optional[str]:
    """
    Finds the most recent validation results JSON file for a given domain pair and model.
    Searches domain/seed/generator folders. Explicit roots also support older layouts.
    """
    if generation_mode is None and no_few_shot is not None:
        generation_mode = "zero-shot" if no_few_shot else "few-shot"
    mode = resolve_generation_mode(prompt_version, generation_mode)
    no_few_shot = mode == "zero-shot"
    clean_model = sanitize_model_name(model_name)
    clean_version = sanitize_model_name(prompt_version) if prompt_version else ""
    is_v1 = not clean_version or clean_version.lower() == "v1"
    version_suffix = "" if is_v1 else f"_{clean_version}"
    if no_few_shot:
        version_suffix += "_no_few_shot"
    elif prompt_version == "v3":
        version_suffix += "_few_shot"
    model_folder = f"{clean_model}{version_suffix}"

    candidate_dir = os.path.join(REPO_ROOT, base_results_dir, domain_pair, model_folder)
    domain_dir = os.path.join(REPO_ROOT, base_results_dir, domain_pair)
    seed_folder = f"seed{sampling_seed}" if sampling_seed is not None else "seed*"
    patterns = [
        os.path.join(domain_dir, seed_folder, generator_folder(model_name, prompt_version, mode), "*.json"),
        os.path.join(candidate_dir, "explanations_*.json"),
        os.path.join(candidate_dir, f"{domain_pair}_{model_folder}_*.json"),
        os.path.join(candidate_dir, f"*{clean_model}*.json"),
        os.path.join(REPO_ROOT, base_results_dir, domain_pair, f"*{clean_model}*.json"),
        os.path.join(REPO_ROOT, base_results_dir, domain_pair, "explanations_*.json"),
    ]

    all_files = []
    for pat in patterns:
        matched = glob.glob(pat)
        if matched:
            all_files.extend(matched)

    if not all_files:
        return None

    # Sort descending by modification time
    all_files.sort(key=os.path.getmtime, reverse=True)
    for path in all_files:
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            if report.get("model") != model_name or report.get("domain_pair") != domain_pair:
                continue
            if report.get("is_shuffled"):
                continue
            # Old single-seed reports cannot prove split/cohort identity.
            if split_seed is not None and report.get("split_seed") != split_seed:
                continue
            if sampling_seed is not None and report.get("sampling_seed") != sampling_seed:
                continue
            if num_users is not None and report.get("num_users") != num_users:
                continue
            if report.get("prompt_version") != prompt_version:
                continue
            if bool(report.get("no_few_shot", False)) != no_few_shot:
                continue
            recorded_mode = report.get("generation_mode", report.get("generation_prompt", {}).get("generation_mode"))
            if recorded_mode is not None and recorded_mode != mode:
                continue
        except (OSError, ValueError, AttributeError):
            continue
        return path
    return None


def run_command_streaming(cmd: List[str], cwd: str = REPO_ROOT) -> Tuple[int, List[str]]:
    """
    Executes a command via subprocess, streams stdout to console in real-time,
    and returns (returncode, captured_stdout_lines).
    """
    cmd_str = " ".join(cmd)
    print(f"\n[PIPELINE SUBPROCESS] Executing:\n  {cmd_str}\n")
    lines: List[str] = []

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    if process.stdout:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)

    returncode = process.wait()
    return returncode, lines


def extract_saved_json_from_logs(logs: List[str]) -> Optional[str]:
    """Parses script stdout logs to extract 'Saved Results JSON: <filepath>'."""
    for line in logs:
        m = re.search(r"Saved Results JSON:\s+(.+)", line)
        if m:
            candidate = m.group(1).strip()
            if os.path.isabs(candidate) and os.path.exists(candidate):
                return candidate
            rel_cand = os.path.join(REPO_ROOT, candidate)
            if os.path.exists(rel_cand):
                return rel_cand
            return candidate
    return None


def extract_judge_outputs_from_logs(logs: List[str]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Find the saved report in logs, then read nullable summary values from JSON."""
    json_path = None
    mean_score = None
    weighted_score = None

    for line in logs:
        # JSON output path
        m_json = re.search(r"JSON:\s+(.+)", line)
        if m_json:
            p = m_json.group(1).strip()
            if os.path.exists(p):
                json_path = p

    if json_path:
        with open(json_path, "r", encoding="utf-8") as report_file:
            report = json.load(report_file)
        mean_score = report.get("global_mean_score")
        weighted_score = report.get("global_mean_weighted_score")
    return json_path, mean_score, weighted_score


def run_seed(args):
    """Execute both phases for one frozen sampling cohort."""
    prompt_metadata = generation_prompt_metadata(args.prompt_version, args.generation_mode)
    start_time = datetime.datetime.now()

    print("\n" + "=" * 90)
    print("      DGCDR COMPLETE EXPLAINER & LLM JUDGE PIPELINE")
    print("=" * 90)
    print(f"Domain Pair:         {args.domain_pair}")
    print(f"Generator Models:    {', '.join(DEFAULT_GENERATOR_MODELS)} ({len(DEFAULT_GENERATOR_MODELS)} models)")
    print(f"Generation Prompt:   {args.prompt_version}")
    print(f"Generation Mode:     {args.generation_mode}")
    print(f"Save Prompts:        {args.save_prompts}")
    print(f"Generation Template: {prompt_metadata['template_file'] or 'legacy v2 builder'}")
    print(f"Template SHA-256:    {prompt_metadata['template_sha256'] or 'N/A'}")
    print(f"Number of Users:     {args.num_users}")
    print(f"RecBole Split Seed:  {DEFAULT_SPLIT_SEED}")
    print(f"Sampling Seed:       {args.seed}")
    print(f"Inference Seed:      {DEFAULT_INFERENCE_SEED}")
    print(f"Sentence-BERT Model: {args.sbert_model}")
    print(f"Validation Caches:   LLM={'disabled' if DEFAULT_SETTINGS['no_cache_llm'] else 'enabled'}, "
          f"SBERT={'disabled' if DEFAULT_SETTINGS['no_cache_sbert'] else 'enabled'}")
    print(f"Few-Shot Examples:   {'Disabled (--no_few_shot)' if args.no_few_shot else 'Enabled'}")
    print(f"Judge Model:         {args.judge_model}")
    print(f"Judge Prompt:        {resolve_judge_prompt(args)[1]}")
    print(f"Skip Generation:     {args.skip_generation}")
    print(f"Skip Judge:          {args.skip_judge}")
    print(f"Dry Run Mode:        {args.dry_run}")
    print(f"Ollama Server URL:   {args.ollama_url}")
    print(f"Start Timestamp:     {start_time.isoformat()}")
    print("=" * 90 + "\n")

    # Track results across runs
    run_records: Dict[str, Dict[str, Any]] = {}
    for m in DEFAULT_GENERATOR_MODELS:
        run_records[m] = {
            "model": m,
            "generation_mode": args.generation_mode,
            "split_seed": DEFAULT_SPLIT_SEED,
            "sampling_seed": args.seed,
            "gen_status": "Pending",
            "gen_json_file": None,
            "judge_status": "Pending",
            "judge_json_file": None,
            "judge_score": None,
            "judge_weighted": None,
            "error": None,
        }

    # =========================================================================
    # FASE 1: Generazione & Validazione Metriche (run_llm_validation.py)
    # =========================================================================
    if not args.skip_generation:
        print("\n" + "#" * 90)
        print("  >>> AVVIO FASE 1: GENERAZIONE SPIEGAZIONI & VALIDAZIONE (3 LLM)")
        print("#" * 90 + "\n")

        for idx, gen_model in enumerate(DEFAULT_GENERATOR_MODELS, 1):
            print("\n" + "-" * 85)
            print(f"  [Fase 1 - Modello {idx}/{len(DEFAULT_GENERATOR_MODELS)}] Generatore: {gen_model}")
            print("-" * 85)

            cmd = [
                sys.executable,
                os.path.join(REPO_ROOT, "run_llm_validation.py"),
                "--domain_pair", args.domain_pair,
                "--model", gen_model,
                "--prompt_version", args.prompt_version,
                "--generation_mode", args.generation_mode,
                "--num_users", str(args.num_users),
                "--num_ctx", str(args.num_ctx),
                "--split_seed", str(DEFAULT_SPLIT_SEED),
                "--seed", str(args.seed),
                "--inference_seed", str(DEFAULT_INFERENCE_SEED),
                "--cohort_manifest", default_manifest_path(os.path.join(REPO_ROOT, EXPLANATIONS_ROOT),
                    args.domain_pair, args.num_users),
                "--sbert_model", args.sbert_model,
                "--ollama_url", args.ollama_url,
            ]
            if args.dry_run:
                cmd.append("--dry_run")
            for kind in ("llm", "sbert"):
                cmd.append(f"--no_cache_{kind}" if DEFAULT_SETTINGS[f"no_cache_{kind}"] else f"--cache_{kind}")
            cmd.append("--save_prompts" if args.save_prompts else "--no_save_prompts")

            retcode, logs = run_command_streaming(cmd)

            if retcode == 0:
                saved_json = extract_saved_json_from_logs(logs)
                if not saved_json:
                    saved_json = find_latest_results_file(args.domain_pair, gen_model, prompt_version=args.prompt_version, no_few_shot=args.no_few_shot,
                        split_seed=DEFAULT_SPLIT_SEED, sampling_seed=args.seed, num_users=args.num_users)
                if not saved_json:
                    run_records[gen_model]["gen_status"] = "Failed (No source file)"
                    run_records[gen_model]["error"] = "Generation exited without a matching results JSON"
                    continue
                run_records[gen_model]["gen_status"] = "Completed"
                run_records[gen_model]["gen_json_file"] = saved_json
                print(f"\n[OK] Generazione completata con successo per {gen_model}!")
                print(f"     File salvato: {saved_json}")
            else:
                run_records[gen_model]["gen_status"] = f"Failed (exit code {retcode})"
                run_records[gen_model]["error"] = f"Generation exit code {retcode}"
                print(f"\n[ERRORE] Generazione fallita per {gen_model} con codice di uscita {retcode}.")
    else:
        print("\n>>> FASE 1 SALTATA (--skip_generation). Ricerca dei risultati esistenti...")
        for gen_model in DEFAULT_GENERATOR_MODELS:
            latest_file = find_latest_results_file(args.domain_pair, gen_model, prompt_version=args.prompt_version, no_few_shot=args.no_few_shot,
                split_seed=DEFAULT_SPLIT_SEED, sampling_seed=args.seed, num_users=args.num_users)
            if latest_file:
                run_records[gen_model]["gen_status"] = "Pre-existing"
                run_records[gen_model]["gen_json_file"] = latest_file
                print(f"  • {gen_model}: Trovato {latest_file}")
            else:
                run_records[gen_model]["gen_status"] = "Not Found"
                print(f"  • {gen_model}: [ATTENZIONE] Nessun file di risultato trovato in {EXPLANATIONS_ROOT}/{args.domain_pair}/!")

    # =========================================================================
    # FASE 2: Valutazione LLM-as-a-Judge (test_geval_prompts.py)
    # =========================================================================
    if not args.skip_judge:
        print("\n\n" + "#" * 90)
        print(f"  >>> AVVIO FASE 2: VALUTAZIONE LLM-AS-A-JUDGE (Judge: {args.judge_model})")
        print("#" * 90 + "\n")

        for idx, gen_model in enumerate(DEFAULT_GENERATOR_MODELS, 1):
            rec = run_records[gen_model]
            target_json = rec.get("gen_json_file")

            print("\n" + "-" * 85)
            print(f"  [Fase 2 - Modello {idx}/{len(DEFAULT_GENERATOR_MODELS)}] Valutazione spiegazioni di: {gen_model}")
            print(f"  Target File: {target_json}")
            print("-" * 85)

            if not target_json or not os.path.exists(target_json):
                print(f"[SKIP] Impossibile eseguire il Judge: file risultati non presente per {gen_model}.")
                rec["judge_status"] = "Skipped (No source file)"
                continue

            judge_output_dir = judge_directory(REPO_ROOT, dict(domain_pair=args.domain_pair,
                sampling_seed=args.seed, model=gen_model, prompt_version=args.prompt_version,
                generation_mode=args.generation_mode))
            os.makedirs(judge_output_dir, exist_ok=True)

            cmd_judge = [
                sys.executable,
                os.path.join(REPO_ROOT, "llm_explainer", "judge", "test_geval_prompts.py"),
                "--model", args.judge_model,
                "--results_file", target_json,
                "--num_users", str(args.num_users),
                "--seed", str(DEFAULT_INFERENCE_SEED),
                "--output_dir", judge_output_dir,
                "--ollama_url", args.ollama_url,
                "--max_tokens", str(args.judge_max_tokens),
                "--num_ctx", str(args.judge_num_ctx),
                "--top_logprobs", str(args.judge_top_logprobs),
            ]
            if args.no_weighted_scoring:
                cmd_judge.append("--no_weighted_scoring")
            if args.judge_num_gpu is not None:
                cmd_judge.extend(["--num_gpu", str(args.judge_num_gpu)])
            if args.system_prompt:
                cmd_judge.extend(["--system_prompt", args.system_prompt])
            else:
                cmd_judge.extend(["--judge_prompt", args.judge_prompt])
            if args.instance_prompt:
                cmd_judge.extend(["--instance_prompt", args.instance_prompt])

            try:
                retcode_judge, logs_judge = run_command_streaming(cmd_judge)
                j_path, m_score, w_score = extract_judge_outputs_from_logs(logs_judge)
                rec["judge_json_file"] = j_path
                if retcode_judge == 0:
                    if not j_path:
                        raise RuntimeError("Judge exited without a saved JSON report")
                    rec["judge_status"] = "Completed"
                    with open(j_path, "r", encoding="utf-8") as report_file:
                        counts = json.load(report_file).get("judgment_counts", {})
                    if counts.get("failed_count", 0):
                        rec["judge_status"] = "Completed with errors"
                    elif m_score is None:
                        rec["judge_status"] = "Completed (unassessable)"
                    rec["judge_score"] = m_score
                    rec["judge_weighted"] = w_score
                    print(f"\n[OK] Valutazione Judge completata per {gen_model}!")
                    if m_score is not None:
                        print(f"     Alignment Score: {m_score}/5 | Weighted Score: {'N/A' if w_score is None else w_score}")
                else:
                    # Model might be absent (e.g. qwen3.8:27b on current machine)
                    rec["judge_status"] = f"Failed (exit code {retcode_judge})"
                    rec["error"] = f"Judge exit code {retcode_judge}"
                    print(
                        f"\n[AVVISO/ERRORE] Valutazione Judge non completata per {gen_model} "
                        f"(codice {retcode_judge}). Nota: il modello '{args.judge_model}' potrebbe non essere "
                        f"installato o disponibile localmente su Ollama. I dati generati nella Fase 1 sono preservati."
                    )
            except Exception as e:
                rec["judge_status"] = f"Exception: {type(e).__name__}"
                rec["error"] = str(e)
                print(f"\n[ECCEZIONE] Errore durante l'invocazione del Judge per {gen_model}: {e}")
    else:
        print("\n>>> FASE 2 SALTATA (--skip_judge).")
        for gen_model in DEFAULT_GENERATOR_MODELS:
            run_records[gen_model]["judge_status"] = "Skipped (--skip_judge)"

    # =========================================================================
    # RIEPILOGO FINALE CONSOLIDATO
    # =========================================================================
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("\n\n" + "=" * 115)
    print("                        RIEPILOGO CONSOLIDATO PIPELINE DGCDR")
    print("=" * 115)
    print(f"Coppia di Dominio:        {args.domain_pair}")
    print(f"Generation Prompt:       {args.prompt_version} / {args.generation_mode}")
    print(f"Save Generation Prompts: {args.save_prompts}")
    print(f"Utenti Valutati per Run:  {args.num_users}")
    print(f"RecBole Split Seed:       {DEFAULT_SPLIT_SEED}")
    print(f"Sampling Seed:            {args.seed}")
    print(f"Modello Judge:            {args.judge_model}")
    print(f"Judge Prompt:             {resolve_judge_prompt(args)[1]}")
    print(f"Modello SBERT:            {args.sbert_model}")
    print(f"Durata Totale Pipeline:   {str(duration).split('.')[0]}")
    print("-" * 115)
    print(
        f"{'Modello Generatore':<18} | {'Fase 1 (Generazione)':<20} | {'Fase 2 (Judge 27b)':<22} | "
        f"{'Score':<8} | {'Weighted':<8}"
    )
    print("-" * 115)
    for gen_model in DEFAULT_GENERATOR_MODELS:
        rec = run_records[gen_model]
        score_str = f"{rec['judge_score']:.3f}" if rec['judge_score'] is not None else "-"
        weight_str = f"{rec['judge_weighted']:.3f}" if rec['judge_weighted'] is not None else "-"
        print(
            f"{gen_model:<18} | {rec['gen_status']:<20} | {rec['judge_status']:<22} | "
            f"{score_str:<8} | {weight_str:<8}"
        )
    print("-" * 115)
    print("Dettaglio File Generati:")
    for gen_model in DEFAULT_GENERATOR_MODELS:
        rec = run_records[gen_model]
        print(f"  [{gen_model}]")
        print(f"    • Spiegazioni JSON: {rec['gen_json_file'] or 'N/A'}")
        print(f"    • Judge JSON:       {rec['judge_json_file'] or 'N/A'}")
    print("=" * 115 + "\n")
    return list(run_records.values())


def main():
    args = parse_args()
    sampling_seeds = COHORT_SEEDS if args.all_seeds else (args.seed,)
    records = []
    for seed in sampling_seeds:
        # Keep all other CLI choices identical across sampling cohorts.
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        records.extend(run_seed(seed_args))
    if args.all_seeds:
        print("\nALL SAMPLING SEEDS — RecBole split seed: 42")
        print(f"{'Seed':<6} | {'Generator':<18} | {'Generation':<28} | Judge")
        for rec in records:
            print(f"{rec['sampling_seed']:<6} | {rec['model']:<18} | "
                  f"{rec['gen_status']:<28} | {rec['judge_status']}")
    return records


if __name__ == "__main__":
    main()
