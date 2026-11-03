"""Shared directory conventions for explainer and judge artifacts."""

import os
from pathlib import Path

EXPLANATIONS_ROOT = "results_explanations"
JUDGE_ROOT = "results_judge"
SHUFFLED_JUDGE_ROOT = "results_judge_shuffled"
LEGACY_ROOT = "legacy_results"


def generator_folder(model, prompt_version, generation_mode):
    from llm_explainer.markdown_exporter import sanitize_model_name
    return f"{sanitize_model_name(model)}_{prompt_version}_{generation_mode}"


def explanation_directory(results_dir, domain_pair, sampling_seed, model, prompt_version, generation_mode):
    return os.path.join(results_dir, domain_pair, f"seed{sampling_seed}",
                        generator_folder(model, prompt_version, generation_mode))


def judge_directory(repo_root, source, shuffled=False):
    directory = os.path.join(repo_root, SHUFFLED_JUDGE_ROOT if shuffled else JUDGE_ROOT,
                             source.get("domain_pair", "Cloth-Elec"))
    if source.get("sampling_seed") is not None:
        directory = os.path.join(directory, f"seed{source['sampling_seed']}")
    model = source.get("generator_model") or source.get("model")
    version = source.get("prompt_version")
    if model and version:
        mode = source.get("generation_mode") or ("zero-shot" if source.get("no_few_shot") else "few-shot")
        directory = os.path.join(directory, generator_folder(model, version, mode))
    return directory


def resolve_result_reference(path, repo_root):
    """Resolve an explicit old reference after archiving, without scanning archives."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(repo_root) / candidate
    candidate = candidate.resolve()
    if candidate.exists():
        return str(candidate)
    try:
        relative = candidate.relative_to(Path(repo_root).resolve())
    except ValueError:
        return str(candidate)
    if relative.parts and relative.parts[0] in ("results", JUDGE_ROOT, "saved_prompts"):
        archived = Path(repo_root) / LEGACY_ROOT / relative
        if archived.exists():
            return str(archived)
    return str(candidate)
