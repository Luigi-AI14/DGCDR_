"""Frozen user cohorts shared by all explanation models and sampling runs.

Like the leakage audit, sample each cohort from the remaining sorted eligible
pool. Generation demonstrations are synthetic, so audit-specific demonstration
exclusions do not apply here.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile


DEFAULT_SPLIT_SEED = 42
DEFAULT_INFERENCE_SEED = 42
COHORT_SEEDS = (11, 22, 33, 44, 55)
logger = logging.getLogger(__name__)


def default_manifest_path(results_dir, domain_pair, num_users):
    """Use the same domain/user-count snapshot in standalone and pipeline runs."""
    return os.path.join(results_dir, "cohorts", domain_pair, f"users{num_users}.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def fingerprint_splits(source, history, test):
    return {name: digest({uid: sorted(items) for uid, items in split.items()})
            for name, split in (("source_all", source), ("target_history", history),
                                ("target_test", test))}


def cohort_settings(domain_pair, num_users, split_seed, min_rating, min_words,
                    prompt_version="v3", num_ctx=None):
    from llm_explainer.history_budget import budget_settings
    if num_users < 1:
        raise ValueError("num_users must be positive")
    return dict(domain_pair=domain_pair, num_users=num_users, split_seed=split_seed,
                min_rating=min_rating, min_review_words=min_words,
                cohort_seeds=list(COHORT_SEEDS),
                prompt_budget=budget_settings(prompt_version, num_ctx))


def sample_with_replacements(users, num_users, seed, check_user):
    """Keep the initial sample's eligible users; continue the same RNG for replacements."""
    rng = random.Random(seed)
    initial = rng.sample(users, num_users)
    attempted = {user["user_id"] for user in initial}
    remaining = [user for user in users if user["user_id"] not in attempted]
    selected, excluded, replacements, checks = [], [], [], []
    pending = initial
    while pending:
        for user in pending:
            result = check_user(user)
            checks.append(result)
            if result["eligible"]:
                selected.append(user)
            else:
                excluded.append(result)
                logger.info("Excluded user %s: prompt exceeds the token budget", user["user_id"])
        needed = num_users - len(selected)
        if not needed:
            break
        if len(remaining) < needed:
            raise ValueError(f"Not enough users fit the prompt budget for seed {seed}: "
                             f"accepted {len(selected)} of {num_users}, excluded {len(excluded)}. "
                             "Reduce --num_users or increase --num_ctx.")
        pending = rng.sample(remaining, needed)
        ids = {user["user_id"] for user in pending}
        remaining = [user for user in remaining if user["user_id"] not in ids]
        replacements.extend(user["user_id"] for user in pending)
    audit = dict(initial_user_ids=[user["user_id"] for user in initial],
                 replacement_user_ids=replacements, excluded_users=excluded, checks=checks,
                 final_user_ids=sorted(user["user_id"] for user in selected))
    selected.sort(key=lambda user: user["user_id"])
    return selected, audit


def make_manifest(users, settings, split_fingerprints, budget=None):
    remaining = sorted(users, key=lambda user: user["user_id"])
    if len({user["user_id"] for user in remaining}) != len(remaining):
        raise ValueError("Eligible pool contains duplicate user IDs")
    required = len(COHORT_SEEDS) * settings["num_users"]
    if len(remaining) < required:
        raise ValueError(f"Need {required} eligible users for five disjoint cohorts; "
                         f"only {len(remaining)} available. Reduce --num_users.")
    if budget is None:
        from llm_explainer.history_budget import HistoryBudget
        budget = HistoryBudget(settings["prompt_budget"])
    cohorts, rejected = [], set()
    for seed in COHORT_SEEDS:
        if len(remaining) < settings["num_users"]:
            raise ValueError(f"Not enough users fit the prompt budget for seed {seed}; reduce --num_users.")
        selected, audit = sample_with_replacements(remaining, settings["num_users"], seed, budget.check_user)
        rejected.update(result["user_id"] for result in audit["excluded_users"])
        chosen = {user["user_id"] for user in selected}
        remaining = [user for user in remaining if user["user_id"] not in chosen | rejected]
        cohorts.append(dict(seed=seed, users=selected, selection_audit=audit))
        logger.info("Cohort seed %s: kept %s users, excluded %s oversized users", seed,
                    len(selected), len(audit["excluded_users"]))
    manifest = dict(schema_version=2, settings=settings, tokenizer_metadata=budget.metadata,
                    eligible_pool_size=len(users), split_fingerprints=split_fingerprints, cohorts=cohorts)
    manifest["sha256"] = digest(manifest)
    return manifest


def validate_manifest(manifest, settings):
    if manifest.get("schema_version") != 2 or manifest.get("settings") != settings:
        raise ValueError("Cohort manifest settings differ or it predates token-budget filtering. "
                         "Archive the old manifest or use a new --cohort_manifest path.")
    if manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Cohort manifest checksum mismatch")
    cohorts = manifest["cohorts"]
    if [cohort["seed"] for cohort in cohorts] != list(COHORT_SEEDS):
        raise ValueError("Cohort manifest must contain all five ordered sampling seeds")
    seen = set()
    for cohort in cohorts:
        ids = [user["user_id"] for user in cohort["users"]]
        if len(ids) != settings["num_users"] or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Cohorts must have the requested size and disjoint user IDs")
        seen.update(ids)
        audit = cohort["selection_audit"]
        accepted = [result["user_id"] for result in audit["checks"] if result["eligible"]]
        if sorted(accepted) != sorted(ids) or audit["final_user_ids"] != sorted(ids):
            raise ValueError("Cohort selection audit differs from the frozen users")


def load_or_prepare_manifest(path, settings, extractor_factory=None):
    """Extract all eligible profiles once; later models read the frozen snapshot."""
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        validate_manifest(manifest, settings)
        return manifest
    if extractor_factory is None:
        from llm_explainer.data_extractor import DataExtractor
        extractor_factory = DataExtractor
    extractor = extractor_factory(domain_pair=settings["domain_pair"],
                                  split_seed=settings["split_seed"],
                                  sampling_seed=DEFAULT_SPLIT_SEED,
                                  min_rating=settings["min_rating"],
                                  min_held_out_words=settings["min_review_words"])
    # Bypass the finite candidate-pool optimisation, as the leakage audit does.
    users = extractor.get_users_for_validation(num_users=sys.maxsize)
    manifest = make_manifest(users, settings, extractor.split_fingerprints)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers must never observe a partially written snapshot.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                         suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return manifest


def select_cohort(manifest, sampling_seed):
    for cohort in manifest["cohorts"]:
        if cohort["seed"] == sampling_seed:
            return cohort["users"]
    raise ValueError(f"Sampling seed must be one of {list(COHORT_SEEDS)}")
