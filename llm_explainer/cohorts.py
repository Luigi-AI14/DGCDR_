"""Frozen user cohorts shared by all explanation models and sampling runs.

Like the leakage audit, sample each cohort from the remaining sorted eligible
pool. Generation demonstrations are synthetic, so audit-specific demonstration
exclusions do not apply here.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile


DEFAULT_SPLIT_SEED = 42
DEFAULT_INFERENCE_SEED = 42
COHORT_SEEDS = (11, 22, 33, 44, 55)


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


def cohort_settings(domain_pair, num_users, split_seed, min_rating, min_words):
    if num_users < 1:
        raise ValueError("num_users must be positive")
    return dict(domain_pair=domain_pair, num_users=num_users, split_seed=split_seed,
                min_rating=min_rating, min_review_words=min_words,
                cohort_seeds=list(COHORT_SEEDS))


def make_manifest(users, settings, split_fingerprints):
    remaining = sorted(users, key=lambda user: user["user_id"])
    if len({user["user_id"] for user in remaining}) != len(remaining):
        raise ValueError("Eligible pool contains duplicate user IDs")
    required = len(COHORT_SEEDS) * settings["num_users"]
    if len(remaining) < required:
        raise ValueError(f"Need {required} eligible users for five disjoint cohorts; "
                         f"only {len(remaining)} available. Reduce --num_users.")
    cohorts = []
    for seed in COHORT_SEEDS:
        selected = random.Random(seed).sample(remaining, settings["num_users"])
        selected.sort(key=lambda user: user["user_id"])
        chosen = {user["user_id"] for user in selected}
        remaining = [user for user in remaining if user["user_id"] not in chosen]
        cohorts.append(dict(seed=seed, users=selected))
    manifest = dict(schema_version=1, settings=settings,
                    split_fingerprints=split_fingerprints, cohorts=cohorts)
    manifest["sha256"] = digest(manifest)
    return manifest


def validate_manifest(manifest, settings):
    if manifest.get("schema_version") != 1 or manifest.get("settings") != settings:
        raise ValueError("Cohort manifest settings differ; use a new --cohort_manifest path")
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
