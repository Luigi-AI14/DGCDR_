"""Freeze DGCDR's original split, then sample disjoint user cohorts."""

import logging
import os
import platform
import random
import sys
from importlib import metadata
from pathlib import Path

from .common import (SCHEMA_VERSION, digest, file_hash, normalize, now, read_json,
                     valid_title, validate_dataset, write_json)

LOG = logging.getLogger(__name__)


def extract_dataset(domain_pair, split_seed, min_rating, min_words, cache_dir=None):
    # Imports stay lazy: probing, reporting and unit tests do not require RecBole,
    # PyTorch, transformers, SGLang, numpy or GPU packages.
    try:
        from llm_explainer.data_extractor import DataExtractor
    except ImportError as error:
        raise RuntimeError("Preparation needs the DGCDR/RecBole environment. "
                           "Activate it, or import an existing --dataset snapshot. "
                           "Import failure: " + str(error)) from error

    class FrozenExtractor(DataExtractor):
        def load_recbole_splits(self):
            if not hasattr(self, "audit_splits"):
                # The repository's YAML contains './dataset/' paths. Resolve
                # them against DGCDR even if the caller works elsewhere.
                from llm_explainer.config import BASE_DIR
                previous = os.getcwd()
                try:
                    os.chdir(BASE_DIR)
                    self.audit_splits = super().load_recbole_splits()
                finally:
                    os.chdir(previous)
            return self.audit_splits

    extractor = FrozenExtractor(domain_pair=domain_pair, seed=split_seed,
                                min_rating=min_rating, min_held_out_words=min_words,
                                cache_dir=str(cache_dir) if cache_dir else None)
    # sys.maxsize intentionally selects ALL eligible users, bypassing the normal
    # finite candidate-pool optimisation and final user subsampling.
    users = extractor.get_users_for_validation(num_users=sys.maxsize)
    source, history, test = extractor.audit_splits
    for uid, held_items in test.items():
        if set(held_items).intersection(history.get(uid, [])):
            raise ValueError("Target train/validation and test overlap for user " + uid)
    from llm_explainer import data_extractor
    from llm_explainer.config import BASE_DIR
    files = []
    cfg = extractor.config_info
    paths = [cfg["source_inter"], cfg["target_inter"]] + list(cfg["config_file_list"])
    paths += [data_extractor.__file__, str(Path(BASE_DIR) / "recbole_cdr/data/dataset.py")]
    for name in ("source_item_meta", "target_item_meta", "source_review_meta", "target_review_meta"):
        paths.append(cfg[name])
    cache = Path(extractor.cache_dir)
    compact = cache / ("compact_" + domain_pair + ".pkl")
    if not compact.exists():
        compact = compact.with_suffix(".json")
    if compact.exists():
        paths.append(str(compact))
    for filename in paths:
        path = Path(filename)
        stat = path.stat()
        # Multi-GB raw metadata uses size+mtime; exact extracted content is hashed
        # below. Configs, interactions, extractor and compact cache use SHA-256.
        entry = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if str(path) not in [cfg[k] for k in ("source_item_meta", "target_item_meta",
                                             "source_review_meta", "target_review_meta")]:
            entry["sha256"] = file_hash(path)
        files.append(entry)
    versions = {}
    for package in ("recbole", "torch", "numpy", "pandas"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    dataset = {
        "schema_version": SCHEMA_VERSION, "domain_pair": domain_pair, "split_seed": split_seed,
        "filters": {"min_rating": min_rating, "min_held_out_words": min_words,
                    "word_count_includes_review_title": True},
        "provenance": {"created_at": now(), "python": platform.python_version(), "packages": versions,
                       "repo_root": BASE_DIR, "files": files,
                       "extraction": "DataExtractor.get_users_for_validation(all eligible users)"},
        "split_fingerprints": {name: {"sha256": digest({u: sorted(items) for u, items in split.items()}),
                                      "users": len(split), "interactions": sum(map(len, split.values()))}
                               for name, split in (("source_all", source), ("target_history", history),
                                                   ("target_test", test))},
        "users": [{"user_id": u["user_id"], "held_out_items": u["held_out_items"]}
                  for u in users],
    }
    validate_dataset(dataset)
    LOG.info("Extracted complete eligible pool: %s users, %s held-out reviews", len(dataset["users"]),
             sum(len(user["held_out_items"]) for user in dataset["users"]))
    return dataset


def make_manifest(dataset, seeds, num_users, few_shot_count=3, demo_seed=314159,
                  negative_controls=3):
    validate_dataset(dataset)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Provide exactly five different cohort seeds")
    if num_users < 1 or few_shot_count < 1 or negative_controls < 1:
        raise ValueError("User, demonstration and negative-control counts must be positive")
    users = sorted(dataset["users"], key=lambda user: user["user_id"])
    demo_rng = random.Random(demo_seed)
    candidates = list(users)
    demo_rng.shuffle(candidates)
    demos, demo_uids, demo_iids, demo_titles, demo_texts = [], set(), set(), set(), set()
    for user in candidates:
        items = sorted(user["held_out_items"], key=lambda item: item["item_id"])
        demo_rng.shuffle(items)
        for item in items:
            title, text = normalize(item.get("item_title")), normalize(item["ground_truth_review"])
            if valid_title(item) and item["item_id"] not in demo_iids and title not in demo_titles and text not in demo_texts:
                demos.append(dict(item, user_id=user["user_id"]))
                demo_uids.add(user["user_id"])
                demo_iids.add(item["item_id"])
                demo_titles.add(title)
                demo_texts.add(text)
                break
        if len(demos) == few_shot_count:
            break
    if len(demos) != few_shot_count:
        raise ValueError("Not enough distinct users/items with metadata for few-shot demonstrations")
    remaining = []
    excluded = 0
    for user in users:
        if user["user_id"] in demo_uids:
            continue
        items = [item for item in user["held_out_items"]
                 if item["item_id"] not in demo_iids
                 and normalize(item.get("item_title")) not in demo_titles
                 and normalize(item["ground_truth_review"]) not in demo_texts]
        excluded += len(user["held_out_items"]) - len(items)
        if items:
            remaining.append({"user_id": user["user_id"], "held_out_items": items})
    required = len(seeds) * num_users
    if len(remaining) < required:
        raise ValueError("Need {} eligible users after demonstration exclusion; only {} available. "
                         "Reduce --num-users or --few-shot-count.".format(required, len(remaining)))
    cohorts = []
    for seed in seeds:
        selected = random.Random(seed).sample(remaining, num_users)
        selected.sort(key=lambda user: user["user_id"])
        chosen = {user["user_id"] for user in selected}
        remaining = [user for user in remaining if user["user_id"] not in chosen]
        cohorts.append({"seed": seed, "users": selected})
    # Generate once, against the entire eligible snapshot, and reuse across
    # models AND prompting modes. They are abstention controls, not nonmembers.
    known_users = {user["user_id"] for user in users}
    known_items = {item["item_id"] for user in users for item in user["held_out_items"]}
    synthetic = {}
    for seed in seeds:
        rng = random.Random("negative-controls:" + str(seed))
        controls = []
        while len(controls) < negative_controls:
            iid = "B0" + "".join(rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(8))
            uid = "A" + "".join(rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(27))
            if iid not in known_items and uid not in known_users:
                controls.append({"item_id": iid, "user_id": uid})
                known_items.add(iid)
                known_users.add(uid)
        synthetic[str(seed)] = controls
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_sha256": digest(dataset),
        "domain_pair": dataset["domain_pair"], "split_seed": dataset["split_seed"],
        "cohort_seeds": list(seeds), "num_users": num_users,
        "demo_seed": demo_seed, "few_shot_count": few_shot_count,
        "negative_controls": negative_controls, "demonstrations": demos, "cohorts": cohorts,
        "synthetic_controls": synthetic,
        "excluded_demo_overlap_records": excluded,
    }
    validate_manifest(manifest, dataset)
    return manifest


def validate_manifest(manifest, dataset):
    if manifest["dataset_sha256"] != digest(dataset):
        raise ValueError("Dataset fingerprint differs from the frozen cohort manifest")
    if manifest["split_seed"] != dataset["split_seed"] or manifest["domain_pair"] != dataset["domain_pair"]:
        raise ValueError("Manifest and dataset split/domain mismatch")
    seeds = manifest["cohort_seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5 or [c["seed"] for c in manifest["cohorts"]] != seeds:
        raise ValueError("Manifest must have exactly five distinct ordered cohorts")
    originals = {u["user_id"]: {i["item_id"]: i for i in u["held_out_items"]} for u in dataset["users"]}
    demos = manifest["demonstrations"]
    if len(demos) != manifest["few_shot_count"] or not demos:
        raise ValueError("Missing demonstration records")
    seen = {d["user_id"] for d in demos}
    demo_iids = {d["item_id"] for d in demos}
    demo_titles = {normalize(d["item_title"]) for d in demos}
    demo_texts = {normalize(d["ground_truth_review"]) for d in demos}
    for demo in demos:
        item = {k: v for k, v in demo.items() if k != "user_id"}
        if originals.get(demo["user_id"], {}).get(demo["item_id"]) != item:
            raise ValueError("Demonstration differs from original record")
    known_users = set(originals)
    known_items = {iid for items in originals.values() for iid in items}
    for seed in seeds:
        controls = manifest["synthetic_controls"][str(seed)]
        if len(controls) != manifest["negative_controls"]:
            raise ValueError("Wrong synthetic-control count")
        for control in controls:
            if control["user_id"] in known_users or control["item_id"] in known_items:
                raise ValueError("Synthetic ID overlaps the frozen dataset or another control")
            known_users.add(control["user_id"])
            known_items.add(control["item_id"])
    for cohort in manifest["cohorts"]:
        if len(cohort["users"]) != manifest["num_users"]:
            raise ValueError("Wrong cohort size")
        for user in cohort["users"]:
            uid = user["user_id"]
            if uid in seen:
                raise ValueError("User overlap between cohorts or demonstrations: " + uid)
            seen.add(uid)
            if not user["held_out_items"]:
                raise ValueError("Empty held-out user")
            iids = set()
            for item in user["held_out_items"]:
                iid = item["item_id"]
                if iid in iids or originals.get(uid, {}).get(iid) != item:
                    raise ValueError("Duplicated or modified evaluation record")
                iids.add(iid)
                if iid in demo_iids or normalize(item.get("item_title")) in demo_titles or normalize(item["ground_truth_review"]) in demo_texts:
                    raise ValueError("Demonstration answer overlaps evaluation")


def prepare(output_dir, seeds, num_users, few_shot_count=3, demo_seed=314159,
            negative_controls=3, dataset_path=None, **extraction):
    output = Path(output_dir)
    if (output / "manifest.json").exists() or (output / "dataset.json").exists():
        raise ValueError("Preparation already exists. Reuse it with run/report, or choose a new --output-dir.")
    dataset = read_json(dataset_path) if dataset_path else extract_dataset(**extraction)
    manifest = make_manifest(dataset, seeds, num_users, few_shot_count, demo_seed, negative_controls)
    write_json(output / "dataset.json", dataset)
    write_json(output / "manifest.json", manifest)
    for cohort in manifest["cohorts"]:
        write_json(output / "cohorts" / ("seed_" + str(cohort["seed"]) + ".json"), cohort)
    LOG.info("Frozen split seed=%s; five disjoint cohorts x %s users; manifest=%s",
             dataset["split_seed"], num_users, digest(manifest)[:12])
    return manifest
