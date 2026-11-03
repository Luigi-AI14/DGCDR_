"""
Persistent on-disk cache for LLM-generated explanations.
Enables rapid evaluation across different Sentence-BERT embedding models without
re-running time-consuming LLM generation when prompts and temperature (0.0) are deterministic.
"""

import json
import glob
import logging
import os
import pickle
import re
from typing import Dict, List, Optional

from llm_explainer.config import BASE_DIR
from llm_explainer.result_paths import EXPLANATIONS_ROOT, generator_folder
from llm_explainer.markdown_exporter import sanitize_model_name

logger = logging.getLogger(__name__)


class ExplanationCache:
    """
    Manages caching and retrieval of LLM explanations per user and item.
    Organized by domain_pair and sanitized LLM model name.
    """

    def __init__(
        self,
        domain_pair: str,
        model_name: str,
        prompt_version: str = "v3",
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        sampling_seed: Optional[int] = None,
        split_seed: Optional[int] = None,
        inference_seed: Optional[int] = None,
    ):
        self.domain_pair = domain_pair
        self.model_name = model_name
        self.clean_model = sanitize_model_name(model_name)
        self.prompt_version = (prompt_version or "").strip()
        self.prompt_hash = prompt_hash
        self.clean_version = sanitize_model_name(self.prompt_version) if self.prompt_version else ""
        self.is_v1 = not self.clean_version or self.clean_version.lower() == "v1"
        self.cache_dir = cache_dir or os.path.join(BASE_DIR, "cache", "explanations")
        self.results_dir = results_dir or os.path.join(BASE_DIR, EXPLANATIONS_ROOT)
        self.sampling_seed = sampling_seed
        self.split_seed = split_seed
        self.inference_seed = inference_seed
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not self.is_v1:
            self.cache_file = os.path.join(
                self.cache_dir, f"expl_{self.domain_pair}_{self.clean_model}_{self.clean_version}.pkl"
            )
        else:
            self.cache_file = os.path.join(
                self.cache_dir, f"expl_{self.domain_pair}_{self.clean_model}.pkl"
            )
        if self.prompt_hash:
            # Template edits under the same version must not reuse older text.
            self.cache_file = self.cache_file[:-4] + "_" + self.prompt_hash + ".pkl"
        context = "".join(f"_{label}{value}" for label, value in (
            ("sample", sampling_seed), ("split", split_seed), ("inference", inference_seed)) if value is not None)
        if context:
            self.cache_file = self.cache_file[:-4] + context + ".pkl"
        # Internal store: user_id -> {item_id: explanation_string}
        self._cache: Dict[str, Dict[str, str]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                num_items = sum(len(v) for v in self._cache.values())
                logger.info(
                    f"Loaded {num_items} cached explanations for {len(self._cache)} users from {self.cache_file}"
                )
                return
            except Exception as e:
                logger.warning(
                    f"Could not load explanation cache from {self.cache_file}: {e}. Initializing empty cache."
                )
                self._cache = {}

        # Auto-import from existing result JSON files if no dedicated cache file exists
        self._auto_import_from_results()

    def _auto_import_from_results(self) -> None:
        """
        Scans the active explanation root to pre-populate the cache with existing explanations.
        Filters strictly for runs with temperature=0.0 and generated on or after 2026-09-28
        matching the specific prompt_version to avoid mixing explanations across prompt versions.
        """
        if not self.is_v1:
            model_dir = os.path.join(self.results_dir, self.domain_pair, f"{self.clean_model}_{self.clean_version}")
        else:
            model_dir = os.path.join(self.results_dir, self.domain_pair, self.clean_model)

        version = self.prompt_version
        mode = "few-shot"
        for suffix, selected_mode in (("_no_few_shot", "zero-shot"), ("_few_shot", "few-shot")):
            if version.endswith(suffix):
                version, mode = version[:-len(suffix)], selected_mode
                break
        seed_folder = f"seed{self.sampling_seed}" if self.sampling_seed is not None else "seed*"
        patterns = [os.path.join(self.results_dir, self.domain_pair, seed_folder,
                                generator_folder(self.model_name, version, mode), "*.json")]
        # Callers explicitly pointing at an older/custom root retain flat-layout support.
        patterns.append(os.path.join(model_dir, "*.json"))
        imported_count = 0
        json_files = sorted(
            {path for pattern in patterns for path in glob.glob(pattern)},
            key=os.path.getmtime,
            reverse=True,
        )

        for jf in json_files:
            fname = os.path.basename(jf)
            # Only consider files from 2026-09-28 onwards (when temperature=0.0 was certified)
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}", fname)
            if date_match is None:
                date_match = re.search(r"(\d{8})_\d{6}", fname)
            if not date_match or date_match.group(1).replace("-", "") < "20260928":
                continue

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("is_shuffled"):
                    continue
                if any(value is not None and data.get(field) != value for field, value in (
                    ("sampling_seed", self.sampling_seed), ("split_seed", self.split_seed),
                    ("inference_seed", self.inference_seed))):
                    continue

                # Verify deterministic temperature=0.0
                if data.get("temperature") != 0.0:
                    continue

                # Reports store version and the demonstration switch separately.
                file_ver = data.get("prompt_version")
                expected_mode = "zero-shot" if data.get("no_few_shot", False) else "few-shot"
                recorded_mode = data.get("generation_mode", data.get("generation_prompt", {}).get("generation_mode"))
                if recorded_mode is not None and recorded_mode != expected_mode:
                    continue
                report_cache_version = file_ver or ""
                if data.get("no_few_shot", False):
                    report_cache_version += "_no_few_shot"
                elif file_ver == "v3":
                    report_cache_version += "_few_shot"
                if self.is_v1:
                    if file_ver not in (None, "", "v1"):
                        continue
                else:
                    if report_cache_version != self.prompt_version:
                        continue
                if self.prompt_hash:
                    prompt_info = data.get("generation_prompt", {})
                    report_hash = prompt_info.get("cache_sha256", prompt_info.get("template_sha256"))
                    if report_hash != self.prompt_hash:
                        continue

                users = data.get("users", [])
                for u in users:
                    uid = u.get("user_id")
                    if not uid:
                        continue
                    if uid not in self._cache:
                        self._cache[uid] = {}
                    for it in u.get("items", []):
                        iid = it.get("id_item")
                        expl = it.get("llm_explanation", "").strip()
                        if iid and expl and iid not in self._cache[uid]:
                            self._cache[uid][iid] = expl
                            imported_count += 1
            except Exception:
                continue

        if imported_count > 0:
            logger.info(
                f"Auto-imported {imported_count} verified explanations (temp=0.0, prompt_version={self.prompt_version or 'v1'}) from {self.results_dir}"
            )
            self.save()

    def get_user_explanations(
        self, user_id: str, expected_item_ids: List[str]
    ) -> Optional[Dict[str, str]]:
        """
        Returns explanations dictionary {item_id: explanation} for the user
        if ALL expected_item_ids exist with non-empty text in the cache.
        If any expected item is missing, returns None.
        """
        if user_id not in self._cache:
            return None

        user_items = self._cache[user_id]
        for iid in expected_item_ids:
            if iid not in user_items or not user_items[iid].strip():
                return None

        return {iid: user_items[iid] for iid in expected_item_ids}

    def set_user_explanations(
        self, user_id: str, explanations: List[Dict[str, str]]
    ) -> None:
        """Stores explanations for a user."""
        if user_id not in self._cache:
            self._cache[user_id] = {}

        for entry in explanations:
            iid = entry.get("item_id")
            expl = entry.get("explanation", "").strip()
            if iid and expl:
                self._cache[user_id][iid] = expl
                self._dirty = True

    def save(self) -> None:
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            num_items = sum(len(v) for v in self._cache.values())
            logger.info(
                f"Saved explanation cache ({num_items} items across {len(self._cache)} users): {self.cache_file}"
            )
            self._dirty = False
        except Exception as e:
            logger.warning(f"Failed to save explanation cache to {self.cache_file}: {e}")

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def total_explanations(self) -> int:
        return sum(len(v) for v in self._cache.values())

    @property
    def total_users(self) -> int:
        return len(self._cache)

    def __len__(self) -> int:
        return self.total_explanations
