"""
Persistent on-disk cache for LLM-generated explanations.
Enables rapid evaluation across different Sentence-BERT embedding models without
re-running time-consuming LLM generation when prompts and temperature (0.0) are deterministic.
"""

import json
import logging
import os
import pickle
import re
from typing import Dict, List, Optional

from llm_explainer.config import BASE_DIR
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
        prompt_version: str = "v2",
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
    ):
        self.domain_pair = domain_pair
        self.model_name = model_name
        self.clean_model = sanitize_model_name(model_name)
        self.prompt_version = (prompt_version or "").strip()
        self.clean_version = sanitize_model_name(self.prompt_version) if self.prompt_version else ""
        self.is_v1 = not self.clean_version or self.clean_version.lower() == "v1"
        self.cache_dir = cache_dir or os.path.join(BASE_DIR, "cache", "explanations")
        self.results_dir = results_dir or os.path.join(BASE_DIR, "results")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not self.is_v1:
            self.cache_file = os.path.join(
                self.cache_dir, f"expl_{self.domain_pair}_{self.clean_model}_{self.clean_version}.pkl"
            )
        else:
            self.cache_file = os.path.join(
                self.cache_dir, f"expl_{self.domain_pair}_{self.clean_model}.pkl"
            )
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
        Scans previous run results in results/ to pre-populate the cache with existing explanations.
        Filters strictly for runs with temperature=0.0 and generated on or after 2026-09-28
        matching the specific prompt_version to avoid mixing explanations across prompt versions.
        """
        if not self.is_v1:
            model_dir = os.path.join(self.results_dir, self.domain_pair, f"{self.clean_model}_{self.clean_version}")
        else:
            model_dir = os.path.join(self.results_dir, self.domain_pair, self.clean_model)

        if not os.path.exists(model_dir):
            return

        imported_count = 0
        json_files = sorted(
            [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".json")],
            key=os.path.getmtime,
            reverse=True,
        )

        for jf in json_files:
            fname = os.path.basename(jf)
            # Only consider files from 2026-09-28 onwards (when temperature=0.0 was certified)
            date_match = re.search(r"(\d{8})_\d{6}", fname)
            if not date_match or date_match.group(1) < "20260928":
                continue

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Verify deterministic temperature=0.0
                if data.get("temperature") != 0.0:
                    continue

                # Verify matching prompt_version
                file_ver = data.get("prompt_version")
                if self.is_v1:
                    if file_ver not in (None, "", "v1"):
                        continue
                else:
                    if file_ver != self.prompt_version:
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
                f"Auto-imported {imported_count} verified explanations (temp=0.0, prompt_version={self.prompt_version or 'v1'}) from {model_dir}"
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
