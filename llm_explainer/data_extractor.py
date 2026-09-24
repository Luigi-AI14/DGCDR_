"""
Data extraction and splitting module for DGCDR.
Uses RecBole CDR dataset loading with seed 42 to guarantee exact match with model splits.
Filters interactions by rating >= 4.0 and removes null/empty/whitespace reviews (check_review_null.py criteria).
Extracts item titles and review metadata efficiently with local caching.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset

from llm_explainer.config import BASE_DIR, DOMAIN_CONFIGS

logger = logging.getLogger(__name__)


def is_valid_review_text(text: Any) -> bool:
    """
    Check if review text is valid according to check_review_null.py rules:
    - Key exists and is not None
    - Is a string
    - Length > 0
    - Stripped length > 0 (not just whitespace)
    """
    if text is None:
        return False
    if not isinstance(text, str):
        return False
    if len(text.strip()) == 0:
        return False
    return True


class DataExtractor:
    def __init__(
        self,
        domain_pair: str = "Cloth-Elec",
        seed: int = 42,
        min_rating: float = 4.0,
        cache_dir: Optional[str] = None,
    ):
        if domain_pair not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain pair: {domain_pair}. Available: {list(DOMAIN_CONFIGS.keys())}")

        self.domain_pair = domain_pair
        self.config_info = DOMAIN_CONFIGS[domain_pair]
        self.seed = seed
        self.min_rating = min_rating
        self.cache_dir = cache_dir or os.path.join(BASE_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.source_name = self.config_info["source_name"]
        self.target_name = self.config_info["target_name"]
        self.source_display = self.config_info["source_display_name"]
        self.target_display = self.config_info["target_display_name"]

    def load_recbole_splits(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Loads dataset using RecBole CDR and builds splits with exact seed.
        Returns:
            user_source_all: dict user_id -> list of source items (100% train+valid)
            user_target_history: dict user_id -> list of target items (80% train+valid)
            user_target_test: dict user_id -> list of target test items (20% held-out)
        """
        logger.info(f"Loading RecBole dataset for {self.domain_pair} with seed {self.seed}...")
        config = CDRConfig(
            model="DGCDR",
            config_file_list=self.config_info["config_file_list"],
            config_dict={"seed": self.seed},
        )
        dataset = create_dataset(config)
        built = dataset.build()

        src_ds = dataset.source_domain_dataset
        tgt_ds = dataset.target_domain_dataset

        inv_src_user = {v: k for k, v in dataset.source_user_ID_remap_dict.items()}
        inv_src_item = {v: k for k, v in dataset.source_item_ID_remap_dict.items()}
        inv_tgt_user = {v: k for k, v in dataset.target_user_ID_remap_dict.items()}
        inv_tgt_item = {v: k for k, v in dataset.target_item_ID_remap_dict.items()}

        # 1. Source domain (Part 0: 100% train+valid)
        src_all_ds = built[0]
        src_u_indices = src_all_ds.inter_feat[src_ds.uid_field].numpy()
        src_i_indices = src_all_ds.inter_feat[src_ds.iid_field].numpy()
        user_source_all = defaultdict(list)
        for u, i in zip(src_u_indices, src_i_indices):
            user_source_all[inv_src_user[u]].append(inv_src_item[i])

        # 2. Target domain train (Part 2) + valid (Part 3) -> 80% history
        tgt_train_ds = built[2]
        tgt_valid_ds = built[3]
        user_target_history = defaultdict(list)

        for ds_part in [tgt_train_ds, tgt_valid_ds]:
            u_indices = ds_part.inter_feat[tgt_ds.uid_field].numpy()
            i_indices = ds_part.inter_feat[tgt_ds.iid_field].numpy()
            for u, i in zip(u_indices, i_indices):
                user_target_history[inv_tgt_user[u]].append(inv_tgt_item[i])

        # 3. Target domain test (Part 4: 20% held-out)
        tgt_test_ds = built[4]
        test_u_indices = tgt_test_ds.inter_feat[tgt_ds.uid_field].numpy()
        test_i_indices = tgt_test_ds.inter_feat[tgt_ds.iid_field].numpy()
        user_target_test = defaultdict(list)
        for u, i in zip(test_u_indices, test_i_indices):
            user_target_test[inv_tgt_user[u]].append(inv_tgt_item[i])

        logger.info(
            f"Splits extracted: {len(user_source_all)} source users, "
            f"{len(user_target_history)} target history users, "
            f"{len(user_target_test)} target test users."
        )
        return user_source_all, user_target_history, user_target_test

    def extract_metadata_for_users(
        self,
        candidate_uids: Set[str],
        source_item_ids: Set[str],
        target_item_ids: Set[str],
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[Tuple[str, str], Dict], Dict[Tuple[str, str], Dict]]:
        """
        Fast streaming extraction of item titles and reviews for candidate users.
        Uses substring check for high throughput across multi-GB files.
        """
        # Cache file path based on hash of candidate users
        cand_str = "_".join(sorted(list(candidate_uids)))
        uid_hash = hashlib.md5(cand_str.encode("utf-8")).hexdigest()[:12]
        cache_id = f"meta_{self.domain_pair}_{len(candidate_uids)}_{uid_hash}"
        cache_file = os.path.join(self.cache_dir, f"{cache_id}.json")

        if os.path.exists(cache_file):
            logger.info(f"Loading metadata from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            src_titles = cdata["src_titles"]
            tgt_titles = cdata["tgt_titles"]
            src_reviews = {tuple(k.split(":::")): v for k, v in cdata["src_reviews"].items()}
            tgt_reviews = {tuple(k.split(":::")): v for k, v in cdata["tgt_reviews"].items()}
            return src_titles, tgt_titles, src_reviews, tgt_reviews

        logger.info("Extracting metadata from JSONL files...")

        def get_asin_from_line(line_str: str) -> Optional[str]:
            idx = line_str.rfind('"parent_asin": "')
            if idx != -1:
                end_idx = line_str.find('"', idx + 16)
                if end_idx != -1:
                    return line_str[idx + 16 : end_idx]
            return None

        def get_uid_from_line(line_str: str) -> Optional[str]:
            idx = line_str.rfind('"user_id": "')
            if idx != -1:
                end_idx = line_str.find('"', idx + 12)
                if end_idx != -1:
                    return line_str[idx + 12 : end_idx]
            return None

        # 1. Source Item Titles
        src_titles = {}
        src_meta_path = self.config_info["source_item_meta"]
        if os.path.exists(src_meta_path):
            logger.info(f"Scanning source item metadata: {os.path.basename(src_meta_path)}")
            with open(src_meta_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, 1):
                    if len(src_titles) >= len(source_item_ids):
                        break
                    asin = get_asin_from_line(line)
                    if asin and asin in source_item_ids and asin not in src_titles:
                        try:
                            d = json.loads(line)
                            src_titles[asin] = d.get("title", f"Item {asin}")
                        except Exception:
                            continue
                    if line_idx % 5_000_000 == 0:
                        logger.info(f"  [Item Meta Source] Processed {line_idx:,} lines - found {len(src_titles)}/{len(source_item_ids)} titles")

        # 2. Target Item Titles
        tgt_titles = {}
        tgt_meta_path = self.config_info["target_item_meta"]
        if os.path.exists(tgt_meta_path):
            logger.info(f"Scanning target item metadata: {os.path.basename(tgt_meta_path)}")
            with open(tgt_meta_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, 1):
                    if len(tgt_titles) >= len(target_item_ids):
                        break
                    asin = get_asin_from_line(line)
                    if asin and asin in target_item_ids and asin not in tgt_titles:
                        try:
                            d = json.loads(line)
                            tgt_titles[asin] = d.get("title", f"Item {asin}")
                        except Exception:
                            continue
                    if line_idx % 5_000_000 == 0:
                        logger.info(f"  [Item Meta Target] Processed {line_idx:,} lines - found {len(tgt_titles)}/{len(target_item_ids)} titles")

        # 3. Source Reviews
        src_reviews = {}
        src_rev_path = self.config_info["source_review_meta"]
        if os.path.exists(src_rev_path):
            logger.info(f"Scanning source review metadata: {os.path.basename(src_rev_path)}")
            with open(src_rev_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, 1):
                    uid = get_uid_from_line(line)
                    if uid and uid in candidate_uids:
                        asin = get_asin_from_line(line)
                        if asin and asin in source_item_ids:
                            try:
                                d = json.loads(line)
                                rating = float(d.get("rating", 0.0))
                                text = d.get("text", "")
                                if rating >= self.min_rating and is_valid_review_text(text):
                                    src_reviews[(uid, asin)] = {
                                        "rating": rating,
                                        "title": d.get("title", ""),
                                        "text": text,
                                    }
                            except Exception:
                                continue
                    if line_idx % 5_000_000 == 0:
                        logger.info(f"  [Reviews Source] Processed {line_idx:,} lines - found {len(src_reviews)} reviews")

        # 4. Target Reviews
        tgt_reviews = {}
        tgt_rev_path = self.config_info["target_review_meta"]
        if os.path.exists(tgt_rev_path):
            logger.info(f"Scanning target review metadata: {os.path.basename(tgt_rev_path)}")
            with open(tgt_rev_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, 1):
                    uid = get_uid_from_line(line)
                    if uid and uid in candidate_uids:
                        asin = get_asin_from_line(line)
                        if asin and asin in target_item_ids:
                            try:
                                d = json.loads(line)
                                rating = float(d.get("rating", 0.0))
                                text = d.get("text", "")
                                if rating >= self.min_rating and is_valid_review_text(text):
                                    tgt_reviews[(uid, asin)] = {
                                        "rating": rating,
                                        "title": d.get("title", ""),
                                        "text": text,
                                    }
                            except Exception:
                                continue
                    if line_idx % 5_000_000 == 0:
                        logger.info(f"  [Reviews Target] Processed {line_idx:,} lines - found {len(tgt_reviews)} reviews")

        # Save to cache
        cache_data = {
            "src_titles": src_titles,
            "tgt_titles": tgt_titles,
            "src_reviews": {f"{k[0]}:::{k[1]}": v for k, v in src_reviews.items()},
            "tgt_reviews": {f"{k[0]}:::{k[1]}": v for k, v in tgt_reviews.items()},
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)

        return src_titles, tgt_titles, src_reviews, tgt_reviews

    def load_compact_cache(
        self,
    ) -> Optional[Tuple[Dict[str, str], Dict[str, str], Dict[Tuple[str, str], Dict], Dict[Tuple[str, str], Dict]]]:
        """
        Loads preprocessed compact metadata from cache/ if available.
        Checks for binary .pkl first (< 0.1s), then .json.
        Returns:
            (src_titles, tgt_titles, src_reviews, tgt_reviews) or None.
        """
        pkl_path = os.path.join(self.cache_dir, f"compact_{self.domain_pair}.pkl")
        json_path = os.path.join(self.cache_dir, f"compact_{self.domain_pair}.json")

        if os.path.exists(pkl_path):
            logger.info(f"Loading preprocessed compact dataset from binary cache: {pkl_path}")
            t0 = time.time()
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded compact dataset in {time.time() - t0:.3f}s!")
            return (
                data["source_items"],
                data["target_items"],
                data["source_reviews"],
                data["target_reviews"],
            )

        if os.path.exists(json_path):
            logger.info(f"Loading preprocessed compact dataset from JSON cache: {json_path}")
            t0 = time.time()
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            src_titles = data["source_items"]
            tgt_titles = data["target_items"]
            src_reviews = {tuple(k.split(":::")): v for k, v in data["source_reviews"].items()}
            tgt_reviews = {tuple(k.split(":::")): v for k, v in data["target_reviews"].items()}
            logger.info(f"Loaded compact dataset from JSON in {time.time() - t0:.3f}s!")
            return src_titles, tgt_titles, src_reviews, tgt_reviews

        return None

    def get_users_for_validation(self, num_users: int = 5) -> List[Dict]:
        """
        Extracts complete, verified profiles for num_users.
        Guarantees:
        - History items have rating >= 4.0 and valid text.
        - Held-out items have rating >= 4.0 and valid ground-truth text.
        """
        user_source_all, user_target_history, user_target_test = self.load_recbole_splits()

        # Find users who have at least 1 item in source, target history, and target test
        overlapping_uids = [
            u
            for u in user_target_test.keys()
            if len(user_source_all.get(u, [])) >= 2
            and len(user_target_history.get(u, [])) >= 2
            and len(user_target_test.get(u, [])) >= 1
        ]
        overlapping_uids.sort()

        logger.info(f"Found {len(overlapping_uids)} overlapping users with interactions in all splits.")

        # Check if compact preprocessed cache exists
        compact_data = self.load_compact_cache()
        if compact_data is not None:
            src_titles, tgt_titles, src_reviews, tgt_reviews = compact_data
            pool_uids = None
        else:
            logger.info(
                f"No compact cache found for {self.domain_pair}. "
                "Falling back to on-the-fly streaming scan. "
                "(TIP: Run 'python preprocess_compact_data.py' once for instant sub-second loading!)"
            )
            # Take candidate pool (e.g. first num_users * 4 to account for review text filtering)
            pool_uids = set(overlapping_uids[: max(num_users * 4, 20)])

            all_src_items = set()
            all_tgt_items = set()
            for u in pool_uids:
                all_src_items.update(user_source_all.get(u, []))
                all_tgt_items.update(user_target_history.get(u, []))
                all_tgt_items.update(user_target_test.get(u, []))

            src_titles, tgt_titles, src_reviews, tgt_reviews = self.extract_metadata_for_users(
                candidate_uids=pool_uids,
                source_item_ids=all_src_items,
                target_item_ids=all_tgt_items,
            )

        valid_users = []

        for uid in overlapping_uids:
            if pool_uids is not None and uid not in pool_uids:
                continue

            # 1. Source History: items with rating >= 4 and valid review
            src_hist = []
            for iid in user_source_all.get(uid, []):
                rev = src_reviews.get((uid, iid))
                if rev and rev["rating"] >= self.min_rating and is_valid_review_text(rev["text"]):
                    src_hist.append(
                        {
                            "item_id": iid,
                            "item_title": src_titles.get(iid, f"Item {iid}"),
                            "rating": rev["rating"],
                            "review_title": rev.get("title", ""),
                            "review_text": rev["text"],
                        }
                    )

            # 2. Target History: items with rating >= 4 and valid review
            tgt_hist = []
            for iid in user_target_history.get(uid, []):
                rev = tgt_reviews.get((uid, iid))
                if rev and rev["rating"] >= self.min_rating and is_valid_review_text(rev["text"]):
                    tgt_hist.append(
                        {
                            "item_id": iid,
                            "item_title": tgt_titles.get(iid, f"Item {iid}"),
                            "rating": rev["rating"],
                            "review_title": rev.get("title", ""),
                            "review_text": rev["text"],
                        }
                    )

            # 3. Held-out Target Items: items with rating >= 4 and valid review
            held_out = []
            for iid in user_target_test.get(uid, []):
                rev = tgt_reviews.get((uid, iid))
                if rev and rev["rating"] >= self.min_rating and is_valid_review_text(rev["text"]):
                    held_out.append(
                        {
                            "item_id": iid,
                            "item_title": tgt_titles.get(iid, f"Item {iid}"),
                            "ground_truth_rating": rev["rating"],
                            "ground_truth_review": rev["text"],
                            "ground_truth_title": rev.get("title", ""),
                        }
                    )

            # Check eligibility: must have at least 1 in each
            if src_hist and tgt_hist and held_out:
                valid_users.append(
                    {
                        "user_id": uid,
                        "source_domain": self.source_display,
                        "target_domain": self.target_display,
                        "source_history": src_hist,
                        "target_history": tgt_hist,
                        "held_out_items": held_out,
                    }
                )

            if len(valid_users) >= num_users:
                break

        logger.info(f"Successfully selected {len(valid_users)} eligible users for validation.")
        return valid_users
