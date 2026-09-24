"""
Compact Data Preprocessor for DGCDR.
Extracts only the items and reviews present in the RecBole .inter files,
saving a lightweight, unified compact JSON and PKL in the cache/ directory.
Reduces SSD scan time from minutes to milliseconds (< 0.1s loading).
"""

import os
import json
import time
import pickle
import logging
from typing import Dict, Tuple, Set, Optional

from .config import DOMAIN_CONFIGS, BASE_DIR

logger = logging.getLogger("CompactBuilder")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def is_valid_review_text(text: Optional[str]) -> bool:
    """Validates review text according to check_review_null.py rules."""
    if text is None:
        return False
    if not isinstance(text, str):
        return False
    return len(text.strip()) > 0


def load_inter_pairs(inter_path: str) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
    """Reads a RecBole .inter file and returns (users, items, user_item_pairs)."""
    users = set()
    items = set()
    pairs = set()

    with open(inter_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                u, i = parts[0], parts[1]
                users.add(u)
                items.add(i)
                pairs.add((u, i))

    return users, items, pairs


def scan_item_titles(meta_path: str, item_ids: Set[str]) -> Dict[str, str]:
    """Scans an item metadata JSONL file to extract titles for relevant item IDs."""
    titles = {}
    if not os.path.exists(meta_path):
        logger.warning(f"Item metadata file not found: {meta_path}")
        return titles

    filename = os.path.basename(meta_path)
    logger.info(f"Scanning item titles from {filename} for {len(item_ids):,} items...")
    t0 = time.time()

    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f, 1):
            if len(titles) >= len(item_ids):
                break

            idx = line.rfind('"parent_asin": "')
            if idx != -1:
                end_idx = line.find('"', idx + 16)
                if end_idx != -1:
                    asin = line[idx + 16 : end_idx]
                    if asin in item_ids and asin not in titles:
                        try:
                            d = json.loads(line)
                            titles[asin] = d.get("title", f"Item {asin}")
                        except Exception:
                            titles[asin] = f"Item {asin}"

            if line_idx % 2_000_000 == 0:
                logger.info(f"  [{filename}] Processed {line_idx:,} lines - found {len(titles):,}/{len(item_ids):,} titles")

    elapsed = time.time() - t0
    logger.info(f"  [{filename}] Completed in {elapsed:.1f}s. Extracted {len(titles):,} item titles.")
    return titles


def scan_reviews(
    review_path: str,
    target_pairs: Set[Tuple[str, str]],
    target_users: Set[str],
) -> Dict[Tuple[str, str], Dict]:
    """
    Scans a review JSONL file to extract rating, title, and text for exact (user, item) pairs.
    Uses C-speed string slicing before json.loads.
    """
    reviews = {}
    if not os.path.exists(review_path):
        logger.warning(f"Review metadata file not found: {review_path}")
        return reviews

    filename = os.path.basename(review_path)
    logger.info(f"Scanning reviews from {filename} for {len(target_pairs):,} interaction pairs...")
    t0 = time.time()

    with open(review_path, "r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f, 1):
            u_idx = line.rfind('"user_id": "')
            a_idx = line.rfind('"parent_asin": "')

            if u_idx != -1 and a_idx != -1:
                u_end = line.find('"', u_idx + 12)
                a_end = line.find('"', a_idx + 16)

                if u_end != -1 and a_end != -1:
                    uid = line[u_idx + 12 : u_end]
                    if uid in target_users:
                        asin = line[a_idx + 16 : a_end]
                        pair = (uid, asin)
                        if pair in target_pairs and pair not in reviews:
                            try:
                                d = json.loads(line)
                                raw_text = d.get("text", "")
                                if is_valid_review_text(raw_text):
                                    reviews[pair] = {
                                        "rating": float(d.get("rating", 0.0)),
                                        "title": str(d.get("title", "")),
                                        "text": str(raw_text).strip(),
                                    }
                            except Exception:
                                continue

            if line_idx % 5_000_000 == 0:
                rate = line_idx / (time.time() - t0)
                logger.info(
                    f"  [{filename}] Processed {line_idx:,} lines ({rate:,.0f} lines/s) - found {len(reviews):,} reviews"
                )

    elapsed = time.time() - t0
    logger.info(f"  [{filename}] Completed in {elapsed:.1f}s. Extracted {len(reviews):,} valid reviews.")
    return reviews


def build_compact_dataset(
    domain_pair: str = "Cloth-Elec",
    cache_dir: str = os.path.join(BASE_DIR, "cache"),
) -> Tuple[str, str]:
    """
    Builds the compact unified JSON and PKL files for a given domain pair.
    Saves to cache/compact_<domain_pair>.pkl and cache/compact_<domain_pair>.json.
    Returns (pkl_path, json_path).
    """
    if domain_pair not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain pair: {domain_pair}. Available: {list(DOMAIN_CONFIGS.keys())}")

    cfg = DOMAIN_CONFIGS[domain_pair]
    os.makedirs(cache_dir, exist_ok=True)

    json_path = os.path.join(cache_dir, f"compact_{domain_pair}.json")
    pkl_path = os.path.join(cache_dir, f"compact_{domain_pair}.pkl")

    logger.info("=" * 80)
    logger.info(f"PREPROCESSING COMPACT METADATA: {domain_pair}")
    logger.info(f"Source: {cfg['source_display_name']} | Target: {cfg['target_display_name']}")
    logger.info(f"Output PKL:  {pkl_path}")
    logger.info(f"Output JSON: {json_path}")
    logger.info("=" * 80)

    # 1. Load interaction pairs from .inter files
    logger.info(f"Loading source interactions: {os.path.basename(cfg['source_inter'])}")
    src_users, src_items, src_pairs = load_inter_pairs(cfg["source_inter"])
    logger.info(f"  Source interactions: {len(src_pairs):,} pairs, {len(src_users):,} users, {len(src_items):,} items")

    logger.info(f"Loading target interactions: {os.path.basename(cfg['target_inter'])}")
    tgt_users, tgt_items, tgt_pairs = load_inter_pairs(cfg["target_inter"])
    logger.info(f"  Target interactions: {len(tgt_pairs):,} pairs, {len(tgt_users):,} users, {len(tgt_items):,} items")

    # 2. Extract item titles
    src_titles = scan_item_titles(cfg["source_item_meta"], src_items)
    tgt_titles = scan_item_titles(cfg["target_item_meta"], tgt_items)

    # 3. Extract reviews
    src_reviews = scan_reviews(cfg["source_review_meta"], src_pairs, src_users)
    tgt_reviews = scan_reviews(cfg["target_review_meta"], tgt_pairs, tgt_users)

    # 4. Serialize to Pickle (binary, loaded in < 0.1s)
    compact_obj = {
        "domain_pair": domain_pair,
        "source_name": cfg["source_name"],
        "target_name": cfg["target_name"],
        "source_items": src_titles,
        "target_items": tgt_titles,
        "source_reviews": src_reviews,  # Tuple[str, str] -> dict
        "target_reviews": tgt_reviews,  # Tuple[str, str] -> dict
    }

    logger.info(f"Saving binary compact cache to {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(compact_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 5. Serialize to JSON (human-readable, keys as "user_id:::item_id")
    logger.info(f"Saving JSON compact cache to {json_path}...")
    compact_json = {
        "domain_pair": domain_pair,
        "source_name": cfg["source_name"],
        "target_name": cfg["target_name"],
        "source_items": src_titles,
        "target_items": tgt_titles,
        "source_reviews": {f"{u}:::{i}": v for (u, i), v in src_reviews.items()},
        "target_reviews": {f"{u}:::{i}": v for (u, i), v in tgt_reviews.items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact_json, f, indent=2, ensure_ascii=False)

    pkl_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    json_size_mb = os.path.getsize(json_path) / (1024 * 1024)
    logger.info("=" * 80)
    logger.info(f"COMPACT METADATA SUCCESSFULLY CREATED:")
    logger.info(f"  PKL Cache:  {pkl_path} ({pkl_size_mb:.2f} MB)")
    logger.info(f"  JSON Cache: {json_path} ({json_size_mb:.2f} MB)")
    logger.info(f"  Source Items: {len(src_titles):,} | Source Reviews: {len(src_reviews):,}")
    logger.info(f"  Target Items: {len(tgt_titles):,} | Target Reviews: {len(tgt_reviews):,}")
    logger.info("=" * 80)

    return pkl_path, json_path
