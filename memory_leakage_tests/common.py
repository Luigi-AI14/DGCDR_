"""Small, standard-library-only helpers shared by preparation and inference."""

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize(value):
    """Ignore Unicode compatibility, case and whitespace; preserve punctuation."""
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def word_count(value):
    return len(re.findall(r"\b\w+\b", value or ""))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    """Atomically replace a JSON artifact, including on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def valid_title(item):
    title = item.get("item_title")
    return bool(normalize(title)) and normalize(title) != normalize("Item " + item["item_id"])


def validate_dataset(dataset):
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema_version")
    if not dataset.get("domain_pair") or not isinstance(dataset.get("split_seed"), int):
        raise ValueError("Dataset needs domain_pair and integer split_seed")
    seen_users = set()
    titles = {}
    for user in dataset["users"]:
        uid = user["user_id"]
        if not isinstance(uid, str) or not uid or uid in seen_users:
            raise ValueError("Missing or duplicate original user ID")
        seen_users.add(uid)
        seen_items = set()
        if not user["held_out_items"]:
            raise ValueError("Every user must have held-out records")
        for item in user["held_out_items"]:
            iid = item["item_id"]
            if not isinstance(iid, str) or not iid or iid in seen_items:
                raise ValueError("Missing or duplicate held-out item for user " + uid)
            seen_items.add(iid)
            if not normalize(item.get("ground_truth_review")):
                raise ValueError("Missing review text for " + uid + "/" + iid)
            if not isinstance(item.get("ground_truth_title"), str):
                raise ValueError("Review titles must be strings (empty is allowed)")
            rating = item.get("ground_truth_rating")
            if isinstance(rating, bool) or not isinstance(rating, (float, int)) or not 1 <= rating <= 5:
                raise ValueError("Invalid review rating")
            if valid_title(item):
                if iid in titles and normalize(titles[iid]) != normalize(item["item_title"]):
                    raise ValueError("Conflicting titles for original item ID " + iid)
                titles[iid] = item["item_title"]
    if not seen_users:
        raise ValueError("No eligible users")
