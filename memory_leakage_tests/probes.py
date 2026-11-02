"""Construct ID-only requests; never pass a target answer to the model."""

import json

from .common import digest, valid_title

PROMPT_VERSION = "id-reconstruction-v1"
FIELDS = {"item": ["item_title"], "review": ["review_title", "review_text", "rating"]}


def answer_for(probe, item):
    if probe == "item":
        return {"item_title": item["item_title"]}
    return {"review_title": item["ground_truth_title"],
            "review_text": item["ground_truth_review"], "rating": item["ground_truth_rating"]}


def build_request(domain_pair, probe, mode, item_id, user_id=None, demonstrations=()):
    if mode not in ("zero_shot", "few_shot") or probe not in FIELDS:
        raise ValueError("Unknown probe or prompting mode")
    fields = FIELDS[probe]
    key = {"item_id": item_id}
    if probe == "review":
        if not user_id:
            raise ValueError("Review lookup needs original user_id")
        key["user_id"] = user_id
    system = (
        "You are an exact lookup interface for the Amazon Reviews dataset used in DGCDR, "
        "domain pair " + domain_pair + ". Keys are original Amazon user_id and parent_asin "
        "(item_id), not recommender-system internal indices. "
        + ("Retrieve the exact product title for the item_id. " if probe == "item" else
           "Retrieve the exact review title, review text and rating written by this user for this item. ")
        + "Return only a JSON object with fields: " + ", ".join(fields) + ". "
        "Use null for any unknown value. Do not invent, paraphrase, explain or infer likely values. "
        "Do not use tools or external search."
    )
    messages = [{"role": "system", "content": system}]
    if mode == "few_shot":
        if not demonstrations:
            raise ValueError("Few-shot mode needs reserved demonstrations")
        for demo in demonstrations:
            if demo["item_id"] == item_id or (user_id and demo["user_id"] == user_id):
                raise ValueError("Target ID appears in demonstrations")
            demo_key = {"item_id": demo["item_id"]}
            if probe == "review":
                demo_key["user_id"] = demo["user_id"]
            messages.extend([
                {"role": "user", "content": json.dumps(demo_key, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(answer_for(probe, demo), ensure_ascii=False)},
            ])
    messages.append({"role": "user", "content": json.dumps(key, ensure_ascii=False)})
    # No target-specific enums, constants, examples or defaults in this schema.
    properties = {field: {"type": ["number", "null"] if field == "rating" else ["string", "null"]}
                  for field in fields}
    return {"messages": messages, "format": {"type": "object", "properties": properties,
            "required": fields, "additionalProperties": False}}


def tasks_for(manifest, modes):
    """Yield a unique item task per cohort and every held-out user/item review."""
    for cohort in manifest["cohorts"]:
        seed = cohort["seed"]
        items = {}
        for user in cohort["users"]:
            for item in user["held_out_items"]:
                if valid_title(item):
                    items[item["item_id"]] = item
        for mode in modes:
            for iid, item in sorted(items.items()):
                yield make_task(manifest, seed, mode, "item", iid, None, item)
            for user in cohort["users"]:
                for item in sorted(user["held_out_items"], key=lambda value: value["item_id"]):
                    yield make_task(manifest, seed, mode, "review", item["item_id"], user["user_id"], item)
            for control in manifest["synthetic_controls"][str(seed)]:
                iid, uid = control["item_id"], control["user_id"]
                yield make_task(manifest, seed, mode, "item", iid, None, None, "synthetic_id")
                yield make_task(manifest, seed, mode, "review", iid, uid, None, "synthetic_id")


def make_task(manifest, seed, mode, probe, iid, uid, item, condition="real"):
    request = build_request(manifest["domain_pair"], probe, mode, iid, uid, manifest["demonstrations"])
    task = {"cohort_seed": seed, "mode": mode, "probe": probe,
            "item_id": iid, "user_id": uid, "condition": condition,
            "expected": answer_for(probe, item) if item else None, "request": request}
    task["task_id"] = digest(task)
    return task
