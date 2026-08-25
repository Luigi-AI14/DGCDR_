import numpy as np
import json
import os
import random
import sys

# This script lives three levels below the repository root, so running it by
# path puts its own directory on sys.path instead of the root, and neither
# recbole_cdr nor extensions resolves. Put the root back before importing them.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     os.pardir, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

np.float = float
np.int = int
np.bool = bool

from recbole.evaluator.metrics import Hit, Recall, NDCG, MRR
from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.ranking_reports.core.ranking_payload import (
    domain_names,
    user_json,
)

MODEL_PATH = 'saved/DGCDR-Aug-06-2026_17-41-44.pth'
TARGET_META_FILE = 'meta_Musical_Instruments.jsonl'
SOURCE_META_FILE = 'meta_CDs_and_Vinyl.jsonl'
TOPK = 20

# Same protocol note as extract_user_ranking.py: the model is evaluated with
# repeatable=True, so RecBole masks nothing but the padding item. Hiding the
# already-seen items reads better in a demo but moves the numbers off that
# protocol, so both rankings are scored for every user.
HIDE_SEEN_ITEMS = True
NUM_USERS_TO_SAMPLE = 50
SEED = 42


def id2token(dataset, field):
    """Internal id -> external token.

    ``field2id_token`` cannot be indexed by id: ``get_keys_from_chainmap_by_order``
    keeps insertion order and '[PAD]' is inserted last, so every entry of an
    overlapping section is shifted by one. ``field2token_id`` is the map the
    interactions were remapped with, so invert that instead.
    """
    return {internal: token for token, internal in dataset.field2token_id[field].items()}


def items_of_user(dataset, uid_field, iid_field, uid):
    """Item ids the user interacts with in one split, in file order."""
    users = dataset.inter_feat[uid_field].numpy()
    items = dataset.inter_feat[iid_field].numpy()
    return [int(i) for u, i in zip(users, items) if int(u) == uid]


def resolve_meta(name):
    """Path of a metadata dump: it sits inside a directory of the same name."""
    for candidate in (os.path.join(name, name), name):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"metadata dump not found: tried {os.path.join(name, name)} and {name}")


def load_titles(meta_file, wanted):
    """ASIN -> title, reading the metadata dump one line at a time.

    The dumps are hundreds of MB, so this must stay a single streamed pass over
    the ASINs of *every* sampled user -- calling it once per user re-reads the
    whole file once per user.
    """
    titles = {}
    if not wanted:
        return titles
    path = resolve_meta(meta_file)
    print(f"Scanning {path} for {len(wanted)} titles...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated line must not abandon the whole scan
            asin = data.get('parent_asin', '')
            if asin in wanted:
                titles[asin] = data.get('title', 'Unknown Title')
                if len(titles) == len(wanted):
                    break
    return titles


def user_metrics(ranked_ids, positives, config):
    """Hit/Recall/MRR/NDCG @TOPK for a single user, via RecBole's own metrics."""
    pos_index = np.isin(ranked_ids, positives).reshape(1, -1).astype(int)
    pos_len = np.array([len(positives)])
    return {
        'hit': Hit(config).metric_info(pos_index)[0, -1].item(),
        'recall': Recall(config).metric_info(pos_index, pos_len)[0, -1].item(),
        'mrr': MRR(config).metric_info(pos_index)[0, -1].item(),
        'ndcg': NDCG(config).metric_info(pos_index, pos_len)[0, -1].item(),
    }


def md_cell(text):
    """Titles are not truncated, but a raw '|' or newline would break the table."""
    return str(text).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ').strip()


def render_user(record, item_titles, source_titles):
    """The per-user markdown report, identical in shape to extract_user_ranking.py."""
    shown, native = record['shown'], record['native']
    lines = []
    lines.append(f"## RANKING FOR USER {record['external_uid']} (internal id {record['target_uid']})\n")
    lines.append("| Rank | Relevant | Item ID | Title |")
    lines.append("|---|---|---|---|")
    for rank, (ext_id, is_relevant) in enumerate(zip(record['ranked_ids'], record['ranked_relevant']), 1):
        title = md_cell(item_titles.get(ext_id, 'Unknown Title'))
        lines.append(f"| {rank} | {'Yes' if is_relevant else 'No'} | {ext_id} | {title} |")

    lines.append("\n<br>\n")
    lines.append(f"**User Metrics (@{TOPK}), over {record['n_positives']} held-out items:**\n")
    if HIDE_SEEN_ITEMS:
        lines.append("> **ranking above, already-seen items hidden:**")
        lines.append(f"> Hit Ratio: {shown['hit']:.4f} &nbsp;|&nbsp; Recall: {shown['recall']:.4f} "
                     f"&nbsp;|&nbsp; MRR: {shown['mrr']:.4f} &nbsp;|&nbsp; NDCG: {shown['ndcg']:.4f}")
        lines.append("> ")
    lines.append("> **RecBole protocol (repeatable=True, nothing masked):**")
    lines.append(f"> Hit Ratio: {native['hit']:.4f} &nbsp;|&nbsp; Recall: {native['recall']:.4f} "
                 f"&nbsp;|&nbsp; MRR: {native['mrr']:.4f} &nbsp;|&nbsp; NDCG: {native['ndcg']:.4f}")
    lines.append("\n<br>\n")
    lines.append("---\n")

    # === Held-out target items (the test ground truth) ===
    lines.append("## HELD-OUT TARGET ITEMS (test ground truth)\n")
    lines.append("| Item ID | Title |")
    lines.append("|---|---|")
    for ext_id in record['ground_truth_ids']:
        lines.append(f"| {ext_id} | {md_cell(item_titles.get(ext_id, 'Unknown Title'))} |")

    # === Source Domain History ===
    lines.append("\n<br>\n")
    lines.append("## SOURCE DOMAIN HISTORY (AmazonCDs)\n")
    lines.append("| Item ID | Title |")
    lines.append("|---|---|")
    for ext_id in record['source_item_ids']:
        lines.append(f"| {ext_id} | {md_cell(source_titles.get(ext_id, 'Unknown Title'))} |")

    # === Target Domain History (train + valid, i.e. what the model was fitted on) ===
    lines.append("\n<br>\n")
    lines.append("## TARGET DOMAIN HISTORY (AmazonInstruments, train + valid)\n")
    lines.append("| Item ID | Title |")
    lines.append("|---|---|")
    for ext_id in record['seen_item_ids']:
        lines.append(f"| {ext_id} | {md_cell(item_titles.get(ext_id, 'Unknown Title'))} |")

    return "\n".join(lines)


def main():
    print(f"Loading {MODEL_PATH}...")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(MODEL_PATH)
    model.eval()

    target_ds = dataset.target_domain_dataset
    source_ds = dataset.source_domain_dataset
    target_uid_field, target_iid_field = target_ds.uid_field, target_ds.iid_field
    source_uid_field, source_iid_field = source_ds.uid_field, source_ds.iid_field

    uid_to_token = id2token(target_ds, target_uid_field)
    target_iid_to_token = id2token(target_ds, target_iid_field)
    source_iid_to_token = id2token(source_ds, source_iid_field)

    # --- pick the users first, so only the sample gets scored -----------------
    # FullSortEvalDataLoader yields (interaction, history_index, positive_u,
    # positive_i), where positive_u indexes the *row of the batch*, not the user
    # id. This first pass touches no model, so it is cheap.
    print("Collecting test users...")
    all_uids = []
    for interaction, _, positive_u, _ in test_data:
        for row in torch.unique(positive_u).cpu().numpy():
            all_uids.append(int(interaction[target_uid_field][int(row)]))

    random.seed(SEED)
    sample_size = min(NUM_USERS_TO_SAMPLE, len(all_uids))
    sampled_uids = set(random.sample(all_uids, sample_size))
    print(f"Sampled {sample_size} users out of {len(all_uids)} available test users.")

    # --- score the sampled users --------------------------------------------
    records = []
    for interaction, _, positive_u, positive_i in test_data:
        for row in torch.unique(positive_u).cpu().numpy():
            row = int(row)
            target_uid = int(interaction[target_uid_field][row])
            if target_uid not in sampled_uids:
                continue

            user_interaction = interaction[row:row + 1].to(config['device'])
            user_pos_items = positive_i[positive_u == row].cpu().numpy()
            external_uid = uid_to_token.get(target_uid, str(target_uid))
            print(f"Processing user {len(records) + 1}/{sample_size}: {external_uid}...")

            # Scores for every target-domain item, exactly as at evaluation time.
            with torch.no_grad():
                scores = model.full_sort_predict(user_interaction)
            scores = scores.view(-1)
            scores[0] = -np.inf  # padding item

            # The user's own history: everything the split hid from the test set.
            train_items = items_of_user(train_data.target_dataloader.dataset,
                                        target_uid_field, target_iid_field, target_uid)
            valid_items = items_of_user(valid_data.dataset,
                                        target_uid_field, target_iid_field, target_uid)
            seen_items = train_items + [i for i in valid_items if i not in train_items]

            shown_scores = scores.clone()
            if HIDE_SEEN_ITEMS:
                for item_id in seen_items:
                    if item_id not in user_pos_items:
                        shown_scores[item_id] = -np.inf

            topk_values, topk_indices = torch.topk(shown_scores, TOPK)
            topk_indices = topk_indices.cpu().numpy()
            topk_values = topk_values.cpu().numpy()
            recbole_indices = torch.topk(scores, TOPK)[1].cpu().numpy()

            source_items = items_of_user(source_ds, source_uid_field, source_iid_field, target_uid)

            records.append({
                'target_uid': target_uid,
                'external_uid': external_uid,
                'n_positives': len(user_pos_items),
                'ranked_ids': [target_iid_to_token[int(i)] for i in topk_indices],
                'ranked_scores': [float(s) for s in topk_values],
                'ranked_relevant': [int(i) in user_pos_items for i in topk_indices],
                'ground_truth_ids': [target_iid_to_token[int(i)] for i in user_pos_items],
                'seen_item_ids': [target_iid_to_token[i] for i in seen_items],
                'source_item_ids': [source_iid_to_token[i] for i in source_items],
                'shown': user_metrics(topk_indices, user_pos_items, config),
                'native': user_metrics(recbole_indices, user_pos_items, config),
            })

    if not records:
        print("No test user could be scored; nothing to write.")
        return

    # --- one pass per metadata dump, for every sampled user at once ----------
    wanted_target = set()
    wanted_source = set()
    for record in records:
        wanted_target.update(record['ranked_ids'])
        wanted_target.update(record['ground_truth_ids'])
        wanted_target.update(record['seen_item_ids'])
        wanted_source.update(record['source_item_ids'])

    item_titles = load_titles(TARGET_META_FILE, wanted_target)
    source_titles = load_titles(SOURCE_META_FILE, wanted_source)

    # --- write the reports ---------------------------------------------------
    log_dir = "ranking_logs"
    users_dir = os.path.join(log_dir, "users")
    os.makedirs(users_dir, exist_ok=True)

    metric_key = 'shown' if HIDE_SEEN_ITEMS else 'native'
    domains = domain_names(config, SOURCE_META_FILE, TARGET_META_FILE)
    aggregate = {'hit': 0.0, 'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0}
    summaries = []
    for record in records:
        metrics = record[metric_key]
        for key in aggregate:
            aggregate[key] += metrics[key]

        user_file = f"user_{record['external_uid']}.md"
        json_file = f"user_{record['external_uid']}.json"
        summaries.append({'external_uid': record['external_uid'],
                          'target_uid': record['target_uid'],
                          'hit': metrics['hit'], 'ndcg': metrics['ndcg'],
                          'file': user_file, 'json_file': json_file})

        with open(os.path.join(users_dir, user_file), "w", encoding="utf-8") as f:
            f.write(render_user(record, item_titles, source_titles))

        with open(os.path.join(users_dir, json_file), "w", encoding="utf-8") as f:
            json.dump(user_json(record, item_titles, source_titles, domains,
                                TOPK, HIDE_SEEN_ITEMS),
                      f, indent=2, ensure_ascii=False)

    for key in aggregate:
        aggregate[key] /= len(records)

    protocol = ("already-seen items hidden" if HIDE_SEEN_ITEMS
                else "RecBole protocol, repeatable=True, nothing masked")
    summary_lines = [
        "# Qualitative Analysis Summary",
        f"\nRandomly sampled **{len(records)}** users from the test dataset (Seed {SEED}).",
        f"\n## Aggregate Metrics @{TOPK} (sample average, {protocol})\n",
        f"- **Hit Ratio**: {aggregate['hit']:.4f}",
        f"- **Recall**: {aggregate['recall']:.4f}",
        f"- **MRR**: {aggregate['mrr']:.4f}",
        f"- **NDCG**: {aggregate['ndcg']:.4f}",
        "\n## User Directory\n",
        "| User ID (External) | Internal ID | Hit Ratio | NDCG | Link |",
        "|---|---|---|---|---|",
    ]
    for entry in summaries:
        summary_lines.append(f"| {entry['external_uid']} | {entry['target_uid']} | "
                             f"{entry['hit']:.4f} | {entry['ndcg']:.4f} | "
                             f"[View Report](users/{entry['file']}) |")

    summary_file_path = os.path.join(log_dir, "analysis_summary.md")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    summary_json_path = os.path.join(log_dir, "analysis_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            'run': {
                'model_path': MODEL_PATH,
                'topk': TOPK,
                'hide_seen_items': HIDE_SEEN_ITEMS,
                'seed': SEED,
                'n_users': len(records),
                'metric_protocol': protocol,
                'source_domain': domains['source'],
                'target_domain': domains['target'],
            },
            'aggregate_metrics': aggregate,
            'users': summaries,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved analysis summary to {summary_file_path} / {summary_json_path} "
          f"and individual user reports to {users_dir}/")


if __name__ == '__main__':
    main()
