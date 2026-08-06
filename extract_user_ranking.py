import numpy as np
import json
import torch
import os

np.float = float
np.int = int
np.bool = bool

from recbole.evaluator.metrics import Hit, Recall, NDCG, MRR
from recbole_cdr.quick_start.quick_start import load_data_and_model

MODEL_PATH = 'saved/DGCDR-Aug-06-2026_17-41-44.pth'
TARGET_META_FILE = 'meta_Musical_Instruments.jsonl/meta_Musical_Instruments.jsonl'
SOURCE_META_FILE = 'meta_CDs_and_Vinyl.jsonl/meta_CDs_and_Vinyl.jsonl'
TOPK = 20

# This model is evaluated with repeatable=True, so the sampler hands the loader an
# empty history and RecBole masks nothing but the padding item: train and valid items
# compete in the ranking and count as misses. Hiding them reads better in a demo but
# moves the numbers off that protocol, so both rankings are scored below.
HIDE_SEEN_ITEMS = True


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


def load_titles(meta_file, wanted):
    """ASIN -> title, reading the metadata dump one line at a time."""
    titles = {}
    print(f"Scanning {meta_file} for titles...")
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
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


def truncate(title, width):
    return title if len(title) <= width else title[:width - 3] + "..."


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

    # The very first user of the test data, i.e. row 0 of the first batch.
    # FullSortEvalDataLoader yields (interaction, history_index, positive_u, positive_i),
    # where positive_u indexes the *row of the batch*, not the user id.
    row = 0
    interaction, history_index, positive_u, positive_i = next(iter(test_data))
    user_interaction = interaction[row:row + 1].to(config['device'])
    target_uid = int(interaction[target_uid_field][row])
    external_uid = uid_to_token[target_uid]
    user_pos_items = positive_i[positive_u == row].cpu().numpy()

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

    topk_indices = torch.topk(shown_scores, TOPK)[1].cpu().numpy()
    recbole_indices = torch.topk(scores, TOPK)[1].cpu().numpy()

    external_item_ids = [target_iid_to_token[int(i)] for i in topk_indices]
    ground_truth_ids = [target_iid_to_token[int(i)] for i in user_pos_items]
    seen_item_ids = [target_iid_to_token[i] for i in seen_items]

    item_metadata = load_titles(TARGET_META_FILE,
                                set(external_item_ids + ground_truth_ids + seen_item_ids))

    shown = user_metrics(topk_indices, user_pos_items, config)
    native = user_metrics(recbole_indices, user_pos_items, config)

    output_lines = []
    output_lines.append("=" * 90)
    title_str = f"RANKING FOR USER {external_uid} (internal id {target_uid})"
    output_lines.append(f"{title_str:^90}")
    output_lines.append("=" * 90)
    output_lines.append(f" {'Rank':<4} | {'Relevant':<8} | {'Item ID':<10} | {'Title':<50}")
    output_lines.append("-" * 90)

    for i, ext_id in enumerate(external_item_ids, 1):
        title = truncate(item_metadata.get(ext_id, 'Unknown Title'), 47)
        rel_str = "Yes" if int(topk_indices[i - 1]) in user_pos_items else "No"
        output_lines.append(f" {i:<4} | {rel_str:<8} | {ext_id:<10} | {title:<50}")

    output_lines.append("=" * 90)
    output_lines.append(f"User Metrics (@{TOPK}), over {len(user_pos_items)} held-out items:")
    if HIDE_SEEN_ITEMS:
        output_lines.append(" ranking above, already-seen items hidden:")
        output_lines.append(f" - Hit Ratio: {shown['hit']:.4f}   Recall: {shown['recall']:.4f}   "
                            f"MRR: {shown['mrr']:.4f}   NDCG: {shown['ndcg']:.4f}")
        output_lines.append(" RecBole protocol (repeatable=True, nothing masked):")
    output_lines.append(f" - Hit Ratio: {native['hit']:.4f}   Recall: {native['recall']:.4f}   "
                        f"MRR: {native['mrr']:.4f}   NDCG: {native['ndcg']:.4f}")
    output_lines.append("=" * 90)

    # === Held-out target items (the test ground truth) ===
    output_lines.append(f"{'HELD-OUT TARGET ITEMS (test ground truth)':^90}")
    output_lines.append("=" * 90)
    output_lines.append(f" {'Item ID':<12} | {'Title':<72}")
    output_lines.append("-" * 90)
    for ext_id in ground_truth_ids:
        output_lines.append(f" {ext_id:<12} | {truncate(item_metadata.get(ext_id, 'Unknown Title'), 72):<72}")

    # === Source Domain History ===
    source_items = items_of_user(source_ds, source_uid_field, source_iid_field, target_uid)
    source_item_ids = [source_iid_to_token[i] for i in source_items]
    source_item_metadata = load_titles(SOURCE_META_FILE, set(source_item_ids))

    output_lines.append("=" * 90)
    output_lines.append(f"{'SOURCE DOMAIN HISTORY (AmazonCDs)':^90}")
    output_lines.append("=" * 90)
    output_lines.append(f" {'Item ID':<12} | {'Title':<72}")
    output_lines.append("-" * 90)
    for item_id in source_item_ids:
        output_lines.append(f" {item_id:<12} | {truncate(source_item_metadata.get(item_id, 'Unknown Title'), 72):<72}")

    # === Target Domain History (train + valid, i.e. what the model was fitted on) ===
    output_lines.append("=" * 90)
    output_lines.append(f"{'TARGET DOMAIN HISTORY (AmazonInstruments, train + valid)':^90}")
    output_lines.append("=" * 90)
    output_lines.append(f" {'Item ID':<12} | {'Title':<72}")
    output_lines.append("-" * 90)
    for item_id in seen_item_ids:
        output_lines.append(f" {item_id:<12} | {truncate(item_metadata.get(item_id, 'Unknown Title'), 72):<72}")

    output_lines.append("=" * 90)

    final_output = "\n".join(output_lines)
    print("\n" + final_output)

    log_dir = "ranking_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "user_ranking_output.txt")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\nSaved formatted ranking to {log_file_path}")


if __name__ == '__main__':
    main()
