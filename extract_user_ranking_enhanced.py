import numpy as np
import json
import torch
import os
import random

np.float = float
np.int = int
np.bool = bool

from recbole.evaluator.metrics import Hit, Recall, NDCG, MRR
from recbole_cdr.quick_start.quick_start import load_data_and_model

MODEL_PATH = 'saved/DGCDR-Aug-06-2026_11-46-49-CDs_Instruments-42.pth'
TARGET_META_FILE = 'item_metadata/meta_Musical_Instruments.jsonl'
SOURCE_META_FILE = 'item_metadata/meta_CDs_and_Vinyl.jsonl'
TOPK = 20
HIDE_SEEN_ITEMS = True
NUM_USERS_TO_SAMPLE = 50

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
    # Titles are no longer truncated for markdown tables
    return title


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

    print("Collecting test users...")
    valid_users_info = []
    for interaction, history_index, positive_u, positive_i in test_data:
        rows_with_positives = torch.unique(positive_u).cpu().numpy()
        for row in rows_with_positives:
            user_interaction = interaction[row:row + 1]
            pos_items = positive_i[positive_u == row].cpu().numpy()
            target_uid = int(interaction[target_uid_field][row])
            valid_users_info.append({
                'user_interaction': user_interaction,
                'pos_items': pos_items,
                'target_uid': target_uid
            })

    # Randomly sample users (or fewer if not enough)
    random.seed(42)
    sample_size = min(NUM_USERS_TO_SAMPLE, len(valid_users_info))
    sampled_users = random.sample(valid_users_info, sample_size)
    print(f"Sampled {sample_size} users out of {len(valid_users_info)} available test users.")

    log_dir = "ranking_logs"
    users_dir = os.path.join(log_dir, "users")
    os.makedirs(users_dir, exist_ok=True)

    aggregate_metrics = {'hit': 0.0, 'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0}
    user_summaries = []

    for idx, user_info in enumerate(sampled_users, 1):
        target_uid = user_info['target_uid']
        user_interaction = user_info['user_interaction'].to(config['device'])
        user_pos_items = user_info['pos_items']
        external_uid = uid_to_token.get(target_uid, str(target_uid))

        print(f"Processing user {idx}/{sample_size}: {external_uid}...")

        with torch.no_grad():
            scores = model.full_sort_predict(user_interaction)
        scores = scores.view(-1)
        scores[0] = -np.inf  # padding item

        train_items = items_of_user(train_data.target_dataloader.dataset, target_uid_field, target_iid_field, target_uid)
        valid_items = items_of_user(valid_data.dataset, target_uid_field, target_iid_field, target_uid)
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

        item_metadata = load_titles(TARGET_META_FILE, set(external_item_ids + ground_truth_ids + seen_item_ids))
        
        shown = user_metrics(topk_indices, user_pos_items, config)
        native = user_metrics(recbole_indices, user_pos_items, config)

        # We'll use the 'shown' metrics for the summary
        metrics_to_use = shown if HIDE_SEEN_ITEMS else native
        for k in aggregate_metrics:
            aggregate_metrics[k] += metrics_to_use[k]

        user_file = f"user_{external_uid}.md"
        user_summaries.append({
            'external_uid': external_uid,
            'target_uid': target_uid,
            'hit': metrics_to_use['hit'],
            'ndcg': metrics_to_use['ndcg'],
            'file': user_file
        })

        output_lines = []
        title_str = f"RANKING FOR USER {external_uid} (internal id {target_uid})"
        output_lines.append(f"## {title_str}\n")
        output_lines.append("| Rank | Relevant | Item ID | Title |")
        output_lines.append("|---|---|---|---|")

        for i, ext_id in enumerate(external_item_ids, 1):
            title = truncate(item_metadata.get(ext_id, 'Unknown Title'), 47)
            rel_str = "Yes" if int(topk_indices[i - 1]) in user_pos_items else "No"
            output_lines.append(f"| {i} | {rel_str} | {ext_id} | {title} |")

        output_lines.append("\n<br>\n")
        output_lines.append(f"**User Metrics (@{TOPK}), over {len(user_pos_items)} held-out items:**\n")
        if HIDE_SEEN_ITEMS:
            output_lines.append("> **ranking above, already-seen items hidden:**")
            output_lines.append(f"> Hit Ratio: {shown['hit']:.4f} &nbsp;|&nbsp; Recall: {shown['recall']:.4f} &nbsp;|&nbsp; MRR: {shown['mrr']:.4f} &nbsp;|&nbsp; NDCG: {shown['ndcg']:.4f}")
            output_lines.append("> ")
            output_lines.append("> **RecBole protocol (repeatable=True, nothing masked):**")
        else:
            output_lines.append("> **RecBole protocol (repeatable=True, nothing masked):**")
            
        output_lines.append(f"> Hit Ratio: {native['hit']:.4f} &nbsp;|&nbsp; Recall: {native['recall']:.4f} &nbsp;|&nbsp; MRR: {native['mrr']:.4f} &nbsp;|&nbsp; NDCG: {native['ndcg']:.4f}")
        output_lines.append("\n<br>\n")
        output_lines.append("---\n")
        
        # === Held-out target items (the test ground truth) ===
        output_lines.append("## HELD-OUT TARGET ITEMS (test ground truth)\n")
        output_lines.append("| Item ID | Title |")
        output_lines.append("|---|---|")
        for ext_id in ground_truth_ids:
            output_lines.append(f"| {ext_id} | {truncate(item_metadata.get(ext_id, 'Unknown Title'), 72)} |")

        # === Source Domain History ===
        source_items = items_of_user(source_ds, source_uid_field, source_iid_field, target_uid)
        source_item_ids = [source_iid_to_token[i] for i in source_items]
        source_item_metadata = load_titles(SOURCE_META_FILE, set(source_item_ids))

        output_lines.append("\n<br>\n")
        output_lines.append("## SOURCE DOMAIN HISTORY (AmazonCDs)\n")
        output_lines.append("| Item ID | Title |")
        output_lines.append("|---|---|")
        for item_id in source_item_ids:
            output_lines.append(f"| {item_id} | {truncate(source_item_metadata.get(item_id, 'Unknown Title'), 72)} |")

        # === Target Domain History (train + valid, i.e. what the model was fitted on) ===
        output_lines.append("\n<br>\n")
        output_lines.append("## TARGET DOMAIN HISTORY (AmazonInstruments, train + valid)\n")
        output_lines.append("| Item ID | Title |")
        output_lines.append("|---|---|")
        for item_id in seen_item_ids:
            output_lines.append(f"| {item_id} | {truncate(item_metadata.get(item_id, 'Unknown Title'), 72)} |")

        final_output = "\n".join(output_lines)
        log_file_path = os.path.join(users_dir, user_file)
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(final_output)

    # Generate Summary file
    for k in aggregate_metrics:
        aggregate_metrics[k] /= sample_size

    summary_lines = []
    summary_lines.append("# Qualitative Analysis Summary")
    summary_lines.append(f"\nRandomly sampled **{sample_size}** users from the test dataset (Seed 42).")
    summary_lines.append("\n## Aggregate Metrics (Sample Average)\n")
    summary_lines.append(f"- **Hit Ratio**: {aggregate_metrics['hit']:.4f}")
    summary_lines.append(f"- **Recall**: {aggregate_metrics['recall']:.4f}")
    summary_lines.append(f"- **MRR**: {aggregate_metrics['mrr']:.4f}")
    summary_lines.append(f"- **NDCG**: {aggregate_metrics['ndcg']:.4f}")
    
    summary_lines.append("\n## User Directory\n")
    summary_lines.append("| User ID (External) | Internal ID | Hit Ratio | NDCG | Link |")
    summary_lines.append("|---|---|---|---|---|")
    for us in user_summaries:
        summary_lines.append(f"| {us['external_uid']} | {us['target_uid']} | {us['hit']:.4f} | {us['ndcg']:.4f} | [View Report](users/{us['file']}) |")
    
    summary_file_path = os.path.join(log_dir, "analysis_summary.md")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nSaved analysis summary to {summary_file_path} and individual user reports to {users_dir}/")


if __name__ == '__main__':
    main()
