import sys
import numpy as np
import json
import torch
import os

np.float = float
np.int = int
np.bool = bool

from recbole.data.interaction import Interaction
from recbole.evaluator.metrics import Hit, Recall, NDCG, MRR
from recbole_cdr.quick_start.quick_start import load_data_and_model

def main():
    model_path = 'saved/DGCDR-Aug-06-2026_11-46-49-CDs_Instruments-42.pth'
    print(f"Loading {model_path}...")
    config, model, dataset, _, _, test_data = load_data_and_model(model_path)
    model.eval()

    # Get the very first user interaction from test data
    first_batch = next(iter(test_data))
    if isinstance(first_batch, tuple):
        # FullSortEvalDataLoader yields (interaction, history_index, positive_u, positive_i)
        interaction, history_index, positive_u, positive_i = first_batch
        user_interaction = interaction[0:1].to(config['device'])
        # Extract ground truth items for the first user
        target_uid = interaction['target_user_id'][0] if 'target_user_id' in interaction else interaction['user_id'][0]
        user_pos_items = positive_i[positive_u == target_uid].cpu().numpy()
    else:
        user_interaction = first_batch[0:1].to(config['device'])
        user_pos_items = np.array([])

    
    # Perform full sort prediction (scores for all items for this user)
    with torch.no_grad():
        scores = model.full_sort_predict(user_interaction)
    
    scores = scores.view(-1)
    
    # Exclude padding item (0)
    scores[0] = -np.inf
    
    # Retrieve the item ID map from the target dataset early for masking
    target_ds = dataset.target_domain_dataset
    target_uid_field = target_ds.uid_field
    target_iid_field = target_ds.iid_field
    
    # Extract Target Domain History
    target_inter_file = 'dataset/AmazonInstruments_AmazonCDs_commonUser_3-core/AmazonInstruments_AmazonCDs_commonUser_3-core.inter'
    target_item_ids = []
    with open(target_inter_file, 'r', encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0] == target_ds.field2id_token[target_uid_field][target_uid]:
                target_item_ids.append(parts[1])

    # Manually mask target domain history (excluding the test ground truth)
    for ext_id in target_item_ids:
        try:
            internal_idx = target_ds.token2id(target_iid_field, ext_id)
            if internal_idx not in user_pos_items:
                scores[internal_idx] = -np.inf
        except ValueError:
            pass

    # Get top 20 items
    topk_scores, topk_indices = torch.topk(scores, 20)
    topk_indices = topk_indices.cpu().numpy()

    # Map back to external ASIN tokens
    external_item_ids = [target_ds.field2id_token[target_iid_field][idx] for idx in topk_indices]
    user_pos_items_mapped = [target_ds.field2id_token[target_iid_field][idx] for idx in user_pos_items]
    ids_to_find = set(external_item_ids + user_pos_items_mapped + target_item_ids)
    
    # Read item metadata line by line
    item_metadata = {}
    print(f"Scanning item_metadata/meta_Musical_Instruments.jsonl for titles...")
    with open('item_metadata/meta_Musical_Instruments.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            asin = data.get('parent_asin', '')
            if asin in ids_to_find:
                item_metadata[asin] = data.get('title', 'Unknown Title')
                if len(item_metadata) == len(ids_to_find):
                    break
    
    # Calculate pos_index (boolean array of hits) and pos_len
    pos_index_bool = np.isin(topk_indices, user_pos_items).reshape(1, -1)
    pos_index_np = pos_index_bool.astype(int)
    pos_len_np = np.array([len(user_pos_items)])
    
    # Calculate Native Metrics (@20 is the last element of the array returned for the single user)
    hit_val = Hit(config).metric_info(pos_index_np)[0, -1].item()
    recall_val = Recall(config).metric_info(pos_index_np, pos_len_np)[0, -1].item()
    mrr_val = MRR(config).metric_info(pos_index_np)[0, -1].item()
    ndcg_val = NDCG(config).metric_info(pos_index_np, pos_len_np)[0, -1].item()

    external_uid = target_ds.field2id_token[target_uid_field][target_uid]

    output_lines = []
    output_lines.append("=" * 90)
    title_str = f"RANKING FOR USER {external_uid}"
    output_lines.append(f"{title_str:^90}")
    output_lines.append("=" * 90)
    output_lines.append(f" {'Rank':<4} | {'Relevant':<8} | {'Item ID':<10} | {'Title':<50}")
    output_lines.append("-" * 90)
    
    for i, (ext_id, is_hit) in enumerate(zip(external_item_ids, pos_index_bool[0]), 1):
        title = item_metadata.get(ext_id, 'Unknown Title')
        if len(title) > 47:
            title = title[:44] + "..."
        rel_str = "Yes" if is_hit else "No"
        output_lines.append(f" {i:<4} | {rel_str:<8} | {ext_id:<10} | {title:<50}")
    
    output_lines.append("=" * 90)
    output_lines.append("User Metrics (@20):")
    output_lines.append(f" - Hit Ratio: {hit_val:.4f}")
    output_lines.append(f" - Recall:    {recall_val:.4f}")
    output_lines.append(f" - MRR:       {mrr_val:.4f}")
    output_lines.append(f" - NDCG:      {ndcg_val:.4f}")
    output_lines.append("=" * 90)
    
    # === Extract Source Domain History ===
    
    source_inter_file = 'dataset/AmazonCDs_AmazonInstruments_commonUser_3-core/AmazonCDs_AmazonInstruments_commonUser_3-core.inter'
    source_item_ids = []
    
    with open(source_inter_file, 'r', encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0] == external_uid:
                source_item_ids.append(parts[1])
                
    source_metadata_file = 'item_metadata/meta_CDs_and_Vinyl.jsonl'
    source_item_metadata = {}
    ids_to_find_source = set(source_item_ids)
    
    print(f"Scanning {source_metadata_file} for titles...")
    with open(source_metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            asin = data.get('parent_asin', '')
            if asin in ids_to_find_source:
                source_item_metadata[asin] = data.get('title', 'Unknown Title')
                if len(source_item_metadata) == len(ids_to_find_source):
                    break
                    
    output_lines.append("==========================================================================================")
    output_lines.append(f"{'SOURCE DOMAIN HISTORY (AmazonCDs)':^90}")
    output_lines.append("==========================================================================================")
    output_lines.append(f" {'Item ID':<12} | {'Title':<72}")
    output_lines.append("-" * 90)
    
    for item_id in source_item_ids:
        title = source_item_metadata.get(item_id, 'Unknown Title')
        if len(title) > 69:
            title = title[:66] + "..."
        output_lines.append(f" {item_id:<12} | {title:<72}")
    
    # === Extract Target Domain History ===
    output_lines.append("==========================================================================================")
    output_lines.append(f"{'TARGET DOMAIN HISTORY (AmazonInstruments)':^90}")
    output_lines.append("==========================================================================================")
    output_lines.append(f" {'Item ID':<12} | {'Title':<72}")
    output_lines.append("-" * 90)
    
    for item_id in target_item_ids:
        title = item_metadata.get(item_id, 'Unknown Title')
        if len(title) > 69:
            title = title[:66] + "..."
        output_lines.append(f" {item_id:<12} | {title:<72}")
        
    output_lines.append("==========================================================================================")
    
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
