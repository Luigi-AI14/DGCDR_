"""
Configuration module for LLM explanation generation and validation on DGCDR.
Provides dataset paths, RecBole config files, and metadata mapping per domain pair.
"""

import os

# Base directory of the repository
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAIN_CONFIGS = {
    "Cloth-Elec": {
        "source_name": "Cloth",
        "target_name": "Elec",
        "source_display_name": "Clothing, Shoes & Jewelry (Cloth)",
        "target_display_name": "Electronics (Elec)",
        "source_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonCloth_AmazonElec_commonUser_10-core", "AmazonCloth_AmazonElec_commonUser_10-core.inter"
        ),
        "target_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonElec_AmazonCloth_commonUser_10-core", "AmazonElec_AmazonCloth_commonUser_10-core.inter"
        ),
        "source_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Cloth.jsonl"),
        "target_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Elec.jsonl"),
        "source_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Cloth.jsonl"),
        "target_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Elec.jsonl"),
        "config_file_list": [
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "overall.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "dataset", "AmazonCloth_AmazonElec_commonUser_10-core.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "model", "DGCDR.yaml"),
        ],
    },
    "Elec-Cloth": {
        "source_name": "Elec",
        "target_name": "Cloth",
        "source_display_name": "Electronics (Elec)",
        "target_display_name": "Clothing, Shoes & Jewelry (Cloth)",
        "source_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonElec_AmazonCloth_commonUser_10-core", "AmazonElec_AmazonCloth_commonUser_10-core.inter"
        ),
        "target_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonCloth_AmazonElec_commonUser_10-core", "AmazonCloth_AmazonElec_commonUser_10-core.inter"
        ),
        "source_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Elec.jsonl"),
        "target_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Cloth.jsonl"),
        "source_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Elec.jsonl"),
        "target_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Cloth.jsonl"),
        "config_file_list": [
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "overall.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "dataset", "AmazonElec_AmazonCloth_commonUser_10-core.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "model", "DGCDR.yaml"),
        ],
    },
    "Cloth-Sport": {
        "source_name": "Cloth",
        "target_name": "Sport",
        "source_display_name": "Clothing, Shoes & Jewelry (Cloth)",
        "target_display_name": "Sports & Outdoors (Sport)",
        "source_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonCloth_AmazonSport_commonUser_5-core", "AmazonCloth_AmazonSport_commonUser_5-core.inter"
        ),
        "target_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonSport_AmazonCloth_commonUser_5-core", "AmazonSport_AmazonCloth_commonUser_5-core.inter"
        ),
        "source_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Cloth.jsonl"),
        "target_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Sport.jsonl"),
        "source_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Cloth.jsonl"),
        "target_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Sport.jsonl"),
        "config_file_list": [
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "overall.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "dataset", "AmazonCloth_AmazonSport_commonUser_5-core.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "model", "DGCDR.yaml"),
        ],
    },
    "Sport-Cloth": {
        "source_name": "Sport",
        "target_name": "Cloth",
        "source_display_name": "Sports & Outdoors (Sport)",
        "target_display_name": "Clothing, Shoes & Jewelry (Cloth)",
        "source_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonSport_AmazonCloth_commonUser_5-core", "AmazonSport_AmazonCloth_commonUser_5-core.inter"
        ),
        "target_inter": os.path.join(
            BASE_DIR, "dataset", "AmazonCloth_AmazonSport_commonUser_5-core", "AmazonCloth_AmazonSport_commonUser_5-core.inter"
        ),
        "source_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Sport.jsonl"),
        "target_item_meta": os.path.join(BASE_DIR, "item_metadata", "item_meta_Cloth.jsonl"),
        "source_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Sport.jsonl"),
        "target_review_meta": os.path.join(BASE_DIR, "review_metadata", "review_Cloth.jsonl"),
        "config_file_list": [
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "overall.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "dataset", "AmazonSport_AmazonCloth_commonUser_5-core.yaml"),
            os.path.join(BASE_DIR, "recbole_cdr", "properties", "model", "DGCDR.yaml"),
        ],
    },
}

DEFAULT_SETTINGS = {
    "domain_pair": "Cloth-Elec",
    "seed": 42,
    "temperature": 0.0,
    "rating_threshold": 4.0,
    "num_users": 5,
    "model_name": "qwen3.5:9b",
    "sbert_model": "all-MiniLM-L6-v2",
    "min_review_words": 5,
    "ollama_url": "http://localhost:11434",
    "prompts_dir": os.path.join(BASE_DIR, "saved_prompts"),
    "output_dir": os.path.join(BASE_DIR, "results"),
}

