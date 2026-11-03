"""
Configuration module for LLM explanation generation and validation on DGCDR.
Provides dataset paths, RecBole config files, and metadata mapping per domain pair.
"""

import os
from llm_explainer.result_paths import EXPLANATIONS_ROOT
from llm_explainer.cohorts import COHORT_SEEDS, DEFAULT_INFERENCE_SEED

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
    "seed": COHORT_SEEDS[0],
    "temperature": 0.0,
    "rating_threshold": 4.0,
    "num_users": 200,
    "model_name": "qwen3.5:9b",
    "sbert_model": "BAAI/bge-large-en-v1.5",
    "no_cache_llm": True,
    "no_cache_sbert": True,
    "min_review_words": 5,
    "use_title_in_quintiles": True,
    "num_ctx": 32768,
    "ollama_url": "http://localhost:11434",
    "prompts_dir": os.path.join(BASE_DIR, "saved_prompts"),
    "save_prompts": False,
    "output_dir": os.path.join(BASE_DIR, EXPLANATIONS_ROOT),
    "prompt_version": "v3",
    "generation_mode": "few-shot",
    "no_few_shot": False,
    "bertscore_model": "roberta-large",
    "bertscore_batch_size": 16,
    "bertscore_rescale_with_baseline": True,
}

# Pipeline and standalone entry points use these same defaults.
GENERATOR_MODELS = (DEFAULT_SETTINGS["model_name"], "llama3.1:8b", "gemma4:12b")
JUDGE_DEFAULTS = {
    "model": "qwen3.8:27b",
    "num_users": DEFAULT_SETTINGS["num_users"],
    "metrics": ["alignment"],
    "judge_prompt": "zero-shot",
    "instance_prompt": os.path.join(BASE_DIR, "llm_explainer", "prompts", "JUDGE_INSTANCE_PROMPT.txt"),
    "temperature": DEFAULT_SETTINGS["temperature"],
    "seed": DEFAULT_INFERENCE_SEED,
    "num_ctx": 8192,
    "max_tokens": 256,
    "max_retries": 1,
    "top_logprobs": 20,
    "no_weighted_scoring": False,
    "num_gpu": None,
    "ollama_url": DEFAULT_SETTINGS["ollama_url"],
}

