# DGCDR Cross-Domain LLM Explanation & Validation Framework

This framework enables generating personalized cross-domain recommendation explanations with a local LLM (**Qwen 3.5 9B** via Ollama) for the **DGCDR** (*Disentangled Graph Cross Domain Recommender*) model, and validating those explanations against actual user reviews on held-out items using syntactic metrics (**BLEU**, **ROUGE-1**, **ROUGE-2**, **ROUGE-L**).

---

## 📁 Project Architecture

```
DGCDR_/
├── llm_explainer/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Domain pairs, file paths, default parameters
│   ├── metrics.py            # Sentence BLEU (smoothing) & ROUGE-1/2/L (P, R, F1)
│   ├── ollama_client.py      # Ollama REST client (supports Qwen 3.5 9B with 16k context)
│   ├── prompt_builder.py     # Prompt formatter in English + prompt file writer
│   ├── data_extractor.py     # RecBole split loader (seed 42), text filter, instant cache loader
│   └── compact_builder.py    # Preprocessor for lightweight compact metadata & reviews
├── saved_prompts/            # Saved prompt text files for every evaluated user
│   └── prompt_<user_id>.txt
├── results/                  # Detailed validation JSON reports
│   └── validation_<domain_pair>_<N>users_<timestamp>.json
├── cache/                    # Compact dataset caches (compact_<pair>.pkl & compact_<pair>.json)
├── preprocess_compact_data.py # CLI script to build compact datasets (< 1s load time)
├── run_llm_validation.py     # Main CLI entry point
└── README_LLM_VALIDATION.md  # Documentation and usage guide
```

---

## 🚀 Quick Start

### 1. Requirements & Conda Environment
Use the provided `dgcdr38` conda environment:
```powershell
conda activate dgcdr38
```

Ensure Ollama is running with Qwen 3.5 9B:
```powershell
ollama serve
# Verify model availability
ollama list
```

---

### 2. Running the Validation

#### Single user test:
```powershell
python run_llm_validation.py --num_users 1
```

#### Multi-user evaluation (e.g., 5 users):
```powershell
python run_llm_validation.py --num_users 5
```

#### Change domain pair (e.g., Clothing -> Sports & Outdoors):
```powershell
python run_llm_validation.py --domain_pair Cloth-Sport --num_users 3
```

#### Preprocessing Compact Datasets (< 1s load time):
To build or update the compact cache for a domain pair:
```powershell
python preprocess_compact_data.py --domain_pair Cloth-Elec
```
Or for all pairs at once:
```powershell
python preprocess_compact_data.py --all
```
This generates `cache/compact_<pair>.pkl` and `cache/compact_<pair>.json`, enabling instant sub-second loading for any number of users.

---

## ⚙️ CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num_users` | `int` | `1` | Number of test users to evaluate |
| `--domain_pair` | `str` | `Cloth-Elec` | Cross-domain pair (`Cloth-Elec`, `Elec-Cloth`, `Cloth-Sport`, `Sport-Cloth`) |
| `--seed` | `int` | `42` | Random seed matching RecBole split reproducibility |
| `--temperature` | `float` | `0.0` | LLM sampling temperature (0.0 for deterministic reproducibility) |
| `--rating_threshold` | `float` | `4.0` | Minimum rating for interaction history and held-out items |
| `--model` | `str` | `qwen3.5:9b` | Local Ollama model name |
| `--ollama_url` | `str` | `http://localhost:11434` | Ollama API endpoint |
| `--prompts_dir` | `str` | `saved_prompts` | Directory where user prompts are saved as `.txt` |
| `--output_dir` | `str` | `results` | Directory where validation reports are saved as `.json` |
| `--dry_run` | `flag` | `False` | Run without querying the LLM |

---

## 📝 Generated Output Structure

### 1. Saved Prompts (`saved_prompts/prompt_<user_id>.txt`)
Each prompt includes:
- **DGCDR Model Definition**: Full name (*Disentangled Graph Cross Domain Recommender*) and cross-domain disentangled preference representation.
- **Source Domain History**: 100% (train + valid) items with rating $\ge 4.0$, title, review title, and review text.
- **Target Domain History**: 80% (train + valid) items with rating $\ge 4.0$, title, review title, and review text.
- **Recommended Items**: 20% target domain held-out items presented blindly as recommendations from DGCDR.
- **Strict English Instructions & JSON Schema**.

### 2. Validation Report (`results/validation_*.json`)
```json
{
  "timestamp": "2026-09-24T11:00:42.251",
  "domain_pair": "Cloth-Elec",
  "source_domain": "Clothing, Shoes & Jewelry (Cloth)",
  "target_domain": "Electronics (Elec)",
  "model": "qwen3.5:9b",
  "seed": 42,
  "rating_threshold": 4.0,
  "num_users": 1,
  "total_held_out_items_evaluated": 2,
  "users": [
    {
      "user_id": "AE22AMGLNKESKOBMOMY7C2SURQTA",
      "prompt_file": ".../saved_prompts/prompt_AE22AMGLNKESKOBMOMY7C2SURQTA.txt",
      "items": [
        {
          "id_item": "B08R3D8R9W",
          "item_title": "GoPro HERO6 Black ...",
          "user_review_text": "It has clear image, and i really love it",
          "llm_explanation": "This recommendation leverages your demonstrated interest in waterproof and durable equipment...",
          "metrics": {
            "bleu": 0.0033,
            "rouge1_f1": 0.0308,
            "rouge1_p": 0.0179,
            "rouge1_r": 0.1111,
            "rouge2_f1": 0.0,
            "rouge2_p": 0.0,
            "rouge2_r": 0.0,
            "rougeL_f1": 0.0308,
            "rougeL_p": 0.0179,
            "rougeL_r": 0.1111
          }
        }
      ],
      "user_averages": {
        "avg_bleu": 0.0043,
        "avg_rouge1_f1": 0.0859,
        "avg_rouge2_f1": 0.0,
        "avg_rougeL_f1": 0.068
      }
    }
  ],
  "global_averages": {
    "macro_avg_bleu": 0.0043,
    "macro_avg_rouge1_f1": 0.0859,
    "macro_avg_rouge2_f1": 0.0,
    "macro_avg_rougeL_f1": 0.068
  }
}
```
