"""Check complete user histories before sampling cohorts; never edit review text."""

import hashlib
import logging
from pathlib import Path

from llm_explainer.config import BASE_DIR, DEFAULT_SETTINGS, GENERATOR_MODELS
from llm_explainer.ollama_client import GENERATION_OUTPUT_TOKENS, generation_messages
from llm_explainer.prompt_builder import (
    GENERATION_MODES, build_item_prompt, generation_prompt_metadata,
)

logger = logging.getLogger(__name__)

# Public tokenizer files only: no model weights or model inference are needed.
# Pin revisions so preparing the same experiment later uses the same vocabulary.
TOKENIZERS = {
    "qwen3.5:9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "llama3.1:8b": ("Xenova/Meta-Llama-3.1-Tokenizer", "d49ea33e96b3fcf8738bb79ff9d85da81742fa82"),
    "gemma4:12b": ("google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d"),
}
CHAT_SAFETY_TOKENS = 256


def budget_settings(prompt_version="v3", num_ctx=None):
    num_ctx = DEFAULT_SETTINGS["num_ctx"] if num_ctx is None else num_ctx
    input_budget = num_ctx - GENERATION_OUTPUT_TOKENS - CHAT_SAFETY_TOKENS
    if input_budget < 1:
        raise ValueError("num_ctx must exceed the output reserve plus chat safety margin (4352 tokens)")
    # Check both modes so zero-shot and few-shot runs can share the frozen users.
    return dict(version=1, num_ctx=num_ctx, output_tokens=GENERATION_OUTPUT_TOKENS,
                chat_safety_tokens=CHAT_SAFETY_TOKENS, input_budget=input_budget,
                prompt_version=prompt_version,
                prompt_hashes={mode: generation_prompt_metadata(prompt_version, mode)["cache_sha256"]
                               for mode in GENERATION_MODES},
                tokenizers={model: dict(repo=TOKENIZERS[model][0], revision=TOKENIZERS[model][1])
                            for model in GENERATOR_MODELS})


class HistoryBudget:
    def __init__(self, settings):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from tokenizers import Tokenizer

        self.settings = settings
        self.tokenizers = {}
        self.metadata = {}
        for model, source in settings["tokenizers"].items():
            logger.info("Loading history-budget tokenizer for %s", model)
            try:
                download = dict(repo_id=source["repo"], filename="tokenizer.json", revision=source["revision"],
                                cache_dir=str(Path(BASE_DIR) / "cache/generation_tokenizers"))
                try:
                    path = hf_hub_download(**download, local_files_only=True)
                except LocalEntryNotFoundError:
                    path = hf_hub_download(**download)
                tokenizer = Tokenizer.from_file(path)
                tokenizer.no_truncation()
                tokenizer.no_padding()
            except Exception as exc:
                raise RuntimeError(f"Cannot load the tokenizer for {model}; check tokenizer dependencies "
                                   "and Hugging Face connectivity/cache. No word-count fallback is used.") from exc
            self.tokenizers[model] = tokenizer
            self.metadata[model] = dict(source, resolved_revision=Path(path).parent.name,
                                       sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def check_user(self, user):
        """Count every candidate in both modes; held-out review text is never used."""
        maxima = {model: dict(input_tokens=0, item_id=None, generation_mode=None)
                  for model in self.tokenizers}
        for mode in GENERATION_MODES:
            for item in user["held_out_items"]:
                prompt, _ = build_item_prompt(
                    user_id=user["user_id"], source_domain_name=user["source_domain"],
                    target_domain_name=user["target_domain"], source_history=user["source_history"],
                    target_history=user["target_history"],
                    target_item=dict(item_id=item["item_id"], item_title=item["item_title"]),
                    prompt_version=self.settings["prompt_version"], generation_mode=mode,
                    save_prompts=False)
                messages = generation_messages(prompt)
                for model, tokenizer in self.tokenizers.items():
                    count = sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids)
                                for m in messages)
                    if count > maxima[model]["input_tokens"]:
                        maxima[model] = dict(input_tokens=count, item_id=item["item_id"], generation_mode=mode)
        fits = all(result["input_tokens"] <= self.settings["input_budget"] for result in maxima.values())
        return dict(user_id=user["user_id"], eligible=fits, max_input_tokens_by_model=maxima,
                    reason=None if fits else "prompt_exceeds_input_budget")
