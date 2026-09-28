"""
Evaluation metrics module: ROUGE (1, 2, L) and BLEU calculation.
Self-contained, mathematically rigorous implementation with graceful edge-case handling.
"""

import hashlib
import logging
import math
import os
import pickle
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    # Tokenize words, removing punctuation
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a list of tokens."""
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def compute_rouge_n(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> Dict[str, float]:
    """
    Compute ROUGE-N (Precision, Recall, F1).
    Overlap is calculated using minimum counts of matching n-grams.
    """
    cand_ngrams = get_ngrams(candidate_tokens, n)
    ref_ngrams = get_ngrams(reference_tokens, n)

    cand_len = max(0, len(candidate_tokens) - n + 1)
    ref_len = max(0, len(reference_tokens) - n + 1)

    if cand_len == 0 or ref_len == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = 0
    for ngram, count in cand_ngrams.items():
        if ngram in ref_ngrams:
            overlap += min(count, ref_ngrams[ngram])

    prec = overlap / cand_len if cand_len > 0 else 0.0
    rec = overlap / ref_len if ref_len > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": prec, "recall": rec, "f1": f1}


def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the length of Longest Common Subsequence (LCS) using dynamic programming."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    # Memory optimized DP with 2 rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)

    return curr[n]


def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> Dict[str, float]:
    """Compute ROUGE-L based on Longest Common Subsequence (LCS)."""
    cand_len = len(candidate_tokens)
    ref_len = len(reference_tokens)

    if cand_len == 0 or ref_len == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(candidate_tokens, reference_tokens)
    prec = lcs / cand_len
    rec = lcs / ref_len
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"precision": prec, "recall": rec, "f1": f1}


def compute_bleu(
    candidate_tokens: List[str], reference_tokens: List[str], max_n: int = 4, weights: Tuple[float, ...] = None
) -> float:
    """
    Compute sentence-level BLEU score with smoothing (Chen and Cherry, 2014, Method 1).
    Smoothing adds a small epsilon when higher order n-gram precision is 0.
    """
    c_len = len(candidate_tokens)
    r_len = len(reference_tokens)

    if c_len == 0 or r_len == 0:
        return 0.0

    # Brevity Penalty (BP)
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(r_len) / float(c_len))

    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)

    p_ns = []
    smooth_applied = False

    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate_tokens, n)
        ref_ngrams = get_ngrams(reference_tokens, n)
        tot_cand_ngrams = max(0, c_len - n + 1)

        if tot_cand_ngrams == 0:
            p_ns.append(0.0)
            continue

        overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items() if ngram in ref_ngrams)

        if overlap > 0:
            p_ns.append(overlap / tot_cand_ngrams)
        else:
            # Smoothing Method 1: replace 0 with small value epsilon
            smooth_applied = True
            p_ns.append(0.1 / tot_cand_ngrams)

    # Compute weighted geometric mean
    log_sum = 0.0
    for w, p in zip(weights, p_ns):
        if p <= 0:
            return 0.0
        log_sum += w * math.log(p)

    bleu = bp * math.exp(log_sum)
    return float(bleu)



SBERT_ALIASES: Dict[str, str] = {
    # BAAI BGE models
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "bge-large-en": "BAAI/bge-large-en",
    "bge-base-en": "BAAI/bge-base-en",
    "bge-small-en": "BAAI/bge-small-en",
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
    "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
    # Microsoft / Intfloat E5 models
    "e5-large-v2": "intfloat/e5-large-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-large": "intfloat/e5-large",
    "e5-base": "intfloat/e5-base",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    # Alibaba GTE models
    "gte-large": "Alibaba-NLP/gte-large-en-v1.5",
    "gte-base": "Alibaba-NLP/gte-base-en-v1.5",
    "gte-large-en-v1.5": "Alibaba-NLP/gte-large-en-v1.5",
    "gte-base-en-v1.5": "Alibaba-NLP/gte-base-en-v1.5",
}


def validate_and_resolve_sbert_model(model_name: str) -> str:
    """
    Validates that a Sentence-BERT model identifier exists (locally or on Hugging Face Hub).
    Automatically maps common shorthand aliases (e.g. 'bge-large-en-v1.5' -> 'BAAI/bge-large-en-v1.5').
    Raises ValueError with a helpful message if the model cannot be found or accessed.
    """
    raw_name = model_name.strip()
    # 1. Resolve shorthand alias if known
    target_name = SBERT_ALIASES.get(raw_name, raw_name)

    # 2. If it is an existing local directory, it's valid
    if os.path.isdir(target_name):
        return target_name

    # 3. Check Hugging Face Hub config existence
    try:
        from transformers import AutoConfig

        candidate_names = [target_name]
        if "/" not in target_name:
            candidate_names.append(f"sentence-transformers/{target_name}")

        last_error = None
        for cand in candidate_names:
            try:
                AutoConfig.from_pretrained(cand)
                return cand
            except Exception as e:
                last_error = e

        # Build helpful error message
        alias_matches = [
            f"'{k}' -> '{v}'"
            for k, v in SBERT_ALIASES.items()
            if raw_name.lower() in k.lower() or k.lower() in raw_name.lower()
        ]
        suggestion_str = (
            f"\nDid you mean one of these known aliases:\n  " + "\n  ".join(alias_matches)
            if alias_matches
            else ""
        )

        raise ValueError(
            f"Sentence-BERT model '{model_name}' could not be resolved or downloaded from Hugging Face Hub.\n"
            f"Underlying error: {last_error}"
            f"{suggestion_str}\n\n"
            "Tip: Models from organizations other than 'sentence-transformers' must include the "
            "organization prefix, for example:\n"
            "  --sbert_model BAAI/bge-large-en-v1.5\n"
            "  --sbert_model intfloat/e5-large-v2\n"
            "  --sbert_model sentence-transformers/all-MiniLM-L6-v2"
        )
    except ImportError:
        return target_name


_DEFAULT_SBERT_MODEL = None


def get_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Lazy-loads and caches the Sentence-BERT model (singleton pattern).
    Uses CUDA if available, falling back gracefully to CPU.
    """
    global _DEFAULT_SBERT_MODEL
    if _DEFAULT_SBERT_MODEL is None:
        resolved_name = validate_and_resolve_sbert_model(model_name)
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                _DEFAULT_SBERT_MODEL = SentenceTransformer(resolved_name, device=device)
            except Exception:
                # Fallback to CPU if device initialization issues occur
                _DEFAULT_SBERT_MODEL = SentenceTransformer(resolved_name, device="cpu")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic evaluation. "
                "Install it with 'pip install sentence-transformers'."
            )
    return _DEFAULT_SBERT_MODEL


def format_reference_with_title(review_text: str, review_title: Optional[str] = None) -> str:
    """
    Combines user review title and review body text into a coherent sentence/paragraph
    for semantic embedding comparison.
    """
    title = (review_title or "").strip()
    text = (review_text or "").strip()
    if title and text:
        if title[-1] in ".!?,:;":
            return f"{title} {text}"
        return f"{title}. {text}"
    elif title:
        return title
    return text


def compute_sbert_similarity(
    candidate_text: str,
    reference_text: str,
    model: Optional[Any] = None,
) -> float:
    """
    Computes cosine similarity between candidate explanation and reference review.
    Returns:
        similarity: float in [-1.0, 1.0] (typically [0.0, 1.0] for topical text).
    """
    cand = candidate_text.strip() if candidate_text else ""
    ref = reference_text.strip() if reference_text else ""

    if not cand or not ref:
        return 0.0

    if model is None:
        model = get_sbert_model()

    import numpy as np

    embs = model.encode([cand, ref], normalize_embeddings=True, show_progress_bar=False)
    sim = float(np.dot(embs[0], embs[1]))
    return round(float(sim), 4)


def batch_compute_sbert_similarity(
    candidate_texts: List[str],
    reference_texts: List[str],
    model: Optional[Any] = None,
    batch_size: int = 64,
) -> List[float]:
    """
    Batch computes cosine similarities between pairs of candidates and references.
    """
    if not candidate_texts or not reference_texts:
        return []

    if len(candidate_texts) != len(reference_texts):
        raise ValueError("candidate_texts and reference_texts must have the same length.")

    if model is None:
        model = get_sbert_model()

    import numpy as np

    cand_embs = model.encode(candidate_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    ref_embs = model.encode(reference_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

    sims = np.sum(cand_embs * ref_embs, axis=1)
    return [round(float(s), 4) for s in sims]


def compute_syntactic_metrics(
    candidate_text: str,
    reference_text: str,
) -> Dict[str, float]:
    """
    Computes syntactic metrics (BLEU, ROUGE-1, ROUGE-2, ROUGE-L) between
    candidate explanation and reference review.
    Fast CPU-only token operations.
    """
    cand_tokens = tokenize(candidate_text)
    ref_tokens = tokenize(reference_text)

    r1 = compute_rouge_n(cand_tokens, ref_tokens, n=1)
    r2 = compute_rouge_n(cand_tokens, ref_tokens, n=2)
    rl = compute_rouge_l(cand_tokens, ref_tokens)
    bleu = compute_bleu(cand_tokens, ref_tokens)

    return {
        "bleu": round(bleu, 4),
        "rouge1_f1": round(r1["f1"], 4),
        "rouge1_p": round(r1["precision"], 4),
        "rouge1_r": round(r1["recall"], 4),
        "rouge2_f1": round(r2["f1"], 4),
        "rouge2_p": round(r2["precision"], 4),
        "rouge2_r": round(r2["recall"], 4),
        "rougeL_f1": round(rl["f1"], 4),
        "rougeL_p": round(rl["precision"], 4),
        "rougeL_r": round(rl["recall"], 4),
    }


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for safe filesystem naming across OS platforms."""
    cleaned = model_name.replace(":", "_").replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9._-]", "_", cleaned)


class ReferenceEmbeddingCache:
    """
    Persistent on-disk cache for reference review Sentence-BERT embeddings.
    Since held-out target reviews remain constant across runs with fixed seed/splits,
    pre-computing and caching them avoids redundant GPU encoding when testing different
    LLM models, prompts, or temperatures.
    """

    def __init__(
        self,
        domain_pair: str,
        sbert_model_name: str,
        cache_dir: Optional[str] = None,
    ):
        self.domain_pair = domain_pair
        self.sbert_model_name = sbert_model_name
        self.clean_model_name = sanitize_model_name(sbert_model_name)
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cache_dir = os.path.join(base_dir, "cache", "embeddings")
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(
            self.cache_dir, f"ref_emb_{self.domain_pair}_{self.clean_model_name}.pkl"
        )
        self._cache: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(
                    f"Loaded {len(self._cache)} cached reference embeddings from {self.cache_file}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load reference embedding cache from {self.cache_file}: {e}. Starting with empty cache."
                )
                self._cache = {}

    def save(self) -> None:
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} reference embeddings to cache: {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save reference embedding cache to {self.cache_file}: {e}")

    @staticmethod
    def get_text_key(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get_or_compute_batch(
        self,
        reference_texts: List[str],
        sbert_model: Any,
        batch_size: int = 64,
        use_cache: bool = True,
    ) -> Any:
        """
        Retrieves normalized embeddings for a list of reference texts.
        Uses in-memory & disk cache for already known texts, and encodes missing texts
        in batches with the provided Sentence-BERT model.
        Returns:
            np.ndarray of shape (len(reference_texts), embedding_dim)
        """
        import numpy as np

        if not reference_texts:
            return np.empty((0, 0))

        keys = [self.get_text_key(t) for t in reference_texts]

        # Identify missing texts that need encoding
        missing_keys_to_text: Dict[str, str] = {}
        for k, t in zip(keys, reference_texts):
            if not use_cache or k not in self._cache:
                if k not in missing_keys_to_text:
                    missing_keys_to_text[k] = t.strip()

        if missing_keys_to_text:
            num_hits = len(reference_texts) - len(missing_keys_to_text)
            logger.info(
                f"Reference review embeddings: {num_hits} loaded from cache, "
                f"{len(missing_keys_to_text)} missing (encoding in batch on device: {getattr(sbert_model, 'device', 'cpu')})..."
            )
            unique_missing_keys = list(missing_keys_to_text.keys())
            unique_missing_texts = [missing_keys_to_text[k] for k in unique_missing_keys]

            new_embs = sbert_model.encode(
                unique_missing_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(unique_missing_texts) > 50,
            )

            for k, emb in zip(unique_missing_keys, new_embs):
                self._cache[k] = emb

            if use_cache:
                self.save()
        else:
            logger.info(
                f"Reference review embeddings: 100% cache hit ({len(reference_texts)}/{len(reference_texts)} loaded from disk cache)."
            )

        result = [self._cache[k] for k in keys]
        return np.array(result)


def batch_compute_candidate_embeddings(
    candidate_texts: List[str],
    sbert_model: Any,
    batch_size: int = 64,
) -> Any:
    """
    Computes normalized embeddings for candidate explanations in batches.
    Handles empty/whitespace candidate texts gracefully by returning zero vectors.
    """
    import numpy as np

    if not candidate_texts:
        return np.empty((0, 0))

    valid_indices = []
    valid_texts = []
    for idx, text in enumerate(candidate_texts):
        cleaned = text.strip() if text else ""
        if cleaned:
            valid_indices.append(idx)
            valid_texts.append(cleaned)

    if not valid_texts:
        dim = (
            sbert_model.get_sentence_embedding_dimension()
            if hasattr(sbert_model, "get_sentence_embedding_dimension")
            else 384
        )
        return np.zeros((len(candidate_texts), dim), dtype=np.float32)

    logger.info(
        f"Encoding {len(valid_texts)} candidate explanations in batches (batch_size={batch_size}, device: {getattr(sbert_model, 'device', 'cpu')})..."
    )
    encoded_valid = sbert_model.encode(
        valid_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(valid_texts) > 50,
    )

    dim = encoded_valid.shape[1]
    all_embs = np.zeros((len(candidate_texts), dim), dtype=np.float32)
    for idx, emb in zip(valid_indices, encoded_valid):
        all_embs[idx] = emb

    return all_embs


def batch_compute_cosine_similarities(
    cand_embs: Any,
    ref_embs: Any,
    candidate_texts: Optional[List[str]] = None,
) -> List[float]:
    """
    Computes vector cosine similarity between normalized candidate and reference embeddings.
    Empty candidate explanations receive a score of 0.0.
    """
    import numpy as np

    if cand_embs.shape[0] != ref_embs.shape[0]:
        raise ValueError(
            f"Shape mismatch: {cand_embs.shape[0]} candidates vs {ref_embs.shape[0]} references"
        )

    if cand_embs.shape[0] == 0:
        return []

    dots = np.sum(cand_embs * ref_embs, axis=1)

    results = []
    for i, sim in enumerate(dots):
        if candidate_texts is not None and (not candidate_texts[i] or not candidate_texts[i].strip()):
            results.append(0.0)
        else:
            clamped = float(np.clip(sim, -1.0, 1.0))
            results.append(round(clamped, 4))
    return results


def evaluate_explanation_vs_review(
    candidate_text: str,
    reference_text: str,
    reference_title: Optional[str] = None,
    sbert_model: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Evaluate candidate explanation against ground-truth user review.
    Computes syntactic metrics (BLEU, ROUGE-1, ROUGE-2, ROUGE-L) and semantic
    similarity via Sentence-BERT (incorporating the review title).

    Returns:
        bleu: float
        rouge1_f1: float
        rouge1_p: float
        rouge1_r: float
        rouge2_f1: float
        rouge2_p: float
        rouge2_r: float
        rougeL_f1: float
        rougeL_p: float
        rougeL_r: float
        sbert_similarity: float
    """
    metrics = compute_syntactic_metrics(candidate_text, reference_text)
    semantic_ref = format_reference_with_title(reference_text, reference_title)
    sbert_sim = compute_sbert_similarity(candidate_text, semantic_ref, model=sbert_model)
    metrics["sbert_similarity"] = round(sbert_sim, 4)
    return metrics

