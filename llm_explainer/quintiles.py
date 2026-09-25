"""
Quintile Stratification and Management Module for DGCDR.
Implements Scenario A:
- Excludes ultra-short uninformative reviews (default: < 5 words)
- Partitions the review length distribution into 5 equiprobable Quintiles (Q1-Q5, 20% each)
- Follows the Principle of Maximum Shannon Entropy (H = log2(5) = 2.3219 bits)
- Supports arbitrary Amazon domain pairs and caches calculated cutoffs.
"""

import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import BASE_DIR, DOMAIN_CONFIGS


def count_words(text: str) -> int:
    """Counts alphanumeric words, ignoring punctuation."""
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


class QuintileManager:
    """
    Manages computation, persistence, and classification of review length quintiles
    for any Amazon domain pair in DGCDR.
    """

    def __init__(
        self,
        domain_pair: str = "Cloth-Elec",
        min_words: int = 5,
        min_rating: float = 4.0,
        cache_dir: Optional[str] = None,
        use_title: bool = False,
    ):
        self.domain_pair = domain_pair
        self.min_words = min_words
        self.min_rating = min_rating
        self.use_title = use_title
        self.cache_dir = cache_dir or os.path.join(BASE_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cutoffs_file = os.path.join(
            self.cache_dir,
            f"quintiles_{self.domain_pair}_minw{self.min_words}{'_title' if self.use_title else ''}.json",
        )
        self.quintile_info = self._load_or_compute_quintiles()

    def _load_or_compute_quintiles(self) -> Dict:
        """Loads cached quintiles if existing, otherwise computes them from compact data."""
        if os.path.exists(self.cutoffs_file):
            with open(self.cutoffs_file, "r", encoding="utf-8") as f:
                return json.load(f)

        return self.compute_and_save_quintiles()

    def compute_and_save_quintiles(self) -> Dict:
        """
        Extracts review lengths from the compact cache of the target domain,
        computes the 20%, 40%, 60%, 80% percentiles, and saves the quintile schema.
        """
        pkl_path = os.path.join(self.cache_dir, f"compact_{self.domain_pair}.pkl")
        json_path = os.path.join(self.cache_dir, f"compact_{self.domain_pair}.json")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            tgt_revs = data.get("target_reviews", {})
        elif os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tgt_revs = data.get("target_reviews", {})
        else:
            raise FileNotFoundError(
                f"Compact cache not found for {self.domain_pair}. "
                f"Expected {pkl_path} or {json_path}. Run preprocess_compact_data.py first."
            )

        lengths = []
        for k, v in tgt_revs.items():
            if v.get("rating", 0) >= self.min_rating:
                txt = v.get("text", "")
                title = v.get("title", "")
                if self.use_title:
                    w = count_words(f"{title} {txt}".strip())
                else:
                    w = count_words(txt)

                if w >= self.min_words:
                    lengths.append(w)

        if not lengths:
            raise ValueError(f"No reviews found in {self.domain_pair} matching rating >= {self.min_rating} and words >= {self.min_words}")

        arr = np.sort(np.array(lengths))
        N = len(arr)

        # Exact 20%, 40%, 60%, 80% percentiles
        p20 = int(round(np.percentile(arr, 20.0)))
        p40 = int(round(np.percentile(arr, 40.0)))
        p60 = int(round(np.percentile(arr, 60.0)))
        p80 = int(round(np.percentile(arr, 80.0)))
        max_w = int(np.max(arr))

        # Enforce strict monotonically increasing boundaries
        p20 = max(self.min_words, p20)
        p40 = max(p20 + 1, p40)
        p60 = max(p40 + 1, p60)
        p80 = max(p60 + 1, p80)

        raw_ranges = [
            ("Q1", "Micro", self.min_words, p20),
            ("Q2", "Short", p20 + 1, p40),
            ("Q3", "Medium", p40 + 1, p60),
            ("Q4", "Detailed", p60 + 1, p80),
            ("Q5", "In-Depth", p80 + 1, max_w),
        ]

        quintiles_list = []
        for q_id, label, low, high in raw_ranges:
            if q_id == "Q5":
                count = int(np.sum(arr >= low))
            else:
                count = int(np.sum((arr >= low) & (arr <= high)))
            pct = round(count / N * 100, 2)
            quintiles_list.append(
                {
                    "quintile": q_id,
                    "label": label,
                    "min_words": low,
                    "max_words": high if q_id != "Q5" else None,
                    "count": count,
                    "percentage": pct,
                }
            )

        # Calculate empirical Shannon entropy
        probs = [q["count"] / N for q in quintiles_list]
        shannon_h = float(-sum(p * np.log2(p) for p in probs if p > 0))
        h_max = float(np.log2(5))

        result = {
            "domain_pair": self.domain_pair,
            "min_words_filter": self.min_words,
            "min_rating": self.min_rating,
            "use_title": self.use_title,
            "total_reviews": N,
            "shannon_entropy_bits": round(shannon_h, 4),
            "max_entropy_bits": round(h_max, 4),
            "entropy_efficiency_pct": round((shannon_h / h_max) * 100, 2),
            "quintiles": quintiles_list,
        }

        with open(self.cutoffs_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

    def classify_text(self, text: str, title: Optional[str] = None) -> Tuple[Optional[str], Optional[str], int]:
        """
        Classifies an input review text into its corresponding quintile (Q1 - Q5).
        Returns:
            (quintile_id, label, word_count)
            e.g. ("Q2", "Short", 22)
            If word_count < min_words, returns (None, "Filtered_UltraShort", word_count).
        """
        if self.use_title and title:
            full_text = f"{title} {text}".strip()
            w = count_words(full_text)
        else:
            w = count_words(text)

        if w < self.min_words:
            return None, "Filtered_UltraShort", w

        for q in self.quintile_info["quintiles"]:
            q_id = q["quintile"]
            low = q["min_words"]
            high = q["max_words"]
            if high is None:
                if w >= low:
                    return q_id, q["label"], w
            else:
                if low <= w <= high:
                    return q_id, q["label"], w

        # Fallback to Q5 if above all
        return "Q5", "In-Depth", w

    def get_cutoffs_summary(self) -> str:
        """Returns a formatted tabular string summarizing the quintiles."""
        lines = []
        lines.append(f"Domain Pair: {self.domain_pair} (Min Words: {self.min_words}, Min Rating: {self.min_rating})")
        lines.append(f"Total Evaluated Reviews: {self.quintile_info['total_reviews']:,}")
        lines.append(f"Shannon Entropy: {self.quintile_info['shannon_entropy_bits']:.4f} / {self.quintile_info['max_entropy_bits']:.4f} bits (Efficiency: {self.quintile_info['entropy_efficiency_pct']}%)")
        lines.append("-" * 75)
        lines.append(f"{'Quintile':<10} {'Label':<12} {'Range (Words)':<18} {'Count':<15} {'Percentage':<12}")
        lines.append("-" * 75)
        for q in self.quintile_info["quintiles"]:
            range_str = f"{q['min_words']} - {q['max_words']}w" if q['max_words'] else f">= {q['min_words']}w"
            lines.append(f"{q['quintile']:<10} {q['label']:<12} {range_str:<18} {q['count']:<15,d} {q['percentage']:<10.2f}%")
        lines.append("-" * 75)
        return "\n".join(lines)
