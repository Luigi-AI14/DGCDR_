"""
Prompt builder module for generating LLM explanation prompts in English.
Formats user history across domains, presents recommended items from DGCDR,
enforces JSON response schema, and saves each prompt to a dedicated file.
"""

import json
import os
from typing import Dict, List, Optional, Tuple


def build_user_prompt(
    user_id: str,
    source_domain_name: str,
    target_domain_name: str,
    source_history: List[Dict],
    target_history: List[Dict],
    recommended_items: List[Dict],
    prompts_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Builds a comprehensive, personalized prompt in English for a single user.
    Also saves the prompt text to prompts_dir/prompt_<user_id>.txt if prompts_dir is provided.

    Returns:
        (prompt_text, saved_file_path)
    """
    lines = []
    lines.append("You are an expert AI assistant specializing in Recommender Systems and explanation generation.")
    lines.append(
        "Your task is to generate convincing, natural, and personalized explanations for recommendations produced by the DGCDR (Disentangled Graph Cross Domain Recommender) model."
    )
    lines.append("")
    lines.append(
        "DGCDR is an advanced cross-domain recommendation system that models user preferences by disentangling domain-specific and domain-shared behavioral graphs, effectively transferring user preference patterns from a source domain to a target domain."
    )
    lines.append("")
    lines.append("Scenario configuration:")
    lines.append(f"- Source Domain: {source_domain_name}")
    lines.append(f"- Target Domain: {target_domain_name}")
    lines.append("")
    lines.append(
        "Below is the user's cross-domain historical profile consisting of previously purchased items with high user ratings (rating >= 4.0), including item IDs, item titles, ratings, review titles, and review texts."
    )
    lines.append("")
    lines.append("=" * 60)
    lines.append("USER INTERACTION HISTORY")
    lines.append(f"User ID: {user_id}")
    lines.append("=" * 60)
    lines.append("")

    # 1. Source Domain History (100% train+valid)
    lines.append(f"--- Source Domain: {source_domain_name} ---")
    if not source_history:
        lines.append("No historical reviews recorded in this domain.")
    else:
        for idx, item in enumerate(source_history, 1):
            lines.append(f"{idx}. Item ID: {item['item_id']}")
            lines.append(f"   Item Title: {item.get('item_title', 'Unknown Title')}")
            lines.append(f"   User Rating: {float(item.get('rating', 5.0)):.1f} / 5.0")
            rev_title = item.get("review_title", "").strip() or "No review title"
            lines.append(f"   Review Title: {rev_title}")
            rev_text = item.get("review_text", "").strip() or "No review text"
            lines.append(f"   Review Text: {rev_text}")
            lines.append("")

    # 2. Target Domain History (80% train+valid)
    lines.append(f"--- Target Domain: {target_domain_name} ---")
    if not target_history:
        lines.append("No prior historical interactions recorded in this domain.")
    else:
        for idx, item in enumerate(target_history, 1):
            lines.append(f"{idx}. Item ID: {item['item_id']}")
            lines.append(f"   Item Title: {item.get('item_title', 'Unknown Title')}")
            lines.append(f"   User Rating: {float(item.get('rating', 5.0)):.1f} / 5.0")
            rev_title = item.get("review_title", "").strip() or "No review title"
            lines.append(f"   Review Title: {rev_title}")
            rev_text = item.get("review_text", "").strip() or "No review text"
            lines.append(f"   Review Text: {rev_text}")
            lines.append("")

    # 3. Recommended Items (The 20% held-out target items)
    lines.append("=" * 60)
    lines.append("RECOMMENDED ITEMS BY DGCDR MODEL")
    lines.append("=" * 60)
    lines.append(
        f"Based on the user's cross-domain profile and preference representations, the DGCDR (Disentangled Graph Cross Domain Recommender) model has recommended the following items in {target_domain_name}:"
    )
    lines.append("")
    for idx, item in enumerate(recommended_items, 1):
        lines.append(f"{idx}. Item ID: {item['item_id']}")
        lines.append(f"   Item Title: {item.get('item_title', 'Unknown Title')}")
        lines.append("")

    # 4. Instructions and JSON Output Schema
    lines.append("=" * 60)
    lines.append("INSTRUCTIONS FOR EXPLANATIONS")
    lines.append("=" * 60)
    lines.append("For each recommended item listed above:")
    lines.append(
        "1. Provide a concise, clear, and personalized explanation in English explaining WHY the DGCDR model recommended this item to this specific user."
    )
    lines.append(
        "2. Connect the recommendation to the user's historical preferences and behavioral traits (e.g., preference for high comfort, durable build quality, ease of use, reliable daily performance)."
    )
    lines.append(
        "3. Explain how the item's key features satisfy the user's implicit and explicit needs transferred across domains."
    )
    lines.append(
        "4. Keep each explanation concise and focused (strictly 2 to 3 sentences per item)."
    )
    lines.append(
        "5. Directly produce the final JSON immediately without preamble."
    )
    lines.append("")
    lines.append(
        "CRITICAL: You must reply ONLY with a valid JSON object. Do not include any introductory or concluding text, explanations, or markdown code blocks outside the JSON. Use the following schema:"
    )

    # Build schema example matching actual recommended items
    example_explanations = []
    for it in recommended_items:
        example_explanations.append(
            {
                "item_id": it["item_id"],
                "item_title": it.get("item_title", ""),
                "explanation": f"<Write the personalized explanation in English for item {it['item_id']} here>",
            }
        )
    example_json = json.dumps({"explanations": example_explanations}, indent=2)
    lines.append(example_json)

    prompt_text = "\n".join(lines)

    # Save prompt to file if directory is specified
    saved_path = None
    if prompts_dir:
        os.makedirs(prompts_dir, exist_ok=True)
        # Clean user_id for filename
        safe_uid = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        saved_path = os.path.join(prompts_dir, f"prompt_{safe_uid}.txt")
        with open(saved_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

    return prompt_text, saved_path
