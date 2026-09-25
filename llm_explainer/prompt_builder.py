"""
Prompt builder module for generating LLM explanation prompts in English.
Formats user history across domains, presents recommended items from DGCDR,
enforces JSON response schema, and saves each prompt to a dedicated file.
"""

import json
import os
from typing import Dict, List, Optional, Tuple


TARGET_DOMAIN_PROFILES: Dict[str, Dict[str, str]] = {
    "Elec": {
        "domain_label": "Electronics",
        "concrete_attributes": (
            "e.g., battery longevity, sound clarity, connector sturdiness, cable length, "
            "screen resolution, portability, ease of setup, heat management"
        ),
        "few_shot_poor": (
            '"Based on your cross-domain profile transferred by the DGCDR model, this product matches '
            'your preferences for electronics. It satisfies the behavioral patterns detected in your previous purchases."'
        ),
        "few_shot_good": (
            '"Given your active daily routine and preference for durable, hassle-free gear, this 6ft cable '
            'gives you the extra reach needed to use your devices comfortably while charging without straining the connectors. '
            'The reinforced braided design and reliable 60W power delivery ensure long-lasting durability, whether at your desk or on the go."'
        ),
        "few_shot_note": "Mentions specific physical attributes (6ft length, reinforced connectors, 60W speed) that reflect what real users review in electronics.",
    },
    "Cloth": {
        # --- CASO DOMINIO CLOTHING (da completare in seguito) ---
        "domain_label": "Clothing, Shoes & Jewelry",
        "concrete_attributes": "",  # TODO: inserire attributi concreti per abbigliamento (es. fit, tessuto, comfort)
        "few_shot_poor": "",        # TODO: esempio negativo per abbigliamento
        "few_shot_good": "",        # TODO: esempio positivo per abbigliamento
        "few_shot_note": "",
    },
    "Sport": {
        # --- CASO DOMINIO SPORTS & OUTDOORS (da completare in seguito) ---
        "domain_label": "Sports & Outdoors",
        "concrete_attributes": "",  # TODO: inserire attributi concreti per sport (es. grip, traspirabilità, resistenza)
        "few_shot_poor": "",        # TODO: esempio negativo per sport
        "few_shot_good": "",        # TODO: esempio positivo per sport
        "few_shot_note": "",
    },
}

DEFAULT_DOMAIN_PROFILE: Dict[str, str] = {
    "domain_label": "Target Domain",
    "concrete_attributes": "",
    "few_shot_poor": "",
    "few_shot_good": "",
    "few_shot_note": "",
}


def resolve_target_domain_key(target_domain_name: str) -> str:
    """
    Identifies domain key ('Elec', 'Cloth', 'Sport') from display names
    like 'Electronics (Elec)', 'Cloth', etc.
    """
    name_upper = target_domain_name.upper()
    if "ELEC" in name_upper:
        return "Elec"
    elif "CLOTH" in name_upper:
        return "Cloth"
    elif "SPORT" in name_upper:
        return "Sport"
    return "Default"


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
    Builds a comprehensive, personalized prompt in English for a single user,
    dynamically tailored to the target domain.
    Also saves the prompt text to prompts_dir/prompt_<user_id>.txt if prompts_dir is provided.

    Returns:
        (prompt_text, saved_file_path)
    """
    domain_key = resolve_target_domain_key(target_domain_name)
    domain_profile = TARGET_DOMAIN_PROFILES.get(domain_key, DEFAULT_DOMAIN_PROFILE)

    lines = []
    lines.append("You are an expert personal shopping assistant and product specialist.")
    lines.append(
        f"Your goal is to write natural, compelling, and personalized product recommendations for items in {target_domain_name}."
    )
    lines.append("")
    lines.append("Scenario:")
    lines.append(f"- Source Domain: {source_domain_name}")
    lines.append(f"- Target Domain: {target_domain_name}")
    lines.append("")
    lines.append(
        "Below is the user's cross-domain profile. Pay special attention to what the user values in their reviews (e.g., build quality, durability, comfort, practical convenience, daily reliability):"
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
    lines.append("RECOMMENDED ITEMS TO EXPLAIN")
    lines.append("=" * 60)
    lines.append(
        f"The following items have been recommended for the user in {target_domain_name}:"
    )
    lines.append("")
    for idx, item in enumerate(recommended_items, 1):
        lines.append(f"{idx}. Item ID: {item['item_id']}")
        lines.append(f"   Item Title: {item.get('item_title', 'Unknown Title')}")
        lines.append("")

    # 4. Instructions and Guidelines
    lines.append("=" * 60)
    lines.append("INSTRUCTIONS & GUIDELINES")
    lines.append("=" * 60)
    lines.append(
        "For each recommended item, explain what practical features, ergonomic details, and performance qualities this specific user will appreciate."
    )
    lines.append("")
    lines.append("Follow these strict rules:")
    lines.append("1. FOCUS ON CONCRETE PRODUCT ATTRIBUTES:")
    attr_hint = domain_profile.get("concrete_attributes", "").strip()
    if attr_hint:
        lines.append(f"   - Anticipate what would make this user leave a 5-star review ({attr_hint}).")
    else:
        lines.append(f"   - Anticipate what would make this user leave a 5-star review for products in {target_domain_name}.")
    lines.append(
        "   - Relate these attributes to the traits the user consistently praised in past purchases (e.g., comfort, long-term durability, hassle-free daily setup)."
    )
    lines.append("")
    lines.append("2. NEGATIVE CONSTRAINTS (STRICTLY FORBIDDEN):")
    lines.append("   - Do NOT mention 'DGCDR', 'recommendation algorithm', 'cross-domain model', or 'graph'.")
    lines.append(
        "   - Do NOT start every sentence with repetitive meta-phrases like 'Based on your history' or 'The system recommended this because'."
    )
    lines.append("   - Speak directly, naturally, and conversationally to the user ('You\\'ll appreciate...', 'This offers...').")
    lines.append("")
    lines.append("3. LENGTH:")
    lines.append("   - Strictly 2 to 3 concise, punchy sentences per item.")
    lines.append("")

    # 5. Few-Shot Demonstrations (if defined for the target domain)
    few_shot_good = domain_profile.get("few_shot_good", "").strip()
    few_shot_poor = domain_profile.get("few_shot_poor", "").strip()
    few_shot_note = domain_profile.get("few_shot_note", "").strip()
    if few_shot_good:
        lines.append("=" * 60)
        lines.append("EXAMPLES OF EXPECTED STYLE (FEW-SHOT DEMONSTRATIONS)")
        lines.append("=" * 60)
        if few_shot_poor:
            lines.append("[POOR STYLE - DO NOT WRITE LIKE THIS]:")
            lines.append(few_shot_poor)
            lines.append("-> Reason: Empty algorithmic jargon, zero mentions of actual product features.")
            lines.append("")
        lines.append("[EXCELLENT STYLE - WRITE LIKE THIS]:")
        lines.append(few_shot_good)
        if few_shot_note:
            lines.append(f"-> Reason: {few_shot_note}")
        lines.append("")

    # 6. Output Format and JSON Schema
    lines.append("=" * 60)
    lines.append("OUTPUT FORMAT (JSON ONLY)")
    lines.append("=" * 60)
    lines.append(
        "CRITICAL: You must reply ONLY with a valid JSON object. Do not include any introductory or concluding text, explanations, or markdown code blocks outside the JSON. Use the following schema:"
    )
    lines.append("")

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

