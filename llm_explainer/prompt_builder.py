"""
Prompt builder module for generating LLM explanation prompts in English.
Formats user history across domains, presents recommended items from DGCDR,
enforces JSON response schema, and saves each prompt to a dedicated file.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


TARGET_DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "Elec": {
        "domain_label": "Electronics",
        "concrete_attributes": (
            "e.g., battery longevity, sound clarity, connector sturdiness, cable length, "
            "screen resolution, portability, ease of setup, heat management"
        ),
        # V1 legacy few-shots
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
        # V2 formal multi few-shots
        "few_shots_poor_v2": [
            (
                '"This versatile solution is the ultimate companion to elevate your daily routine. '
                'Crafted with innovative technology, it seamlessly integrates into your lifestyle for an unparalleled experience."',
                "Empty commercial buzzwords with zero mention of tangible product features, build quality, or real-world utility.",
            ),
            (
                '"Based on your cross-domain profile transferred by the DGCDR model, this product '
                'satisfies the behavioral patterns detected in your previous purchases."',
                "Robotic system jargon. It describes the recommendation algorithm rather than providing meaningful product context to the user.",
            ),
            (
                '"Given your preference for durable gear, this high-quality item provides '
                'dependable performance whether at your desk or on the go."',
                "Rigid formulaic phrasing that repeats a generic template without describing any specific physical quality or usage scenario.",
            ),
        ],
        "few_shots_good_v2": [
            (
                '"The durable braided exterior and reinforced connectors provide reliable protection against daily bending and wear. '
                'Its six-foot length provides ample reach for charging comfortably from distant outlets without placing unnecessary tension on the device."',
                "Demonstrates formal, clear description of build durability (braided exterior, reinforced connectors) and practical utility (6ft reach, no tension on device).",
            ),
            (
                '"The weighted, non-slip base ensures steady support on vehicle dashboards without requiring adhesives that could damage the interior. '
                'The secure mounting mechanism allows for straightforward one-handed placement and smooth viewing angle adjustments while traveling."',
                "Highlights physical stability, surface protection, and convenient one-handed operation in formal English.",
            ),
            (
                '"Offering extended battery performance of up to thirty hours, these earbuds deliver clear sound throughout long workdays and travel. '
                'The intuitive controls and comfortable ergonomic fit provide dependable noise isolation and effortless operation without accidental interruptions."',
                "Focuses on battery longevity, clear audio, ergonomic comfort, and functional reliability without engineering jargon.",
            ),
        ],
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
    prompt_version: str = "v2",
) -> Tuple[str, Optional[str]]:
    """
    Builds a comprehensive, personalized prompt in English for a single user,
    dynamically tailored to the target domain and prompt version (v1 or v2).
    Also saves the prompt text to prompts_dir/prompt_<user_id>.txt if prompts_dir is provided.

    Returns:
        (prompt_text, saved_file_path)
    """
    domain_key = resolve_target_domain_key(target_domain_name)
    domain_profile = TARGET_DOMAIN_PROFILES.get(domain_key, DEFAULT_DOMAIN_PROFILE)

    lines = []
    if (prompt_version or "").strip().lower() == "v1":
        lines.append("You are an expert personal shopping assistant and product specialist.")
        lines.append(
            f"Your goal is to write natural, compelling, and personalized product recommendations for items in {target_domain_name}."
        )
    else:
        lines.append("You are an objective consumer experience analyst and pragmatic product reviewer.")
        lines.append(
            f"Your goal is to explain why items in {target_domain_name} specifically fit this user based on real-world utility, verified physical characteristics, and practical daily performance."
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
    attr_hint = domain_profile.get("concrete_attributes", "").strip()

    if (prompt_version or "").strip().lower() == "v1":
        lines.append(
            "For each recommended item, explain what practical features, ergonomic details, and performance qualities this specific user will appreciate."
        )
        lines.append("")
        lines.append("Follow these strict rules:")
        lines.append("1. FOCUS ON CONCRETE PRODUCT ATTRIBUTES:")
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
    else:
        lines.append(
            "For each recommended item, explain what practical features, ergonomic details, setup ease, and real-world performance qualities this specific user will value."
        )
        lines.append("")
        lines.append("Follow these strict rules:")
        lines.append("1. CONCRETE REAL-WORLD UTILITY:")
        if attr_hint:
            lines.append(
                f"   - Explain what specific real-world benefits, ergonomics, setup ease, and practical performance aspects a real user will highlight in a 5-star review ({attr_hint})."
            )
        else:
            lines.append(
                f"   - Explain what specific real-world benefits, ergonomics, setup ease, and practical performance aspects a real user will highlight in a 5-star review for products in {target_domain_name}."
            )
        lines.append(
            "   - Ground these benefits in the user's prior interaction history (e.g., relate them to traits the user consistently praised in past purchases, such as build quality, comfort, durability, or low-maintenance daily reliability)."
        )
        lines.append("")
        lines.append("2. NEGATIVE CONSTRAINTS (STRICTLY FORBIDDEN):")
        lines.append(
            "   - Avoid generic marketing buzzwords such as 'seamless functionality', 'dependable companion', 'versatile solution', 'elevate your routine', 'game changer', or 'must-have addition', unless tied directly to a specific physical feature."
        )
        lines.append("   - Do NOT mention 'DGCDR', 'recommendation algorithm', 'cross-domain model', or 'graph'.")
        lines.append(
            "   - Do NOT start every sentence with repetitive meta-phrases like 'Based on your history' or 'The system recommended this because'."
        )
        lines.append(
            "   - Vary sentence structures across recommendations. Do NOT repeat formulaic openings such as 'Given your preference for...' or 'Considering your appreciation for...'. State the item's concrete qualities and practical benefits directly in clear, formal English."
        )
        lines.append("   - Speak directly, naturally, and objectively to the user ('You\\'ll appreciate...', 'This provides...').")
        lines.append("")
        lines.append("3. LENGTH & FOCUS:")
        lines.append("   - Strictly 2 to 3 concise, information-dense sentences per item.")
        lines.append("   - Focus strictly on tangible user experience and functional durability rather than promotional sales copy.")
        lines.append("")

    # 5. Few-Shot Demonstrations (if defined for the target domain)
    is_v1 = (prompt_version or "").strip().lower() == "v1"
    few_shots_good_v2 = domain_profile.get("few_shots_good_v2")
    few_shots_poor_v2 = domain_profile.get("few_shots_poor_v2")

    if not is_v1 and few_shots_good_v2:
        lines.append("=" * 60)
        lines.append("EXAMPLES OF EXPECTED FORMAL STYLE (FEW-SHOT DEMONSTRATIONS)")
        lines.append("=" * 60)
        if few_shots_poor_v2:
            lines.append("[POOR STYLE - DO NOT WRITE LIKE THESE]:")
            for idx, (ex_text, reason) in enumerate(few_shots_poor_v2, 1):
                lines.append(f"{idx}. {ex_text}")
                lines.append(f"   -> Reason: {reason}")
            lines.append("")
        lines.append("[EXCELLENT FORMAL STYLE - WRITE LIKE THESE]:")
        for idx, (ex_text, note) in enumerate(few_shots_good_v2, 1):
            lines.append(f"{idx}. {ex_text}")
            if note:
                lines.append(f"   -> Note: {note}")
        lines.append("")
    else:
        # Fallback / V1 baseline few-shot
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

