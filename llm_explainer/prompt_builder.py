"""
Prompt builder module for generating LLM explanation prompts in English.
Formats user history across domains, presents a single recommended item from DGCDR,
enforces JSON response schema, and optionally saves each prompt to a dedicated file.
"""

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from llm_explainer.config import DEFAULT_SETTINGS


DEFAULT_PROMPT_VERSION = "v3"
GENERATION_PROMPT_VERSIONS = ("v2", "v3")
V3_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "prompts", "EXPLANATION_V3.txt")
V3_FEW_SHOT_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "prompts", "EXPLANATION_V3_FEW_SHOT.txt")
GENERATION_MODES = ("zero-shot", "few-shot")
V3_REQUEST_PROTOCOL = "system_instance_json_schema_v1"


def resolve_generation_mode(prompt_version, generation_mode=None, no_few_shot=False):
    """Default generation to few-shot; honor explicit modes and the legacy alias."""
    mode = generation_mode or ("zero-shot" if no_few_shot else DEFAULT_SETTINGS["generation_mode"])
    if mode not in GENERATION_MODES:
        raise ValueError("Unsupported generation mode: %s" % mode)
    return mode


def add_generation_mode_arguments(parser, no_few_shot_default=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--generation_mode", choices=GENERATION_MODES, default=None,
        help=f"Explanation generation examples: zero-shot or few-shot (default: {DEFAULT_SETTINGS['generation_mode']}). Independent of --judge_prompt.",
    )
    group.add_argument(
        "--no_few_shot", action="store_true", default=no_few_shot_default,
        help="Compatibility alias for --generation_mode zero-shot.",
    )


def add_prompt_saving_arguments(parser, default=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save_prompts", dest="save_prompts", action="store_true",
                       help="Enable saving generation prompts and the template snapshot.")
    group.add_argument("--no_save_prompts", dest="save_prompts", action="store_false",
                       help="Do not write files to saved_prompts (default).")
    parser.set_defaults(save_prompts=default)


def normalize_generation_arguments(args):
    args.generation_mode = resolve_generation_mode(args.prompt_version, args.generation_mode, args.no_few_shot)
    args.no_few_shot = args.generation_mode == "zero-shot"
    return args


def generation_run_version(prompt_version, generation_mode):
    """Separate few-shot v3 artifacts while keeping previous run paths valid."""
    if generation_mode == "zero-shot":
        return prompt_version + "_no_few_shot"
    return prompt_version + "_few_shot" if prompt_version == "v3" else prompt_version


def split_generation_prompt(prompt: str) -> Tuple[Optional[str], str]:
    """Separate v3 instructions from data without editing either section."""
    instructions, separator, instance = prompt.partition("\nINPUT\n")
    if prompt.startswith("TASK\n") and separator:
        return instructions, "INPUT\n" + instance
    return None, prompt


def load_generation_template(prompt_version: str, generation_mode: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Resolve an actual template version rather than just labeling outputs."""
    if prompt_version not in GENERATION_PROMPT_VERSIONS:
        raise ValueError("Unsupported generation prompt version: %s" % prompt_version)
    if prompt_version == "v2":
        return None, None
    mode = resolve_generation_mode(prompt_version, generation_mode)
    path = V3_FEW_SHOT_TEMPLATE_FILE if mode == "few-shot" else V3_TEMPLATE_FILE
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read(), path


def generation_prompt_metadata(prompt_version: str, generation_mode: Optional[str] = None) -> Dict[str, Any]:
    mode = resolve_generation_mode(prompt_version, generation_mode)
    template, path = load_generation_template(prompt_version, mode)
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest() if template is not None else None
    return {
        "version": prompt_version,
        "generation_mode": mode,
        "demonstration_set": "v3_evidence_v1" if prompt_version == "v3" and mode == "few-shot" else ("v2_style" if mode == "few-shot" else None),
        "template_file": path,
        "template_sha256": template_hash,
        "history_format": "labeled_text",
        "request_protocol": V3_REQUEST_PROTOCOL if template_hash else "legacy",
        # Keep the template hash faithful to its wording while invalidating
        # explanations generated with the previous JSON history rendering.
        "cache_sha256": hashlib.sha256((template_hash + ":labeled_text_v1:" + V3_REQUEST_PROTOCOL).encode("utf-8")).hexdigest() if template_hash else None,
    }


def _render_v3_history(history: List[Dict]) -> str:
    # Use the legacy labeled-text layout without copying its instructions
    # or examples into v3. Only historical evidence fields are included.
    if not history:
        return "No historical reviews recorded in this domain."
    lines = []
    for idx, item in enumerate(history, 1):
        rating = item.get("rating")
        rating_text = f"{float(rating):.1f} / 5.0" if rating is not None else "Unknown"
        lines.extend([
            f"{idx}. Item ID: {item.get('item_id', '')}",
            f"   Item Title: {item.get('item_title', 'Unknown Title')}",
            f"   User Rating: {rating_text}",
            f"   Review Title: {(item.get('review_title') or '').strip() or 'No review title'}",
            f"   Review Text: {(item.get('review_text') or '').strip() or 'No review text'}",
            "",
        ])
    return "\n".join(lines).rstrip()


def _save_prompt(prompt_text: str, prompts_dir: Optional[str], user_id: str, item_id: str) -> Optional[str]:
    if not prompts_dir:
        return None
    os.makedirs(prompts_dir, exist_ok=True)
    safe_uid = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_"))
    safe_iid = "".join(c for c in str(item_id) if c.isalnum() or c in ("-", "_"))
    path = os.path.join(prompts_dir, f"prompt_{safe_uid}_{safe_iid}.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(prompt_text)
    return path


TARGET_DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "Elec": {
        "domain_label": "Electronics",
        "concrete_attributes": (
            "e.g., battery longevity, sound clarity, connector sturdiness, cable length, "
            "screen resolution, portability, ease of setup, heat management"
        ),
        "few_shots_poor_v2": [
            (
                '"This versatile solution is the ultimate companion to elevate your daily routine. '
                'Crafted with innovative technology, it seamlessly integrates into your lifestyle for an unparalleled experience."',
                "Empty commercial buzzwords with zero mention of tangible product features, build quality, or real-world utility.",
            ),
            (
                '"Based on your cross-domain profile, this product '
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
        "domain_label": "Clothing, Shoes & Jewelry",
        "concrete_attributes": "",  # TODO: attributi concreti per abbigliamento (es. fit, tessuto, comfort)
        "few_shots_poor_v2": [],
        "few_shots_good_v2": [],
    },
    "Sport": {
        "domain_label": "Sports & Outdoors",
        "concrete_attributes": "",  # TODO: attributi concreti per sport (es. grip, traspirabilità, resistenza)
        "few_shots_poor_v2": [],
        "few_shots_good_v2": [],
    },
}

DEFAULT_DOMAIN_PROFILE: Dict[str, Any] = {
    "domain_label": "Target Domain",
    "concrete_attributes": "",
    "few_shots_poor_v2": [],
    "few_shots_good_v2": [],
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


def build_item_prompt(
    user_id: str,
    source_domain_name: str,
    target_domain_name: str,
    source_history: List[Dict],
    target_history: List[Dict],
    target_item: Dict,
    prompts_dir: Optional[str] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    include_few_shots: Optional[bool] = None,
    generation_mode: Optional[str] = None,
    save_prompts: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Builds a single-item prompt using v3 (file template) or legacy v2.
    V3 selects a separate evidence-demonstration template for few-shot generation.
    Saves prompt text only when prompts_dir is provided and save_prompts is True.

    Returns:
        (prompt_text, saved_file_path)
    """
    if not save_prompts:
        prompts_dir = None
    if generation_mode is None and include_few_shots is not None:
        generation_mode = "few-shot" if include_few_shots else "zero-shot"
    mode = resolve_generation_mode(prompt_version, generation_mode)
    include_few_shots = mode == "few-shot"
    template, _ = load_generation_template(prompt_version, mode)
    if template is not None:
        values = {
            "user_id": str(user_id), "source_domain_name": source_domain_name,
            "target_domain_name": target_domain_name,
            "source_history": _render_v3_history(source_history),
            "target_history": _render_v3_history(target_history),
            "item_id": str(target_item["item_id"]),
            "item_title": target_item.get("item_title", "Unknown Title"),
        }
        # Replace named input slots only: preserve literal JSON braces and never
        # interpret braces/placeholders contained in the supplied input data.
        slots = set(re.findall(r"\{([a-z_]+)\}", template))
        unknown = slots - values.keys()
        if unknown:
            raise ValueError("Unknown v3 prompt placeholders: %s" % ", ".join(sorted(unknown)))
        prompt_text = re.sub(r"\{([a-z_]+)\}", lambda match: values[match.group(1)], template)
        return prompt_text, _save_prompt(prompt_text, prompts_dir, user_id, target_item["item_id"])

    domain_key = resolve_target_domain_key(target_domain_name)
    domain_profile = TARGET_DOMAIN_PROFILES.get(domain_key, DEFAULT_DOMAIN_PROFILE)

    lines = []
    lines.append("Act as an objective consumer experience analyst and pragmatic product reviewer.")
    lines.append(
        f"Your goal is to explain why items in {target_domain_name} specifically fit this user interests and values."
    )
    lines.append("")
    lines.append("Scenario:")
    lines.append(f"- Source Domain: {source_domain_name}")
    lines.append(f"- Target Domain: {target_domain_name}")
    lines.append("")
    lines.append(
        "Below is the user's cross-domain profile. Pay special attention to what the user values in their reviews:"
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

    # 3. Recommended Item (The single held-out target item)
    lines.append("=" * 60)
    lines.append("RECOMMENDED ITEM TO EXPLAIN")
    lines.append("=" * 60)
    lines.append(
        f"The following item has been recommended for the user in {target_domain_name}:"
    )
    lines.append("")
    lines.append(f"Item ID: {target_item['item_id']}")
    lines.append(f"Item Title: {target_item.get('item_title', 'Unknown Title')}")
    lines.append("")

    # 4. Instructions and Guidelines
    lines.append("=" * 60)
    lines.append("INSTRUCTIONS & GUIDELINES")
    lines.append("=" * 60)
    lines.append("For the recommended item, follow these strict rules:")
    lines.append("")
    lines.append("1. CONCRETE REAL-WORLD UTILITY:")
    lines.append(
        "   - Explain what specific real-world benefits and practical aspects a real user could value and highlight."
    )
    lines.append(
        "   - Ground these benefits in the user's prior interaction history (e.g., relate them to traits the user consistently praised in past purchases)."
    )
    lines.append(
        "   - Pay attention on verifiable physical characteristics that the user could value given its history."
    )
    lines.append("")
    lines.append("2. NEGATIVE CONSTRAINTS:")
    lines.append(
        "   - Avoid generic marketing buzzwords such as 'seamless functionality', 'dependable companion', 'versatile solution', 'elevate your routine', 'game changer', or 'must-have addition', unless tied directly to a specific physical feature."
    )
    lines.append(
        "   - Do NOT start every sentence with repetitive meta-phrases like 'Based on your history' or 'The system recommended this because'."
    )
    lines.append("")
    lines.append("3. LENGTH & FOCUS:")
    lines.append("   - Strictly less than 50 words, using information-dense sentences.")
    lines.append("   - Focus strictly on tangible user experience.")
    lines.append("   - Speak directly, naturally, and objectively to the user.")
    lines.append("")

    # 5. Few-Shot Demonstrations (if enabled and defined for the target domain)
    few_shots_good_v2 = domain_profile.get("few_shots_good_v2", [])
    few_shots_poor_v2 = domain_profile.get("few_shots_poor_v2", [])

    if include_few_shots and few_shots_good_v2:
        lines.append("=" * 60)
        lines.append("EXAMPLES OF EXPECTED FORMAL STYLE")
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

    # 6. Output Format and JSON Schema
    lines.append("=" * 60)
    lines.append("OUTPUT FORMAT (JSON ONLY)")
    lines.append("=" * 60)
    lines.append(
        "CRITICAL: You must reply ONLY with a valid JSON object. Do not include any introductory or concluding text, explanations, or markdown code blocks outside the JSON. Use the following schema:"
    )
    lines.append("")

    example_obj = {
        "item_id": target_item["item_id"],
        "item_title": target_item.get("item_title", ""),
        "explanation": f"<Write the personalized explanation in English for item {target_item['item_id']} here (strictly under 50 words)>",
    }
    example_json = json.dumps(example_obj, indent=2)
    lines.append(example_json)

    prompt_text = "\n".join(lines)

    return prompt_text, _save_prompt(prompt_text, prompts_dir, user_id, target_item["item_id"])


def build_user_prompt(
    user_id: str,
    source_domain_name: str,
    target_domain_name: str,
    source_history: List[Dict],
    target_history: List[Dict],
    recommended_items: List[Dict],
    prompts_dir: Optional[str] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    include_few_shots: Optional[bool] = None,
    generation_mode: Optional[str] = None,
    save_prompts: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Backward-compatibility wrapper for build_item_prompt.
    If multiple items are passed, builds prompt for the first item.
    """
    if recommended_items:
        return build_item_prompt(
            user_id=user_id,
            source_domain_name=source_domain_name,
            target_domain_name=target_domain_name,
            source_history=source_history,
            target_history=target_history,
            target_item=recommended_items[0],
            prompts_dir=prompts_dir,
            prompt_version=prompt_version,
            include_few_shots=include_few_shots,
            generation_mode=generation_mode,
            save_prompts=save_prompts,
        )
    return "", None
