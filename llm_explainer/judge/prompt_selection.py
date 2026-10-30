"""Shared selection of the two judge prompt variants."""

import os


PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
JUDGE_PROMPTS = {
    "zero-shot": os.path.join(PROMPTS_DIR, "TEST_PROMPT_ALIGNMENT_NO_NULL_ZERO_SHOT.txt"),
    "few-shot": os.path.join(PROMPTS_DIR, "TEST_PROMPT_ALIGNMENT_NO_NULL_FEW_SHOT.txt"),
}


def add_judge_prompt_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--judge_prompt", choices=tuple(JUDGE_PROMPTS), default="zero-shot",
        help="Judge prompt variant (default: zero-shot).",
    )
    group.add_argument(
        "--system_prompt", default=None,
        help="Custom judge system prompt file; logged as custom instead of a built-in variant.",
    )


def resolve_judge_prompt(args):
    """Return the selected system prompt path and its report label."""
    if getattr(args, "system_prompt", None):
        return args.system_prompt, "custom"
    version = getattr(args, "judge_prompt", "zero-shot")
    return JUDGE_PROMPTS[version], version
