"""Strict JSON protocol for the review-alignment judge (standard library only)."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "minLength": 1},
        "alignment_score": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
    },
    "required": ["reason", "alignment_score"],
    "additionalProperties": False,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_judgment(text: str) -> Dict[str, Any]:
    """Never infer a score from prose, specifications, or malformed JSON."""
    data = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) != {"reason", "alignment_score"}:
        raise ValueError("Expected exactly reason and alignment_score")
    if not isinstance(data["reason"], str) or not data["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    score = data["alignment_score"]
    # bool is a subclass of int, but is not a valid grade.
    if score is not None and (type(score) is not int or not 1 <= score <= 5):
        raise ValueError("alignment_score must be an integer 1-5 or null")
    return {
        "reason": data["reason"].strip(),
        "discrete_score": score,
        "status": "insufficient_evidence" if score is None else "scored",
    }


def judgment_counts(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = sum(it.get("status") == "scored" for it in items)
    insufficient = sum(it.get("status") == "insufficient_evidence" for it in items)
    return {
        "attempted_count": len(items),
        "scored_count": scored,
        "insufficient_evidence_count": insufficient,
        "failed_count": len(items) - scored - insufficient,
        "score_coverage_pct": round(100 * scored / len(items), 2) if items else None,
    }


def call_ollama_judge(
    system_prompt: str,
    instance_prompt: str,
    model: str = "llama3.1:8b",
    temperature: float = 0.0,
    seed: int = 42,
    ollama_url: str = "http://localhost:11434",
    timeout: int = 180,
    max_tokens: int = 256,
    num_ctx: int = 8192,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Request two-field JSON, retaining failures and one bounded retry by default."""
    if max_tokens < 1 or num_ctx < 1 or max_retries not in (0, 1):
        raise ValueError("Positive token limits and max_retries of 0 or 1 are required")
    attempts = []
    token_budget = max_tokens
    for _ in range(max_retries + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ],
            "stream": False,
            "think": False,
            "format": RESPONSE_SCHEMA,
            "options": {"temperature": temperature, "seed": seed,
                        "num_predict": token_budget, "num_ctx": num_ctx},
        }
        attempt = {"max_tokens": token_budget, "response_text": "",
                   "done_reason": None, "status": "request_error", "error": None}
        parsed = None
        try:
            request = urllib.request.Request(
                ollama_url.rstrip("/") + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8")
            attempt["raw_response"] = raw_body
            attempt["status"] = "invalid_response"
            response_data = json.loads(raw_body)
            message = response_data.get("message", {})
            content = message.get("content", "")
            if not isinstance(content, str):
                raise ValueError("Judge message content must be a string")
            attempt["response_text"] = content
            attempt["done_reason"] = response_data.get("done_reason")
            if attempt["done_reason"] == "length":
                attempt.update(status="truncated", error="Output token limit reached")
                token_budget = max(512, token_budget * 2)
            elif response_data.get("error"):
                attempt.update(status="request_error", error=str(response_data["error"]))
            else:
                parsed = parse_judgment(content)
                attempt["status"] = parsed["status"]
        except urllib.error.HTTPError as exc:
            attempt["error"] = "HTTP {}: {}".format(exc.code, exc.read().decode("utf-8", errors="replace"))
        except (OSError, TimeoutError) as exc:
            attempt["error"] = str(exc)
        except (ValueError, AttributeError, TypeError) as exc:
            attempt.update(status="invalid_response", error=str(exc))
        attempts.append(attempt)
        if parsed is not None:
            break

    last = attempts[-1]
    return {
        "error": last["error"], "status": last["status"],
        "response_text": last["response_text"], "done_reason": last["done_reason"],
        "reason": parsed["reason"] if parsed else "",
        "discrete_score": parsed["discrete_score"] if parsed else None,
        "weighted_score": None,
        "weighted_score_status": "not_computed",
        "attempts": attempts,
    }
