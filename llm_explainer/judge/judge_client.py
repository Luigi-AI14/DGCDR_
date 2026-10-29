"""Strict JSON protocol for the review-alignment judge (standard library only)."""

import json
import math
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


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


def weighted_score_counts(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = sum(it.get("status") == "scored" for it in items)
    computed = sum(it.get("weighted_score") is not None for it in items)
    return {"eligible_count": eligible, "computed_count": computed,
            "unavailable_count": eligible - computed,
            "coverage_pct": round(100 * computed / eligible, 2) if eligible else None}


def extract_weighted_score(content: str, logprobs: Any) -> Dict[str, Any]:
    """Normalize grade alternatives at the JSON score value, never numbers in prose.

    Call only after parse_judgment has validated a numeric judgment. Probabilities
    are conditional on the actual preceding output; JSON key order is recorded.
    """
    result = {"weighted_score": None, "weighted_score_status": "missing_logprobs",
              "score_probabilities": {}, "score_token_index": None,
              "weighted_score_conditioning": None}
    if not logprobs:
        return result
    try:
        # Decode top-level members to avoid matching escaped keys inside reason.
        decoder = json.JSONDecoder()
        offset = content.index("{") + 1
        reason_seen = False
        while True:
            while content[offset].isspace():
                offset += 1
            key, offset = decoder.raw_decode(content, offset)
            offset = content.index(":", offset) + 1
            while content[offset].isspace():
                offset += 1
            value_start = offset
            value, offset = decoder.raw_decode(content, offset)
            if key == "alignment_score":
                break
            reason_seen = reason_seen or key == "reason"
            offset = content.index(",", offset) + 1

        result["weighted_score_conditioning"] = (
            "input_and_preceding_reason" if reason_seen else "input_and_preceding_output")
        start = len(content[:value_start].encode("utf-8"))
        end = len(content[:offset].encode("utf-8"))

        def token_bytes(entry):
            raw = entry.get("bytes")
            return bytes(raw) if raw is not None else entry["token"].encode("utf-8")

        pieces = [token_bytes(entry) for entry in logprobs]
        if b"".join(pieces) != content.encode("utf-8"):
            result["weighted_score_status"] = "token_alignment_unavailable"
            return result
        cursor = 0
        for index, piece in enumerate(pieces):
            next_cursor = cursor + len(piece)
            if cursor <= start and end <= next_cursor:
                if piece.strip() != str(value).encode("ascii"):
                    break  # A digit merged with punctuation is not a grade token.
                result["score_token_index"] = index
                entry = logprobs[index]
                alternatives = list(entry.get("top_logprobs", []))
                if not any(token_bytes(alt) == piece for alt in alternatives):
                    alternatives.append(entry)
                grades = {}
                seen_tokens = set()
                for alternative in alternatives:
                    raw = token_bytes(alternative)
                    label = raw.strip()
                    if label not in (b"1", b"2", b"3", b"4", b"5") or raw in seen_tokens:
                        continue
                    seen_tokens.add(raw)
                    lp = alternative["logprob"]
                    if type(lp) not in (int, float) or math.isnan(lp) or lp > 0:
                        raise ValueError("Invalid grade log probability")
                    grades.setdefault(label.decode("ascii"), []).append(lp)
                if len(grades) != 5:
                    result["weighted_score_status"] = "incomplete_score_distribution"
                    return result
                largest = max(lp for values in grades.values() for lp in values)
                if not math.isfinite(largest):
                    raise ValueError("No finite grade log probability")
                masses = {grade: sum(math.exp(lp - largest) for lp in values)
                          for grade, values in grades.items()}
                total = sum(masses.values())
                probabilities = {str(s): masses[str(s)] / total for s in range(1, 6)}
                result.update(weighted_score=float(sum(int(s) * p for s, p in probabilities.items())),
                              weighted_score_status="computed", score_probabilities=probabilities)
                return result
            cursor = next_cursor
        result["weighted_score_status"] = "unsupported_score_token"
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
        result["weighted_score_status"] = "invalid_logprobs"
    return result


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
    weighted_scoring: bool = True,
    top_logprobs: int = 20,
    num_gpu: Optional[int] = None,
) -> Dict[str, Any]:
    """Request two-field JSON, retaining failures and one bounded retry by default."""
    if max_tokens < 1 or num_ctx < 1 or max_retries not in (0, 1):
        raise ValueError("Positive token limits and max_retries of 0 or 1 are required")
    if not 1 <= top_logprobs <= 20 or (num_gpu is not None and num_gpu < 0):
        raise ValueError("top_logprobs must be 1-20; num_gpu must be nonnegative")
    attempts = []
    response_data = {}
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
        if weighted_scoring:
            payload.update(logprobs=True, top_logprobs=top_logprobs)
        if num_gpu is not None:
            payload["options"]["num_gpu"] = num_gpu
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
    weighted = {"weighted_score": None, "weighted_score_status": "disabled" if not weighted_scoring
                else ("not_applicable" if parsed else "judgment_failed"),
                "score_probabilities": {}, "score_token_index": None,
                "weighted_score_conditioning": None}
    if weighted_scoring and parsed is not None and parsed["discrete_score"] is not None:
        weighted = extract_weighted_score(last["response_text"], response_data.get("logprobs"))
    return {
        "error": last["error"], "status": last["status"],
        "response_text": last["response_text"], "done_reason": last["done_reason"],
        "reason": parsed["reason"] if parsed else "",
        "discrete_score": parsed["discrete_score"] if parsed else None,
        **weighted,
        "attempts": attempts,
    }
