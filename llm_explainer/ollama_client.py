"""
Ollama client for local LLM inference with Qwen 3.5 9B / Llama 3.1 8B.
Uses standard urllib for zero-dependency HTTP communication.
Enforces JSON format mode, handles single-item and batched explanations, and includes retry logic.
"""

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:9b",
        temperature: float = 0.0,
        seed: int = 42,
        timeout: int = 300,
        num_ctx: int = 32768,
        think: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.think = think

    def check_health(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def _call_ollama_api(self, prompt: str) -> str:
        """Internal helper to send a prompt to Ollama /api/chat and return raw text."""
        url = f"{self.base_url}/api/chat"
        system_instruction = (
            "You are an expert Recommender System AI assistant specializing in personalized explanation generation. "
            "You MUST reply strictly with a raw JSON object conforming to the requested schema. "
            "Never output chain-of-thought, reasoning notes, drafts, or markdown fences before or after the JSON."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": 4096,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                if resp_data.get("done_reason") == "length":
                    logger.warning(
                        "Ollama response generation was truncated due to context window limit (done_reason='length')!"
                    )
                message = resp_data.get("message", {})
                raw_response = message.get("content", "")
                if not raw_response and "response" in resp_data:
                    raw_response = resp_data.get("response", "")
                return raw_response
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to communicate with Ollama at {url}: {e}")

    def generate_explanation(
        self,
        prompt: str,
        mock: bool = False,
        expected_item: Optional[Dict[str, str]] = None,
        max_retries: int = 1,
    ) -> Dict[str, str]:
        """
        Send single-item prompt to Ollama and return parsed explanation dict:
            {"item_id": str, "item_title": str, "explanation": str}
        Includes retry mechanism (up to max_retries retries).
        """
        item_id = str(expected_item.get("item_id", "")).strip() if expected_item else ""
        item_title = str(expected_item.get("item_title", item_id)).strip() if expected_item else item_id

        if mock:
            return {
                "item_id": item_id,
                "item_title": item_title,
                "explanation": (
                    f"The DGCDR model recommended '{item_title}' based on user preference "
                    "for reliable durability and high performance demonstrated across past interactions."
                ),
            }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                raw_response = self._call_ollama_api(prompt)
                explanations = self._parse_json_response(raw_response, expected_item=expected_item)
                if explanations:
                    return explanations[0]
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} for item '{item_id}' failed ({e}). Retrying..."
                    )
                else:
                    logger.error(f"All {max_retries + 1} attempts failed for item '{item_id}': {e}")

        # Fallback if all attempts fail
        return {
            "item_id": item_id,
            "item_title": item_title,
            "explanation": "",
        }

    def generate_explanations(
        self,
        prompt: str,
        mock: bool = False,
        expected_items: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Send prompt to Ollama and return parsed explanations list.
        Each item in the returned list is:
            {"item_id": str, "item_title": str, "explanation": str}
        """
        if mock:
            # Generate deterministic mock explanations for dry-run testing
            results = []
            if expected_items:
                for it in expected_items:
                    results.append(
                        {
                            "item_id": it["item_id"],
                            "item_title": it.get("item_title", ""),
                            "explanation": f"The DGCDR model recommended '{it.get('item_title', it['item_id'])}' based on the user's strong affinity for durable, comfortable and highly rated products demonstrated across their multi-domain history.",
                        }
                    )
            return results

        raw_response = self._call_ollama_api(prompt)
        explanations = self._parse_json_response(raw_response)
        return explanations

    def _parse_json_response(
        self, raw_text: str, expected_item: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Extract and validate explanations from model JSON output (single dict or list)."""
        parsed = None
        cleaned = raw_text.strip()

        # 1. Try finding complete code block
        if "```" in cleaned:
            code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if code_match:
                try:
                    parsed = json.loads(code_match.group(1))
                except Exception:
                    pass

        # 2. Try parsing between outer brackets
        if parsed is None:
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate_str = cleaned[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(candidate_str)
                except Exception:
                    pass

        # 3. Try direct json.loads
        if parsed is None:
            try:
                parsed = json.loads(cleaned)
            except Exception:
                pass

        # If parsed into dict or list
        if isinstance(parsed, dict) and "explanations" in parsed:
            items = parsed["explanations"]
        elif isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict) and ("item_id" in parsed or "explanation" in parsed):
            items = [parsed]
        else:
            items = []

        exp_iid = str(expected_item.get("item_id", "")).strip() if expected_item else ""
        exp_title = str(expected_item.get("item_title", "")).strip() if expected_item else ""

        validated = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get("item_id", "")).strip() or exp_iid
            item_title = str(entry.get("item_title", "")).strip() or exp_title
            explanation = str(entry.get("explanation", "")).strip()

            if item_id and explanation:
                validated.append(
                    {
                        "item_id": item_id,
                        "item_title": item_title,
                        "explanation": explanation,
                    }
                )

        if validated:
            return validated

        # 4. Fallback: regex search for single explanation
        expl_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', cleaned)
        if expl_match and exp_iid:
            return [
                {
                    "item_id": exp_iid,
                    "item_title": exp_title,
                    "explanation": expl_match.group(1).strip(),
                }
            ]

        # 5. Fallback: regex search for individual JSON objects
        item_matches = re.finditer(
            r'\{\s*"item_id"\s*:\s*"([^"]+)"\s*,\s*"item_title"\s*:\s*"([^"]*)"\s*,\s*"explanation"\s*:\s*"([^"]+)"',
            cleaned,
        )
        for m in item_matches:
            validated.append(
                {
                    "item_id": m.group(1).strip(),
                    "item_title": m.group(2).strip(),
                    "explanation": m.group(3).strip(),
                }
            )

        if validated:
            return validated

        logger.error(f"Failed to extract explanations from LLM response:\n{raw_text}")
        raise ValueError(f"Model output did not contain valid explanations: {raw_text}")

    def get_loaded_models(self) -> List[str]:
        """Returns list of currently loaded model names in Ollama VRAM."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/ps", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception as e:
            logger.debug(f"Could not check loaded models via /api/ps: {e}")
            return []

    def unload_model(self, model: Optional[str] = None) -> bool:
        """
        Explicitly unloads a model from GPU VRAM by sending keep_alive=0 to Ollama.
        """
        target_model = model or self.model
        url = f"{self.base_url}/api/generate"
        payload = {"model": target_model, "keep_alive": 0}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    logger.info(f"Successfully unloaded LLM model '{target_model}' from GPU VRAM.")
                    return True
        except Exception as e:
            logger.warning(f"Could not unload model '{target_model}' from Ollama: {e}")
        return False

    def unload_all_models(self) -> None:
        """
        Unloads self.model and any other active models in Ollama VRAM to ensure
        maximum GPU memory availability for downstream embedding models.
        """
        loaded = self.get_loaded_models()
        targets = set(loaded)
        if self.model:
            targets.add(self.model)

        for m in targets:
            self.unload_model(m)
