"""Constrained verbalisation of a numeric attribution.

The LLM is deliberately *not* asked to figure out why an item was recommended.
That question is already answered exactly by the attribution matrix.  The LLM
only renders an answer it is handed, under an explicit numeric constraint, so
the generated text stays anchored to a quantity the model actually computed.

This is what separates the approach from prompt-a-model-with-the-history
explanation baselines, whose text is a plausible rationalisation with no
guaranteed relation to the recommender's internals.
"""

import json
import os
import re
import urllib.error
import urllib.request

from extensions.explainability.channels import BASE, SHARED, SPECIFIC

SYSTEM_PROMPT = """You explain recommendations produced by a cross-domain recommender system.

The system has already decomposed each recommendation into two preference channels:
- TRANSFERRED: taste imported from the user's activity in the SOURCE domain.
- NATIVE: taste specific to the user's activity in the TARGET domain.

You are given the exact numeric split (transfer ratio). Your job is ONLY to put it
into natural language.

Hard rules:
1. Never contradict the transfer ratio. Above 0.6 the recommendation is driven mainly
   by transferred taste; below 0.4 mainly by native taste; in between it is mixed.
2. Only mention items that appear in the user's listed history.
3. Do not invent attributes, ratings, prices or reasons that are not given to you.
4. If the evidence is thin, say so instead of embellishing.
5. Two or three sentences, addressed to the user, no bullet points, no preamble."""


def _format_items(item_ids, catalogue, limit=10):
    if not item_ids:
        return "(none)"
    shown = [catalogue.describe(i) for i in item_ids[:limit]]
    suffix = f" (+{len(item_ids) - limit} more)" if len(item_ids) > limit else ""
    return "; ".join(shown) + suffix


def source_history(model, user_id, limit=None):
    """Target-domain user ids share the id space with the source domain."""
    inter = model.source_interaction_matrix
    items = [int(i) for i in inter.col[inter.row == user_id]]
    return items[:limit] if limit else items


def describe_ratio(tau):
    if tau >= 0.6:
        return "mainly transferred from the source domain"
    if tau <= 0.4:
        return "mainly native to the target domain"
    return "a roughly even mix of transferred and native taste"


def build_prompt(attribution, catalogue, target_history, src_history,
                 source_domain_name='source domain',
                 target_domain_name='target domain'):
    """Render the numeric attribution into an LLM prompt.

    Every number in the prompt comes from the exact decomposition, so the text
    the LLM produces can be checked against it afterwards.
    """
    totals = attribution.user_channel_totals
    lines = [
        f"SOURCE DOMAIN: {source_domain_name}",
        f"TARGET DOMAIN: {target_domain_name}",
        "",
        f"User's history in the source domain: {_format_items(src_history, catalogue)}",
        f"User's history in the target domain: {_format_items(target_history, catalogue)}",
        "",
        f"RECOMMENDED ITEM (rank {attribution.rank}): {catalogue.describe(attribution.item_id)}",
        "",
        "Exact score decomposition computed by the recommender:",
        f"  total score                     : {attribution.score:+.4f}",
        f"  transferred (shared) channel    : {totals.get(SHARED, 0.0):+.4f}",
        f"  native (specific) channel       : {totals.get(SPECIFIC, 0.0):+.4f}",
        f"  raw collaborative channel       : {totals.get(BASE, 0.0):+.4f}",
        f"  TRANSFER RATIO                  : {attribution.tau:.3f}"
        f"  -> {describe_ratio(attribution.tau)}",
        "",
        "Write the explanation.",
    ]
    return "\n".join(lines)


THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


class LLMClient:
    """Minimal client for any OpenAI-compatible chat endpoint.

    Works with a local Ollama / vLLM / LM Studio server as well as hosted
    providers, so the framework does not depend on a specific vendor.

    Reasoning models need care here. Qwen3.5 thinks before answering, and its
    reasoning goes to a separate field: with a small token budget the whole
    budget is spent thinking and the answer comes back empty. When the server is
    Ollama the thinking is switched off through its own endpoint, since a
    six-line explanation does not need it.
    """

    def __init__(self, model, base_url=None, api_key=None, temperature=0.2,
                 max_tokens=400, timeout=300, think=False):
        self.model = model
        self.base_url = (base_url or os.environ.get(
            'EXPLAIN_LLM_BASE_URL', 'http://localhost:11434/v1')).rstrip('/')
        self.api_key = api_key or os.environ.get('EXPLAIN_LLM_API_KEY', 'not-needed')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.think = think

    def _is_ollama(self):
        return self.base_url.endswith('/v1') and '11434' in self.base_url

    def _post(self, url, payload):
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {self.api_key}'},
            method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(
                f"LLM request to {url} failed: {exc}. Start the server or run "
                f"without --llm_model to keep the numeric attribution only."
            ) from exc

    def generate(self, prompt, system_prompt=SYSTEM_PROMPT):
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}]

        if self._is_ollama():
            body = self._post(self.base_url[:-3] + '/api/chat', {
                'model': self.model, 'messages': messages, 'stream': False,
                'think': self.think,
                'options': {'temperature': self.temperature,
                            'num_predict': self.max_tokens}})
            text = (body.get('message') or {}).get('content', '')
            reason = None
        else:
            body = self._post(f"{self.base_url}/chat/completions", {
                'model': self.model, 'messages': messages,
                'temperature': self.temperature, 'max_tokens': self.max_tokens})
            choices = body.get('choices') or []
            if not choices:
                raise RuntimeError(f"LLM returned no choices: {body}")
            message = choices[0].get('message') or {}
            text = message.get('content') or ''
            reason = choices[0].get('finish_reason')

        text = THINK_BLOCK.sub('', text).strip()
        if not text:
            # Silently returning '' is how a whole file of empty explanations
            # gets written without anyone noticing.
            raise RuntimeError(
                f"'{self.model}' returned an empty explanation"
                + (f" (finish_reason={reason})" if reason else "")
                + ". A reasoning model can spend the entire token budget "
                  "thinking; raise max_tokens or disable the thinking mode.")
        return text
