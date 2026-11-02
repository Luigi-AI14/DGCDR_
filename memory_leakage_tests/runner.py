"""Ollama transport, immutable run configuration and resumable JSONL journal."""

import json
import logging
import math
import os
import platform
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import digest, file_hash, now, read_json, validate_dataset, write_json
from .data import validate_manifest
from .probes import FIELDS, PROMPT_VERSION, tasks_for

LOG = logging.getLogger(__name__)
DEFAULT_MODELS = ["gemma4:12b", "qwen3.5:9b", "llama3.1:8b"]
DEFAULT_MODES = ["zero_shot", "few_shot"]


def http_json(base_url, endpoint, payload=None, timeout=300):
    data = json.dumps(payload, allow_nan=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(base_url.rstrip("/") + endpoint, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError("Ollama HTTP {}: {}".format(error.code, detail)) from error


def parse_answer(content, probe):
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != set(FIELDS[probe]):
        raise ValueError("Response must contain exactly the requested JSON fields")
    for key, field in value.items():
        if field is None:
            continue
        if key == "rating":
            if isinstance(field, bool) or not isinstance(field, (int, float)) or not math.isfinite(field) or not 1 <= field <= 5:
                raise ValueError("rating must be null or a finite number from 1 to 5")
        elif not isinstance(field, str):
            raise ValueError(key + " must be a string or null")
    return value


def infer(base_url, model, task, options, timeout, retries, thinking):
    payload = dict(task["request"], model=model, stream=False, options=options)
    if thinking != "default":
        payload["think"] = thinking == "on"
    response, raw, answer, error = None, "", None, None
    status = "transport_error"
    attempts = 0
    start = time.monotonic()
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            response = http_json(base_url, "/api/chat", payload, timeout)
            if not isinstance(response, dict) or response.get("error"):
                raise RuntimeError("Invalid Ollama envelope: " + str(response)[:500])
            raw = response.get("message", {}).get("content", "")
            if not response.get("done") or response.get("done_reason") not in ("stop",):
                # A length-limited response is never evidence of non-memorization,
                # even when its initial JSON happens to parse.
                status = "truncated" if response.get("done_reason") == "length" else "incomplete"
                error = "Incomplete generation: " + str(response.get("done_reason"))
                break
            try:
                answer = parse_answer(raw, task["probe"])
                status = "abstained" if all(v is None or (isinstance(v, str) and not v.strip())
                                            for v in answer.values()) else "ok"
                error = None
            except (ValueError, TypeError) as exc:
                status, error = "format_error", str(exc)
            break  # Never repair a response by revealing the correct answer.
        except (OSError, RuntimeError, ValueError) as exc:
            error = str(exc)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 5))
    return {"status": status, "answer": answer, "raw_response": raw,
            "response_envelope": response, "error": error, "attempts": attempts,
            "seconds": round(time.monotonic() - start, 3), "created_at": now()}


def load_prepared(output_dir):
    output = Path(output_dir)
    dataset = read_json(output / "dataset.json")
    manifest = read_json(output / "manifest.json")
    validate_dataset(dataset)
    validate_manifest(manifest, dataset)
    return dataset, manifest


def read_journal(path, repair_tail=False):
    """Ignore only a crash-partial final line; reject corruption in the middle."""
    path = Path(path)
    if not path.exists():
        return {}
    records = {}
    mode = "r+b" if repair_tail else "rb"
    with path.open(mode) as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                LOG.warning("Ignoring incomplete journal tail at byte %s", start)
                if repair_tail:
                    stream.truncate(start)
                break
            try:
                record = json.loads(line)
                key = (record["model"], record["task_id"])
                if "request" in record and digest(record["request"]) != record["request_sha256"]:
                    raise ValueError("Logged request fingerprint mismatch")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("Corrupt response journal at byte " + str(start)) from exc
            # Raw requests/envelopes remain in the disk journal. Avoid keeping
            # three repeated few-shot reviews per request in RAM during resume.
            record.pop("request", None)
            record.pop("response_envelope", None)
            records[key] = record  # A retry explicitly supersedes an older failure.
    return records


def preview(output_dir, modes):
    _, manifest = load_prepared(output_dir)
    counts, examples = {}, []
    for task in tasks_for(manifest, modes):
        key = "{} / {} / {}".format(task["mode"], task["probe"], task["condition"])
        counts[key] = counts.get(key, 0) + 1
        if not any(e["mode"] == task["mode"] and e["probe"] == task["probe"]
                   and e["condition"] == task["condition"] for e in examples):
            # Ground truth stays out of request preview files.
            examples.append({k: v for k, v in task.items() if k != "expected"})
    result = {"manifest_sha256": digest(manifest), "requests_per_model": counts,
              "total_per_model": sum(counts.values()), "examples": examples}
    write_json(Path(output_dir) / "request_preview.json", result)
    LOG.info("Dry run: %s logical requests per model; no model contacted", result["total_per_model"])
    return result


def run(output_dir, models=DEFAULT_MODELS, modes=DEFAULT_MODES,
        base_url="http://localhost:11434", inference_seed=42, num_ctx=8192,
        max_tokens=4096, timeout=300, retries=1, thinking="off", run_name="default",
        retry_errors=False, permutations=499, bootstrap_samples=499):
    _, manifest = load_prepared(output_dir)
    run_dir = Path(output_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".running"
    try:
        lock_handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError("Run is already locked. If its process crashed, remove " + str(lock)) from exc
    try:
        lock_handle.write(str(os.getpid()))
        lock_handle.close()
        tags = http_json(base_url, "/api/tags", timeout=timeout)
        available = {m.get("name", m.get("model")): m for m in tags["models"]}
        missing = [model for model in models if model not in available]
        if missing:
            raise ValueError("Requested Ollama models are not installed: " + ", ".join(missing))
        infos = {}
        for model in models:
            details = http_json(base_url, "/api/show", {"model": model}, timeout)
            infos[model] = {"digest": available[model]["digest"],
                            "details": details.get("details"), "model_info": details.get("model_info"),
                            "capabilities": details.get("capabilities"),
                            "parameters": details.get("parameters"), "template": details.get("template"),
                            "system": details.get("system")}
        version = http_json(base_url, "/api/version", timeout=timeout)
        options = {"temperature": 0, "top_p": 1, "top_k": 0, "repeat_penalty": 1.0,
                   "seed": inference_seed, "num_ctx": num_ctx, "num_predict": max_tokens}
        config = {"manifest_sha256": digest(manifest), "prompt_version": PROMPT_VERSION,
                  "models": list(models), "modes": list(modes), "model_info": infos,
                  "base_url": base_url.rstrip("/"), "ollama_version": version,
                  "options": options, "thinking": thinking,
                  "permutations": permutations, "bootstrap_samples": bootstrap_samples,
                  "evidence_rule": {"alpha": 0.05, "min_distinct_records": 2,
                                    "min_unique_review_words": 12},
                  "code_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            existing = read_json(config_path)
            if existing["config"] != config:
                raise ValueError("Run settings, code, model digest or backend changed. Use a new --run-name.")
        else:
            write_json(config_path, {"created_at": now(), "platform": platform.platform(),
                                     "python": platform.python_version(), "config": config})
        config_sha = digest(config)
        journal_path = run_dir / "responses.jsonl"
        records = read_journal(journal_path, repair_tail=True)
        for record in records.values():
            if record["config_sha256"] != config_sha:
                raise ValueError("Response journal belongs to different settings")
        total = sum(1 for _ in tasks_for(manifest, modes)) * len(models)
        done = 0
        LOG.info("Audit: %s logical requests; %s models; split seed=%s", total, len(models), manifest["split_seed"])
        with journal_path.open("a", encoding="utf-8", newline="\n") as stream:
            for model in models:
                shared = {}
                for task in tasks_for(manifest, modes):
                    key = (model, task["task_id"])
                    request_sha = digest(task["request"])
                    old = records.get(key)
                    if old and (old["status"] in ("ok", "abstained") or not retry_errors):
                        if old["status"] in ("ok", "abstained"):
                            shared[request_sha] = old
                        done += 1
                        continue
                    if request_sha in shared:
                        previous = shared[request_sha]
                        result = {k: previous[k] for k in ("status", "answer", "raw_response", "error")}
                        result.update(response_envelope=None, seconds=0, attempts=0, created_at=now(),
                                      reused_from_task=previous["task_id"])
                    else:
                        result = infer(base_url, model, task, options, timeout, retries, thinking)
                    record = {k: v for k, v in task.items() if k not in ("expected", "request")}
                    record.update(result, model=model, config_sha256=config_sha,
                                  request_sha256=request_sha, request=task["request"])
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    cached_record = {k: v for k, v in record.items() if k not in ("request", "response_envelope")}
                    records[key] = cached_record
                    if result["status"] in ("ok", "abstained"):
                        shared[request_sha] = cached_record
                    done += 1
                    if result["status"] not in ("ok", "abstained"):
                        LOG.warning("%s seed=%s %s %s: %s: %s", model, task["cohort_seed"],
                                    task["mode"], task["probe"], result["status"], result["error"])
                    if done % 10 == 0 or done == total:
                        LOG.info("Progress %s/%s | %s | seed=%s | %s | %s", done, total, model,
                                 task["cohort_seed"], task["mode"], task["probe"])
        from .scoring import report
        report(output_dir, run_name)
        return run_dir
    finally:
        if not lock_handle.closed:
            lock_handle.close()
        lock.unlink()
