"""Offline scientific and transport regressions; no models/GPU/RecBole needed."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from memory_leakage_tests.common import digest, normalize, read_json, validate_dataset, write_json
from memory_leakage_tests.data import make_manifest, prepare, validate_manifest
from memory_leakage_tests.probes import answer_for, build_request, tasks_for
from memory_leakage_tests.runner import infer, parse_answer, preview, read_journal, run
from memory_leakage_tests.scoring import cluster_interval, permutation_control, report, score, summarize


def fixture():
    users = []
    for index in range(24):
        items = []
        for number in range(2):
            items.append({"item_id": "B{:08d}".format(index * 2 + number),
                "item_title": "Distinct product {} number {}".format(index, number),
                "ground_truth_title": "Specific review {} {}".format(index, number),
                "ground_truth_review": ("These are sufficiently long and distinctive review words for checking exact "
                                        "identity specific recovery for user {} item {}.").format(index, number),
                "ground_truth_rating": 4.0 if number else 5.0})
        users.append({"user_id": "USER_{:03d}".format(index), "held_out_items": items})
    return {"schema_version": 1, "domain_pair": "Cloth-Elec", "split_seed": 42,
            "filters": {"min_rating": 4.0, "min_held_out_words": 5}, "users": users}


def manifest(dataset=None, **kwargs):
    return make_manifest(dataset or fixture(), [11, 22, 33, 44, 55], 2, **kwargs)


def envelope(answer, reason="stop"):
    return {"message": {"role": "assistant", "content": json.dumps(answer)},
            "done": True, "done_reason": reason}


def recorded(answer, status="ok"):
    return {"status": status, "raw_response": json.dumps(answer)}


class CohortTests(unittest.TestCase):
    def test_reproducible_disjoint_original_ids_and_fixed_split(self):
        dataset = fixture()
        before = copy.deepcopy(dataset)
        first = manifest(dataset)
        self.assertEqual(first, manifest(dataset))
        self.assertEqual(dataset, before)
        uids = [u["user_id"] for c in first["cohorts"] for u in c["users"]]
        self.assertEqual(len(uids), 10)
        self.assertEqual(len(set(uids)), 10)
        self.assertEqual(first["split_seed"], 42)
        self.assertFalse(set(uids) & {d["user_id"] for d in first["demonstrations"]})
        validate_manifest(first, dataset)

    def test_insufficient_users_and_duplicate_seeds_fail(self):
        with self.assertRaisesRegex(ValueError, "Need"):
            make_manifest(fixture(), [1, 2, 3, 4, 5], 100)
        with self.assertRaisesRegex(ValueError, "five different"):
            make_manifest(fixture(), [1, 1, 3, 4, 5], 1)

    def test_tampered_snapshot_and_cohort_overlap_rejected(self):
        dataset = fixture()
        frozen = manifest(dataset)
        changed = copy.deepcopy(dataset)
        changed["split_seed"] = 43
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_manifest(frozen, changed)
        frozen["cohorts"][1]["users"][0] = frozen["cohorts"][0]["users"][0]
        with self.assertRaisesRegex(ValueError, "User overlap"):
            validate_manifest(frozen, dataset)

    def test_demonstration_item_and_answer_text_excluded_globally(self):
        dataset = fixture()
        frozen = manifest(dataset)
        demo = frozen["demonstrations"][0]
        others = [u for u in dataset["users"] if u["user_id"] != demo["user_id"]]
        others[0]["held_out_items"][0]["ground_truth_review"] = demo["ground_truth_review"]
        new = manifest(dataset)
        demo_texts = {normalize(d["ground_truth_review"]) for d in new["demonstrations"]}
        for c in new["cohorts"]:
            for u in c["users"]:
                for item in u["held_out_items"]:
                    self.assertNotIn(normalize(item["ground_truth_review"]), demo_texts)

    def test_duplicate_pairs_and_conflicting_titles_rejected(self):
        dataset = fixture()
        dataset["users"][0]["held_out_items"].append(dataset["users"][0]["held_out_items"][0])
        with self.assertRaisesRegex(ValueError, "duplicate held-out"):
            validate_dataset(dataset)
        dataset = fixture()
        dataset["users"][1]["held_out_items"][0]["item_id"] = dataset["users"][0]["held_out_items"][0]["item_id"]
        with self.assertRaisesRegex(ValueError, "Conflicting titles"):
            validate_dataset(dataset)

    def test_preparation_does_not_overwrite_and_needs_no_recbole_for_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            write_json(path / "input.json", fixture())
            prepare(path / "audit", [1, 2, 3, 4, 5], 2, dataset_path=path / "input.json")
            self.assertTrue((path / "audit" / "cohorts" / "seed_1.json").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(path / "audit", [1, 2, 3, 4, 5], 2, dataset_path=path / "input.json")


class PromptTests(unittest.TestCase):
    def test_requests_never_contain_target_answers_including_schema(self):
        frozen = manifest()
        for task in tasks_for(frozen, ["zero_shot", "few_shot"]):
            if task["condition"] != "real":
                continue
            serialized = json.dumps(task["request"])
            for field, value in task["expected"].items():
                if field != "rating":
                    self.assertNotIn(value, serialized)
            self.assertNotIn("enum", json.dumps(task["request"]["format"]))
            key = json.loads(task["request"]["messages"][-1]["content"])
            self.assertEqual(set(key), {"item_id"} if task["probe"] == "item" else {"item_id", "user_id"})

    def test_synthetic_ids_shared_across_modes_and_absent_from_snapshot(self):
        frozen = manifest()
        grouped = {}
        for mode in ("zero_shot", "few_shot"):
            grouped[mode] = [(t["cohort_seed"], t["probe"], t["item_id"], t["user_id"])
                             for t in tasks_for(frozen, [mode]) if t["condition"] == "synthetic_id"]
        self.assertEqual(grouped["zero_shot"], grouped["few_shot"])

    def test_target_cannot_appear_as_demonstration(self):
        demo = manifest()["demonstrations"][0]
        with self.assertRaisesRegex(ValueError, "Target ID"):
            build_request("Cloth-Elec", "review", "few_shot", demo["item_id"], demo["user_id"], [demo])


class ScoringTests(unittest.TestCase):
    def task(self, probe="review"):
        return next(t for t in tasks_for(manifest(), ["zero_shot"]) if t["probe"] == probe and t["condition"] == "real")

    def test_exact_normalization_preserves_punctuation_and_fuzzy_is_not_exact(self):
        task = self.task("item")
        answer = {"item_title": "  " + task["expected"]["item_title"].upper() + " \n"}
        result = score(task, recorded(answer))
        self.assertTrue(result["primary_exact"])
        self.assertFalse(result["raw_exact"])
        answer["item_title"] += "!"
        result = score(task, recorded(answer))
        self.assertFalse(result["primary_exact"])
        self.assertGreater(result["fuzzy_similarity"], 0.8)

    def test_failures_missing_and_abstentions_never_match(self):
        task = self.task()
        for record in (None, {"status": "transport_error"}, recorded({k: None for k in task["expected"]}, "abstained")):
            self.assertFalse(score(task, record)["primary_exact"])

    def test_short_or_shared_review_is_not_strong_evidence(self):
        task = self.task()
        task["expected"]["review_text"] = "Very good product indeed yes"
        rows = [score(task, recorded(task["expected"]))]
        cfg = {"permutations": 19, "bootstrap_samples": 19, "models": ["m"], "modes": ["zero_shot"],
               "evidence_rule": {"alpha": 0.05, "min_distinct_records": 2, "min_unique_review_words": 12}}
        result = summarize(rows, [], "m", "zero_shot", "ALL", "review", cfg, {}, 0, 1)
        self.assertEqual(result["strong_distinct_matches"], 0)
        self.assertEqual(result["evidence_status"], "RECOVERY_DETECTED_REVIEW_REQUIRED")

    def test_item_pool_deduplicates_cross_seed_repeats(self):
        task = self.task("item")
        rows = [score(task, recorded(task["expected"]))] * 4
        cfg = {"permutations": 19, "bootstrap_samples": 19, "models": ["m"], "modes": ["zero_shot"],
               "evidence_rule": {"alpha": 0.05, "min_distinct_records": 2, "min_unique_review_words": 12}}
        result = summarize(rows, [], "m", "zero_shot", "ALL", "item", cfg, {}, 0, 1)
        self.assertEqual(result["scheduled"], 1)
        self.assertEqual(result["matched"], 1)

    def test_permutation_distinguishes_exact_from_constant_rating_like_guess(self):
        tasks = [t for t in tasks_for(manifest(), ["zero_shot"]) if t["probe"] == "item" and t["condition"] == "real"]
        rows = [score(task, recorded(task["expected"])) for task in tasks]
        result = permutation_control(rows, 499, 42)
        self.assertLess(result["permutation_p"], 0.01)
        self.assertLess(result["shuffled_exact_rate"], 0.2)
        self.assertEqual(cluster_interval(rows, 99, 42), [1.0, 1.0])

    def test_parser_rejects_extra_keys_booleans_and_nan_ratings(self):
        for raw in ('{"review_title":null,"review_text":null,"rating":true}',
                    '{"review_title":null,"review_text":null,"rating":NaN}',
                    '{"review_title":null,"review_text":null,"rating":6}',
                    '{"item_title":"x","extra":"y"}'):
            with self.assertRaises(ValueError):
                parse_answer(raw, "item" if "item_title" in raw else "review")


class TransportTests(unittest.TestCase):
    def task(self):
        return next(tasks_for(manifest(), ["zero_shot"]))

    def test_truncation_and_malformed_json_not_scored_as_unknown(self):
        for reply, status in ((envelope({"item_title": "x"}, "length"), "truncated"),
                              ({"message": {"content": "not json"}, "done": True, "done_reason": "stop"}, "format_error")):
            with patch("memory_leakage_tests.runner.http_json", return_value=reply):
                result = infer("http://unused", "m", self.task(), {}, 1, 0, "off")
            self.assertEqual(result["status"], status)

    def test_transport_retry_does_not_inject_ground_truth(self):
        task = self.task()
        with patch("memory_leakage_tests.runner.http_json", side_effect=[OSError("temporary"), envelope({"item_title": None})]) as client:
            with patch("memory_leakage_tests.runner.time.sleep"):
                result = infer("http://unused", "m", task, {}, 1, 1, "off")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(client.call_args_list[0].args[2], client.call_args_list[1].args[2])

    def test_http_serializes_ollama_payload(self):
        from memory_leakage_tests.runner import http_json
        with patch("memory_leakage_tests.runner.urllib.request.urlopen", return_value=io.BytesIO(b'{"ok":true}')) as client:
            self.assertEqual(http_json("http://localhost:11434/", "/api/chat", {"model": "m"}), {"ok": True})
        request = client.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(json.loads(request.data), {"model": "m"})

    def test_crash_tail_repaired_but_middle_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "responses.jsonl"
            record = {"model": "m", "task_id": "one"}
            path.write_bytes(json.dumps(record).encode() + b'\n{"partial"')
            self.assertEqual(len(read_journal(path, repair_tail=True)), 1)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            path.write_bytes(b'not json\n' + json.dumps(record).encode() + b'\n')
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                read_journal(path)


class EndToEndTests(unittest.TestCase):
    def test_failed_requests_retry_without_repeating_successful_queries(self):
        calls = []
        broken = [True]
        def server(base, endpoint, payload=None, timeout=300):
            if endpoint == "/api/tags":
                return {"models": [{"name": "test:model", "digest": "fixed-digest"}]}
            if endpoint == "/api/show":
                return {}
            if endpoint == "/api/version":
                return {"version": "test"}
            calls.append(payload)
            if broken[0]:
                broken[0] = False
                raise RuntimeError("temporary service failure")
            return envelope({field: None for field in payload["format"]["required"]})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            write_json(path / "input.json", fixture())
            output = path / "audit"
            prepare(output, [11, 22, 33, 44, 55], 2, dataset_path=path / "input.json")
            options = dict(models=["test:model"], modes=["zero_shot"], retries=0,
                           permutations=19, bootstrap_samples=19)
            with patch("memory_leakage_tests.runner.http_json", side_effect=server), redirect_stdout(io.StringIO()):
                run(output, **options)
                initial = len(calls)
                rows = report(output)
                pooled = [r for r in rows if r["cohort_seed"] == "ALL" and r["probe"] == "item"][0]
                self.assertEqual(pooled["evidence_status"], "INCONCLUSIVE")
                self.assertEqual(pooled["failures_or_missing"], 1)
                run(output, retry_errors=True, **options)
                self.assertEqual(len(calls), initial + 1)
                # Completed abstentions are frozen, including after a retry run.
                run(output, **options)
                self.assertEqual(len(calls), initial + 1)
                rows = report(output)
                self.assertTrue(all(r["evidence_status"] == "NO_EVIDENCE_DETECTED"
                                    for r in rows if r["cohort_seed"] == "ALL"))
                # Removing a terminal record simulates interruption before that task.
                journal = output / "runs/default/responses.jsonl"
                lines = journal.read_text(encoding="utf-8").splitlines()
                journal.write_text("\n".join(lines[1:-1]) + "\n", encoding="utf-8")
                rows = report(output)
                self.assertEqual(sum(r["failures_or_missing"] for r in rows if r["cohort_seed"] == "ALL"), 1)

    def test_prepare_preview_run_resume_report_and_settings_guard(self):
        dataset = fixture()
        items = {i["item_id"]: i for u in dataset["users"] for i in u["held_out_items"]}
        pairs = {(u["user_id"], i["item_id"]): i for u in dataset["users"] for i in u["held_out_items"]}
        calls = []
        def server(base, endpoint, payload=None, timeout=300):
            if endpoint == "/api/tags":
                return {"models": [{"name": "test:model", "digest": "fixed-digest"}]}
            if endpoint == "/api/show":
                return {"details": {"quantization_level": "TEST"}}
            if endpoint == "/api/version":
                return {"version": "test"}
            self.assertEqual(endpoint, "/api/chat")
            calls.append(payload)
            key = json.loads(payload["messages"][-1]["content"])
            probe = "review" if "user_id" in key else "item"
            item = pairs.get((key.get("user_id"), key["item_id"])) if probe == "review" else items.get(key["item_id"])
            return envelope(answer_for(probe, item) if item else {field: None for field in payload["format"]["required"]})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            write_json(path / "input.json", dataset)
            output = path / "audit"
            prepare(output, [11, 22, 33, 44, 55], 2, dataset_path=path / "input.json")
            preview_result = preview(output, ["zero_shot", "few_shot"])
            self.assertNotIn("ground_truth", (output / "request_preview.json").read_text())
            with patch("memory_leakage_tests.runner.http_json", side_effect=server), redirect_stdout(io.StringIO()):
                run(output, models=["test:model"], permutations=99, bootstrap_samples=99)
                initial = len(calls)
                self.assertEqual(initial, preview_result["total_per_model"])
                run(output, models=["test:model"], permutations=99, bootstrap_samples=99)
                self.assertEqual(len(calls), initial)
                rows = report(output)
                pooled = [r for r in rows if r["cohort_seed"] == "ALL"]
                self.assertEqual(len(pooled), 4)
                self.assertTrue(all(r["exact_rate"] == 1 for r in pooled))
                self.assertTrue(all(r["negative_control_hallucinations"] == 0 for r in pooled))
                self.assertTrue(all(r["evidence_status"] == "EVIDENCE_DETECTED" for r in pooled))
                with self.assertRaisesRegex(ValueError, "settings"):
                    run(output, models=["test:model"], inference_seed=43, permutations=99, bootstrap_samples=99)
            self.assertFalse((output / "runs/default/.running").exists())
            for name in ("summary.txt", "summary.csv", "summary.json", "recovery_candidates.csv", "responses.jsonl"):
                self.assertTrue((output / "runs/default" / name).exists())


if __name__ == "__main__":
    unittest.main()
