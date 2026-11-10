# ID-based memory audit

Adaptation of Di Palma et al., **Do LLMs Memorize Recommendation Datasets?
A Preliminary Study on MovieLens-1M**, SIGIR 2025,
[DOI: 10.1145/3726302.3730178](https://doi.org/10.1145/3726302.3730178).

The audit asks whether `gemma4:12b`, `qwen3.5:9b` and `llama3.1:8b` can recover
Amazon records from their original IDs. It does not generate explanations or run
an LLM judge. The existing explanation pipeline is unchanged.

## Quick start

Run these commands from the **DGCDR_ repository root**, in your existing DGCDR
environment. Commands are identical on Windows and Linux. Use a new output
directory when changing preparation settings.

```sh
python -B -m memory_leakage_tests prepare --output-dir memory_leakage_tests/results/cloth_elec --domain-pair Cloth-Elec --split-seed 42 --cohort-seeds 11 22 33 44 55 --num-users 200
python -B -m memory_leakage_tests preview --output-dir memory_leakage_tests/results/cloth_elec
python -B -m memory_leakage_tests run --output-dir memory_leakage_tests/results/cloth_elec
```

1. **prepare** reconstructs the fixed RecBole split and exports the complete
   eligible held-out pool. It reserves three demonstrations, then samples five
   cohorts of 200 users without replacement. The default cohort seeds are
   `11 22 33 44 55`; they never change the split. Preparation fails if fewer than
   1,000 eligible users remain after demonstration exclusions.
2. **preview** validates the snapshot and manifest, lists requests per model, and
   saves example requests without target answers. It never contacts Ollama.
3. **run** uses the same cohorts for all three models, in zero-shot and few-shot
   modes. Ollama must already serve the exact requested tags. It does not pull or
   substitute models. Inference runs sequentially and saves each response before
   continuing. This can be a substantial run: every retained held-out review is
   tested, not just one review per user. Preview shows the logical request budget;
   identical item requests across cohorts reuse their first successful response.

The first preparation loads the full compact metadata cache (about 750 MB for
Cloth-Elec) and constructs profiles for the complete eligible-user pool. Allow
additional RAM for decoded metadata/profiles. Without a compact cache, the
existing extractor scans the raw files; preprocessing beforehand is recommended:

```sh
python -B preprocess_compact_data.py --domain_pair Cloth-Elec
```

For a small smoke run use a **new** directory and `--num-users 1`. This still
extracts the complete eligible pool before sampling, so it reduces inference
work, not preparation work.

## What is queried

| Probe | Input | Reconstructed fields |
|---|---|---|
| Item | Original `parent_asin` as `item_id`, dataset/domain identity | `item_title` |
| Review | Original `user_id` + `parent_asin`, dataset/domain identity | `review_title`, `review_text`, `rating` |

Item metadata currently means **product title**, matching the existing
`DataExtractor` output and the paper's MovieID-to-title experiment. Additional
product fields (brand, features, descriptions, etc.) are not scored in this
version. Review metadata includes all three fields already extracted by DGCDR.

There is no target title, review, history, candidate list, explanation or gold
value in the target prompt or response schema. Every query starts a fresh
conversation. The few-shot examples are frozen independently using demo seed
`314159`; their users, item IDs, normalized item titles and review texts are
excluded from **all five** evaluation cohorts, including the zero-shot condition.
Both conditions therefore evaluate identical records. Users remain disjoint;
items can occur for different users or seeds.

The snapshot keeps DGCDR's existing eligibility rules: the same source/history
requirements, ratings >= 4, nonblank review text, and >= 5 words across the
held-out review title and text. All surviving held-out items per sampled user
are retained. This is an audit of that filtered population, not all Amazon users
or reviews. Missing/placeholder product titles are excluded from item scoring
and counted explicitly; their reviews can still be tested.

## Read the result

Start with:

```text
memory_leakage_tests/results/cloth_elec/runs/default/summary.txt
```

The text report shows pooled and per-seed recovery counts, coverage, shuffled
baseline, failures, uncertainty and an evidence status. `summary.csv` and
`summary.json` additionally contain per-field matches, complete-record matches,
abstentions, majority-rating baselines and negative-control results.

- **EVIDENCE_DETECTED:** pooled recovery exceeds the shuffled-reference null
  with Bonferroni-adjusted p <= 0.05 across all model/mode/probe comparisons,
  and at least two distinct strong matches exist. For items these are different
  item IDs. For reviews these must be complete-record matches whose review texts
  contain >= 12 words and occur exactly once in the **complete eligible pool**.
  This is a flag for inspection, not proof of training-set membership.
- **RECOVERY_DETECTED_REVIEW_REQUIRED:** at least one exact recovery exists,
  but the stronger pooled criteria were not met. Inspect it manually.
- **NO_EVIDENCE_DETECTED:** no exact recovery under these prompts, with complete
  requests/controls and scorable metadata. This does **not** prove no leakage.
- **INCONCLUSIVE:** no exact recovery and incomplete/failed requests, failed
  controls, or unavailable item metadata. Seed rows use the descriptive
  `RECOVERY_DETECTED` label when any match exists; they are not extra hypothesis
  tests.

`recovery_candidates.csv` contains exact and high-similarity examples, including
expected and generated values and a blank `manual_verdict` column. Compare them
before claiming user-specific memorization. `responses.jsonl` records every
request, raw response, model, task ID, inference/config hashes, timing, error and
retry attempt count. Ground truth is in the frozen snapshot, outside inference
payloads. These artifacts contain original IDs and review texts.

## Scoring and controls

- Primary coverage uses NFKC normalization, case folding and collapsed whitespace;
  punctuation remains significant. Raw exact matches are also reported.
  Full-record review recovery requires title, text and rating to match.
  Empty reference titles do not inflate title-field coverage.
- All scheduled, scorable requests are in the coverage denominator, including
  abstentions, format/transport failures, truncated responses and missing tasks.
  Failures are reported separately and never treated as successful abstentions.
- Unique items are tested once per cohort. Pooled item coverage deduplicates
  items across seeds; repeated item prompts reuse a successful response.
  Review coverage counts distinct held-out user/item pairs.
- The shuffled-reference control permutes the references while holding model
  outputs fixed (499 deterministic permutations by default; plus-one p-value).
  Permutations retain reference frequencies, including common repeated texts.
  This null is an ID/reference-association diagnostic, not a calibrated test of
  pretraining membership. It does not establish that a recovered review depended
  specifically on the supplied user ID rather than product-level knowledge.
- Three synthetic item IDs and three synthetic user/item pairs per seed/mode
  test whether the model abstains. They are absent from the frozen eligible
  snapshot and identical across modes/models. Their global nonexistence or
  absence from LLM training data is **not** assumed.
- 95% percentile bootstrap intervals resample users for reviews and unique items
  for item coverage (499 samples by default). With fewer than two clusters the
  interval is unavailable. All-zero/all-one intervals can be degenerate; these
  are descriptive intervals, not guarantees about unseen data.
- Fuzzy similarity is exploratory only (difflib SequenceMatcher, normalized
  strings, first 10,000 characters). It never triggers an evidence flag.
  Rating accuracy alone is weak evidence, especially after filtering ratings >= 4;
  compare it with the reported most-frequent-rating baseline.

Product-title recovery indicates item knowledge. Verbatim, distinctive review
recovery is stronger evidence relevant to evaluation contamination. Neither
directly proves inclusion of the specific dataset in pretraining, causal use of
memorization during explanation generation, or inflated LLM-as-a-Judge scores.
The judge intentionally receiving a reference review for evaluation is different
from the explanation generator receiving its held-out answer.

## Resume, move to Linux, or change settings

Re-run the same `run` command to resume. Successful/abstained responses are frozen;
prior failures are retried only with `--retry-errors`. Transport retries use the
identical request and never reveal ground truth. JSON format errors are not
repaired. `--retry-errors` keeps the last attempt for each task while retaining
all attempts in the journal. A crash-partial final journal line is safely removed
on resume; corruption in the middle is rejected. A `.running` lock prevents two
writers; after a process crash, remove it only after checking the process exited.

```sh
python -B -m memory_leakage_tests run --output-dir memory_leakage_tests/results/cloth_elec --retry-errors
python -B -m memory_leakage_tests report --output-dir memory_leakage_tests/results/cloth_elec
```

`report` can also summarize an interrupted run, marking pending requests as
missing. It needs neither RecBole nor a running Ollama server.

Preparation and inference are separate: copy the entire prepared directory to
Linux and run inference there without reconstructing its split. Alternatively,
prepare directly on Linux in the existing DGCDR environment. Frozen snapshots
can be imported to build different disjoint cohorts without reloading RecBole:

```sh
python -B -m memory_leakage_tests prepare --dataset memory_leakage_tests/results/cloth_elec/dataset.json --output-dir memory_leakage_tests/results/other_cohorts --cohort-seeds 61 62 63 64 65 --num-users 200
```

On import, the snapshot's domain, split and filters are authoritative. Cohort
disjointness is guaranteed within each prepared audit, not across independently
prepared directories or domain-pair audits.

Default inference uses temperature 0, top_p 1, top_k 0, repeat penalty 1,
inference seed 42, context 8192, output limit 4096, and thinking disabled. Settings
are configurable with `--num-ctx`, `--max-tokens`, `--inference-seed`,
`--thinking off|on|default`, `--ollama-url`, `--models`, `--modes`,
`--permutations` and `--bootstrap-samples`. Choose sufficient context/output
limits for long reviews and demonstrations; Ollama token counts and finish
reasons are preserved in the journal. A length-limited response is a failure.
The available backend/model version may constrain thinking controls; use
`--thinking default` if explicit thinking control is unsupported.

Resume refuses changes to the manifest, audit code, requested models/digests,
Ollama version, endpoint or inference/analysis settings. Use a new `--run-name`
for those changes. Copying results between operating systems may also change
code bytes or backend metadata; start a new run name rather than mixing runs.
Temperature 0 does not promise bit-for-bit equivalence across platforms.

## Dependencies and validation

The audit package itself uses **only Python's standard library** (Python 3.8+).
Real data preparation lazily imports the existing DGCDR/RecBole environment;
do not install the reference inspector's requirements just to run this audit.
SGLang is not required on Windows or Linux. Ollama serves the same HTTP interface
on either platform. No new packages or model downloads are performed by the code.

```sh
python -B -m unittest discover -s memory_leakage_tests/tests -v
```

Tests use synthetic records and a simulated Ollama transport, not the real three
models. They check sampling/isolation, prompt/schema exposure, scoring, controls,
fingerprint guards, crash recovery and a prepare-to-report/resume run.

Provenance includes split hashes, interaction/config/extractor/compact-cache
SHA-256 hashes, dependency versions and the exact extracted snapshot hash. Raw
multi-GB metadata files use path/size/mtime rather than full-file hashes. Existing
compact caches are reused, not silently rebuilt: if you replace raw metadata,
rebuild the compact cache first. Outputs are ignored by Git under this folder.
