# DGCDR

Pytorch implementation for paper "**Enhancing Transferability and Consistency in Cross-Domain Recommendations via Supervised Disentanglement**"[[1]](#1). The paper has been accepted for presentation at the [RecSys 2025 conference](https://recsys.acm.org/recsys25/accepted-contributions/). Thank you for your support and citation!

## Datasets
The dataset is processed based on [Amazon](https://amazon-reviews-2023.github.io/index.html)[[2]](#2), Douban[[3]](#3). 

Each domain contains a interaction file in the following format:

| Suffix | Content | Format |
|---|---|---|
| *.inter* | User-item interaction | user_id, item_id, rating |

## Training
You can use this command to train the model:

`python run_recbole_cdr.py`

The configuration settings are in a YAML file in  DGCDR/recbole_cdr/properties/.

## LLM explanation pipeline: split and sampling seeds

The pipeline always uses **42** for the RecBole dataset split and for generation
and judge inference. `--seed` selects the user cohort; it does not change the split.

```powershell
# One sampling cohort (default seed: 11; default users per cohort: 200)
python run_llm_explainer_pipeline.py --seed 33

# All five cohorts: 11, 22, 33, 44, 55, with all three generators and the judge
python run_llm_explainer_pipeline.py --all_seeds

# Judge existing reports for all five cohorts
python run_llm_explainer_pipeline.py --all_seeds --skip_generation
```

`--seed` and `--all_seeds` are mutually exclusive. Existing generation-mode,
domain-pair, user-count, dry-run and phase-skip options still apply to every run.

Explanation generation defaults to **v3 few-shot**; judge evaluation defaults to
**zero-shot**. These are independent: use `--generation_mode zero-shot` (or
`--no_few_shot`) to disable generation examples, or `--judge_prompt few-shot` to
enable judge examples.

Generation selects `llm_explainer/prompts/EXPLANATION_V3_<source>-<target>_FEW_SHOT.txt`
or `EXPLANATION_V3_<source>-<target>_ZERO_SHOT.txt` using `--domain_pair` and
`--generation_mode`. For example, the default is
`EXPLANATION_V3_Cloth-Elec_FEW_SHOT.txt`. Reports, cache hashes and cohort
token-budget checks use the selected pair's template.

Each v3 generation file contains both instructions and an `INPUT` section.
After filling the placeholders, the client sends the instructions (including
few-shot demonstrations) in the system message and the `INPUT` section in the
user message. Generation does not need a separate instance-template file.

Judge evaluation uses `JUDGE_PROMPT_ALIGNMENT_ZERO_SHOT.txt` by default;
`--judge_prompt few-shot` selects `JUDGE_PROMPT_ALIGNMENT_FEW_SHOT.txt`.
`--system_prompt` continues to accept a custom judge file.
Per-item judge data is formatted using `JUDGE_INSTANCE_PROMPT.txt`;
`--instance_prompt` overrides this template.

Generation prompt saving is disabled by default in both `run_llm_explanations.py`
and `run_llm_explainer_pipeline.py`. Use `--save_prompts` to enable it, or
`--no_save_prompts` to explicitly keep it disabled:

```powershell
python run_llm_explainer_pipeline.py --all_seeds
python run_llm_explanations.py --seed 11 --save_prompts
```

The pipeline forwards this choice to every generation run. When disabled, no
prompt directories, item prompt files or generation-template copies are written;
existing saved prompts are left untouched. Prompts are still built in memory,
and their SHA-256 hashes remain in the explanation reports, including cache hits.
Reports record `save_prompts: false`, `prompts_dir: null`, null prompt-file paths
and empty user prompt-file lists. Explanations, metrics and judge evaluation work
normally. `--prompts_dir` is used only when saving is enabled. Python callers of
`build_item_prompt` and `build_user_prompt` can also pass `save_prompts=False`.

On first generation, the pipeline freezes the complete eligible pool into five
disjoint cohorts, sampling from the remaining users in seed order, as the leakage
extension does. Full profiles for the selected users are saved in
`results_explanations/cohorts/<domain_pair>/users<num_users>.json` and reused across
models, prompting modes and later invocations. Even a single-seed run prepares
the five cohort definitions, so it requires at least `5 * num_users` eligible
users (1,000 by default). The cohorts are deterministic; a single-seed run uses
the same users as that seed in batch mode. Leakage-audit demonstration exclusions
are specific to that audit and are not applied to the explainer's synthetic examples.

The manifest is a frozen snapshot. To prepare new cohorts after changing the data
or filtering rules, archive the existing manifest before rerunning. Its settings
and checksum are validated before reuse.

Before freezing cohorts, each seed samples its initial users and checks their
complete generation prompts with the Qwen, Llama, and Gemma tokenizers. Users
whose prompts exceed the input budget are discarded and replaced by additional
draws from the same seeded random generator, without resetting it. Accepted
users keep their full histories: no review is shortened. Rejected users are not
drawn again, and the five final cohorts remain disjoint. If too few users fit,
preparation fails without saving a partial manifest.

The default context is 32,768 tokens. The check reserves 4,096 output tokens and
256 tokens for chat formatting and safety, leaving **28,416 message-content
tokens**. It checks every candidate title and both generation modes so all three
models and zero-shot/few-shot runs share the same final users. The candidate's
held-out review is not used in this check. `--num_ctx` controls generation context
in both the pipeline and standalone validator; `--judge_num_ctx` remains separate.

The manifest and explanation JSON reports record the initial user IDs, excluded
users with token counts and reasons, replacement IDs in draw order, final IDs,
budget settings, and tokenizer revisions/checksums. The exclusion count makes
the restriction to users with fitting histories measurable.

Only tokenizer files are downloaded, once, into `cache/generation_tokenizers/`;
cached runs work offline. Install the tokenizer dependencies from
`requirements.txt` in the execution environment. No model weights are loaded
for this preprocessing step. `--independent_sampling` applies the same check
and replacement rule to its single sample. Mock `--dry_run` generation also
performs the eligibility check.

**Existing manifests from before this filtering rule are rejected.** Archive
`results_explanations/cohorts/<domain_pair>/users<num_users>.json` before the first
new run (or give the standalone validator a new `--cohort_manifest` path). Old
manifests are never overwritten automatically. A change to the context budget,
prompt version, or template also requires a new manifest.

Output filenames use single underscores between fields and readable dates:

```text
explanations_qwen3.5-9b_v3-few-shot_seed-11_2026-11-03_12-00-00.json
judge_qwen3.5-9b_by-qwen3.8-27b_zero-shot_seed-11_2026-11-03_12-00-00.json
```

Companion Markdown files use the same basename. Timestamps stop at seconds;
only if a name already exists, a short suffix such as `_2` or `_3` is added.
The fixed split seed is recorded inside reports, not in filenames or directory
names. All output types use domain/seed/generator folders:

```text
results_explanations/
  cohorts/Cloth-Elec/users200.json
  Cloth-Elec/seed11/qwen3.5_9b_v3_few-shot/
    explanations_....json
    explanations_....md
    shuffled/
      explanations_..._shuffled-....json
      explanations_..._shuffled-....md
results_judge/
  Cloth-Elec/seed11/qwen3.5_9b_v3_few-shot/
    judge_....json
    judge_....md
    benchmark_judge_congruent_vs_shuffled_....json
    benchmark_judge_congruent_vs_shuffled_....md
results_judge_shuffled/
  Cloth-Elec/seed11/qwen3.5_9b_v3_few-shot/
    judge_..._shuffled_....json
    judge_..._shuffled_....md
legacy_results/
  results/
  results_judge/
  saved_prompts/
  archive_manifest.json
```

The same structure applies to seeds 22, 33, 44 and 55 and the other generators.
Generation folder names now use `zero-shot` / `few-shot`, consistently with judge
folders. Explicit `--output_dir` overrides remain supported. The standalone judge
automatically chooses the normal or shuffled root from the input report.

JSON and Markdown reports record the distinct seed roles; JSON also records cohort IDs and
split fingerprints. Judge-only result discovery requires matching split seed,
sampling seed and user count. Legacy reports without both seed fields must be
regenerated for these runs.

`llm_explainer/judge/shuffle_for_judge.py` supports the same readable filenames
and nested judge directories. Its `--latest` selection reads report metadata and
ignores cohort manifests. Automatic comparison selects a compatible original and
shuffled pair from the same generator, sampling cohort, source run and judge
protocol. Its automatic comparison searches both judge roots. Existing report
names remain supported when an older report is explicitly selected.

Older outputs and their saved prompts are archived under `legacy_results`, with
their original filenames, directory structure and contents. `archive_manifest.json`
records file locations, sizes and SHA-256 checksums. Paths embedded in historical
reports remain original provenance; explicit old result references are resolved to
their archived locations by the judge, shuffler and Markdown exporter. Automatic
discovery and default cache imports use the active roots and do not scan the archive.

Shuffled datasets also use single underscores and timestamps ending at seconds,
for example `explanations_qwen3.5-9b_v3-few-shot_shuffled-inter-user_seed-11_shuffle-seed-42_2026-11-03_12-00-00.json`.
Here `seed-11` identifies the source user cohort; `shuffle-seed-42` identifies the
shuffle permutation. The split seed stays inside the JSON and Markdown reports.
The shuffled JSON and Markdown card share a basename, and duplicate names receive
the same short numeric suffix used by generation and judge reports.

Standalone validation uses the same defaults as the pipeline: 200 users, sampling
seed 11, split/inference seeds 42, `BAAI/bge-large-en-v1.5`, disabled explanation
and embedding caches, v3 few-shot generation, and no saved prompts. The default
generator is `qwen3.5:9b`; select another pipeline generator with `--model`.

Frozen cohorts are selected automatically using
`<output_dir>/cohorts/<domain_pair>/users<num_users>.json`, shared with the pipeline
when the default output root is used. `--cohort_manifest <path>` overrides that
location. Use `--independent_sampling` to bypass frozen cohorts and allow any
sampling seed. Use `--cache_llm` / `--cache_sbert` to explicitly enable caches.
Existing Python callers of
`DataExtractor(seed=...)` retain their original behavior; new callers can specify
`split_seed` and `sampling_seed` separately.

Standalone judging defaults to `qwen3.8:27b` and 200 users, matching the pipeline.
Run it with `python llm_explainer/judge/run_llm_judge.py --results_file <report.json>`
or `python -m llm_explainer.judge.run_llm_judge --results_file <report.json>`.
Other shared defaults are zero-shot alignment judging, temperature 0, inference
seed 42, context 8192, initial token limit 256, one retry, weighted scoring enabled,
and 20 top log probabilities. `--results_file` remains required; domain and sampling
seed come from the input report. `--all_users` removes the 200-user limit.
The shuffle script also uses the same judge model and user limit, with an
`--all_users` override.

Generation and judge defaults are shared in `llm_explainer/config.py` to prevent
pipeline and standalone defaults from drifting apart. For example:

```powershell
# Same cohort/settings as the pipeline, running only one generator
python run_llm_explanations.py --model llama3.1:8b

# Explicit opt-ins for an independent five-user run with caches
python run_llm_explanations.py --independent_sampling --seed 42 --num_users 5 --cache_llm --cache_sbert
```

## Acknowledgement
The implementation is based on the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR).


## References
## References
<a id="1">[1]</a> Yuhan Wang, Qing Xie, Zhifeng Bao, Mengzi Tang, Lin Li, and Yongjian Liu. 2025. Enhancing Transferability and Consistency in Cross-Domain Recommendations via Supervised Disentanglement. In Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys ’25). Association for Computing Machinery, New York, NY, USA, 104–113. https://doi.org/10.1145/3705328.3748044

<a id="2">[2]</a> Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2024. Bridging Language and Items for Retrieval and Recommendation. arXiv preprint arXiv:2403.03952 (2024).

<a id="3">[3]</a> Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu
Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, et al., Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms,in: Proceedings of the 30th ACM International Conference on Information & Knowledge Management, Association for Computing Machinery, New York, NY, USA, 2021, pp. 4653–4664


