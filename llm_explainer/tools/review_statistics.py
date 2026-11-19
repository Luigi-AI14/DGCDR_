"""Input and output shared by the review-level statistical test scripts."""

from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional

from llm_explainer.markdown_exporter import sanitize_model_name

QUINTILES = ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')
GRADES = (1, 2, 3, 4, 5)
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results_judge'


class ScoredPair(NamedTuple):
    quintile: str
    word_count: Optional[int]
    grade: int


@dataclass
class JudgeRun:
    domain: str
    model: str
    prompting: str
    seed: int
    path: Path
    pairs: list
    skipped: Counter
    judge: tuple
    quintile_bounds: tuple

    @property
    def condition(self):
        return self.domain, self.model, self.prompting


def add_arguments(parser, output_root):
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR,
                        help='Root of the saved judge reports (default: results_judge in this repo).')
    parser.add_argument('--domain', default='Cloth-Elec',
                        help='Domain direction or "all" (default: Cloth-Elec).')
    parser.add_argument('--prompting', choices=('few-shot', 'zero-shot', 'all'), default='few-shot',
                        help='Explanation generation mode (default: few-shot).')
    parser.add_argument('--models', '--llms', nargs='+', metavar='MODEL',
                        help='Generator model IDs or prefixes, e.g. gemma llama qwen (default: all).')
    parser.add_argument('--seeds', nargs='+', type=int, metavar='SEED',
                        help='Sampling seeds to include (default: all available).')
    parser.add_argument('--holm', action='store_true',
                        help='Apply Holm across all computed tests in this invocation and use adjusted p-values for 5%% significance labels.')
    parser.add_argument('--output', type=Path, metavar='CSV',
                        help='Override the CSV path for one domain, model, and prompting mode '
                             '(default: write one CSV per combination under {} in this repo).'.format(output_root.name))


def load_runs(args):
    if not args.results_dir.is_dir():
        raise ValueError('Judge results directory does not exist: {}'.format(args.results_dir))
    requested_models = {model.casefold() for model in args.models or ()}
    requested_seeds = set(args.seeds or ())
    matched_models = set()
    runs = {}
    for path in sorted(args.results_dir.glob('*/seed*/*/*.json')):
        domain, seed_folder = path.relative_to(args.results_dir).parts[:2]
        if args.domain != 'all' and domain != args.domain:
            continue
        seed_suffix = seed_folder[4:]
        if not seed_suffix.isdecimal():
            continue
        seed = int(seed_suffix)
        if requested_seeds and seed not in requested_seeds:
            continue

        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('is_shuffled'):
            continue
        metadata = data['metadata']
        prompting = metadata['generation_mode']
        if args.prompting != 'all' and prompting != args.prompting:
            continue
        model = metadata['generator_model']
        matches = {prefix for prefix in requested_models if model.casefold().startswith(prefix)}
        if requested_models and not matches:
            continue
        matched_models.update(matches)
        if data['sampling_seed'] != seed:
            raise ValueError('Sampling seed differs from directory in {}'.format(path))
        key = (domain, model, prompting, seed)
        if key in runs:
            raise ValueError('Multiple reports for {}: {} and {}'.format(key, runs[key].path, path))
        runs[key] = read_run(data, path, domain, model, prompting, seed)

    if not runs:
        raise ValueError('No judge reports match the selected filters.')
    if requested_models - matched_models:
        raise ValueError('No matching reports for models: {}'.format(', '.join(sorted(requested_models - matched_models))))
    missing_seeds = requested_seeds - {run.seed for run in runs.values()}
    if missing_seeds:
        raise ValueError('No matching reports for seeds: {}'.format(', '.join(map(str, sorted(missing_seeds)))))
    return [runs[key] for key in sorted(runs)]


def read_run(data, path, domain, model, prompting, seed):
    pairs = []
    skipped = Counter()
    seen = set()
    for user in data['users']:
        for item in user['items']:
            case = (user['user_id'], item['item_id'])
            if case in seen:
                raise ValueError('Duplicate review case {!r} in {}'.format(case, path))
            seen.add(case)
            quintile = item.get('quintile')
            score = item.get('score')
            if score is None or item.get('status', 'scored') != 'scored':
                skipped[quintile] += 1
                continue
            if isinstance(score, bool) or score not in GRADES or quintile not in QUINTILES:
                raise ValueError('Invalid scored grade or quintile for {!r} in {}'.format(case, path))
            word_count = item.get('review_word_count')
            pairs.append(ScoredPair(quintile, word_count, int(score)))

    metadata = data['metadata']
    judge = tuple(metadata.get(key) for key in ('model', 'system_prompt_sha256', 'instance_prompt_sha256'))
    by_quintile = data.get('quintile_metrics', {}).get('by_quintile', {})
    bounds = tuple((by_quintile.get(q, {}).get('min_words'), by_quintile.get(q, {}).get('max_words'))
                   for q in QUINTILES)
    return JudgeRun(domain, model, prompting, seed, path, pairs, skipped, judge, bounds)


def write_results(records, output, holm, *, output_root, filename_suffix, fields):
    records = [row.copy() for row in records]
    if holm:
        valid = sorted((row for row in records if row['p_value'] is not None), key=lambda row: row['p_value'])
        running_max = 0.0
        for rank, row in enumerate(valid):
            running_max = max(running_max, min(1.0, (len(valid) - rank) * row['p_value']))
            row['p_value_holm'] = running_max
    else:
        fields = tuple(field for field in fields if field != 'p_value_holm')

    p_value_field = 'p_value_holm' if holm else 'p_value'
    for row in records:
        p_value = row.get(p_value_field)
        if p_value is None:
            row['significance'] = 'not tested'
        elif p_value < 0.05:
            row['significance'] = 'significant'
        else:
            row['significance'] = 'not significant'

    by_condition = {}
    for row in records:
        condition = (row['domain'], row['model'], row['prompting'])
        by_condition.setdefault(condition, []).append(row)
    if output is not None and len(by_condition) != 1:
        raise ValueError('--output requires one domain, model, and prompting mode. '
                         'Omit --output to save a CSV for each combination.')

    files = {}
    for (domain, model, prompting), rows in by_condition.items():
        model_name = sanitize_model_name(model)
        prompting_name = prompting.replace('-', '_')
        path = output or (output_root / domain / f'{model_name}_{prompting_name}' /
                          f'{model_name}_{filename_suffix}.csv')
        if path in files:
            raise ValueError(f'Multiple model IDs map to the same CSV path: {path}')
        files[path] = rows
    for path, rows in files.items():
        write_csv(path, fields, rows)


def write_csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False,
                                newline='', encoding='utf-8') as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n',
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
