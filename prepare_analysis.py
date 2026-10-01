"""Prepare a DGCDR experiment without LightGCN, training, or full-ranking evaluation.

python prepare_analysis.py --analysis transfer_results/elec_to_cloth \
    --checkpoint-dir trained_checkpoints/Elec_to_Cloth/DGCDR --device cpu

Domain names and seeds are read from the checkpoints; optional filters disambiguate
directories containing multiple experiments. Only use trusted checkpoint files.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from analysis_common import (
    ROOT, REFERENCE_FIELDS, attention_checkpoint_config, path_from_root,
    read_reference, reconstruct_data, sha256_file, split_fingerprints,
    token_histories, token_map, token_pairs,
)


def discover_checkpoints(directory, overrides, seeds=None, source=None, target=None):
    import torch

    explicit = {}
    for entry in overrides:
        try:
            seed_text, filename = entry.split('=', 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError('Use --checkpoint SEED=PATH') from exc
        if seed in explicit:
            raise ValueError('Duplicated explicit seed: %s' % seed)
        explicit[seed] = path_from_root(filename)
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError('Missing or duplicated seeds')
    if seeds is not None and set(explicit) - set(seeds):
        raise ValueError('Explicit checkpoints contain unexpected seeds')
    paths = list(explicit.values())
    if directory:
        folder = path_from_root(directory)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths.extend(path for path in sorted(folder.rglob('*.pth'))
                     if 'dataloader' not in path.name and path.resolve() not in explicit.values())
    if not paths:
        raise ValueError('Supply --checkpoint-dir or --checkpoint SEED=PATH')
    found, domains = {}, set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        if 'config' not in state or 'state_dict' not in state:
            raise ValueError('%s must contain config and state_dict, not just bare weights' % path)
        config = state['config']
        if config['model'] != 'DGCDR':
            if path in explicit.values():
                raise ValueError('Explicit checkpoint is not DGCDR: %s' % path)
            continue
        seed = int(config['seed'])
        pair = (config['source_domain']['dataset'], config['target_domain']['dataset'])
        if path in explicit.values() and explicit.get(seed) != path:
            raise ValueError('Explicit checkpoint seed differs from its config: %s' % path)
        matches = ((seeds is None or seed in seeds)
                   and (source is None or source == pair[0])
                   and (target is None or target == pair[1]))
        if not matches:
            if path in explicit.values():
                raise ValueError('Explicit checkpoint does not match requested experiment: %s' % path)
            continue
        if seed in explicit and path != explicit[seed]:
            continue
        found.setdefault(seed, []).append(path.resolve())
        domains.add(pair)
    if len(domains) != 1:
        raise ValueError('Expected one DGCDR domain pair; found %s. Use --source/--target.' % sorted(domains))
    selected_seeds = list(seeds) if seeds is not None else sorted(found)
    for seed in selected_seeds:
        if len(found.get(seed, [])) != 1:
            raise ValueError('Expected one DGCDR checkpoint for seed %s; found %s. '
                             'Use --checkpoint SEED=PATH.' % (seed, found.get(seed, [])))
    source, target = domains.pop()
    return {'source': source, 'target': target, 'seeds': selected_seeds}, {
        seed: found[seed][0] for seed in selected_seeds}


def prepare(analysis, spec, checkpoints, data_root, device='auto', threads=4):
    import torch

    if threads:
        torch.set_num_threads(threads)
    rows, audits, population = [], {}, None
    for seed in spec['seeds']:
        path = checkpoints[seed]
        logging.info('Preparing DGCDR seed %s from %s', seed, path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        config = attention_checkpoint_config(state, path, seed, spec['source'], spec['target'],
                                             data_root, device, analysis / 'preparation' / '_scratch')
        train, valid, test, target, source, model_dataset = reconstruct_data(config, 'DGCDR', seed)
        splits = {name: token_pairs(dataset) for name, dataset in (
            ('train', target), ('validation', valid.dataset), ('test', test.dataset))}
        if (splits['train'] & splits['validation'] or splits['train'] & splits['test']
                or splits['validation'] & splits['test']):
            raise ValueError('Target split overlap at seed %s' % seed)
        splits['source_train'] = token_pairs(source)
        source_histories = token_histories(source)
        target_histories = token_histories(target)
        test_histories = token_histories(test.dataset)
        users = sorted(test_histories)
        if not users:
            raise ValueError('No target test users at seed %s' % seed)
        source_ids = {token: index for index, token in token_map(source, source.uid_field).items()}
        target_ids = {token: index for index, token in token_map(target, target.uid_field).items()}
        # Never silently drop target-only users: legacy attention rejects them too.
        for user in users:
            if (user not in source_ids or user not in target_ids
                    or source_ids[user] != target_ids[user]
                    or target_ids[user] >= model_dataset.num_overlap_user):
                raise ValueError('Target test user %s lacks aligned source/target representations' % user)
        if population is not None and population != set(users):
            raise ValueError('Target test user population differs across seeds')
        population = set(users)
        rows.extend(dict(user_id=user, seed=seed,
                         n_source_train=len(source_histories.get(user, set())),
                         n_target_train=len(target_histories.get(user, set()))) for user in users)
        audits[str(seed)] = {
            'checkpoints': {'DGCDR': str(path)}, 'checkpoint_sha256': sha256_file(path),
            'split_sha256': split_fingerprints(splits),
            'train': len(splits['train']), 'validation': len(splits['validation']),
            'test': len(splits['test']), 'test_users': len(users),
        }
        del state, train, valid, test, model_dataset
    frame = pd.DataFrame(rows, columns=REFERENCE_FIELDS)
    # If transfer results already exist, require an identical reference and leave them untouched.
    if (analysis / 'config.json').is_file() and (analysis / 'per_user.csv').is_file():
        metadata, reference, seeds = read_reference(analysis)
        if seeds != spec['seeds'] or any(metadata['spec'][key] != spec[key] for key in ('source', 'target')):
            raise ValueError('Preparation differs from existing transfer domains/seeds')
        expected = reference[REFERENCE_FIELDS].sort_values(['seed', 'user_id']).reset_index(drop=True)
        actual = frame.sort_values(['seed', 'user_id']).reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError('Preparation users/counts differ from existing transfer')
        for seed in seeds:
            recorded = Path(metadata['seeds'][str(seed)]['checkpoints']['DGCDR'])
            if recorded.is_file() and sha256_file(recorded) != audits[str(seed)]['checkpoint_sha256']:
                raise ValueError('Preparation DGCDR differs from existing transfer at seed %s' % seed)
    output = analysis / 'preparation'
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'reference_kind': 'dgcdr_preparation', 'spec': spec,
                'data_root': str(data_root), 'seeds': audits}
    csv_temp = output / 'users.csv.tmp'
    frame.to_csv(csv_temp, index=False)
    json_temp = output / 'config.json.tmp'
    json_temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    csv_temp.replace(output / 'users.csv')
    json_temp.replace(output / 'config.json')
    logging.info('Prepared %d users across %d seeds in %s', len(population), len(spec['seeds']), output)
    return metadata, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, help='Experiment directory; writes only ANALYSIS/preparation/')
    parser.add_argument('--checkpoint-dir', help='Recursively discover DGCDR .pth checkpoints by saved metadata')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='SEED=PATH')
    parser.add_argument('--source', help='Optional source dataset filter')
    parser.add_argument('--target', help='Optional target dataset filter')
    parser.add_argument('--seeds', nargs='+', type=int, help='Seed order (default: existing reference, otherwise all discovered seeds, sorted)')
    parser.add_argument('--data-root', default=str(ROOT / 'dataset'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    analysis = path_from_root(args.analysis)
    seeds = args.seeds
    if seeds is None and ((analysis / 'config.json').is_file()
                          or (analysis / 'preparation/config.json').is_file()):
        _, _, seeds = read_reference(analysis)
    spec, checkpoints = discover_checkpoints(args.checkpoint_dir, args.checkpoint, seeds,
                                             args.source, args.target)
    prepare(analysis, spec, checkpoints, path_from_root(args.data_root),
            args.device, args.threads)


if __name__ == '__main__':
    main()
