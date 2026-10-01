"""Shared experiment preparation; importing this module does not load model libraries."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE_FIELDS = ['user_id', 'seed', 'n_source_train', 'n_target_train']


def pairs(dataset):
    f = dataset.inter_feat
    return set(zip(f[dataset.uid_field].tolist(), f[dataset.iid_field].tolist()))


def histories(dataset):
    result = {}
    for u, i in pairs(dataset):
        result.setdefault(u, set()).add(i)
    return result


def token_map(dataset, field):
    # In some DGCDR datasets id2token is stale after remapping; invert the live map.
    inverse = {int(index): str(token) for token, index in dataset.field2token_id[field].items()}
    if len(inverse) != len(dataset.field2token_id[field]):
        raise ValueError('Non-unique token mapping for ' + field)
    return inverse


def token_pairs(dataset):
    users = token_map(dataset, dataset.uid_field)
    items = token_map(dataset, dataset.iid_field)
    return {(users[u], items[i]) for u, i in pairs(dataset)}


def token_histories(dataset):
    result = {}
    for user, item in token_pairs(dataset):
        result.setdefault(user, set()).add(item)
    return result


def checkpoint_config(state, path, model, seed, spec, out):
    import torch
    config = state['config']
    if config['model'] != model or int(config['seed']) != seed:
        raise ValueError(f'{path}: checkpoint model/seed differs from requested {model}/{seed}')
    target = config['target_domain']['dataset'] if model == 'DGCDR' else config['dataset']
    if target != spec['target']:
        raise ValueError(f'{path}: target domain {target} differs from {spec["target"]}')
    if model == 'DGCDR' and config['source_domain']['dataset'] != spec['source']:
        raise ValueError(f'{path}: unexpected source domain')
    # Checkpoints may contain absolute paths from another computer. Change paths only.
    if model == 'DGCDR':
        for domain in ('source', 'target'):
            name = config[domain + '_domain']['dataset']
            config[domain + '_domain']['data_path'] = str(ROOT / 'dataset' / name)
    else:
        config['data_path'] = str(ROOT / 'dataset' / target)
    config['use_gpu'] = bool(spec.get('use_gpu', True))
    config['device'] = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
    config['checkpoint_dir'] = str(out / '_no_saved_dataloaders' / model / str(seed))
    config['dataloaders_save_path'] = None
    config['save_dataloaders'] = False
    config['save_dataset'] = False
    return config


def path_from_root(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def resolve_checkpoints(metadata, seeds, overrides, checkpoint_dir):
    explicit = {}
    for entry in overrides:
        try:
            seed_text, filename = entry.split('=', 1)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError('Use --checkpoint SEED=PATH') from exc
        if seed not in seeds or seed in explicit:
            raise ValueError('Unexpected or duplicated explicit seed: ' + str(seed))
        explicit[seed] = path_from_root(filename)
    directory = path_from_root(checkpoint_dir) if checkpoint_dir else None
    paths = {}
    for seed in seeds:
        if seed in explicit:
            path = explicit[seed]
        elif directory:
            matches = [candidate for candidate in directory.rglob('DGCDR-*.pth')
                       if candidate.parent.name == str(seed)]
            if len(matches) != 1:
                raise ValueError('Expected one DGCDR checkpoint for seed %s in %s; found %s. '
                                 'Use --checkpoint SEED=PATH.' % (seed, directory, matches))
            path = matches[0].resolve()
        else:
            recorded = Path(metadata['seeds'][str(seed)]['checkpoints']['DGCDR'])
            if recorded.is_file():
                path = recorded.resolve()
            else:
                raise FileNotFoundError('Missing DGCDR checkpoint for seed %s: %s. '
                                        'Supply --checkpoint-dir or --checkpoint.' % (seed, recorded))
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[seed] = path
    if len(set(paths.values())) != len(paths):
        raise ValueError('The same checkpoint is assigned to multiple seeds')
    return paths


def distinct_counts(matrix):
    """Count distinct item IDs for each user in a sparse training graph."""
    if matrix.nnz == 0:
        return np.zeros(matrix.shape[0], dtype=np.int64)
    order = np.lexsort((matrix.col, matrix.row))
    rows, cols = matrix.row[order], matrix.col[order]
    unique = np.empty(len(rows), dtype=bool)
    unique[0] = True
    unique[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    return np.bincount(rows[unique], minlength=matrix.shape[0])


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference(analysis):
    config_path, users_path = reference_paths(analysis)
    metadata = json.loads(config_path.read_text(encoding='utf-8'))
    seeds = [int(seed) for seed in metadata['spec']['seeds']]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Missing or duplicated reference seeds')
    required = {'user_id', 'seed', 'n_source_train', 'n_target_train'}
    frame = pd.read_csv(users_path, dtype={'user_id': str})
    if not required.issubset(frame.columns):
        raise ValueError('Reference per_user.csv is missing required columns')
    if frame[list(required)].isna().any().any() or frame.duplicated(['user_id', 'seed']).any():
        raise ValueError('Reference contains null fields or duplicate user/seed pairs')
    coverage = frame.groupby('user_id').seed.agg(['size', 'nunique'])
    if set(frame.seed) != set(seeds) or not coverage.eq(len(seeds)).all().all():
        raise ValueError('Reference has incomplete or unexpected seeds')
    for name in ('n_source_train', 'n_target_train'):
        values = pd.to_numeric(frame[name], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError('Invalid training counts in reference: ' + name)
    return metadata, frame, seeds


def reference_paths(analysis):
    """Prefer existing transfer results; never replace them with a minimal CSV."""
    if (analysis / 'config.json').is_file() and (analysis / 'per_user.csv').is_file():
        return analysis / 'config.json', analysis / 'per_user.csv'
    prepared = analysis / 'preparation'
    if (prepared / 'config.json').is_file() and (prepared / 'users.csv').is_file():
        return prepared / 'config.json', prepared / 'users.csv'
    raise FileNotFoundError(
        'No reference found in %s. Supply existing transfer config.json/per_user.csv '
        'or run prepare_analysis.py --analysis %s --checkpoint-dir PATH first.'
        % (analysis, analysis))


def reconstruct_data(config, model, seed):
    """Reproduce the original per-seed data preparation without ranking either model."""
    from recbole.utils import init_seed

    init_seed(seed, config['reproducibility'])
    if model == 'DGCDR':
        from recbole_cdr.data import create_dataset, data_preparation
        dataset = create_dataset(config)
        train, valid, test = data_preparation(config, dataset)
        target, source, model_dataset = train.target_dataset, train.source_dataset, train.dataset
    else:
        from recbole.data import create_dataset, data_preparation
        dataset = create_dataset(config)
        train, valid, test = data_preparation(config, dataset)
        target, source, model_dataset = train.dataset, None, train.dataset
    return train, valid, test, target, source, model_dataset


def load_predictor(state, config, model, model_dataset, reseed=None):
    """Keep the original RNG order: transfer reseeds; attention does not."""
    if reseed is not None:
        from recbole.utils import init_seed
        init_seed(reseed, config['reproducibility'])
    if model == 'DGCDR':
        from recbole_cdr.model.cross_domain_recommender.dgcdr import DGCDR
        predictor = DGCDR(config, model_dataset).to(config['device'])
    else:
        from recbole.model.general_recommender.lightgcn import LightGCN
        predictor = LightGCN(config, model_dataset).to(config['device'])
    predictor.load_state_dict(state['state_dict'])
    if state.get('other_parameter') is not None:
        predictor.load_other_parameter(state['other_parameter'])
    return predictor


def attention_checkpoint_config(state, path, seed, source, target, data_root, device_name, scratch):
    import torch

    if 'config' not in state or 'state_dict' not in state:
        raise ValueError('%s must contain both config and state_dict, not just bare weights' % path)
    config = state['config']
    if config['model'] != 'DGCDR' or int(config['seed']) != seed:
        raise ValueError('%s has the wrong model or seed' % path)
    if config['source_domain']['dataset'] != source or config['target_domain']['dataset'] != target:
        raise ValueError('%s has the wrong source/target domains' % path)
    if not config['preference_disentangle'] or config['fuse_mode'] != 'attention':
        raise ValueError('%s does not use user disentanglement with attention' % path)
    if config['attention_mode'] not in ('all', 'part'):
        raise ValueError('Unsupported attention_mode in ' + str(path))
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but not available')
    device = torch.device(device_name)
    for domain in ('source', 'target'):
        location = data_root / config[domain + '_domain']['dataset']
        if not location.is_dir():
            raise FileNotFoundError(location)
        config[domain + '_domain']['data_path'] = str(location)
    config['device'] = device
    config['use_gpu'] = device.type == 'cuda'
    config['checkpoint_dir'] = str(scratch / str(seed))
    config['dataloaders_save_path'] = None
    config['save_dataloaders'] = False
    config['save_dataset'] = False
    return config


def split_fingerprints(splits):
    """Hash original token pairs in a stable order, independently of internal IDs."""
    return {name: hashlib.sha256(json.dumps(sorted(values), ensure_ascii=False,
                                           separators=(',', ':')).encode('utf-8')).hexdigest()
            for name, values in splits.items()}


def prepared_seed_check(metadata, seed, checkpoint, splits):
    audit = metadata['seeds'][str(seed)]
    if sha256_file(checkpoint) != audit['checkpoint_sha256']:
        raise ValueError('DGCDR checkpoint differs from preparation at seed %s' % seed)
    if split_fingerprints(splits) != audit['split_sha256']:
        raise ValueError('Reconstructed splits differ from preparation at seed %s' % seed)


def validate_transfer_preparation(spec, analysis, seed, checkpoint, result):
    """When a preparation exists, require the full transfer to match it exactly."""
    if not (analysis / 'preparation' / 'config.json').is_file():
        return
    prepared = analysis / 'preparation'
    metadata = json.loads((prepared / 'config.json').read_text(encoding='utf-8'))
    for field in ('source', 'target', 'seeds'):
        if metadata['spec'][field] != spec[field]:
            raise ValueError('Transfer %s differs from preparation' % field)
    splits = dict(result['splits'])
    splits['source_train'] = {(user, item) for user, items in result['source'].items() for item in items}
    prepared_seed_check(metadata, seed, checkpoint, splits)
    frame = pd.read_csv(prepared / 'users.csv', dtype={'user_id': str})
    reference = frame[frame.seed == seed].set_index('user_id')
    if set(reference.index) != set(result['results']):
        raise ValueError('Transfer users differ from preparation at seed %s' % seed)
    for domain in ('source', 'target'):
        actual = [len(result[domain].get(user, set())) for user in reference.index]
        if not np.array_equal(reference['n_%s_train' % domain].to_numpy(), actual):
            raise ValueError('Transfer %s counts differ from preparation at seed %s' % (domain, seed))
