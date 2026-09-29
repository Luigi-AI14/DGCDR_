"""Inspect DGCDR user attention and cross-domain shared alignment.

Example (from the project root, using the paper_env environment):
  python run_attention_analysis.py \
    --analysis transfer_results/cds_to_instruments_checkpoint_analysis \
    --checkpoint-dir trained_checkpoints/CDs_to_Instruments/DGCDR \
    --stage all --device cpu

Use --stage report to regenerate the Markdown from saved CSVs without checkpoints.
Only load checkpoint files produced by a trusted local training run.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SEED_FIELDS = [
    'user_id', 'seed', 'n_source_train', 'n_target_train',
    'attention_shared_target', 'attention_specific_target', 'shared_similarity_st',
]
USER_FIELDS = [
    'user_id', 'n_source_train_mean', 'n_target_train_mean',
    'attention_shared_target_mean', 'attention_specific_target_mean',
    'shared_similarity_st_mean', 'source_bin', 'target_bin',
]
ZERO_NORM_TOL = 1e-12


def path_from_root(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def write_csv_atomic(path, frame, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, columns=fields, index=False, float_format='%.17g')
    temp.replace(path)


def write_text_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content, encoding='utf-8')
    temp.replace(path)


def read_reference(analysis):
    metadata = json.loads((analysis / 'config.json').read_text(encoding='utf-8'))
    seeds = [int(seed) for seed in metadata['spec']['seeds']]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Missing or duplicated reference seeds')
    required = {'user_id', 'seed', 'n_source_train', 'n_target_train'}
    frame = pd.read_csv(analysis / 'per_user.csv', dtype={'user_id': str})
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


def live_token_map(dataset, field):
    mapping = dataset.field2token_id[field]
    reverse = {int(index): str(token) for token, index in mapping.items()}
    if len(reverse) != len(mapping):
        raise ValueError('Non-unique token mapping for ' + field)
    return reverse


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


def channel_measurements(g_target, c_target, s_target, c_source, torch):
    """Match DGCDR.fuse_and_update attention and measure shared cosine."""
    logits = torch.stack(((g_target * c_target).sum(dim=1),
                          (g_target * s_target).sum(dim=1)), dim=1)
    attention = torch.softmax(logits, dim=1)
    source_norm = torch.linalg.vector_norm(c_source, dim=1)
    target_norm = torch.linalg.vector_norm(c_target, dim=1)
    if (source_norm <= ZERO_NORM_TOL).any() or (target_norm <= ZERO_NORM_TOL).any():
        raise ValueError('Zero or near-zero shared representation; cosine undefined')
    cosine = (c_source * c_target).sum(dim=1) / (source_norm * target_norm)
    if not torch.isfinite(attention).all() or not torch.isfinite(cosine).all():
        raise ValueError('Non-finite attention or shared similarity')
    if not torch.allclose(attention.sum(dim=1), torch.ones_like(cosine), atol=1e-6):
        raise ValueError('Attention weights do not sum to one')
    if (cosine.abs() > 1.00001).any():
        raise ValueError('Cosine outside [-1, 1]')
    return attention, cosine


def extract_seed(seed, path, reference, source, target, data_root, device_name, scratch):
    # Keep all model dependencies out of the report-only path.
    import torch
    from recbole.utils import init_seed
    from recbole_cdr.data import create_dataset, data_preparation
    from recbole_cdr.model.cross_domain_recommender.dgcdr import DGCDR

    state = torch.load(path, map_location='cpu', weights_only=False)
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
        dataset_name = config[domain + '_domain']['dataset']
        location = data_root / dataset_name
        if not location.is_dir():
            raise FileNotFoundError(location)
        config[domain + '_domain']['data_path'] = str(location)
    config['device'] = device
    config['use_gpu'] = device.type == 'cuda'
    config['checkpoint_dir'] = str(scratch / str(seed))
    config['dataloaders_save_path'] = None
    config['save_dataloaders'] = False
    config['save_dataset'] = False
    init_seed(seed, config['reproducibility'])
    dataset = create_dataset(config)
    train, _, _ = data_preparation(config, dataset)
    predictor = DGCDR(config, train.dataset).to(device)
    predictor.load_state_dict(state['state_dict'])
    if state.get('other_parameter') is not None:
        predictor.load_other_parameter(state['other_parameter'])
    predictor.eval()

    target_map = live_token_map(train.target_dataset, train.target_dataset.uid_field)
    source_map = live_token_map(train.source_dataset, train.source_dataset.uid_field)
    target_ids = {token: index for index, token in target_map.items()}
    source_ids = {token: index for index, token in source_map.items()}
    users = reference.user_id.tolist()
    missing = (set(users) - target_ids.keys()) | (set(users) - source_ids.keys())
    if missing:
        raise ValueError('Users missing from reconstructed source/target mapping: %s' % list(sorted(missing))[:5])
    indices = np.array([target_ids[user] for user in users], dtype=np.int64)
    if any(source_ids[user] != target_ids[user] for user in users):
        raise ValueError('Source and target user indices differ for overlapping users')
    if (indices >= predictor.overlapped_num_users).any():
        raise ValueError('Requested user lacks source/target disentangled representations')

    source_counts = distinct_counts(predictor.source_interaction_matrix)[indices]
    target_counts = distinct_counts(predictor.target_interaction_matrix)[indices]
    expected_source = reference.n_source_train.to_numpy(dtype=np.int64)
    expected_target = reference.n_target_train.to_numpy(dtype=np.int64)
    if not np.array_equal(source_counts, expected_source) or not np.array_equal(target_counts, expected_target):
        raise ValueError('Reconstructed training counts differ from reference at seed %s' % seed)

    with torch.inference_mode():
        components, _, _, _, _ = predictor.forward()
        sr_c, tg_c, _, tg_s, _, tg_user = components[:6]
        selected = torch.as_tensor(indices, device=device)
        attention, cosine = channel_measurements(tg_user[selected], tg_c[selected],
                                                  tg_s[selected], sr_c[selected], torch)
        values = np.column_stack((attention.cpu().numpy(), cosine.cpu().numpy()))
    if not np.isfinite(values).all():
        raise ValueError('Non-finite extracted values at seed %s' % seed)
    result = pd.DataFrame({
        'user_id': users, 'seed': seed,
        'n_source_train': source_counts, 'n_target_train': target_counts,
        'attention_shared_target': values[:, 0],
        'attention_specific_target': values[:, 1],
        'shared_similarity_st': values[:, 2],
    })
    return result, {
        'path': str(path), 'size_bytes': path.stat().st_size,
        'sha256': sha256_file(path),
        'attention_mode': config['attention_mode'],
        'feature_mapping_way': config['feature_mapping_way'],
        'overlapped_num_users': int(predictor.overlapped_num_users),
        'evaluated_users': len(result), 'device': str(device),
    }


def table(headers, rows):
    result = ['| ' + ' | '.join(headers) + ' |',
              '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    result.extend('| ' + ' | '.join(str(value).replace('|', '\\|') for value in row) + ' |'
                  for row in rows)
    return '\n'.join(result)


def format_count(value):
    return str(int(round(value))) if abs(value - round(value)) < 1e-9 else ('%.2f' % value)


def exact_cut(value):
    return repr(float(value))


def bin_labels(values, cuts):
    if not len(cuts):
        return ['Tutti gli utenti']
    integer_counts = np.equal(values, np.floor(values)).all()
    if integer_counts and np.equal(cuts, np.floor(cuts)).all():
        limits = [int(cut) for cut in cuts]
        return ['N ≤ %d' % limits[0]] + [
            '%d ≤ N ≤ %d' % (lower + 1, upper)
            for lower, upper in zip(limits[:-1], limits[1:])
        ] + ['N ≥ %d' % (limits[-1] + 1)]
    return ['N ≤ %s' % exact_cut(cuts[0])] + [
        '%s < N ≤ %s' % (exact_cut(lower), exact_cut(upper))
        for lower, upper in zip(cuts[:-1], cuts[1:])
    ] + ['N > %s' % exact_cut(cuts[-1])]


def summarize(details, seeds):
    if set(details.columns) != set(SEED_FIELDS) or details[SEED_FIELDS].isna().any().any():
        raise ValueError('Incomplete attention per_user_seed.csv')
    if details.duplicated(['user_id', 'seed']).any() or set(details.seed) != set(seeds):
        raise ValueError('Duplicate or unexpected user/seed observations')
    coverage = details.groupby('user_id').seed.agg(['size', 'nunique'])
    if not coverage.eq(len(seeds)).all().all():
        raise ValueError('Incomplete seed coverage for at least one user')
    numeric = [name for name in SEED_FIELDS if name not in ('user_id', 'seed')]
    if not np.isfinite(details[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite attention observations')
    attention = details[['attention_shared_target', 'attention_specific_target']].to_numpy(dtype=float)
    cosine = details.shared_similarity_st.to_numpy(dtype=float)
    if ((attention < -1e-6) | (attention > 1 + 1e-6)).any() or not np.allclose(
            attention.sum(axis=1), 1, atol=1e-6) or (np.abs(cosine) > 1.00001).any():
        raise ValueError('Attention or cosine outside valid range')
    if (details[['n_source_train', 'n_target_train']].to_numpy(dtype=float) < 0).any():
        raise ValueError('Negative training interaction counts')
    summary = details.groupby('user_id', sort=True).mean(numeric_only=True)
    summary = summary.rename(columns={name: name + '_mean' for name in numeric})
    labels = {}
    cuts = {}
    for domain in ('source', 'target'):
        values = summary['n_' + domain + '_train_mean'].to_numpy(dtype=float)
        cuts[domain] = np.unique(np.quantile(values, [1 / 3, 2 / 3]))
        summary[domain + '_bin'] = np.searchsorted(cuts[domain], values, side='left')
        labels[domain] = bin_labels(values, cuts[domain])
    return summary.reset_index()[USER_FIELDS], labels, cuts


def generate_report(summary, labels, metadata, analysis):
    spec = metadata['reference']['spec']
    seeds = metadata['reference']['seeds']
    grouped = summary.groupby(['source_bin', 'target_bin'], sort=True)
    rows = []
    for source_bin in range(len(labels['source'])):
        for target_bin in range(len(labels['target'])):
            key = (source_bin, target_bin)
            group = grouped.get_group(key) if key in grouped.groups else None
            rows.append([source_bin, target_bin, 0 if group is None else len(group), *(
                ['n/d'] * 3 if group is None else [
                    '%.6f' % group[column].mean() if column != 'shared_similarity_st_mean'
                    else '%.4f' % group[column].mean() for column in (
                        'attention_shared_target_mean', 'attention_specific_target_mean',
                        'shared_similarity_st_mean')])])
    sample = summary.sort_values('user_id', kind='stable').head(10)
    sample_rows = [[row.user_id, format_count(row.n_source_train_mean),
                    format_count(row.n_target_train_mean),
                    '%.6f' % row.attention_shared_target_mean,
                    '%.6f' % row.attention_specific_target_mean,
                    '%.4f' % row.shared_similarity_st_mean]
                   for row in sample.itertuples(index=False)]
    lines = [
        '# Attention DGCDR: %s → %s' % (spec['source'], spec['target']),
        '',
        'Analisi di checkpoint DGCDR già addestrati; nessun nuovo training. '
        'Utenti del confronto originale: **%d**. Seed: %s.' % (len(summary), seeds),
        '',
        'Per ogni utente si calcola prima la media sui seed. Ogni fascia è poi la media '
        'dei propri utenti: ciascun utente pesa una volta.',
        '',
        'L’attenzione shared e specific è calcolata sulla rappresentazione utente target; '
        'i due pesi sommano a 1. Indicano come DGCDR bilancia i due canali, non la '
        'percentuale di informazione proveniente dal source. La similarità shared S–T '
        'confronta la direzione dei vettori shared dello stesso utente nei due domini: '
        'valori vicini a 1 indicano maggiore allineamento, vicini a 0 poco allineamento. '
        'Non misura il beneficio del source sul ranking.',
        '',
        '## Fasce di attività source × target',
        '',
        'Le fasce sono terzili dei conteggi di interazioni distinte nel training, '
        'mediati sui seed per utente. Le soglie coincidenti vengono accorpate.',
        '',
        table(['Fascia', 'Interazioni source medie', 'Interazioni target medie'], [
            [index, labels['source'][index] if index < len(labels['source']) else '—',
             labels['target'][index] if index < len(labels['target']) else '—']
            for index in range(max(len(labels['source']), len(labels['target'])))]),
        '',
        table(['Fascia S', 'Fascia T', 'Utenti', 'Attenzione shared media',
               'Attenzione specific media', 'Similarità shared S–T media'], rows),
        '',
        '## Esempio di utenti',
        '',
        'Primi dieci ID in ordine alfabetico: estratto di consultazione, non campione '
        'rappresentativo.',
        '',
        table(['Utente', 'Interazioni source medie', 'Interazioni target medie',
               'Attenzione shared target media', 'Attenzione specific target media',
               'Similarità shared S–T media'], sample_rows),
        '',
        '[Tutti gli utenti](per_user.csv) · '
        '[Dettaglio utente × seed](per_user_seed.csv) · '
        '[Report del confronto originale](../report.md).',
        '',
        'Dati e split sono stati ricostruiti dai checkpoint con il protocollo '
        'dell’analisi originale; in assenza dei dataloader storici, i soli pesi '
        'non dimostrano l’identità esatta degli split storici.',
        '',
    ]
    return '\n'.join(lines)


def report(analysis, output):
    metadata = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    if Path(metadata['reference']['analysis']).resolve() != analysis.resolve():
        raise ValueError('Saved attention data belongs to another reference analysis')
    seeds = [int(value) for value in metadata['reference']['seeds']]
    details = pd.read_csv(output / 'per_user_seed.csv', dtype={'user_id': str})
    summary, labels, cuts = summarize(details, seeds)
    _, reference, reference_seeds = read_reference(analysis)
    if seeds != reference_seeds or len(details) != len(reference):
        raise ValueError('Saved attention seeds/rows differ from the reference')
    expected_detail = reference[['user_id', 'seed', 'n_source_train', 'n_target_train']].sort_values(
        ['user_id', 'seed'], kind='stable').reset_index(drop=True)
    actual_detail = details[['user_id', 'seed', 'n_source_train', 'n_target_train']].sort_values(
        ['user_id', 'seed'], kind='stable').reset_index(drop=True)
    if not actual_detail.equals(expected_detail):
        raise ValueError('Saved per-seed user/count rows differ from the reference')
    expected = reference.groupby('user_id')[['n_source_train', 'n_target_train']].mean()
    observed = summary.set_index('user_id')
    if len(summary) != len(expected) or not observed.index.equals(expected.index) or not np.allclose(
            observed[['n_source_train_mean', 'n_target_train_mean']].to_numpy(),
            expected.to_numpy(), atol=1e-9):
        raise ValueError('Attention users/counts differ from the reference analysis')
    metadata['bin_cuts'] = {domain: values.tolist() for domain, values in cuts.items()}
    metadata['bin_labels'] = labels
    metadata['population'] = len(summary)
    write_csv_atomic(output / 'per_user.csv', summary, USER_FIELDS)
    write_text_atomic(output / 'report.md', generate_report(summary, labels, metadata, analysis))
    write_text_atomic(output / 'config.json', json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    logging.info('Report written for %d users in %s', len(summary), output)


def extract(analysis, output, args):
    import torch
    metadata, reference, seeds = read_reference(analysis)
    source = metadata['spec']['source']
    target = metadata['spec']['target']
    checkpoint_paths = resolve_checkpoints(metadata, seeds, args.checkpoint, args.checkpoint_dir)
    rows, checks = [], {}
    if args.threads:
        torch.set_num_threads(args.threads)
    for seed in seeds:
        logging.info('Extracting seed %s from %s', seed, checkpoint_paths[seed])
        result, check = extract_seed(
            seed, checkpoint_paths[seed], reference[reference.seed == seed].copy(),
            source, target, path_from_root(args.data_root), args.device, output / '_scratch')
        rows.append(result)
        checks[str(seed)] = check
        logging.info('Seed %s: %d users', seed, len(result))
    details = pd.concat(rows, ignore_index=True)
    if len(details) != len(reference):
        raise ValueError('Extraction does not cover the reference population')
    # Check all values before committing any output.
    summarize(details, seeds)
    record = {
        'reference': {'analysis': str(analysis), 'spec': {'source': source, 'target': target},
                      'seeds': seeds, 'reference_rows': len(reference)},
        'extraction': {'zero_norm_tolerance': ZERO_NORM_TOL,
                       'data_root': str(path_from_root(args.data_root)),
                       'checkpoint_checks': checks},
    }
    write_csv_atomic(output / 'per_user_seed.csv', details, SEED_FIELDS)
    write_text_atomic(output / 'config.json', json.dumps(record, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, help='Directory containing the reference config.json/per_user.csv')
    parser.add_argument('--output', help='Attention output directory (default: ANALYSIS/attention)')
    parser.add_argument('--stage', choices=['all', 'extract', 'report'], default='all')
    parser.add_argument('--checkpoint-dir', help='Directory with DGCDR checkpoints, optionally grouped by seed')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='SEED=PATH')
    parser.add_argument('--data-root', default=str(ROOT / 'dataset'), help='Root containing the source/target dataset folders')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    # RecBole reads sys.argv, so hide this script's arguments from it.
    sys.argv = [sys.argv[0]]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    analysis = path_from_root(args.analysis)
    output = path_from_root(args.output) if args.output else analysis / 'attention'
    if args.stage in ('all', 'extract'):
        extract(analysis, output, args)
    if args.stage in ('all', 'report'):
        report(analysis, output)


if __name__ == '__main__':
    main()
