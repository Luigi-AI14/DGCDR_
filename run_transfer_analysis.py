r"""Evaluate existing DGCDR and LightGCN checkpoints without training.

python run_transfer_analysis.py \
  --source 'NOME_DATASET_SOURCE_PET' \
  --target 'NOME_DATASET_TARGET_BEAUTY' \
  --checkpoint-dir saved \
  --output transfer_results/pet_to_beauty

python run_transfer_analysis.py \
  --source AmazonCDs_AmazonInstruments_commonUser_3-core \
  --target AmazonInstruments_AmazonCDs_commonUser_3-core \
  --checkpoint-dir saved \
  --output transfer_results/cds_to_instruments
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from recbole.data import create_dataset as create_light_dataset
from recbole.data import data_preparation as prepare_light_data
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.utils import init_seed
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.model.cross_domain_recommender.dgcdr import DGCDR

ROOT = Path(__file__).resolve().parent
# Defaults can be edited here; command-line options and YAML entries override them.
DEFAULT_CHECKPOINT_DIR = ROOT / 'saved'
CHECKPOINT_OVERRIDES = {
    # 'DGCDR': {2022: '/absolute/path/to/DGCDR-checkpoint.pth'},
    # 'LightGCN': {2022: '/absolute/path/to/LightGCN-checkpoint.pth'},
}
def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


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


def checkpoint_paths(spec, args):
    """Explicit YAML/CLI paths take precedence; discover remaining .pth by metadata."""
    explicit = {}
    for model, entries in {**CHECKPOINT_OVERRIDES, **spec.get('checkpoints', {})}.items():
        if model not in ('DGCDR', 'LightGCN'):
            raise ValueError('Unknown checkpoint model: ' + model)
        for seed, filename in entries.items():
            explicit[(model, int(seed))] = Path(filename).expanduser().resolve()
    for entry in args.checkpoint:
        try:
            key, filename = entry.split('=', 1)
            model, seed = key.split(':', 1)
            if model not in ('DGCDR', 'LightGCN'):
                raise ValueError()
            explicit[(model, int(seed))] = Path(filename).expanduser().resolve()
        except ValueError as exc:
            raise ValueError('Use --checkpoint MODEL:SEED=/path/to/model.pth') from exc
    wanted = {(model, seed) for seed in spec['seeds'] for model in ('DGCDR', 'LightGCN')}
    for key, path in explicit.items():
        if key not in wanted or not path.is_file():
            raise ValueError(f'Unexpected or missing checkpoint {key}: {path}')
    found = {key: [path] for key, path in explicit.items()}
    if wanted - explicit.keys():
        directory = Path(args.checkpoint_dir or spec.get('checkpoint_dir', DEFAULT_CHECKPOINT_DIR)).expanduser()
        if not directory.is_absolute():
            directory = ROOT / directory
        if not directory.is_dir():
            raise FileNotFoundError(f'Checkpoint directory not found: {directory}')
        for path in directory.rglob('*.pth'):
            if 'dataloader' in path.name or path in explicit.values():
                continue
            if not path.name.startswith(('DGCDR-', 'LightGCN-')):
                continue
            state = torch.load(path, map_location='cpu', weights_only=False)
            if 'config' not in state or 'state_dict' not in state:
                continue
            config = state['config']
            key = (config['model'], int(config['seed']))
            target = config['target_domain']['dataset'] if key[0] == 'DGCDR' else config['dataset']
            source_matches = (key[0] != 'DGCDR' or
                              config['source_domain']['dataset'] == spec['source'])
            if key in wanted and target == spec['target'] and source_matches:
                found.setdefault(key, []).append(path)
    for key in sorted(wanted):
        matches = found.get(key, [])
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {key} checkpoint; found {matches}. '
                             'Specify the intended file with --checkpoint or checkpoints in YAML.')
    return {key: paths[0] for key, paths in found.items()}


@torch.no_grad()
def individual_metrics(model, loader, device, k):
    """Common full-ranking; exact ties broken by increasing shared item ID."""
    model.eval()
    if hasattr(model, 'init_restore_e'):
        model.init_restore_e()
    else:
        model.restore_user_e = model.restore_item_e = None
    dataset = loader.dataset
    output = {}
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)
    for interaction, history, positive_u, positive_i in loader:
        users = interaction[dataset.uid_field].tolist()
        scores = model.full_sort_predict(interaction.to(device)).view(len(users), dataset.item_num)
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite unmasked scores')
        scores[:, 0] = -torch.inf
        if history is not None:
            scores[history] = -torch.inf
        # Stable sorting makes cutoff ties reproducible for both models.
        top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(k, scores.shape[1])]
        finite = torch.isfinite(scores.gather(1, top)).cpu().numpy()
        top = top.cpu().numpy()
        positives = [set() for _ in users]
        for row, item in zip(positive_u.tolist(), positive_i.tolist()):
            positives[row].add(item)
        for row, u in enumerate(users):
            if not positives[row] or u in output:
                raise ValueError('Empty positives or repeated user')
            hits = np.array([item in positives[row] and finite[row, j]
                             for j, item in enumerate(top[row])], dtype=float)
            output[u] = (float(np.dot(hits, discounts[:len(hits)]) / ideal[min(k, len(positives[row])) - 1]),
                         float(hits.sum() / len(positives[row])))
    if not output:
        raise ValueError('No evaluable users')
    return output


def reconstruct(spec, out, model, seed, path):
    logging.info('Reconstructing %s seed=%d from %s', model, seed, path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint_config(state, path, model, seed, spec, out)
    init_seed(seed, config['reproducibility'])
    if model == 'DGCDR':
        dataset = create_dataset(config)
        train, valid, test = data_preparation(config, dataset)
        target = train.target_dataset
        source = train.source_dataset
        model_dataset = train.dataset
    else:
        dataset = create_light_dataset(config)
        train, valid, test = prepare_light_data(config, dataset)
        target = train.dataset
        source = None
        model_dataset = target
    datasets = {'train': target, 'validation': valid.dataset, 'test': test.dataset}
    splits = {name: token_pairs(data) for name, data in datasets.items()}
    if splits['train'] & splits['validation'] or splits['train'] & splits['test'] or splits['validation'] & splits['test']:
        raise ValueError(f'{model}/{seed}: target split overlap')
    candidates = set(token_map(target, target.iid_field).values()) - {'[PAD]'}
    init_seed(seed, config['reproducibility'])
    predictor = (DGCDR(config, model_dataset) if model == 'DGCDR'
                 else LightGCN(config, model_dataset)).to(config['device'])
    predictor.load_state_dict(state['state_dict'])
    if state.get('other_parameter') is not None:
        predictor.load_other_parameter(state['other_parameter'])
    matrix = predictor.target_interaction_matrix if model == 'DGCDR' else predictor.interaction_matrix
    if set(zip(matrix.row.tolist(), matrix.col.tolist())) != pairs(target):
        raise ValueError(f'{model}/{seed}: model graph differs from reconstructed training data')
    values = individual_metrics(predictor, test, config['device'], spec['k'])
    users = token_map(test.dataset, test.dataset.uid_field)
    result = {users[user]: metrics for user, metrics in values.items()}
    if len(result) != len(values):
        raise ValueError(f'{model}/{seed}: duplicate original user ID')
    logging.info('Evaluated %s seed=%d: %d users, NDCG@%d=%.8f', model, seed,
                 len(result), spec['k'], np.mean([v[0] for v in result.values()]))
    return dict(results=result, splits=splits, candidates=candidates,
                source=token_histories(source) if source is not None else None,
                target=token_histories(target), test=token_histories(test.dataset),
                repeatable=bool(config['repeatable']),
                threshold=(config['target_domain']['threshold'] if model == 'DGCDR'
                           else config['threshold']))


def compare_pair(seed, light, dgcdr):
    for phase in ('train', 'validation', 'test'):
        if light['splits'][phase] != dgcdr['splits'][phase]:
            raise ValueError(f'Seed {seed}: DGCDR/LightGCN {phase} interactions differ; '
                             'cannot make a paired user comparison')
    if light['candidates'] != dgcdr['candidates']:
        raise ValueError(f'Seed {seed}: target candidate item sets differ')
    if light['repeatable'] != dgcdr['repeatable']:
        raise ValueError(f'Seed {seed}: evaluation repeatable policies differ')
    if set(light['results']) != set(dgcdr['results']):
        raise ValueError(f'Seed {seed}: evaluated target users differ')
    logging.info('Seed %d: paired splits match (%d train, %d valid, %d test)', seed,
                 *[len(light['splits'][phase]) for phase in ('train', 'validation', 'test')])


def write_csv(spec, out, results):
    rows = []
    for seed in spec['seeds']:
        light, dgcdr = results[('LightGCN', seed)], results[('DGCDR', seed)]
        for user in sorted(light['results']):
            left, right = light['results'][user], dgcdr['results'][user]
            ns = len(dgcdr['source'].get(user, set()))
            nt = len(dgcdr['target'].get(user, set()))
            rows.append(dict(user_id=user, seed=seed,
                ndcg_lightgcn=left[0], ndcg_dgcdr=right[0], delta_ndcg=right[0]-left[0],
                recall_lightgcn=left[1], recall_dgcdr=right[1], delta_recall=right[1]-left[1],
                n_source_train=ns, n_target_train=nt,
                n_test_positives=len(dgcdr['test'][user]),
                log_activity_ratio=float(np.log((ns+1)/(nt+1)))))
    if not rows:
        raise ValueError('No users to compare')
    path = out / 'per_user.csv'
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def report(spec, out):
    import pandas as pd
    df = pd.read_csv(out / 'per_user.csv', dtype={'user_id': str}, encoding='utf-8')
    if df.duplicated(['user_id', 'seed']).any() or not df.groupby('user_id').seed.apply(
            lambda s: set(s) == set(spec['seeds'])).all():
        raise ValueError('Missing or duplicated seed results')
    g = df.groupby('user_id', sort=True)
    summary = g.mean(numeric_only=True)
    eps = spec['epsilon']
    summary['negative_seeds'] = g.delta_ndcg.apply(lambda x: int((x < -eps).sum()))
    summary['std_delta'] = g.delta_ndcg.std().fillna(0)
    summary['class'] = np.where(summary.delta_ndcg > eps, 'Positive',
                        np.where(summary.delta_ndcg < -eps, 'Negative', 'Neutral'))
    summary['both_zero'] = g.apply(lambda x: bool(((x.ndcg_dgcdr == 0) & (x.ndcg_lightgcn == 0)).all()))
    for domain in ('source', 'target'):
        values = summary['n_%s_train' % domain]
        cuts = np.unique(np.quantile(values, [1/3, 2/3]))
        summary[domain + '_bin'] = np.searchsorted(cuts, values, side='left')
    summary = summary.sort_values(['delta_ndcg'], kind='stable')
    n = len(summary)
    rng = np.random.RandomState(2022)
    boot = []
    values = summary.delta_ndcg.to_numpy()
    for _ in range(spec['bootstrap_samples']):
        v = values[rng.randint(0, n, n)]
        boot.append([v.mean(), (v < -eps).mean(), (v > eps).mean()])
    ci = np.quantile(boot, [.025, .975], axis=0)
    def table(headers, rows):
        return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + ''.join(
            '| ' + ' | '.join(str(v).replace('|', '\\|') for v in row) + ' |\n' for row in rows)
    audits = json.loads((out / 'config.json').read_text(encoding='utf-8'))['seeds']
    policies = {(a['repeatable'], str(a['threshold_dgcdr']), str(a['threshold_lightgcn']))
                for a in audits.values()}
    if len(policies) != 1:
        raise ValueError('Evaluation or relevance policy varies across seeds')
    repeatable, threshold_dgcdr, threshold_lightgcn = policies.pop()
    def audit_counts(field):
        return ', '.join(str(value) for value in sorted({a[field] for a in audits.values()}))
    lines = ['# Transfer: %s → %s\n' % (spec['source'], spec['target']),
             'Analisi di checkpoint già addestrati; nessun training o tuning.\n',
             'Ogni coppia DGCDR/LightGCN usa lo stesso split target ricostruito dal seed. '
             'Gli split possono cambiare fra seed. Checkpoint selezionati in origine con Recall@20 validation. '
             'repeatable=%s; soglie salvate nei checkpoint: DGCDR %s, LightGCN %s.\n' % (
                 repeatable, threshold_dgcdr, threshold_lightgcn),
             'Train target: **%s**. Validation target: **%s**. Test target: **%s**. '
             'Utenti test distinti: **%d**. Seed: %s. Epsilon: %g.\n' % (
                 audit_counts('train'), audit_counts('validation'), audit_counts('test'),
                 n, spec['seeds'], eps),
             '## Risultati globali\n',
             table(['Modello', 'NDCG@20', 'Recall@20'], [[name, '%.6f'%summary['ndcg_'+key].mean(),
                 '%.6f'%summary['recall_'+key].mean()] for name,key in [('LightGCN','lightgcn'),('DGCDR','dgcdr')]]),
             'Delta medio NDCG: **%.6f**, IC bootstrap utenti 95%% [%.6f, %.6f].\n' % (values.mean(),ci[0,0],ci[1,0]),
             table(['Classe','Utenti','Percentuale'], [[c,int((summary['class']==c).sum()),
                   '%.2f%%'%(100*(summary['class']==c).mean())] for c in ['Negative','Neutral','Positive']]),
             'NDCG sempre zero per entrambi: %d utenti.\n' % summary.both_zero.sum(),
             'Quota negative: IC 95%% [%.2f%%, %.2f%%]; positive: [%.2f%%, %.2f%%].\n' % tuple(100*ci[:,1:].T.flatten()),
             'Guadagno medio tra positive: %s; perdita media tra negative: %s.\n' % (
                 ('%.6f' % summary.loc[summary['class']=='Positive','delta_ndcg'].mean()) if (summary['class']=='Positive').any() else 'n/d',
                 ('%.6f' % -summary.loc[summary['class']=='Negative','delta_ndcg'].mean()) if (summary['class']=='Negative').any() else 'n/d'),
             '## Variabilità fra seed\n',
             table(['Seed','NDCG LightGCN','NDCG DGCDR','Delta'], [[seed, *['%.6f'%v for v in group[
                 ['ndcg_lightgcn','ndcg_dgcdr','delta_ndcg']].mean()]] for seed,group in df.groupby('seed')]),
             '## Attività source × target\nN source e N target sono i numeri di interazioni '
             'nel training di ciascun utente. Per ciascun dominio, i conteggi sono divisi '
             'in base ai terzili: fascia 0 = attività bassa, 1 = media, 2 = alta. '
             'Se due soglie coincidono, le fasce vengono accorpate e possono essere meno di tre. '
             'La tabella mostra gli intervalli effettivamente osservati per ogni combinazione.\n']
    cells = []
    for (s,t), group in summary.groupby(['source_bin','target_bin']):
        neg = group[group['class']=='Negative']
        cells.append([s,t,'%d–%d'%(group.n_source_train.min(), group.n_source_train.max()),
                      '%d–%d'%(group.n_target_train.min(), group.n_target_train.max()), len(group),
                      '%.6f'%group.delta_ndcg.mean(),'%.2f%%'%(100*(group['class']=='Negative').mean()),
                      '%.6f'%(-neg.delta_ndcg.mean()) if len(neg) else 'n/d'])
    lines.append(table(['Fascia S','Fascia T','N source','N target','Utenti','Delta','Negative','Perdita negative'],cells))
    lines.extend(['## Sensibilità alla soglia\n',table(['Epsilon','Negative','Neutral','Positive'],[
        [e,int((values < -e).sum()),int((np.abs(values)<=e).sum()),int((values>e).sum())] for e in spec['epsilon_grid']]),
        '## Confronto individuale\n[Tutti gli utenti, ordinati per delta e separati per classe](users.md). '
        '[Risultati completi per utente e seed](per_user.csv).\n',
        '## Controlli e limiti\nPer ogni seed sono stati verificati split target e item candidati identici fra modelli; '
        'nessuna interazione target di validation/test entra nel grafo train. '
        'In assenza dei dataloader originali, gli split sono ricostruiti da checkpoint, dataset e seed: '
        'la coincidenza con il test storico non è dimostrabile dai soli pesi. '
        'Gli intervalli ricampionano gli utenti e sono condizionati ai seed e split osservati; '
        'le differenze DGCDR/LightGCN non isolano causalmente l’effetto del source.\n'])
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    user_lines = ['# Confronto utenti test\n[Report generale](report.md)\n',
                  'Metriche medie sui seed; conteggi medi riferiti al training. [Negative](#negative) · [Positive](#positive) · [Neutral](#neutral)\n']
    for cls in ['Negative','Positive','Neutral']:
        group = summary[summary['class']==cls]
        user_lines.extend(['## '+cls+'\n',table(['Utente','N source','N target','NDCG LGCN','NDCG DGCDR','Delta','Seed negativi','Std delta'],[
            [u,int(r.n_source_train),int(r.n_target_train),'%.6f'%r.ndcg_lightgcn,'%.6f'%r.ndcg_dgcdr,
             '%.6f'%r.delta_ndcg,'%d/%d'%(r.negative_seeds,len(spec['seeds'])),'%.6f'%r.std_delta] for u,r in group.iterrows()]),
            '### Recall — '+cls+'\n', table(['Utente','Recall LGCN','Recall DGCDR','Delta'],[
                [u,'%.6f'%r.recall_lightgcn,'%.6f'%r.recall_dgcdr,'%.6f'%r.delta_recall] for u,r in group.iterrows()])])
    (out / 'users.md').write_text('\n'.join(user_lines), encoding='utf-8')
    for old_plot in ('delta.png', 'activity_delta.png', 'activity_negative.png'):
        (out / old_plot).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/transfer_analysis.yaml'))
    parser.add_argument('--output')
    parser.add_argument('--checkpoint-dir', help='Search this directory recursively for model .pth files (default: saved/)')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='MODEL:SEED=PATH',
                        help='Explicit model file; repeat for each model/seed. Overrides YAML and discovery.')
    parser.add_argument('--source', help='Override source dataset name from YAML')
    parser.add_argument('--target', help='Override target dataset name from YAML')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2022, 2023, 42, 24, 1])
    parser.add_argument('--stage', choices=['run','report'], default='run')
    args = parser.parse_args()
    # RecBole parses sys.argv itself; keep this entry point's arguments out of it.
    sys.argv = [sys.argv[0]]
    spec = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    spec['seeds'] = args.seeds
    if args.source or args.target:
        if not args.output:
            parser.error('--source/--target requires --output to keep domains separate')
        spec['source'] = args.source or spec['source']
        spec['target'] = args.target or spec['target']
    out = Path(args.output or ROOT / spec['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(spec['threads'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.stage == 'report':
        saved = json.loads((out/'config.json').read_text(encoding='utf-8'))
        report(saved['spec'],out)
        return
    if spec.get('use_gpu', True) and torch.cuda.is_available():
        logging.info('Evaluation device: %s (%s)', torch.device('cuda'), torch.cuda.get_device_name())
    elif spec.get('use_gpu', True):
        logging.warning('CUDA unavailable; falling back to CPU for evaluation')
    else:
        logging.info('Evaluation device: CPU')
    paths = checkpoint_paths(spec, args)
    results = {}
    audits = {}
    for seed in spec['seeds']:
        for name in ['LightGCN','DGCDR']:
            results[(name,seed)] = reconstruct(spec, out, name, seed, paths[(name, seed)])
        light, dgcdr = results[('LightGCN', seed)], results[('DGCDR', seed)]
        compare_pair(seed, light, dgcdr)
        audits[str(seed)] = dict(train=len(light['splits']['train']),
                                validation=len(light['splits']['validation']),
                                test=len(light['splits']['test']), test_users=len(light['results']),
                                repeatable=light['repeatable'],
                                threshold_lightgcn=light['threshold'],
                                threshold_dgcdr=dgcdr['threshold'],
                                checkpoints={name: str(paths[(name,seed)]) for name in ('LightGCN','DGCDR')})
    atomic_json(out / 'config.json', dict(spec=spec, seeds=audits))
    write_csv(spec,out,results)
    report(spec,out)
    logging.info('Report ready: %s',out/'report.md')


if __name__ == '__main__':
    main()
