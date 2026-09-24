"""Paired DGCDR / native RecBole LightGCN experiment (RecBole 1.0.1)."""
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
import yaml
from recbole.config import Config
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.model.cross_domain_recommender.dgcdr import DGCDR
from recbole_cdr.trainer import CrossDomainTrainer

ROOT = Path(__file__).resolve().parent
# Optional plotting dependencies installed locally; leave the training environment unchanged.
if (ROOT / '.transfer_plotting').exists():
    sys.path.append(str(ROOT / '.transfer_plotting'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'transfer_results/.matplotlib'))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def pairs(dataset):
    f = dataset.inter_feat
    return set(zip(f[dataset.uid_field].tolist(), f[dataset.iid_field].tolist()))


def histories(dataset):
    result = {}
    for u, i in pairs(dataset):
        result.setdefault(u, set()).add(i)
    return result


def cdr_config(spec, out):
    override = dict(seed=spec['split_seed'], use_gpu=spec['use_gpu'], repeatable=False,
                    train_epochs=['BOTH:%d' % spec['epochs']],
                    stopping_step=spec['patience'], train_batch_size=spec['train_batch_size'],
                    eval_batch_size=spec['eval_batch_size'], metrics=['Recall', 'NDCG'],
                    topk=[spec['k']], valid_metric='Recall@%d' % spec['k'],
                    metric_decimal_place=10, show_progress=False, log_wandb=False,
                    save_dataset=False, save_dataloaders=False, dataset_save_path=None,
                    dataloaders_save_path=None, checkpoint_dir=str(out),
                    rm_dup_inter='first', threshold=None,
                    eval_args={'split': {'RS': [0.6, 0.2, 0.2]}, 'order': 'RO',
                               'group_by': 'user', 'mode': 'full'})
    for domain in ('source', 'target'):
        override[domain + '_domain'] = dict(dataset=spec[domain], data_path=str(ROOT / 'dataset'),
                                           threshold=None, load_col={'inter': ['user_id', 'item_id']},
                                           rm_dup_inter='first', val_interval=None)
    return CDRConfig(model='DGCDR', config_file_list=[str(ROOT / spec['dataset_config']),
                     str(ROOT / 'recbole_cdr/properties/model/DGCDR.yaml')], config_dict=override)


def validate_data(train, valid, test):
    t = train.target_dataset
    tp, vp, ep = (pairs(x) for x in (t, valid.dataset, test.dataset))
    if tp & vp or tp & ep or vp & ep:
        raise ValueError('Target train/validation/test overlap')
    for d in (t, valid.dataset, test.dataset, train.source_dataset):
        if len(pairs(d)) != len(d):
            raise ValueError('Duplicate interactions remain')
    th, vh, eh = (histories(x) for x in (t, valid.dataset, test.dataset))
    for loader, expected, positives in ((valid, th, vh),
                                       (test, {u: th.get(u, set()) | vh.get(u, set())
                                               for u in set(th) | set(vh)}, eh)):
        for inter, history, pu, pi in loader:
            users = inter[t.uid_field].tolist()
            actual_history = {u: set() for u in users}
            if history is not None:
                for row, item in zip(history[0].tolist(), history[1].tolist()):
                    actual_history[users[row]].add(item)
            actual_pos = {u: set() for u in users}
            for row, item in zip(pu.tolist(), pi.tolist()):
                actual_pos[users[row]].add(item)
            for u in users:
                if actual_history[u] != expected.get(u, set()) or actual_pos[u] != positives[u]:
                    raise ValueError('Evaluation masks or positives differ from frozen split')
    if not set(eh).issubset(th):
        raise ValueError('Test users without target training history')
    return {'target_train': len(tp), 'target_valid': len(vp), 'target_test': len(ep),
            'source_train': len(train.source_dataset), 'test_users': len(eh),
            'target_items': t.item_num - 1,
            'users_without_test': len(set(th) - set(eh)),
            'test_items_without_train': len({i for _, i in ep} - {i for _, i in tp})}


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


class SharedEvaluation:
    """Use exactly the same ranking rules during validation and final test."""
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if load_best_model:
            state = torch.load(model_file or self.saved_model_file, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state['state_dict'])
        metrics = individual_metrics(self.model, eval_data, self.device, self.config['topk'][0])
        values = np.array(list(metrics.values()))
        k = self.config['topk'][0]
        return {'ndcg@%d' % k: float(values[:, 0].mean()), 'recall@%d' % k: float(values[:, 1].mean())}


class PairedCDRTrainer(SharedEvaluation, CrossDomainTrainer):
    pass


class PairedLightTrainer(SharedEvaluation, Trainer):
    pass


def prepare(spec, out):
    inputs = {}
    for domain in ('source', 'target'):
        path = ROOT / 'dataset' / spec[domain] / (spec[domain] + '.inter')
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (ROOT / spec['dataset_config'], ROOT / 'recbole_cdr/properties/model/DGCDR.yaml',
                 ROOT / 'recbole_cdr/properties/overall.yaml', Path(__file__),
                 ROOT / 'recbole_cdr/model/cross_domain_recommender/dgcdr.py',
                 ROOT / 'recbole_cdr/data/dataset.py', ROOT / 'recbole_cdr/data/dataloader.py'):
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    import recbole, scipy, pandas
    environment = dict(python=sys.version, torch=torch.__version__, recbole=recbole.__version__,
                       numpy=np.__version__, scipy=scipy.__version__, pandas=pandas.__version__)
    signature = dict(spec=spec, input_hashes=inputs, environment=environment)
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    manifest = out / 'config.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Output belongs to another configuration/code/input: choose a new --output')
    config = cdr_config(spec, out)
    frozen = out / 'split.pkl'
    if frozen.exists():
        with frozen.open('rb') as f:
            payload = pickle.load(f)
        if payload['fingerprint'] != fingerprint:
            raise ValueError('Frozen data belong to a different experiment')
        train, valid, test = payload['loaders']
    else:
        init_seed(spec['split_seed'], True)
        dataset = create_dataset(config)
        train, valid, test = data_preparation(config, dataset)
        audit = validate_data(train, valid, test)
        with frozen.with_suffix('.tmp').open('wb') as f:
            pickle.dump(dict(fingerprint=fingerprint, loaders=(train, valid, test)), f, protocol=4)
        frozen.with_suffix('.tmp').replace(frozen)
    audit = validate_data(train, valid, test)
    signature.update(fingerprint=fingerprint, audit=audit,
                     dgcdr_resolved=config.final_config_dict)
    atomic_json(manifest, signature)
    logging.info('Frozen split verified: %s', audit)
    return fingerprint


def run_model(spec, out, fingerprint, name, seed):
    with (out / 'split.pkl').open('rb') as f:
        payload = pickle.load(f)
        train, valid, test = payload['loaders']
    target = train.target_dataset
    config = cdr_config(spec, out)
    run_dir = out / ('%s_%s' % (name, seed))
    run_dir.mkdir(exist_ok=True)
    if name == 'LightGCN':
        config = Config(model='LightGCN', dataset=spec['target'], config_dict=dict(
            **spec['lightgcn'], USER_ID_FIELD=target.uid_field, ITEM_ID_FIELD=target.iid_field,
            NEG_PREFIX='neg_', use_gpu=spec['use_gpu'], seed=seed, epochs=spec['epochs'],
            train_batch_size=spec['train_batch_size'], eval_batch_size=spec['eval_batch_size'],
            stopping_step=spec['patience'], metrics=['Recall', 'NDCG'], topk=[spec['k']],
            valid_metric='Recall@%d' % spec['k'], metric_decimal_place=10,
            checkpoint_dir=str(run_dir), log_wandb=False, show_progress=False,
            neg_sampling={'uniform': 1}, require_pow=False))
    config['seed'] = seed
    config['checkpoint_dir'] = str(run_dir)
    init_seed(seed, True)
    model = (DGCDR(config, train.dataset) if name == 'DGCDR' else LightGCN(config, target)).to(config['device'])
    matrix = model.target_interaction_matrix if name == 'DGCDR' else model.interaction_matrix
    if set(zip(matrix.row.tolist(), matrix.col.tolist())) != pairs(target):
        raise ValueError('Model graph differs from target train')
    checkpoint = run_dir / 'best.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
    if state is not None and state.get('transfer_fingerprint') == fingerprint and state.get('transfer_complete'):
        logging.info('Reusing completed %s seed %d', name, seed)
        model.load_state_dict(state['state_dict'])
    else:
        log_handler = logging.FileHandler(run_dir / 'training.log', mode='w')
        logging.getLogger().addHandler(log_handler)
        trainer = (PairedCDRTrainer if name == 'DGCDR' else PairedLightTrainer)(config, model)
        trainer.saved_model_file = str(checkpoint)
        logging.info('START %s seed=%d', name, seed)
        try:
            trainer.fit(train if name == 'DGCDR' else train.target_dataloader, valid, show_progress=False)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['state_dict'])
            state.update(transfer_fingerprint=fingerprint, transfer_complete=True)
            tmp = checkpoint.with_suffix('.tmp')
            torch.save(state, tmp)
            tmp.replace(checkpoint)
        finally:
            trainer.tensorboard.close()
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()
    result = individual_metrics(model, test, config['device'], spec['k'])
    # Compare the native evaluator on this checkpoint (tie policies may differ).
    native = Trainer(config, model)
    native_result = native.evaluate(test, load_best_model=False, show_progress=False)
    native.tensorboard.close()
    avg = np.mean(list(result.values()), axis=0)
    discrepancy = max(abs(native_result['ndcg@%d' % spec['k']] - avg[0]),
                      abs(native_result['recall@%d' % spec['k']] - avg[1]))
    if discrepancy > 1e-7:
        raise ValueError('Native/per-user metric discrepancy %.12g; inspect cutoff ties' % discrepancy)
    logging.info('DONE %s seed=%d best_epoch=%d test NDCG=%.8f Recall=%.8f',
                 name, seed, state['epoch'] + 1, avg[0], avg[1])
    return result


def write_csv(spec, out, results):
    with (out / 'split.pkl').open('rb') as f:
        payload = pickle.load(f)
        train, valid, test = payload['loaders']
    target = train.target_dataset
    sh, th, eh = histories(train.source_dataset), histories(target), histories(test.dataset)
    expected = set(eh)
    rows = []
    for seed in spec['seeds']:
        left, right = results[('LightGCN', seed)], results[('DGCDR', seed)]
        if set(left) != expected or set(right) != expected:
            raise ValueError('Incomplete user comparison')
        for u in sorted(expected):
            rows.append(dict(user_id=str(target.id2token(target.uid_field, u)), seed=seed,
                ndcg_lightgcn=left[u][0], ndcg_dgcdr=right[u][0], delta_ndcg=right[u][0]-left[u][0],
                recall_lightgcn=left[u][1], recall_dgcdr=right[u][1], delta_recall=right[u][1]-left[u][1],
                n_source_train=len(sh.get(u, set())), n_target_train=len(th[u]), n_test_positives=len(eh[u]),
                log_activity_ratio=float(np.log((len(sh.get(u, set()))+1)/(len(th[u])+1)))))
    path = out / 'per_user.csv'
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def report(spec, out):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.read_csv(out / 'per_user.csv', dtype={'user_id': str})
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
    audit = json.loads((out / 'config.json').read_text())['audit']
    lines = ['# Transfer: CDs → Instruments\n',
             '**PROVA BREVE — non risultati conclusivi.**\n' if spec.get('smoke') else 'Esperimento con configurazioni fisse, senza tuning.\n',
             'Rilevanza = presenza dell’interazione; nessun uso delle stelle. Split 60/20/20 condiviso; '
             'checkpoint scelto con Recall@20 validation. `repeatable=False` per escludere gli item già osservati '
             '(override della configurazione originale).\n',
             'Utenti test: **%d**. Seed: %s. Epsilon: %g.\n' % (n, spec['seeds'], eps),
             'Split effettivo: %d train, %d validation, %d test. Source: %d interazioni. '
             '%d item test distinti non hanno archi target train (rimangono candidati per entrambi).\n' % (
                 audit['target_train'],audit['target_valid'],audit['target_test'],audit['source_train'],audit['test_items_without_train']),
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
             '## Attività source × target\nLe fasce 0/1/2 derivano dai terzili dei conteggi train; '
             'soglie coincidenti vengono unite. Gli intervalli osservati sono indicati in tabella.\n']
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
        '![Distribuzione delle differenze](delta.png)\n',
        '## Controlli e limiti\nSplit disgiunti; grafi target train-only; maschere train/validation verificate; '
        'ID condivisi; cinque coppie per utente nelle run complete; metriche individuali confrontate con RecBole. '
        'Gli intervalli ricampionano utenti mantenendo insieme i seed e sono condizionati allo split e ai modelli osservati. '
        'Nessuna analisi causale del source; iperparametri non ottimizzati.\n'])
    (out / 'report.md').write_text('\n'.join(lines))
    user_lines = ['# Confronto utenti test\n[Report generale](report.md)\n',
                  'Metriche medie sui seed; conteggi riferiti al training. [Negative](#negative) · [Neutral](#neutral) · [Positive](#positive)\n']
    for cls in ['Negative','Neutral','Positive']:
        group = summary[summary['class']==cls]
        user_lines.extend(['## '+cls+'\n',table(['Utente','N source','N target','NDCG LGCN','NDCG DGCDR','Delta','Seed negativi','Std delta'],[
            [u,int(r.n_source_train),int(r.n_target_train),'%.6f'%r.ndcg_lightgcn,'%.6f'%r.ndcg_dgcdr,
             '%.6f'%r.delta_ndcg,'%d/%d'%(r.negative_seeds,len(spec['seeds'])),'%.6f'%r.std_delta] for u,r in group.iterrows()]),
            '### Recall — '+cls+'\n', table(['Utente','Recall LGCN','Recall DGCDR','Delta'],[
                [u,'%.6f'%r.recall_lightgcn,'%.6f'%r.recall_dgcdr,'%.6f'%r.delta_recall] for u,r in group.iterrows()])])
    (out / 'users.md').write_text('\n'.join(user_lines))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(values, bins=50, color='#4878a8')
    ax.axvline(0,color='black',linewidth=1)
    ax.set(xlabel='Delta medio NDCG@20 (DGCDR − LightGCN)',ylabel='Utenti')
    fig.tight_layout()
    fig.savefig(out / 'delta.png',dpi=160)
    plt.close(fig)
    for field, filename, label in [('delta_ndcg','activity_delta.png','Delta medio NDCG@20'),
                                    ('negative_fraction','activity_negative.png','Quota negative')]:
        temp = summary.assign(negative_fraction=(summary['class']=='Negative').astype(float))
        grid = temp.pivot_table(index='source_bin',columns='target_bin',values=field,aggfunc='mean')
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(grid.to_numpy(), cmap='coolwarm' if field=='delta_ndcg' else 'Reds')
        for row in range(len(grid)):
            for col in range(len(grid.columns)):
                ax.text(col,row,'%.3f'%grid.iloc[row,col],ha='center',va='center')
        ax.set(xticks=range(len(grid.columns)),xticklabels=grid.columns,yticks=range(len(grid)),
               yticklabels=grid.index,xlabel='Fascia attività target',ylabel='Fascia attività source',title=label)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / filename,dpi=160)
        plt.close(fig)
    with (out / 'report.md').open('a') as f:
        f.write('\n![Delta per attività](activity_delta.png)\n\n![Negative per attività](activity_negative.png)\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/transfer_analysis.yaml'))
    parser.add_argument('--output')
    parser.add_argument('--stage', choices=['prepare','run','report'], default='run')
    parser.add_argument('--smoke', action='store_true', help='Two epochs, one seed; separate output')
    args = parser.parse_args()
    # RecBole parses sys.argv itself; keep this entry point's arguments out of it.
    sys.argv = [sys.argv[0]]
    spec = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        spec.update(epochs=2, seeds=[2022], smoke=True, bootstrap_samples=100)
        spec['output'] += '_smoke'
    out = Path(args.output or ROOT / spec['output']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(spec['threads'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.stage == 'report':
        saved = json.loads((out/'config.json').read_text())
        report(saved['spec'],out)
        return
    fingerprint = prepare(spec,out)
    if args.stage == 'prepare':
        return
    results = {}
    for seed in spec['seeds']:
        for name in ['LightGCN','DGCDR']:
            results[(name,seed)] = run_model(spec,out,fingerprint,name,seed)
    write_csv(spec,out,results)
    report(spec,out)
    logging.info('Report ready: %s',out/'report.md')


if __name__ == '__main__':
    main()
