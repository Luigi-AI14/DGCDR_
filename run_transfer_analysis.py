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
from analysis_common import (
    pairs, histories, token_map, token_pairs, token_histories, checkpoint_config,
    reconstruct_data, load_predictor, validate_transfer_preparation,
)
from transfer_statistics_test import global_paired_ttest, validate_ndcg_values

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
    train, valid, test, target, source, model_dataset = reconstruct_data(config, model, seed)
    datasets = {'train': target, 'validation': valid.dataset, 'test': test.dataset}
    splits = {name: token_pairs(data) for name, data in datasets.items()}
    if splits['train'] & splits['validation'] or splits['train'] & splits['test'] or splits['validation'] & splits['test']:
        raise ValueError(f'{model}/{seed}: target split overlap')
    candidates = set(token_map(target, target.iid_field).values()) - {'[PAD]'}
    predictor = load_predictor(state, config, model, model_dataset, reseed=seed)
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


def activity_distribution(summary, out):
    """Plot both domains with common count intervals and retain exact frequencies."""
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    bounds = [-np.inf, 5, 10, 20, np.inf]
    labels = ['0 ≤ N ≤ 5', '5 < N ≤ 10', '10 < N ≤ 20', 'N > 20']
    n = len(summary)
    grouped = {}
    detailed = []
    for domain in ('source', 'target'):
        activity = summary['n_%s_train' % domain]
        if not np.isfinite(activity).all() or (activity < 0).any():
            raise ValueError('Invalid training activity for ' + domain)
        grouped[domain] = (pd.cut(activity, bounds, labels=labels, right=True)
                           .value_counts(sort=False).reindex(labels, fill_value=0))
        counts = activity.value_counts().sort_index()
        detailed.extend(dict(domain=domain, n_train_mean=value, users=int(count),
                             percent=100 * count / n) for value, count in counts.items())
    pd.DataFrame(detailed).to_csv(out / 'activity_distribution.csv', index=False)

    fig = Figure(figsize=(11, 4.8), layout='constrained', facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    positions = np.arange(len(labels))
    maximum = max(int(counts.max()) for counts in grouped.values())
    for domain, offset, color in [('source', -.19, '#2563a6'), ('target', .19, '#df8031')]:
        counts = grouped[domain].to_numpy()
        bars = ax.barh(positions + offset, counts, height=.34,
                       label=domain.capitalize(), color=color)
        for bar, count in zip(bars, counts):
            ax.text(count + maximum * .012, bar.get_y() + bar.get_height() / 2,
                    ('%.2f%%' % (100 * count / n)).replace('.', ','),
                    va='center', fontsize=9, color='#334155')
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'.replace(',', '.')))
    ax.set_xlabel('Numero di utenti')
    ax.set_ylabel('Interazioni di training (N)')
    ax.set_title('Conteggi medi sui seed · %s utenti per dominio\n'
                 'Intervalli uguali per source e target; etichette = percentuale di utenti'
                 % f'{n:,}'.replace(',', '.'), fontsize=10, loc='left', pad=16)
    fig.suptitle('Distribuzione degli utenti per attività', fontsize=17, fontweight='bold')
    ax.legend(loc='lower right', frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e2e8f0')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0)
    fig.savefig(out / 'activity_distribution.png', dpi=180)
    return grouped


def activity_bin_labels(values, cuts):
    """Describe global bin boundaries, with inclusive bounds for integer counts."""
    integer_counts = np.equal(values, np.floor(values)).all()
    limits = [str(int(np.floor(cut))) if integer_counts else
              np.format_float_positional(float(cut), trim='-') for cut in cuts]
    labels = ['N ≤ ' + limits[0]]
    for lower, upper in zip(limits[:-1], limits[1:]):
        labels.append(f'{int(lower) + 1} ≤ N ≤ {upper}' if integer_counts else
                      f'{lower} < N ≤ {upper}')
    labels.append(f'N ≥ {int(limits[-1]) + 1}' if integer_counts else
                  'N > ' + limits[-1])
    return labels


def transfer_classes(lightgcn, dgcdr):
    """Classify NDCG changes using a fixed inclusive 1% relative threshold."""
    delta = dgcdr - lightgcn
    return np.select(
        [(lightgcn == 0) & (dgcdr == 0),
         delta >= 0.01 * lightgcn, delta <= -0.01 * lightgcn],
        ['NDCG@20 nullo', 'Positive', 'Negative'], default='Neutral')


def report(spec, out):
    import pandas as pd
    df = pd.read_csv(out / 'per_user.csv', dtype={'user_id': str}, encoding='utf-8')
    if df.duplicated(['user_id', 'seed']).any() or not df.groupby('user_id').seed.apply(
            lambda s: set(s) == set(spec['seeds'])).all():
        raise ValueError('Missing or duplicated seed results')
    df['ndcg_dgcdr'] = validate_ndcg_values(df['ndcg_dgcdr'], 'DGCDR')
    df['ndcg_lightgcn'] = validate_ndcg_values(df['ndcg_lightgcn'], 'LightGCN')
    g = df.groupby('user_id', sort=True)
    summary = g.mean(numeric_only=True)
    seed_classes = pd.Series(transfer_classes(df.ndcg_lightgcn, df.ndcg_dgcdr), index=df.index)
    summary['negative_seeds'] = seed_classes.eq('Negative').groupby(df.user_id).sum()
    summary['std_delta'] = g.delta_ndcg.std().fillna(0)
    summary['class'] = transfer_classes(summary.ndcg_lightgcn, summary.ndcg_dgcdr)
    bin_labels = {}
    for domain in ('source', 'target'):
        values = summary['n_%s_train' % domain]
        cuts = np.unique(np.quantile(values, [1/3, 2/3]))
        summary[domain + '_bin'] = np.searchsorted(cuts, values, side='left')
        bin_labels[domain] = activity_bin_labels(values, cuts)
    summary = summary.sort_values(['delta_ndcg'], kind='stable')
    n = len(summary)
    paired_test = global_paired_ttest(summary.ndcg_dgcdr.to_numpy(),
                                      summary.ndcg_lightgcn.to_numpy())
    classification_note = (
        'La classe di ciascun utente è definita sui suoi NDCG@20 medi sui seed.\n\n'
        '- **Negative:** peggioramento di DGCDR di almeno l’1% rispetto a LightGCN.\n'
        '- **Positive:** miglioramento di DGCDR di almeno l’1% rispetto a LightGCN; '
        'include il caso LightGCN = 0 e DGCDR > 0, per il quale la percentuale individuale non è definita.\n'
        '- **Neutral:** variazione relativa strettamente compresa tra −1% e +1%, con LightGCN > 0.\n'
        '- **NDCG@20 nullo:** NDCG@20 pari a zero per entrambi i modelli.\n')
    def table(headers, rows):
        return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + ''.join(
            '| ' + ' | '.join(str(v).replace('|', '\\|') for v in row) + ' |\n' for row in rows)
    test_rows = [['Utenti confrontati', paired_test.n],
                 ['Delta medio (punti NDCG)',
                  '%.6f' % paired_test.mean_delta if paired_test.mean_delta is not None else 'n/d']]
    if paired_test.reason:
        test_rows.extend([['IC 95% del Delta medio', 'n/d'],
                          ['Statistica t', 'n/d'],
                          ['Gradi di libertà', 'n/d'],
                          ['p-value bilaterale', 'n/d']])
        test_conclusion = 'Test non calcolabile: %s.' % paired_test.reason
    else:
        pvalue_text = ('< 1e-300' if paired_test.pvalue == 0
                       else '%.2e' % paired_test.pvalue)
        test_rows.extend([['IC 95% del Delta medio', '[%.6f, %.6f]' % (
                              paired_test.ci_low, paired_test.ci_high)],
                          ['Statistica t', '%.4f' % paired_test.statistic],
                          ['Gradi di libertà', paired_test.degrees_of_freedom],
                          ['p-value bilaterale', pvalue_text]])
        if paired_test.pvalue < 0.05:
            winner = 'DGCDR' if paired_test.mean_delta > 0 else 'LightGCN'
            test_conclusion = 'Differenza statisticamente significativa a favore di %s (α = 0,05).' % winner
        else:
            test_conclusion = ('Il test non rileva una differenza statisticamente '
                               'significativa al livello del 5%.')
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
             'Utenti test distinti: **%d**. Seed: %s.\n' % (
                 audit_counts('train'), audit_counts('validation'), audit_counts('test'),
                 n, spec['seeds']),
             '## Risultati globali\n',
             table(['Modello', 'NDCG@20', 'Recall@20'], [[name, '%.6f'%summary['ndcg_'+key].mean(),
                 '%.6f'%summary['recall_'+key].mean()] for name,key in [('LightGCN','lightgcn'),('DGCDR','dgcdr')]]),
             '**Definizione di Delta per utente:**\n\n'
             '$$\n'
             r'\Delta_u = \overline{\mathrm{NDCG@20}}_{u,\mathrm{DGCDR}}'
             r' - \overline{\mathrm{NDCG@20}}_{u,\mathrm{LightGCN}}'
             '\n$$\n\n'
             'Per ciascun utente u, Delta è la differenza tra l’NDCG@20 di DGCDR e quello '
             'di LightGCN. La barra indica la media sui seed per lo stesso utente. '
             'Il risultato è espresso in punti NDCG: un valore positivo indica un miglioramento, '
             'uno negativo un peggioramento e zero indica valori medi uguali. '
             'Per esempio, con DGCDR = 0,22 e LightGCN = 0,20, Delta vale 0,02.\n',
             classification_note,
             table(['Classe','Utenti','Percentuale'], [[c,int((summary['class']==c).sum()),
                   '%.2f%%'%(100*(summary['class']==c).mean())] for c in ['Negative','Positive','Neutral','NDCG@20 nullo']]),
             '### T-test appaiato globale su NDCG@20\n'
             'Confronto bilaterale sugli NDCG@20 medi sui seed degli stessi utenti, '
             'inclusi quelli con NDCG nullo. Livello di significatività: 5%.\n',
             table(['Misura', 'Valore'], test_rows),
             test_conclusion + ' Il confronto riguarda i checkpoint e gli split osservati '
             'e assume indipendenza tra utenti; la significatività statistica va valutata '
             'insieme all’entità del miglioramento.\n',
             '## Variabilità fra seed\n',
             table(['Seed','NDCG LightGCN','NDCG DGCDR','Delta'], [[seed, *['%.6f'%v for v in group[
                 ['ndcg_lightgcn','ndcg_dgcdr','delta_ndcg']].mean()]] for seed,group in df.groupby('seed')])]
    grouped_activity = activity_distribution(summary, out)
    lines.extend(['## Distribuzione delle interazioni per dominio\n',
                  'Il grafico e le tabelle considerano gli stessi utenti test inclusi nel confronto. '
                  'Per ogni utente si usa il numero di interazioni distinte nel training, '
                  'mediato sui seed, come per la suddivisione in fasce. Ogni utente è contato '
                  'una sola volta in ciascun dominio. Gli intervalli sono fissi e uguali per '
                  'source e target; includono le eventuali medie decimali secondo i limiti indicati. '
                  'Questi intervalli descrittivi sono distinti dalle fasce a terzili della sezione successiva. '
                  'Le barre mostrano il numero di utenti per intervallo; le percentuali sono riferite '
                  'al totale degli utenti analizzati.\n',
                  '![Distribuzione utenti source e target per intervalli comuni di interazioni]'
                  '(activity_distribution.png)\n',
                  '[Distribuzione completa per singolo valore (CSV)](activity_distribution.csv).\n'])
    for domain in ('source', 'target'):
        counts = grouped_activity[domain]
        distribution = [[interval, int(count),
                         '%.2f%%' % (100 * count / n)]
                        for interval, count in counts.items()]
        lines.extend(['### Dominio %s — %s\n' % (domain, spec[domain]),
                      table(['Interazioni train (media sui seed)', 'Utenti', 'Percentuale'],
                            distribution)])
    lines.append('## Attività source × target\n'
                 'Le fasce sono definite sui numeri di interazioni nel training di ciascun utente, '
                 'mediati sui seed. Le soglie sono globali per ciascun dominio e restano uguali '
                 'in tutte le combinazioni; se due soglie coincidono, le fasce vengono accorpate.\n')
    lines.append('**Legenda delle fasce:**\n')
    for domain, prefix in (('source', 'S'), ('target', 'T')):
        lines.append(f'**{domain.capitalize()} ({prefix}):**\n')
        lines.append(table(['Fascia', 'Numero di interazioni nel training N'], [
            [f'{prefix} = {index}', label]
            for index, label in enumerate(bin_labels[domain])]))
    lines.append('**Legenda delle colonne:**\n\n'
                 '- **Fascia S:** livello di attività nel dominio source.\n'
                 '- **Fascia T:** livello di attività nel dominio target.\n'
                 '- **Utenti:** numero di utenti nella combinazione delle due fasce.\n'
                 '- **Negative:** percentuale di utenti della combinazione con peggioramento '
                 'di almeno l’1% rispetto a LightGCN.\n'
                 '- **Positive:** percentuale di utenti della combinazione con miglioramento '
                 'di almeno l’1% rispetto a LightGCN, inclusi i miglioramenti da LightGCN = 0.\n'
                 '- **P̄ (%):** perdita percentuale dell’NDCG@20 medio dei soli utenti Negative '
                 'rispetto alla loro media LightGCN.\n\n'
                 '  $$\n'
                 r'  \bar{P}(\%) = 100 \times '
                 r'\frac{\text{NDCG medio LightGCN}_{\text{Negative}}-'
                 r'\text{NDCG medio DGCDR}_{\text{Negative}}}'
                 r'{\text{NDCG medio LightGCN}_{\text{Negative}}}'
                 '\n  $$\n\n'
                 '- **Ḡ (%):** guadagno percentuale dell’NDCG@20 medio dei soli utenti Positive '
                 'rispetto alla loro media LightGCN.\n\n'
                 '  $$\n'
                 r'  \bar{G}(\%) = 100 \times '
                 r'\frac{\text{NDCG medio DGCDR}_{\text{Positive}}-'
                 r'\text{NDCG medio LightGCN}_{\text{Positive}}}'
                 r'{\text{NDCG medio LightGCN}_{\text{Positive}}}'
                 '\n  $$\n\n'
                 '- **Delta (%):** variazione percentuale relativa dell’NDCG@20 medio '
                 'di DGCDR rispetto a quello medio di LightGCN nella combinazione. '
                 'Le medie includono tutti gli utenti della combinazione.\n\n'
                 '  $$\n'
                 r'  \text{Delta (\%)} = 100 \times '
                 r'\frac{\text{NDCG medio DGCDR}-\text{NDCG medio LightGCN}}'
                 r'{\text{NDCG medio LightGCN}}'
                 '\n  $$\n\n'
                 'La quota restante dopo Negative e Positive comprende Neutral e NDCG@20 nullo. '
                 'Tutte le classi sono incluse in Utenti e nei denominatori delle percentuali di classe.\n\n'
                 '**n/d** indica che la classe corrispondente è vuota oppure che la media '
                 'LightGCN usata come denominatore è zero.\n')
    cells = []
    for (s,t), group in summary.groupby(['source_bin','target_bin']):
        neg = group[group['class']=='Negative']
        pos = group[group['class']=='Positive']
        mean_lightgcn = group.ndcg_lightgcn.mean()
        neg_lightgcn = neg.ndcg_lightgcn.mean()
        pos_lightgcn = pos.ndcg_lightgcn.mean()
        loss_pct = ('%.2f%%' % (-100 * neg.delta_ndcg.mean() / neg_lightgcn)
                    if len(neg) and neg_lightgcn else 'n/d')
        gain_pct = ('%.2f%%' % (100 * pos.delta_ndcg.mean() / pos_lightgcn)
                    if len(pos) and pos_lightgcn else 'n/d')
        relative_delta = ('%.2f%%' % (100 * group.delta_ndcg.mean() / mean_lightgcn)
                          if mean_lightgcn else 'n/d')
        cells.append([s,t,len(group),
                      '%.2f%%'%(100*(group['class']=='Negative').mean()),
                      '%.2f%%'%(100*(group['class']=='Positive').mean()),
                      loss_pct,gain_pct,
                      relative_delta])
    lines.append(table(['Fascia S','Fascia T','Utenti','Negative',
                        'Positive','P̄ (%)','Ḡ (%)','Delta (%)'],cells))
    lines.extend([
        '## Confronto individuale\n[Tutti gli utenti, ordinati per delta e separati per classe](users.md). '
        '[Risultati completi per utente e seed](per_user.csv).\n'])
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    user_lines = ['# Confronto utenti test\n[Report generale](report.md)\n',
                  'Metriche medie sui seed; conteggi medi riferiti al training. '
                  '[Negative](#negative) · [Positive](#positive) · [Neutral](#neutral) · [NDCG@20 nullo](#ndcg20-nullo)\n',
                  classification_note,
                  'Seed negativi conta i seed in cui il peggioramento relativo è almeno dell’1%.\n']
    for cls in ['Negative','Positive','Neutral','NDCG@20 nullo']:
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
        validate_transfer_preparation(spec, out, seed, paths[('DGCDR', seed)], dgcdr)
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
