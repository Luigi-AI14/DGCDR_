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
from dataclasses import asdict
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
from transfer_statistics_test import (
    global_paired_ttest, holm_adjust, validate_ndcg_values, welch_transfer_test,
)

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


def transfer_activity_row(source_bin, target_bin, group):
    """Summarize one population from unrounded per-user seed means."""
    neg = group[group['class'] == 'Negative']
    pos = group[group['class'] == 'Positive']
    mean_lightgcn = group.ndcg_lightgcn.mean()
    neg_lightgcn = neg.ndcg_lightgcn.mean()
    pos_lightgcn = pos.ndcg_lightgcn.mean()
    loss_pct = ('%.2f%%' % (-100 * neg.delta_ndcg.mean() / neg_lightgcn)
                if len(neg) and neg_lightgcn else 'n/d')
    gain_pct = ('%.2f%%' % (100 * pos.delta_ndcg.mean() / pos_lightgcn)
                if len(pos) and pos_lightgcn else 'n/d')
    relative_delta = ('%.2f%%' % (100 * group.delta_ndcg.mean() / mean_lightgcn)
                      if mean_lightgcn else 'n/d')
    return [source_bin, target_bin, len(group),
            '%.2f%%' % (100 * (group['class'] == 'Negative').mean()),
            '%.2f%%' % (100 * (group['class'] == 'Positive').mean()),
            loss_pct, gain_pct, relative_delta]


def activity_band_statistics(summary, bin_labels):
    """Compare all planned source/target cells, then Holm-adjust as one family."""
    rows = []
    for s, source_range in enumerate(bin_labels['source']):
        for t, target_range in enumerate(bin_labels['target']):
            group = summary[(summary.source_bin == s) & (summary.target_bin == t)]
            result = global_paired_ttest(group.ndcg_dgcdr.to_numpy(),
                                         group.ndcg_lightgcn.to_numpy())
            rows.append(dict(
                source_bin=s, target_bin=t,
                source_range=source_range, target_range=target_range,
                ndcg_lightgcn=float(group.ndcg_lightgcn.mean()) if result.n else None,
                ndcg_dgcdr=float(group.ndcg_dgcdr.mean()) if result.n else None,
                **asdict(result)))
    # Keep empty/degenerate cells in the planned family, without inventing
    # a reported p-value or making the other comparisons less conservative.
    adjusted = holm_adjust([r['pvalue'] if r['pvalue'] is not None else 1.0
                            for r in rows])
    for row, corrected in zip(rows, adjusted):
        row['pvalue_holm'] = float(corrected) if row['pvalue'] is not None else None
        if row['reason']:
            row['outcome'] = 'Non calcolabile'
        elif row['pvalue_holm'] >= 0.05:
            row['outcome'] = 'Evidenza insufficiente'
        elif row['mean_delta'] > 0:
            row['outcome'] = 'Miglioramento significativo'
        else:
            row['outcome'] = 'Peggioramento significativo'
    return rows


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + ''.join(
        '| ' + ' | '.join(str(v).replace('|', '\\|') for v in row) + ' |\n' for row in rows)


def activity_band_heatmap(band_stats, out):
    """Plot effect size by source (rows) and target (columns), marking Holm significance."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    source_labels = {r['source_bin']: r['source_range'] for r in band_stats}
    target_labels = {r['target_bin']: r['target_range'] for r in band_stats}
    n_source, n_target = len(source_labels), len(target_labels)
    values = [abs(r['mean_delta']) for r in band_stats if r['mean_delta'] is not None]
    limit = max(values, default=0) or 0.000001
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = LinearSegmentedColormap.from_list('transfer_delta', ['#b96325', '#fbfaf7', '#08618b'])
    fig = Figure(figsize=(10.8, 6.3), facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.03, 0.32, 0.94, 0.61])
    ax.set_xlim(-1.05, n_target)
    ax.set_ylim(n_source + 1, 0)
    ax.set_axis_off()
    ink, border, header = '#17324d', '#d7e1e7', '#e8eff3'

    def cell(x, y, width, color):
        ax.add_patch(Rectangle((x, y), width, 1, facecolor=color,
                               edgecolor=border, linewidth=0.8))

    def label(index, labels, prefix):
        level = ['Bassa', 'Media', 'Alta'][index] if len(labels) == 3 else 'Fascia ' + str(index)
        return '%s (%s%d)\n%s' % (level, prefix, index, labels[index])

    cell(-1.05, 0, 1.05, header)
    ax.text(-0.525, 0.5, 'Source → Target\n(righe)     (colonne)',
            ha='center', va='center', fontsize=11, fontweight='bold', color=ink)
    for t in sorted(target_labels):
        cell(t, 0, 1, header)
        ax.text(t + 0.5, 0.5, label(t, target_labels, 'T'),
                ha='center', va='center', fontsize=12, fontweight='bold', color=ink)
    for s in sorted(source_labels):
        cell(-1.05, s + 1, 1.05, '#f8fafb')
        ax.text(-0.95, s + 1.5, label(s, source_labels, 'S'),
                ha='left', va='center', fontsize=12, color=ink)
    for row in band_stats:
        s, t, delta = row['source_bin'], row['target_bin'], row['mean_delta']
        color = cmap(norm(delta)) if delta is not None else (0.91, 0.93, 0.94, 1)
        cell(t, s + 1, 1, color)
        rgb = np.asarray(color[:3])
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luminance = np.dot(linear, [0.2126, 0.7152, 0.0722])
        text_color = '#ffffff' if luminance < 0.179 else '#000000'
        marker = ('†' if row['pvalue_holm'] is None else
                  '*' if row['pvalue_holm'] < 0.05 else '')
        value = ('%+.6f' % delta).replace('.', ',').replace('-', '−') if delta is not None else 'n/d'
        ax.text(t + 0.5, s + 1.4, value + marker, ha='center', va='center',
                fontsize=16, fontweight='bold', color=text_color)
        ax.text(t + 0.5, s + 1.72, '(%s utenti)' % f"{row['n']:,}".replace(',', '.'),
                ha='center', va='center', fontsize=11, color=text_color)
    fig.text(0.03, 0.965, 'Differenza media NDCG@20 · DGCDR − LightGCN',
             fontsize=16, fontweight='bold', color=ink)
    bar_ax = fig.add_axes([0.22, 0.225, 0.72, 0.035])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax,
                            orientation='horizontal', ticks=np.linspace(-limit, limit, 5))
    colorbar.outline.set_edgecolor(border)
    colorbar.ax.tick_params(labelsize=10, length=0, pad=5)
    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _: ('%.6f' % value).replace('.', ',').replace('-', '−')))
    fig.text(0.22, 0.14, 'LightGCN migliore', fontsize=12, color=ink)
    fig.text(0.94, 0.14, 'DGCDR migliore', ha='right', fontsize=12, color=ink)
    legend = '* p Holm < 0,05; il colore rappresenta il Delta medio.'
    if any(r['pvalue_holm'] is None for r in band_stats):
        legend += '   † Test non calcolabile.'
    fig.text(0.03, 0.075, legend, fontsize=11, color=ink)
    fig.text(0.03, 0.035, 'Interazioni nel training medie sui seed. Scala simmetrica adattata a questo report.',
             fontsize=10, color=ink)
    fig.savefig(out / 'activity_band_heatmap.png', dpi=180)


def activity_band_report(summary, bin_labels, out):
    """Write the band statistics CSV and return the final Markdown section."""
    lines = []
    band_stats = activity_band_statistics(summary, bin_labels)
    activity_band_heatmap(band_stats, out)
    csv_path = out / 'activity_band_statistics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(band_stats[0]))
        writer.writeheader()
        writer.writerows(band_stats)

    def number(value, fmt='%.6f'):
        return 'n/d' if value is None else fmt % value

    def pvalue(value):
        return '< 1e-300' if value == 0 else number(value, '%.2e')

    untestable = [r for r in band_stats if r['reason']]
    notes = []
    if untestable:
        notes.append('**Test non calcolabili:**\n\n' + '\n'.join(
            '- **S%d × T%d:** %s.' % (r['source_bin'], r['target_bin'], r['reason'])
            for r in untestable) + '\n')

    lines.extend([
        '## Significatività del confronto DGCDR–LightGCN per fascia\n',
        'T-test appaiato bilaterale su NDCG@20: ipotesi nulla di differenza media '
        'pari a zero. Si usano le medie sui seed per ciascun utente, così ogni '
        'utente pesa una volta; sono inclusi anche gli utenti con NDCG nullo. '
        'Le fasce coincidono con quelle della sezione Attività source × target.\n',
        'La correzione di Holm considera insieme le **%d combinazioni** source × target '
        'del report (nove quando entrambi i domini hanno tre fasce; le soglie '
        'coincidenti accorpano le fasce). Anche le combinazioni vuote o con test '
        'non calcolabile restano nel numero di confronti della correzione. '
        'Il t-test globale è un confronto distinto e non entra in questa famiglia. '
        'La decisione usa **p Holm < 0,05**, calcolato prima di arrotondare.\n' % len(band_stats),
        '**Limiti delle fasce — interazioni train medie sui seed:**\n',
        table(['Dominio', 'Fascia', 'Interazioni train N'], [
            [domain.capitalize(), '%s%d' % (prefix, index), label]
            for domain, prefix in [('source', 'S'), ('target', 'T')]
            for index, label in enumerate(bin_labels[domain])]),
        '**Risultati per fascia:** Delta = NDCG medio DGCDR − NDCG medio LightGCN, '
        'espresso in punti NDCG. Gli esiti riguardano il comportamento medio della '
        'fascia e non le etichette individuali basate sulla soglia relativa dell’1%.\n',
        table(['Fascia', 'Utenti', 'NDCG LightGCN', 'NDCG DGCDR', 'Delta medio',
               'p Holm', 'Esito'], [
            ['S%d × T%d' % (r['source_bin'], r['target_bin']), r['n'],
             number(r['ndcg_lightgcn']), number(r['ndcg_dgcdr']), number(r['mean_delta']),
             pvalue(r['pvalue_holm']), r['outcome']] for r in band_stats]),
        '![Heatmap del Delta medio NDCG@20 per fascia source × target; '
        'asterisco per p Holm < 0,05](activity_band_heatmap.png)\n',
        '**Dettagli statistici:** l’esito finale dipende dal p-value Holm.\n',
        table(['Fascia', 't', 'p iniziale'], [
            ['S%d × T%d' % (r['source_bin'], r['target_bin']),
             number(r['statistic'], '%.4f'), pvalue(r['pvalue'])] for r in band_stats]),
        *notes,
        '**Riepilogo:** ' + '; '.join('%s: **%d**' % (
            outcome, sum(r['outcome'] == outcome for r in band_stats))
            for outcome in ['Miglioramento significativo', 'Peggioramento significativo',
                            'Evidenza insufficiente', 'Non calcolabile']) + '.\n',
        'Evidenza insufficiente non dimostra equivalenza tra i modelli. '
        'Il confronto riguarda i checkpoint e gli split osservati e assume '
        'indipendenza tra utenti e un’approssimazione adeguata della distribuzione '
        'della media. La significatività va letta insieme all’entità del Delta.\n',
        '[Risultati statistici completi per fascia (CSV)](activity_band_statistics.csv).\n'])
    return '\n'.join(lines)


def source_activity_contrast_report(summary, bin_labels, out):
    """Write one exploratory S2 vs S0 contrast within T0, without multiplicity adjustment."""
    low = summary[(summary.source_bin == 0) & (summary.target_bin == 0)]
    high = summary[(summary.source_bin == 2) & (summary.target_bin == 0)]
    result = welch_transfer_test(
        (high.ndcg_dgcdr - high.ndcg_lightgcn).to_numpy(),
        (low.ndcg_dgcdr - low.ndcg_lightgcn).to_numpy())
    row = dict(source_low_range=bin_labels['source'][0],
               source_high_range=bin_labels['source'][2] if len(bin_labels['source']) == 3 else None,
               target_low_range=bin_labels['target'][0], **asdict(result))
    if len(bin_labels['source']) != 3 or len(bin_labels['target']) != 3:
        row['reason'] = 'terzili accorpati: il confronto previsto S2 × T0 vs S0 × T0 non è disponibile'
    if row['reason']:
        row['outcome'] = 'Non calcolabile'
        # Do not expose a test for a different, collapsed binning protocol.
        row['statistic'] = row['degrees_of_freedom'] = row['pvalue'] = None
    elif row['pvalue'] >= 0.05:
        row['outcome'] = 'Evidenza insufficiente'
    elif row['difference'] > 0:
        row['outcome'] = 'Vantaggio maggiore con source alto'
    else:
        row['outcome'] = 'Vantaggio maggiore con source basso'
    row.update(pvalue_holm=None, family_size=1, pvalue_adjustment='none',
               outcome_uncorrected=row['outcome'])
    with (out / 'source_activity_contrast.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return source_activity_contrast_markdown(row)


def source_activity_contrast_markdown(row):
    """Render raw or family-adjusted Welch results, keeping both p-values visible."""
    corrected = row.get('pvalue_adjustment') == 'holm'

    def number(value, fmt='%.6f'):
        return 'n/d' if value is None else fmt % value

    pvalue = '< 1e-300' if row['pvalue'] == 0 else number(row['pvalue'], '%.2e')
    headers = ['D (punti NDCG)', 't Welch', 'p bilaterale non corretto']
    cells = [number(row['difference']), number(row['statistic'], '%.4f'), pvalue]
    if corrected:
        headers.append('p Holm')
        cells.append('< 1e-300' if row['pvalue_holm'] == 0 else number(row['pvalue_holm'], '%.2e'))
    headers.append('Esito dopo Holm (5%)' if corrected else 'Esito esplorativo al 5%')
    cells.append(row['outcome'])
    lines = [
        '## Confronto esplorativo: source alto vs basso con target basso\n',
        '**Research question:** tra gli utenti con bassa attività nel target, '
        'un’elevata attività nel source è associata a un maggiore vantaggio di '
        'DGCDR rispetto a LightGCN?\n',
        'Si confrontano S2 × T0 e S0 × T0 con un **t-test di Welch bilaterale** '
        'sui Delta individuali (NDCG@20 DGCDR − LightGCN), mediati sui seed. '
        'Ogni utente pesa una volta, inclusi gli utenti con NDCG nullo per entrambi '
        'i modelli. I gruppi sono distinti; il test non assume varianze uguali. '
        'Target basso: **%s** interazioni train medie sui seed.\n' % row['target_low_range'],
        table(['Gruppo', 'Interazioni source', 'Utenti', 'Delta medio DGCDR − LightGCN'], [
            ['Source basso (S0 × T0)', row['source_low_range'], row['n_low'], number(row['mean_low'])],
            ['Source alto (S2 × T0)', row['source_high_range'] or 'n/d', row['n_high'],
             number(row['mean_high'])]]),
        '**Differenza D = Delta medio source alto − Delta medio source basso.** '
        'L’ipotesi nulla è D = 0. D positivo indica un vantaggio relativo maggiore '
        'con source alto; non implica, da solo, che DGCDR superi LightGCN in valore assoluto.\n',
        table(headers, [cells]),
    ]
    if row['reason']:
        lines.append('Test non calcolabile: %s.\n' % row['reason'])
    lines.extend([
        'L’analisi assume indipendenza fra utenti e riguarda i checkpoint e gli split '
        'osservati. Descrive un’associazione fra gruppi, non un effetto causale '
        'dell’aggiunta di dati source. Non verifica se l’effetto sia più forte '
        'rispetto alla condizione di target alto.\n',
        '[Risultato completo del confronto Welch (CSV)](source_activity_contrast.csv).\n'])
    if corrected:
        lines.append(
            '**Nota sui confronti multipli:** correzione di **Holm applicata insieme '
            'ai %d confronti Welch** della famiglia di direzioni analizzate. '
            'L’esito usa **p Holm < 0,05** prima di arrotondare; il p-value iniziale '
            'è conservato per confronto. Anche i test non calcolabili restano '
            'conteggiati nella famiglia. La correzione non cambia medie e differenze '
            'osservate. Questa famiglia resta distinta dai nove t-test per fascia, '
            'che hanno la propria correzione di Holm. L’analisi resta esplorativa.\n'
            % row['family_size'])
    else:
        lines.append('**Nota sui confronti multipli:** il p-value di questo confronto esplorativo '
        'è non corretto. Un unico confronto effettivamente previsto non richiede '
        'Holm; se vengono definiti più confronti o direzioni, occorre considerarli '
        'insieme e correggere i p-value della famiglia prevista. I risultati non '
        'vanno selezionati in base ai p-value. Questa analisi resta distinta dai '
        'nove t-test per fascia, per i quali la correzione di Holm è già applicata.\n')
    return '\n'.join(lines)


def apply_source_contrast_holm(output_dirs, summary_out):
    """Correct one explicit family of saved Welch comparisons and update its reports."""
    outputs = [Path(path).resolve() for path in output_dirs]
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError('Specificare una famiglia non vuota di cartelle distinte')
    heading = '## Confronto esplorativo: source alto vs basso con target basso'
    entries = []
    for out in outputs:
        with (out / 'source_activity_contrast.csv').open(encoding='utf-8', newline='') as f:
            saved_rows = list(csv.DictReader(f))
        if len(saved_rows) != 1:
            raise ValueError('Atteso un solo confronto Welch: ' + str(out))
        row = saved_rows[0]
        for key in ('n_high', 'n_low'):
            row[key] = int(row[key])
        for key in ('mean_high', 'mean_low', 'std_high', 'std_low', 'difference',
                    'statistic', 'degrees_of_freedom', 'pvalue'):
            row[key] = float(row[key]) if row[key] else None
        row['reason'] = row['reason'] or None
        if row['pvalue'] is None and not row['reason']:
            raise ValueError('p-value mancante senza motivazione: ' + str(out))
        original = (out / 'report.md').read_text(encoding='utf-8')
        if original.count(heading) != 1:
            raise ValueError('Sezione Welch mancante o duplicata: ' + str(out))
        spec = json.loads((out / 'config.json').read_text(encoding='utf-8'))['spec']
        entries.append((out, row, original, spec))
    adjusted = holm_adjust([r['pvalue'] if not r['reason'] else 1.0
                            for _, r, _, _ in entries])
    results = []
    for (out, row, original, spec), p_holm in zip(entries, adjusted):
        if row['reason']:
            raw_outcome = outcome = 'Non calcolabile'
        else:
            favored = ('Vantaggio maggiore con source alto' if row['difference'] > 0
                       else 'Vantaggio maggiore con source basso')
            raw_outcome = favored if row['pvalue'] < 0.05 else 'Evidenza insufficiente'
            outcome = favored if p_holm < 0.05 else 'Evidenza insufficiente'
        row.update(pvalue_holm=float(p_holm) if not row['reason'] else None,
                   family_size=len(entries), pvalue_adjustment='holm',
                   outcome_uncorrected=raw_outcome, outcome=outcome)
        with (out / 'source_activity_contrast.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        prefix = original.split('\n' + heading, 1)[0]
        (out / 'report.md').write_text(prefix.rstrip() + '\n\n' +
                                       source_activity_contrast_markdown(row), encoding='utf-8')
        results.append(dict(direction=out.name, source=spec['source'], target=spec['target'], **row))
    summary_out = Path(summary_out)
    summary_out.mkdir(parents=True, exist_ok=True)
    with (summary_out / 'source_activity_contrast_summary.csv').open(
            'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    def number(value, fmt='%+.6f'):
        return 'n/d' if value is None else fmt % value

    outcomes = ['Vantaggio maggiore con source alto', 'Vantaggio maggiore con source basso',
                'Evidenza insufficiente', 'Non calcolabile']
    lines = [
        '# Confronto dell’attività source con target basso — correzione di Holm\n',
        '**Research question:** tra gli utenti con target basso, un’attività source '
        'alta è associata a un maggiore vantaggio NDCG@20 di DGCDR rispetto a LightGCN?\n',
        'Welch bilaterale su S2 × T0 vs S0 × T0, usando i Delta individuali medi '
        'sui seed. D = Delta source alto − Delta source basso. '
        '**Holm applicata ai %d confronti**; gli esiti usano p Holm < 0,05. '
        'La famiglia include tutte le direzioni specificate, senza selezione '
        'basata sui p-value.\n' % len(entries),
        table(['Direzione', 'Delta source basso', 'Delta source alto', 'D (punti NDCG)',
               'p iniziale', 'p Holm', 'Esito dopo Holm'], [
            [r['direction'].replace('_to_', ' → ').replace('_', ' ').title(),
             number(r['mean_low']), number(r['mean_high']), number(r['difference']),
             number(r['pvalue'], '%.3e'), number(r['pvalue_holm'], '%.3e'), r['outcome']]
            for r in results]),
        table(['Esito', 'Prima di Holm', 'Dopo Holm'], [
            [name, sum(r['outcome_uncorrected'] == name for r in results),
             sum(r['outcome'] == name for r in results)] for name in outcomes]),
        'Le medie e D non cambiano: Holm corregge la decisione considerando '
        'l’insieme dei confronti. Evidenza insufficiente non dimostra equivalenza. '
        'L’analisi resta esplorativa, assume indipendenza fra utenti e riguarda '
        'i checkpoint e gli split osservati. Descrive associazioni, non causalità.\n',
        '[Risultati completi e limiti delle fasce (CSV)](source_activity_contrast_summary.csv).\n',
        '**Nota sui confronti multipli:** questa correzione riguarda i confronti '
        'Welch tra direzioni; resta distinta dalla correzione dei nove t-test per '
        'fascia all’interno di ogni report.\n']
    (summary_out / 'source_activity_contrast_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return results


def report(spec, out, source_contrast=False):
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
    test_rows = [['Utenti confrontati', paired_test.n],
                 ['Delta medio (punti NDCG)',
                  '%.6f' % paired_test.mean_delta if paired_test.mean_delta is not None else 'n/d']]
    if paired_test.reason:
        test_rows.extend([['IC 95% del Delta medio', 'n/d'],
                          ['Statistica t', 'n/d'],
                          ['p-value bilaterale', 'n/d']])
        test_conclusion = 'Test non calcolabile: %s.' % paired_test.reason
    else:
        pvalue_text = ('< 1e-300' if paired_test.pvalue == 0
                       else '%.2e' % paired_test.pvalue)
        test_rows.extend([['IC 95% del Delta medio', '[%.6f, %.6f]' % (
                              paired_test.ci_low, paired_test.ci_high)],
                          ['Statistica t', '%.4f' % paired_test.statistic],
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
             'repeatable=%s.\n\n'
             'La rilevanza è basata sulla presenza dell’interazione. '
             'Il parametro `threshold` salvato nei checkpoint (DGCDR %s, LightGCN %s) '
             'genera etichette binarie, ma non filtra le interazioni; tali etichette '
             'non vengono utilizzate dal training BPR né per selezionare i positivi '
             'nella valutazione full-ranking.\n' % (
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
                 'La riga “Tutti gli utenti” usa l’intera popolazione analizzata: prima si '
                 'calcola la media sui seed per utente, poi ogni utente pesa una volta. '
                 'Le misure sono ricalcolate sui valori individuali non arrotondati: '
                 'P̄ sui soli Negative, Ḡ sui soli Positive e Delta su tutti gli utenti.\n\n'
                 '**n/d** indica che la classe corrispondente è vuota oppure che la media '
                 'LightGCN usata come denominatore è zero.\n')
    cells = []
    for (s,t), group in summary.groupby(['source_bin','target_bin']):
        cells.append(transfer_activity_row(s, t, group))
    cells.append(['**%s**' % value for value in
                  transfer_activity_row('Tutti gli utenti', '—', summary)])
    lines.append(table(['Fascia S','Fascia T','Utenti','Negative',
                        'Positive','P̄ (%)','Ḡ (%)','Delta (%)'],cells))
    lines.extend([
        '## Confronto individuale\n[Tutti gli utenti, ordinati per delta e separati per classe](users.md). '
        '[Risultati completi per utente e seed](per_user.csv).\n'])
    lines.append(activity_band_report(summary, bin_labels, out))
    if source_contrast:
        lines.append(source_activity_contrast_report(summary, bin_labels, out))
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
    parser.add_argument('--source-contrast', action='store_true',
                        help='Add an exploratory Welch S2 vs S0 comparison within T0 for this direction')
    parser.add_argument('--holm-source-contrasts', nargs='+', metavar='OUTPUT_DIR',
                        help='Apply Holm to the explicit family of saved Welch comparison directories')
    parser.add_argument('--contrast-summary-dir', default=str(ROOT / 'transfer_results/integrated_analysis'),
                        help='Destination for the combined Welch/Holm summary')
    args = parser.parse_args()
    # RecBole parses sys.argv itself; keep this entry point's arguments out of it.
    sys.argv = [sys.argv[0]]
    if args.holm_source_contrasts:
        apply_source_contrast_holm(args.holm_source_contrasts, args.contrast_summary_dir)
        return
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
        report(saved['spec'], out, source_contrast=args.source_contrast)
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
    report(spec, out, source_contrast=args.source_contrast)
    logging.info('Report ready: %s',out/'report.md')


if __name__ == '__main__':
    main()
