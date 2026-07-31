"""Controlled checks on the claims DGCDR makes about its own mechanisms.

Each check targets one published claim and pairs it with the control that claim
was never given. Neither needs retraining.

  anchor   Sec. 3.3, the paper's second contribution. The decoder's anchor
           supervision is meant to align a user's domain-shared features across
           domains. Does the hierarchy it optimises actually hold, does it hold
           *per user* (permutation null), and can the shared channel identify a
           user across domains better than the raw GNN embedding it started
           from?

  fusion   Table 5, the -Pers row. Replacing personalised fusion with
           concatenation costs 22.69%, the largest ablation in the paper. That
           variant changes the fusion's form as well as its personalisation;
           this isolates the personalisation by handing each user a constant --
           or another user's -- attention weights, at inference.

Examples:
    python audit_paper_claims.py -m saved/DGCDR-Jul-25-2026_11-32-27.pth
    python audit_paper_claims.py -m saved/model.pth --only fusion -n 500
"""

import argparse
import json
import os
from datetime import datetime

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.anchor_audit import (
    hierarchy_with_null,
    identification_by_channel,
)
from extensions.explainability import fusion_audit
from explain_dgcdr import build_ground_truth, select_users


def report_anchor(results, identification):
    print(f"\n{'=' * 70}\nA. LA SUPERVISIONE AD ANCORA (Sec. 3.3, contributo C2)\n{'=' * 70}")
    for direction, data in results.items():
        paired, null, gap = data['paired'], data['null'], data['gap']
        print(f"\n  {direction}  ({data['n_users']} utenti sovrapposti)")
        print(f"    {'':28} {'appaiato':>10} {'permutato':>10} {'divario':>10}")
        for key, label in (('cos_shared', 'cos(ancora, shared)'),
                           ('cos_gnn', 'cos(ancora, gnn)'),
                           ('cos_specific', 'cos(ancora, specific)')):
            print(f"    {label:28} {paired[key]:>10.4f} {null[key]:>10.4f} {gap[key]:>+10.4f}")
        for key, label in (('rate_shared_over_gnn', 'shared > gnn'),
                           ('rate_gnn_over_specific', 'gnn > specific'),
                           ('rate_full_ordering', 'ordinamento completo')):
            print(f"    {label:28} {paired[key] * 100:>9.1f}% {null[key] * 100:>9.1f}% "
                  f"{gap[key] * 100:>+9.1f}%")

    print(f"\n{'=' * 70}\nB. IL CANALE SHARED IDENTIFICA L'UTENTE FRA I DOMINI?\n{'=' * 70}")
    any_channel = next(iter(identification.values()))
    print(f"  Ogni utente contro {any_channel['n_candidates']} candidati.")
    print(f"  Caso: hit@1 = {any_channel['chance_hit_at_1'] * 100:.2f}%, "
          f"rango mediano = {any_channel['chance_median_rank']:.0f}\n")
    print(f"    {'canale':18} {'hit@1':>9} {'hit@10':>9} {'MRR':>9} {'rango med.':>11}")
    for name, data in identification.items():
        print(f"    {name:18} {data['hit_at_1'] * 100:>8.2f}% {data['hit_at_10'] * 100:>8.2f}% "
              f"{data['mrr']:>9.4f} {data['median_rank']:>11.0f}")


def report_fusion(results, topk):
    print(f"\n{'=' * 70}\nC. LA FUSIONE PERSONALIZZATA E' PERSONALE? (Tabella 5, -Pers)\n{'=' * 70}")
    stats = results['attention_stats']
    print(f"  Peso shared: media {stats['mean_shared_weight']:.4f}, "
          f"deviazione {stats['std_shared_weight']:.4f}, "
          f"intervallo [{stats['min_shared_weight']:.3f}, {stats['max_shared_weight']:.3f}]\n")
    print(f"    {'attention':14} {'Recall@' + str(topk):>12} {'NDCG@' + str(topk):>12} "
          f"{'top-k uguale':>14}   {'variazione (IC 95%)':>26}")
    for name in ('intact', 'constant', 'permuted'):
        data = results[name]
        recall, ndcg = data[f'recall_at_{topk}'], data[f'ndcg_at_{topk}']
        overlap = data['mean_topk_overlap']
        change = ''
        boot = data.get('vs_intact')
        if boot:
            mark = '' if boot['significant'] else '  n.s.'
            change = (f"  {boot['relative_change'] * 100:+.1f}% "
                      f"[{boot['ci_low'] * 100:+.1f}, {boot['ci_high'] * 100:+.1f}]{mark}")
        print(f"    {name:14} {recall:>12.4f} {ndcg:>12.4f} {overlap * 100:>13.1f}%{change}")


def main():
    parser = argparse.ArgumentParser(description="Controlled checks on DGCDR's published claims")
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, default='audit_out')
    parser.add_argument('--only', choices=['anchor', 'fusion', 'all'], default='all')
    parser.add_argument('--num_users', '-n', type=int, default=1000,
                        help="Users sampled for the fusion evaluation")
    parser.add_argument('--id_sample', type=int, default=2000,
                        help="Candidate-set size for cross-domain identification")
    parser.add_argument('--topk', '-k', type=int, default=20)
    parser.add_argument('--permutations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Carico {args.model_path}...")
    config, model, dataset, _, _, test_data = load_data_and_model(args.model_path)
    model.eval()

    domains = f"{config['source_domain']['dataset']} -> {config['target_domain']['dataset']}"
    print(f"Domini: {domains} | fuse_mode={config['fuse_mode']}")

    output = {'model_checkpoint': args.model_path, 'domains': domains,
              'hyper': {name: config[name] for name in
                        ('cl_sim_weight', 'cl_org_weight', 'cl_decoder_weight',
                         'item_cl_weight', 'temperature', 'fuse_mode')
                        if config[name] is not None}}

    if args.only in ('anchor', 'all'):
        anchor = hierarchy_with_null(model, args.permutations, args.seed)
        ident = identification_by_channel(model, args.id_sample, args.seed)
        output['anchor'] = anchor
        output['identification'] = ident
        report_anchor(anchor, ident)

    if args.only in ('fusion', 'all'):
        ground_truth = build_ground_truth(test_data)
        users = select_users(model, args.num_users, args.seed)
        fusion = fusion_audit.run(model, users, ground_truth, args.topk, args.seed)
        output['fusion'] = fusion
        report_fusion(fusion, args.topk)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f"audit_{os.path.basename(args.model_path).replace('.pth', '')}_{timestamp}"
    path = os.path.join(args.output_dir, f'{stem}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON: {path}")


if __name__ == '__main__':
    main()
