"""Contribution to the score vs contribution to the prediction.

tau measures how much of a *score* a channel carries. This measures how much of
the *decision* it carries -- the exact signed share of the margin between a
relevant item and an irrelevant one, which is the only thing a ranking metric
can see.

    python -m extensions.explainability.decomposition.run_discrimination -m saved/DGCDR-Jul-25-2026_11-32-27.pth -n 300

A channel with high tau and delta near zero adds score mass without moving the
ranking, and that is a testable explanation of why zeroing the shared channel
costs no accuracy while tau still predicts which recommendations collapse.
"""

import argparse
import json
import os
import sys
from datetime import datetime

# This script lives three levels below the repository root, so running it by
# path puts its own directory on sys.path instead of the root, and neither
# recbole_cdr nor extensions resolves. Put the root back before importing them.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     os.pardir, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
# RecBole 1.0.1 still uses np.float/np.int/np.bool, removed in numpy>=1.24.
# CDRConfig patches them in its constructor, but load_data_and_model unpickles
# the config instead of constructing one, so the patch has to happen here.
np.float = float
np.int = int
np.bool = bool

import torch

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.decomposition.core.attribution import explain_user, pooled_transfer_ratio
from extensions.explainability.decomposition.core.channels import (
    decompose_target_domain,
    verify_decomposition,
)
from extensions.explainability.decomposition.core.discrimination import analyse_user, summarise
from extensions.explainability.decomposition.core.data import build_ground_truth, select_users


def _fmt(value, width, places=4):
    """Right-aligned number, or a dash when the metric was not computed."""
    return f"{value:>{width}.{places}f}" if value is not None else f"{'-':>{width}}"


def main():
    parser = argparse.ArgumentParser(description="Score contribution vs prediction contribution")
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, default='discrimination_out')
    parser.add_argument('--topk', '-k', type=int, default=20)
    parser.add_argument('--num_users', '-n', type=int, default=3000)
    parser.add_argument('--negatives', type=int, default=200,
                        help="Irrelevant items sampled per relevant one")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Carico {args.model_path}...")
    config, model, dataset, _, _, test_data = load_data_and_model(args.model_path)
    model.eval()

    decomposition = decompose_target_domain(model)
    verification = verify_decomposition(model, decomposition)
    if not verification['passed']:
        raise RuntimeError("La decomposizione non ricostruisce il modello.")

    ground_truth = build_ground_truth(test_data)
    users = select_users(model, args.num_users, args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    results = []
    for user_id in users:
        outcome = analyse_user(model, decomposition, user_id,
                               ground_truth.get(user_id, []), args.topk,
                               args.negatives, generator=generator)
        if outcome is not None:
            results.append(outcome)

    summary = summarise(results, args.topk)
    if not summary:
        raise RuntimeError("Nessun utente utilizzabile: tutti senza item rilevanti nel test set.")

    # tau on the same users, so the two ratios are directly comparable.
    attributions = []
    for user_id in (r['user_id'] for r in results):
        attributions.extend(explain_user(model, decomposition, user_id,
                                         topk=args.topk).attributions)
    summary['tau_pooled'] = pooled_transfer_ratio(attributions)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f"disc_{os.path.basename(args.model_path).replace('.pth', '')}_{timestamp}"
    path = os.path.join(args.output_dir, f'{stem}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'model_checkpoint': args.model_path, 'summary': summary}, f, indent=2)

    print(f"\n{'=' * 72}\n1. OGNI CANALE DA SOLO, COME RANKER\n{'=' * 72}")
    print(f"   ({summary['n_users']} utenti, {summary['n_positives']} item rilevanti)\n")
    print(f"    {'canale':12} {'Recall@' + str(args.topk):>12} {'NDCG@' + str(args.topk):>12} {'AUC':>10}")
    for name, data in summary['quality'].items():
        # A missing value must not drop the whole row, as it did when the
        # f-string was the branch of a conditional expression.
        cells = [_fmt(data.get(f'recall_at_{args.topk}'), 12),
                 _fmt(data.get(f'ndcg_at_{args.topk}'), 12),
                 _fmt(data.get('auc'), 10)]
        print(f"    {name:12} " + " ".join(cells))

    disentangled_share = summary['decision'].pop('_disentangled_margin_share')
    print(f"\n{'=' * 72}\n2. CONTRIBUTO AL PUNTEGGIO CONTRO CONTRIBUTO ALLA DECISIONE\n{'=' * 72}")
    print(f"    {'canale':12} {'delta':>10} {'quota con segno':>18} {'voto corretto':>16}")
    for name, data in summary['decision'].items():
        print(f"    {name:12} {_fmt(data['delta'], 10)} {_fmt(data['signed_share'], 18)} "
              f"{data['vote_accuracy'] * 100:>15.1f}%")

    print(f"\n    Quota del margine che passa dal disentanglement : {_fmt(disentangled_share, 6)}")
    print(f"    (il resto lo porta il canale collaborativo grezzo)")
    print(f"\n    Confronto a denominatore uguale, solo shared/specific:")
    print(f"      tau   (quota del punteggio) : {summary['tau_pooled']:.4f}")
    print(f"      delta (quota del margine)   : {_fmt(summary['decision']['shared']['delta_disentangled'], 6)}")
    print(f"\n    accuratezza sulle coppie, punteggio pieno : {summary['pair_accuracy'] * 100:.1f}%"
          f"   (caso: 50.0%)")

    print(f"\n{'=' * 72}\n3. I CANALI ORDINANO IL CATALOGO ALLO STESSO MODO?\n{'=' * 72}")
    for key, value in summary['redundancy'].items():
        print(f"    Spearman({key:20}) = {value:+.4f}")

    print(f"\nJSON: {path}")


if __name__ == '__main__':
    main()
