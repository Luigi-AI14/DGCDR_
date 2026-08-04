"""Counterfactual test: what breaks when a channel is removed.

Zeroes the shared channel on the user side, re-ranks every sampled user, and
reports what it cost and whether tau predicted it.

    python run_counterfactual.py -m saved/DGCDR-Jul-25-2026_11-32-27.pth

The second question is the one that matters. tau claims a recommendation leans
on the shared channel; the honest test of that claim is whether the
recommendation actually falls apart when the channel is taken away.
"""

import argparse
import json
import os
from datetime import datetime

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.attribution import explain_user
from extensions.explainability.channels import (
    SHARED,
    decompose_target_domain,
    verify_decomposition,
)
from extensions.explainability.counterfactual import counterfactual_for_user, summarise
from extensions.explainability.data import build_ground_truth, select_users


def main():
    parser = argparse.ArgumentParser(description="Counterfactual channel ablation")
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, default='counterfactual_out')
    parser.add_argument('--topk', '-k', type=int, default=20)
    parser.add_argument('--num_users', '-n', type=int, default=500)
    parser.add_argument('--drop', nargs='*', default=[SHARED],
                        help="Channels to zero on the user side")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_path}...")
    config, model, dataset, _, _, test_data = load_data_and_model(args.model_path)
    model.eval()

    decomposition = decompose_target_domain(model)
    verification = verify_decomposition(model, decomposition)
    if not verification['passed']:
        raise RuntimeError("La decomposizione non ricostruisce il modello.")

    ground_truth = build_ground_truth(test_data)
    users = select_users(model, args.num_users, args.seed)
    print(f"Utenti: {len(users)} | canale azzerato: {args.drop}")

    # tau from the intact model, so the prediction is made before the ablation
    taus = {}
    for user_id in users:
        explanation = explain_user(model, decomposition, user_id, topk=args.topk)
        for attribution in explanation.attributions:
            taus[(user_id, attribution.item_id)] = attribution.tau

    results = [counterfactual_for_user(model, decomposition, u, args.topk,
                                       set(args.drop), ground_truth.get(u, []))
               for u in users]
    summary = summarise(results, taus)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f"cf_{os.path.basename(args.model_path).replace('.pth', '')}_{timestamp}"
    path = os.path.join(args.output_dir, f'{stem}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'model_checkpoint': args.model_path, 'dropped': args.drop,
                   'summary': summary, 'per_user': results}, f, indent=2)

    print(f"\n{'=' * 58}\n1. QUANTO COSTA TOGLIERE IL CANALE\n{'=' * 58}")
    print(f"  Recall@{args.topk} intatto  : {summary['recall_full']:.4f}")
    print(f"  Recall@{args.topk} ablato   : {summary['recall_ablated']:.4f}")
    if summary['recall_full']:
        delta = (summary['recall_ablated'] - summary['recall_full']) / summary['recall_full']
        print(f"  variazione            : {delta * 100:+.1f}%")
    print(f"  top-{args.topk} che sopravvive : {summary['mean_topk_overlap'] * 100:.1f}%")

    print(f"\n{'=' * 58}\n2. TAU PREVEDE COSA SPARISCE?\n{'=' * 58}")
    print(f"  tau medio, raccomandazioni cadute    : {summary['tau_dropped']:.4f} "
          f"(n={summary['n_dropped']})")
    print(f"  tau medio, raccomandazioni rimaste   : {summary['tau_survived']:.4f} "
          f"(n={summary['n_survived']})")
    if 'tau_gap' in summary:
        print(f"  differenza                           : {summary['tau_gap']:+.4f}"
              f"  (permutazione, p = {summary['tau_gap_p']:.4f})")
    if 'corr_tau_newrank' in summary:
        print(f"  corr(tau, nuovo rango)               : {summary['corr_tau_newrank']:+.4f}")
    print(f"\nJSON: {path}")


if __name__ == '__main__':
    main()
