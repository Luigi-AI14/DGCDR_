import argparse
import json
import os
from datetime import datetime

import numpy as np
# Patch numpy deprecated aliases used by RecBole
np.float = float
np.int = int
np.bool = bool

import torch

from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.trainer import CrossDomainTrainer

from extensions.explainability.attribution import explain_user, pooled_transfer_ratio
from extensions.explainability.channels import (
    decompose_target_domain,
    verify_decomposition,
)
from extensions.explainability.discrimination import analyse_user, summarise
from extensions.explainability.data import build_ground_truth, select_users


def _fmt(value, width, places=4):
    """Right-aligned number, or a dash when the metric was not computed."""
    return f"{value:>{width}.{places}f}" if value is not None else f"{'-':>{width}}"


def main():
    parser = argparse.ArgumentParser(description="Score contribution vs prediction contribution (RecBole native)")
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

    # 1. Run the standard diagnosis (delta, tau, redundancy, etc.)
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

    attributions = []
    for user_id in (r['user_id'] for r in results):
        attributions.extend(explain_user(model, decomposition, user_id,
                                         topk=args.topk).attributions)
    summary['tau_pooled'] = pooled_transfer_ratio(attributions)

    # 2. Run the RecBole Native Evaluation for Ranking Quality
    print("\nRunning RecBole native evaluation for isolated channels...")

    # The evaluator is built from the config when the trainer is constructed, so
    # a topk RecBole was not asked for would come back as a row of None.
    if args.topk not in (config['topk'] or []):
        config['topk'] = sorted(set(config['topk'] or []) | {args.topk})
    trainer = CrossDomainTrainer(config, model)
    original_get_restore_e = model.get_restore_e

    recbole_quality = {}
    channels = ['base', 'shared', 'specific', 'full']

    try:
        for channel in channels:
            print(f"Evaluating channel: {channel}")
            if channel == 'full':
                model.__dict__.pop('get_restore_e', None)  # back to the real method
            else:
                # Only the *user* side is isolated; the item side stays fused,
                # exactly as channel_score_vectors defines a channel's scores.
                def get_channel_restore_e(chan=channel):
                    _, restore_item_e = original_get_restore_e()
                    return decomposition.user_channels[chan], restore_item_e
                model.get_restore_e = get_channel_restore_e

            test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
            recbole_quality[channel] = {
                f'recall_at_{args.topk}': test_result.get(f'recall@{args.topk}'),
                f'ndcg_at_{args.topk}': test_result.get(f'ndcg@{args.topk}'),
                'mrr': test_result.get(f'mrr@{args.topk}'),
                'hit': test_result.get(f'hit@{args.topk}'),
            }
    finally:
        # Leaving the patch in place would silently corrupt anything scored later.
        model.__dict__.pop('get_restore_e', None)

    # Replace the quality section in summary with RecBole's metrics. The two are
    # not interchangeable: RecBole scores every test user and, with
    # repeatable=True, masks nothing but [PAD]; quality_original scores the
    # sampled overlapping users with their training history masked out.
    summary['quality_original'] = summary['quality']  # micro-averaged, sampled users
    summary['quality'] = recbole_quality
    summary['quality_protocol'] = {
        'quality': 'RecBole evaluator, all test users, repeatable=True (nothing masked)',
        'quality_original': f"{summary['n_users']} sampled overlapping users, training history masked",
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f"disc_recbole_{os.path.basename(args.model_path).replace('.pth', '')}_{timestamp}"
    path = os.path.join(args.output_dir, f'{stem}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'model_checkpoint': args.model_path, 'summary': summary}, f, indent=2)

    print(f"\n{'=' * 72}\n1. OGNI CANALE DA SOLO, COME RANKER (RECBOLE)\n{'=' * 72}")
    print(f"   (tutti gli utenti di test, protocollo RecBole: niente e' mascherato)\n")
    print(f"    {'canale':12} {'Recall@' + str(args.topk):>12} {'NDCG@' + str(args.topk):>12} {'MRR':>10}")
    for name, data in summary['quality'].items():
        # A missing value must not drop the whole row, as it did when the
        # f-string was the branch of a conditional expression.
        cells = [_fmt(data.get(f'recall_at_{args.topk}'), 12),
                 _fmt(data.get(f'ndcg_at_{args.topk}'), 12),
                 _fmt(data.get('mrr'), 10)]
        print(f"    {name:12} " + " ".join(cells))

    disentangled_share = summary['decision'].pop('_disentangled_margin_share')
    print(f"\n   (le sezioni 2 e 3 usano {summary['n_users']} utenti sovrapposti campionati, "
          f"{summary['n_positives']} item rilevanti, con lo storico di training mascherato)")
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
