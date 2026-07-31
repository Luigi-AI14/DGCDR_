"""What happens to the recommendations when a channel is removed.

The attribution says how a score was composed. It does not say what the model
would have done without one of its parts -- those are different questions, and
only the second one is causal.

This module answers the second. It zeroes a channel, re-ranks, and asks two
things:

1. how much accuracy the model loses without that channel;
2. whether tau predicts *which* recommendations change.

The second is the real test of Contribution 1. If the items that disappear when
the shared channel is removed are the ones tau called shared-driven, then tau
measures something the model actually does. If they are not, tau is bookkeeping
that happens to sum correctly.
"""

import numpy as np
import torch


def ablated_user_embeddings(decomposition, drop):
    """Fused user embeddings with one or more channels zeroed."""
    return sum(v for k, v in decomposition.user_channels.items() if k not in drop)


def _scores(user_emb, item_emb, user_id, history, n_items):
    scores = torch.matmul(item_emb[:n_items], user_emb[user_id])
    if history:
        scores[torch.as_tensor(history, dtype=torch.long, device=scores.device)] = float('-inf')
    scores[0] = float('-inf')  # [PAD]
    return scores


def counterfactual_for_user(model, decomposition, user_id, topk, drop,
                            ground_truth=None):
    """Re-rank one user with and without a channel.

    Returns a dict with the original top-k, which of them survive in the new
    top-k, their new ranks, and hits before and after.
    """
    n_items = model.target_num_items
    item_emb = decomposition.fused_item_embeddings()

    inter = model.target_interaction_matrix
    history = [int(i) for i in inter.col[inter.row == user_id] if int(i) < n_items]

    full = _scores(decomposition.fused_user_embeddings(), item_emb, user_id, history, n_items)
    ablated = _scores(ablated_user_embeddings(decomposition, drop), item_emb,
                      user_id, history, n_items)

    top_full = torch.topk(full, k=min(topk, n_items - 1)).indices
    top_ablated = torch.topk(ablated, k=min(topk, n_items - 1)).indices
    survivors = set(int(i) for i in top_ablated)

    # New rank of each originally recommended item: how many items now beat it.
    new_ranks = [int((ablated > ablated[item]).sum().item()) + 1 for item in top_full]

    held_out = set(ground_truth or [])
    return {
        'user_id': user_id,
        'top_full': [int(i) for i in top_full],
        'survived': [int(i) in survivors for i in top_full],
        'new_ranks': new_ranks,
        'overlap': len(survivors & set(int(i) for i in top_full)) / len(top_full),
        'hits_full': sum(1 for i in top_full if int(i) in held_out),
        'hits_ablated': sum(1 for i in top_ablated if int(i) in held_out),
        'n_held_out': len(held_out),
    }


def summarise(results, taus_by_item):
    """Aggregate, and test whether tau predicts what disappears.

    ``taus_by_item`` maps (user_id, item_id) to the tau the attribution gave
    that recommendation before the ablation.
    """
    dropped_tau, survived_tau, displacements = [], [], []
    overlap, hits_full, hits_ablated, n_held = [], 0, 0, 0

    for r in results:
        overlap.append(r['overlap'])
        hits_full += r['hits_full']
        hits_ablated += r['hits_ablated']
        n_held += r['n_held_out']
        for item, survived, new_rank in zip(r['top_full'], r['survived'], r['new_ranks']):
            tau = taus_by_item.get((r['user_id'], item))
            if tau is None:
                continue
            (survived_tau if survived else dropped_tau).append(tau)
            displacements.append((tau, new_rank))

    summary = {
        'n_users': len(results),
        'mean_topk_overlap': float(np.mean(overlap)) if overlap else None,
        'recall_full': hits_full / n_held if n_held else None,
        'recall_ablated': hits_ablated / n_held if n_held else None,
        'n_dropped': len(dropped_tau),
        'n_survived': len(survived_tau),
        'tau_dropped': float(np.mean(dropped_tau)) if dropped_tau else None,
        'tau_survived': float(np.mean(survived_tau)) if survived_tau else None,
    }

    if dropped_tau and survived_tau:
        summary['tau_gap'] = summary['tau_dropped'] - summary['tau_survived']
        summary['tau_gap_p'] = _permutation_p(dropped_tau, survived_tau)

    if len(displacements) > 2:
        taus = np.array([d[0] for d in displacements])
        ranks = np.array([float(d[1]) for d in displacements])
        summary['corr_tau_newrank'] = _spearman(taus, ranks)

    return summary


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _permutation_p(a, b, n=10000, seed=42):
    """Two-sided permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    observed = abs(np.mean(a) - np.mean(b))
    n_a = len(a)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        if abs(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])) >= observed:
            count += 1
    return count / n
