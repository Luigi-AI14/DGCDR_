"""Does a channel help the *prediction*, or only the score?

``tau`` says how a score was composed. It is a share of magnitude, and magnitude
is not what a recommender is judged on. A channel that adds roughly the same
amount to every item raises the score of all of them and leaves the ranking
exactly where it was: it carries contribution mass and no decision. The
prediction is an ordering, so only the *differences between items* can move it.

This module measures that second quantity, and it stays exact. For a relevant
item ``i+`` and an irrelevant ``i-`` the model ranks correctly iff
``s(u,i+) > s(u,i-)``, and that margin decomposes by the same distributive
property the whole framework rests on::

    s(u,i+) - s(u,i-) = dC_base + dC_shared + dC_specific

so every channel has an exact, signed contribution *to the decision*. It is also
the form of the BPR loss the model was trained with, which makes it the natural
object rather than a convenient one.

From it comes **delta**, the counterpart of tau:

    tau   = share of the score's magnitude carried by the channel
    delta = share of the ranking margin carried by the channel

The diagnostic is the **gap between them**. A channel with high tau and delta
near zero is decorative: it is in the arithmetic of the score and absent from
the decision. That is a testable explanation for the standing puzzle of the
counterfactual -- tau predicts which recommendations collapse when the shared
channel is removed, yet removing it costs no accuracy.

Three controls make the numbers readable:

* ``vote_accuracy`` has a chance level of 0.5, so a channel that ranks no better
  than a coin is visible as such;
* the ``base`` channel is the natural reference, being the plain collaborative
  signal before any disentanglement;
* ``rank_redundancy`` asks whether two channels order the catalogue the same
  way. If they do, the second cannot add discrimination however much score mass
  it carries -- which is what the collapsed item gates predict.
"""

import numpy as np
import torch

from extensions.explainability.decomposition.core.channels import BASE, SHARED, SPECIFIC


def channel_score_vectors(model, decomposition, user_id):
    """Per-channel score over the whole catalogue, for one user.

    ``C_k(u, i) = <U_k[u], sum_l I_l[i]>``, and the item channels sum to the
    fused item embedding, so each channel's scores over every item are a single
    matrix-vector product. The channels sum back to the model's score by
    construction; :func:`verify` checks it rather than assuming it.
    """
    n_items = model.target_num_items
    item_emb = decomposition.fused_item_embeddings()[:n_items]
    with torch.no_grad():
        return {name: torch.matmul(item_emb, vector[user_id])
                for name, vector in decomposition.user_channels.items()}


def _valid_mask(model, user_id, n_items, device):
    """Items the user could still be recommended: not history, not [PAD]."""
    inter = model.target_interaction_matrix
    mask = torch.ones(n_items, dtype=torch.bool, device=device)
    history = [int(i) for i in inter.col[inter.row == user_id] if int(i) < n_items]
    if history:
        mask[torch.as_tensor(history, dtype=torch.long, device=device)] = False
    mask[0] = False
    return mask


def _auc(positive_scores, negative_scores):
    """P(score of a relevant item > score of an irrelevant one), ties at 0.5.

    Computed against every negative rather than a sample: the whole point of
    this module is to compare channels whose differences may be small, and a
    sampled AUC would add noise on the same order.
    """
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return None
    ordered = torch.sort(negative_scores).values
    below = torch.searchsorted(ordered, positive_scores, right=False).float()
    at_or_below = torch.searchsorted(ordered, positive_scores, right=True).float()
    ties = at_or_below - below
    return ((below + 0.5 * ties) / ordered.numel()).mean().item()


def _dcg(gains):
    return float((gains / np.log2(np.arange(2, gains.size + 2))).sum())


def _ranking_quality(scores, mask, positives, topk):
    """Recall@k, NDCG@k and AUC when this channel is the only ranker."""
    masked = scores.clone()
    masked[~mask] = float('-inf')
    top = [int(i) for i in torch.topk(masked, k=min(topk, int(mask.sum()))).indices]

    gains = np.array([1.0 if i in positives else 0.0 for i in top])
    ideal = np.zeros_like(gains)
    ideal[:min(len(positives), gains.size)] = 1.0

    positive_ids = torch.as_tensor(sorted(positives), dtype=torch.long, device=scores.device)
    positive_mask = torch.zeros_like(mask)
    positive_mask[positive_ids] = True

    return {
        'hits': int(gains.sum()),
        'ndcg': _dcg(gains) / _dcg(ideal) if ideal.sum() > 0 else 0.0,
        'auc': _auc(scores[mask & positive_mask], scores[mask & ~positive_mask]),
    }


def _margins(channel_scores, mask, positives, n_negatives, generator):
    """Signed per-channel contribution to (relevant vs irrelevant) decisions."""
    device = mask.device
    positive_ids = torch.as_tensor(sorted(positives), dtype=torch.long, device=device)
    positive_mask = torch.zeros_like(mask)
    positive_mask[positive_ids] = True

    candidates = torch.nonzero(mask & ~positive_mask, as_tuple=False).squeeze(1)
    kept = positive_ids[mask[positive_ids]]
    if candidates.numel() == 0 or kept.numel() == 0:
        return None

    n_draw = min(n_negatives, candidates.numel())
    picked = candidates[torch.randperm(candidates.numel(), generator=generator,
                                       device='cpu')[:n_draw].to(device)]

    # [n_positives, n_negatives] margin per channel.
    return {name: scores[kept].unsqueeze(1) - scores[picked].unsqueeze(0)
            for name, scores in channel_scores.items()}


def _spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = ra.norm() * rb.norm()
    return float((ra @ rb / denominator).item()) if denominator > 0 else 0.0


def analyse_user(model, decomposition, user_id, positives, topk=20,
                 n_negatives=200, n_redundancy_items=2000, generator=None):
    """Everything this module measures, for one user. None if unusable."""
    if not positives:
        return None
    n_items = model.target_num_items
    channel_scores = channel_score_vectors(model, decomposition, user_id)
    device = next(iter(channel_scores.values())).device
    mask = _valid_mask(model, user_id, n_items, device)

    positives = {i for i in positives if 0 < i < n_items and bool(mask[i])}
    if not positives:
        return None

    total = sum(channel_scores.values())
    scored = dict(channel_scores)
    scored['full'] = total

    quality = {name: _ranking_quality(scores, mask, positives, topk)
               for name, scores in scored.items()}

    margins = _margins(channel_scores, mask, positives, n_negatives, generator)
    if margins is None:
        return None

    absolute = sum(m.abs() for m in margins.values())
    full_margin = sum(margins.values())
    decision = {}
    for name, m in margins.items():
        decision[name] = {
            'abs_margin': m.abs().sum().item(),
            'signed_margin': m.sum().item(),
            'vote_accuracy': (m > 0).float().mean().item(),
        }

    # Do two channels order the catalogue the same way? Sub-sampled: a full
    # Spearman over the catalogue costs a sort per user and adds nothing.
    valid = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if valid.numel() > n_redundancy_items:
        idx = valid[torch.randperm(valid.numel(), generator=generator,
                                   device='cpu')[:n_redundancy_items].to(device)]
    else:
        idx = valid
    redundancy = {}
    names = [n for n in (BASE, SHARED, SPECIFIC) if n in channel_scores]
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            redundancy[f'{first}|{second}'] = _spearman(
                channel_scores[first][idx], channel_scores[second][idx])

    return {
        'user_id': user_id,
        'n_positives': len(positives),
        'quality': quality,
        'decision': decision,
        'total_abs_margin': absolute.sum().item(),
        'total_signed_margin': full_margin.sum().item(),
        'pair_accuracy': (full_margin > 0).float().mean().item(),
        'redundancy': redundancy,
    }


def summarise(results, topk=20):
    """Pool across users, then put tau and delta side by side.

    Pooled rather than averaged, for the same reason ``pooled_transfer_ratio``
    is: a mean over users weights a user whose channels nearly cancel the same
    as one carrying real contribution mass.
    """
    if not results:
        return {}

    channels = list(results[0]['decision'].keys())
    held = sum(r['n_positives'] for r in results)

    quality = {}
    for name in results[0]['quality']:
        hits = sum(r['quality'][name]['hits'] for r in results)
        aucs = [r['quality'][name]['auc'] for r in results
                if r['quality'][name]['auc'] is not None]
        quality[name] = {
            f'recall_at_{topk}': hits / held if held else None,
            f'ndcg_at_{topk}': float(np.mean([r['quality'][name]['ndcg'] for r in results])),
            'auc': float(np.mean(aucs)) if aucs else None,
        }

    total_abs = sum(r['total_abs_margin'] for r in results)
    total_signed = sum(r['total_signed_margin'] for r in results)
    absolute = {name: sum(r['decision'][name]['abs_margin'] for r in results)
                for name in channels}
    # tau's denominator excludes the raw collaborative channel, so a delta
    # normalised over all three is *not* comparable to it. Both normalisations
    # are reported, and only ``delta_disentangled`` may be read against tau.
    disentangled_total = sum(absolute[name] for name in (SHARED, SPECIFIC)
                             if name in absolute)

    decision = {}
    for name in channels:
        signed = sum(r['decision'][name]['signed_margin'] for r in results)
        decision[name] = {
            'delta': absolute[name] / total_abs if total_abs else None,
            'delta_disentangled': (absolute[name] / disentangled_total
                                   if disentangled_total and name in (SHARED, SPECIFIC)
                                   else None),
            'signed_share': signed / total_signed if total_signed else None,
            'vote_accuracy': float(np.mean([r['decision'][name]['vote_accuracy']
                                            for r in results])),
        }
    # How much of the decision the disentanglement accounts for at all, as
    # against the plain collaborative signal it was built on top of.
    decision['_disentangled_margin_share'] = (disentangled_total / total_abs
                                              if total_abs else None)

    redundancy = {key: float(np.mean([r['redundancy'][key] for r in results]))
                  for key in results[0]['redundancy']}

    return {
        'n_users': len(results),
        'n_positives': held,
        'quality': quality,
        'decision': decision,
        'redundancy': redundancy,
        'pair_accuracy': float(np.mean([r['pair_accuracy'] for r in results])),
    }
