"""Is DGCDR's personalised fusion personal?

Table 5 of the paper reports that replacing the attention fusion of Eq. (5) with
direct concatenation (the ``-Pers`` variant) costs 22.69% on average. It is by a
wide margin the largest ablation effect in the paper, and it carries the claim
that fusion has to be tailored to the individual user -- "some users may
prioritize domain-shared attributes ... while others may focus on
domain-specific attributes".

That ablation moves two things at once:

  (a) the weights stop being per-user, and
  (b) an additive fusion of three d-dimensional vectors is replaced by a
      concatenation into 3d followed by an MLP.

Only (a) is what the personalisation claim is about. This module isolates it,
and needs no retraining: the attention weights of a trained checkpoint are
replaced at inference by

  * ``constant``  -- every user gets the same weights (the checkpoint's own
    mean, so the fusion stays on the scale the parameters were trained for);
  * ``permuted``  -- every user gets *another user's* weights.

Architecture, parameters and dimensions are untouched in both. ``permuted`` is
the sharper of the two: it keeps the exact multiset of weights the model
produced and destroys only the pairing. If personalisation is what carries the
22.69%, handing a user somebody else's fusion weights has to hurt.

The claim is about users, so the audit is about users: item-side attention is
left exactly as the model computed it.
"""

import numpy as np
import torch

from extensions.explainability.channels import (
    forward_outputs,
    refuse_users,
    user_fusion_parts,
)


def _dcg(gains):
    positions = np.arange(2, gains.size + 2)
    return float((gains / np.log2(positions)).sum())


def evaluate(model, user_embeddings, item_embeddings, users, ground_truth, topk=20):
    """Recall@k and NDCG@k over sampled users, masking each user's history.

    Deliberately the same sampled-user protocol as the counterfactual module
    rather than RecBole's full evaluation: every number this module reports is a
    *comparison* between fusion variants on identical users and identical
    masking, and that comparison is what has to be trustworthy. The absolute
    level is not comparable to the paper's table, and is not used as if it were.
    """
    n_items = model.target_num_items
    inter = model.target_interaction_matrix
    item_emb = item_embeddings[:n_items]

    hits, held, ndcgs, rankings, per_user = 0, 0, [], {}, {}
    for user_id in users:
        truth = set(ground_truth.get(user_id, []))
        if not truth:
            continue
        with torch.no_grad():
            scores = torch.matmul(item_emb, user_embeddings[user_id])
        history = [int(i) for i in inter.col[inter.row == user_id] if int(i) < n_items]
        if history:
            scores[torch.as_tensor(history, dtype=torch.long, device=scores.device)] = float('-inf')
        scores[0] = float('-inf')  # [PAD]

        top = torch.topk(scores, k=min(topk, n_items - 1)).indices
        top = [int(i) for i in top]
        rankings[user_id] = top

        gains = np.array([1.0 if i in truth else 0.0 for i in top])
        ideal = np.zeros_like(gains)
        ideal[:min(len(truth), gains.size)] = 1.0

        hits += int(gains.sum())
        held += len(truth)
        per_user[user_id] = (int(gains.sum()), len(truth))
        ndcgs.append(_dcg(gains) / _dcg(ideal) if ideal.sum() > 0 else 0.0)

    return {
        'n_users_scored': len(ndcgs),
        f'recall_at_{topk}': hits / held if held else None,
        f'ndcg_at_{topk}': float(np.mean(ndcgs)) if ndcgs else None,
        '_rankings': rankings,
        '_per_user': per_user,
    }


def _paired_bootstrap(variant, intact, n_resamples=2000, seed=42):
    """95% CI on the relative Recall difference, resampling users.

    The differences this module measures sit in the same few-percent band the
    channel ablations do, and that band is where this project has already found
    noise. Reading a sign off a point estimate there is not safe; users are the
    independent unit, so they are what gets resampled, and the pairing is kept
    so the comparison stays within-user.
    """
    shared_users = [u for u in variant if u in intact]
    if not shared_users:
        return None
    variant_hits = np.array([variant[u][0] for u in shared_users], dtype=float)
    intact_hits = np.array([intact[u][0] for u in shared_users], dtype=float)
    held = np.array([intact[u][1] for u in shared_users], dtype=float)

    def relative(sample):
        denominator = held[sample].sum()
        if denominator == 0:
            return 0.0
        base = intact_hits[sample].sum() / denominator
        return (variant_hits[sample].sum() / denominator - base) / base if base else 0.0

    rng = np.random.default_rng(seed)
    n = len(shared_users)
    draws = [relative(rng.integers(0, n, n)) for _ in range(n_resamples)]

    observed = relative(np.arange(n))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        'relative_change': observed,
        'ci_low': float(low),
        'ci_high': float(high),
        # A CI straddling zero means the sign of the point estimate is not
        # established, which is the only thing that may be reported.
        'significant': bool(low > 0 or high < 0),
    }


def attention_variants(parts, seed=42):
    """The attention matrices to compare against the one the model computed."""
    attention = parts['attention']
    n = attention.shape[0]

    constant = attention.mean(dim=0, keepdim=True).repeat(n, 1)

    generator = torch.Generator().manual_seed(seed)
    permuted = attention[torch.randperm(n, generator=generator).to(attention.device)]

    return {'intact': attention, 'constant': constant, 'permuted': permuted}


def run(model, users, ground_truth, topk=20, seed=42, domain='target'):
    """Evaluate every fusion variant on the same users.

    Also reports how much of each user's top-k survives the swap. Accuracy can
    stay flat while the list churns underneath, and the two say different
    things: the first is whether personalisation matters to the metric, the
    second whether it moves the recommendations at all.
    """
    parts = user_fusion_parts(model, domain)
    item_embeddings = forward_outputs(model)[f'{domain}_item'].detach()

    variants = attention_variants(parts, seed)
    results, reference, intact_per_user = {}, None, None
    for name, attention in variants.items():
        user_embeddings = refuse_users(model, parts, attention)
        outcome = evaluate(model, user_embeddings, item_embeddings, users,
                           ground_truth, topk)
        rankings = outcome.pop('_rankings')
        per_user = outcome.pop('_per_user')
        if name == 'intact':
            reference, intact_per_user = rankings, per_user
            outcome['mean_topk_overlap'] = 1.0
        else:
            overlaps = [len(set(rankings[u]) & set(reference[u])) / len(reference[u])
                        for u in rankings if u in reference and reference[u]]
            outcome['mean_topk_overlap'] = float(np.mean(overlaps)) if overlaps else None
            outcome['vs_intact'] = _paired_bootstrap(per_user, intact_per_user, seed=seed)
        results[name] = outcome

    attention = parts['attention']
    results['attention_stats'] = {
        'mean_shared_weight': attention[:, 0].mean().item(),
        'std_shared_weight': attention[:, 0].std().item(),
        'min_shared_weight': attention[:, 0].min().item(),
        'max_shared_weight': attention[:, 0].max().item(),
    }
    return results
