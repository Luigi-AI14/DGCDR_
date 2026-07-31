"""Exact attribution of DGCDR recommendation scores to disentangled channels.

Given the additive channels recovered in :mod:`channels`, the score of a
(user, item) pair splits exactly::

    score(u, i) = <sum_k U_k[u], sum_l I_l[i]> = sum_{k,l} <U_k[u], I_l[i]>

so every recommendation comes with a (user-channel x item-channel) matrix of
signed contributions that provably sums to the score the model computed.

From that matrix we derive the **transfer ratio** tau(u, i): how much of the
recommendation is carried by knowledge imported from the source domain, as
opposed to preferences native to the target domain.
"""

import torch

from extensions.explainability.channels import BASE, SHARED, SPECIFIC


class ItemAttribution:
    """Attribution of a single (user, item) score.

    Attributes:
        item_id (int): internal target-domain item id.
        score (float): the model's score, equal to the sum of ``matrix``.
        matrix (dict): user_channel -> item_channel -> signed contribution.
        user_channel_totals (dict): contribution aggregated per user channel.
        item_channel_totals (dict): contribution aggregated per item channel.
        tau (float): transfer ratio in [0, 1] over the disentangled channels.
        tau_full (float): transfer ratio including the raw GNN channel.
        magnitude (float): |C_shared| + |C_specific|, the denominator of tau.
            tau is a ratio of signed sums, so it becomes numerically
            meaningless when this is close to zero -- see :func:`is_reliable`.
        rank (int): position in the user's recommendation list (1-based).
    """

    def __init__(self, item_id, score, matrix, user_channel_totals,
                 item_channel_totals, tau, tau_full, magnitude, rank=None):
        self.item_id = item_id
        self.score = score
        self.matrix = matrix
        self.user_channel_totals = user_channel_totals
        self.item_channel_totals = item_channel_totals
        self.tau = tau
        self.tau_full = tau_full
        self.magnitude = magnitude
        self.rank = rank
        # True when the item is in the user's held-out set, i.e. the
        # recommendation was a hit. Set by explain_user.
        self.relevant = None

    @property
    def dominant_channel(self):
        """Which disentangled user channel drives this recommendation."""
        shared = abs(self.user_channel_totals.get(SHARED, 0.0))
        specific = abs(self.user_channel_totals.get(SPECIFIC, 0.0))
        return SHARED if shared >= specific else SPECIFIC

    def to_dict(self):
        return {
            'item_id': self.item_id,
            'rank': self.rank,
            'relevant': self.relevant,
            'score': self.score,
            'transfer_ratio': self.tau,
            'transfer_ratio_full': self.tau_full,
            'magnitude': self.magnitude,
            'dominant_channel': self.dominant_channel,
            'user_channel_totals': self.user_channel_totals,
            'item_channel_totals': self.item_channel_totals,
            'contribution_matrix': self.matrix,
        }


class UserExplanation:
    """All attributions for one user's recommendation list."""

    def __init__(self, user_id, user_token, attributions, is_overlapping,
                 ground_truth=None, history=None, source_history=None):
        self.user_id = user_id
        self.user_token = user_token
        self.attributions = attributions
        self.is_overlapping = is_overlapping
        self.ground_truth = ground_truth or []
        self.history = history or []
        self.source_history = source_history or []

    @property
    def mean_tau(self):
        if not self.attributions:
            return 0.0
        return sum(a.tau for a in self.attributions) / len(self.attributions)

    @property
    def n_hits(self):
        return sum(1 for a in self.attributions if a.relevant)

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'user_token': self.user_token,
            'is_overlapping': self.is_overlapping,
            'mean_transfer_ratio': self.mean_tau,
            'n_hits': self.n_hits,
            'ground_truth_items': self.ground_truth,
            'history_items': self.history,
            'source_history_items': self.source_history,
            'recommendations': [a.to_dict() for a in self.attributions],
        }


def _transfer_ratios(user_totals):
    """Turn signed channel totals into bounded transfer ratios.

    Contributions are signed (a channel can push a score *down*), so a plain
    share ``shared / (shared + specific)`` is unbounded and flips sign near the
    denominator's zero.  We normalise by absolute magnitude instead, which is
    the standard convention for signed attributions and keeps tau in [0, 1]:

        tau      = |C_shared| / (|C_shared| + |C_specific|)
        tau_full = |C_shared| / (|C_base| + |C_shared| + |C_specific|)

    ``tau`` answers "of the disentangled preference, how much is transferred?".
    ``tau_full`` additionally discounts for the part of the score that the raw
    collaborative signal explains on its own.  Both are reported because they
    answer different questions; the signed totals are kept in the output so any
    other normalisation can be recomputed downstream.
    """
    shared = abs(user_totals.get(SHARED, 0.0))
    specific = abs(user_totals.get(SPECIFIC, 0.0))
    base = abs(user_totals.get(BASE, 0.0))

    denom = shared + specific
    tau = shared / denom if denom > 0 else 0.0

    denom_full = shared + specific + base
    tau_full = shared / denom_full if denom_full > 0 else 0.0
    return tau, tau_full, denom


def pooled_transfer_ratio(attributions):
    """Aggregate tau over many recommendations without averaging ratios.

    A mean of per-item ratios weights an item whose channels nearly cancel the
    same as one carrying the bulk of the score.  Pooling the numerators and
    denominators instead answers the population-level question directly -- of
    all the disentangled contribution mass, how much is transferred -- and is
    the aggregate that should be reported.
    """
    numerator = sum(abs(a.user_channel_totals.get(SHARED, 0.0)) for a in attributions)
    denominator = sum(a.magnitude for a in attributions)
    return numerator / denominator if denominator > 0 else 0.0


def reliable_attributions(attributions, min_magnitude_ratio=0.1):
    """Drop attributions whose tau is dominated by numerical noise.

    tau is a ratio of signed sums: when ``magnitude`` approaches zero the two
    channels nearly cancel and tau swings across the whole [0, 1] range without
    meaning anything.  On a real catalogue this affects low-scoring items, so
    tau statistics must be reported over the items that carry actual
    contribution mass.  The threshold is relative to the median magnitude of
    the set, so it adapts to the scale of the checkpoint.
    """
    if not attributions:
        return [], 0.0
    magnitudes = sorted(a.magnitude for a in attributions)
    median = magnitudes[len(magnitudes) // 2]
    threshold = median * min_magnitude_ratio
    return [a for a in attributions if a.magnitude >= threshold], threshold


def attribute_scores(decomposition, user_id, item_ids):
    """Compute the exact contribution matrix for one user over a set of items.

    Args:
        decomposition (ChannelDecomposition): from ``decompose_target_domain``.
        user_id (int): internal user id.
        item_ids (Tensor|list[int]): internal target-domain item ids.

    Returns:
        list[ItemAttribution], ordered as ``item_ids``.
    """
    if not torch.is_tensor(item_ids):
        item_ids = torch.as_tensor(item_ids, dtype=torch.long)
    device = next(iter(decomposition.user_channels.values())).device
    item_ids = item_ids.to(device)

    user_names = decomposition.user_channel_names
    item_names = decomposition.item_channel_names

    # contributions[k][l] -> [n_items]
    contributions = {}
    with torch.no_grad():
        for k in user_names:
            u_vec = decomposition.user_channels[k][user_id]
            contributions[k] = {}
            for l in item_names:
                i_mat = decomposition.item_channels[l][item_ids]
                contributions[k][l] = torch.matmul(i_mat, u_vec)

    results = []
    for idx in range(item_ids.shape[0]):
        matrix = {k: {l: contributions[k][l][idx].item() for l in item_names}
                  for k in user_names}
        user_totals = {k: sum(matrix[k].values()) for k in user_names}
        item_totals = {l: sum(matrix[k][l] for k in user_names) for l in item_names}
        score = sum(user_totals.values())
        tau, tau_full, magnitude = _transfer_ratios(user_totals)

        results.append(ItemAttribution(
            item_id=int(item_ids[idx].item()),
            score=score,
            matrix=matrix,
            user_channel_totals=user_totals,
            item_channel_totals=item_totals,
            tau=tau,
            tau_full=tau_full,
            magnitude=magnitude,
        ))
    return results


def recommend(model, decomposition, user_id, topk=10, mask_history=True):
    """Score every target-domain item for a user and return the top-k.

    Returns:
        (item_ids (list[int]), history (list[int]))
    """
    n_items = model.target_num_items
    user_emb = decomposition.fused_user_embeddings()[user_id]
    item_emb = decomposition.fused_item_embeddings()[:n_items]

    with torch.no_grad():
        scores = torch.matmul(item_emb, user_emb)

    history = []
    if mask_history:
        inter = model.target_interaction_matrix
        history = [int(i) for i in inter.col[inter.row == user_id] if int(i) < n_items]
        if history:
            scores[torch.as_tensor(history, dtype=torch.long, device=scores.device)] = float('-inf')
    scores[0] = float('-inf')  # RecBole reserves internal id 0 as [PAD]

    top_items = torch.topk(scores, k=min(topk, n_items - 1)).indices
    return [int(i) for i in top_items], history


def explain_user(model, decomposition, user_id, topk=10, mask_history=True,
                 user_token=None, ground_truth=None, source_history=None):
    """Produce the full attributed recommendation list for one user."""
    item_ids, history = recommend(model, decomposition, user_id, topk, mask_history)
    attributions = attribute_scores(decomposition, user_id, item_ids)

    held_out = set(ground_truth or [])
    for rank, attribution in enumerate(attributions, start=1):
        attribution.rank = rank
        attribution.relevant = attribution.item_id in held_out

    return UserExplanation(
        user_id=user_id,
        user_token=user_token,
        attributions=attributions,
        is_overlapping=bool(decomposition.overlap_mask[user_id].item()),
        ground_truth=ground_truth,
        history=history,
        source_history=source_history,
    )
