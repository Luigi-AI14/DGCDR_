"""Audit of the contrastive alignment decoder -- DGCDR's own claimed novelty.

The paper's second contribution (Sec. 3.3) is anchor-based supervision. The
GNN-enhanced embedding of one domain is used as an anchor, and the decoder is
trained so that the *other* domain's decoded features order themselves as

    cos(anchor_u, e^c_u)  >  cos(anchor_u, e^g_u)  >  cos(anchor_u, e^s_u)

(``decoder_loss_function`` L2-normalises every term, so these really are cosines
and the norm shortcut of the encoder loss does not apply here.)

The published evidence for the mechanism is one ablation on accuracy: removing
the decoder loss costs 2.66%. Two questions were never asked, and neither needs
retraining.

1. **Does the ordering hold at convergence?** The loss pushes towards it; that
   does not mean it is reached.
2. **If it holds, does it hold for *this* user?** This is the one the claim
   rests on. The loss is written per user, so it reads as aligning a user's
   shared preferences across domains. But nothing in the objective makes the
   alignment personal: an ordering satisfied by every ``(anchor_u, decoded_v)``
   pair, ``u != v`` included, minimises the loss exactly as well while carrying
   no information about *u*. Separating the two costs one permutation.

The module then asks the question the alignment claim ultimately reduces to:
**can the shared channel identify a user across domains at all?** If a user's
target-domain ``e^c`` does not retrieve their own source-domain ``e^c`` better
than the raw GNN embedding already does, then the alignment objectives -- the
encoder's ``dis(e^c_A, e^c_B)`` and the decoder's hierarchy -- bought nothing
that graph propagation had not already provided. That comparison is the point of
:func:`identification`, and the raw GNN channel is the control it needs.
"""

import numpy as np
import torch
import torch.nn.functional as F

from extensions.explainability.channels import forward_outputs

# Order of ``DGCDR.disentangle_layer``'s user return list, which ``forward``
# passes through unchanged. Hard-coding the positions is unavoidable (the model
# returns a bare list) so they are named once, here.
_USER_PARTS = (
    'source_common', 'target_common', 'source_specific', 'target_specific',
    'source_gnn', 'target_gnn',
    'source_decoded_gnn', 'source_decoded_common', 'source_decoded_specific',
    'target_decoded_gnn', 'target_decoded_common', 'target_decoded_specific',
)


def decoder_views(model):
    """Recover the exact tensors the decoder loss is computed on.

    Returns:
        dict: every entry of ``_USER_PARTS``, restricted to overlapping users,
        detached on the CPU.
    """
    if not model.preference_disentangle:
        raise ValueError("the checkpoint has no disentangled channels to audit")
    if model.training:
        raise RuntimeError("model must be in eval() mode: dropout would make "
                           "the audit non-deterministic")

    with torch.no_grad():
        user_list = forward_outputs(model)['user_disentangled']
    if len(user_list) != len(_USER_PARTS):
        raise RuntimeError(
            f"expected {len(_USER_PARTS)} user tensors from forward(), got "
            f"{len(user_list)}; the model's return signature changed and the "
            f"positional mapping in this module is no longer valid.")
    return {name: tensor.detach().cpu() for name, tensor in zip(_USER_PARTS, user_list)}


def _cos(a, b):
    return F.cosine_similarity(a, b, dim=1)


def hierarchy(anchor, decoded_common, decoded_gnn, decoded_specific, permutation=None):
    """Measure the ordering the decoder loss is trained to produce.

    Args:
        permutation: if given, the decoded features are read for a *different*
            user than the anchor. This is the null that separates "the ordering
            holds" from "the ordering holds for this user": under it the loss is
            still satisfiable, but nothing personal is being aligned.
    """
    if permutation is not None:
        decoded_common = decoded_common[permutation]
        decoded_gnn = decoded_gnn[permutation]
        decoded_specific = decoded_specific[permutation]

    m_c = _cos(anchor, decoded_common)
    m_g = _cos(anchor, decoded_gnn)
    m_s = _cos(anchor, decoded_specific)

    return {
        'cos_shared': m_c.mean().item(),
        'cos_gnn': m_g.mean().item(),
        'cos_specific': m_s.mean().item(),
        # The two pairwise comparisons the two loss terms encode, plus the
        # full three-way ordering they jointly ask for.
        'rate_shared_over_gnn': (m_c > m_g).float().mean().item(),
        'rate_gnn_over_specific': (m_g > m_s).float().mean().item(),
        'rate_full_ordering': ((m_c > m_g) & (m_g > m_s)).float().mean().item(),
        'margin_shared_over_gnn': (m_c - m_g).mean().item(),
        'margin_gnn_over_specific': (m_g - m_s).mean().item(),
    }


def hierarchy_with_null(model, n_permutations=5, seed=42):
    """The ordering, paired and permuted, for both anchor directions.

    The gap between the two is the only part of the result that says the
    supervision did something *per user*. The paired numbers on their own do
    not: they are equally consistent with an ordering that is a property of the
    channels rather than of the pairing.
    """
    parts = decoder_views(model)
    generator = torch.Generator().manual_seed(seed)

    results = {}
    for anchor_domain, decoded_domain in (('source', 'target'), ('target', 'source')):
        anchor = parts[f'{anchor_domain}_gnn']
        dec_c = parts[f'{decoded_domain}_decoded_common']
        dec_g = parts[f'{decoded_domain}_decoded_gnn']
        dec_s = parts[f'{decoded_domain}_decoded_specific']

        paired = hierarchy(anchor, dec_c, dec_g, dec_s)

        nulls = []
        for _ in range(n_permutations):
            perm = torch.randperm(anchor.shape[0], generator=generator)
            nulls.append(hierarchy(anchor, dec_c, dec_g, dec_s, permutation=perm))

        null = {k: float(np.mean([n[k] for n in nulls])) for k in nulls[0]}
        null_sd = {k: float(np.std([n[k] for n in nulls])) for k in nulls[0]}

        results[f'anchor_{anchor_domain}'] = {
            'n_users': int(anchor.shape[0]),
            'paired': paired,
            'null': null,
            'null_sd': null_sd,
            'gap': {k: paired[k] - null[k] for k in paired},
        }
    return results


def identification(source, target, sample_size=2000, seed=42):
    """Can this channel match a user to themselves across domains?

    Cross-domain alignment, stated as a measurable claim: user *u*'s vector in
    one domain should be closest to user *u*'s vector in the other, out of all
    candidates. Retrieval rank measures exactly that, and unlike a mean cosine
    it has a chance level to read against.

    Returns:
        dict with hit@1, hit@10, MRR, the median rank, and the chance values
        for the sampled candidate-set size.
    """
    n = min(source.shape[0], target.shape[0])
    generator = torch.Generator().manual_seed(seed)
    idx = torch.arange(1, n)  # skip the [PAD] row
    if sample_size and idx.numel() > sample_size:
        idx = idx[torch.randperm(idx.numel(), generator=generator)[:sample_size]]

    a = F.normalize(source[idx], dim=1)
    b = F.normalize(target[idx], dim=1)
    n_cand = idx.numel()

    similarity = a @ b.t()
    true_scores = similarity.diagonal().unsqueeze(1)
    # Rank of the true partner: how many candidates score at least as high.
    # Ties count against the model, which is the conservative direction and
    # matters on collapsed checkpoints where many scores coincide.
    ranks = (similarity >= true_scores).sum(dim=1).float()

    return {
        'n_candidates': int(n_cand),
        'hit_at_1': (ranks <= 1).float().mean().item(),
        'hit_at_10': (ranks <= 10).float().mean().item(),
        'mrr': (1.0 / ranks).mean().item(),
        'median_rank': ranks.median().item(),
        'chance_hit_at_1': 1.0 / n_cand,
        'chance_hit_at_10': min(10.0 / n_cand, 1.0),
        'chance_median_rank': (n_cand + 1) / 2.0,
    }


def identification_by_channel(model, sample_size=2000, seed=42):
    """Cross-domain user identification, per channel.

    ``gnn`` is the control. It is what the model has before any alignment
    objective is applied, so a shared channel that does not beat it has not been
    aligned by the losses -- it has inherited whatever the graph already put
    there. ``specific`` is the opposite control: by design it should identify a
    user *worse*, since it is meant to hold what does not transfer.
    """
    parts = decoder_views(model)
    channels = {
        'gnn': ('source_gnn', 'target_gnn'),
        'shared': ('source_common', 'target_common'),
        'specific': ('source_specific', 'target_specific'),
        'shared_decoded': ('source_decoded_common', 'target_decoded_common'),
    }
    return {name: identification(parts[src], parts[tgt], sample_size, seed)
            for name, (src, tgt) in channels.items()}
