"""Channel decomposition of DGCDR embeddings.

DGCDR fuses the disentangled preferences with an *additive* rule::

    e = e_gnn + a_c * e_common + a_s * e_specific

Because the fusion is additive, the final embedding is an exact sum of
independent *channels*.  The dot-product score therefore decomposes exactly
into one term per (user-channel, item-channel) pair -- no approximation, no
gradient estimate, no surrogate model.  This module recovers those channels.

Nothing here modifies the model: the graph propagation is re-composed from the
model's own public primitives (``get_ego_embeddings``, ``graph_layer``) and the
result is checked against ``model.forward()`` by :func:`verify_decomposition`.

Written against the defaults in ``properties/model/DGCDR.yaml``, which are also
the settings the paper reports: ``fuse_mode=attention``, ``attention_mode=all``,
``connect_way=concat``, ``feature_mapping_way=projection``,
``item_disentangle=True``, ``preference_disentangle=True``.
"""

import torch
import torch.nn.functional as F

# Channel names. BASE is the plain LightGCN embedding (no disentanglement),
# SHARED is the domain-common preference e^c, SPECIFIC is the domain-specific
# preference e^s.
BASE = 'base'
SHARED = 'shared'
SPECIFIC = 'specific'


class ChannelDecomposition:
    """Per-channel user and item embeddings for the target domain.

    Attributes:
        user_channels (dict): channel name -> tensor [total_num_users, D].
        item_channels (dict): channel name -> tensor [total_num_items, D].
        overlap_mask (Tensor): bool [total_num_users], True for users that were
            actually disentangled (only overlapping users are).
    """

    def __init__(self, user_channels, item_channels, overlap_mask):
        self.user_channels = user_channels
        self.item_channels = item_channels
        self.overlap_mask = overlap_mask

    @property
    def user_channel_names(self):
        return list(self.user_channels.keys())

    @property
    def item_channel_names(self):
        return list(self.item_channels.keys())

    def fused_user_embeddings(self):
        return sum(self.user_channels.values())

    def fused_item_embeddings(self):
        return sum(self.item_channels.values())


def _check_supported(model):
    """Fail loudly when the decomposition would not be exact."""
    if not model.preference_disentangle:
        raise ValueError(
            "Channel attribution requires preference_disentangle=True; the "
            "model has no disentangled channels to attribute to."
        )
    if model.fuse_mode != 'attention':
        raise NotImplementedError(
            f"Exact channel attribution requires fuse_mode='attention' (the "
            f"additive fusion), but the checkpoint uses fuse_mode="
            f"'{model.fuse_mode}'. With 'concat' the channels pass through an "
            f"MLP and the score is not additively separable; attributing it "
            f"would need an approximation (e.g. Shapley over the channels), "
            f"which defeats the exact-by-construction guarantee."
        )
    if model.overlapped_num_users <= 1:
        raise ValueError(
            "Channel attribution is only defined for overlapping users, and "
            "this dataset has none."
        )
    if model.training:
        raise RuntimeError(
            "model must be in eval() mode: dropout in graph_layer would make "
            "the decomposition non-deterministic."
        )


def _propagate(model):
    """Re-run the LightGCN propagation of ``DGCDR.forward`` on the target domain."""
    all_embeddings, norm_adj_matrix = model.get_ego_embeddings(domain='target')

    embeddings_list = [all_embeddings]
    for _ in range(model.n_layers):
        all_embeddings = model.graph_layer(norm_adj_matrix, all_embeddings)
        embeddings_list.append(all_embeddings)

    lightgcn_all_embeddings = torch.cat(embeddings_list, 1)
    return torch.split(lightgcn_all_embeddings,
                       [model.total_num_users, model.total_num_items])


def _encode(model, embeddings, is_user):
    """Split embeddings into (common, specific) exactly as ``disentangle_layer``."""
    if is_user:
        common_layers = model.target_en_common_layers
        specific_layers = model.target_en_specific_layers
    else:
        common_layers = model.target_en_item_common_layers
        specific_layers = model.target_en_item_specific_layers

    common = embeddings * torch.sigmoid(common_layers(embeddings))
    specific = embeddings * torch.sigmoid(specific_layers(embeddings))
    return common, specific


def _attention_channels(base, common, specific):
    """Reproduce the attention fusion, keeping the addends separate.

    Mirrors ``DGCDR.fuse_and_update``. The logits go into the softmax unscaled:
    Eq. (4) of the paper divides them by sqrt(d), the released implementation
    does not divide at all, and this module follows the implementation.
    """
    a_1 = torch.sum(torch.mul(base, common), dim=1)
    a_2 = torch.sum(torch.mul(base, specific), dim=1)

    att = torch.cat((a_1.unsqueeze(1), a_2.unsqueeze(1)), dim=1)
    softed_att = F.softmax(att, dim=1)

    e_c = softed_att[:, 0].unsqueeze(1) * common
    e_s = softed_att[:, 1].unsqueeze(1) * specific
    return e_c, e_s


def decompose_target_domain(model):
    """Decompose the target-domain embeddings into additive channels.

    The target domain is the one recommendations are served from, so it is the
    one being explained. Each domain's embedding table only covers its own
    items, so reading source items out of this decomposition would return
    untrained rows -- noise that clusters into equal sized groups and looks like
    structure.

    Returns:
        ChannelDecomposition
    """
    _check_supported(model)

    with torch.no_grad():
        target_user_e, target_item_e = _propagate(model)
        n_overlap = model.overlapped_num_users

        # ---- users -----------------------------------------------------
        # Only overlapping users go through the disentanglement; the rest keep
        # their plain LightGCN embedding (see DGCDR.disentangle_layer).
        target_overlap = target_user_e[:n_overlap]
        tg_common, tg_specific = _encode(model, target_overlap, is_user=True)
        e_c, e_s = _attention_channels(target_overlap, tg_common, tg_specific)

        user_shared = torch.zeros_like(target_user_e)
        user_specific = torch.zeros_like(target_user_e)
        user_shared[:n_overlap] = e_c
        user_specific[:n_overlap] = e_s

        user_channels = {BASE: target_user_e.clone(),
                         SHARED: user_shared,
                         SPECIFIC: user_specific}

        # ---- items -----------------------------------------------------
        it_common, it_specific = _encode(model, target_item_e, is_user=False)
        i_c, i_s = _attention_channels(target_item_e, it_common, it_specific)
        item_channels = {BASE: target_item_e.clone(), SHARED: i_c, SPECIFIC: i_s}

    overlap_mask = torch.zeros(model.total_num_users, dtype=torch.bool,
                               device=target_user_e.device)
    overlap_mask[:n_overlap] = True

    return ChannelDecomposition(user_channels, item_channels, overlap_mask)


def forward_outputs(model):
    """``model.forward()`` as a named dict, so callers do not index by position."""
    user_list, src_user, src_item, tgt_user, tgt_item = model.forward()
    return {'user_disentangled': user_list,
            'source_user': src_user, 'source_item': src_item,
            'target_user': tgt_user, 'target_item': tgt_item}


def verify_decomposition(model, decomposition, atol=1e-4):
    """Check that the channels sum back to what the model actually uses.

    This is the guarantee that makes the attribution *exact by construction*
    rather than a plausible-looking post-hoc story, so it is worth running (and
    reporting) on every checkpoint.

    Returns:
        dict with the max absolute reconstruction error on user embeddings,
        item embeddings and recommendation scores, plus a boolean ``passed``.
    """
    with torch.no_grad():
        outputs = forward_outputs(model)
        target_user_e = outputs['target_user']
        target_item_e = outputs['target_item']

        user_err = (decomposition.fused_user_embeddings() - target_user_e).abs().max().item()
        item_err = (decomposition.fused_item_embeddings() - target_item_e).abs().max().item()

        # Score-level check on a sample of users, against the model's own
        # scoring path rather than against the embeddings alone.
        n_users = min(64, model.overlapped_num_users)
        users = torch.arange(n_users, device=target_user_e.device)
        n_items = model.target_num_items

        reference = torch.matmul(target_user_e[users], target_item_e[:n_items].t())

        reconstructed = torch.zeros_like(reference)
        for u_emb in decomposition.user_channels.values():
            for i_emb in decomposition.item_channels.values():
                reconstructed += torch.matmul(u_emb[users], i_emb[:n_items].t())

        score_err = (reconstructed - reference).abs().max().item()
        score_scale = reference.abs().max().item()

    return {
        'user_embedding_max_abs_error': user_err,
        'item_embedding_max_abs_error': item_err,
        'score_max_abs_error': score_err,
        'score_max_abs_value': score_scale,
        'score_relative_error': score_err / score_scale if score_scale > 0 else 0.0,
        'passed': max(user_err, item_err, score_err) < atol,
        'atol': atol,
    }
