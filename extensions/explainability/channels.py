"""Channel decomposition of DGCDR embeddings.

DGCDR fuses the disentangled preferences with an *additive* rule when
``fuse_mode='attention'`` (see ``DGCDR.fuse_and_update``)::

    attention_mode='all'   ->  e = e_gnn + a_c * e_common + a_s * e_specific
    attention_mode='part'  ->  e =         a_c * e_common + a_s * e_specific

Because the fusion is additive, the final embedding is an exact sum of
independent *channels*.  The dot-product score therefore decomposes exactly
into one term per (user-channel, item-channel) pair -- no approximation, no
gradient estimate, no surrogate model.  This module recovers those channels.

Nothing here modifies the model: the graph propagation is re-composed from the
model's own public primitives (``get_ego_embeddings``, ``graph_layer``) and the
result is checked against ``model.forward()`` by :func:`verify_decomposition`.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Channel names. BASE is the plain LightGCN embedding (no disentanglement),
# SHARED is the domain-common preference e^c, SPECIFIC is the domain-specific
# preference e^s.
BASE = 'base'
SHARED = 'shared'
SPECIFIC = 'specific'


class ChannelDecomposition:
    """Per-channel user and item embeddings for one domain.

    Attributes:
        user_channels (dict): channel name -> tensor [total_num_users, D].
        item_channels (dict): channel name -> tensor [total_num_items, D].
        overlap_mask (Tensor): bool [total_num_users], True for users that were
            actually disentangled (only overlapping users are).
    """

    def __init__(self, user_channels, item_channels, overlap_mask, domain):
        self.user_channels = user_channels
        self.item_channels = item_channels
        self.overlap_mask = overlap_mask
        self.domain = domain

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


def _propagate(model, domain):
    """Re-run the LightGCN propagation of ``DGCDR.forward`` for one domain."""
    all_embeddings, norm_adj_matrix = model.get_ego_embeddings(domain=domain)

    embeddings_list = [all_embeddings]
    for _ in range(model.n_layers):
        all_embeddings = model.graph_layer(norm_adj_matrix, all_embeddings)
        embeddings_list.append(all_embeddings)

    if model.connect_way == 'concat':
        lightgcn_all_embeddings = torch.cat(embeddings_list, 1)
    elif model.connect_way == 'mean':
        lightgcn_all_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)
    else:
        raise NotImplementedError(f"unknown connect_way: {model.connect_way}")

    return torch.split(lightgcn_all_embeddings,
                       [model.total_num_users, model.total_num_items])


def _encode(model, embeddings, domain, is_user):
    """Split embeddings into (common, specific) exactly as ``disentangle_layer``."""
    # Only the layers matching feature_mapping_way exist on the model, so they
    # are resolved lazily inside each branch.
    if model.feature_mapping_way == 'projection':
        if is_user:
            common_layers = model.source_en_common_layers if domain == 'source' else model.target_en_common_layers
            specific_layers = model.source_en_specific_layers if domain == 'source' else model.target_en_specific_layers
        else:
            common_layers = model.source_en_item_common_layers if domain == 'source' else model.target_en_item_common_layers
            specific_layers = model.source_en_item_specific_layers if domain == 'source' else model.target_en_item_specific_layers
        common = embeddings * torch.sigmoid(common_layers(embeddings))
        specific = embeddings * torch.sigmoid(specific_layers(embeddings))
    elif model.feature_mapping_way == 'mlp':
        if is_user:
            mlp_layers = model.source_en_layers if domain == 'source' else model.target_en_layers
        else:
            mlp_layers = model.source_en_item_layers if domain == 'source' else model.target_en_item_layers
        common = mlp_layers(embeddings)
        specific = embeddings - common
    else:
        raise NotImplementedError(
            f"unknown feature_mapping_way: {model.feature_mapping_way}")
    return common, specific


def _attention_channels(model, base, common, specific):
    """Reproduce the attention fusion, keeping the addends separate.

    Mirrors ``DGCDR.fuse_and_update`` for ``fuse_mode='attention'``.
    """
    a_1 = torch.sum(torch.mul(base, common), dim=1)
    a_2 = torch.sum(torch.mul(base, specific), dim=1)

    # Ask the model for its own scale; a checkpoint decomposed with the wrong
    # denominator fails the reconstruction check.
    #
    # The fallback is 1.0, not sqrt(d), because that is what the *published*
    # DGCDR does: Eq. (4) of the paper divides the attention logits by sqrt(d),
    # and the released implementation does not divide at all --
    #
    #     att = torch.cat((b_1, b_2), dim=1)      # fuse_and_update, upstream
    #
    # This repository's copy added the division to match the equation, and
    # exposes softmax_scale to make it configurable. A model without that method
    # is therefore an unmodified upstream one, which applies no scaling.
    scale = model.softmax_scale(base.shape[-1]) if hasattr(model, 'softmax_scale') \
        else 1.0
    att = torch.cat((a_1.unsqueeze(1), a_2.unsqueeze(1)), dim=1) / scale
    softed_att = F.softmax(att, dim=1)

    e_c = softed_att[:, 0].unsqueeze(1) * common
    e_s = softed_att[:, 1].unsqueeze(1) * specific
    return e_c, e_s, softed_att


def decompose_target_domain(model):
    """Decompose the target-domain embeddings. See :func:`decompose_domain`."""
    return decompose_domain(model, 'target')


def decompose_domain(model, domain='target'):
    """Decompose one domain's embeddings into additive channels.

    The target domain is the one recommendations are served from, so it is the
    one Contribution 1 explains. The source domain is needed by the semantic
    audit, which compares the two.

    Which domain is used is not a detail: each domain's embedding table only
    covers its own items, and the rows for the other domain's items are left
    untrained. Reading source items out of the target decomposition returns
    vectors ~27x smaller than the real ones -- noise that clusters into equal
    sized groups and looks like structure.

    Returns:
        ChannelDecomposition
    """
    if domain not in ('source', 'target'):
        raise ValueError(f"domain must be 'source' or 'target', got '{domain}'")
    _check_supported(model)

    with torch.no_grad():
        target_user_e, target_item_e = _propagate(model, domain)

        n_overlap = model.overlapped_num_users

        # ---- users -----------------------------------------------------
        # Only overlapping users go through the disentanglement; the rest keep
        # their plain LightGCN embedding (see DGCDR.disentangle_layer).
        target_overlap = target_user_e[:n_overlap]
        tg_common, tg_specific = _encode(model, target_overlap, domain, is_user=True)
        e_c, e_s, user_att = _attention_channels(model, target_overlap, tg_common, tg_specific)

        user_shared = torch.zeros_like(target_user_e)
        user_specific = torch.zeros_like(target_user_e)
        user_shared[:n_overlap] = e_c
        user_specific[:n_overlap] = e_s

        user_base = target_user_e.clone()
        if model.attention_mode == 'part':
            # Overlapping users drop the raw GNN term; non-overlapping users
            # are untouched by the disentanglement and keep it.
            user_base[:n_overlap] = 0.0
        elif model.attention_mode != 'all':
            raise NotImplementedError(
                f"unknown attention_mode: {model.attention_mode}")

        user_channels = {BASE: user_base, SHARED: user_shared, SPECIFIC: user_specific}

        # ---- items -----------------------------------------------------
        if model.item_disentangle:
            it_common, it_specific = _encode(model, target_item_e, domain, is_user=False)
            i_c, i_s, _ = _attention_channels(model, target_item_e, it_common, it_specific)

            item_base = torch.zeros_like(target_item_e) if model.attention_mode == 'part' \
                else target_item_e.clone()
            item_channels = {BASE: item_base, SHARED: i_c, SPECIFIC: i_s}
        elif model.item_mapping:
            mapped = target_item_e * torch.sigmoid(getattr(model, f'{domain}_item_mapping_layer')(target_item_e))
            item_channels = {BASE: mapped}
        else:
            item_channels = {BASE: target_item_e}

    overlap_mask = torch.zeros(model.total_num_users, dtype=torch.bool,
                               device=target_user_e.device)
    overlap_mask[:n_overlap] = True

    decomposition = ChannelDecomposition(user_channels, item_channels, overlap_mask, domain)
    decomposition.user_attention = user_att
    return decomposition


def forward_outputs(model):
    """``model.forward()`` as a named dict, whatever arity it returns.

    The published DGCDR returns five tensors; this repository's copy returns
    six, having inserted ``item_disentangled_list`` in second position. Reading
    the outputs by index therefore breaks silently when the module is pointed at
    the other tree -- silently, because the shapes still line up and only the
    meaning shifts. Resolving by length once, here, keeps every caller correct
    on both and means the model file needs no modification at all.

    Returns:
        dict with ``user_disentangled``, ``source_user``, ``source_item``,
        ``target_user``, ``target_item``, and ``item_disentangled`` (None on the
        five-tensor signature, which does not expose it).
    """
    outputs = model.forward()
    if len(outputs) == 6:
        user_list, item_list, src_user, src_item, tgt_user, tgt_item = outputs
    elif len(outputs) == 5:
        user_list, src_user, src_item, tgt_user, tgt_item = outputs
        item_list = None
    else:
        raise RuntimeError(
            f"model.forward() returned {len(outputs)} tensors; this module knows "
            f"the 5-tensor (published) and 6-tensor signatures. Add the new one "
            f"here rather than indexing into it at the call sites.")
    return {'user_disentangled': user_list, 'item_disentangled': item_list,
            'source_user': src_user, 'source_item': src_item,
            'target_user': tgt_user, 'target_item': tgt_item}


def user_fusion_parts(model, domain='target'):
    """The tensors Eq. (5) fuses, kept *before* the attention weights hit them.

    ``decompose_domain`` returns the channels already weighted, which is what an
    attribution needs. Auditing the fusion itself needs the opposite: the raw
    addends plus the weights, so a caller can re-fuse them under a different
    attention -- constant, permuted -- without touching a single parameter.

    Returns:
        dict with ``base``/``common``/``specific`` for the overlapping users,
        the ``attention`` the model computed for them, and ``all_users`` (the
        full propagated table, so non-overlapping rows can be carried over).
    """
    _check_supported(model)
    with torch.no_grad():
        user_e, item_e = _propagate(model, domain)
        n_overlap = model.overlapped_num_users
        base = user_e[:n_overlap]
        common, specific = _encode(model, base, domain, is_user=True)
        _, _, attention = _attention_channels(model, base, common, specific)
    return {'base': base, 'common': common, 'specific': specific,
            'attention': attention, 'all_users': user_e, 'items': item_e,
            'n_overlap': n_overlap}


def refuse_users(model, parts, attention):
    """Re-run Eq. (5) with the attention weights replaced.

    Non-overlapping users never enter the disentanglement, so their rows are
    carried over untouched -- swapping them in would change what is being
    measured from "does personalised fusion matter" to "does fusion matter".
    """
    n_overlap = parts['n_overlap']
    fused = parts['all_users'].clone()
    overlap = (attention[:, 0].unsqueeze(1) * parts['common']
               + attention[:, 1].unsqueeze(1) * parts['specific'])
    if model.attention_mode == 'all':
        overlap = overlap + parts['base']
    elif model.attention_mode != 'part':
        raise NotImplementedError(f"unknown attention_mode: {model.attention_mode}")
    fused[:n_overlap] = overlap
    return fused


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
        side = 'source' if getattr(decomposition, 'domain', 'target') == 'source' else 'target'
        target_user_e = outputs[f'{side}_user']
        target_item_e = outputs[f'{side}_item']

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
