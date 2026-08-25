"""Exact channel-level attribution for DGCDR recommendations.

Contribution 1 of the explainability framework: decompose the recommendation
score into the disentangled channels the model itself uses (domain-shared vs
domain-specific), exactly and without any post-hoc approximation.
"""

from extensions.explainability.decomposition.core.channels import (
    ChannelDecomposition,
    decompose_target_domain,
    verify_decomposition,
)
from extensions.explainability.decomposition.core.attribution import (
    ItemAttribution,
    UserExplanation,
    attribute_scores,
    explain_user,
)

__all__ = [
    'ChannelDecomposition',
    'decompose_target_domain',
    'verify_decomposition',
    'ItemAttribution',
    'UserExplanation',
    'attribute_scores',
    'explain_user',
]
