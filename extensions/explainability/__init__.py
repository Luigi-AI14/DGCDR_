"""Exact channel-level attribution for DGCDR recommendations.

Contribution 1 of the explainability framework: decompose the recommendation
score into the disentangled channels the model itself uses (domain-shared vs
domain-specific), exactly and without any post-hoc approximation.
"""

from extensions.explainability.channels import (
    ChannelDecomposition,
    decompose_target_domain,
    verify_decomposition,
)
from extensions.explainability.attribution import (
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
