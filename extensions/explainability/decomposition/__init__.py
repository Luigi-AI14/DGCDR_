"""Channel decomposition of DGCDR scores, and what can be measured from it.

``core`` holds the machinery: the additive channels, the score attribution
(tau), the ranking-margin split (delta) and the channel ablations. This level
holds what is run against it -- the measurement scripts and the self-test
that checks the decomposition against ``model.forward()``.

The scripts import by absolute path, so run them from the repository root:

    python -m extensions.explainability.decomposition.run_discrimination -m saved/<ckpt>.pth
    python -m extensions.explainability.decomposition.test_decomposition -m saved/<ckpt>.pth
"""
