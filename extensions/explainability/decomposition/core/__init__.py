"""The decomposition machinery the scripts in the parent package call.

Every module here rests on one property: DGCDR fuses its disentangled
preferences additively, so a score is an exact sum of per-channel terms. What
differs is the question asked of that sum.

    channels        recovers the channels from a loaded checkpoint
    attribution     splits a score across them (tau: share of magnitude)
    discrimination  splits a ranking margin across them (delta: share of decision)
    counterfactual  removes a channel and re-ranks, which is the causal question
    data            users, held-out items and histories, read from the checkpoint
"""
