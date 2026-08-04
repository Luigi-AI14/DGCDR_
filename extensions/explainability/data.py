"""Reading users, held-out items and histories out of a loaded checkpoint.

These helpers touch the dataset rather than the model's channels, so they live
apart from the attribution itself, and keeping them here is what lets a
measurement script depend on the package alone.
"""

import random
from collections import defaultdict


def build_ground_truth(test_data):
    """user id -> list of held-out target-domain item ids.

    "Relevant" is not a score the model computes: it is an interaction the split
    hid from training. Every row of the test split counts, whatever rating it
    carried -- the implicit-feedback convention RecBole applies here.
    """
    ground_truth = defaultdict(list)
    try:
        dataset = test_data.dataset
        uid_field, iid_field = dataset.uid_field, dataset.iid_field
        users = dataset.inter_feat[uid_field].numpy()
        items = dataset.inter_feat[iid_field].numpy()
        for u, i in zip(users, items):
            ground_truth[int(u)].append(int(i))
    except (AttributeError, KeyError) as exc:
        print(f"[warn] could not read test interactions ({exc}); "
              f"explanations will have no ground truth.")
    return ground_truth


def select_users(model, num_users, seed=42):
    """Sample overlapping users -- the only ones with a defined transfer ratio."""
    candidates = list(range(1, model.overlapped_num_users))  # skip [PAD]
    if num_users and num_users < len(candidates):
        random.Random(seed).shuffle(candidates)
        candidates = sorted(candidates[:num_users])
    return candidates


def source_history(model, user_id, limit=None):
    """The user's interactions in the *source* domain.

    Target- and source-domain user ids share one id space, so the same index
    reads both.
    """
    inter = model.source_interaction_matrix
    items = [int(i) for i in inter.col[inter.row == user_id]]
    return items[:limit] if limit else items
