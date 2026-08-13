"""Self-test for the channel decomposition.

Checks, against a real checkpoint, that:
  1. the channels sum back to the model's own forward() outputs;
  2. the contribution matrix sums to the score of each attributed item;
  3. the guard rails refuse the configurations where the decomposition would
     not be exact, instead of silently returning an approximation.

Run it after touching anything in this module, or on a new checkpoint before
trusting the attributions it produces:

    python -m extensions.explainability.decomposition.test_decomposition -m saved/<ckpt>.pth
"""

import argparse
import os
import sys

# This script lives three levels below the repository root, so running it by
# path puts its own directory on sys.path instead of the root, and neither
# recbole_cdr nor extensions resolves. Put the root back before importing them.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     os.pardir, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.decomposition.core.attribution import explain_user
from extensions.explainability.decomposition.core.channels import (
    decompose_target_domain,
    verify_decomposition,
)

# The score is a sum of float32 dot products, so exact bitwise equality is not
# achievable; anything above this is a real bug, not rounding.
TOLERANCE = 1e-4


def _check_raises(label, expected, fn, results):
    try:
        fn()
    except expected as exc:
        print(f"  [ok] {label:36s} refused: {str(exc).splitlines()[0][:50]}...")
        results.append(True)
        return
    print(f"  [FAIL] {label:36s} should have raised {expected.__name__}")
    results.append(False)


def main():
    parser = argparse.ArgumentParser(description="Self-test the channel decomposition")
    parser.add_argument('--model_path', '-m', type=str, required=True,
                        help="Path to a .pth checkpoint trained with "
                             "preference_disentangle=True and fuse_mode=attention")
    args = parser.parse_args()

    config, model, dataset, _, _, _ = load_data_and_model(args.model_path)
    model.eval()

    results = []

    print("\n1. Reconstruction against model.forward()")
    decomposition = decompose_target_domain(model)
    verification = verify_decomposition(model, decomposition, atol=TOLERANCE)
    ok = verification['passed']
    print(f"  [{'ok' if ok else 'FAIL'}] "
          f"users={decomposition.user_channel_names} "
          f"items={decomposition.item_channel_names} "
          f"score_err={verification['score_max_abs_error']:.2e}")
    results.append(ok)

    print("\n2. Contribution matrix sums to the score")
    explanation = explain_user(model, decomposition, user_id=1, topk=5)
    worst = 0.0
    for attribution in explanation.attributions:
        total = sum(sum(row.values()) for row in attribution.matrix.values())
        worst = max(worst, abs(total - attribution.score))
    ok = worst < TOLERANCE
    print(f"  [{'ok' if ok else 'FAIL'}] max |matrix_sum - score| = {worst:.3e}")
    results.append(ok)

    print("\n3. Guard rails")
    model.fuse_mode = 'concat'
    _check_raises("fuse_mode=concat", NotImplementedError,
                  lambda: decompose_target_domain(model), results)
    model.fuse_mode = 'attention'

    model.train()
    _check_raises("model in train() mode", RuntimeError,
                  lambda: decompose_target_domain(model), results)
    model.eval()

    model.preference_disentangle = False
    _check_raises("preference_disentangle=False", ValueError,
                  lambda: decompose_target_domain(model), results)
    model.preference_disentangle = True

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
