"""Self-test for the channel decomposition.

Checks, against a real checkpoint, that:
  1. the channels sum back to the model's own forward() outputs, in every
     supported configuration (attention_mode all/part x item_disentangle on/off);
  2. the contribution matrix sums to the score of each attributed item;
  3. the guard rails refuse the configurations where the decomposition would
     not be exact, instead of silently returning an approximation.

Run it after touching anything in this module, or on a new checkpoint before
trusting the attributions it produces:

    python -m extensions.explainability.test_decomposition -m saved/<ckpt>.pth
"""

import argparse
import sys

from recbole_cdr.quick_start.quick_start import load_data_and_model

from extensions.explainability.attribution import explain_user
from extensions.explainability.channels import (
    decompose_target_domain,
    forward_outputs,
    refuse_users,
    user_fusion_parts,
    verify_decomposition,
)

# The score is a sum of float32 dot products, so exact bitwise equality is not
# achievable; anything above this is a real bug, not rounding.
TOLERANCE = 1e-4


def _check_config(model, label, results):
    decomposition = decompose_target_domain(model)
    verification = verify_decomposition(model, decomposition, atol=TOLERANCE)
    ok = verification['passed']
    print(f"  [{'ok' if ok else 'FAIL'}] {label:36s} "
          f"users={decomposition.user_channel_names} "
          f"items={decomposition.item_channel_names} "
          f"score_err={verification['score_max_abs_error']:.2e}")
    results.append(ok)
    return decomposition


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
    original_attention, original_item = model.attention_mode, model.item_disentangle

    print("\n1. Reconstruction against model.forward()")
    decomposition = _check_config(model, "attention=all, item_disentangle=True", results)
    for attention_mode in ('all', 'part'):
        for item_disentangle in (True, False):
            if (attention_mode, item_disentangle) == ('all', True):
                continue
            model.attention_mode = attention_mode
            model.item_disentangle = item_disentangle
            _check_config(
                model,
                f"attention={attention_mode}, item_disentangle={item_disentangle}",
                results)
    model.attention_mode, model.item_disentangle = original_attention, original_item

    print("\n2. Contribution matrix sums to the score")
    explanation = explain_user(model, decomposition, user_id=1, topk=5)
    worst = 0.0
    for attribution in explanation.attributions:
        total = sum(sum(row.values()) for row in attribution.matrix.values())
        worst = max(worst, abs(total - attribution.score))
    ok = worst < TOLERANCE
    print(f"  [{'ok' if ok else 'FAIL'}] max |matrix_sum - score| = {worst:.3e}")
    results.append(ok)

    print("\n3. Re-fusing with the model's own attention is a no-op")
    # The fusion audit replaces the attention weights and reads the accuracy
    # change as the cost of personalisation. That reading is only valid if
    # re-fusing with the *original* weights reproduces the model exactly --
    # otherwise the measured cost includes whatever the re-fusion itself broke.
    parts = user_fusion_parts(model, 'target')
    refused = refuse_users(model, parts, parts['attention'])
    reference = forward_outputs(model)['target_user'].detach()
    worst = (refused - reference).abs().max().item()
    ok = worst < TOLERANCE
    print(f"  [{'ok' if ok else 'FAIL'}] max |refused - forward()| = {worst:.3e}")
    results.append(ok)

    print("\n4. Guard rails")
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
