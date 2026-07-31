"""Does the generated text actually say what the numbers say?

The point of the framework is that the explanation is anchored to a quantity the
model computed. That is a claim about the *text*, and it has to be tested on the
text: a second model reads the explanation alone -- no numbers, no history, no
decomposition -- and estimates the transfer ratio it conveys. If the estimate
tracks the real tau, the sentence carries the attribution. If it does not, the
text is fluent and unmoored, which is precisely what the framework is supposed
to avoid.

The reader must be a different model from the writer: one asked to grade its own
prose measures its own consistency.
"""

import numpy as np

READER_SYSTEM = """You read an explanation a recommender system gave to a user.

The system splits a user's taste into two parts:
- TRANSFERRED: taste carried over from a different product category.
- NATIVE: taste specific to the category being recommended.

From the explanation alone, estimate how much of the recommendation the text
attributes to TRANSFERRED taste.

Answer with a single number from 0 to 100, where 0 means entirely native, 50
means an even mix, and 100 means entirely transferred. Nothing else."""


def read_back(explanations, client):
    """Estimate tau from each explanation, blind to everything else.

    ``explanations`` is a list of (true_tau, text).
    Returns per-item estimates plus the correlation with the true values.
    """
    results = []
    for true_tau, text in explanations:
        prompt = f'Explanation:\n"{text}"\n\nHow much does it attribute to transferred taste? Answer 0-100.'
        estimate = client.rate(prompt, system=READER_SYSTEM, lo=0, hi=100)
        if estimate is None:
            continue
        results.append({'true_tau': true_tau, 'read_tau': estimate / 100.0})

    if len(results) < 3:
        return {'n': len(results), 'per_item': results}

    true = np.array([r['true_tau'] for r in results])
    read = np.array([r['read_tau'] for r in results])
    return {
        'n': len(results),
        'pearson': _pearson(true, read),
        'spearman': _spearman(true, read),
        'mean_true': float(true.mean()),
        'mean_read': float(read.mean()),
        'mean_abs_error': float(np.abs(true - read).mean()),
        'per_item': results,
    }


def _pearson(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return _pearson(ra, rb)
