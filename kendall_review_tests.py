"""Test Kendall's tau-b between exact review word counts and judge grades.

The default runs one test per selected generator, domain, prompting mode, and
seed. Use --pool-seeds for one test on all selected observations per generator,
domain, and prompting mode. P-values are two-sided tie-corrected normal
approximations. The CSV reports how often the longer review scored higher,
both excluding and including grade ties. Both percentages use comparisons
between reviews of different lengths.
Each scored review-explanation pair is treated as independent.
"""

import argparse
from collections import Counter
from itertools import combinations, groupby
import math
from statistics import NormalDist

from llm_explainer.tools.review_statistics import REPO_ROOT, add_arguments, load_runs, write_results


NORMAL_95 = NormalDist().inv_cdf(0.975)
OUTPUT_ROOT = REPO_ROOT / 'results_kendall'
CSV_FIELDS = (
    'seed', 'n', 'skipped', 'status', 'tau_b', 'p_value', 'p_value_holm', 'significance',
    'pct_longer_higher_excluding_grade_ties', 'pct_longer_higher_including_grade_ties',
)


def tie_terms(counts):
    """Numbers of tied pairs and two sums needed by the normal null variance."""
    return (
        sum(t * (t - 1) // 2 for t in counts),
        sum(t * (t - 1) * (t - 2) for t in counts),
        sum(t * (t - 1) * (2 * t + 5) for t in counts),
    )


def jackknife_ci(rows, contributions, score, tau, words, grades, word_ties, grade_ties):
    """Compute leave-one-out coefficients by subtracting affected pairs and ties."""
    n = len(rows)
    remaining_pairs = (n - 1) * (n - 2) // 2
    estimates = []
    for (word_count, grade), contribution in zip(rows, contributions):
        different_words = remaining_pairs - word_ties + words[word_count] - 1
        different_grades = remaining_pairs - grade_ties + grades[grade] - 1
        if not different_words or not different_grades:
            return None, None
        estimates.append((score - contribution) / math.sqrt(different_words * different_grades))

    mean_estimate = math.fsum(estimates) / n
    se = math.sqrt((n - 1) / n * math.fsum((estimate - mean_estimate)**2 for estimate in estimates))
    margin = NORMAL_95 * se
    return max(-1.0, tau - margin), min(1.0, tau + margin)


def kendall_tau_b(rows):
    """Use the tie-corrected null variance of the net concordance score for p."""
    n = len(rows)
    result = dict(status='computed', tau_b=None, z=None, p_value=None,
                  ci95_low=None, ci95_high=None, ci_status='unavailable')
    words = Counter(word_count for word_count, grade in rows)
    grades = Counter(grade for word_count, grade in rows)
    higher = lower = 0
    contributions = [0] * n
    for (i, (x1, y1)), (j, (x2, y2)) in combinations(enumerate(rows), 2):
        if x1 == x2 or y1 == y2:
            continue
        direction = ((x2 > x1) - (x2 < x1)) * ((y2 > y1) - (y2 < y1))
        higher += direction > 0
        lower += direction < 0
        contributions[i] += direction
        contributions[j] += direction

    m = n * (n - 1)
    total_pairs = m // 2
    word_ties, word_triplets, word_variance = tie_terms(words.values())
    different_lengths = total_pairs - word_ties
    different_lengths_and_grades = higher + lower
    result.update(
        longer_higher_count=higher,
        longer_lower_count=lower,
        longer_same_grade_count=different_lengths - different_lengths_and_grades,
        pct_longer_higher_excluding_grade_ties=(
            100 * higher / different_lengths_and_grades if different_lengths_and_grades else None
        ),
        pct_longer_higher_including_grade_ties=(
            100 * higher / different_lengths if different_lengths else None
        ),
    )
    if n < 3:
        result['status'] = 'insufficient_observations'
        return result
    if len(words) < 2 or len(grades) < 2:
        result['status'] = 'constant_word_count_or_grade'
        return result

    score = higher - lower
    grade_ties, grade_triplets, grade_variance = tie_terms(grades.values())
    tau = score / math.sqrt(different_lengths * (total_pairs - grade_ties))
    # Same normal null variance as scipy.stats.kendalltau(method='asymptotic').
    variance = ((m * (2 * n + 5) - word_variance - grade_variance) / 18
                + 2 * word_ties * grade_ties / m
                + word_triplets * grade_triplets / (9 * m * (n - 2)))
    z = score / math.sqrt(variance)
    low, high = jackknife_ci(rows, contributions, score, tau, words, grades, word_ties, grade_ties)
    result.update(tau_b=tau, z=z, p_value=math.erfc(abs(z) / math.sqrt(2)),
                  ci95_low=low, ci95_high=high,
                  ci_status='computed' if low is not None else 'constant_leave_one_out_sample')
    return result


def analyze_runs(runs):
    first = runs[0]
    values = []
    for run in runs:
        if run.judge != first.judge:
            raise ValueError('Judge definitions differ between {} and {}'.format(first.path, run.path))
        for pair in run.pairs:
            if type(pair.word_count) is not int or pair.word_count < 0:
                raise ValueError('Invalid review_word_count in {}'.format(run.path))
            values.append((pair.word_count, pair.grade))
    row = dict(domain=first.domain, model=first.model, prompting=first.prompting,
               seed='+'.join(str(run.seed) for run in runs), n=len(values),
               skipped=sum(sum(run.skipped.values()) for run in runs))
    row.update(kendall_tau_b(values))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, OUTPUT_ROOT)
    parser.add_argument('--pool-seeds', action='store_true',
                        help='Pool selected seeds within each generator, domain, and prompting mode.')
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args)
        if args.pool_seeds:
            groups = (list(group) for _, group in groupby(runs, key=lambda run: run.condition))
            filename_suffix = 'kendall_pooled_seeds'
        else:
            groups = ([run] for run in runs)
            filename_suffix = 'kendall'
        write_results([analyze_runs(group) for group in groups], args.output, args.holm,
                      output_root=OUTPUT_ROOT, filename_suffix=filename_suffix, fields=CSV_FIELDS)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
