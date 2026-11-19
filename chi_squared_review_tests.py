"""Compare grade distributions using Pearson's chi-square test for 2 x 5 tables.

The default compares all ten pairs of quintiles within each selected run.
With --across-seeds, compare each quintile across all pairs of selected seeds.
P-values use the chi-square distribution with four degrees of freedom. A grade
absent from both groups makes a four-df test unavailable. Each scored
review-explanation pair is treated as independent.
"""

import argparse
from collections import Counter
from itertools import combinations, groupby
import math

from llm_explainer.tools.review_statistics import (
    GRADES, QUINTILES, REPO_ROOT, add_arguments, load_runs, write_results,
)

OUTPUT_ROOT = REPO_ROOT / 'results_chisquared'
COMPARISON_FIELDS = (
    'quintile_right', 'chi_squared', 'p_value', 'significance', 'p_value_holm',
    'n_left', 'n_right', 'skipped_left', 'skipped_right',
    *(f'grade_{grade}_{side}' for side in ('left', 'right') for grade in GRADES),
    'status', 'df',
)


def poisson_probabilities(mean):
    """Evaluate Poisson weights around their mode to avoid exp(-mean) underflow."""
    if mean == 0:
        return [1.0]
    mode = int(mean)
    # This upper limit leaves negligible mass for the CDF inversion.
    last = mode + max(30, math.ceil(12 * math.sqrt(mean)))
    weights = [0.0] * (last + 1)
    weights[mode] = 1.0
    for k in range(mode, 0, -1):
        weights[k - 1] = weights[k] * k / mean
    for k in range(mode + 1, last + 1):
        weights[k] = weights[k - 1] * mean / k
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def cramers_v_ci(statistic, n):
    """Two-sided 95% noncentral chi-square interval for V in a 2 x 5 table.

    A noncentral chi-square with 4 degrees of freedom is a Poisson mixture of
    central chi-squares with 4+2*j degrees of freedom. Each central CDF at x is
    a Poisson(x/2) upper tail starting at j+2. Invert the mixture CDF and
    transform the noncentrality parameters to V = sqrt(lambda / n).
    """
    probabilities = poisson_probabilities(statistic / 2)
    tails = [0.0] * (len(probabilities) + 1)
    for k in range(len(probabilities) - 1, -1, -1):
        tails[k] = tails[k + 1] + probabilities[k]

    def cdf(noncentrality):
        return math.fsum(weight * tail for weight, tail in
                         zip(poisson_probabilities(noncentrality / 2), tails[2:]))

    central_cdf = cdf(0)
    if central_cdf <= 0.025:
        # No positive upper root exists. The full parameter range preserves
        # coverage and avoids reporting a collapsed interval at zero.
        return 0.0, 1.0, 'uninformative_low_statistic'

    def bound(target):
        if central_cdf <= target:
            return 0.0
        if cdf(n) >= target:
            return 1.0
        low, high = 0.0, float(n)
        for _ in range(48):
            mid = (low + high) / 2
            if cdf(mid) > target:
                low = mid
            else:
                high = mid
        return math.sqrt((low + high) / (2 * n))

    return bound(0.975), bound(0.025), 'computed'


def pearson_chi_squared(left, right):
    """Compare two ordered lists of counts for integer grades 1 through 5."""
    if len(left) != len(GRADES) or len(right) != len(GRADES):
        raise ValueError('Expected five grade counts for each group.')
    n_left, n_right = sum(left), sum(right)
    columns = [a + b for a, b in zip(left, right)]
    result = dict(status='computed', chi_squared=None, df=4, p_value=None,
                  cramers_v=None, ci95_low=None, ci95_high=None, ci_status='unavailable',
                  minimum_expected=None, expected_cells_below_5=None, sparse_expected_counts=None)
    if not n_left or not n_right:
        result['status'] = 'empty_group'
        return result
    if 0 in columns:
        result['status'] = 'zero_grade_margin'
        return result

    n = n_left + n_right
    expected = [row_n * col_n / n for row_n in (n_left, n_right) for col_n in columns]
    observed = [*left, *right]
    statistic = math.fsum((o - e)**2 / e for o, e in zip(observed, expected))
    # Closed form of the chi-square survival function for df=4.
    half = statistic / 2
    p_value = math.exp(math.log1p(half) - half)
    ci_low, ci_high, ci_status = cramers_v_ci(statistic, n)
    minimum = min(expected)
    below_5 = sum(e < 5 for e in expected)
    result.update(chi_squared=statistic, p_value=p_value, cramers_v=math.sqrt(statistic / n),
                  ci95_low=ci_low, ci95_high=ci_high, ci_status=ci_status, minimum_expected=minimum,
                  expected_cells_below_5=below_5,
                  sparse_expected_counts=minimum < 1 or below_5 / len(expected) > 0.20)
    return result


def compare(left_run, left_q, left_counts, right_run, right_q, right_counts, mode):
    row = dict(domain=left_run.domain, model=left_run.model, prompting=left_run.prompting,
               comparison=mode, seed_left=left_run.seed, quintile_left=left_q,
               seed_right=right_run.seed, quintile_right=right_q,
               n_left=sum(left_counts), n_right=sum(right_counts),
               skipped_left=left_run.skipped[left_q], skipped_right=right_run.skipped[right_q])
    for side, counts in (('left', left_counts), ('right', right_counts)):
        row.update(('grade_{}_{}'.format(grade, side), count) for grade, count in zip(GRADES, counts))
    row.update(pearson_chi_squared(left_counts, right_counts))
    return row


def run_comparisons(runs, across_seeds):
    counts = {}
    for run in runs:
        histogram = Counter((pair.quintile, pair.grade) for pair in run.pairs)
        counts[run.path] = {q: [histogram[q, grade] for grade in GRADES] for q in QUINTILES}

    if not across_seeds:
        for run in runs:
            for left, right in combinations(QUINTILES, 2):
                yield compare(run, left, counts[run.path][left], run, right,
                              counts[run.path][right], 'within-seed')
        return

    for _, condition_runs in groupby(runs, key=lambda run: run.condition):
        for left, right in combinations(condition_runs, 2):
            if left.judge != right.judge or left.quintile_bounds != right.quintile_bounds:
                raise ValueError('Judge or quintile definitions differ between {} and {}'.format(left.path, right.path))
            for quintile in QUINTILES:
                yield compare(left, quintile, counts[left.path][quintile], right, quintile,
                              counts[right.path][quintile], 'across-seeds')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, OUTPUT_ROOT)
    parser.add_argument('--across-seeds', action='store_true',
                        help='Compare the same quintile across every pair of selected seeds.')
    args = parser.parse_args(argv)
    try:
        rows = list(run_comparisons(load_runs(args), args.across_seeds))
        if not rows:
            raise ValueError('Across-seed comparisons require two matching seeds for a generator and condition.')
        if args.across_seeds:
            filename_suffix = 'chisquared_across_seeds'
            seed_fields = ('seed_left', 'quintile_left', 'seed_right')
        else:
            filename_suffix = 'chisquared_within_seed'
            seed_fields = ('seed', 'quintile_left')
            for row in rows:
                row['seed'] = row['seed_left']
        write_results(rows, args.output, args.holm, output_root=OUTPUT_ROOT,
                      filename_suffix=filename_suffix, fields=seed_fields + COMPARISON_FIELDS)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
