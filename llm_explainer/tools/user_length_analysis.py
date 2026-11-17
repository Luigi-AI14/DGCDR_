"""User-level inference for count_explanation_words.py (requires NumPy).

One independent observation per user, with equal weights for their distinct
items. This analyzes only held-out reviews present in the selected reports.
"""

import math
import statistics
from collections import defaultdict

import numpy as np


GROUP_FIELDS = ("domain_pair", "model", "prompt_version", "generation_mode")


def aggregate_users(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS) + (row["user_id"],)
        groups[key][row["item_id"]].append(row)
    users = []
    for key, items in sorted(groups.items()):
        reviews, explanations, quintiles = [], [], []
        for item_rows in items.values():
            # A changed review or classification is not another independent item.
            signatures = {(r["review_words"], r["quintile"],
                           r["quintile_range_words"], r["quintile_uses_review_title"])
                          for r in item_rows}
            if len(signatures) != 1:
                raise ValueError("Conflicting review/quintile for user {} item {}".format(
                    key[-1], item_rows[0]["item_id"]))
            first = item_rows[0]
            quintiles.append(first["quintile"])
            valid = [r for r in item_rows if r["review_words"] and r["explanation_words"]]
            if valid:
                reviews.append(first["review_words"])
                explanations.append(statistics.mean(r["explanation_words"] for r in valid))
        if not reviews:
            continue
        users.append(dict(
            zip(GROUP_FIELDS, key[:-1]), user_id=key[-1],
            observed_items=len(items), paired_items=len(reviews),
            complete_pairs=len(reviews) == len(items),
            mean_review_words=statistics.mean(reviews),
            mean_explanation_words=statistics.mean(explanations),
            q5_items=quintiles.count("Q5"),
            q5_fraction=quintiles.count("Q5") / len(items),
            q5_only=all(q == "Q5" for q in quintiles),
            sampling_seeds=";".join(sorted({str(r["seed"]) for rs in items.values() for r in rs})),
        ))
    return users


def average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlation_tests(xs, ys, rng, resamples):
    """Two-sided random-pairing tests, preserving ties for Spearman ranks."""
    result = dict(pearson_r=None, pearson_p=None, spearman_rho=None, spearman_p=None)
    if len(xs) < 3 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return result
    x = np.array([xs, average_ranks(xs)], dtype=float)
    y = np.array([ys, average_ranks(ys)], dtype=float)
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    x /= np.sqrt((x * x).sum(axis=1, keepdims=True))
    y /= np.sqrt((y * y).sum(axis=1, keepdims=True))
    observed = (x * y).sum(axis=1)
    extreme = np.zeros(2, dtype=int)
    for start in range(0, resamples, 128):
        permutations = np.array([rng.permutation(len(xs))
                                 for _ in range(min(128, resamples - start))])
        null = (x[:, None, :] * y[:, permutations]).sum(axis=2)
        extreme += (np.abs(null) >= np.abs(observed[:, None]) - 1e-12).sum(axis=1)
    pvalues = (extreme + 1) / (resamples + 1)
    result.update(pearson_r=float(observed[0]), pearson_p=float(pvalues[0]),
                  spearman_rho=float(observed[1]), spearman_p=float(pvalues[1]))
    return result


def studentized_difference(high, rest, axis=-1):
    difference = high.mean(axis=axis) - rest.mean(axis=axis)
    se = np.sqrt(high.var(axis=axis, ddof=1) / high.shape[axis]
                 + rest.var(axis=axis, ddof=1) / rest.shape[axis])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, difference / se,
                        np.where(difference == 0, 0.0, np.sign(difference) * np.inf))


def compare_q5(users, rng=None, resamples=9999):
    """Studentized permutation test and basic bootstrap CI for disjoint users."""
    if rng is None:
        rng = np.random.default_rng(42)
    high = np.array([u["mean_explanation_words"] for u in users if u["q5_only"]])
    rest = np.array([u["mean_explanation_words"] for u in users if not u["q5_only"]])
    overall = float(np.mean(np.concatenate((high, rest)))) if len(users) else None
    result = dict(
        q5_only_users=len(high), other_users=len(rest), overall_user_mean_words=overall,
        q5_only_mean_words=float(high.mean()) if len(high) else None,
        other_users_mean_words=float(rest.mean()) if len(rest) else None,
        difference_vs_overall=None, percent_vs_overall=None,
        difference_vs_others=None, difference_ci95_low=None, difference_ci95_high=None,
        vs_overall_ci95_low=None, vs_overall_ci95_high=None,
        studentized_statistic=None, difference_p=None, status="insufficient_users",
    )
    if len(high) and len(rest):
        delta = float(high.mean() - rest.mean())
        fraction = len(rest) / len(users)
        result.update(difference_vs_others=delta, difference_vs_overall=delta * fraction,
                      percent_vs_overall=100 * delta * fraction / overall if overall else None)
    if len(high) < 2 or len(rest) < 2:
        return result
    v_high, v_rest = high.var(ddof=1) / len(high), rest.var(ddof=1) / len(rest)
    se2 = v_high + v_rest
    if se2 == 0:
        result["status"] = "zero_variance"
        return result
    observed = float(studentized_difference(high, rest))
    pooled = np.concatenate((high, rest))
    extreme, bootstraps = 0, []
    for start in range(0, resamples, 128):
        batch = min(128, resamples - start)
        permuted = np.array([rng.permutation(pooled) for _ in range(batch)])
        null = studentized_difference(permuted[:, :len(high)], permuted[:, len(high):])
        extreme += int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
        boot_high = rng.choice(high, size=(batch, len(high)), replace=True).mean(axis=1)
        boot_rest = rng.choice(rest, size=(batch, len(rest)), replace=True).mean(axis=1)
        bootstraps.extend(boot_high - boot_rest)
    # Basic (reverse-percentile) interval; resampling units are individual users.
    upper_quantile, lower_quantile = np.quantile(bootstraps, [.975, .025])
    low, high_ci = float(2 * delta - upper_quantile), float(2 * delta - lower_quantile)
    result.update(
        difference_ci95_low=low, difference_ci95_high=high_ci,
        vs_overall_ci95_low=fraction * low, vs_overall_ci95_high=fraction * high_ci,
        studentized_statistic=observed, difference_p=(extreme + 1) / (resamples + 1),
        status="ok",
    )
    return result


def adjust_holm(tests):
    """One family across all models, modes, domains, tests and sensitivity runs."""
    valid = []
    for row, field in tests:
        row[field + "_holm"] = None
        if row[field] is not None and math.isfinite(row[field]):
            valid.append((row, field))
    previous = 0.0
    for index, (row, field) in enumerate(sorted(valid, key=lambda entry: entry[0][entry[1]])):
        previous = min(1.0, max(previous, (len(valid) - index) * row[field]))
        row[field + "_holm"] = previous
    return len(valid)


def analyze_users(rows, minimums=(1, 2), resamples=9999, seed=42):
    users = aggregate_users(rows)
    groups = defaultdict(list)
    for user in users:
        groups[tuple(user[field] for field in GROUP_FIELDS)].append(user)
    correlations, comparisons, tests = [], [], []
    rng = np.random.default_rng(seed)
    for key, group in sorted(groups.items()):
        for minimum in sorted(set(minimums)):
            selected = [u for u in group if u["complete_pairs"] and u["paired_items"] >= minimum]
            metadata = dict(zip(GROUP_FIELDS, key), min_reviews=minimum, users=len(selected))
            xs = np.array([u["mean_review_words"] for u in selected])
            ys = np.array([u["mean_explanation_words"] for u in selected])
            correlation = dict(metadata, resamples=resamples, random_seed=seed,
                               **correlation_tests(xs, ys, rng, resamples))
            comparison = dict(metadata, resamples=resamples, random_seed=seed,
                              **compare_q5(selected, rng, resamples))
            correlations.append(correlation)
            comparisons.append(comparison)
            tests.extend([(correlation, "pearson_p"), (correlation, "spearman_p"),
                          (comparison, "difference_p")])
    family_size = adjust_holm(tests)
    for row in correlations + comparisons:
        row["holm_family_size"] = family_size
    return users, correlations, comparisons


def write_report(path, correlations, comparisons, include_title):
    def number(value, places=2):
        return "N/A" if value is None else format(value, ".{}f".format(places))

    lines = ["# User review length and explanation length", "",
             "Review lengths use **{}**. Only held-out reviews in the selected reports are available; "
             "Q5-only does not mean all reviews the user has ever written, or their prompt history.".format(
                 "title + body" if include_title else "body"), "",
             "Each user has equal weight. Each distinct item has equal weight within that user; "
             "repeat generations for one item are averaged. Users with incomplete pairs are excluded "
             "from inference. Saved quintile assignments are preserved.", "",
             "Settings: {} permutations and bootstrap samples per comparison; random seed {}; "
             "Holm family: {} tests. Monte Carlo p-values have resolution 1 / (resamples + 1); "
             "identical small adjusted p-values can reflect this resolution limit.".format(
                 correlations[0]["resamples"], correlations[0]["random_seed"],
                 correlations[0]["holm_family_size"]), "",
             "## Q5-only users", "",
             "The overall mean is the mean of user means. Significance uses a two-sided studentized permutation test "
             "against all OTHER users, avoiding an overlapping comparison. The difference versus "
             "the overall mean is (other users / all users) times the difference versus other users. "
             "Its 95% basic bootstrap CI resamples users within each group and uses the same scaling, "
             "conditional on the observed group proportions. CIs are approximate and unadjusted; "
             "p-values receive Holm correction across all "
             "available tests in this run, including both correlations and minimum-review sensitivities.", "",
             "| Domain | LLM | Prompt | Mode | Min reviews | Q5 users | Other users | Overall mean | Q5 mean | "
             "Difference vs overall (95% CI), words | Change | Holm p |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for row in comparisons:
        cells = [str(row[k]) for k in GROUP_FIELDS] + [str(row[k]) for k in (
            "min_reviews", "q5_only_users", "other_users")]
        cells += [number(row[k]) for k in ("overall_user_mean_words", "q5_only_mean_words")]
        cells += ["{} [{}, {}]".format(*(number(row[k]) for k in (
            "difference_vs_overall", "vs_overall_ci95_low", "vs_overall_ci95_high"))),
                  number(row["percent_vs_overall"]) + "%", number(row["difference_p_holm"], 4)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Correlation of user averages", "",
              "Two-sided permutation p-values use random pairings of user means (plus-one correction). "
              "Spearman uses average ranks for ties. Minimum 1 review is the primary analysis; "
              "minimum 2 reviews checks sensitivity to users represented by a single review.", "",
              "| Domain | LLM | Prompt | Mode | Min reviews | Users | Pearson r | Holm p | Spearman rho | Holm p |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in correlations:
        cells = [str(row[k]) for k in GROUP_FIELDS] + [str(row[k]) for k in ("min_reviews", "users")]
        cells += [number(row[k], 4) for k in ("pearson_r", "pearson_p_holm", "spearman_rho", "spearman_p_holm")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "These are exploratory associations, assuming independent users within each group. "
              "Permutation calibration assumes exchangeability under the null; studentization handles "
              "unequal variance asymptotically, but small samples remain approximate. "
              "Product mix, user selection and prompt length limits may affect results; significance "
              "does not establish that review verbosity causes longer explanations. Model groups may "
              "contain different users. Confidence intervals and tests can be unstable for small Q5 groups.", "",
              "Methods: [bootstrap intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), "
              "[permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), "
              "[studentized permutation inference](https://arxiv.org/abs/1304.5939), "
              "[Holm adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
