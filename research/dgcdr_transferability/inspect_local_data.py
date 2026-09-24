"""Reproduce descriptive statistics for the six local .inter files (stdlib only).

These are full-file inventory statistics, not training-only predictors or results
of a recommender experiment. Run from any directory; output goes beside this file.
"""
import collections
import csv
import json
import math
from pathlib import Path


def gini(values):
    values = sorted(values)
    return (2 * sum(k * x for k, x in enumerate(values, 1)) /
            (len(values) * sum(values)) - (len(values) + 1) / len(values))


def main():
    root = Path(__file__).resolve().parents[2]
    results, identities = [], {}
    for path in sorted((root / 'dataset').glob('*/*.inter')):
        users, items, ratings = (collections.Counter() for _ in range(3))
        with path.open() as stream:
            reader = csv.reader(stream, delimiter='\t')
            header = next(reader)
            for row in reader:
                users[row[0]] += 1
                items[row[1]] += 1
                ratings[int(float(row[2]))] += 1
        n = sum(ratings.values())
        name = path.parent.name
        identities[name] = (set(users), set(items))
        results.append({
            'dataset': name, 'file': str(path.relative_to(root)),
            'header': header, 'users': len(users), 'items': len(items),
            'interactions': n, 'density': n / (len(users) * len(items)),
            'interactions_per_user': n / len(users),
            'interactions_per_item': n / len(items),
            'minimum_user_degree': min(users.values()),
            'minimum_item_degree': min(items.values()),
            'rating_counts': dict(sorted(ratings.items())),
            'mean_rating': sum(r * c for r, c in ratings.items()) / n,
            'fraction_rating_1_or_2': (ratings[1] + ratings[2]) / n,
            'fraction_rating_4_or_5': (ratings[4] + ratings[5]) / n,
            'item_degree_gini': gini(items.values()),
            'top_1_percent_items_interaction_share':
                sum(sorted(items.values(), reverse=True)[:math.ceil(len(items) * .01)]) / n,
        })
    pair_names = [
        ('AmazonElec_AmazonCloth_commonUser_10-core', 'AmazonCloth_AmazonElec_commonUser_10-core'),
        ('AmazonSport_AmazonCloth_commonUser_5-core', 'AmazonCloth_AmazonSport_commonUser_5-core'),
        ('DoubanMovie_DoubanBook_commonUser_10-core', 'DoubanBook_DoubanMovie_commonUser_10-core'),
    ]
    pairs = []
    for a, b in pair_names:
        ua, ia = identities[a]
        ub, ib = identities[b]
        pairs.append({'domains': [a, b], 'overlap_users': len(ua & ub),
                      'user_jaccard': len(ua & ub) / len(ua | ub),
                      'overlap_items': len(ia & ib)})
    output = {'scope': 'Full local files before runtime splitting; counts are rows, not a deduplication audit.',
              'datasets': results, 'pairs': pairs}
    dest = Path(__file__).with_name('local_data_inventory.json')
    dest.write_text(json.dumps(output, indent=2) + '\n')
    for row in results:
        print(row['dataset'], 'low ratings:', round(100 * row['fraction_rating_1_or_2'], 2),
              '%, item Gini:', round(row['item_degree_gini'], 3))
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()
