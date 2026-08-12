"""Machine-readable view of one user's ranking report.

extract_user_ranking.py (a single user) and extract_user_ranking_enhanced.py (a
sample of users) both emit per-user JSON in this shape, so run_llm_relevance.py
has one schema to consume instead of two that can drift apart.

Nothing here reads a module global: ``topk`` and ``hide_seen_items`` are passed
in, so neither script can silently write a report that disagrees with the
constants it actually ran with.
"""

import numpy as np


def json_score(value):
    """Masked items score -inf, and `-Infinity` is not valid JSON."""
    return value if np.isfinite(value) else None


def item_entries(ext_ids, titles):
    return [{'item_id': ext_id, 'title': titles.get(ext_id, 'Unknown Title')}
            for ext_id in ext_ids]


def domain_names(config, source_fallback, target_fallback):
    """Dataset names of both domains, for labelling the reports."""
    names = {}
    for side, fallback in (('source', source_fallback), ('target', target_fallback)):
        section = config['{}_domain'.format(side)]
        names[side] = (section or {}).get('dataset', fallback)
    return names


def user_json(record, item_titles, source_titles, domains, topk, hide_seen_items):
    """One user's report as data -- same content as the markdown, machine-readable.

    Written for downstream LLM consumption, so the ranking is a list of objects
    rather than the parallel arrays of ``record``: nothing has to be zipped back
    together to know which title a relevance flag belongs to.
    """
    return {
        'user': {
            'external_id': record['external_uid'],
            'internal_id': record['target_uid'],
            'n_held_out': record['n_positives'],
        },
        'config': {
            'topk': topk,
            'hide_seen_items': hide_seen_items,
            'source_domain': domains['source'],
            'target_domain': domains['target'],
        },
        'ranking': [
            {
                'rank': rank,
                'item_id': ext_id,
                'title': item_titles.get(ext_id, 'Unknown Title'),
                'score': json_score(score),
                # Relevance is implicit feedback: False only means the item is
                # absent from the held-out set, not that the user dislikes it.
                'relevant': bool(is_relevant),
            }
            for rank, (ext_id, score, is_relevant) in enumerate(
                zip(record['ranked_ids'], record['ranked_scores'], record['ranked_relevant']), 1)
        ],
        'metrics': {
            'shown': record['shown'] if hide_seen_items else None,
            'native': record['native'],
        },
        'ground_truth': item_entries(record['ground_truth_ids'], item_titles),
        'source_history': item_entries(record['source_item_ids'], source_titles),
        'target_history': item_entries(record['seen_item_ids'], item_titles),
    }
