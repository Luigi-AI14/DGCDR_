"""LLM-assisted relevance audit of a DGCDR ranking.

Reads the per-user JSON written by extract_user_ranking.py /
extract_user_ranking_enhanced.py and asks a locally served LLM two things:

1. whether the ranking is coherent with the user's actual interactions, across
   both domains;
2. for every item the evaluation counts as *non*-relevant, whether it is
   semantically consistent enough with the user's profile to be a plausible
   false negative -- an item the user would likely have interacted with but
   never saw.

The second question is the point. Held-out test sets are implicit feedback: an
item is labelled irrelevant because the user did not buy it, not because they
rejected it. A verdict here is therefore a *hypothesis about labelling noise*,
never ground truth, and every metric this script prints that depends on one is
named `llm_` so it can never be mistaken for a measured result.

Serving is Ollama's REST API on localhost, with constrained decoding against a
JSON schema, so the output parses by construction rather than by prompt luck.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time

import requests

DEFAULT_MODEL = 'qwen3.5:9b'
DEFAULT_HOST = 'http://localhost:11434'

# qwen3.5 ships temperature 1 / presence_penalty 1.5, which are the wrong
# defaults for a judge: identical input must give identical verdicts.
SAMPLING = {
    'temperature': 0,
    'top_p': 1,
    'presence_penalty': 0,
    'seed': 42,
}

VERDICTS = ['likely_relevant', 'uncertain', 'likely_irrelevant']

# The rubric is spelled out for the model, because "1 to 5" alone is ambiguous:
# asked for a bare integer it will happily read the scale as a rank and return 1
# for everything.
RUBRIC = """\
5 = functional duplicate or direct replacement of something the user owns, or the \
obvious next purchase for a specific owned item. You must be able to name that item.
4 = clearly complementary to a specific owned item; a natural accessory for this \
exact setup.
3 = plausible for this instrument in general, but with no specific link to anything \
the user owns.
2 = weak link: fits the broad category, but not this user's evident setup, level or taste.
1 = implies an instrument, genre or activity the user shows no sign of."""

# The model scores; the verdict is derived from the score in code. Letting it emit
# both invites the two to disagree, and a threshold is more useful as a knob than
# as a model whim.
RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'user_profile': {'type': 'string'},
        'ranking_coherence': {
            'type': 'object',
            'properties': {
                'score': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                'explanation': {'type': 'string'},
            },
            'required': ['score', 'explanation'],
        },
        'judgments': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'candidate_id': {'type': 'integer'},
                    'closest_owned_item': {'type': 'string'},
                    'rationale': {'type': 'string'},
                    'semantic_consistency': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                },
                # Ordered so the model commits to a justification before a number,
                # rather than picking a number and rationalising it.
                'required': ['candidate_id', 'closest_owned_item', 'rationale',
                             'semantic_consistency'],
            },
        },
    },
    'required': ['user_profile', 'ranking_coherence', 'judgments'],
}


def verdict_for(score, threshold):
    if score >= threshold:
        return 'likely_relevant'
    if score == threshold - 1:
        return 'uncertain'
    return 'likely_irrelevant'


def normalise_tokens(text):
    return set(re.sub(r'[^a-z0-9]+', ' ', text.lower()).split())


def verify_citation(citation, owned_titles, candidate_title, threshold=0.6):
    """Check that the item cited as 'closest owned' is one the user actually owns.

    The top score is supposed to require naming a specific owned item, which makes
    the citation the load-bearing part of the judgment -- and the easiest thing for
    the model to fabricate. Observed failure: it cites the *candidate itself* as the
    owned item ("a duplicate of the picks they already own", for a user who owns no
    picks) and awards a 5 on that basis, so self-citation is reported separately.

    Matching is token containment rather than equality: the model paraphrases and
    truncates long product titles, which no exact match would survive.
    """
    blank = {'citation_verified': None, 'citation_match': None, 'citation_self': False}
    if not citation or citation.strip().lower() in ('none', 'n/a', 'na', '-', ''):
        return blank
    cited = normalise_tokens(citation)
    if not cited:
        return blank

    def containment(other):
        return len(cited & normalise_tokens(other)) / float(len(cited))

    best_title, best = None, 0.0
    for title in owned_titles:
        value = containment(title)
        if value > best:
            best, best_title = value, title
    self_match = containment(candidate_title)
    return {
        'citation_verified': best >= threshold,
        'citation_match': best_title if best >= threshold else None,
        'citation_self': self_match > best and self_match >= threshold,
    }

SYSTEM_PROMPT = """\
You audit a cross-domain recommender system. You are given one user's purchase \
history in two domains and a ranked list of items the model recommended.

Your job is to judge SEMANTIC AND FUNCTIONAL CONSISTENCY between recommended \
items and what the user actually owns.

Critical framing: a recommended item is labelled "not relevant" only because the \
user did not happen to buy it in the test period. That is NOT evidence they \
disliked it. Many such items are false negatives: the user never saw them. Your \
task is to identify those.

Judge on:
- functional equivalence or complementarity to items the user owns (a strap lock \
  and a strap-lock set are the same purchase; a capo and a guitar are complementary)
- consistency with the instruments, genres and brands the user has shown
- whether the item fits the user's evident skill level and setup

Rate each candidate's semantic consistency on this scale:

{rubric}

Calibration matters more than generosity. A competent recommender's top-20 is \
almost entirely superficially plausible, so "plausible" is not enough to earn a 4 \
or 5 -- those require a specific owned item you can name. Expect a spread across \
the scale: if you rate more than a third of the candidates 4 or above, you are \
being too permissive, so re-check which ones really have a concrete link. Equally, \
do not damn an item merely because it duplicates something owned -- strings, picks \
and cables are consumables that get rebought.

Never justify a score with the item's rank or position; judge the product, not the \
model.
""".format(rubric=RUBRIC)


def load_payload(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def bullet_list(entries, limit):
    """Titles as a bullet list, truncated so a long history cannot blow context."""
    shown = entries[:limit]
    lines = ['- {}'.format(e['title']) for e in shown]
    if len(entries) > limit:
        lines.append('- (... and {} more)'.format(len(entries) - limit))
    return '\n'.join(lines) if lines else '- (none)'


def build_candidates(payload, mode, probe, rng):
    """The items the LLM must judge, as (candidate_id, entry) records.

    In every mode the candidates are the ranking's non-relevant items -- the ones
    the evaluation counts as misses. ``probe`` additionally smuggles the held-out
    ground-truth items into the same list, unlabelled: they are known true
    positives, so the share of them the judge recovers measures the judge itself.
    """
    candidates = []
    for item in payload['ranking']:
        if item['relevant']:
            continue
        candidates.append({
            'source': 'ranking',
            'rank': item['rank'],
            'item_id': item['item_id'],
            'title': item['title'],
            'score': item['score'],
        })

    if probe:
        ranked_ids = set(i['item_id'] for i in payload['ranking'])
        for entry in payload['ground_truth']:
            if entry['item_id'] in ranked_ids:
                continue  # already in the list, and there as a genuine miss
            candidates.append({
                'source': 'probe',
                'rank': None,
                'item_id': entry['item_id'],
                'title': entry['title'],
                'score': None,
            })
        rng.shuffle(candidates)  # position must not leak which is which

    for i, entry in enumerate(candidates, 1):
        entry['candidate_id'] = i
    return candidates


def render_prompt(payload, candidates, mode, max_history):
    cfg = payload['config']
    parts = []

    parts.append('## USER PROFILE\n')
    parts.append('### Source domain -- music the user bought ({})\n'.format(cfg['source_domain']))
    parts.append(bullet_list(payload['source_history'], max_history))
    parts.append('\n### Target domain -- items the user already owns ({})\n'.format(cfg['target_domain']))
    parts.append(bullet_list(payload['target_history'], max_history))

    parts.append('\n\n## MODEL RANKING (top {})\n'.format(cfg['topk']))
    if mode == 'anchor':
        # The evaluation's own labels, shown as anchors. This is what the user
        # asked for, but it does mean the verdicts cannot then be used to score
        # the model without circularity -- see --mode blind.
        known = [i for i in payload['ranking'] if i['relevant']]
        if known:
            parts.append('Items marked [KNOWN RELEVANT] were genuinely bought by the user '
                         'in the held-out period. Use them as positive anchors.\n')
        else:
            parts.append('None of the ranked items was bought in the held-out period, so '
                         'judge consistency against the owned items above.\n')
        for item in payload['ranking']:
            flag = ' [KNOWN RELEVANT]' if item['relevant'] else ''
            parts.append('{}. {}{}'.format(item['rank'], item['title'], flag))
    else:
        for item in payload['ranking']:
            parts.append('{}. {}'.format(item['rank'], item['title']))

    parts.append('\n\n## CANDIDATES TO JUDGE\n')
    parts.append('For each candidate decide whether it could actually be relevant '
                 'to this user despite not being bought.\n')
    for entry in candidates:
        parts.append('[{}] {}'.format(entry['candidate_id'], entry['title']))

    parts.append('\n\n## OUTPUT\n')
    parts.append('Return JSON with:')
    parts.append('- "user_profile": 2-3 sentences on this user\'s tastes, instruments and setup, '
                 'inferred from both domains.')
    parts.append('- "ranking_coherence": score 1-5 for how well the ranking matches that profile, '
                 'plus a short explanation naming specific items.')
    parts.append('- "judgments": exactly one entry per candidate above, using its [id] as '
                 '"candidate_id". Give "closest_owned_item" (quote an item the user owns, or '
                 '"none"), then "rationale", then "semantic_consistency" on the 1-5 scale.')
    parts.append('\nScale reminder:')
    parts.append(RUBRIC)
    return '\n'.join(parts)


def call_llm(host, model, system_prompt, user_prompt, num_ctx, think, timeout, retries=3):
    """One judged batch, retried on transient failures.

    A 50-user run is half an hour of GPU time; without this a single dropped
    connection -- the Ollama server exiting, say -- throws all of it away.
    """
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'stream': False,
        'think': think,
        'format': RESPONSE_SCHEMA,
        'options': dict(SAMPLING, num_ctx=num_ctx),
    }
    last = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(host.rstrip('/') + '/api/chat', json=payload,
                                     timeout=timeout)
            response.raise_for_status()
            body = response.json()
            return json.loads(body['message']['content']), body
        except (requests.exceptions.RequestException, ValueError) as exc:
            # ValueError covers the JSONDecodeError of a truncated response; the
            # schema makes that unlikely, but a retry is cheaper than a crash.
            last = exc
            if attempt < retries:
                wait = 5 * attempt
                print('    call failed ({}: {}); retry {}/{} in {}s'.format(
                    type(exc).__name__, str(exc)[:120], attempt, retries - 1, wait))
                time.sleep(wait)
    raise RuntimeError('LLM call failed after {} attempts: {}'.format(retries, last))


def merge_judgments(candidates, parsed, threshold, owned_titles, require_citation):
    """Attach each verdict to its candidate.

    The join is on the integer candidate_id rather than a title or an ASIN: those
    are long enough that a model will occasionally paraphrase or mistype one,
    and a silent mismatch would misattribute a verdict.
    """
    by_id = {entry['candidate_id']: entry for entry in candidates}
    merged = []
    unmatched = []
    for judgment in parsed.get('judgments', []):
        entry = by_id.pop(judgment.get('candidate_id'), None)
        if entry is None:
            unmatched.append(judgment)
            continue
        score = judgment['semantic_consistency']
        citation = verify_citation(judgment['closest_owned_item'], owned_titles, entry['title'])
        verdict = verdict_for(score, threshold)
        # A top score rests on naming a real owned item. If that name checks out
        # against nothing the user owns, the score has no support left.
        demoted = (require_citation and verdict == 'likely_relevant'
                   and citation['citation_verified'] is not True)
        if demoted:
            verdict = 'uncertain'
        row = dict(entry)
        row.update(citation)
        row.update({
            'semantic_consistency': score,
            'verdict': verdict,
            'demoted_unverified': demoted,
            'closest_owned_item': judgment['closest_owned_item'],
            'rationale': judgment['rationale'],
        })
        merged.append(row)
    merged.sort(key=lambda e: e['candidate_id'])
    return merged, list(by_id.values()), unmatched


def summarise(payload, merged, probe):
    """Counts, plus the payoff metric -- all clearly marked as LLM-assessed."""
    ranking_side = [e for e in merged if e['source'] == 'ranking']
    rescued = [e for e in ranking_side if e['verdict'] == 'likely_relevant']
    n_relevant = sum(1 for i in payload['ranking'] if i['relevant'])

    summary = {
        'n_candidates': len(merged),
        # Probes are candidates too, so the rescue rate must divide by the ranking
        # side alone -- mixing them understates it on a --probe run.
        'n_ranking_candidates': len(ranking_side),
        'verdict_counts': dict((v, sum(1 for e in merged if e['verdict'] == v)) for v in VERDICTS),
        # A judge that gives every item the same score has no discriminative power,
        # whatever its rationales say. Keep the distribution visible.
        'score_distribution': dict((str(s), sum(1 for e in merged
                                                if e['semantic_consistency'] == s))
                                   for s in range(1, 6)),
        'citations': {
            'unverified': sum(1 for e in merged if e['citation_verified'] is False),
            'self_cited': sum(1 for e in merged if e['citation_self']),
            'none_given': sum(1 for e in merged if e['citation_verified'] is None),
            'demoted': sum(1 for e in merged if e['demoted_unverified']),
        },
        'n_true_relevant_in_topk': n_relevant,
        'llm_rescued_in_topk': len(rescued),
        # Hit@K if the LLM's rescued items were counted as relevant. A hypothesis
        # about the labels, not a measurement of the model.
        'llm_assisted_hit_at_k': 1 if (n_relevant or rescued) else 0,
    }

    if probe:
        probes = [e for e in merged if e['source'] == 'probe']
        recovered = [e for e in probes if e['verdict'] == 'likely_relevant']
        summary['probe'] = {
            'n_probes': len(probes),
            'n_recovered': len(recovered),
            # Known positives the judge failed to flag: its miss rate, on items
            # we know the answer for.
            'recall_on_known_positives': (round(len(recovered) / float(len(probes)), 3)
                                          if probes else None),
        }
    return summary


def render_report(payload, parsed, merged, summary, meta):
    user = payload['user']
    lines = ['# LLM Relevance Audit -- user {}'.format(user['external_id']), '']
    lines.append('- model: `{}`  (mode: {}{})'.format(meta['model'], meta['mode'],
                                                      ', probe' if meta['probe'] else ''))
    lines.append('- held-out items: {} | measured Hit@{}: {}'.format(
        user['n_held_out'], payload['config']['topk'],
        payload['metrics']['shown']['hit'] if payload['metrics']['shown']
        else payload['metrics']['native']['hit']))
    lines.append('')
    lines.append('## Inferred profile\n')
    lines.append(parsed['user_profile'])
    lines.append('')
    coherence = parsed['ranking_coherence']
    lines.append('## Ranking coherence: {}/5\n'.format(coherence['score']))
    lines.append(coherence['explanation'])
    lines.append('')
    lines.append('## Verdicts on non-relevant items\n')
    lines.append('| Cand | Rank | Verdict | Cons. | Title | Closest owned item | Cited item exists | Rationale |')
    lines.append('|---|---|---|---|---|---|---|---|')
    marks = {True: 'yes', False: 'NO', None: '-'}
    for e in merged:
        cell = lambda t: str(t).replace('|', '\\|').replace('\n', ' ').strip()
        mark = marks[e['citation_verified']]
        if e['citation_self']:
            mark = 'NO (self)'
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            e['candidate_id'],
            e['rank'] if e['rank'] is not None else 'probe',
            e['verdict'] + (' (demoted)' if e['demoted_unverified'] else ''),
            e['semantic_consistency'],
            cell(e['title']), cell(e['closest_owned_item']), mark, cell(e['rationale'])))
    lines.append('')
    lines.append('## Summary\n')
    lines.append('```json')
    lines.append(json.dumps(summary, indent=2))
    lines.append('```')
    lines.append('')
    lines.append('> `llm_*` figures treat the model\'s verdicts as labels. They are a '
                 'hypothesis about false negatives in the test set, not a measured result.')
    return '\n'.join(lines)


def audit_user(path, args, rng):
    payload = load_payload(path)
    uid = payload['user']['external_id']
    stem = os.path.join(args.out_dir, 'llm_audit_{}'.format(uid))

    if not args.overwrite and os.path.isfile(stem + '.json'):
        # Resume: a half-hour batch that died partway should not re-spend the GPU
        # time it already paid for.
        with open(stem + '.json', 'r', encoding='utf-8') as f:
            done = json.load(f)
        print('  {}: already audited, reusing'.format(uid))
        summary = done['summary']
        summary['external_id'] = uid
        # Audits written before this field existed would otherwise fall back to a
        # denominator that includes probes, deflating the base rate the lift is
        # measured against. The judgments carry what is needed to backfill it.
        if 'n_ranking_candidates' not in summary:
            summary['n_ranking_candidates'] = sum(1 for j in done['judgments']
                                                  if j['source'] == 'ranking')
        return summary

    candidates = build_candidates(payload, args.mode, args.probe, rng)
    if not candidates:
        print('  {}: every ranked item is relevant, nothing to judge'.format(uid))
        return None

    prompt = render_prompt(payload, candidates, args.mode, args.max_history)
    print('  {}: {} candidates, prompt ~{} chars'.format(uid, len(candidates), len(prompt)))

    parsed, body = call_llm(args.host, args.model, SYSTEM_PROMPT, prompt,
                            args.num_ctx, args.think, args.timeout)

    prompt_tokens = body.get('prompt_eval_count')
    if prompt_tokens and prompt_tokens >= args.num_ctx:
        print('  WARNING: prompt filled the {}-token window; raise --num-ctx'.format(args.num_ctx))

    owned_titles = [e['title'] for e in payload['target_history'] + payload['source_history']]
    merged, missing, unmatched = merge_judgments(candidates, parsed, args.relevant_threshold,
                                                 owned_titles, args.require_citation)
    n_bad = sum(1 for e in merged if e['citation_verified'] is False)
    if n_bad:
        print('  NOTE: {} citation(s) match nothing the user owns ({} cite the candidate '
              'itself){}'.format(n_bad, sum(1 for e in merged if e['citation_self']),
                                 '; demoted' if args.require_citation else ''))
    if len(set(e['semantic_consistency'] for e in merged)) == 1 and len(merged) > 1:
        print('  WARNING: every candidate got the same score; the judge is not '
              'discriminating and its verdicts carry no information')
    if missing:
        print('  WARNING: {} candidate(s) not judged: {}'.format(
            len(missing), [e['candidate_id'] for e in missing]))
    if unmatched:
        print('  WARNING: {} verdict(s) had no matching candidate id'.format(len(unmatched)))

    summary = summarise(payload, merged, args.probe)
    meta = {'model': args.model, 'mode': args.mode, 'probe': args.probe,
            'relevant_threshold': args.relevant_threshold,
            'source_file': os.path.basename(path),
            'prompt_eval_count': prompt_tokens, 'eval_count': body.get('eval_count')}

    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'user': payload['user'], 'summary': summary,
                   'user_profile': parsed['user_profile'],
                   'ranking_coherence': parsed['ranking_coherence'],
                   'judgments': merged}, f, indent=2, ensure_ascii=False)
    with open(stem + '.md', 'w', encoding='utf-8') as f:
        f.write(render_report(payload, parsed, merged, summary, meta))

    print('  -> {}.json / .md  ({} rescued of {} candidates)'.format(
        stem, summary['llm_rescued_in_topk'], summary['n_candidates']))
    summary['external_id'] = uid
    return summary


def aggregate(summaries, args):
    """Totals over every audited user, so a 50-user run has one thing to read."""
    totals = {
        'n_users': len(summaries),
        'model': args.model,
        'mode': args.mode,
        # Not 'probe': that key holds the probe *statistics* below, and a bool
        # under the same name silently shadows them.
        'probe_enabled': args.probe,
        'relevant_threshold': args.relevant_threshold,
        'require_citation': args.require_citation,
        'n_candidates': sum(s['n_candidates'] for s in summaries),
        'n_ranking_candidates': sum(s.get('n_ranking_candidates', s['n_candidates'])
                                    for s in summaries),
        'llm_rescued': sum(s['llm_rescued_in_topk'] for s in summaries),
        'true_relevant_in_topk': sum(s['n_true_relevant_in_topk'] for s in summaries),
        'measured_hit_at_k': sum(1 for s in summaries if s['n_true_relevant_in_topk'] > 0),
        'llm_assisted_hit_at_k': sum(s['llm_assisted_hit_at_k'] for s in summaries),
        'score_distribution': dict(
            (str(s), sum(t['score_distribution'][str(s)] for t in summaries))
            for s in range(1, 6)),
        'citations': dict(
            (k, sum(t['citations'][k] for t in summaries))
            for k in ('unverified', 'self_cited', 'none_given', 'demoted')),
    }
    totals['rescued_rate'] = (round(totals['llm_rescued'] / float(totals['n_ranking_candidates']), 3)
                              if totals['n_ranking_candidates'] else None)
    # Users whose every non-relevant item was rescued: the judge told us nothing
    # about them, and a high count here means the rubric is too permissive.
    totals['users_rescuing_everything'] = sum(
        1 for s in summaries if s['n_candidates'] and s['llm_rescued_in_topk'] == s['n_candidates'])

    probes = [s['probe'] for s in summaries if 'probe' in s]
    if probes:
        n_probe = sum(p['n_probes'] for p in probes)
        n_rec = sum(p['n_recovered'] for p in probes)
        recall = round(n_rec / float(n_probe), 3) if n_probe else None
        totals['probe'] = {
            'n_probes': n_probe, 'n_recovered': n_rec,
            'recall_on_known_positives': recall,
            # The number that decides whether any of this works. Recall alone is
            # meaningless: a judge flagging 37% of everything scores ~37% on known
            # positives by luck. Only the ratio to that base rate is evidence, and
            # a lift near 1.0 means the judge cannot tell bought from not-bought.
            'base_rate_on_ranking_negatives': totals['rescued_rate'],
            'lift_over_base_rate': (round(recall / totals['rescued_rate'], 2)
                                    if recall and totals['rescued_rate'] else None),
        }
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='ranking_logs/user_ranking_output.json',
                        help='per-user JSON, or a glob for many')
    parser.add_argument('--out-dir', default='ranking_logs/llm_audit')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--mode', choices=['anchor', 'blind'], default='anchor',
                        help='anchor: show which ranked items are truly relevant, as positive '
                             'anchors. blind: hide the labels, so the verdicts stay independent '
                             'of the ground truth and can be validated against it.')
    parser.add_argument('--probe', action='store_true',
                        help='also judge the held-out items, unlabelled and shuffled in. They '
                             'are known positives, so recall on them measures the judge.')
    parser.add_argument('--num-ctx', type=int, default=16384,
                        help="Ollama's default context is small enough to silently truncate a "
                             'long history')
    parser.add_argument('--max-history', type=int, default=80,
                        help='cap on titles listed per domain')
    parser.add_argument('--relevant-threshold', type=int, default=4, choices=[2, 3, 4, 5],
                        help='semantic_consistency at or above which an item counts as a '
                             'plausible false negative (default 4)')
    parser.add_argument('--overwrite', action='store_true',
                        help='re-judge users that already have an audit in --out-dir '
                             '(default: reuse them, so an interrupted run can resume)')
    parser.add_argument('--require-citation', action='store_true',
                        help='demote likely_relevant to uncertain when the cited owned item '
                             'matches nothing the user actually owns')
    parser.add_argument('--think', action='store_true', help='enable the model\'s thinking mode')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--seed', type=int, default=42, help='shuffle seed for --probe')
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input)) if any(c in args.input for c in '*?[') else [args.input]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit('no input JSON matched {} -- run extract_user_ranking.py first'
                         .format(args.input))

    print('Auditing {} user(s) with {} (mode={}{})'.format(
        len(paths), args.model, args.mode, ', probe' if args.probe else ''))
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    summaries, failed = [], []
    for path in paths:
        try:
            summary = audit_user(path, args, rng)
        except Exception as exc:  # one bad user must not discard the whole batch
            print('  FAILED {}: {}: {}'.format(os.path.basename(path),
                                               type(exc).__name__, str(exc)[:200]))
            failed.append(os.path.basename(path))
            continue
        if summary:
            summaries.append(summary)
    if failed:
        print('\n{} user(s) failed: {}'.format(len(failed), ', '.join(failed)))
        print('re-run the same command to retry only those (finished users are reused)')
    if not summaries:
        raise SystemExit('no user was audited successfully')

    if len(summaries) > 1:
        totals = aggregate(summaries, args)
        rows = sorted(summaries, key=lambda s: -s['llm_rescued_in_topk'])
        lines = ['# LLM Relevance Audit -- {} users'.format(totals['n_users']), '',
                 '```json', json.dumps(totals, indent=2), '```', '',
                 '## Per user', '',
                 '| User | Cands | Rescued | True rel. in top-k | Bad citations |',
                 '|---|---|---|---|---|']
        for s in rows:
            lines.append('| {} | {} | {} | {} | {} |'.format(
                s['external_id'], s['n_candidates'], s['llm_rescued_in_topk'],
                s['n_true_relevant_in_topk'], s['citations']['unverified']))
        lines.append('')
        lines.append('> `llm_*` figures treat the model\'s verdicts as labels: a hypothesis '
                     'about false negatives in the test set, not a measured result.')
        with open(os.path.join(args.out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'totals': totals, 'users': rows}, f, indent=2, ensure_ascii=False)

        print('\n{} users: {} of {} non-relevant items flagged as plausible false negatives '
              '({}%)'.format(totals['n_users'], totals['llm_rescued'], totals['n_candidates'],
                             round(100 * (totals['rescued_rate'] or 0))))
        print('measured Hit@k: {}/{} users | LLM-assisted: {}/{}'.format(
            totals['measured_hit_at_k'], totals['n_users'],
            totals['llm_assisted_hit_at_k'], totals['n_users']))
        print('bad citations: {} ({} self-cited) | users where everything was rescued: {}'.format(
            totals['citations']['unverified'], totals['citations']['self_cited'],
            totals['users_rescuing_everything']))
        if 'probe' in totals:
            probe = totals['probe']
            print('probe recall on known positives: {} (n={}) vs base rate {} '
                  '-> lift {}x'.format(probe['recall_on_known_positives'], probe['n_probes'],
                                       probe['base_rate_on_ranking_negatives'],
                                       probe['lift_over_base_rate']))
        print('-> {}/summary.md'.format(args.out_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
