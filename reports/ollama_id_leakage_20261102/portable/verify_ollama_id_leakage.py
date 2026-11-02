#!/usr/bin/env python3
"""Portable Amazon Cloth-Elec ID recall audit; Python 3.8+, no pip packages.

The fixture embedded below contains exactly the samples/prompts used in the
Gemma/Llama/Qwen audit. Reference answers are NEVER sent in the ID-only requests.
Run --help for options. No models are automatically downloaded.
"""
import argparse
import collections
import copy
import difflib
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FIELDS = {'item_title', 'review_title', 'review_text'}
PRIMARY = {'ids_cautious', 'ids_direct'}
CONDITIONS = ['ids_cautious', 'ids_direct', 'swapped_user', 'ids_plus_title', 'transport']
TRANSPORT = {'item_title': 'SYNTHETIC LANTERN 7281', 'review_title': 'Transport check',
             'review_text': 'The violet umbrella waited beside seven copper clocks.'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_text(path, text):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def save(path, obj):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def words(value):
    text = unicodedata.normalize('NFKC', html.unescape(value or '')).casefold()
    return re.findall(r'\w+', text)


def abstains(value):
    return value is None or not str(value).strip() or ' '.join(words(value)) == 'null'


def compare(value, truth):
    p, t = words(value), words(truth)
    match = difflib.SequenceMatcher(None, p, t, autojunk=False).find_longest_match()
    overlap = sum((collections.Counter(p) & collections.Counter(t)).values())
    return {'literal_exact': not abstains(value) and value.strip() == truth.strip(),
            'normalized_exact': not abstains(value) and p == t,
            'word_f1': round(2 * overlap / (len(p) + len(t)), 4) if p and t else 0,
            'longest_contiguous_words': match.size,
            'longest_contiguous_text': ' '.join(t[match.b:match.b + match.size])}


def parse_answer(response):
    raw = response.get('message', {}).get('content', '')
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != FIELDS:
        return None
    if any(value is not None and not isinstance(value, str) for value in obj.values()):
        return None
    return obj


def usable(record):
    response = record.get('response', {})
    return (not record.get('error') and response.get('done') is True
            and response.get('done_reason') == 'stop' and parse_answer(response) is not None)


def http(base, endpoint, timeout, payload=None):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = urllib.request.Request(base.rstrip('/') + endpoint, data=data,
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode('utf-8', errors='replace')
        raise RuntimeError('Ollama HTTP {} at {}: {}'.format(exc.code, endpoint, detail)) from exc


def prepare(args):
    fixture = json.loads(_EMBEDDED_FIXTURE_JSON)
    if digest(fixture) != _FIXTURE_SHA256:
        raise ValueError('Embedded test fixture checksum mismatch.')
    requests = copy.deepcopy(fixture['requests'])
    for row in requests:
        row['payload']['model'] = args.model
        if args.num_gpu is not None:
            row['payload']['options']['num_gpu'] = args.num_gpu
        if args.num_ctx != 4096:
            row['payload']['options']['num_ctx'] = args.num_ctx
        if args.max_tokens != 1536:
            row['payload']['options']['num_predict'] = args.max_tokens
    if len(requests) != 65 or len({r['probe_id'] for r in requests}) != 65:
        raise ValueError('Expected the frozen set of 65 distinct probes.')
    samples = {s['sample_id']: s for s in fixture['ground_truth']['samples']}
    for row in requests:
        payload = row['payload']
        if len(payload['messages']) != 2:
            raise ValueError('Each request must have its own fresh two-message context.')
        if row['condition'] in PRIMARY:
            reference = samples[row['reference_id']]
            text = canonical(payload)
            if reference['review_text'] in text or reference['item_title'] in text:
                raise ValueError('Ground truth unexpectedly included in an ID-only prompt.')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {'model': args.model, 'base_url': args.base_url.rstrip('/'),
              'fixture_sha256': _FIXTURE_SHA256, 'requests_sha256': digest(requests),
              'expected_quantization': args.expected_quantization,
              'same_payloads_as_baseline_except_model': args.num_gpu is None
                  and args.num_ctx == 4096 and args.max_tokens == 1536}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if read(manifest_path)['config'] != config:
            raise ValueError('Output directory belongs to different settings/model/server. '
                             'Use a new --output-dir to preserve the previous run.')
    else:
        save(manifest_path, {'prepared_at_utc': utc_now(), 'config': config,
                            'samples': 25, 'requests': 65,
                            'source_audit': '2026-11-02 Amazon Cloth-Elec ID recall',
                            'source_ground_truth_sha256': digest(fixture['ground_truth'])})
    for name, obj in [('requests.json', requests), ('ground_truth.json', fixture['ground_truth'])]:
        path = out / name
        if path.exists() and read(path) != obj:
            raise ValueError('Saved {} has changed. Use a new output directory.'.format(name))
        if not path.exists():
            save(path, obj)
    return out, requests, fixture


def load_attempts(out, requests):
    expected = {r['probe_id']: digest(r['payload']) for r in requests}
    latest, counts = {}, collections.Counter()
    for path in sorted((out / 'attempts').glob('*.json')):
        row = read(path)
        probe = row['probe_id']
        if probe not in expected or row.get('request_sha256') != expected[probe]:
            raise ValueError('Saved attempt does not match the prepared requests: {}'.format(path))
        counts[probe] = max(counts[probe], row['attempt'])
        if probe not in latest or row['attempt'] > latest[probe]['attempt']:
            latest[probe] = row
    return latest, counts


def model_metadata(args, out):
    tags = http(args.base_url, '/api/tags', args.timeout)
    installed = [m for m in tags.get('models', [])
                 if args.model in (m.get('name'), m.get('model'))
                 or args.model + ':latest' in (m.get('name'), m.get('model'))]
    if not installed:
        names = ', '.join(m.get('name', '') for m in tags.get('models', []))
        raise ValueError('Model {!r} is not installed on this server. Supply the exact '
                         '`ollama list` name with --model. Available: {}. '
                         'This script does not pull models.'.format(args.model, names))
    show = http(args.base_url, '/api/show', args.timeout, {'model': args.model})
    details = show.get('details', {})
    quant = details.get('quantization_level')
    if args.expected_quantization and quant != args.expected_quantization:
        raise ValueError('Expected quantization {}, server reports {}.'.format(args.expected_quantization, quant))
    model_digest = installed[0].get('digest')
    if not model_digest:
        raise ValueError('Ollama did not return the model digest; cannot verify resume identity.')
    path = out / 'model_metadata.json'
    if path.exists() and read(path)['model_digest'] != model_digest:
        raise ValueError('Model digest changed since this run started. Use a new --output-dir.')
    metadata = {'checked_at_utc': utc_now(), 'model': args.model, 'model_digest': model_digest,
                'version': http(args.base_url, '/api/version', args.timeout),
                'installed_model': installed[0],
                'show': {k: v for k, v in show.items() if k not in {'tensors', 'license'}}}
    save(path, metadata)
    print('Model: {} | parameters: {} | quantization: {} | digest: {}'.format(
          args.model, details.get('parameter_size'), quant, model_digest), flush=True)
    return metadata


def analyze(out, requests, fixture, latest):
    samples = {s['sample_id']: s for s in fixture['ground_truth']['samples']}
    rows = []
    raw_lines = []
    for request in requests:
        record = latest.get(request['probe_id'])
        if record is None:
            continue
        raw_lines.append(canonical(record))
        response = record.get('response', {})
        parsed = parse_answer(response)
        valid = parsed is not None
        row = {'probe_id': request['probe_id'], 'condition': request['condition'],
               'sample_id': request['sample_id'], 'reference_id': request['reference_id'],
               'domain': request['domain'], 'attempt': record['attempt'], 'valid_json': valid,
               'usable': usable(record), 'error': record.get('error'),
               'done_reason': response.get('done_reason'), 'parsed': parsed,
               'raw_content': response.get('message', {}).get('content', '')}
        reference = samples.get(request['reference_id'])
        if reference:
            for field in sorted(FIELDS):
                value = parsed[field] if valid else None
                row[field + '_comparison'] = compare(value, reference[field])
            row['both_exact'] = (row['item_title_comparison']['normalized_exact']
                                 and row['review_text_comparison']['normalized_exact'])
        if request['condition'] == 'transport':
            row['transport_pass'] = row['usable'] and parsed == TRANSPORT
        rows.append(row)
    summary = {'model': requests[0]['payload']['model'], 'generated_at_utc': utc_now(),
               'expected_calls': len(requests), 'attempted_calls': len(rows),
               'usable_calls': sum(r['usable'] for r in rows), 'conditions': {}}
    for condition in CONDITIONS:
        part = [r for r in rows if r['condition'] == condition]
        good = [r for r in part if r['usable']]
        stats = {'expected': sum(q['condition'] == condition for q in requests),
                 'attempted': len(part), 'usable': len(good),
                 'errors': sum(bool(r['error']) for r in part),
                 'invalid_json': sum(not r['valid_json'] for r in part),
                 'truncated': sum(r['done_reason'] == 'length' for r in part),
                 'item_answers': sum(not abstains(r['parsed']['item_title']) for r in good),
                 'review_answers': sum(not abstains(r['parsed']['review_text']) for r in good)}
        for field in ['item_title', 'review_text']:
            for metric in ['literal_exact', 'normalized_exact']:
                stats[field + '_' + metric] = sum(r.get(field + '_comparison', {}).get(metric, False) for r in good)
        stats['both_exact'] = sum(r.get('both_exact', False) for r in good)
        summary['conditions'][condition] = stats
    primary = [r for r in rows if r['condition'] in PRIMARY and r['usable']]
    summary['primary'] = {'expected_calls': 50, 'usable_calls': len(primary),
        'distinct_usable_pairs': len({r['sample_id'] for r in primary}),
        'item_exact_calls': sum(r['item_title_comparison']['normalized_exact'] for r in primary),
        'review_exact_calls': sum(r['review_text_comparison']['normalized_exact'] for r in primary),
        'both_exact_calls': sum(r['both_exact'] for r in primary),
        'item_exact_pairs': len({r['sample_id'] for r in primary if r['item_title_comparison']['normalized_exact']}),
        'review_exact_pairs': len({r['sample_id'] for r in primary if r['review_text_comparison']['normalized_exact']})}
    control = next((r for r in rows if r['condition'] == 'transport'), None)
    summary['transport_pass'] = bool(control and control['transport_pass'])
    summary['complete_and_usable'] = len(rows) == len(requests) and all(r['usable'] for r in rows)
    summary['audit_checks_pass'] = summary['complete_and_usable'] and summary['transport_pass']
    save(out / 'summary.json', summary)
    save(out / 'scored.json', rows)
    atomic_text(out / 'responses.jsonl', '\n'.join(raw_lines) + ('\n' if raw_lines else ''))
    report(out, requests, fixture, rows, summary)
    return summary


def report(out, requests, fixture, rows, summary):
    primary = summary['primary']
    lines = ['# Amazon Cloth-Elec: {} ID recall audit'.format(summary['model']), '',
        'Generated (UTC): {}'.format(summary['generated_at_utc']), '',
        '**Audit checks passed:** {}. Calls attempted: {}/65; usable: {}/65; synthetic control: {}.'.format(
            summary['audit_checks_pass'], summary['attempted_calls'], summary['usable_calls'], summary['transport_pass']), '',
        '| Condition | Expected | Attempted | Usable | Exact item title | Exact review body | Both | Errors / invalid JSON / truncated |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition, x in summary['conditions'].items():
        item = x['item_title_normalized_exact'] if condition not in {'ids_plus_title', 'transport'} else 'n/a'
        review = x['review_text_normalized_exact'] if condition != 'transport' else 'n/a'
        both = x['both_exact'] if condition not in {'ids_plus_title', 'transport'} else 'n/a'
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} / {} / {} |'.format(
            condition, x['expected'], x['attempted'], x['usable'], item, review, both,
            x['errors'], x['invalid_json'], x['truncated']))
    lines += ['', 'Exact scores in this table use normalized complete strings and only usable responses. '
        'Usable means complete (`done=true`, `done_reason=stop`) with the requested JSON schema. '
        'Failed, invalid, missing, and truncated responses are not evidence of a miss. '
        'Both JSON null and the string "null" count as abstentions.', '']
    if not summary['audit_checks_pass']:
        lines += ['**Incomplete or failed checks:** Inspect failures/control output and rerun the same command '
                  'to retry unusable responses. Do not interpret zero matches as evidence against leakage yet.', '']
    elif primary['review_exact_calls'] or primary['item_exact_calls']:
        lines += ['**Matches observed:** Inspect the matching records below. Product-title recall alone '
                  'does not establish recall of a particular user review. A short generic review match '
                  'is weaker evidence than a distinctive long passage tied to the correct user.', '']
    else:
        lines += ['**No exact item-title or user-review recovery was observed from IDs under these settings.** '
                  'Inspect nonempty guesses for product-name variants and partial memorization before drawing conclusions.', '']
    lines += ['## Comparison with the previous runs', '',
        '| Model | Usable primary calls | Exact item titles | Exact reviews | Both |',
        '|---|---:|---:|---:|---:|']
    for b in fixture['baselines']:
        lines.append('| {} | {} | {} | {} | {} |'.format(b['model'], b['primary_calls'],
            b['exact_title_calls'], b['exact_review_calls'], b['both_exact_calls']))
    lines += ['| {} (this run) | {} | {} | {} | {} |'.format(summary['model'], primary['usable_calls'],
              primary['item_exact_calls'], primary['review_exact_calls'], primary['both_exact_calls']), '',
        'The baseline data above are saved observations from 2026-11-02; they are not rerun by this script. '
        'The current run uses the identical 25 pairs and 65 prompt payloads, changing only the model '
        'unless explicit context/token/GPU overrides were supplied (recorded in manifest.json).', '',
        '## Method and interpretation', '',
        '15 held-out Electronics pairs (3 per saved review-length quintile) and 10 Clothing pairs '
        '(2 per body-length bin: 5-10, 11-30, 31-60, 61-120, 121-400 words), 25 distinct users and items. '
        'Two prompt styles produce 50 primary calls. Controls: 10 swapped-user pairs absent from '
        'the local interaction files/compact reviews, 4 title-provided probes, and 1 synthetic copy probe. '
        'Absence of swapped pairs from all Amazon data is not established.', '',
        'Every inference uses an independent two-message conversation, no tools, no retrieval, no history. '
        'Primary prompts contain original user_id/parent_asin and dataset/category labels, but no titles, '
        'ratings, reference reviews, or user histories. The four title-provided probes are separate. '
        'No reference review is included in any real recall prompt. '
        'Default settings: temperature=0, seed=42, think=false, num_ctx=4096, num_predict=1536, format=json.', '',
        'Literal comparisons trim outer whitespace. Normalized comparisons use HTML unescaping, Unicode NFKC, '
        'case folding, and word tokenization, matching the previous audit. Token F1 and longest contiguous '
        'word overlap are inspection aids. A model may know a product without knowing this user\'s review. '
        'Swapped-user matches are compared to the original user\'s reference only as a control, not as recall hits. '
        'The synthetic probe checks transport/formatting; it is not a positive test of pretrained memory.', '',
        'Failure to retrieve these records does not prove absence from pretraining, exclude partial '
        'memorization, or establish that explanation/judge evaluations with richer inputs are leakage-free. '
        'Exact item-title scoring may miss legitimate title variants; inspect nonempty responses. '
        'Quantization, model digest, server metadata, and overridden settings are saved for interpretation.', '',
        '## Full comparisons', '']
    samples = {s['sample_id']: s for s in fixture['ground_truth']['samples']}
    by_probe = {q['probe_id']: q for q in requests}
    for row in rows:
        q = by_probe[row['probe_id']]
        lines += ['### {}'.format(row['probe_id']), '',
                  'Usable: {}; user: `{}`; item: `{}`.'.format(row['usable'], q['user_id'], q['item_id']), '']
        s = samples.get(row['reference_id'])
        if s:
            lines += ['True item: {}'.format(s['item_title']), '',
                      'True review title: {}'.format(s['review_title']), '',
                      'True review body:', '', '> ' + s['review_text'].replace('\n', '\n> '), '']
        lines += ['Model answer:', '', '```json', json.dumps(row['parsed'], ensure_ascii=False, indent=2), '```', '']
        if not row['usable']:
            lines += ['Error: {}; completion reason: {}.'.format(row['error'], row['done_reason']), '',
                      'Raw output:', '', '```text', row['raw_content'], '```', '']
        if s:
            lines += ['Exact item: {}; exact review: {}; longest review overlap: {} words.'.format(
                row['item_title_comparison']['normalized_exact'], row['review_text_comparison']['normalized_exact'],
                row['review_text_comparison']['longest_contiguous_words']), '']
    atomic_text(out / 'report.md', '\n'.join(lines) + '\n')


def run(args):
    out, requests, fixture = prepare(args)
    print('Output: {}'.format(out), flush=True)
    if args.dry_run:
        print('Prepared 65 requests / 25 pairs. No server contacted. No inference performed.', flush=True)
        return 0
    latest, counts = load_attempts(out, requests)
    if not args.analyze_only:
        metadata = model_metadata(args, out)
        allowed_models = {args.model, args.model + ':latest',
                          metadata['installed_model'].get('name'), metadata['installed_model'].get('model')}
        allowed_models.discard(None)
        attempts_dir = out / 'attempts'
        attempts_dir.mkdir(exist_ok=True)
        try:
            for index, request in enumerate(requests, 1):
                probe = request['probe_id']
                if (probe in latest and usable(latest[probe]) and
                        (request['condition'] != 'transport' or
                         parse_answer(latest[probe]['response']) == TRANSPORT)):
                    continue
                record = {'probe_id': probe, 'attempt': counts[probe] + 1,
                          'request_sha256': digest(request['payload']), 'started_at_utc': utc_now()}
                started = time.monotonic()
                try:
                    record['response'] = http(args.base_url, '/api/chat', args.timeout, request['payload'])
                    if record['response'].get('model') not in allowed_models:
                        raise ValueError('Response model differs from the selected installed model.')
                except Exception as exc:
                    record['error'] = '{}: {}'.format(type(exc).__name__, exc)
                record['elapsed_seconds'] = round(time.monotonic() - started, 3)
                save(attempts_dir / '{}__{:04d}.json'.format(probe, record['attempt']), record)
                latest[probe] = record
                counts[probe] = record['attempt']
                summary = analyze(out, requests, fixture, latest)
                response = record.get('response', {})
                print('{}/65 {} | {:.2f}s | {} | {}'.format(index, probe, record['elapsed_seconds'],
                    'usable' if usable(record) else 'UNUSABLE',
                    record.get('error') or response.get('message', {}).get('content', '')[:180]), flush=True)
                if record.get('error'):
                    print('Stopped after a request error. Fix the server/connection and rerun to resume.', file=sys.stderr)
                    return 2
        except KeyboardInterrupt:
            analyze(out, requests, fixture, latest)
            print('\nInterrupted. Completed attempts are saved; rerun the same command to resume.', file=sys.stderr)
            return 130
    summary = analyze(out, requests, fixture, latest)
    p = summary['primary']
    print('Primary usable: {}/50 | exact items: {} | exact reviews: {} | both: {}'.format(
          p['usable_calls'], p['item_exact_calls'], p['review_exact_calls'], p['both_exact_calls']), flush=True)
    print('Audit checks passed: {} | report: {}'.format(summary['audit_checks_pass'], out / 'report.md'), flush=True)
    return 0 if summary['audit_checks_pass'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='qwen3.8:27b', help='Exact installed Ollama tag (including your quantization alias).')
    parser.add_argument('--base-url', default='http://localhost:11434')
    parser.add_argument('--output-dir', type=Path, help='Results directory; reuse it to resume, use a new one for another build/config.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Prepare and validate inputs without any server access.')
    mode.add_argument('--analyze-only', action='store_true', help='Recompute reports from saved attempts without server access.')
    parser.add_argument('--timeout', type=float, default=900, help='HTTP timeout in seconds per request (default: 900).')
    parser.add_argument('--expected-quantization', help='Optional exact quantization check, e.g. the value reported by Ollama.')
    parser.add_argument('--num-gpu', type=int, help='Optional Ollama num_gpu override; 0 requests CPU execution.')
    parser.add_argument('--num-ctx', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=1536)
    args = parser.parse_args(argv)
    if not args.model.strip() or args.timeout <= 0 or args.num_ctx <= 0 or args.max_tokens <= 0:
        parser.error('Model must be nonempty and timeout/context/token limits must be positive.')
    if args.num_gpu is not None and args.num_gpu < 0:
        parser.error('--num-gpu must be nonnegative.')
    if args.output_dir is None:
        slug = re.sub(r'[^A-Za-z0-9._-]+', '_', args.model).strip('._') or 'model'
        args.output_dir = Path.cwd() / 'reports' / ('ollama_id_leakage_' + slug)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        return 2


# Frozen source data are inserted here during authoring; no external files needed.
_EMBEDDED_FIXTURE_JSON = r'''
{
  "requests": [
    {
      "probe_id": "ids_cautious__elec_01",
      "sample_id": "elec_01",
      "reference_id": "elec_01",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AFCMUVTNPRQW4XQEI5QOWF6ILB2A",
      "item_id": "B01E6MTP0S",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFCMUVTNPRQW4XQEI5QOWF6ILB2A\nparent_asin (item_id): B01E6MTP0S\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_02",
      "sample_id": "elec_02",
      "reference_id": "elec_02",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA",
      "item_id": "B0C49KQRX4",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA\nparent_asin (item_id): B0C49KQRX4\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_03",
      "sample_id": "elec_03",
      "reference_id": "elec_03",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AGUFMAT6HE42URXYLNSZR5XKJ34Q",
      "item_id": "B00GXOOL4S",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGUFMAT6HE42URXYLNSZR5XKJ34Q\nparent_asin (item_id): B00GXOOL4S\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_04",
      "sample_id": "elec_04",
      "reference_id": "elec_04",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AFBU2OKYDAWEKQZSGU4KWUMXUTOQ",
      "item_id": "B007ZG03DG",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFBU2OKYDAWEKQZSGU4KWUMXUTOQ\nparent_asin (item_id): B007ZG03DG\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_05",
      "sample_id": "elec_05",
      "reference_id": "elec_05",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AGONYK3OBT7GPDSXMYCVK3MN26VA",
      "item_id": "B073JYC4XM",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGONYK3OBT7GPDSXMYCVK3MN26VA\nparent_asin (item_id): B073JYC4XM\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_06",
      "sample_id": "elec_06",
      "reference_id": "elec_06",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AGYDGTJJDOFV2RWXOCLTZMHCAMKA",
      "item_id": "B00L9BUM48",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGYDGTJJDOFV2RWXOCLTZMHCAMKA\nparent_asin (item_id): B00L9BUM48\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_07",
      "sample_id": "elec_07",
      "reference_id": "elec_07",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AEIRRIZSGY3BW7QG5O7P2AFLTDQA",
      "item_id": "B0BNW6HXW1",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AEIRRIZSGY3BW7QG5O7P2AFLTDQA\nparent_asin (item_id): B0BNW6HXW1\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_08",
      "sample_id": "elec_08",
      "reference_id": "elec_08",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AFT7EBVQSP3CORRXUWMYW6MD2T5Q",
      "item_id": "B07VTK654B",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFT7EBVQSP3CORRXUWMYW6MD2T5Q\nparent_asin (item_id): B07VTK654B\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_09",
      "sample_id": "elec_09",
      "reference_id": "elec_09",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AHMZ7E3STONPGXXC2EDWMO5SRHOQ",
      "item_id": "B0846MH1CJ",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AHMZ7E3STONPGXXC2EDWMO5SRHOQ\nparent_asin (item_id): B0846MH1CJ\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_10",
      "sample_id": "elec_10",
      "reference_id": "elec_10",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AGUMRMORGZZPZX7MYBWRT4CATTPQ",
      "item_id": "B01EZWFNKA",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGUMRMORGZZPZX7MYBWRT4CATTPQ\nparent_asin (item_id): B01EZWFNKA\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_11",
      "sample_id": "elec_11",
      "reference_id": "elec_11",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AEZ5NUGJPQVHL5SGP3QMFFGL3LIA",
      "item_id": "B0BQJB1SS8",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AEZ5NUGJPQVHL5SGP3QMFFGL3LIA\nparent_asin (item_id): B0BQJB1SS8\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_12",
      "sample_id": "elec_12",
      "reference_id": "elec_12",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AFLJT24IDBBBAI5XDTCAC5FDLA6Q",
      "item_id": "B0BT7VSWH4",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFLJT24IDBBBAI5XDTCAC5FDLA6Q\nparent_asin (item_id): B0BT7VSWH4\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_13",
      "sample_id": "elec_13",
      "reference_id": "elec_13",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AHZTIGQW7ZWYVM7NYLVV5AZHCFZA",
      "item_id": "B0083J78GY",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AHZTIGQW7ZWYVM7NYLVV5AZHCFZA\nparent_asin (item_id): B0083J78GY\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_14",
      "sample_id": "elec_14",
      "reference_id": "elec_14",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AFJXPTIVTQGVSQ4ILSSWQS64HEGA",
      "item_id": "B000GIW4XC",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFJXPTIVTQGVSQ4ILSSWQS64HEGA\nparent_asin (item_id): B000GIW4XC\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__elec_15",
      "sample_id": "elec_15",
      "reference_id": "elec_15",
      "condition": "ids_cautious",
      "domain": "Elec",
      "user_id": "AELO5FBBZJZB5GC5QCOLUCIUW77A",
      "item_id": "B0BVNZZBY8",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AELO5FBBZJZB5GC5QCOLUCIUW77A\nparent_asin (item_id): B0BVNZZBY8\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_01",
      "sample_id": "cloth_01",
      "reference_id": "cloth_01",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AHYCHLTEQHVIAQWUETYJTE3DEKFA",
      "item_id": "B0BH113WW9",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AHYCHLTEQHVIAQWUETYJTE3DEKFA\nparent_asin (item_id): B0BH113WW9\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_02",
      "sample_id": "cloth_02",
      "reference_id": "cloth_02",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AHIAMFRGZ44A6JCFAAJNONFYI4EQ",
      "item_id": "B0C4HS52X9",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AHIAMFRGZ44A6JCFAAJNONFYI4EQ\nparent_asin (item_id): B0C4HS52X9\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_03",
      "sample_id": "cloth_03",
      "reference_id": "cloth_03",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AEODJ6TRQBC4BYQSWCIAWK7LEVOQ",
      "item_id": "B0C2XZDN8G",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEODJ6TRQBC4BYQSWCIAWK7LEVOQ\nparent_asin (item_id): B0C2XZDN8G\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_04",
      "sample_id": "cloth_04",
      "reference_id": "cloth_04",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AFMN2ITMRKPSIW3GGQFUGTXYHNIQ",
      "item_id": "B019961SG0",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AFMN2ITMRKPSIW3GGQFUGTXYHNIQ\nparent_asin (item_id): B019961SG0\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_05",
      "sample_id": "cloth_05",
      "reference_id": "cloth_05",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AEHAYO7YTALZUKJTS3QBEQNE3KVQ",
      "item_id": "B0BV25DTYP",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEHAYO7YTALZUKJTS3QBEQNE3KVQ\nparent_asin (item_id): B0BV25DTYP\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_06",
      "sample_id": "cloth_06",
      "reference_id": "cloth_06",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AGGYYRRZ6UPNN5SHZZM5STQLGF6A",
      "item_id": "B0B7RJX4LB",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AGGYYRRZ6UPNN5SHZZM5STQLGF6A\nparent_asin (item_id): B0B7RJX4LB\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_07",
      "sample_id": "cloth_07",
      "reference_id": "cloth_07",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AEQDVQBSQGKAFZXI4YOIOOEPWKWA",
      "item_id": "B0C2L2HN19",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEQDVQBSQGKAFZXI4YOIOOEPWKWA\nparent_asin (item_id): B0C2L2HN19\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_08",
      "sample_id": "cloth_08",
      "reference_id": "cloth_08",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AGTD6WFUY6AKSV3NR6ZDC3LOVIHA",
      "item_id": "B0B67LKB2F",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AGTD6WFUY6AKSV3NR6ZDC3LOVIHA\nparent_asin (item_id): B0B67LKB2F\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_09",
      "sample_id": "cloth_09",
      "reference_id": "cloth_09",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AGDIO3EQA2WNLRP3ZMWA37E2UMYA",
      "item_id": "B0BN4T68WV",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AGDIO3EQA2WNLRP3ZMWA37E2UMYA\nparent_asin (item_id): B0BN4T68WV\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_cautious__cloth_10",
      "sample_id": "cloth_10",
      "reference_id": "cloth_10",
      "condition": "ids_cautious",
      "domain": "Cloth",
      "user_id": "AFAC5JZAWUPAYENYHI4EV7PEWR6Q",
      "item_id": "B0BZ82TCMX",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AFAC5JZAWUPAYENYHI4EV7PEWR6Q\nparent_asin (item_id): B0BZ82TCMX\nCan you identify this exact product and retrieve the exact review this user wrote for it, using only what you already know? Return the exact product title, original review title, and complete verbatim review body. Do not infer a review from typical product opinions or invent one. Use null independently for each unknown field. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_01",
      "sample_id": "elec_01",
      "reference_id": "elec_01",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AFCMUVTNPRQW4XQEI5QOWF6ILB2A",
      "item_id": "B01E6MTP0S",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFCMUVTNPRQW4XQEI5QOWF6ILB2A\nparent_asin (item_id): B01E6MTP0S\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_02",
      "sample_id": "elec_02",
      "reference_id": "elec_02",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA",
      "item_id": "B0C49KQRX4",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA\nparent_asin (item_id): B0C49KQRX4\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_03",
      "sample_id": "elec_03",
      "reference_id": "elec_03",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AGUFMAT6HE42URXYLNSZR5XKJ34Q",
      "item_id": "B00GXOOL4S",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGUFMAT6HE42URXYLNSZR5XKJ34Q\nparent_asin (item_id): B00GXOOL4S\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_04",
      "sample_id": "elec_04",
      "reference_id": "elec_04",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AFBU2OKYDAWEKQZSGU4KWUMXUTOQ",
      "item_id": "B007ZG03DG",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFBU2OKYDAWEKQZSGU4KWUMXUTOQ\nparent_asin (item_id): B007ZG03DG\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_05",
      "sample_id": "elec_05",
      "reference_id": "elec_05",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AGONYK3OBT7GPDSXMYCVK3MN26VA",
      "item_id": "B073JYC4XM",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGONYK3OBT7GPDSXMYCVK3MN26VA\nparent_asin (item_id): B073JYC4XM\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_06",
      "sample_id": "elec_06",
      "reference_id": "elec_06",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AGYDGTJJDOFV2RWXOCLTZMHCAMKA",
      "item_id": "B00L9BUM48",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGYDGTJJDOFV2RWXOCLTZMHCAMKA\nparent_asin (item_id): B00L9BUM48\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_07",
      "sample_id": "elec_07",
      "reference_id": "elec_07",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AEIRRIZSGY3BW7QG5O7P2AFLTDQA",
      "item_id": "B0BNW6HXW1",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AEIRRIZSGY3BW7QG5O7P2AFLTDQA\nparent_asin (item_id): B0BNW6HXW1\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_08",
      "sample_id": "elec_08",
      "reference_id": "elec_08",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AFT7EBVQSP3CORRXUWMYW6MD2T5Q",
      "item_id": "B07VTK654B",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFT7EBVQSP3CORRXUWMYW6MD2T5Q\nparent_asin (item_id): B07VTK654B\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_09",
      "sample_id": "elec_09",
      "reference_id": "elec_09",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AHMZ7E3STONPGXXC2EDWMO5SRHOQ",
      "item_id": "B0846MH1CJ",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AHMZ7E3STONPGXXC2EDWMO5SRHOQ\nparent_asin (item_id): B0846MH1CJ\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_10",
      "sample_id": "elec_10",
      "reference_id": "elec_10",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AGUMRMORGZZPZX7MYBWRT4CATTPQ",
      "item_id": "B01EZWFNKA",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGUMRMORGZZPZX7MYBWRT4CATTPQ\nparent_asin (item_id): B01EZWFNKA\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_11",
      "sample_id": "elec_11",
      "reference_id": "elec_11",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AEZ5NUGJPQVHL5SGP3QMFFGL3LIA",
      "item_id": "B0BQJB1SS8",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AEZ5NUGJPQVHL5SGP3QMFFGL3LIA\nparent_asin (item_id): B0BQJB1SS8\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_12",
      "sample_id": "elec_12",
      "reference_id": "elec_12",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AFLJT24IDBBBAI5XDTCAC5FDLA6Q",
      "item_id": "B0BT7VSWH4",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFLJT24IDBBBAI5XDTCAC5FDLA6Q\nparent_asin (item_id): B0BT7VSWH4\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_13",
      "sample_id": "elec_13",
      "reference_id": "elec_13",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AHZTIGQW7ZWYVM7NYLVV5AZHCFZA",
      "item_id": "B0083J78GY",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AHZTIGQW7ZWYVM7NYLVV5AZHCFZA\nparent_asin (item_id): B0083J78GY\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_14",
      "sample_id": "elec_14",
      "reference_id": "elec_14",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AFJXPTIVTQGVSQ4ILSSWQS64HEGA",
      "item_id": "B000GIW4XC",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFJXPTIVTQGVSQ4ILSSWQS64HEGA\nparent_asin (item_id): B000GIW4XC\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__elec_15",
      "sample_id": "elec_15",
      "reference_id": "elec_15",
      "condition": "ids_direct",
      "domain": "Elec",
      "user_id": "AELO5FBBZJZB5GC5QCOLUCIUW77A",
      "item_id": "B0BVNZZBY8",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AELO5FBBZJZB5GC5QCOLUCIUW77A\nparent_asin (item_id): B0BVNZZBY8\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_01",
      "sample_id": "cloth_01",
      "reference_id": "cloth_01",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AHYCHLTEQHVIAQWUETYJTE3DEKFA",
      "item_id": "B0BH113WW9",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AHYCHLTEQHVIAQWUETYJTE3DEKFA\nparent_asin (item_id): B0BH113WW9\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_02",
      "sample_id": "cloth_02",
      "reference_id": "cloth_02",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AHIAMFRGZ44A6JCFAAJNONFYI4EQ",
      "item_id": "B0C4HS52X9",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AHIAMFRGZ44A6JCFAAJNONFYI4EQ\nparent_asin (item_id): B0C4HS52X9\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_03",
      "sample_id": "cloth_03",
      "reference_id": "cloth_03",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AEODJ6TRQBC4BYQSWCIAWK7LEVOQ",
      "item_id": "B0C2XZDN8G",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEODJ6TRQBC4BYQSWCIAWK7LEVOQ\nparent_asin (item_id): B0C2XZDN8G\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_04",
      "sample_id": "cloth_04",
      "reference_id": "cloth_04",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AFMN2ITMRKPSIW3GGQFUGTXYHNIQ",
      "item_id": "B019961SG0",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AFMN2ITMRKPSIW3GGQFUGTXYHNIQ\nparent_asin (item_id): B019961SG0\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_05",
      "sample_id": "cloth_05",
      "reference_id": "cloth_05",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AEHAYO7YTALZUKJTS3QBEQNE3KVQ",
      "item_id": "B0BV25DTYP",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEHAYO7YTALZUKJTS3QBEQNE3KVQ\nparent_asin (item_id): B0BV25DTYP\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_06",
      "sample_id": "cloth_06",
      "reference_id": "cloth_06",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AGGYYRRZ6UPNN5SHZZM5STQLGF6A",
      "item_id": "B0B7RJX4LB",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AGGYYRRZ6UPNN5SHZZM5STQLGF6A\nparent_asin (item_id): B0B7RJX4LB\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_07",
      "sample_id": "cloth_07",
      "reference_id": "cloth_07",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AEQDVQBSQGKAFZXI4YOIOOEPWKWA",
      "item_id": "B0C2L2HN19",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEQDVQBSQGKAFZXI4YOIOOEPWKWA\nparent_asin (item_id): B0C2L2HN19\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_08",
      "sample_id": "cloth_08",
      "reference_id": "cloth_08",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AGTD6WFUY6AKSV3NR6ZDC3LOVIHA",
      "item_id": "B0B67LKB2F",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AGTD6WFUY6AKSV3NR6ZDC3LOVIHA\nparent_asin (item_id): B0B67LKB2F\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_09",
      "sample_id": "cloth_09",
      "reference_id": "cloth_09",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AGDIO3EQA2WNLRP3ZMWA37E2UMYA",
      "item_id": "B0BN4T68WV",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AGDIO3EQA2WNLRP3ZMWA37E2UMYA\nparent_asin (item_id): B0BN4T68WV\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_direct__cloth_10",
      "sample_id": "cloth_10",
      "reference_id": "cloth_10",
      "condition": "ids_direct",
      "domain": "Cloth",
      "user_id": "AFAC5JZAWUPAYENYHI4EV7PEWR6Q",
      "item_id": "B0BZ82TCMX",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AFAC5JZAWUPAYENYHI4EV7PEWR6Q\nparent_asin (item_id): B0BZ82TCMX\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_elec_01",
      "sample_id": "swapped_elec_01",
      "reference_id": "elec_01",
      "condition": "swapped_user",
      "domain": "Elec",
      "user_id": "AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA",
      "item_id": "B01E6MTP0S",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA\nparent_asin (item_id): B01E6MTP0S\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_elec_02",
      "sample_id": "swapped_elec_02",
      "reference_id": "elec_02",
      "condition": "swapped_user",
      "domain": "Elec",
      "user_id": "AGUFMAT6HE42URXYLNSZR5XKJ34Q",
      "item_id": "B0C49KQRX4",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGUFMAT6HE42URXYLNSZR5XKJ34Q\nparent_asin (item_id): B0C49KQRX4\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_elec_03",
      "sample_id": "swapped_elec_03",
      "reference_id": "elec_03",
      "condition": "swapped_user",
      "domain": "Elec",
      "user_id": "AFBU2OKYDAWEKQZSGU4KWUMXUTOQ",
      "item_id": "B00GXOOL4S",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFBU2OKYDAWEKQZSGU4KWUMXUTOQ\nparent_asin (item_id): B00GXOOL4S\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_elec_04",
      "sample_id": "swapped_elec_04",
      "reference_id": "elec_04",
      "condition": "swapped_user",
      "domain": "Elec",
      "user_id": "AGONYK3OBT7GPDSXMYCVK3MN26VA",
      "item_id": "B007ZG03DG",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGONYK3OBT7GPDSXMYCVK3MN26VA\nparent_asin (item_id): B007ZG03DG\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_elec_05",
      "sample_id": "swapped_elec_05",
      "reference_id": "elec_05",
      "condition": "swapped_user",
      "domain": "Elec",
      "user_id": "AGYDGTJJDOFV2RWXOCLTZMHCAMKA",
      "item_id": "B073JYC4XM",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AGYDGTJJDOFV2RWXOCLTZMHCAMKA\nparent_asin (item_id): B073JYC4XM\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_cloth_01",
      "sample_id": "swapped_cloth_01",
      "reference_id": "cloth_01",
      "condition": "swapped_user",
      "domain": "Cloth",
      "user_id": "AHIAMFRGZ44A6JCFAAJNONFYI4EQ",
      "item_id": "B0BH113WW9",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AHIAMFRGZ44A6JCFAAJNONFYI4EQ\nparent_asin (item_id): B0BH113WW9\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_cloth_02",
      "sample_id": "swapped_cloth_02",
      "reference_id": "cloth_02",
      "condition": "swapped_user",
      "domain": "Cloth",
      "user_id": "AEODJ6TRQBC4BYQSWCIAWK7LEVOQ",
      "item_id": "B0C4HS52X9",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEODJ6TRQBC4BYQSWCIAWK7LEVOQ\nparent_asin (item_id): B0C4HS52X9\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_cloth_03",
      "sample_id": "swapped_cloth_03",
      "reference_id": "cloth_03",
      "condition": "swapped_user",
      "domain": "Cloth",
      "user_id": "AFMN2ITMRKPSIW3GGQFUGTXYHNIQ",
      "item_id": "B0C2XZDN8G",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AFMN2ITMRKPSIW3GGQFUGTXYHNIQ\nparent_asin (item_id): B0C2XZDN8G\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_cloth_04",
      "sample_id": "swapped_cloth_04",
      "reference_id": "cloth_04",
      "condition": "swapped_user",
      "domain": "Cloth",
      "user_id": "AEHAYO7YTALZUKJTS3QBEQNE3KVQ",
      "item_id": "B019961SG0",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AEHAYO7YTALZUKJTS3QBEQNE3KVQ\nparent_asin (item_id): B019961SG0\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "swapped_user__swapped_cloth_05",
      "sample_id": "swapped_cloth_05",
      "reference_id": "cloth_05",
      "condition": "swapped_user",
      "domain": "Cloth",
      "user_id": "AGGYYRRZ6UPNN5SHZZM5STQLGF6A",
      "item_id": "B0BV25DTYP",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AGGYYRRZ6UPNN5SHZZM5STQLGF6A\nparent_asin (item_id): B0BV25DTYP\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_plus_title__elec_01",
      "sample_id": "elec_01",
      "reference_id": "elec_01",
      "condition": "ids_plus_title",
      "domain": "Elec",
      "user_id": "AFCMUVTNPRQW4XQEI5QOWF6ILB2A",
      "item_id": "B01E6MTP0S",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AFCMUVTNPRQW4XQEI5QOWF6ILB2A\nparent_asin (item_id): B01E6MTP0S\nKnown product title: Fintie Rotating Case for iPad 6th / 5th Generation (2018 2017 Model, 9.7 inch), iPad Air 2 / iPad Air 1-360 Degree Rotating Protective Stand Cover Auto Sleep Wake, Composition Book\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_plus_title__elec_15",
      "sample_id": "elec_15",
      "reference_id": "elec_15",
      "condition": "ids_plus_title",
      "domain": "Elec",
      "user_id": "AELO5FBBZJZB5GC5QCOLUCIUW77A",
      "item_id": "B0BVNZZBY8",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Electronics.\nuser_id: AELO5FBBZJZB5GC5QCOLUCIUW77A\nparent_asin (item_id): B0BVNZZBY8\nKnown product title: Yottamaster 10-in-1 USB C Docking Station, USB C Hub with 4K HDMI, VGA, 3 USB 3.0, TF/SD Reader, Ethernet, Audio, 100W Power Delivery Compatible with MacBook, Surface, iPad or Other Type C Devices\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_plus_title__cloth_01",
      "sample_id": "cloth_01",
      "reference_id": "cloth_01",
      "condition": "ids_plus_title",
      "domain": "Cloth",
      "user_id": "AHYCHLTEQHVIAQWUETYJTE3DEKFA",
      "item_id": "B0BH113WW9",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AHYCHLTEQHVIAQWUETYJTE3DEKFA\nparent_asin (item_id): B0BH113WW9\nKnown product title: Hanes Boxer Briefs, Cool Dri Moisture-Wicking Underwear, Cotton No-Ride-up for Men, Multi-Packs Available\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "ids_plus_title__cloth_10",
      "sample_id": "cloth_10",
      "reference_id": "cloth_10",
      "condition": "ids_plus_title",
      "domain": "Cloth",
      "user_id": "AFAC5JZAWUPAYENYHI4EV7PEWR6Q",
      "item_id": "B0BZ82TCMX",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Amazon Reviews dataset; Cloth-Elec common-user 10-core subset.\nCategory: Clothing, Shoes & Jewelry.\nuser_id: AFAC5JZAWUPAYENYHI4EV7PEWR6Q\nparent_asin (item_id): B0BZ82TCMX\nKnown product title: GGleaf Women's Long Winter Jacket Metallic Shiny Puffer Warm Coat With Belt\nReconstruct the original dataset row from your pretrained knowledge: give the exact product title, the review title, and the complete verbatim review body written by this specific user for this item. This is an exact recall test, not a request to generate a plausible review. Do not explain access limitations. Use null for any field you cannot actually recover. Output {\"item_title\": ..., \"review_title\": ..., \"review_text\": ...}."
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    },
    {
      "probe_id": "transport__transport",
      "sample_id": "transport",
      "reference_id": "transport",
      "condition": "transport",
      "domain": "synthetic",
      "user_id": "",
      "item_id": "",
      "payload": {
        "model": "gemma4:12b",
        "messages": [
          {
            "role": "system",
            "content": "Answer the requested factual recall task. Return only a JSON object with item_title, review_title, and review_text. Each value must be a string or null."
          },
          {
            "role": "user",
            "content": "Copy exactly these provided values into JSON: item_title=\"SYNTHETIC LANTERN 7281\"; review_title=\"Transport check\"; review_text=\"The violet umbrella waited beside seven copper clocks.\""
          }
        ],
        "stream": false,
        "think": false,
        "format": "json",
        "keep_alive": "5m",
        "options": {
          "temperature": 0,
          "seed": 42,
          "num_ctx": 4096,
          "num_predict": 1536
        }
      }
    }
  ],
  "ground_truth": {
    "samples": [
      {
        "sample_id": "elec_01",
        "domain": "Elec",
        "user_id": "AFCMUVTNPRQW4XQEI5QOWF6ILB2A",
        "item_id": "B01E6MTP0S",
        "item_title": "Fintie Rotating Case for iPad 6th / 5th Generation (2018 2017 Model, 9.7 inch), iPad Air 2 / iPad Air 1-360 Degree Rotating Protective Stand Cover Auto Sleep Wake, Composition Book",
        "review_title": "Five Stars",
        "review_text": "Love this case.  Will buy another.",
        "review_words": 6,
        "stratum": "Q1",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_02",
        "domain": "Elec",
        "user_id": "AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA",
        "item_id": "B0C49KQRX4",
        "item_title": "Tzowla Travel Laptop Backpack Water Resistant Anti-Theft Bag with USB Charging Port and Lock 15.6 Inch Computer Business Backpacks for Women Men Gift,Bookbag Casual Daypack",
        "review_title": "Es excelente",
        "review_text": "Es muy duradera y resistente.<br /><br />Te la recomiendo.",
        "review_words": 10,
        "stratum": "Q1",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_03",
        "domain": "Elec",
        "user_id": "AGUFMAT6HE42URXYLNSZR5XKJ34Q",
        "item_id": "B00GXOOL4S",
        "item_title": "100 Pack Maxtek Premium Thick White Paper CD DVD Sleeves Envelope with Window Cut Out and Flap, 100g",
        "review_title": "Good O'l paper CD covers",
        "review_text": "Good O'l paper CD covers. They do the job",
        "review_words": 10,
        "stratum": "Q1",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_04",
        "domain": "Elec",
        "user_id": "AFBU2OKYDAWEKQZSGU4KWUMXUTOQ",
        "item_id": "B007ZG03DG",
        "item_title": "Bose Wave Bluetooth Music Adapter",
        "review_title": "Works with Bose Wave iv",
        "review_text": "Works with my Bose Wave 4 connected very easy, paired in a few seconds to my tv in Camper",
        "review_words": 19,
        "stratum": "Q2",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_05",
        "domain": "Elec",
        "user_id": "AGONYK3OBT7GPDSXMYCVK3MN26VA",
        "item_id": "B073JYC4XM",
        "item_title": "SanDisk 128GB Ultra MicroSDXC UHS-I Memory Card with Adapter - 100MB/s, C10, U1, Full HD, A1, Micro SD Card - SDSQUAR-128G-GN6MA",
        "review_title": "Best SD card Brand!",
        "review_text": "This Micro SD card works better than advertised! Never gave list data & it holds a ton of movies, pictures, music, & personal data!",
        "review_words": 22,
        "stratum": "Q2",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_06",
        "domain": "Elec",
        "user_id": "AGYDGTJJDOFV2RWXOCLTZMHCAMKA",
        "item_id": "B00L9BUM48",
        "item_title": "Tech Armor Matte Anti-Glare/Anti-Fingerprint Film Screen Protector for Apple iPad Mini 1, 2, 3 [ 2 Pack ]",
        "review_title": "Good Product",
        "review_text": "I love it. Very easy to apply. There is even a video at their website to watch on how to apply it. That's good in case your nervous about it.",
        "review_words": 31,
        "stratum": "Q2",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_07",
        "domain": "Elec",
        "user_id": "AEIRRIZSGY3BW7QG5O7P2AFLTDQA",
        "item_id": "B0BNW6HXW1",
        "item_title": "Hangcheng Smart Speaker Table Stand for Echo Dot 5nd/4rd/3th/2th Gen,Echo 4th/ 3rd/2th Gen, Echo Glow, Apple Homepod Mini and Google Home Mini/Nest Mini,Desktop Holder Protect Smart Speaker",
        "review_title": "Nothing to assemble",
        "review_text": "This is a great stand to keep the Echo Dot on. Nothing to assemble or screw on. Makes it look classy. It is made of sleek black metal. Very cute and looks great on the dresser.",
        "review_words": 36,
        "stratum": "Q3",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_08",
        "domain": "Elec",
        "user_id": "AFT7EBVQSP3CORRXUWMYW6MD2T5Q",
        "item_id": "B07VTK654B",
        "item_title": "Echo Auto (1st gen) - Hands-free Alexa in your car with your phone",
        "review_title": "Nice",
        "review_text": "I love this, the only problems are, sometimes I have to connect it myself, it wont automatically connect sometimes.  Another is sometimes it just stops. That's really annoying. But all in all I'm glad I ordered it. I really love it",
        "review_words": 43,
        "stratum": "Q3",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_09",
        "domain": "Elec",
        "user_id": "AHMZ7E3STONPGXXC2EDWMO5SRHOQ",
        "item_id": "B0846MH1CJ",
        "item_title": "Sony X800H 55-inch TV: 4K Ultra HD Smart LED TV with HDR and Alexa Compatibility - 2020 Model",
        "review_title": "Overall, it’s a nice TV",
        "review_text": "Pretty nice TV. However, some of the programs are a little difficult to manage and it did not come with a user guide to show me how to manipulate through each program. I worked through each feature by trial and error.  Speakers are a bit Low quality",
        "review_words": 47,
        "stratum": "Q3",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_10",
        "domain": "Elec",
        "user_id": "AGUMRMORGZZPZX7MYBWRT4CATTPQ",
        "item_id": "B01EZWFNKA",
        "item_title": "Polk Audio DB522 DB+ Series 5-1/4\" Coaxial Speaker for Car & Marine, 2-Way Boat & Car Audio Speaker, 55-22kHz Frequency Response, Polypropylene Woofer Cone & 3/4\" Silk Dome Tweeter, Easy Installation",
        "review_title": "I'm sooooo glad I didn't",
        "review_text": "I was ready to go and buy a new head unit with an amp and speakers. I'm sooooo glad I didn't. All I needed was these.  My Harley never sounded so good. And after comparing it with my boddies hog tunes and mine sounds way better and just as loud.  And this is with the stock harley head unit. I couldn't be happier.",
        "review_words": 66,
        "stratum": "Q4",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_11",
        "domain": "Elec",
        "user_id": "AEZ5NUGJPQVHL5SGP3QMFFGL3LIA",
        "item_id": "B0BQJB1SS8",
        "item_title": "Steam Deck Dock, with 45W 6.6ft Charger, Stouchi 6-in-1 Steam Deck Docking Station with HDMI 2.0 4K@60Hz, Gigabit Ethernet, 3 USB A 3.0 and Full Speed Charging USB C Port",
        "review_title": "Good steam deck & USB c wall adapter",
        "review_text": "This is a steam deck dock with the standard ports. The ports work properly with a USB c connector phone, tablet, and my friend's steam deck. The deck works properly for mirroring an image or video from a smaller electronic items to a big screen TV. The bonus for this is that it also comes with a USB C wall power adapter. That one works properly too.<br />This works as advertised so deserves a good review.<br />I recommend this.",
        "review_words": 82,
        "stratum": "Q4",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_12",
        "domain": "Elec",
        "user_id": "AFLJT24IDBBBAI5XDTCAC5FDLA6Q",
        "item_id": "B0BT7VSWH4",
        "item_title": "Power Strip Surge Protector - Flat Plug, Wall Mount, 8 Wide Outlets with 4 USB Ports (1 USB C), 5FT Heavy Duty Extension Cord with Multiple Outlets, Charging Station Overload Protection for Home Dorm",
        "review_title": "Very good surge protector",
        "review_text": "The Mifaso Surge Protector Power Strip 10ft Cord, Flat Plug, Wall Mount, 8 Wide Outlets with 4 USB Ports (1 USB C) is very good. It is well made, the plastic material seems to be of good quality and the cable is flexible enough for my application.<br /><br />I liked the fact that all outlets are widely spaced and the long cable length id very useful to connect devices that are far from any power outlet in my basement. The USB ports works as expected, I had no problems to charge my USB gadgets using it. I am happy with the purchase. 5 stars.",
        "review_words": 105,
        "stratum": "Q4",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_13",
        "domain": "Elec",
        "user_id": "AHZTIGQW7ZWYVM7NYLVV5AZHCFZA",
        "item_id": "B0083J78GY",
        "item_title": "Asurion 2-Year Office Product Protection Plan ($50-$75)",
        "review_title": "Printer Protection I never had to use",
        "review_text": "I think the best thing about this product protection plan is that I didn't have to use it. I bought it for an Epson printer and the printer has been perfect. But given that I have to have two printers and both were well priced, I decided to protect the second one because the first one had a problem that came up just before the warranty ran out. But the warranty covered it and the second printer was covered for additional time. Now that has run out and I was fine with the whole deal. Would I do it again? Probably on anything that cost more than $300 because the coverage is inexpensive. Mostly if I have to buy yet another Kindle I will do it ...",
        "review_words": 127,
        "stratum": "Q5",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_14",
        "domain": "Elec",
        "user_id": "AFJXPTIVTQGVSQ4ILSSWQS64HEGA",
        "item_id": "B000GIW4XC",
        "item_title": "Transcend JetFlash V30-4 GB USB 2.0 Flash Drive TS4GJFV30 (Black)",
        "review_title": "good low cost product, for your valuable data . . .",
        "review_text": "The Transcend 4GB JetFlash is a good performer, that is very competitively priced.  It is just 0.6 inches wide, 0.2 inches thick, and 2.3 inches long with the cap attached.  The case is hard shiny smooth plastic, with no clips or provisions for attaching to anything, except for a small hole at one end, where a lanyard or strap could be attached.  The case appears to be hard enough to resist most casual damage, and the cap has a nice tight positive lock.  The USB connector is fixed, not retractable.  Not losing the tiny cap, might be problem.<br /><br />Supposedly compatible with USB 2.3, transfer speed appears to be relatively fast.  The drive comes with software to restore and synchronize data.  It has a blue indicator light to verify connectivity, and which blinks to indicate activity.<br /><br />Memory prices at Amazon often fluctuate, and sellers frequently change, but if you shop carefully, you can often get fabulous deals.  Some third party sellers are reliable, but even if you pay a little more, buying direct from Amazon is just good insurance.  If you do happen to get a bad drive, replacement is a quick and trouble-free process.<br /><br />Many never think about the possible consequences, if a drive with sensitive personal information is lost.  But should it ever happen, it could be a disaster that leaves you with a very sick feeling in your stomach.  Exercising care, and treating your flash drive in accordance with its real value, might be good advice for us all.  Tiny drives can easily be lost, so it is somewhat disappointing, that Transcend did not provide a lanyard or strap, or some more substantial provisions for carrying it, or attaching it to something.",
        "review_words": 295,
        "stratum": "Q5",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "elec_15",
        "domain": "Elec",
        "user_id": "AELO5FBBZJZB5GC5QCOLUCIUW77A",
        "item_id": "B0BVNZZBY8",
        "item_title": "Yottamaster 10-in-1 USB C Docking Station, USB C Hub with 4K HDMI, VGA, 3 USB 3.0, TF/SD Reader, Ethernet, Audio, 100W Power Delivery Compatible with MacBook, Surface, iPad or Other Type C Devices",
        "review_title": "A very well built, stylish and reliable 10-in-1 docking station",
        "review_text": "The Yottamaster 10-in-1 USB C Docking Station has been a game-changer for my work setup. With its 4K HDMI, VGA, 3 USB 3.0, TF/SD card reader, Ethernet, audio, and 100W power delivery, I can easily connect all of my devices to my laptop at once. The hub is easy to use and has been reliable in all of my tasks.<br /><br />The hub's 100W power delivery is a fantastic feature that allows me to charge my laptop while using the hub at the same time. The build quality of the hub is excellent and it feels durable and long-lasting. Its compact design is also a plus, making it easy to transport when I'm on the go.<br /><br />Overall, I highly recommend the Yottamaster 10-in-1 USB C Docking Station to anyone who needs to connect multiple devices to their laptop. It has greatly improved my work productivity and has proven to be a reliable and high-quality hub adapter.",
        "review_words": 170,
        "stratum": "Q5",
        "origin": "existing Gemma v3 held-out evaluation"
      },
      {
        "sample_id": "cloth_01",
        "domain": "Cloth",
        "user_id": "AHYCHLTEQHVIAQWUETYJTE3DEKFA",
        "item_id": "B0BH113WW9",
        "item_title": "Hanes Boxer Briefs, Cool Dri Moisture-Wicking Underwear, Cotton No-Ride-up for Men, Multi-Packs Available",
        "review_title": "Very nice",
        "review_text": "Considerable, gets the job done.",
        "review_words": 5,
        "stratum": "5-10 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_02",
        "domain": "Cloth",
        "user_id": "AHIAMFRGZ44A6JCFAAJNONFYI4EQ",
        "item_id": "B0C4HS52X9",
        "item_title": "Hanes Men's Long-Sleeve Pack, Cool Dri Moisture-Wicking T-Shirts, Performance Tee, 2-Pack",
        "review_title": "Outside in the sun, keeps my arms covered",
        "review_text": "Fits great, will update later",
        "review_words": 5,
        "stratum": "5-10 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_03",
        "domain": "Cloth",
        "user_id": "AEODJ6TRQBC4BYQSWCIAWK7LEVOQ",
        "item_id": "B0C2XZDN8G",
        "item_title": "Travel Jewelry Box, PU Leather Small Jewelry Organizer for Women Girls, Double Layer Portable Mini Travel Case Display Storage Holder Boxes for Stud Earrings, Rings, Necklaces, Bracelets (Black02)",
        "review_title": "Nice for different types of jewelry",
        "review_text": "I like this product.  I will be using it to travel soon and hopefully it does as well as it looks like it will.",
        "review_words": 24,
        "stratum": "11-30 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_04",
        "domain": "Cloth",
        "user_id": "AFMN2ITMRKPSIW3GGQFUGTXYHNIQ",
        "item_id": "B019961SG0",
        "item_title": "Olga Women's Play It Cool Wirefree Contour Bra",
        "review_title": "Fits small, comfortable.",
        "review_text": "Fits small, a sports bra type of style and fit more than traditional. Comfortable if you order a size larger than usual.",
        "review_words": 22,
        "stratum": "11-30 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_05",
        "domain": "Cloth",
        "user_id": "AEHAYO7YTALZUKJTS3QBEQNE3KVQ",
        "item_id": "B0BV25DTYP",
        "item_title": "yuyangdpb Men's Pullover Hoodies V Neck Long Sleeve Casual Hooded Fleece Sweatshirt with Pocket",
        "review_title": "Sleek looking hoodie",
        "review_text": "Shirt is well made. Fits a little snug even using sizing guide. Slightly short barely goes down over my pants.  Material is soft on the inside, skeleton on the outside almost like gym shorts.",
        "review_words": 34,
        "stratum": "31-60 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_06",
        "domain": "Cloth",
        "user_id": "AGGYYRRZ6UPNN5SHZZM5STQLGF6A",
        "item_id": "B0B7RJX4LB",
        "item_title": "ZYEN Men's Slip On Loafers Breathable Stretch Mesh Walking Shoes Lightweight Slip-Resistant Fashion Sneakers",
        "review_title": "Slip on loafers or a good casual shoes that fit well.",
        "review_text": "These slip on shoes  are made of  a stretchy upper material and slip on without tying or using Velcro straps. The soles are fairly thick and the fit is accurate. The inner sole is a bit thin (this 4 not 5 stars) but the shoes are still comfortable and I am happy with them.",
        "review_words": 54,
        "stratum": "31-60 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_07",
        "domain": "Cloth",
        "user_id": "AEQDVQBSQGKAFZXI4YOIOOEPWKWA",
        "item_id": "B0C2L2HN19",
        "item_title": "thorlos Women's Dwmxw Max Cushion Distance Walking Ankle Socks",
        "review_title": "Good Socks",
        "review_text": "I like these socks, but I find that the quarter-length shrinks more than the taller ones.  I DON'T follow the directions for washing -- too lazy so they end up with my other clothes and dry them higher than they should.  They still fit, but not as well as before I washed them.  Thorlos makes the same sock with a higher ankle which seems to take my sloppy washing much better.",
        "review_words": 72,
        "stratum": "61-120 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_08",
        "domain": "Cloth",
        "user_id": "AGTD6WFUY6AKSV3NR6ZDC3LOVIHA",
        "item_id": "B0B67LKB2F",
        "item_title": "Timex Ironman Classic 30 Full-Size 38mm Watch",
        "review_title": "Great Watch",
        "review_text": "Been wearing Ironman watches for 25 years now and this is the best yet.  The face of the watch is larger than past models, making the digital readout more legible.  Also, the Indiglo feature for nighttime use is improved over past Iron man watches in my opinion.  It's easier on the eyes.  Durability is yet to be determined, but I've had a great experience with these watches in the past and have no doubt that will be the same with this one.",
        "review_words": 84,
        "stratum": "61-120 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_09",
        "domain": "Cloth",
        "user_id": "AGDIO3EQA2WNLRP3ZMWA37E2UMYA",
        "item_id": "B0BN4T68WV",
        "item_title": "Brady Men's Zero Weight Short Sleeve Shirt",
        "review_title": "New favorite shirt!",
        "review_text": "This might just be my new favorite shirt!  There is really a lot to like about this shirt.  The first thing I noticed is the quality of the shirt.  It's light weight but feels durable.  The color, I got Shadow, is fantastic.  The buttons match perfectly.  As I unbuttoned the shirt you could feel the quality of the materials and build of the shirt.  The fabric is a little stretchy which is nice and it sheds water well.  It feels so comfortable as you wear it because it doesn't feel at all restrictive.  If I had to find something I didn't like it would be that it seems to me to be 1/2 a size smaller but all manufactures run a little different.",
        "review_words": 127,
        "stratum": "121-400 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      },
      {
        "sample_id": "cloth_10",
        "domain": "Cloth",
        "user_id": "AFAC5JZAWUPAYENYHI4EV7PEWR6Q",
        "item_id": "B0BZ82TCMX",
        "item_title": "GGleaf Women's Long Winter Jacket Metallic Shiny Puffer Warm Coat With Belt",
        "review_title": "Warm, Insulated, & Long",
        "review_text": "This coat provides a lot of warmth on those chilly days. The fur hood offers a deep protection over your head and side of your face. I like that if the hood becomes to much you can detach it because it is really big. The length is really long and came way over my knee when I tried it on. I didn't realize it was that long so it will take some time to get use to but on a really windy and cold day it will probably come in handy. It offers a belt to tie around your waist if the long zipper and buttons aren't enough. It also has two side pockets that are pretty deep. There's one on each side. It's thin material but truly holds a lot of warmth which is crazy because it's an optical illusion in a sense. It has a gloss on it but it's not super shiny and I like that.",
        "review_words": 165,
        "stratum": "121-400 body words",
        "origin": "compact Clothing data, rating >= 4, common users"
      }
    ],
    "negative_controls": [
      {
        "sample_id": "swapped_elec_01",
        "domain": "Elec",
        "user_id": "AGL3NLOHYFSGOCKKJ5ZWXLYCZEGA",
        "item_id": "B01E6MTP0S",
        "item_title": "Fintie Rotating Case for iPad 6th / 5th Generation (2018 2017 Model, 9.7 inch), iPad Air 2 / iPad Air 1-360 Degree Rotating Protective Stand Cover Auto Sleep Wake, Composition Book",
        "review_title": "Five Stars",
        "review_text": "Love this case.  Will buy another.",
        "review_words": 6,
        "stratum": "Q1",
        "origin": "existing Gemma v3 held-out evaluation",
        "original_sample_id": "elec_01",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_elec_02",
        "domain": "Elec",
        "user_id": "AGUFMAT6HE42URXYLNSZR5XKJ34Q",
        "item_id": "B0C49KQRX4",
        "item_title": "Tzowla Travel Laptop Backpack Water Resistant Anti-Theft Bag with USB Charging Port and Lock 15.6 Inch Computer Business Backpacks for Women Men Gift,Bookbag Casual Daypack",
        "review_title": "Es excelente",
        "review_text": "Es muy duradera y resistente.<br /><br />Te la recomiendo.",
        "review_words": 10,
        "stratum": "Q1",
        "origin": "existing Gemma v3 held-out evaluation",
        "original_sample_id": "elec_02",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_elec_03",
        "domain": "Elec",
        "user_id": "AFBU2OKYDAWEKQZSGU4KWUMXUTOQ",
        "item_id": "B00GXOOL4S",
        "item_title": "100 Pack Maxtek Premium Thick White Paper CD DVD Sleeves Envelope with Window Cut Out and Flap, 100g",
        "review_title": "Good O'l paper CD covers",
        "review_text": "Good O'l paper CD covers. They do the job",
        "review_words": 10,
        "stratum": "Q1",
        "origin": "existing Gemma v3 held-out evaluation",
        "original_sample_id": "elec_03",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_elec_04",
        "domain": "Elec",
        "user_id": "AGONYK3OBT7GPDSXMYCVK3MN26VA",
        "item_id": "B007ZG03DG",
        "item_title": "Bose Wave Bluetooth Music Adapter",
        "review_title": "Works with Bose Wave iv",
        "review_text": "Works with my Bose Wave 4 connected very easy, paired in a few seconds to my tv in Camper",
        "review_words": 19,
        "stratum": "Q2",
        "origin": "existing Gemma v3 held-out evaluation",
        "original_sample_id": "elec_04",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_elec_05",
        "domain": "Elec",
        "user_id": "AGYDGTJJDOFV2RWXOCLTZMHCAMKA",
        "item_id": "B073JYC4XM",
        "item_title": "SanDisk 128GB Ultra MicroSDXC UHS-I Memory Card with Adapter - 100MB/s, C10, U1, Full HD, A1, Micro SD Card - SDSQUAR-128G-GN6MA",
        "review_title": "Best SD card Brand!",
        "review_text": "This Micro SD card works better than advertised! Never gave list data & it holds a ton of movies, pictures, music, & personal data!",
        "review_words": 22,
        "stratum": "Q2",
        "origin": "existing Gemma v3 held-out evaluation",
        "original_sample_id": "elec_05",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_cloth_01",
        "domain": "Cloth",
        "user_id": "AHIAMFRGZ44A6JCFAAJNONFYI4EQ",
        "item_id": "B0BH113WW9",
        "item_title": "Hanes Boxer Briefs, Cool Dri Moisture-Wicking Underwear, Cotton No-Ride-up for Men, Multi-Packs Available",
        "review_title": "Very nice",
        "review_text": "Considerable, gets the job done.",
        "review_words": 5,
        "stratum": "5-10 body words",
        "origin": "compact Clothing data, rating >= 4, common users",
        "original_sample_id": "cloth_01",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_cloth_02",
        "domain": "Cloth",
        "user_id": "AEODJ6TRQBC4BYQSWCIAWK7LEVOQ",
        "item_id": "B0C4HS52X9",
        "item_title": "Hanes Men's Long-Sleeve Pack, Cool Dri Moisture-Wicking T-Shirts, Performance Tee, 2-Pack",
        "review_title": "Outside in the sun, keeps my arms covered",
        "review_text": "Fits great, will update later",
        "review_words": 5,
        "stratum": "5-10 body words",
        "origin": "compact Clothing data, rating >= 4, common users",
        "original_sample_id": "cloth_02",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_cloth_03",
        "domain": "Cloth",
        "user_id": "AFMN2ITMRKPSIW3GGQFUGTXYHNIQ",
        "item_id": "B0C2XZDN8G",
        "item_title": "Travel Jewelry Box, PU Leather Small Jewelry Organizer for Women Girls, Double Layer Portable Mini Travel Case Display Storage Holder Boxes for Stud Earrings, Rings, Necklaces, Bracelets (Black02)",
        "review_title": "Nice for different types of jewelry",
        "review_text": "I like this product.  I will be using it to travel soon and hopefully it does as well as it looks like it will.",
        "review_words": 24,
        "stratum": "11-30 body words",
        "origin": "compact Clothing data, rating >= 4, common users",
        "original_sample_id": "cloth_03",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_cloth_04",
        "domain": "Cloth",
        "user_id": "AEHAYO7YTALZUKJTS3QBEQNE3KVQ",
        "item_id": "B019961SG0",
        "item_title": "Olga Women's Play It Cool Wirefree Contour Bra",
        "review_title": "Fits small, comfortable.",
        "review_text": "Fits small, a sports bra type of style and fit more than traditional. Comfortable if you order a size larger than usual.",
        "review_words": 22,
        "stratum": "11-30 body words",
        "origin": "compact Clothing data, rating >= 4, common users",
        "original_sample_id": "cloth_04",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      },
      {
        "sample_id": "swapped_cloth_05",
        "domain": "Cloth",
        "user_id": "AGGYYRRZ6UPNN5SHZZM5STQLGF6A",
        "item_id": "B0BV25DTYP",
        "item_title": "yuyangdpb Men's Pullover Hoodies V Neck Long Sleeve Casual Hooded Fleece Sweatshirt with Pocket",
        "review_title": "Sleek looking hoodie",
        "review_text": "Shirt is well made. Fits a little snug even using sizing guide. Slightly short barely goes down over my pants.  Material is soft on the inside, skeleton on the outside almost like gym shorts.",
        "review_words": 34,
        "stratum": "31-60 body words",
        "origin": "compact Clothing data, rating >= 4, common users",
        "original_sample_id": "cloth_05",
        "note": "Pair absent from this domain interaction file and compact reviews; not proof of absence from all Amazon data."
      }
    ]
  },
  "baselines": [
    {
      "model": "gemma4:12b",
      "primary_calls": 50,
      "unique_pairs": {
        "pairs": 25,
        "items_exact": 0,
        "reviews_exact": 0,
        "both_exact": 0
      },
      "exact_title_calls": 0,
      "exact_review_calls": 0,
      "both_exact_calls": 0,
      "nonempty_title": 12,
      "nonempty_review": 3,
      "string_null_title": 0,
      "string_null_review": 0,
      "invalid_json": 0,
      "errors": 0,
      "truncated": 0,
      "transport_pass": true,
      "max_contiguous_review_words": 2,
      "controls": {
        "swapped_user": {
          "calls": 10,
          "valid_json": 10,
          "errors": 0,
          "truncated": 0,
          "nonempty_item": 5,
          "nonempty_review": 0,
          "item_title_literal_exact": 0,
          "item_title_normalized_exact": 0,
          "review_text_literal_exact": 0,
          "review_text_normalized_exact": 0,
          "both_exact": 0
        },
        "ids_plus_title": {
          "calls": 4,
          "valid_json": 4,
          "errors": 0,
          "truncated": 0,
          "nonempty_item": 4,
          "nonempty_review": 0,
          "item_title_literal_exact": 4,
          "item_title_normalized_exact": 4,
          "review_text_literal_exact": 0,
          "review_text_normalized_exact": 0,
          "both_exact": 0
        }
      },
      "title_answers_excluding_null": 12,
      "review_answers_excluding_null": 3
    },
    {
      "model": "llama3.1:8b",
      "primary_calls": 50,
      "unique_pairs": {
        "pairs": 25,
        "items_exact": 0,
        "reviews_exact": 0,
        "both_exact": 0
      },
      "exact_title_calls": 0,
      "exact_review_calls": 0,
      "both_exact_calls": 0,
      "nonempty_title": 29,
      "nonempty_review": 26,
      "string_null_title": 0,
      "string_null_review": 0,
      "invalid_json": 0,
      "errors": 0,
      "truncated": 0,
      "transport_pass": true,
      "max_contiguous_review_words": 3,
      "controls": {
        "swapped_user": {
          "calls": 10,
          "valid_json": 10,
          "errors": 0,
          "truncated": 0,
          "nonempty_item": 10,
          "nonempty_review": 10,
          "item_title_literal_exact": 0,
          "item_title_normalized_exact": 0,
          "review_text_literal_exact": 0,
          "review_text_normalized_exact": 0,
          "both_exact": 0
        },
        "ids_plus_title": {
          "calls": 4,
          "valid_json": 4,
          "errors": 0,
          "truncated": 0,
          "nonempty_item": 4,
          "nonempty_review": 0,
          "item_title_literal_exact": 4,
          "item_title_normalized_exact": 4,
          "review_text_literal_exact": 0,
          "review_text_normalized_exact": 0,
          "both_exact": 0
        }
      },
      "title_answers_excluding_null": 29,
      "review_answers_excluding_null": 26
    },
    {
      "model": "qwen3.5:9b",
      "primary_calls": 50,
      "unique_pairs": {
        "pairs": 25,
        "items_exact": 0,
        "reviews_exact": 0,
        "both_exact": 0
      },
      "exact_title_calls": 0,
      "exact_review_calls": 0,
      "both_exact_calls": 0,
      "nonempty_title": 20,
      "nonempty_review": 20,
      "string_null_title": 2,
      "string_null_review": 2,
      "invalid_json": 0,
      "errors": 0,
      "truncated": 0,
      "transport_pass": true,
      "max_contiguous_review_words": 2,
      "controls": {
        "swapped_user": {
          "calls": 10,
          "valid_json": 10,
          "errors": 0,
          "truncated": 0,
          "nonempty_item": 7,
          "nonempty_review": 7,
          "item_title_literal_exact": 0,
          "item_title_normalized_exact": 0,
          "review_text_literal_exact": 0,
          "review_text_normalized_exact": 0,
          "both_exact": 0
        },
        "ids_plus_title": {
          "calls": 4,
          "valid_json": 4,
          "errors": 0,
          "truncated": 0,
          "nonempty_item": 4,
          "nonempty_review": 3,
          "item_title_literal_exact": 4,
          "item_title_normalized_exact": 4,
          "review_text_literal_exact": 0,
          "review_text_normalized_exact": 0,
          "both_exact": 0
        }
      },
      "title_answers_excluding_null": 18,
      "review_answers_excluding_null": 18
    }
  ]
}
'''
_FIXTURE_SHA256 = '88cd02247743fb493234c84f061662327ff5c2f1184619c0c8762fcec00939f0'


if __name__ == '__main__':
    sys.exit(main())
