#!/usr/bin/env python3
"""
regression-check.py — Compare two tokenizer-comparison CSV outputs to find
regressions (sentences that parsed with the baseline grammar but not the
modified grammar).

Usage:
  python3 utils/regression-check.py BASELINE.csv MODIFIED.csv

Reads both CSVs (with errors='replace' for encoding safety) and reports:
  - Sentences that regressed (parsed in baseline, not in modified) per tokenizer
  - Sentences that improved (parsed in modified, not in baseline) per tokenizer
  - Net change per tokenizer
"""

import csv
import sys

def load_results(path):
    results = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            sid = row['sentence_id']
            results[sid] = {
                'jieba': row['jieba_parsed'] == 'True',
                'spacy': row['spacy_parsed'] == 'True',
                'stanza': row['stanza_parsed'] == 'True',
                'raw_text': row.get('raw_text', ''),
            }
    return results

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} BASELINE.csv MODIFIED.csv")
        sys.exit(1)

    baseline_path = sys.argv[1]
    modified_path = sys.argv[2]

    print(f"Baseline: {baseline_path}")
    print(f"Modified: {modified_path}")

    baseline = load_results(baseline_path)
    modified = load_results(modified_path)

    common_ids = set(baseline.keys()) & set(modified.keys())
    print(f"Common sentences: {len(common_ids)}")
    print()

    for tok in ['jieba', 'spacy', 'stanza']:
        regressions = []
        improvements = []

        for sid in sorted(common_ids, key=int):
            b = baseline[sid][tok]
            m = modified[sid][tok]
            text = modified[sid]['raw_text']

            if b and not m:
                regressions.append((sid, text))
            elif m and not b:
                improvements.append((sid, text))

        b_total = sum(1 for sid in common_ids if baseline[sid][tok])
        m_total = sum(1 for sid in common_ids if modified[sid][tok])

        print(f"=== {tok} ===")
        print(f"  Baseline parses: {b_total}")
        print(f"  Modified parses: {m_total}")
        print(f"  Net change: {m_total - b_total:+d}")
        print(f"  Regressions: {len(regressions)}")
        for sid, text in regressions[:20]:
            print(f"    [{sid}] {text[:80]}")
        if len(regressions) > 20:
            print(f"    ... and {len(regressions) - 20} more")
        print(f"  Improvements: {len(improvements)}")
        if len(improvements) <= 10:
            for sid, text in improvements:
                print(f"    [{sid}] {text[:80]}")
        else:
            print(f"    (showing first 10 of {len(improvements)})")
            for sid, text in improvements[:10]:
                print(f"    [{sid}] {text[:80]}")
        print()

if __name__ == '__main__':
    main()