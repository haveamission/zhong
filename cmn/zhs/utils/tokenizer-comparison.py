#!/usr/bin/env python3
"""
tokenizer-comparison.py — Compare Jieba vs spaCy vs Stanza tokenization
for Zhong grammar parsing coverage.

Queries all Chinese sentences from article_sentence, tokenizes with all
three tokenizers, feeds each to ACE, records parse success and reading
count. Deduplicates ACE calls when tokenizers agree.

Usage (inside grammarengine Docker container):
  python3 utils/tokenizer-comparison.py

Required environment variables:
  DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

Optional environment variables:
  GRAMMAR_FILE  — path to .dat file (default: /runpod-volume/grammars/zhs.dat)
  TIMEOUT       — ACE timeout in seconds (default: 10)
  OUTPUT_FILE   — CSV output path (default: /tmp/tokenizer-comparison-3way.csv)
"""

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import jieba
jieba.setLogLevel(20)

import spacy
SPACY_MODEL_BASE = Path('/runpod-volume/spacy_models/zh_core_web_sm')
model_paths = list(SPACY_MODEL_BASE.glob('zh_core_web_sm-*'))
nlp_spacy = spacy.load(str(model_paths[0]) if model_paths else str(SPACY_MODEL_BASE))

import stanza
nlp_stanza = stanza.Pipeline('zh', processors='tokenize', download_method=None, model_dir='/runpod-volume/stanza_models', verbose=False)

import pymysql

DB_HOST = os.environ.get('DB_HOST', 'EDIT_ME')
DB_USER = os.environ.get('DB_USER', 'EDIT_ME')
DB_PASS = os.environ.get('DB_PASSWORD', 'EDIT_ME')
DB_NAME = os.environ.get('DB_NAME', 'EDIT_ME')
DB_PORT = int(os.environ.get('DB_PORT', '3306'))

GRAMMAR_FILE = os.environ.get('GRAMMAR_FILE', '/runpod-volume/grammars/zhs.dat')
ACE_BINARY = 'ace'
TIMEOUT_SECONDS = int(os.environ.get('TIMEOUT', '10'))
OUTPUT_FILE = os.environ.get('OUTPUT_FILE', '/tmp/tokenizer-comparison-3way.csv')
SUMMARY_FILE = OUTPUT_FILE.replace('.csv', '-summary.txt')

CHINESE_PUNCT = '。，！？；：、""''（）《》·—…「」『』【】〈〉～'

def strip_non_chinese(text):
    return ''.join(
        char for char in text
        if re.search('[\u4e00-\u9fff]', char)
        or char.isdigit()
        or char.isspace()
        or char in CHINESE_PUNCT
    )

def has_english(text):
    return bool(re.search('[a-zA-Z]', text))

def tokenize_jieba(text):
    return list(jieba.cut(text))

def tokenize_spacy(text):
    doc = nlp_spacy(text)
    return [token.text for token in doc]

def tokenize_stanza(text):
    doc = nlp_stanza(text)
    return [w.text for s in doc.sentences for w in s.words]

def parse_with_ace(segmented_text, timeout=TIMEOUT_SECONDS):
    preprocessed = strip_non_chinese(segmented_text)
    preprocessed = re.sub(r' +', ' ', preprocessed).strip()
    if not preprocessed:
        return {'parsed': False, 'readings': 0, 'time_s': 0, 'ram_kb': 0, 'edges': 0, 'input': preprocessed}

    try:
        result = subprocess.run(
            [ACE_BINARY, '-g', GRAMMAR_FILE, '-1', '--max-chart-megabytes=2000'],
            input=preprocessed,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        stderr = result.stderr
        readings = 0
        parse_time = 0
        ram_kb = 0
        edges = 0

        for line in stderr.split('\n'):
            m = re.search(r'(\d+) readings?, added (\d+)', line)
            if m:
                readings = int(m.group(1))
                edges = int(m.group(2))
            m2 = re.search(r'RAM: (\d+)k', line)
            if m2:
                ram_kb = int(m2.group(1))
            m3 = re.search(r'time (\d+\.\d+)s', line)
            if m3:
                parse_time = float(m3.group(1))

        return {
            'parsed': readings > 0,
            'readings': readings,
            'time_s': parse_time,
            'ram_kb': ram_kb,
            'edges': edges,
            'input': preprocessed
        }
    except subprocess.TimeoutExpired:
        return {'parsed': False, 'readings': -1, 'time_s': timeout, 'ram_kb': 0, 'edges': 0, 'input': preprocessed}
    except Exception as e:
        return {'parsed': False, 'readings': -2, 'time_s': 0, 'ram_kb': 0, 'edges': 0, 'input': preprocessed, 'error': str(e)}

def get_sentences():
    conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME, charset='utf8mb4')
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT s.id, s.sentence, s.per_language_id
                FROM article_sentence s
                WHERE s.language_id = 1
                ORDER BY s.id
            """)
            return cursor.fetchall()
    finally:
        conn.close()

def main():
    print(f"Grammar: {GRAMMAR_FILE}")
    print(f"Timeout: {TIMEOUT_SECONDS}s")
    print(f"Output: {OUTPUT_FILE}")
    print("Fetching sentences...")
    sentences = get_sentences()
    print(f"Found {len(sentences)} Chinese sentences")

    fieldnames = [
        'sentence_id', 'per_language_id', 'raw_text', 'has_english',
        'jieba_seg', 'spacy_seg', 'stanza_seg',
        'jieba_spacy_agree', 'jieba_stanza_agree', 'spacy_stanza_agree', 'all_agree',
        'jieba_parsed', 'jieba_readings', 'jieba_time_s', 'jieba_ram_kb', 'jieba_edges',
        'spacy_parsed', 'spacy_readings', 'spacy_time_s', 'spacy_ram_kb', 'spacy_edges',
        'stanza_parsed', 'stanza_readings', 'stanza_time_s', 'stanza_ram_kb', 'stanza_edges',
        'jieba_ace_input', 'spacy_ace_input', 'stanza_ace_input'
    ]

    stats = {
        'total': 0, 'has_english': 0,
        'all_agree': 0, 'jieba_spacy_agree': 0, 'jieba_stanza_agree': 0, 'spacy_stanza_agree': 0,
        'all_three_parse': 0,
        'only_jieba': 0, 'only_spacy': 0, 'only_stanza': 0,
        'jieba_spacy_only': 0, 'jieba_stanza_only': 0, 'spacy_stanza_only': 0,
        'none_parse': 0,
        'jieba_timeout': 0, 'spacy_timeout': 0, 'stanza_timeout': 0,
        'ace_calls': 0,
    }

    start_time = time.time()

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (sid, sentence, plid) in enumerate(sentences):
            if not sentence or not sentence.strip():
                continue

            stats['total'] += 1
            eng = has_english(sentence)
            if eng:
                stats['has_english'] += 1

            jieba_seg = ' '.join(tokenize_jieba(sentence))
            spacy_seg = ' '.join(tokenize_spacy(sentence))
            stanza_seg = ' '.join(tokenize_stanza(sentence))

            js_agree = (jieba_seg == spacy_seg)
            jt_agree = (jieba_seg == stanza_seg)
            st_agree = (spacy_seg == stanza_seg)
            all_agree = js_agree and jt_agree

            if all_agree: stats['all_agree'] += 1
            if js_agree: stats['jieba_spacy_agree'] += 1
            if jt_agree: stats['jieba_stanza_agree'] += 1
            if st_agree: stats['spacy_stanza_agree'] += 1

            unique_segs = {}
            for name, seg in [('jieba', jieba_seg), ('spacy', spacy_seg), ('stanza', stanza_seg)]:
                if seg not in unique_segs:
                    unique_segs[seg] = parse_with_ace(seg)
                    stats['ace_calls'] += 1

            jieba_result = unique_segs[jieba_seg]
            spacy_result = unique_segs[spacy_seg]
            stanza_result = unique_segs[stanza_seg]

            if jieba_result['readings'] == -1: stats['jieba_timeout'] += 1
            if spacy_result['readings'] == -1: stats['spacy_timeout'] += 1
            if stanza_result['readings'] == -1: stats['stanza_timeout'] += 1

            jp = jieba_result['parsed']
            sp = spacy_result['parsed']
            tp = stanza_result['parsed']

            if jp and sp and tp: stats['all_three_parse'] += 1
            elif jp and sp and not tp: stats['jieba_spacy_only'] += 1
            elif jp and tp and not sp: stats['jieba_stanza_only'] += 1
            elif sp and tp and not jp: stats['spacy_stanza_only'] += 1
            elif jp and not sp and not tp: stats['only_jieba'] += 1
            elif sp and not jp and not tp: stats['only_spacy'] += 1
            elif tp and not jp and not sp: stats['only_stanza'] += 1
            else: stats['none_parse'] += 1

            writer.writerow({
                'sentence_id': sid, 'per_language_id': plid,
                'raw_text': sentence, 'has_english': eng,
                'jieba_seg': jieba_seg, 'spacy_seg': spacy_seg, 'stanza_seg': stanza_seg,
                'jieba_spacy_agree': js_agree, 'jieba_stanza_agree': jt_agree,
                'spacy_stanza_agree': st_agree, 'all_agree': all_agree,
                'jieba_parsed': jp, 'jieba_readings': jieba_result['readings'],
                'jieba_time_s': jieba_result['time_s'], 'jieba_ram_kb': jieba_result['ram_kb'],
                'jieba_edges': jieba_result['edges'],
                'spacy_parsed': sp, 'spacy_readings': spacy_result['readings'],
                'spacy_time_s': spacy_result['time_s'], 'spacy_ram_kb': spacy_result['ram_kb'],
                'spacy_edges': spacy_result['edges'],
                'stanza_parsed': tp, 'stanza_readings': stanza_result['readings'],
                'stanza_time_s': stanza_result['time_s'], 'stanza_ram_kb': stanza_result['ram_kb'],
                'stanza_edges': stanza_result['edges'],
                'jieba_ace_input': jieba_result['input'],
                'spacy_ace_input': spacy_result['input'],
                'stanza_ace_input': stanza_result['input'],
            })

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(sentences) - i - 1) / rate
                print(f"  Processed {i+1}/{len(sentences)} sentences... ({elapsed/60:.0f}m elapsed, ~{remaining/60:.0f}m remaining, {stats['ace_calls']} ACE calls)")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    t = max(stats['total'], 1)

    jieba_total = stats['all_three_parse'] + stats['jieba_spacy_only'] + stats['jieba_stanza_only'] + stats['only_jieba']
    spacy_total = stats['all_three_parse'] + stats['jieba_spacy_only'] + stats['spacy_stanza_only'] + stats['only_spacy']
    stanza_total = stats['all_three_parse'] + stats['jieba_stanza_only'] + stats['spacy_stanza_only'] + stats['only_stanza']

    summary_lines = [
        f"=== Tokenizer Comparison: Jieba vs spaCy vs Stanza ===",
        f"Grammar: {GRAMMAR_FILE}",
        f"Total sentences: {stats['total']}",
        f"Sentences with English: {stats['has_english']}",
        f"Runtime: {elapsed/60:.1f} minutes",
        f"ACE calls: {stats['ace_calls']} (saved {stats['total']*3 - stats['ace_calls']} by dedup)",
        "",
        f"=== Segmentation Agreement ===",
        f"All three agree: {stats['all_agree']} ({100*stats['all_agree']/t:.1f}%)",
        f"Jieba-spaCy agree: {stats['jieba_spacy_agree']} ({100*stats['jieba_spacy_agree']/t:.1f}%)",
        f"Jieba-Stanza agree: {stats['jieba_stanza_agree']} ({100*stats['jieba_stanza_agree']/t:.1f}%)",
        f"spaCy-Stanza agree: {stats['spacy_stanza_agree']} ({100*stats['spacy_stanza_agree']/t:.1f}%)",
        "",
        f"=== Parse Results ===",
        f"All three parse: {stats['all_three_parse']}",
        f"Jieba + spaCy only: {stats['jieba_spacy_only']}",
        f"Jieba + Stanza only: {stats['jieba_stanza_only']}",
        f"spaCy + Stanza only: {stats['spacy_stanza_only']}",
        f"Only Jieba: {stats['only_jieba']}",
        f"Only spaCy: {stats['only_spacy']}",
        f"Only Stanza: {stats['only_stanza']}",
        f"None parse: {stats['none_parse']}",
        "",
        f"=== Total Parses per Tokenizer ===",
        f"Jieba: {jieba_total} ({100*jieba_total/t:.1f}%)",
        f"spaCy: {spacy_total} ({100*spacy_total/t:.1f}%)",
        f"Stanza: {stanza_total} ({100*stanza_total/t:.1f}%)",
        "",
        f"=== Timeouts ===",
        f"Jieba: {stats['jieba_timeout']}",
        f"spaCy: {stats['spacy_timeout']}",
        f"Stanza: {stats['stanza_timeout']}",
        "",
        f"Results: {OUTPUT_FILE}",
    ]

    summary = '\n'.join(summary_lines)
    print("\n" + summary)

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')

if __name__ == '__main__':
    main()