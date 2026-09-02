"""Marathi post-processor: dictionary spelling correction + linguistic rules.

Fixes recognition errors that are deterministic at the TEXT level:
  1. Spelling: word not in dictionary + unique edit-distance-1 candidate in
     dictionary -> replace (शुध्दीपत्रक -> शुद्धीपत्रक, सहायक -> सहाय्यक).
  2. Ordinals: digit+वा/री/रा run together -> insert space (८वा -> ८ वा).
  3. Phrase table: commonly-merged Marathi phrases -> rejoin with space.
  4. Stray ZWJ/ZWNJ/ZWSP removal.

Dictionary sources: curated synthetic vocab + high-frequency real-crop labels
(NEVER the test document).
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_DIGITS = '०१२३४५६७८९'
ZW = '\u200b\u200c\u200d\ufeff'

# Ordinal suffixes that must be space-separated from the preceding digit
ORDINALS = ('वा', 'री', 'रा', 'था', 'वी')

# Commonly-merged phrases (correct form -> split form)
PHRASES = {
    'यानावाऐवजी': 'या नावाऐवजी',
    'नवीनप्रशासन': 'नवीन प्रशासन',
    'सदरशुद्धीपत्रक': 'सदर शुद्धीपत्रक',
}

# Conjunct/order confusion patterns: apply when the result IS in the dictionary
CONFUSIONS = [
    ('ध्द', 'द्ध'),   # शुध्दीपत्रक -> शुद्धीपत्रक
    ('शुध्द', 'शुद्ध'),
    ('बुध्द', 'बुद्ध'),
    ('सिध्द', 'सिद्ध'),
]

# Domain abbreviations that must NOT be "corrected" into longer words
DOMAIN_WORDS = {'विधानभा', 'विधानभा', 'सभापती', 'पक्षनेता', 'नावाने'}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _split_punct(word: str):
    """Split trailing/leading punctuation from the word core."""
    lead = ''
    while word and not ('\u0900' <= word[0] <= '\u097f') and not word[0].isalnum():
        lead += word[0]
        word = word[1:]
    trail = ''
    while word and not ('\u0900' <= word[-1] <= '\u097f') and not word[-1].isalnum():
        trail = word[-1] + trail
        word = word[:-1]
    return lead, word, trail


class MarathiPostprocessor:
    def __init__(self, dict_sources=None, min_freq=2):
        words = Counter()
        # Curated synthetic vocab
        from IndicPhotoOCR.recognition.synthetic_data_v3 import (
            GOV_WORDS, CONJUNCT_WORDS, NAMES_DEV)
        for w in GOV_WORDS + CONJUNCT_WORDS + NAMES_DEV:
            w = w.strip('.,:;()')
            if w:
                words[w] += 10  # curated = trusted
        # Real-crop labels (high-frequency = likely correct)
        for src in (dict_sources or []):
            if not os.path.exists(src):
                continue
            for line in open(src, encoding='utf-8'):
                row = json.loads(line)
                core = _split_punct(row['expected_text'])[1]
                if core:
                    words[core] += 1
        self.dictionary = {w for w, c in words.items() if c >= min_freq or c >= 10}
        self._dict_list = None

    def _candidates(self, word):
        if self._dict_list is None:
            self._dict_list = sorted(self.dictionary, key=len)
        cands = [w for w in self._dict_list if abs(len(w) - len(word)) <= 1
                 and levenshtein(word, w) == 1]
        return cands

    def fix_word(self, word: str) -> str:
        # 0. strip zero-width chars
        word = ''.join(c for c in word if c not in ZW)
        if not word:
            return word

        # 1. phrase table
        if word in PHRASES:
            return PHRASES[word]

        # 2. ordinal split: ८वा -> ८ वा
        m = re.match(rf'^([{''.join(DEV_DIGITS)}0-9]+)({'|'.join(ORDINALS)})$', word)
        if m:
            return f'{m.group(1)} {m.group(2)}'

        lead, core, trail = _split_punct(word)
        if not core:
            return word

        # 3a. domain words are always kept as-is
        if core in DOMAIN_WORDS:
            return lead + core + trail

        # 3b. conjunct confusion patterns (ध्द -> द्ध) when result is a word
        if core not in self.dictionary:
            for bad, good in CONFUSIONS:
                if bad in core:
                    cand = core.replace(bad, good)
                    if cand in self.dictionary:
                        core = cand
                        break

        # 3c. dictionary spelling correction (unique edit-distance-1 candidate)
        if core not in self.dictionary:
            cands = self._candidates(core)
            if len(cands) == 1:
                core = cands[0]

        # 4. ordinal inside trailing punct (e.g. '८वा.' -> '८ वा.')
        m = re.match(rf'^([{''.join(DEV_DIGITS)}0-9]+)({'|'.join(ORDINALS)})$', core)
        if m:
            return f'{lead}{m.group(1)} {m.group(2)}{trail}'

        # 5. ':-' spacing: पत्रक:- -> पत्रक :- (document convention)
        if trail == ':-' and core:
            return f'{lead}{core} :-'

        return lead + core + trail

    def process(self, text: str) -> str:
        return ' '.join(self.fix_word(w) for w in text.split(' ') if w != '') \
            if ' ' in text else self.fix_word(text)
