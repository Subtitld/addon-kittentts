#!/usr/bin/env python3
"""Regenerate kittentts_addon/voice_list.json and the voices in manifest.json.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The built-in voices are the entries of the pinned revision's voices/voices.json
(the Python package's list: name, voice_id, transcript). Each voice_id names the
speaker and, for the non-English ones, the language:

    03_radio_dj_smooth_male       -> Marcus, English: "Marcus (smooth radio DJ, man)"
    185_news_anchor_pt            -> PortugueseNewsAnchor, Portuguese: "News anchor"
    pt_portuguese                 -> Portuguese, Portuguese: "Portuguese (main voice)"

The C++ runtime's voices (cpp/default/voices.json) must be exactly the same
names; the script fails otherwise. Never hand-edit the outputs.

    python tools/regen_voices.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS = REPO_ROOT / 'kittentts_addon' / 'pins.json'
OUT = REPO_ROOT / 'kittentts_addon' / 'voice_list.json'
MANIFEST = REPO_ROOT / 'manifest.json'

LANGUAGE_CODES = ('es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi', 'tr', 'nl', 'pl', 'id')
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
    'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi',
    'tr': 'Turkish', 'nl': 'Dutch', 'pl': 'Polish', 'id': 'Indonesian',
}
# The voice each language starts with when the host names none: upstream's
# default for English, the voice named after the language where there is one.
# Not for Chinese: in a test the voice named Chinese fell into silence on 3
# lines of 6 (the add-on then retries), the other three Chinese voices on none.
DEFAULTS = {'en': 'Bruno', 'zh': 'ChineseNewsCctv', 'ja': 'JapaneseTrainConductor', 'ko': 'KoreanNewsAnchor',
            'tr': 'TurkishNewsAnchor', 'nl': 'DutchWeatherPresenter', 'pl': 'PolishNewsAnchor',
            'id': 'IndonesianNewsAnchor'}
WORDS = {'dj': 'DJ', 'cctv': 'CCTV', 'kdrama': 'K-drama', 'irish': 'Irish'}
# Upstream's own wording for the named English voices (docs/voices-and-expression.md).
CHARACTERS = {
    'Matthew': 'exhausted mechanic', 'Elliot': 'grieving, young', 'Willow': 'hushed, young',
    'Dolores': 'southern, elderly', 'Victor': 'controlled fury, older', 'Dante': 'playful, smooth, young',
    'Alfred': 'tender, older', 'Saoirse': 'joyful, Irish', 'Claire': 'playful professional',
    'Raven': 'controlled fury, young', 'Marcus': 'smooth radio DJ', 'Herbert': 'wise elderly professor',
    'Diana': 'stern military officer', 'Laurence': 'dramatic stage actor', 'Maeve': 'cozy, warm storyteller',
    'Walter': 'warm grandfather', 'Edith': 'soft elderly grandmother', 'Miles': 'gentle librarian',
    'Grace': 'soothing nurse', 'Reginald': 'distinguished elderly diplomat', 'Iris': 'hushed museum guide',
    'Frank': 'gravelly, grizzled war veteran', 'Serena': 'gentle yoga instructor', 'Julian': 'soft, dreamy poet',
    'Eleanor': 'somber historian', 'Otis': 'rumbling blues musician', 'Vincent': 'hushed museum guide',
    'Martha': 'weathered radio journalist', 'Sable': 'hushed conspirator', 'Victoria': 'regal narrator',
}


def fetch_json(pins: dict, path: str) -> dict:
    url = f'https://huggingface.co/{pins["repo"]}/resolve/{pins["revision"]}/{path}'
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def describe(name: str, voice_id: str) -> tuple[str, str]:
    """(language code, display name) for one voice."""
    parts = voice_id.split('_')
    if parts[0] in LANGUAGE_CODES and len(parts) == 2:          # pt_portuguese
        return parts[0], f'{name} (main voice)'
    language = 'en'
    if parts[-1] in LANGUAGE_CODES:                             # 185_news_anchor_pt
        language = parts.pop()
    if parts and parts[0].isdigit():
        parts.pop(0)
    gender = ''
    if parts and parts[-1] in ('male', 'female'):
        gender = 'man' if parts.pop() == 'male' else 'woman'
    if parts[:2] == ['kitten', 'v08']:
        words = 'KittenTTS 0.8 voice'
    else:
        words = ' '.join(WORDS.get(w, w) for w in parts)
    words = words[:1].upper() + words[1:]
    camel = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    stripped = camel.removeprefix(LANGUAGE_NAMES[language] + ' ') if language != 'en' else camel
    spoken = ''.join(parts).lower()
    personal = stripped.lower().replace(' ', '') not in (spoken, spoken + 'female', spoken + 'male')
    if name in CHARACTERS:
        words = CHARACTERS[name]
    if not personal:
        # Named after what it is (AirportAnnouncer, PortugueseNewsAnchor).
        text = words + (f', {gender}' if gender else '')
        return language, text
    detail = words[:1].lower() + words[1:] if words != 'KittenTTS 0.8 voice' else words
    return language, f'{name} ({detail}{", " + gender if gender else ""})'


def main() -> int:
    pins = json.loads(PINS.read_text(encoding='utf-8'))
    listed = fetch_json(pins, 'voices/voices.json')
    native = fetch_json(pins, pins['files']['voices']['path'])
    if set(listed) != set(native):
        raise SystemExit(f'FATAL: the Python and C++ voice lists differ: {sorted(set(listed) ^ set(native))}')
    voices = []
    for name, entry in listed.items():
        language, display = describe(name, entry['voice_id'])
        voices.append({'id': name, 'language': language, 'display_name': display,
                       'reference_seconds': round(float(entry.get('duration_s', 0.0)), 1)})
    order = list(LANGUAGE_NAMES)
    voices.sort(key=lambda v: (order.index(v['language']), v['display_name'].lower()))
    by_language: dict[str, list[str]] = {}
    for voice in voices:
        by_language.setdefault(voice['language'], []).append(voice['id'])
    defaults = {}
    for language, names in by_language.items():
        default = DEFAULTS.get(language, LANGUAGE_NAMES[language])
        if default not in names:
            raise SystemExit(f'FATAL: default voice {default} for {language} is not in {names}')
        defaults[language] = default
    OUT.write_text(json.dumps({'revision': pins['revision'], 'defaults': defaults, 'voices': voices},
                              indent=1, ensure_ascii=False) + '\n', encoding='utf-8')

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    clone = [v for v in manifest['voices'] if v.get('requires')]
    manifest['voices'] = clone + [{'id': v['id'], 'language': v['language'], 'display_name': v['display_name'],
                                   'sample_rate': 24000} for v in voices]
    manifest['languages'] = list(by_language)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{len(voices)} voices in {len(by_language)} languages:',
          ', '.join(f'{k} {len(v)}' for k, v in by_language.items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
