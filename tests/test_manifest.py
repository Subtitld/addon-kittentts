"""manifest.json, pins.json, voice_list.json and the code agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kittentts_addon import ADDON_ID, __version__, voices  # noqa: E402
from kittentts_addon.engine import pins  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_REVISION = re.compile(r'^[0-9a-f]{40}$')


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['voice_clone'] is True
    assert MANIFEST['compute'] == 'cpu'
    assert 'accelerators' not in MANIFEST
    [clone] = [v for v in MANIFEST['voices'] if v.get('requires')]
    assert clone['id'] == voices.CLONE and clone['language'] == '*'


def test_the_licence_is_spelled_out():
    assert 'Stellon Labs' in MANIFEST['license'] and 'Stellon Labs' in MANIFEST['summary']
    assert 'US$1M' in MANIFEST['summary']
    licences = REPO_ROOT / 'THIRD_PARTY_LICENSES'
    assert 'Powered by KittenML' in (licences / 'models' / 'README.md').read_text(encoding='utf-8')
    text = (licences / 'models' / 'STELLON-LABS-COMMUNITY-LICENSE.md').read_text(encoding='utf-8')
    assert text.startswith('STELLON LABS COMMUNITY LICENSE AGREEMENT')


def test_every_language_has_its_default_voice():
    assert set(voices.DEFAULT_VOICE) == set(voices.LANGUAGES)
    for language, voice in voices.DEFAULT_VOICE.items():
        assert voices.BY_ID[voice].language == language
    assert voices.DEFAULT_VOICE['en'] == 'Bruno'
    assert len(voices.BUILT_IN) == len(voices.BY_ID) - 1 == 115


def test_options_match_the_code():
    fields = {f['key']: f for f in MANIFEST['config_schema']['fields']}
    assert set(fields) == {'preset', 'threads', 'voice_ref_audio', 'voice_ref_text', 'skip_ref_text'}
    assert sorted(o['value'] for o in fields['preset']['options']) == ['expressive', 'stable']
    assert fields['voice_ref_audio']['type'] == 'file'
    assert fields['skip_ref_text']['type'] == 'bool'


def test_pins_name_one_revision_and_no_stellon_file_of_ours():
    data = pins()
    assert data['repo'] == 'KittenML/kitten-tts-2'
    assert _REVISION.match(data['revision'])
    assert voices.REVISION == data['revision']
    assert MANIFEST['models'][0]['revision'] == data['revision']
    for key, pin in data['files'].items():
        assert _SHA256.match(pin['sha256']) and pin['size'] > 0, key
        assert 'url' not in pin, key   # KittenML's files come from KittenML's repository
    spk = data['files']['spk_proj']
    assert spk['path'] == 'lm/model-ternary.safetensors' and spk['size'] == 2109440 and spk['offset'] > 0
    assert data['files']['config']['save_as'] != 'config.json'
    encoder = data['ref_encoder']
    assert encoder['url'].startswith('https://github.com/Subtitld/addon-kittentts/releases/download/')
    assert encoder['url'].endswith('/' + encoder['path']) and _SHA256.match(encoder['sha256'])
