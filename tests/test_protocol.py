"""The add-on spoken over its wire format, in a real process.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

These run `python -m kittentts_addon` and talk JSON lines to it, the way the
host does. They cover every answer the add-on owes the host without the
programs, a model or the network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kittentts_addon import voices  # noqa: E402


@pytest.fixture()
def proc(tmp_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    env['KITTENTTS_ASSETS'] = str(tmp_path / 'no-such-assets')
    env['SUBTITLD_MODELS_DIR'] = str(tmp_path / 'models')
    env['HF_ENDPOINT'] = 'http://127.0.0.1:9'  # nothing listens: no test reaches the network
    for name in list(env):
        if name.startswith('KITTENTTS_') and name != 'KITTENTTS_ASSETS':
            env.pop(name)
    process = subprocess.Popen([sys.executable, '-m', 'kittentts_addon'], cwd=tmp_path, env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    yield process
    if process.poll() is None:
        process.kill()
    process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close()


def _send(proc, frame: dict) -> None:
    proc.stdin.write((json.dumps(frame) + '\n').encode('utf-8'))
    proc.stdin.flush()


def _recv(proc) -> dict:
    line = proc.stdout.readline()
    assert line, proc.stderr.read().decode('utf-8', 'replace')
    return json.loads(line)


def _terminal(proc, req_id: str) -> dict:
    while True:
        frame = _recv(proc)
        if frame.get('id') == req_id and frame['type'] in ('result', 'error'):
            return frame


def _synth(proc, req_id: str, tmp_path: Path, **params) -> dict:
    params.setdefault('output_path', str(tmp_path / f'{req_id}.wav'))
    _send(proc, {'id': req_id, 'type': 'tts.synthesize', 'params': params})
    return _terminal(proc, req_id)


def test_hello_lists_languages_voices_and_the_clone(proc):
    hello = _recv(proc)
    assert hello['type'] == 'hello' and hello['addon'] == 'kittentts' and hello['protocol'] == 1
    [capability] = hello['capabilities']
    assert capability['task'] == 'tts.synthesize'
    assert capability['languages'] == list(voices.LANGUAGES)
    assert capability['voices'] == voices.manifest_voices()
    assert capability['voice_clone'] is True


def test_requests_it_cannot_serve_get_clear_errors(proc, tmp_path):
    _recv(proc)
    _send(proc, {'type': 'ready', 'host': 'subtitld', 'host_version': 'test'})
    assert _synth(proc, 'a', tmp_path, text='Hej.', voice='Bruno', language='sv')['code'] == 'unsupported_language'
    empty = _synth(proc, 'b', tmp_path, text='Hello.', voice='Bruno', language='')
    assert empty['code'] == 'unsupported_language' and "project's language" in empty['message']
    assert _synth(proc, 'c', tmp_path, text='Hello.', voice='Cosette', language='en')['code'] == 'unsupported_voice'
    assert _synth(proc, 'd', tmp_path, text='  ', voice='Bruno', language='en')['code'] == 'bad_params'
    _send(proc, {'id': 'e', 'type': 'tts.synthesize', 'params': {'text': 'Hi.', 'language': 'en'}})
    assert _terminal(proc, 'e')['code'] == 'bad_params'
    _send(proc, {'id': 'f', 'type': 'asr.transcribe', 'params': {}})
    assert _terminal(proc, 'f')['code'] == 'bad_params'


def test_a_clone_without_a_clip_says_what_to_do(proc, tmp_path):
    _recv(proc)
    frame = _synth(proc, 'g', tmp_path, text='Olá.', voice=voices.CLONE, language='pt-BR')
    assert frame['code'] == 'bad_params' and 'fallback reference clip' in frame['message']
    frame = _synth(proc, 'h', tmp_path, text='Olá.', voice=voices.CLONE, language='pt',
                   voice_ref_audio=str(tmp_path / 'missing.wav'))
    assert frame['code'] == 'bad_params' and 'missing' in frame['message']


def test_a_download_that_cannot_reach_the_server_is_retryable(proc, tmp_path):
    _recv(proc)
    frame = _synth(proc, 'i', tmp_path, text='Olá.', language='pt-BR')
    assert frame['code'] == 'network_unavailable' and frame['retryable'] is True


def test_shutdown_exits_cleanly(proc):
    _recv(proc)
    _send(proc, {'type': 'shutdown'})
    assert proc.wait(timeout=10) == 0
