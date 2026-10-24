"""The add-on as Subtitld runs it: a process spoken to over stdio, speaking for real.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

By default the process is the source tree (`python -m kittentts_addon`) with
the programs from tools/build_native.py. With KITTENTTS_TEST_BINARY set it is
that executable instead, which is how the release workflow runs these tests:
on the frozen add-on unpacked from the release zip the way Subtitld installs
it, with nothing but what is inside its own bundle.

The files download into $SUBTITLD_MODELS_DIR (a temporary folder when it is
not set): the model, 1.66 GB, and the reference encoder, 542 MB.

The clone is made from a line the add-on itself spoke in English, then asked
to speak Portuguese: the reference clip has to be someone, and KittenTTS's own
output is the add-on's to use.
"""

from __future__ import annotations

import array
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kittentts_addon import engine, voices  # noqa: E402

BINARY = os.environ.get('KITTENTTS_TEST_BINARY')
HAVE_PROGRAMS = bool(BINARY) or engine.binary_path().is_file()

pytestmark = pytest.mark.skipif(not HAVE_PROGRAMS, reason='no kitten-tts; run tools/build_native.py')

_MODELS = Path(os.environ.get('SUBTITLD_MODELS_DIR') or tempfile.mkdtemp(prefix='kittentts-models-'))


def _start(cwd: Path, **extra_env: str) -> subprocess.Popen:
    env = dict(os.environ)
    env['SUBTITLD_MODELS_DIR'] = str(_MODELS)
    env.update(extra_env)
    if BINARY:
        # The bundle's own programs and code, nothing from this checkout.
        env.pop('KITTENTTS_ASSETS', None)
        env.pop('PYTHONPATH', None)
        command = [BINARY]
    else:
        env['PYTHONPATH'] = str(REPO_ROOT)
        command = [sys.executable, '-m', 'kittentts_addon']
    return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _say(proc, req_id: str, out: Path, **params) -> tuple[dict, list[dict]]:
    params = {'output_path': str(out), **params}
    proc.stdin.write((json.dumps({'id': req_id, 'type': 'tts.synthesize', 'params': params}) + '\n').encode())
    proc.stdin.flush()
    progress = []
    while True:
        line = proc.stdout.readline()
        assert line, proc.stderr.read().decode('utf-8', 'replace')
        frame = json.loads(line)
        if frame.get('id') != req_id:
            continue
        if frame['type'] == 'progress':
            progress.append(frame)
            continue
        return frame, progress


def _check_wav(path: Path, at_least_sec: float) -> float:
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        seconds = wav.getnframes() / wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    assert seconds >= at_least_sec
    assert any(frames[i] for i in range(0, len(frames), 97)), 'the audio is silent'
    return seconds


def _to_16k(source: Path, target: Path) -> None:
    """24 kHz to 16 kHz by linear interpolation: crude, and enough for a reference."""
    with wave.open(str(source), 'rb') as wav:
        samples = array.array('h', wav.readframes(wav.getnframes()))
    out = array.array('h')
    for i in range(len(samples) * 2 // 3):
        position = i * 1.5
        low = int(position)
        high = min(low + 1, len(samples) - 1)
        out.append(int(samples[low] + (samples[high] - samples[low]) * (position - low)))
    with wave.open(str(target), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(out.tobytes())


def test_speaks_built_in_voices_and_clones_across_languages(tmp_path):
    proc = _start(tmp_path)
    try:
        hello = json.loads(proc.stdout.readline())
        assert hello['type'] == 'hello' and hello['capabilities'][0]['voice_clone'] is True

        english = ('Look, we have to follow our passions. You have your robotics, '
                   'and I just want to be awesome in space, by the 3rd of May.')
        out = tmp_path / 'out' / 'en.wav'
        frame, progress = _say(proc, 'r1', out, text=english, voice='Bruno', language='en-US', seed=3)
        assert frame['type'] == 'result', frame
        assert frame['data']['sample_rate'] == 24000 and frame['data']['voice'] == 'Bruno'
        assert _check_wav(out, 4.0) == pytest.approx(frame['data']['duration_sec'], abs=0.01)
        assert progress[-1]['value'] == 1.0

        frame, _ = _say(proc, 'r2', tmp_path / 'out' / 'pt.wav', text='Bom dia. Tudo bem com você?',
                        voice=voices.DEFAULT_VOICE['pt'], language='pt-BR')
        assert frame['type'] == 'result', frame
        _check_wav(tmp_path / 'out' / 'pt.wav', 1.0)

        # Text beyond Latin-1, into a folder whose name is not ASCII: on Windows
        # both reach kitten-tts only through its UTF-8 code page.
        chinese = tmp_path / 'saída 声音' / 'zh.wav'
        frame, _ = _say(proc, 'r2b', chinese, text='你好，今天天气很好。', voice=voices.DEFAULT_VOICE['zh'], language='zh-Hans')
        assert frame['type'] == 'result', frame
        _check_wav(chinese, 1.0)

        reference = tmp_path / 'reference.wav'
        _to_16k(out, reference)
        portuguese = 'Estou apavorado! Tenho pesadelos que estou sendo perseguido.'
        frame, progress = _say(proc, 'r3', tmp_path / 'out' / 'clone.wav', text=portuguese,
                               voice=voices.CLONE, language='pt', voice_ref_audio=str(reference),
                               voice_ref_text=english, seed=7)
        assert frame['type'] == 'result', frame
        assert any("speaker's voice" in p['message'] for p in progress)
        _check_wav(tmp_path / 'out' / 'clone.wav', 1.5)

        # Without a transcript, the timbre alone; the prepared voice is reused.
        frame, progress = _say(proc, 'r4', tmp_path / 'out' / 'timbre.wav', text=portuguese,
                               voice=voices.CLONE, language='pt', voice_ref_audio=str(reference), seed=7)
        assert frame['type'] == 'result', frame
        assert not any('Downloading' in p['message'] for p in progress)
        _check_wav(tmp_path / 'out' / 'timbre.wav', 1.5)
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
        proc.stdout.close()
        # The add-on's log, so a failure in CI says what kitten-tts did.
        sys.stderr.write(proc.stderr.read().decode('utf-8', 'replace')[-20000:])
        proc.stderr.close()
