"""engine.py without the real programs: files it writes, audio it reads, processes it runs.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import array
import json
import struct
import sys
import threading
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kittentts_addon import download, engine  # noqa: E402


def _float_wav(path: Path, samples: list[float], rate: int = 24000) -> None:
    data = struct.pack(f'<{len(samples)}f', *samples)
    fmt = struct.pack('<HHIIHH', 3, 1, rate, rate * 4, 4, 32)
    path.write_bytes(b'RIFF' + struct.pack('<I', 4 + 8 + len(fmt) + 8 + len(data)) + b'WAVE'
                     + b'fmt ' + struct.pack('<I', len(fmt)) + fmt
                     + b'LIST' + struct.pack('<I', 4) + b'INFO'   # a chunk to skip
                     + b'data' + struct.pack('<I', len(data)) + data)


def _pcm_wav(path: Path, rate: int, channels: int = 1, seconds: float = 1.0) -> None:
    frames = int(rate * seconds)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(array.array('h', [(i * 37) % 2000 - 1000 for i in range(frames * channels)]).tobytes())


def test_reads_float_wav_and_writes_16_bit(tmp_path):
    raw, out = tmp_path / 'raw.wav', tmp_path / 'out.wav'
    _float_wav(raw, [0.0, 0.5, -0.5, 1.5, -2.0])
    samples, rate = engine.read_float_wav(raw)
    assert rate == 24000 and list(samples) == [0.0, 0.5, -0.5, 1.5, -2.0]
    engine.write_pcm16(out, samples, rate)
    with wave.open(str(out), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        values = array.array('h', wav.readframes(5))
    assert list(values) == [0, 16383, -16383, 32767, -32767]  # clipped, not wrapped


def test_silence_around_the_speech_is_trimmed():
    rate = 1000
    speech = [0.5 if i % 2 else -0.5 for i in range(300)]
    samples = array.array('f', [0.0001] * 1200 + speech + [0.0] * 900)
    trimmed = engine.trim_silence(samples, rate, keep_sec=0.08)
    assert 300 + 2 * 80 <= len(trimmed) <= 300 + 2 * 80 + 2 * 20
    assert max(abs(x) for x in trimmed[:60]) < 0.01 and max(abs(x) for x in trimmed[-60:]) < 0.01
    assert list(engine.trim_silence(array.array('f', [0.0] * 50), rate)) == [0.0] * 50


def test_a_subtitld_clip_is_read_without_ffmpeg(tmp_path, monkeypatch):
    clip = tmp_path / 'clip.wav'
    _pcm_wav(clip, 16000)
    monkeypatch.setattr(engine, '_find_ffmpeg', lambda: None)
    pcm = engine.reference_pcm(clip)
    with wave.open(str(clip), 'rb') as wav:
        assert pcm == wav.readframes(wav.getnframes())


def test_other_audio_needs_ffmpeg(tmp_path, monkeypatch):
    clip = tmp_path / 'clip.wav'
    _pcm_wav(clip, 44100, channels=2)
    monkeypatch.setattr(engine, '_find_ffmpeg', lambda: None)
    with pytest.raises(engine.UnreadableReference, match='ffmpeg'):
        engine.reference_pcm(clip)


def _seed_revision(root: Path, voices: dict) -> None:
    rev = root / 'kittentts' / engine.revision()
    (rev / 'cpp' / 'default').mkdir(parents=True)
    (rev / 'upstream-config.json').write_text(json.dumps({'type': 'KITTEN2', 'default_voice': 'Bruno'}))
    (rev / 'cpp' / 'default' / 'voices.json').write_text(json.dumps(voices))


def test_built_in_voices_are_split_once(tmp_path, monkeypatch):
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    _seed_revision(tmp_path, {'Bruno': {'transcript': 'hi', 'speaker': [1]}, 'Luna': {'transcript': 'yo'}})
    path = engine.builtin_voice_file('Luna')
    assert json.loads(path.read_text()) == {'transcript': 'yo'}
    big = tmp_path / 'kittentts' / engine.revision() / 'cpp' / 'default' / 'voices.json'
    big.write_text('not read again')
    assert json.loads(engine.builtin_voice_file('Bruno').read_text())['speaker'] == [1]
    with pytest.raises(engine.EngineError, match='Nobody'):
        (path.parent / '.split-ok').unlink()
        big.write_text(json.dumps({'Bruno': {}}))
        engine.builtin_voice_file('Nobody')


def test_the_run_config_names_the_downloaded_files_and_one_voice(tmp_path, monkeypatch):
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    _seed_revision(tmp_path, {})
    voice = tmp_path / 'voice.json'
    voice.write_text(json.dumps({'reference_tokens': [1, 2], 'speaker': [0.5]}))
    engine._write_run_files(voice, 'Words of the clip.')
    root = tmp_path / 'kittentts' / engine.revision()
    config = json.loads((root / 'config.json').read_text())
    files = engine.pins()['files']
    assert config['type'] == 'KITTEN2'
    assert config['cpp']['gguf'] == {'file': files['model']['path'], 'size': files['model']['size']}
    default = config['cpp']['decoders']['default']
    assert default['torchscript']['file'] == files['decoder']['path']
    assert default['voices']['file'] == 'run/voice.json'
    written = root / 'run' / 'voice.json'
    assert default['voices']['size'] == written.stat().st_size
    assert json.loads(written.read_text()) == {
        engine.VOICE_NAME: {'reference_tokens': [1, 2], 'speaker': [0.5], 'transcript': 'Words of the clip.'}}


def _fake_program(tmp_path: Path, body: str) -> Path:
    script = tmp_path / 'fake.py'
    script.write_text(body)
    return script


def test_a_failing_program_reports_its_last_lines(tmp_path):
    script = _fake_program(tmp_path, 'import sys\nprint("llama_model_loader: noise", file=sys.stderr)\n'
                                     'print("kitten-tts: unknown voice: Nobody", file=sys.stderr)\nsys.exit(1)\n')
    with pytest.raises(engine.EngineError, match='unknown voice: Nobody') as caught:
        engine._run([sys.executable, str(script)], threading.Event(), 'kitten-tts')
    assert 'llama_model_loader' not in str(caught.value)


def test_a_cancel_kills_the_program(tmp_path):
    script = _fake_program(tmp_path, 'import time\ntime.sleep(60)\n')
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with pytest.raises(download.Cancelled):
        engine._run([sys.executable, str(script)], cancel, 'kitten-tts')


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX signal')
def test_an_illegal_instruction_names_the_cpu_requirement(tmp_path):
    script = _fake_program(tmp_path, 'import os, signal\nos.kill(os.getpid(), signal.SIGILL)\n')
    with pytest.raises(engine.UnsupportedCPU, match='AVX2'):
        engine._run([sys.executable, str(script)], threading.Event(), 'kitten-tts')


def test_a_line_is_finished_only_when_every_chunk_stopped(tmp_path):
    report = tmp_path / 'report.json'
    assert not engine.finished(report)
    report.write_text(json.dumps({'chunks': [{'terminated': True}, {'terminated': False}]}))
    assert not engine.finished(report)
    report.write_text(json.dumps({'chunks': [{'terminated': True}]}))
    assert engine.finished(report)
    report.write_text(json.dumps({'chunks': []}))
    assert not engine.finished(report)


def test_the_token_budget_follows_the_text():
    assert engine.max_tokens('Hi.') == 250
    assert engine.max_tokens('x' * 100) == 600
    assert engine.max_tokens('x' * 5000) == 1000


def test_default_threads_is_between_one_and_eight():
    assert 1 <= engine.default_threads() <= 8
