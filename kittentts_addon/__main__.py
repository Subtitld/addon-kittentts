"""KittenTTS 2 add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline text-to-speech with KittenML's KittenTTS 2, in 16 languages at
24 kHz: in 115 built-in voices, or in the speaker's own voice, cloned from the
clip Subtitld cuts from the video. Serves `tts.synthesize`. KittenML's C++
runtime does the synthesis, and this add-on's kitten-ref prepares cloned voices
(engine.py). The model, about 1.6 GB, downloads on the first line, and the
reference encoder, 540 MB, on the first cloned line (download.py); after that
nothing leaves the machine.

The weights are KittenML's, under the Stellon Labs Community License, and are
downloaded from KittenML's repository, never shipped with the add-on.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:kittentts]` and surfaces.

Options (the host sets them from Configure as KITTENTTS_<KEY>)
--------------------------------------------------------------
preset: `stable` or `expressive`; threads: 0 for automatic; voice_ref_audio
and voice_ref_text: a clone reference to fall back on; skip_ref_text: clone the
timbre only.

Limits
------
One line at a time. A cancel kills the line's process, or stops the download
(which resumes on the next request).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from kittentts_addon import ADDON_ID, PROTOCOL_VERSION, __version__
from kittentts_addon import download, engine, tls, voices

ADDON_VERSION = __version__

ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_UNSUPPORTED_VOICE = 'unsupported_voice'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network_unavailable'

# No space left on the device, or over the user's disk quota.
_DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, 'EDQUOT', 122)}

PRESETS = ('stable', 'expressive')

log = logging.getLogger('kittentts_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`, so nothing else can write a
    stray line into the protocol. Falls back to the Python-level swap if the
    fd dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _options() -> dict:
    preset = _env('KITTENTTS_PRESET').lower() or 'stable'
    if preset not in PRESETS:
        log.warning('KITTENTTS_PRESET=%r is not one of %s; using stable', preset, PRESETS)
        preset = 'stable'
    try:
        threads = max(0, min(256, int(float(_env('KITTENTTS_THREADS') or 0))))
    except ValueError:
        threads = 0
    return {
        'preset': preset,
        'threads': threads,
        'voice_ref_audio': _env('KITTENTTS_VOICE_REF_AUDIO'),
        'voice_ref_text': ' '.join(_env('KITTENTTS_VOICE_REF_TEXT').split()),
        'skip_ref_text': _env('KITTENTTS_SKIP_REF_TEXT').lower() in ('1', 'true', 'yes', 'on'),
    }


def _resolve_language(raw) -> str:
    """The primary tag KittenTTS 2 has voices for (`pt-br`, `en_US`, `zh-Hans` -> pt, en, zh)."""
    tag = str(raw or '').strip().lower().replace('_', '-')
    primary = tag.split('-', 1)[0]
    if primary in voices.LANGUAGES:
        return primary
    spoken = ', '.join(voices.LANGUAGE_NAMES[code] for code in voices.LANGUAGES)
    if not primary:
        raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                          f"Set the project's language first: KittenTTS 2 speaks {spoken}.")
    raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                      f'KittenTTS 2 does not speak {raw!r}. It speaks {spoken}; '
                      f'pick another voice engine for other languages.')


def _seed(params: dict) -> int:
    try:
        return int(params.get('seed', 42)) & 0x7FFFFFFF
    except (TypeError, ValueError):
        return 42


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()
_work_lock = threading.Lock()  # one line, or one download, at a time


def _megabytes(count: int) -> str:
    return f'{count / 1e6:,.0f} MB'


def _ensure_files(state: _Request, clone: bool) -> None:
    pins = engine.base_pins() + (engine.clone_pins() if clone else [])
    todo = download.missing(pins, engine.revision())
    if not todo:
        return
    total = sum(pin['size'] for pin in todo)
    if any(pin is engine.pins()['files']['model'] for pin in todo):
        what = 'the KittenTTS 2 model'
    elif any(pin is engine.pins()['ref_encoder'] for pin in todo):
        what = 'the reference encoder, for cloning'
    else:
        what = 'the model files'
    message = f'Downloading {what} ({_megabytes(total)})...'
    _emit_progress(state.req_id, 0.0, message)
    last = [0.0]

    def progress(done: int, total_bytes: int) -> None:
        now = time.monotonic()
        if now - last[0] >= 0.25 or done >= total_bytes:
            last[0] = now
            _emit_progress(state.req_id, 0.85 * done / max(1, total_bytes), message)

    started = time.monotonic()
    download.ensure(pins, engine.pins()['repo'], engine.revision(), progress, state.cancel.is_set)
    log.info('downloaded %s in %.1fs', _megabytes(total), time.monotonic() - started)


def _clone_reference(params: dict, options: dict) -> tuple[str, str]:
    """(clip, transcript) for a cloned line: the host's, else the configured fallback."""
    sent = str(params.get('voice_ref_audio') or '').strip()
    if sent:
        return sent, ' '.join(str(params.get('voice_ref_text') or '').split())
    if options['voice_ref_audio']:
        return options['voice_ref_audio'], options['voice_ref_text']
    raise _AddonError(ERR_BAD_PARAMS,
                      "The speaker's own voice needs a clip of the speaker, and Subtitld sent none. "
                      "Turn on the original voice for this speaker, pick a fallback reference clip "
                      "in Configure, or pick another voice.")


def _handle_synthesize(state: _Request, params: dict) -> None:
    req_id = state.req_id
    text = ' '.join(str(params.get('text') or '').split())
    output_path = params.get('output_path')
    if not text:
        _emit_error(req_id, ERR_BAD_PARAMS, 'text is empty')
        return
    if not isinstance(output_path, str) or not output_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'output_path is required')
        return
    options = _options()
    try:
        language = _resolve_language(params.get('language'))
        voice_id = str(params.get('voice') or '') or voices.DEFAULT_VOICE[language]
        if voice_id not in voices.BY_ID:
            raise _AddonError(ERR_UNSUPPORTED_VOICE, f'unknown voice {voice_id!r}')
        clone = voice_id == voices.CLONE
        reference = transcript = ''
        if clone:
            reference, transcript = _clone_reference(params, options)
            if not os.path.isfile(reference):
                raise _AddonError(ERR_BAD_PARAMS, f'The reference clip is missing: {reference}')
        elif voices.BY_ID[voice_id].language != language:
            log.warning('voice %s speaks %s, the line is %s; its accent will show',
                        voice_id, voices.BY_ID[voice_id].language, language)
    except _AddonError as exc:
        _emit_error(req_id, exc.code, exc.message)
        return
    seed = _seed(params)
    threads = options['threads'] or engine.default_threads()

    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        if state.cancel.is_set():
            raise download.Cancelled()
        with _work_lock:
            _ensure_files(state, clone)
            if state.cancel.is_set():
                raise download.Cancelled()
            if clone:
                _emit_progress(req_id, 0.87, "Preparing the speaker's voice...")
                voice_file = engine.clone_voice_file(Path(reference), threads, state.cancel)
                use_reference = bool(transcript) and not options['skip_ref_text']
            else:
                voice_file = engine.builtin_voice_file(voice_id)
                transcript, use_reference = None, True
            _emit_progress(req_id, 0.9, 'Speaking...')
            log.info('synthesize %s: voice=%s language=%s seed=%d preset=%s threads=%d reference=%s chars=%d',
                     req_id, voice_id, language, seed, options['preset'], threads,
                     'yes' if use_reference else 'timbre only', len(text))
            speech = engine.synthesize(text, voice_file, Path(output_path), transcript=transcript,
                                       use_reference=use_reference, normalize=language == 'en',
                                       seed=seed, preset=options['preset'], threads=threads,
                                       cancel=state.cancel)
        log.info('synthesize %s: %.2fs of audio in %.2fs', req_id, speech.duration_sec, speech.seconds)
        outcome = ('result', {
            'path': output_path,
            'duration_sec': round(speech.duration_sec, 3),
            'sample_rate': speech.sample_rate,
            'channels': 1,
            'voice': voice_id,
        })
    except download.Cancelled:
        log.info('synthesize %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except download.DownloadError as exc:
        log.warning('synthesize %s: download failed: %s', req_id, exc)
        outcome = ('error', ERR_NETWORK, f'Could not download the model: {exc}', True)
    except engine.UnreadableReference as exc:
        log.warning('synthesize %s: %s', req_id, exc)
        outcome = ('error', ERR_BAD_PARAMS, str(exc), False)
    except engine.EngineError as exc:
        log.warning('synthesize %s failed: %s', req_id, exc)
        outcome = ('error', ERR_INTERNAL, str(exc), False)
    except OSError as exc:
        log.exception('synthesize %s failed', req_id)
        if getattr(exc, 'errno', None) in _DISK_FULL_ERRNOS:
            outcome = ('error', ERR_DISK_FULL, f'Out of disk space: {exc}', False)
        else:
            outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('synthesize %s failed', req_id)
        outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'kittentts-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': 'tts.synthesize',
            'languages': list(voices.LANGUAGES),
            'voices': voices.manifest_voices(),
            'voice_clone': True,
        }],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == 'tts.synthesize':
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('tts.synthesize without id')
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_synthesize, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Check what the frozen bundle needs, without the network or a model.

    Both programs must be in the bundle and start (kitten-ref loads LibTorch to
    print its version), the English grammars must be there, and the TLS and
    hashing modules the downloads use.
    """
    import hashlib
    import ssl

    report: dict = {'version': ADDON_VERSION}
    ok = True
    for name in ('kitten-tts', 'kitten-ref'):
        binary = engine.binary_path(name)
        try:
            engine.ensure_executable(binary)
            done = subprocess.run([str(binary), '--version' if name == 'kitten-ref' else '--help'],
                                  capture_output=True, text=True, timeout=120, env=engine.child_env())
            first = (done.stdout or done.stderr).strip().splitlines()[:1]
            report[name] = first[0] if first else f'exit {done.returncode}'
            ok = ok and done.returncode == 0 and bool(first)
        except (OSError, subprocess.SubprocessError, engine.EngineError) as exc:
            report[name] = f'{type(exc).__name__}: {exc}'
            ok = False
    report['text_data'] = (engine.text_data_dir() / 'en' / 'tagger.fst').is_file()
    report['voices'] = len(voices.BUILT_IN)
    report['openssl'] = ssl.OPENSSL_VERSION
    report['sha256'] = hashlib.sha256(b'').hexdigest()[:8]
    report['ca_bundle'] = os.environ.get('SSL_CERT_FILE', '')
    report['ok'] = ok and report['text_data']
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0 if report['ok'] else 1


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'kittentts-addon {ADDON_VERSION}')
        return 0
    logging.basicConfig(
        level=(_env('KITTENTTS_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    tls.configure_ca_bundle()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('kittentts-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())
    if not engine.binary_path().is_file():
        # Not fatal: better to appear in the engine list and fail the first
        # request with a clear message than to vanish from the UI.
        log.warning('the kitten-tts program is missing: %s', engine.binary_path())

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        _stop_workers()
    return rc


if __name__ == '__main__':
    sys.exit(main())
