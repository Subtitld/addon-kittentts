"""Run kitten-tts and kitten-ref on local files only.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The programs
------------
`kitten-tts` is KittenML's C++ runtime for KittenTTS 2 (kitten-tts-2-cpp, MIT):
a llama.cpp language model writes speech tokens, a TorchScript decoder turns
them into 24 kHz audio. `kitten-ref` is this add-on's own (native/): it turns a
clone reference clip into a voice, with ref_encoder.pt. Both are in the archive
with the LibTorch they link (tools/build_native.py). One process runs per line,
so a cancel kills it and nothing is held between lines.

Local files only
----------------
kitten-tts is pointed at a folder of downloaded files (`--assets`, `--offline`):
it is never handed a URL. That folder is the revision's cache folder, whose
config.json the add-on writes itself: upstream's config, with a manifest naming
the downloaded model and decoder, and `run/voice.json`, rewritten for each line
with the one voice the line needs. The built-in voices come split out of
upstream's 97 MB voices.json once, so a line reads one voice, not all of them.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from kittentts_addon import download

log = logging.getLogger('kittentts_addon.engine')

_PINS_FILE = Path(__file__).resolve().parent / 'pins.json'
_pins: dict | None = None

SAMPLE_RATE = 24000
REFERENCE_RATE = 16000
VOICE_NAME = 'Line'   # the one voice in run/voice.json

# How the process ends when the CPU lacks an instruction the build uses.
# The x86-64 builds need AVX2, FMA and F16C (x86-64-v3): Intel since 2013
# (Haswell), AMD since 2015 (Excavator), but not every Celeron, Pentium or Atom.
_SIGILL = -4
_STATUS_ILLEGAL_INSTRUCTION = 0xC000001D


class EngineError(Exception):
    """A program failed; the message says why."""


class UnsupportedCPU(EngineError):
    """The build needs instructions this CPU lacks."""


class UnreadableReference(EngineError):
    """The clone reference is not audio this add-on can read."""


def pins() -> dict:
    global _pins
    if _pins is None:
        _pins = json.loads(_PINS_FILE.read_text(encoding='utf-8'))
    return _pins


def revision() -> str:
    return pins()['revision']


def base_pins() -> list[dict]:
    """What every line needs: the model, the decoder, the voices, the config, the licence."""
    files = pins()['files']
    return [files[k] for k in ('config', 'licence', 'voices', 'decoder', 'model')]


def clone_pins() -> list[dict]:
    """What a clone needs on top: the speaker head and the reference encoder."""
    return [pins()['files']['spk_proj'], pins()['ref_encoder']]


def _path(key: str) -> Path:
    return download.local_path(pins()['files'][key], revision())


def revision_root() -> Path:
    return download.cache_root() / revision()


# ---------------------------------------------------------------------------
# The programs
# ---------------------------------------------------------------------------
def _exe(name: str) -> str:
    return name + ('.exe' if sys.platform == 'win32' else '')


def assets_dir() -> Path:
    """$KITTENTTS_ASSETS, else the archive's assets/, else build/assets/ in a source tree."""
    override = (os.environ.get('KITTENTTS_ASSETS') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    bundled = getattr(sys, '_MEIPASS', None)  # PyInstaller onedir: the `_internal` folder
    if bundled:
        return Path(bundled) / 'assets'
    return Path(__file__).resolve().parent.parent / 'build' / 'assets'


def binary_path(name: str = 'kitten-tts') -> Path:
    return assets_dir() / _exe(name)


def text_data_dir() -> Path:
    return assets_dir() / 'text-data'


def ensure_executable(binary: Path) -> None:
    """Make a program runnable.

    Subtitld installs an add-on with ZipFile.extractall, which drops permission
    bits, and then marks only the manifest's `executable` as runnable. The
    add-on's install folder is the user's own, so the bit can be set here.
    """
    if sys.platform == 'win32' or os.access(binary, os.X_OK):
        return
    try:
        binary.chmod(binary.stat().st_mode | 0o111)
    except OSError as exc:
        raise EngineError(f'Cannot make {binary} executable: {exc}. Reinstall the add-on.') from exc


def _no_window_startupinfo():
    """Keep a console window from flashing on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def child_env() -> dict:
    """The environment for kitten-tts, kitten-ref and ffmpeg.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH; the programs find their LibTorch through their own rpath
    and must load the system's C++ runtime, so the host's value is restored
    (PyInstaller keeps it in LD_LIBRARY_PATH_ORIG).
    """
    env = dict(os.environ)
    if getattr(sys, 'frozen', False) and sys.platform.startswith('linux'):
        original = env.pop('LD_LIBRARY_PATH_ORIG', None)
        if original is not None:
            env['LD_LIBRARY_PATH'] = original
        else:
            env.pop('LD_LIBRARY_PATH', None)
    for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN_PATH'):
        env.pop(name, None)
    env['HF_HUB_OFFLINE'] = '1'
    return env


def _cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(','):
        if '-' in part:
            low, high = part.split('-')
            cpus.extend(range(int(low), int(high) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def performance_cores() -> int:
    """Physical performance cores of a hybrid CPU; 0 when not hybrid or not known.

    The slower efficiency cores hold every step back: on a 13th-gen Intel with 8
    performance and 8 efficiency cores, 8 threads are faster than 16.
    """
    try:
        if sys.platform.startswith('linux'):
            cpus = _cpu_list(Path('/sys/devices/cpu_core/cpus').read_text())
            siblings = Path(f'/sys/devices/system/cpu/cpu{cpus[0]}/topology/thread_siblings_list')
            return max(1, len(cpus) // max(1, len(_cpu_list(siblings.read_text()))))
        if sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu'],
                                 capture_output=True, text=True, timeout=5)
            counts = [int(line) for line in out.stdout.split()]
            return counts[0] if len(counts) == 2 and counts[1] > 0 else 0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return 0


def default_threads() -> int:
    """Performance cores, else one per core, at most 8.

    One per core is half the logical CPUs, except on a Mac: Apple Silicon has
    no hyper-threading.
    """
    logical = os.cpu_count() or 2
    cores = performance_cores() or (logical if sys.platform == 'darwin' else max(1, logical // 2))
    return max(1, min(8, cores))


def _illegal_instruction(returncode: int) -> bool:
    return returncode == _SIGILL or (sys.platform == 'win32'
                                     and returncode & 0xFFFFFFFF == _STATUS_ILLEGAL_INSTRUCTION)


def _run(cmd: list[str], cancel: threading.Event, what: str) -> list[str]:
    """Run a program to the end, or kill it on cancel; raise EngineError on failure.

    Returns the program's own messages (its last lines, without llama.cpp's
    loading chatter), for the log.
    """
    binary = Path(cmd[0])
    if not binary.is_file():
        raise EngineError(f'{binary.name} is missing from the add-on ({binary}). Reinstall the add-on.')
    ensure_executable(binary)
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=errors, stderr=errors,
                                env=child_env(), startupinfo=_no_window_startupinfo())
        while True:
            try:
                returncode = proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancel.is_set():
                    proc.kill()
                    proc.wait()
                    raise download.Cancelled()
        errors.seek(0)
        lines = [line for line in errors.read().decode('utf-8', 'replace').splitlines()
                 if line.strip() and not line.startswith(('llama_', 'load', 'print_info'))]
        if returncode != 0:
            if _illegal_instruction(returncode):
                raise UnsupportedCPU(
                    'This processor lacks the AVX2, FMA and F16C instructions KittenTTS needs '
                    '(Intel processors since 2013, AMD since 2015, but not every Celeron, '
                    'Pentium or Atom). Pick another voice engine.')
            tail = ' / '.join(lines[-4:])
            raise EngineError(f'{what} failed (exit {returncode}): {tail or "no output"}')
    return lines[-8:]


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
def builtin_voice_file(name: str) -> Path:
    """One built-in voice as its own JSON file, split out of voices.json on first use."""
    folder = revision_root() / 'voices'
    target = folder / f'{name}.json'
    stamp = folder / '.split-ok'
    want = pins()['files']['voices']['sha256']
    try:
        if stamp.read_text(encoding='ascii').strip() == want and target.is_file():
            return target
    except OSError:
        pass
    started = time.monotonic()
    every = json.loads(_path('voices').read_text(encoding='utf-8'))
    folder.mkdir(parents=True, exist_ok=True)
    for voice, entry in every.items():
        scratch = folder / f'{voice}.json.tmp'
        scratch.write_text(json.dumps(entry, separators=(',', ':')), encoding='utf-8')
        os.replace(scratch, folder / f'{voice}.json')
    stamp.write_text(want + '\n', encoding='ascii')
    log.info('split %d built-in voices in %.1fs', len(every), time.monotonic() - started)
    if not target.is_file():
        raise EngineError(f'The voice {name} is not in KittenTTS 2 at revision {revision()[:12]}.')
    return target


def _find_ffmpeg() -> str | None:
    for candidate in ((os.environ.get('SUBTITLD_FFMPEG_EXECUTABLE') or '').strip(),
                      (os.environ.get('FFMPEG_EXECUTABLE') or '').strip()):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def reference_pcm(path: Path) -> bytes:
    """The clip as 16-bit mono PCM at 16 kHz, the encoder's input.

    Subtitld's own clips are already that, and are read with the standard
    library. Anything else goes through ffmpeg.
    """
    try:
        with wave.open(str(path), 'rb') as clip:
            if (clip.getframerate(), clip.getsampwidth(), clip.getnchannels()) == (REFERENCE_RATE, 2, 1):
                data = clip.readframes(clip.getnframes())
                if sys.byteorder == 'big':
                    samples = array.array('h', data)
                    samples.byteswap()
                    data = samples.tobytes()
                return data
    except (wave.Error, EOFError, OSError):
        pass
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise UnreadableReference(f'The reference {path.name} is not 16 kHz mono 16-bit WAV, and no ffmpeg '
                                  f'was found to convert it. Use such a WAV file.')
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', str(path),
           '-vn', '-ac', '1', '-ar', str(REFERENCE_RATE), '-f', 's16le', '-c:a', 'pcm_s16le', '-']
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, env=child_env(),
                          startupinfo=_no_window_startupinfo())
    if done.returncode != 0 or not done.stdout:
        tail = ' / '.join(done.stderr.decode('utf-8', 'replace').strip().splitlines()[-3:])
        raise UnreadableReference(f'Could not read the reference {path.name}: {tail or "no audio"}')
    return done.stdout


def clone_voice_file(reference: Path, threads: int, cancel: threading.Event) -> Path:
    """The clip prepared as a voice (no transcript), cached by the clip's content."""
    pcm = reference_pcm(reference)
    if len(pcm) < REFERENCE_RATE:  # half a second of 16-bit samples
        raise UnreadableReference(f'The reference {reference.name} is shorter than half a second.')
    digest = hashlib.sha256()
    digest.update(pins()['ref_encoder']['sha256'].encode())
    digest.update(pins()['files']['spk_proj']['sha256'].encode())
    digest.update(pcm)
    target = download.cache_root() / 'clones' / f'{digest.hexdigest()[:32]}.json'
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, scratch_name = tempfile.mkstemp(prefix='.reference-', suffix='.s16', dir=target.parent)
    with os.fdopen(fd, 'wb') as out:
        out.write(pcm)
    scratch = Path(scratch_name)
    started = time.monotonic()
    try:
        _run([str(binary_path('kitten-ref')),
              '--encoder', str(download.local_path(pins()['ref_encoder'], revision())),
              '--spk-proj', str(_path('spk_proj')),
              '--input', str(scratch), '--output', str(target), '--threads', str(threads)],
             cancel, 'Preparing the cloned voice')
    finally:
        scratch.unlink(missing_ok=True)
    log.info('prepared a clone voice from %s (%.1f s of audio) in %.1fs', reference.name,
             len(pcm) / 2 / REFERENCE_RATE, time.monotonic() - started)
    return target


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
@dataclass
class Speech:
    path: Path
    duration_sec: float
    sample_rate: int
    seconds: float  # wall time


def _write_run_files(voice_file: Path, transcript: str | None) -> None:
    """run/voice.json with the line's voice, and the config.json naming it."""
    root = revision_root()
    entry = json.loads(voice_file.read_text(encoding='utf-8'))
    if transcript is not None:
        entry['transcript'] = transcript
    entry.setdefault('transcript', '')
    voice_path = root / 'run' / 'voice.json'
    voice_path.parent.mkdir(parents=True, exist_ok=True)
    voice_path.write_text(json.dumps({VOICE_NAME: entry}, ensure_ascii=False, separators=(',', ':')),
                          encoding='utf-8')
    files = pins()['files']
    config = json.loads(_path('config').read_text(encoding='utf-8'))
    config['default_decoder'] = 'default'
    config['cpp'] = {
        'version': 1,
        'gguf': {'file': files['model']['path'], 'size': files['model']['size']},
        'decoders': {'default': {
            'torchscript': {'file': files['decoder']['path'], 'size': files['decoder']['size']},
            'voices': {'file': 'run/voice.json', 'size': voice_path.stat().st_size},
        }},
    }
    scratch = root / 'config.json.tmp'
    scratch.write_text(json.dumps(config, indent=1), encoding='utf-8')
    os.replace(scratch, root / 'config.json')


def read_float_wav(path: Path) -> tuple[array.array, int]:
    """Samples and rate of a mono WAV: 32-bit float (what kitten-tts writes) or 16-bit PCM."""
    data = path.read_bytes()
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise EngineError(f'{path.name} is not a WAV file')
    fmt = None
    position = 12
    while position + 8 <= len(data):
        chunk, size = data[position:position + 4], struct.unpack_from('<I', data, position + 4)[0]
        body = data[position + 8:position + 8 + size]
        if chunk == b'fmt ':
            fmt = struct.unpack_from('<HHIIHH', body, 0)
        elif chunk == b'data' and fmt is not None:
            code, channels, rate, _, _, bits = fmt
            if channels != 1:
                raise EngineError(f'kitten-tts wrote {channels} channels, expected mono')
            if code == 3 and bits == 32:
                samples = array.array('f', body[:len(body) // 4 * 4])
            elif code == 1 and bits == 16:
                samples = array.array('h', body[:len(body) // 2 * 2])
                samples = array.array('f', (s / 32768.0 for s in samples))
            else:
                raise EngineError(f'kitten-tts wrote WAV format {code} at {bits} bits')
            if sys.byteorder == 'big':
                samples.byteswap()
            return samples, rate
        position += 8 + size + (size & 1)
    raise EngineError(f'{path.name} has no audio')


def trim_silence(samples: array.array, rate: int, keep_sec: float = 0.1,
                 below_peak_db: float = 30.0) -> array.array:
    """The samples without the silence before and after the speech.

    KittenTTS often opens or closes a line with up to a second and a half of
    near silence or a breath, and the host places the line at its subtitle's
    start, so the speech would come late. Silence is any 20 ms frame quieter
    than the loudest frame by `below_peak_db`; `keep_sec` of it stays on each
    side, so soft onsets and decaying endings are kept.
    """
    frame = max(1, rate // 50)
    energies = []
    for start in range(0, len(samples), frame):
        chunk = samples[start:start + frame]
        energies.append(sum(x * x for x in chunk) / len(chunk))
    peak = max(energies, default=0.0)
    if peak <= 0.0:
        return samples
    floor = peak * 10.0 ** (-below_peak_db / 10.0)
    loud = [i for i, energy in enumerate(energies) if energy > floor]
    keep = int(keep_sec * rate)
    begin = max(0, loud[0] * frame - keep)
    end = min(len(samples), (loud[-1] + 1) * frame + keep)
    return samples[begin:end]


def write_pcm16(path: Path, samples: array.array, rate: int) -> None:
    pcm = array.array('h', (int(max(-1.0, min(1.0, s)) * 32767.0) for s in samples))
    if sys.byteorder == 'big':
        pcm.byteswap()
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def synthesize(text: str, voice_file: Path, output: Path, *, transcript: str | None = None,
               use_reference: bool = True, normalize: bool = True, seed: int = 42,
               preset: str = 'stable', threads: int = 0,
               cancel: threading.Event | None = None) -> Speech:
    """Speak `text` into `output` (16-bit mono WAV, 24 kHz). The files must be downloaded.

    `transcript` replaces the voice's own (a clone's comes from the host).
    `use_reference=False` leaves the reference clip out of the prompt, so only
    its timbre is used; it has to be when there is no transcript.
    """
    cancel = cancel or threading.Event()
    threads = threads if threads > 0 else default_threads()
    _write_run_files(voice_file, transcript)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the output and renamed, so the host never sees half a file.
    fd, raw_name = tempfile.mkstemp(prefix='.kittentts-', suffix='.wav', dir=output.parent)
    os.close(fd)
    raw = Path(raw_name)
    final = raw.with_name(raw.stem + '-pcm.wav')
    report = raw.with_name(raw.stem + '-report.json')
    cmd = [str(binary_path()), '--assets', str(revision_root()), '--offline',
           '--data', str(text_data_dir()), '--voice', VOICE_NAME, '--text', text,
           '--output', str(raw), '--report', str(report), '--threads', str(threads),
           '--preset', preset, '--max-tokens', str(max_tokens(text))]
    if not normalize:
        cmd.append('--no-normalize')
    if not use_reference:
        cmd.append('--no-reference')
    started = time.monotonic()
    try:
        for attempt in range(ATTEMPTS):
            attempt_seed = (seed + 7919 * attempt) & 0x7FFFFFFF
            messages = _run(cmd + ['--seed', str(attempt_seed)], cancel, 'kitten-tts')
            if finished(report):
                break
            log.warning('kitten-tts did not finish the line with seed %d (attempt %d of %d): %s',
                        attempt_seed, attempt + 1, ATTEMPTS, summary(report) or ' / '.join(messages[-3:]))
        else:
            raise EngineError('KittenTTS could not finish this line: it kept producing silence. '
                              'Try another seed or another voice, or reword the line.')
        samples, rate = read_float_wav(raw)
        samples = trim_silence(samples, rate)
        if not samples:
            raise EngineError('kitten-tts wrote no audio')
        write_pcm16(final, samples, rate)
        os.replace(final, output)
    finally:
        for path in (raw, final, report):
            path.unlink(missing_ok=True)
    return Speech(output, len(samples) / float(rate), rate, time.monotonic() - started)


# A line can go wrong: the model may fall into a run of silence tokens instead
# of speaking, and either reach its token budget or stop with no usable scores
# (native/patches/0001). Either way the report marks the chunk unterminated,
# and the line is tried again with another seed.
ATTEMPTS = 3


def max_tokens(text: str) -> int:
    """The token budget for a line: 25 tokens are a second of speech.

    Upstream gives a single chunk 1000 tokens, 40 seconds. Six tokens per
    character is several times what speech takes in any of the languages (a
    Chinese character, the slowest, is about a quarter of a second), and keeps a
    runaway line from costing a minute before it is retried.
    """
    return max(250, min(1000, 6 * len(text)))


def summary(report: Path) -> str:
    """Per chunk: tokens generated, of which speech, and whether it ended."""
    try:
        chunks = json.loads(report.read_text(encoding='utf-8'))['chunks']
    except (OSError, ValueError, KeyError, TypeError):
        return ''
    return '; '.join(f'{len(c.get("generated", []))} tokens, {len(c.get("audio_tokens", []))} audio, '
                     f'{"ended" if c.get("terminated") else "unfinished"}' for c in chunks)


def finished(report: Path) -> bool:
    """True when every chunk of the line ended on its own stop token."""
    try:
        chunks = json.loads(report.read_text(encoding='utf-8'))['chunks']
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(chunks) and all(chunk.get('terminated') for chunk in chunks)
