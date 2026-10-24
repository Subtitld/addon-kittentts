"""Fetch pinned files: resumable, checked by size and sha256.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Three kinds of pin (kittentts_addon/pins.json, made by tools/pin_models.py):

- a file of KittenML/kitten-tts-2 at the pinned revision;
- a byte range of one (`offset` set): the speaker head, 2 MB inside the
  language model's weights, which are otherwise not needed;
- a file at a full `url`: the reference encoder, attached to a release of
  this add-on.

They land in the host's model cache:

    $SUBTITLD_MODELS_DIR/kittentts/<revision>/<save_as or path in the repository>
    $SUBTITLD_MODELS_DIR/kittentts/ref-encoder/<file>

A file is downloaded to `<name>.part` and renamed into place only once its size
and sha256 match the pin; an interrupted download resumes from the `.part` on
the next request. A file in place is trusted once a `.ok` stamp beside it says
it was checked; the stamp holds the sha256, so a re-pinned file is checked
again. $HF_ENDPOINT points the Hugging Face downloads at a mirror.
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from kittentts_addon import __version__

log = logging.getLogger('kittentts_addon.download')

_CHUNK = 1 << 20


class Cancelled(Exception):
    """The request was cancelled."""


class DownloadError(Exception):
    """A file could not be fetched, or did not check out."""


def cache_root() -> Path:
    """$SUBTITLD_MODELS_DIR/kittentts, else ~/.cache/subtitld/models/kittentts."""
    base = (os.environ.get('SUBTITLD_MODELS_DIR') or '').strip()
    root = Path(os.path.expanduser(base)) if base else Path.home() / '.cache' / 'subtitld' / 'models'
    return root / 'kittentts'


def local_path(pin: dict, revision: str) -> Path:
    if 'url' in pin:
        return cache_root() / 'ref-encoder' / pin['path']
    name = pin.get('save_as') or pin['path']
    return cache_root() / revision / Path(*name.split('/'))


def _stamp(path: Path) -> Path:
    return path.with_name(path.name + '.ok')


def is_present(pin: dict, revision: str) -> bool:
    """True when the file is in place and was checked against this pin."""
    path = local_path(pin, revision)
    try:
        return (path.is_file() and path.stat().st_size == pin['size']
                and _stamp(path).read_text(encoding='ascii').strip() == pin['sha256'])
    except OSError:
        return False


def missing(pins: list[dict], revision: str) -> list[dict]:
    return [pin for pin in pins if not is_present(pin, revision)]


def _ssl_context() -> ssl.SSLContext:
    # kittentts_addon.tls has set SSL_CERT_FILE for the frozen Linux and macOS
    # builds; elsewhere this is the platform's own trust store.
    return ssl.create_default_context()


def _sha256(path: Path, cancel: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_CHUNK):
            if cancel():
                raise Cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _url(pin: dict, repo: str, revision: str) -> str:
    if 'url' in pin:
        return pin['url']
    endpoint = (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
    return f'{endpoint}/{repo}/resolve/{revision}/{pin["path"]}'


def _fetch(pin: dict, repo: str, revision: str, on_bytes: Callable[[int], None],
           cancel: Callable[[], bool]) -> None:
    dest = local_path(pin, revision)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    size = pin['size']
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        url = _url(pin, repo, revision)
        headers = {'User-Agent': f'subtitld-kittentts-addon/{__version__}'}
        start = pin.get('offset', 0) + have
        ranged = 'offset' in pin or have > 0
        if ranged:
            headers['Range'] = f'bytes={start}-{pin.get("offset", 0) + size - 1}'
        log.info('fetching %s (%s bytes%s)', url, f'{size:,}',
                 f' from byte {start:,}' if ranged else '')
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=60, context=_ssl_context()) as response:
                if ranged and response.status != 206:
                    if 'offset' in pin:
                        raise DownloadError(f'{dest.name}: the server ignored the byte range')
                    have = 0  # the server ignored the resume: start over
                on_bytes(have)
                with part.open('ab' if have else 'wb') as out:
                    remaining = size - have
                    while remaining > 0 and (chunk := response.read(min(_CHUNK, remaining))):
                        if cancel():
                            raise Cancelled()
                        out.write(chunk)
                        remaining -= len(chunk)
                        on_bytes(len(chunk))
        except urllib.error.HTTPError as exc:
            raise DownloadError(f'{dest.name}: HTTP {exc.code} from {url}') from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise DownloadError(f'{dest.name}: {getattr(exc, "reason", exc)}') from exc
    else:
        on_bytes(have)
    got = part.stat().st_size
    if got != size:
        raise DownloadError(f'{dest.name}: got {got:,} bytes, expected {size:,}. Try again to resume.')
    if _sha256(part, cancel) != pin['sha256']:
        part.unlink()
        raise DownloadError(f'{dest.name} does not match its pinned sha256; the download was discarded.')
    os.replace(part, dest)
    _stamp(dest).write_text(pin['sha256'] + '\n', encoding='ascii')


def ensure(pins: list[dict], repo: str, revision: str, progress: Callable[[int, int], None],
           cancel: Callable[[], bool]) -> None:
    """Download whichever of `pins` is missing. `progress(done, total)` is in bytes."""
    todo = missing(pins, revision)
    if not todo:
        return
    total = sum(pin['size'] for pin in todo)
    done = 0

    def on_bytes(count: int) -> None:
        nonlocal done
        done += count
        progress(min(done, total), total)

    for pin in todo:
        dest = local_path(pin, revision)
        if dest.is_file() and dest.stat().st_size == pin['size'] and _sha256(dest, cancel) == pin['sha256']:
            _stamp(dest).write_text(pin['sha256'] + '\n', encoding='ascii')  # checked, stamp lost
            on_bytes(pin['size'])
            continue
        _fetch(pin, repo, revision, on_bytes, cancel)
