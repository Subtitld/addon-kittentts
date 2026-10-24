"""The downloader against a local HTTP server: checks, ranges, resumes and cancels.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import http.server
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kittentts_addon import download  # noqa: E402

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
REPO = 'KittenML/kitten-tts-2'
REVISION = 'a' * 40


class _Handler(http.server.BaseHTTPRequestHandler):
    requests: list[tuple[str, str]] = []
    honour_ranges = True

    def do_GET(self):  # noqa: N802
        wanted = self.headers.get('Range') or ''
        type(self).requests.append((self.path, wanted))
        start, end = 0, len(PAYLOAD) - 1
        if wanted and self.honour_ranges:
            low, _, high = wanted.split('=')[1].partition('-')
            start, end = int(low), int(high) if high else len(PAYLOAD) - 1
            self.send_response(206)
        else:
            self.send_response(200)
        body = PAYLOAD[start:end + 1]
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture()
def server(monkeypatch, tmp_path):
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('HF_ENDPOINT', f'http://127.0.0.1:{httpd.server_address[1]}')
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    _Handler.requests = []
    _Handler.honour_ranges = True
    yield httpd
    httpd.shutdown()


def _pin(sha256: str | None = None) -> dict:
    return {'path': 'cpp/model-tq2_1.gguf', 'size': len(PAYLOAD),
            'sha256': sha256 or hashlib.sha256(PAYLOAD).hexdigest()}


def _ensure(pins, cancel=lambda: False, progress=lambda *_: None):
    download.ensure(pins, REPO, REVISION, progress, cancel)


def test_downloads_checks_and_stamps(server, tmp_path):
    pin = _pin()
    seen = []
    _ensure([pin], progress=lambda done, total: seen.append((done, total)))
    path = download.local_path(pin, REVISION)
    assert path == tmp_path / 'kittentts' / REVISION / 'cpp' / 'model-tq2_1.gguf'
    assert path.read_bytes() == PAYLOAD
    assert download.is_present(pin, REVISION)
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert _Handler.requests == [(f'/{REPO}/resolve/{REVISION}/cpp/model-tq2_1.gguf', '')]
    _ensure([pin])  # present now: nothing is fetched again
    assert len(_Handler.requests) == 1


def test_a_range_pin_fetches_only_its_bytes(server, tmp_path):
    piece = PAYLOAD[1000:3048]
    pin = {'path': 'lm/model-ternary.safetensors', 'offset': 1000, 'size': len(piece),
           'sha256': hashlib.sha256(piece).hexdigest(), 'save_as': 'spk_proj.bin'}
    _ensure([pin])
    assert _Handler.requests == [(f'/{REPO}/resolve/{REVISION}/lm/model-ternary.safetensors', 'bytes=1000-3047')]
    assert (tmp_path / 'kittentts' / REVISION / 'spk_proj.bin').read_bytes() == piece


def test_a_range_pin_refuses_a_server_that_sends_the_whole_file(server):
    _Handler.honour_ranges = False
    piece = PAYLOAD[1000:3048]
    pin = {'path': 'lm/x.safetensors', 'offset': 1000, 'size': len(piece),
           'sha256': hashlib.sha256(piece).hexdigest(), 'save_as': 'spk_proj.bin'}
    with pytest.raises(download.DownloadError, match='byte range'):
        _ensure([pin])


def test_a_url_pin_lands_in_its_own_folder(server, tmp_path):
    base = f'http://127.0.0.1:{server.server_address[1]}'
    pin = {'url': f'{base}/releases/download/ref-encoder-1/ref_encoder-1.pt', 'path': 'ref_encoder-1.pt',
           'size': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest()}
    _ensure([pin])
    assert _Handler.requests == [('/releases/download/ref-encoder-1/ref_encoder-1.pt', '')]
    assert (tmp_path / 'kittentts' / 'ref-encoder' / 'ref_encoder-1.pt').read_bytes() == PAYLOAD


def test_resumes_from_the_part_file(server):
    pin = _pin()
    part = download.local_path(pin, REVISION).with_name('model-tq2_1.gguf.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    _ensure([pin])
    assert _Handler.requests[0][1] == f'bytes=1000-{len(PAYLOAD) - 1}'
    assert download.local_path(pin, REVISION).read_bytes() == PAYLOAD


def test_a_wrong_sha256_discards_the_file(server):
    pin = _pin(sha256='0' * 64)
    with pytest.raises(download.DownloadError, match='sha256'):
        _ensure([pin])
    path = download.local_path(pin, REVISION)
    assert not path.exists() and not path.with_name(path.name + '.part').exists()


def test_a_cancel_stops_the_download(server):
    pin = _pin()
    with pytest.raises(download.Cancelled):
        _ensure([pin], cancel=lambda: True)
    assert not download.is_present(pin, REVISION)


def test_a_file_without_its_stamp_is_checked_not_fetched(server):
    pin = _pin()
    path = download.local_path(pin, REVISION)
    path.parent.mkdir(parents=True)
    path.write_bytes(PAYLOAD)
    assert not download.is_present(pin, REVISION)
    _ensure([pin])
    assert download.is_present(pin, REVISION)
    assert _Handler.requests == []


def test_an_unreachable_server_is_a_download_error(monkeypatch, tmp_path):
    monkeypatch.setenv('HF_ENDPOINT', 'http://127.0.0.1:9')  # discard port: nothing listens
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    with pytest.raises(download.DownloadError):
        _ensure([_pin()])
