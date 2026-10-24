#!/usr/bin/env python3
"""Pin the KittenTTS 2 files the add-on downloads: kittentts_addon/pins.json.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Every file is pinned to one revision of KittenML/kitten-tts-2 by size and
sha256 (from the Hub's LFS metadata, or by downloading the small ones). The
speaker head (spk_proj) is not a file: it is 2 MB in the middle of
lm/model-ternary.safetensors, so its byte range is read from the safetensors
header and that range alone is downloaded and hashed. The reference encoder
is this add-on's own file (tools/export_ref_encoder.py), attached to a
prerelease of the add-on's repository; pass its URL and local copy.

    python tools/pin_models.py --revision <sha> --encoder-url URL --encoder ref_encoder.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / 'kittentts_addon' / 'pins.json'
REPO = 'KittenML/kitten-tts-2'
FILES = {
    'config': 'config.json',
    'model': 'cpp/model-tq2_1.gguf',
    'decoder': 'cpp/default/decoder.pt',
    'voices': 'cpp/default/voices.json',
    'licence': 'LICENSE.md',
}
SPK_PROJ_SOURCE = 'lm/model-ternary.safetensors'
SPK_PROJ_TENSORS = ('spk_proj.0.bias', 'spk_proj.0.weight', 'spk_proj.1.bias', 'spk_proj.1.weight')


def api(path: str) -> dict:
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/{path}', timeout=60) as response:
        return json.load(response)


def resolve(revision: str, path: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range:
        headers['Range'] = f'bytes={byte_range[0]}-{byte_range[1] - 1}'
    request = urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{revision}/{path}', headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--encoder-url', required=True)
    parser.add_argument('--encoder', type=Path, required=True, help='local copy of the uploaded ref_encoder.pt')
    args = parser.parse_args(argv)

    info = api(f'revision/{args.revision}?blobs=true')
    revision = info['sha']
    siblings = {s['rfilename']: s for s in info['siblings']}
    files = {}
    for key, path in FILES.items():
        sibling = siblings.get(path)
        if sibling is None:
            raise SystemExit(f'FATAL: {path} is not in {REPO}@{revision}')
        lfs = sibling.get('lfs') or {}
        if lfs.get('sha256'):
            files[key] = {'path': path, 'size': int(lfs['size']), 'sha256': lfs['sha256']}
        else:
            data = resolve(revision, path)
            files[key] = {'path': path, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    # The upstream config is rewritten by the add-on into its own config.json
    # (engine.py), so the downloaded copy keeps another name.
    files['config']['save_as'] = 'upstream-config.json'

    head = resolve(revision, SPK_PROJ_SOURCE, (0, 8))
    header_size = struct.unpack('<Q', head)[0]
    header = json.loads(resolve(revision, SPK_PROJ_SOURCE, (8, 8 + header_size)))
    spans = [header[name]['data_offsets'] for name in SPK_PROJ_TENSORS]
    for name in SPK_PROJ_TENSORS:
        if header[name]['dtype'] != 'BF16':
            raise SystemExit(f'FATAL: {name} is {header[name]["dtype"]}; kitten-ref reads bfloat16')
    if any(spans[i][1] != spans[i + 1][0] for i in range(len(spans) - 1)):
        raise SystemExit(f'FATAL: the spk_proj tensors are not back to back in that order: {spans}')
    start, end = 8 + header_size + spans[0][0], 8 + header_size + spans[-1][1]
    blob = resolve(revision, SPK_PROJ_SOURCE, (start, end))
    if len(blob) != end - start:
        raise SystemExit('FATAL: the server ignored the byte range')
    files['spk_proj'] = {'path': SPK_PROJ_SOURCE, 'offset': start, 'size': end - start,
                         'sha256': hashlib.sha256(blob).hexdigest(), 'save_as': 'spk_proj.bin'}

    encoder = {'url': args.encoder_url, 'path': Path(args.encoder_url).name,
               'size': args.encoder.stat().st_size, 'sha256': sha256_file(args.encoder)}
    pins = {'repo': REPO, 'revision': revision, 'files': files, 'ref_encoder': encoder}
    OUT.write_text(json.dumps(pins, indent=2) + '\n', encoding='utf-8')
    total = sum(f['size'] for k, f in files.items() if k != 'licence')
    print(f'pinned {REPO}@{revision}: {total / 1e6:,.0f} MB to download, encoder {encoder["size"] / 1e6:,.0f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
