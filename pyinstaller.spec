# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld KittenTTS 2 add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Produces `dist/kittentts-addon/` (launcher + `_internal/`). The release step
copies it to `bin/` inside the archive, next to manifest.json.

kitten-tts, kitten-ref, the LibTorch they link and the English normalizer
grammars are copied into `_internal/assets/`. Run tools/build_native.py first:
it builds them into build/assets/.

Run from the add-on root:

    python tools/build_native.py
    pyinstaller pyinstaller.spec --noconfirm
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()
ASSETS = SPEC_ROOT / 'build' / 'assets'
_EXE = '.exe' if sys.platform == 'win32' else ''
PACKAGE = SPEC_ROOT / 'kittentts_addon'

# Fail the build, loudly, if a program or the pins are missing. Otherwise the
# frozen add-on would answer the handshake and then fail the first line.
_MISSING = [str(p) for p in (ASSETS / f'kitten-tts{_EXE}', ASSETS / f'kitten-ref{_EXE}',
                             ASSETS / 'text-data' / 'en' / 'tagger.fst',
                             PACKAGE / 'pins.json', PACKAGE / 'voice_list.json')
            if not p.is_file()]
if _MISSING:
    raise SystemExit(f'FATAL: missing {", ".join(_MISSING)}. '
                     f'Run `python tools/build_native.py` first.')

datas = [(str(PACKAGE / 'pins.json'), 'kittentts_addon'),
         (str(PACKAGE / 'voice_list.json'), 'kittentts_addon')]


# ---------------------------------------------------------------------------
# libexpat sanity check (from addon-whispercpp).
#
# PyInstaller bundles whatever expat it resolved on the build host, for
# CPython's pyexpat. If that host has an OLD expat shadowing the distro one
# (for example a hand-built 2.5.0 in /usr/local/lib), the frozen binary dies
# before reaching Python:
#     Failed to load libpython3.x.so: undefined symbol: XML_SetReparseDeferralEnabled
# Dropping the bundled copy is not a fix either, because the same stale system
# copy is then used. So verify the bundled copy exports the symbol libpython
# needs, and substitute a known-good system copy if it does not.
# ---------------------------------------------------------------------------
_EXPAT_SYMBOL = 'XML_SetReparseDeferralEnabled'


def _exports_symbol(path: str, symbol: str) -> bool:
    try:
        out = subprocess.run(['nm', '-D', '--defined-only', path],
                             capture_output=True, text=True, timeout=30)
        return symbol in out.stdout
    except Exception:
        return True  # no nm (macOS/Windows): assume fine rather than break the build


def _fix_libexpat(entries):
    if not sys.platform.startswith('linux'):
        return entries
    out = []
    for entry in entries:
        name = entry[0] if entry else ''
        src = entry[1] if len(entry) > 1 else ''
        if not os.path.basename(name).startswith('libexpat.so'):
            out.append(entry)
            continue
        if _exports_symbol(src, _EXPAT_SYMBOL):
            out.append(entry)
            continue
        replacement = None
        for cand in ('/usr/lib/x86_64-linux-gnu/libexpat.so.1',
                     '/usr/lib64/libexpat.so.1',
                     '/lib/x86_64-linux-gnu/libexpat.so.1'):
            if os.path.exists(cand) and _exports_symbol(cand, _EXPAT_SYMBOL):
                replacement = cand
                break
        if replacement:
            print(f'WARNING: build host libexpat at {src} lacks {_EXPAT_SYMBOL}; '
                  f'substituting {replacement}')
            out.append((name, replacement) + tuple(entry[2:]))
        else:
            print(f'ERROR: bundled libexpat lacks {_EXPAT_SYMBOL} and no good '
                  f'system copy was found. The frozen binary will not start. '
                  f'Fix the build host (a stale expat in /usr/local/lib is the '
                  f'usual cause).')
            out.append(entry)
    return out


a = Analysis(
    [str(PACKAGE / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6', 'PyQt6', 'PyQt5', 'tkinter', 'Tkinter', '_tkinter',
        'numpy', 'torch', 'torchaudio', 'setuptools', 'pkg_resources', 'unittest', 'pydoc_data', 'pytest',
        'huggingface_hub', 'requests', 'urllib3', 'certifi',
        # Standard modules the add-on never calls, reached only through
        # optional imports (shutil's archive formats, random's hash fallback).
        # Each one would bring a native library, and with it a licence to
        # ship: bzip2, xz, zstd, libmpdec. ssl and hashlib stay: the models
        # download over HTTPS and are checked by sha256, so OpenSSL ships.
        # tests/test_installed.py runs the frozen binary end to end, so a
        # module that turns out to be needed fails the release build, not
        # the user.
        'bz2', '_bz2', 'lzma', '_lzma', 'compression', '_zstd',
        'decimal', '_decimal', '_pydecimal',
    ],
    noarchive=False,
)

a.binaries = _fix_libexpat(a.binaries)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='kittentts-addon',
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False: stripping the native libraries saves little and breaks
    # macOS code signatures.
    strip=False,
    upx=False,
    console=True,  # stdio is the protocol channel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='kittentts-addon',
)

# The programs and LibTorch are things the add-on starts, not libraries it
# loads, so they are copied in after COLLECT rather than handed to PyInstaller.
# Given to PyInstaller, even as data, they are reclassified as binaries: their
# C++ runtime is collected beside them, which they never load (the add-on
# starts them with the system's library path and their own rpath), and on
# macOS they are re-signed. Copied here, they stay byte for byte as built.
# build/assets/licences/ goes into THIRD_PARTY_LICENSES/ instead
# (tools/assemble_licences.py).
import shutil  # noqa: E402

_assets_out = Path(DISTPATH) / 'kittentts-addon' / '_internal' / 'assets'
if _assets_out.exists():
    shutil.rmtree(_assets_out)
shutil.copytree(ASSETS, _assets_out, ignore=shutil.ignore_patterns('licences'), symlinks=False)
