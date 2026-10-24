#!/usr/bin/env python3
"""Build kitten-tts and kitten-ref, and gather what they need into build/assets/.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

kitten-tts is KittenML's C++ runtime for KittenTTS 2 (the kitten-tts-2-cpp
fork of llama.cpp, MIT), built at the commit pinned below. kitten-ref is this
add-on's clone-reference prep tool (native/). Both link LibTorch, taken from
the CPU torch wheel installed in the Python running this script; the wheel's
own licence and third-party notices ship with it.

    python tools/build_native.py            # needs git, cmake, a C++20 compiler, torch==TORCH_VERSION

Output, all of it copied into the frozen add-on:

    build/assets/kitten-tts[.exe], kitten-ref[.exe]
    build/assets/lib/            LibTorch (beside the programs on Windows)
    build/assets/text-data/en/   the English text normalizer's grammars
    build/assets/licences/       notices harvested from the sources and the wheel

The llama.cpp parts are linked statically. ggml is built for a fixed baseline,
not for the build machine: x86-64 needs AVX2, FMA and F16C (as Pocket TTS), Apple
Silicon gets dotprod and fp16. ggml's own OpenMP is off, so LibTorch's runtime
is the only one in the process.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD = REPO_ROOT / 'build'
ASSETS = BUILD / 'assets'

KITTEN_TTS_REPO = 'https://github.com/KittenML/kitten-tts-2-cpp'
KITTEN_TTS_COMMIT = '2b9912e72245f5597c664d2be44a6e7ba0a3f293'
TORCH_VERSION = '2.14.1'


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def exe(name: str) -> str:
    return name + ('.exe' if sys.platform == 'win32' else '')


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
PATCHES = sorted((REPO_ROOT / 'native' / 'patches').glob('*.patch'))


def checkout(source: Path) -> None:
    """kitten-tts-2-cpp at KITTEN_TTS_COMMIT, with the normalizer submodule and native/patches/."""
    if not (source / '.git').exists():
        source.mkdir(parents=True, exist_ok=True)
        run(['git', 'init', '-q'], cwd=source)
        run(['git', 'remote', 'add', 'origin', KITTEN_TTS_REPO], cwd=source)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True).stdout.strip()
    if head != KITTEN_TTS_COMMIT:
        run(['git', 'fetch', '-q', '--depth', '1', 'origin', KITTEN_TTS_COMMIT], cwd=source)
        run(['git', 'checkout', '-q', '--force', KITTEN_TTS_COMMIT], cwd=source)
    run(['git', 'submodule', 'update', '--init', '--depth', '1', 'vendor/kitten-text-processing'], cwd=source)
    run(['git', 'reset', '-q', '--hard'], cwd=source)
    for patch in PATCHES:
        run(['git', 'apply', '--whitespace=nowarn', patch], cwd=source)


def torch_dirs() -> tuple[Path, Path, str]:
    import torch  # noqa: PLC0415 — only the build needs it
    version = torch.__version__.split('+')[0]
    if version != TORCH_VERSION:
        raise SystemExit(f'FATAL: torch {torch.__version__} is installed; this build pins {TORCH_VERSION} '
                         f'(ref_encoder.pt and the decoder are TorchScript, read by this LibTorch)')
    package = Path(torch.__file__).resolve().parent
    return package / 'lib', Path(torch.utils.cmake_prefix_path), torch.__version__


def cpu_flags() -> list[str]:
    machine = platform.machine().lower()
    if machine in ('x86_64', 'amd64'):
        return ['-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
                '-DGGML_AVX512=OFF', '-DGGML_AMX_TILE=OFF', '-DGGML_AMX_INT8=OFF']
    if machine in ('arm64', 'aarch64') and sys.platform == 'darwin':
        return ['-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16']
    return []


def rpath_flags() -> list[str]:
    if sys.platform == 'darwin':
        rpath = '@loader_path/lib'
    elif sys.platform.startswith('linux'):
        rpath = '$ORIGIN/lib'
    else:
        return []
    return [f'-DCMAKE_INSTALL_RPATH={rpath}', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
            '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF']


def common_flags(prefix: Path) -> list[str]:
    flags = ['-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_PREFIX_PATH={prefix}']
    if sys.platform == 'darwin':
        flags.append('-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0')
    if sys.platform == 'win32':
        # UTF-8 as the process code page: the text and the paths arrive intact
        # (native/utf8.manifest).
        manifest = (REPO_ROOT / 'native' / 'utf8.manifest').as_posix()
        flags.append(f'-DCMAKE_EXE_LINKER_FLAGS=/MANIFEST:EMBED /MANIFESTINPUT:{manifest}')
    return flags + rpath_flags()


def build(source: Path, prefix: Path, jobs: int) -> tuple[Path, Path]:
    fork_build = BUILD / 'kitten-tts-build'
    run(['cmake', '-S', source, '-B', fork_build, *common_flags(prefix),
         '-DCMAKE_CXX_STANDARD=20', '-DCMAKE_CXX_STANDARD_REQUIRED=ON',
         '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
         '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_BLAS=OFF', '-DGGML_VULKAN=OFF',
         '-DLLAMA_BUILD_KITTEN_TTS=ON', '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_TESTS=OFF',
         '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_CURL=OFF', f'-DPython3_EXECUTABLE={sys.executable}',
         *cpu_flags(),
         # MSVC makes u8"" literals char8_t in C++20, which llama.cpp's sources
         # do not expect (they build as C++17 upstream).
         *(['-DCMAKE_CXX_FLAGS=/Zc:char8_t- /EHsc /utf-8'] if sys.platform == 'win32' else [])])
    run(['cmake', '--build', fork_build, '--config', 'Release', '--target', 'kitten-tts', '-j', str(jobs)])

    ref_build = BUILD / 'kitten-ref-build'
    run(['cmake', '-S', REPO_ROOT / 'native', '-B', ref_build, *common_flags(prefix),
         f'-DKITTEN_TTS_SOURCE={source}'])
    run(['cmake', '--build', ref_build, '--config', 'Release', '-j', str(jobs)])

    def found(tree: Path, name: str) -> Path:
        hits = [p for p in tree.rglob(exe(name)) if p.is_file() and 'CMakeFiles' not in p.parts]
        if not hits:
            raise SystemExit(f'FATAL: {exe(name)} was not built under {tree}')
        return hits[0]

    return found(fork_build, 'kitten-tts'), found(ref_build, 'kitten-ref')


# ---------------------------------------------------------------------------
# Which LibTorch libraries the programs load
# ---------------------------------------------------------------------------
def _elf_needed(path: Path) -> list[str]:
    out = subprocess.run(['readelf', '-d', str(path)], capture_output=True, text=True, check=True,
                         env={**os.environ, 'LC_ALL': 'C'}).stdout
    return re.findall(r'\(NEEDED\).*\[(.+?)\]', out)


def _macho_needed(path: Path) -> list[str]:
    out = subprocess.run(['otool', '-L', str(path)], capture_output=True, text=True, check=True).stdout
    return [Path(line.strip().split(' ')[0]).name for line in out.splitlines()[1:] if line.strip()]


def _pe_needed(path: Path) -> list[str]:
    """DLL names in a PE file's import table (and its delay-load table)."""
    data = path.read_bytes()
    pe = struct.unpack_from('<I', data, 0x3C)[0]
    sections_count = struct.unpack_from('<H', data, pe + 6)[0]
    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
    optional = pe + 24
    magic = struct.unpack_from('<H', data, optional)[0]
    directories = optional + (112 if magic == 0x20B else 96)
    sections = []
    for i in range(sections_count):
        base = optional + optional_size + 40 * i
        vsize, vaddr, rsize, raddr = struct.unpack_from('<IIII', data, base + 8)
        sections.append((vaddr, max(vsize, rsize), raddr))

    def offset(rva: int) -> int:
        for vaddr, size, raddr in sections:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + raddr
        raise ValueError(f'RVA {rva:#x} is outside every section of {path}')

    def cstring(rva: int) -> str:
        start = offset(rva)
        return data[start:data.index(b'\0', start)].decode('ascii')

    names = []
    for index, stride, name_at in ((1, 20, 12), (13, 32, 4)):  # imports, delay imports
        rva, size = struct.unpack_from('<II', data, directories + 8 * index)
        if not rva or not size:
            continue
        entry = offset(rva)
        while True:
            name_rva = struct.unpack_from('<I', data, entry + name_at)[0]
            if not name_rva:
                break
            names.append(cstring(name_rva))
            entry += stride
    return names


def needed(path: Path) -> list[str]:
    if sys.platform == 'win32':
        return _pe_needed(path)
    if sys.platform == 'darwin':
        return _macho_needed(path)
    return _elf_needed(path)


MSVC_RUNTIME = re.compile(r'^(vcruntime140(_1|_threads)?|msvcp140(_\w+)?|vcomp140|concrt140)\.dll$', re.I)


def torch_closure(programs: list[Path], torch_lib: Path) -> list[Path]:
    """The libraries from torch/lib the programs load, directly or not.

    On Windows also the Visual C++ runtime they load, from the system folder:
    a Windows without the VC++ redistributable installed has none of it.
    """
    available = {p.name.lower(): p for p in torch_lib.iterdir() if p.is_file()}
    if sys.platform == 'win32':
        system = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32'
        for dll in system.glob('*.dll'):
            if MSVC_RUNTIME.match(dll.name) and dll.name.lower() not in available:
                available[dll.name.lower()] = dll
    todo, seen = list(programs), set()
    chosen: list[Path] = []
    while todo:
        for name in needed(todo.pop()):
            hit = available.get(name.lower())
            if hit is not None and hit.name not in seen:
                seen.add(hit.name)
                chosen.append(hit)
                todo.append(hit)
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------
def harvest_licences(source: Path, torch_version: str) -> None:
    out = ASSETS / 'licences'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copies = {
        'kitten-tts-2-cpp/LICENSE': source / 'LICENSE',
        'kitten-tts-2-cpp/LICENSE-KITTENTTS': source / 'tools/kitten-tts/LICENSE-KITTENTTS',
        'kitten-tts-2-cpp/AUTHORS': source / 'AUTHORS',
        'kitten-text-processing/LICENSE': source / 'vendor/kitten-text-processing/LICENSE',
        'kitten-text-processing/NOTICE': source / 'vendor/kitten-text-processing/NOTICE',
        'kitten-text-processing/LICENSE-SACREMOSES': source / 'vendor/kitten-text-processing/LICENSE-SACREMOSES',
        'nlohmann-json/LICENSE': source / 'licenses/LICENSE-jsonhpp',
    }
    for target, origin in copies.items():
        if not origin.is_file():
            raise SystemExit(f'FATAL: {origin} is missing; check the notices for this kitten-tts-2-cpp commit')
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, out / target)
    (out / 'kitten-tts-2-cpp' / 'SOURCE.txt').write_text(
        f'kitten-tts and kitten-ref were built from {KITTEN_TTS_REPO} at {KITTEN_TTS_COMMIT}\n'
        f'(with its vendor/kitten-text-processing submodule), and from native/ in this\n'
        f'add-on\'s repository, against LibTorch {torch_version}.\n\n'
        f'Changed from upstream by this add-on (native/patches/ in its repository):\n'
        + ''.join(f'  {patch.name}\n' for patch in PATCHES), encoding='utf-8')

    from importlib.metadata import distribution  # noqa: PLC0415
    dist = distribution('torch')
    torch_out = out / 'pytorch'
    for entry in dist.files or []:
        parts = Path(str(entry)).parts
        if len(parts) >= 2 and parts[0].endswith('.dist-info') and parts[1] == 'licenses':
            target = torch_out.joinpath(*parts[2:])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dist.locate_file(entry), target)
    if not (torch_out / 'LICENSE').is_file():
        raise SystemExit('FATAL: the torch wheel carries no licenses/LICENSE')
    (torch_out / 'SOURCE.txt').write_text(
        f'LibTorch from the torch {torch_version} wheel (CPU) on PyPI / download.pytorch.org.\n'
        f'LICENSE and third_party/ are the wheel\'s own notices for what it bundles.\n', encoding='utf-8')


# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=BUILD / 'kitten-tts-2-cpp')
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    args = parser.parse_args(argv)

    torch_lib, prefix, torch_version = torch_dirs()
    checkout(args.source)
    kitten_tts, kitten_ref = build(args.source, prefix, args.jobs)

    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True)
    programs = [ASSETS / kitten_tts.name, ASSETS / kitten_ref.name]
    shutil.copy2(kitten_tts, programs[0])
    shutil.copy2(kitten_ref, programs[1])
    lib_dir = ASSETS if sys.platform == 'win32' else ASSETS / 'lib'  # Windows looks beside the .exe
    lib_dir.mkdir(exist_ok=True)
    libraries = torch_closure(programs, torch_lib)
    for library in libraries:
        shutil.copy2(library, lib_dir / library.name)
    if not any(lib.name.lower().startswith(('libtorch_cpu', 'torch_cpu')) for lib in libraries):
        raise SystemExit(f'FATAL: no torch_cpu library among {[lib.name for lib in libraries]}')

    data = next((p for p in (BUILD / 'kitten-tts-build').rglob('data') if (p / 'en' / 'tagger.fst').is_file()), None)
    if data is None:
        raise SystemExit('FATAL: the normalizer grammars (text-processing/data/en) were not built')
    shutil.copytree(data / 'en', ASSETS / 'text-data' / 'en')

    harvest_licences(args.source, torch_version)
    total = sum(p.stat().st_size for p in ASSETS.rglob('*') if p.is_file())
    print(f'build/assets: {total / 2**20:.0f} MiB')
    for path in sorted(ASSETS.rglob('*')):
        if path.is_file() and 'licences' not in path.parts and 'text-data' not in path.parts:
            print(f'  {path.relative_to(ASSETS)}  {path.stat().st_size:,}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
