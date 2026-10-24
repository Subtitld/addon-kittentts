#!/usr/bin/env python3
"""Export ref_encoder.pt, the clone-reference encoder kitten-ref runs.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

To clone a voice, the KittenTTS 2 C++ runtime needs the voice prepared:
the clip's S3 speech tokens, a speaker vector, and the s3gen decoder's
conditioning. Upstream computes them in Python, with the kittenml package,
transformers and diffusers (tools/kitten-tts/prepare_reference.py in
kitten-tts-2-cpp). This puts the same computation in one TorchScript file,
so the add-on needs only the LibTorch it already ships for the decoder:

    forward(wav16 (1, N) float32 at 16 kHz, spk_proj (1054720,) bfloat16)
      -> reference_tokens (T,), speaker (2048,), prompt_token (1, P),
         prompt_feat (1, 2P, 80), embedding (1, 192)

What is inside, all MIT-licensed:
- s3gen's S3 tokenizer, mel extractor and CAMPPlus speaker encoder, from
  ResembleAI/chatterbox-turbo (s3gen_meanflow.safetensors). The tokenizer
  descends from FunAudioLLM's CosyVoice (Apache-2.0).
- the pyannote-derived speaker model in KittenML/kitten-tts-2/speaker/.
- code traced from kittenml (Apache-2.0) and s3tokenizer (Apache-2.0).
Not inside: spk_proj, the speaker head of the KittenTTS 2 language model,
which is under the Stellon Labs Community License. It is an input, which the
add-on downloads from KittenML's repository on the user's machine.

How it was made traceable: traced graphs freeze Python-side arithmetic on
shapes. The tokenizer is traced at the longest input it takes (30 s; it is exact
for anything shorter), kaldi.fbank is rewritten in tensor ops (FBank below),
and the length logic (tiling to 6 s, cutting at 10 and 30 s) is scripted, not
traced. The checks at the end compare every piece with the eager code at
several lengths and fail the export on any difference.

Run with the kittenml package installed (pip install kittenml==KITTENML,
torch==TORCH from the CPU index):

    python tools/export_ref_encoder.py --out ref_encoder.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

KITTENML = '0.9.6'
TORCH = '2.14.1'
CHATTERBOX_REPO = 'ResembleAI/chatterbox-turbo'
CHATTERBOX_REVISION = '749d1c1a46eb10492095d68fbcf55691ccf137cd'
KITTEN_REPO = 'KittenML/kitten-tts-2'
KITTEN_REVISION = 'c93cb7c271baa51d9ae8204d5277988906f235a2'
SPK_PROJ_ELEMENTS = 2048 + 2048 * 512 + 2048 + 2048


def noise(seconds: float, rate: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seconds * 100))
    return (torch.randn(1, int(seconds * rate), generator=generator) * 0.1).float()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    from importlib.metadata import version  # noqa: PLC0415
    if version('kittenml') != KITTENML or torch.__version__.split('+')[0] != TORCH:
        raise SystemExit(f'FATAL: needs kittenml {KITTENML} and torch {TORCH}; '
                         f'found {version("kittenml")} and {torch.__version__}')
    import torchaudio  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415
    from torchaudio.compliance import kaldi  # noqa: PLC0415

    from kittenml.kittentts2.s3gen import S3Gen  # noqa: PLC0415
    from kittenml.kittentts2.speaker_embedding import load_speaker_embedding_model  # noqa: PLC0415

    s3 = S3Gen(meanflow=True)
    s3.load_state_dict(load_file(hf_hub_download(CHATTERBOX_REPO, 's3gen_meanflow.safetensors',
                                                 revision=CHATTERBOX_REVISION)), strict=True)
    s3.eval().requires_grad_(False)
    pyannote = load_speaker_embedding_model(
        hf_hub_download(KITTEN_REPO, 'speaker/model.safetensors', revision=KITTEN_REVISION), device='cpu')
    pyannote.eval().requires_grad_(False)

    class Tokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = s3.tokenizer

        def forward(self, wav16):
            return self.m(wav16)[0]

    class Mel24(torch.nn.Module):
        def forward(self, wav24):
            return s3.mel_extractor(wav24).transpose(1, 2)

    class FBank(torch.nn.Module):
        """kaldi.fbank(num_mel_bins=80) at its defaults, in tensor ops only, then mean-normalised.

        torchaudio's fbank counts frames with Python ints, which a trace freezes.
        """

        def __init__(self):
            super().__init__()
            banks, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
            self.register_buffer('banks', torch.nn.functional.pad(banks, (0, 1)).float())
            self.register_buffer('window', torch.hann_window(400, periodic=False,
                                                             dtype=torch.float64).pow(0.85).float())

        def forward(self, wav16):
            frames = wav16[0].unfold(0, 400, 160)
            frames = frames - frames.mean(dim=1, keepdim=True)
            previous = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
            frames = (frames - 0.97 * previous) * self.window
            power = torch.fft.rfft(frames, n=512).abs().pow(2.0)
            mel = torch.clamp_min(power @ self.banks.t(), torch.finfo(torch.float32).eps).log()
            return (mel - mel.mean(dim=0, keepdim=True)).unsqueeze(0)

    class CAMPPlus(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = s3.speaker_encoder

        def forward(self, feats):
            return self.m.forward(feats)

    class Pyannote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = pyannote

        def forward(self, wav16):
            return self.m(wav16.unsqueeze(0))

    with torch.inference_mode():
        tokenizer = torch.jit.trace(Tokenizer().eval(), (noise(30, 16000),), check_trace=False)
        mel24 = torch.jit.trace(Mel24().eval(), (noise(7, 24000),), check_trace=False)
        fbank = torch.jit.trace(FBank().eval(), (noise(7, 16000),), check_trace=False)
        campplus = torch.jit.trace(CAMPPlus().eval(), (fbank(noise(7, 16000)),), check_trace=False)
        speaker = torch.jit.trace(Pyannote().eval(), (noise(7, 16000),), check_trace=False)

    # Every traced piece against the eager code it replaces.
    problems = []
    with torch.inference_mode():
        for seconds in (0.6, 3.1, 5.55, 9.0, 11.9, 20.0, 29.9):
            x = noise(seconds, 16000)
            tokens = s3.tokenizer(x)[0]
            if not torch.equal(tokens, tokenizer(x)):
                problems.append(f'tokenizer differs at {seconds} s')
            features = kaldi.fbank(x, num_mel_bins=80)
            features = (features - features.mean(dim=0, keepdim=True)).unsqueeze(0)
            if features.shape != fbank(x).shape or float((features - fbank(x)).abs().max()) > 5e-3:
                problems.append(f'fbank differs at {seconds} s')
            if float((s3.speaker_encoder.inference(x) - campplus(fbank(x))).abs().max()) > 1e-4:
                problems.append(f'CAMPPlus differs at {seconds} s')
            if float((pyannote(x.unsqueeze(0)) - speaker(x)).abs().max()) > 1e-4:
                problems.append(f'speaker model differs at {seconds} s')
            x24 = noise(seconds, 24000)
            if not torch.allclose(s3.mel_extractor(x24).transpose(1, 2), mel24(x24), atol=1e-5):
                problems.append(f'mel differs at {seconds} s')
    if problems:
        raise SystemExit('FATAL: the traced encoder is not exact:\n  ' + '\n  '.join(problems))

    class RefEncoder(torch.nn.Module):
        """prepare_reference.py for one 16 kHz mono clip."""

        def __init__(self):
            super().__init__()
            self.tokenizer = tokenizer
            self.mel24 = mel24
            self.fbank = fbank
            self.campplus = campplus
            self.speaker = speaker
            self.up = torchaudio.transforms.Resample(16000, 24000)
            self.down = torchaudio.transforms.Resample(24000, 16000)

        def forward(self, wav16: torch.Tensor, spk_proj: torch.Tensor):
            wav16 = wav16[:, :30 * 16000]
            reference_tokens = self.tokenizer(wav16)[0]

            # spk_proj.0.bias, .0.weight, .1.bias, .1.weight, back to back, as in
            # the language model's safetensors: Linear(512, 2048) + LayerNorm(2048).
            b0 = spk_proj[:2048]
            w0 = spk_proj[2048:2048 + 2048 * 512].reshape(2048, 512)
            b1 = spk_proj[2048 + 2048 * 512:4096 + 2048 * 512]
            w1 = spk_proj[4096 + 2048 * 512:6144 + 2048 * 512]
            vector = self.speaker(wav16).reshape(1, 512).float()
            vector = vector / torch.clamp_min(vector.norm(), 1e-12)
            projected = torch.nn.functional.linear(vector.to(torch.bfloat16), w0, b0)
            projected = torch.nn.functional.layer_norm(projected, [2048], w1, b1)
            speaker_vector = projected.float().flatten()

            # The decoder's conditioning, as kittenml's _load_decoder_reference
            # and s3gen.embed_ref make it: at 24 kHz, the speech tiled to 6 s if
            # shorter (never padded with silence), cut at 10 s.
            wav24 = self.up(wav16)
            needed = 6 * 24000
            if wav24.shape[1] < needed:
                repeats = (needed + wav24.shape[1] - 1) // wav24.shape[1]
                wav24 = wav24.repeat(1, repeats)
            wav24 = wav24[:, :10 * 24000]
            prompt_feat = self.mel24(wav24)
            cond16 = self.down(wav24)
            embedding = self.campplus(self.fbank(cond16))
            prompt_token = self.tokenizer(cond16)
            prompt_token = prompt_token[:, :prompt_feat.shape[1] // 2]
            return reference_tokens, speaker_vector, prompt_token, prompt_feat, embedding

    encoder = torch.jit.freeze(torch.jit.script(RefEncoder().eval()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(str(args.out))

    # The saved file, reloaded, on a 4 s clip (tiled) and a 12 s one (cut).
    loaded = torch.jit.load(str(args.out))
    projection = torch.zeros(SPK_PROJ_ELEMENTS, dtype=torch.bfloat16)
    with torch.inference_mode():
        for seconds in (4.0, 12.0):
            out = loaded(noise(seconds, 16000), projection)
            if out[3].shape[1] != 2 * out[2].shape[1] or out[4].shape != (1, 192) or out[1].shape != (2048,):
                raise SystemExit(f'FATAL: unexpected shapes at {seconds} s: {[tuple(t.shape) for t in out]}')
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f'{args.out}: {args.out.stat().st_size:,} bytes, sha256 {digest}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
