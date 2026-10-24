# KittenTTS 2 add-on for Subtitld

Offline **text-to-speech and voice cloning** for
[Subtitld](https://github.com/Subtitld/subtitld), powered by KittenML's
[KittenTTS 2](https://huggingface.co/KittenML/kitten-tts-2), run by KittenML's
own C++ runtime, [kitten-tts-2-cpp](https://github.com/KittenML/kitten-tts-2-cpp).

Powered by KittenML.

- Serves the `tts.synthesize` task over Subtitld's add-on protocol.
- **The speaker's own voice.** The add-on clones the speaker from the clip
  Subtitld cuts from the video, and speaks the translated line in that voice,
  in another language too.
- **16 languages, 115 built-in voices.** English (50 voices), Spanish, French,
  German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic,
  Hindi, Turkish, Dutch, Polish and Indonesian, at 24 kHz.
- **CPU only, about real time.** On a laptop CPU (i7-13700T), a second of
  speech takes 1.1 to 1.5 seconds to make, with about 2.3 GB of memory. No GPU,
  and no Python packages beyond the standard library.
- **Offline after the first line.** The model, 1.66 GB, downloads on the first
  line, and the reference encoder, 542 MB, on the first cloned line. After
  that nothing leaves the machine: no telemetry, no account.

## The licence of the weights: read this first

The add-on itself is Apache-2.0, and everything in its archive is MIT,
Apache-2.0, BSD or similar. **The KittenTTS 2 weights are not.** The add-on
downloads them from KittenML's repository on your computer, and they are under
the [Stellon Labs Community License](THIRD_PARTY_LICENSES/models/STELLON-LABS-COMMUNITY-LICENSE.md),
which you accept by using them. In short:

- Research, non-commercial and personal use is free.
- Commercial use, dubbing for pay included, is free only while you and your
  affiliates have **less than US$1,000,000 in annual revenue and less than
  US$1,000,000 in total funding**, and you must register with Stellon Labs.
  Above either amount, you need a separate licence from Stellon Labs.
- Use must follow Stellon Labs' Acceptable Use Policy, which is not published
  ("available by contacting us").
- The weights and their output may not be used to train or distill another model.
- The licence is revocable.

This is a summary, not legal advice: read the licence itself before using the
add-on for paid work. Where the terms do not suit you, VoxCPM2 (Apache-2.0)
also clones voices, and Piper, Kokoro and Pocket TTS have permissive built-in
voices.

## Cloning

Pick **Speaker's own voice (cloned from the video)** for a speaker, and turn on
the original voice for them in Subtitld. Subtitld sends a clip of the speaker,
3 to 12 seconds, and the subtitle text spoken in it:

- **With the text**, KittenTTS continues from the clip, which is closest to
  the speaker and steadiest. In a test (two speakers, two Portuguese lines,
  two seeds, English references), the clones scored a speaker similarity of
  0.80, against 0.61 between two different people, with 4% of words wrong.
- **Without it**, or with **Clone the timbre only** on, only the timbre is
  used: a little less close (0.76), and less steady (13% of words wrong,
  mostly one bad line out of eight).

Preparing a voice from a clip takes about 2 seconds, once per clip. Clips
shorter than 6 seconds are repeated to 6 seconds for the decoder, as KittenML's
own code does. Clone only voices you have the right to use.

## Built-in voices

KittenTTS 2 voices carry their language: each is a recording and its
transcript, and the model continues in its accent. So each voice is listed for
its own language, and an English voice reading Portuguese sounds English.
English has 50 voices, among them Bella, Jasper, Luna, Bruno, Rosie, Hugo,
Kiki and Leo (KittenTTS 0.8's), and characters like Marcus (smooth radio DJ)
or Victoria (regal narrator). The other languages have 2 to 7 each, one of
them named after the language (Portuguese, Spanish...), which KittenML
describes as its main voice. KittenML does not say where the voices'
recordings come from.

Now and then the model falls into silence instead of speaking a line. The
add-on notices (the line never ends on its own) and tries again with another
seed, up to three times. In a test of one line in each language with four seeds,
that happened 3 times in 64: twice with the voice named Chinese, which is why
Chinese starts with News CCTV instead (no failure in 18 tries), and once in
Indonesian.

Text in English is normalized first (numbers, dates, currencies,
abbreviations: "the 3rd of May" is read "the third of May"). The normalizer is
English-only, so text in other languages is read as written.

## Options

Set these in **Settings ▸ Add-ons ▸ KittenTTS 2 ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| Delivery | Stable | `KITTENTTS_PRESET` | `stable` or `expressive` (livelier, less predictable). |
| CPU threads | 0 (automatic) | `KITTENTTS_THREADS` | 0 picks the performance cores of a hybrid CPU, else one per core, at most 8. |
| Fallback reference clip | — | `KITTENTTS_VOICE_REF_AUDIO` | For the clone when Subtitld sends no clip. WAV preferred; other formats need ffmpeg. |
| Fallback reference transcript | — | `KITTENTTS_VOICE_REF_TEXT` | The words spoken in the fallback clip. |
| Clone the timbre only | off | `KITTENTTS_SKIP_REF_TEXT` | Leave the clip's words out of the prompt. |

## Requirements

- A 64-bit CPU: x86-64 with AVX2, FMA and F16C (Intel since 2013, AMD since
  2015, but not every Celeron, Pentium or Atom), or Apple Silicon.
- About 3 GB of free memory, and 2.9 GB of disk for the model files.
- Linux (glibc 2.35 or newer), macOS 12 or newer on Apple Silicon, or Windows 10/11.

## How it works

| Piece | What it does | Licence |
|---|---|---|
| `kitten-tts` | KittenML's runtime: the language model on llama.cpp's ggml, then the S3 decoder on LibTorch | MIT (llama.cpp fork), Apache-2.0 (KittenML's code and text normalizer) |
| `kitten-ref` | This add-on's: turns a clip into a voice, with the reference encoder | Apache-2.0 |
| LibTorch | from the PyTorch 2.14.1 CPU wheel, for the decoder and the encoder | BSD-3-Clause |
| `ref_encoder-1.pt` | the S3 tokenizer, CAMPPlus and a pyannote speaker model, as TorchScript (downloaded; see [THIRD_PARTY_LICENSES/ref-encoder](THIRD_PARTY_LICENSES/ref-encoder/README.md)) | MIT |
| KittenTTS 2 weights | downloaded from [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2), pinned by revision and sha256 | Stellon Labs Community License |

The add-on never redistributes KittenML's weights. Cloning needs the 2 MB
speaker head of the language model (`spk_proj`), so the add-on downloads just
that byte range of `lm/model-ternary.safetensors` instead of the whole 954 MB file.

## Building

```
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -e '.[build,dev]'
python tools/build_native.py        # needs git, cmake and a C++20 compiler
pytest
pyinstaller pyinstaller.spec --noconfirm
```

`tools/pin_models.py` pins a new KittenTTS 2 revision, `tools/regen_voices.py`
regenerates the voice list from it, and `tools/export_ref_encoder.py`
re-exports the reference encoder (it needs the `kittenml` package).
`.github/workflows/release.yml` builds and tests the Linux, macOS and Windows
archives and publishes them on a `v*` tag.

## Licence

The add-on's own code is under the Apache License 2.0 (`LICENSE`). Each
release archive lists every bundled component and its licence in
`THIRD_PARTY_LICENSES/README.md`.
