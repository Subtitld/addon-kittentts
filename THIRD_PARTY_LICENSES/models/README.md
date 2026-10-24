# KittenTTS 2 weights and voices

Powered by KittenML.

The add-on archive holds none of KittenML's weights. On the first line, the
add-on downloads them on your computer from KittenML's Hugging Face
repository [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2),
at the revision `kittentts_addon/pins.json` names, each file checked by its
sha256:

| File | What it is |
|---|---|
| `cpp/model-tq2_1.gguf` | the KittenTTS 2 language model (1.7B parameters, ternary weights) |
| `cpp/default/decoder.pt` | the S3 decoder, from speech tokens to 24 kHz audio |
| `cpp/default/voices.json` | the 115 built-in voices |
| `config.json`, `LICENSE.md` | the model's settings, and its licence |
| 2 MB of `lm/model-ternary.safetensors` | the speaker head (`spk_proj`), only for cloning |

## The licence: Stellon Labs Community License

KittenML releases these weights under the
[Stellon Labs Community License](STELLON-LABS-COMMUNITY-LICENSE.md), not an
open-source licence. You accept it when you use them. In short, read the full
text before relying on this summary:

- Research, non-commercial and personal use is free.
- Commercial use (including dubbing for pay) is free only while you and your
  affiliates have less than US$1,000,000 in annual revenue and have received
  less than US$1,000,000 in total funding. Above either, the licence ends and
  you need a separate one from Stellon Labs.
- Commercial users must register with Stellon Labs.
- Use must follow Stellon Labs' Acceptable Use Policy, which is "available by
  contacting" Stellon Labs and is not published.
- You may not use the weights or their output to train or distill another model.
- The licence is revocable, and governed by California law.

"This Stellon Labs Model is licensed under the Stellon Labs Community License,
Copyright © Stellon Labs Inc. All Rights Reserved."

Two parts of that repository have their own terms: `speaker/`, a pyannote
speaker model, under MIT, and the `kittenml` Python package, under Apache-2.0.

## The voices

KittenML does not say where the built-in voices' reference recordings come
from. Each voice is a recording of 8 to 18 seconds and its transcript, which
the model continues from.

## Cloning

A cloned voice is made on your computer from the clip Subtitld cuts from the
video. Clone only voices you have the right to use.
