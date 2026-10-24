# The reference encoder (ref_encoder-1.pt)

Downloaded on the first cloned line, from this add-on's own release
[ref-encoder-1](https://github.com/Subtitld/addon-kittentts/releases/tag/ref-encoder-1),
checked by its sha256. `tools/export_ref_encoder.py` made it. It is a TorchScript
file that turns a clip into a KittenTTS 2 voice, and holds:

| Part | From | Licence |
|---|---|---|
| S3 speech tokenizer, mel extractor and CAMPPlus speaker encoder (weights) | [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) `s3gen_meanflow.safetensors`, revision 749d1c1a | MIT, [chatterbox-LICENSE](chatterbox-LICENSE) |
| The tokenizer's design and original weights | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Apache-2.0, [CosyVoice-LICENSE](CosyVoice-LICENSE) |
| Speaker model (weights) | [KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2) `speaker/`, derived from pyannote | MIT, [pyannote-speaker-LICENSE](pyannote-speaker-LICENSE) |
| Code traced into the file | [kittenml](https://github.com/KittenML/KittenTTS) 0.9.6 and [s3tokenizer](https://github.com/xingchensong/S3Tokenizer) 0.3.0 | Apache-2.0, [kittenml-LICENSE](kittenml-LICENSE), [s3tokenizer-LICENSE](s3tokenizer-LICENSE) |

Changed from the originals: the pieces were traced to TorchScript, with
torchaudio's Kaldi filterbank rewritten in plain tensor operations so the trace
works at any clip length. The weights are unchanged.

Not in the file: the KittenTTS 2 speaker head (`spk_proj`), which is under the
Stellon Labs Community License. The add-on downloads it from KittenML's
repository and hands it to the encoder at run time.
