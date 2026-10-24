// kitten-ref: turn a clone reference clip into a voice for kitten-tts.
//
// Copyright 2026 Subtitld contributors
// SPDX-License-Identifier: Apache-2.0
//
// The kitten-tts runtime reads a voice as a JSON entry: the reference clip's
// S3 tokens and transcript (the in-context prompt), a speaker vector, and the
// s3gen decoder's conditioning. Upstream prepares that entry in Python
// (tools/kitten-tts/prepare_reference.py). This does the same with LibTorch
// alone, running ref_encoder.pt (tools/export_ref_encoder.py), so the add-on
// ships no Python torch.
//
//   kitten-ref --encoder ref_encoder.pt --spk-proj spk_proj.bin
//              --input clip.s16 --output voice.json [--threads N]
//
// --input is raw signed 16-bit little-endian mono PCM at 16 kHz. --spk-proj
// is the KittenTTS 2 speaker head (spk_proj.0.bias, .0.weight, .1.bias,
// .1.weight, bfloat16, back to back) as it sits in lm/model-ternary.safetensors.
// The output holds every field of a voice but the transcript, which the
// add-on adds.

#include <ATen/Parallel.h>
#include <torch/script.h>
#include <torch/version.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

namespace {

constexpr int64_t kSpkProjElements = 2048 + 2048 * 512 + 2048 + 2048;
constexpr int64_t kSampleRate = 16000;

std::vector<char> read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

json tensor_to_json(const torch::Tensor & tensor) {
    auto t = tensor.contiguous().cpu();
    if (t.dim() == 0) {
        return t.is_floating_point() ? json(t.item<double>()) : json(t.item<int64_t>());
    }
    json out = json::array();
    for (int64_t i = 0; i < t.size(0); ++i) out.push_back(tensor_to_json(t[i]));
    return out;
}

json flat_json(const torch::Tensor & tensor, bool integral) {
    auto t = tensor.contiguous().cpu().flatten();
    json out = json::array();
    if (integral) {
        auto v = t.to(torch::kLong);
        auto p = v.data_ptr<int64_t>();
        for (int64_t i = 0; i < v.numel(); ++i) out.push_back(p[i]);
    } else {
        auto v = t.to(torch::kFloat);
        auto p = v.data_ptr<float>();
        for (int64_t i = 0; i < v.numel(); ++i) out.push_back(p[i]);
    }
    return out;
}

// Rows of a (1, ...) tensor, as nested arrays, the shape the runtime reads.
json nested(const torch::Tensor & tensor, bool integral) {
    auto t = integral ? tensor.to(torch::kLong) : tensor.to(torch::kFloat);
    return tensor_to_json(t);
}

int usage() {
    std::cerr << "usage: kitten-ref --encoder FILE --spk-proj FILE --input FILE.s16 --output FILE.json"
                 " [--threads N]\n";
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") return usage();
        if (key == "--version") {
            std::cout << "kitten-ref 1 (libtorch " << TORCH_VERSION << ")\n";
            return 0;
        }
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) return usage();
        args[key] = argv[++i];
    }
    for (const char * required : {"--encoder", "--spk-proj", "--input", "--output"}) {
        if (!args.count(required)) return usage();
    }
    try {
        if (args.count("--threads")) {
            int threads = std::stoi(args["--threads"]);
            if (threads > 0) {
                at::set_num_threads(threads);
                at::set_num_interop_threads(1);
            }
        }
        c10::InferenceMode guard;

        auto pcm = read_file(args["--input"]);
        int64_t samples = static_cast<int64_t>(pcm.size() / 2);
        if (samples < kSampleRate / 2) throw std::runtime_error("the reference clip is shorter than half a second");
        auto wav = torch::from_blob(pcm.data(), {1, samples}, torch::kInt16).to(torch::kFloat).div(32768.0);

        auto proj = read_file(args["--spk-proj"]);
        if (static_cast<int64_t>(proj.size()) != kSpkProjElements * 2) {
            throw std::runtime_error("spk_proj has " + std::to_string(proj.size()) + " bytes, expected "
                                     + std::to_string(kSpkProjElements * 2));
        }
        auto spk_proj = torch::from_blob(proj.data(), {kSpkProjElements}, torch::kBFloat16);

        auto encoder = torch::jit::load(args["--encoder"], torch::kCPU);
        encoder.eval();
        auto outputs = encoder.forward({wav, spk_proj}).toTuple()->elements();
        if (outputs.size() != 5) throw std::runtime_error("ref_encoder.pt returned an unexpected result");

        json voice;
        voice["reference_tokens"] = flat_json(outputs[0].toTensor(), true);
        voice["speaker"] = flat_json(outputs[1].toTensor(), false);
        voice["prompt_token"] = nested(outputs[2].toTensor(), true);
        voice["prompt_feat"] = nested(outputs[3].toTensor(), false);
        voice["embedding"] = nested(outputs[4].toTensor(), false);

        std::string out = args["--output"];
        std::string scratch = out + ".tmp";
        {
            std::ofstream file(scratch, std::ios::binary);
            if (!file) throw std::runtime_error("cannot write " + scratch);
            file << voice.dump();
            if (!file) throw std::runtime_error("cannot write " + scratch);
        }
        std::remove(out.c_str());
        if (std::rename(scratch.c_str(), out.c_str()) != 0) throw std::runtime_error("cannot write " + out);
        std::cerr << "kitten-ref: " << samples / static_cast<double>(kSampleRate) << " s clip, "
                  << outputs[0].toTensor().numel() << " reference tokens\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "kitten-ref: " << e.what() << "\n";
        return 1;
    }
}
