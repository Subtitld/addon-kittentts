# kitten-ref, the clone-reference prep tool (see kitten_ref.cpp).
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# Built by tools/build_native.py against the same LibTorch as kitten-tts, with
# nlohmann/json from the kitten-tts-2-cpp checkout.

cmake_minimum_required(VERSION 3.21)
project(kitten_ref CXX)

set(CMAKE_CXX_STANDARD 20)  # LibTorch 2.14 headers need C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(KITTEN_TTS_SOURCE "" CACHE PATH "kitten-tts-2-cpp checkout (for vendor/nlohmann)")
if(NOT KITTEN_TTS_SOURCE)
    message(FATAL_ERROR "set KITTEN_TTS_SOURCE")
endif()

find_package(Torch REQUIRED)
add_executable(kitten-ref kitten_ref.cpp)
target_include_directories(kitten-ref PRIVATE "${KITTEN_TTS_SOURCE}/vendor/nlohmann")
target_link_libraries(kitten-ref PRIVATE ${TORCH_LIBRARIES})
install(TARGETS kitten-ref RUNTIME DESTINATION .)
