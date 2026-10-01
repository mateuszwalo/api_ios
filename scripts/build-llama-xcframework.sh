#!/usr/bin/env bash
# Builds LlamaCpp.xcframework (device + simulator) from a pinned llama.cpp tag.
#
# Differences from llama.cpp's own build-xcframework.sh, and why:
#   * LLAMA_BUILD_COMMON=ON + libcommon.a archived  -> we need
#     json_schema_to_grammar() from common/, which the upstream script omits.
#   * GGML_METAL_EMBED_LIBRARY=ON -> Metal shaders compiled into the binary, so a
#     sideloaded IPA can never miss ggml-metal.metal at runtime.
#   * iOS deployment target 17.0 (upstream: 16.4).
#
# Usage: scripts/build-llama-xcframework.sh [output-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$(tr -d '[:space:]' < "$ROOT/LLAMA_CPP_TAG")"
SRC="$ROOT/build/llama.cpp"
OUT_DIR="${1:-$ROOT/build}"
XCF="$OUT_DIR/LlamaCpp.xcframework"

echo "==> llama.cpp tag: $TAG"

if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi
echo "==> source commit: $(git -C "$SRC" rev-parse HEAD)"

# Headers consumed by Sources/LlamaBridge (Objective-C++).
stage_headers() {
  local hdr="$OUT_DIR/Headers"
  rm -rf "$hdr"; mkdir -p "$hdr/nlohmann"
  cp "$SRC"/include/llama.h                       "$hdr/"
  cp "$SRC"/ggml/include/*.h                      "$hdr/"
  cp "$SRC"/tools/mtmd/mtmd.h                     "$hdr/" 2>/dev/null || \
    cp "$SRC"/mtmd/mtmd.h                         "$hdr/"
  cp "$SRC"/tools/mtmd/mtmd-helper.h              "$hdr/" 2>/dev/null || true
  cp "$SRC"/common/json-schema-to-grammar.h       "$hdr/"
  cp "$SRC"/common/common.h                       "$hdr/"
  cp "$SRC"/common/sampling.h                     "$hdr/" 2>/dev/null || true
  cp "$SRC"/vendor/nlohmann/json.hpp              "$hdr/nlohmann/"
  cp "$SRC"/vendor/nlohmann/json_fwd.hpp          "$hdr/nlohmann/"
  echo "==> headers staged in $hdr"
}

build_slice() {
  local name="$1" sysroot="$2" archs="$3"
  local dir="$OUT_DIR/$name"
  echo "==> configuring $name ($sysroot, $archs)"
  cmake -S "$SRC" -B "$dir" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sysroot" \
    -DCMAKE_OSX_ARCHITECTURES="$archs" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_METAL_TARGET_OS=ios \
    -DGGML_OPENMP=OFF \
    -DGGML_BLAS=OFF \
    -DMTMD_VIDEO=OFF
  cmake --build "$dir" --config Release -- -quiet
  # One fat static library per slice; xcframework does not accept a pile of .a files.
  local libs; libs=$(find "$dir" -name '*.a' -path '*Release*' | sort)
  echo "==> merging: $(echo "$libs" | xargs -n1 basename | tr '\n' ' ')"
  libtool -static -o "$dir/libllamacpp.a" $libs
}

stage_headers
build_slice device    iphoneos        "arm64"
build_slice simulator iphonesimulator "arm64;x86_64"

rm -rf "$XCF"
xcodebuild -create-xcframework \
  -library "$OUT_DIR/device/libllamacpp.a"    -headers "$OUT_DIR/Headers" \
  -library "$OUT_DIR/simulator/libllamacpp.a" -headers "$OUT_DIR/Headers" \
  -output "$XCF"

echo "$TAG $(git -C "$SRC" rev-parse HEAD)" > "$XCF/BUILD_INFO"
echo "==> done: $XCF"
