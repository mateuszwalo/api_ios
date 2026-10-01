#!/usr/bin/env bash
# Builds LlamaCpp.xcframework (device + simulator) from a pinned llama.cpp tag.
#
# Differences from llama.cpp's own build-xcframework.sh, and why:
#   * LLAMA_BUILD_COMMON=ON and libcommon.a archived -> we need json_schema_to_grammar()
#     from common/, which the upstream script leaves out.
#   * GGML_METAL_EMBED_LIBRARY=ON -> Metal shaders compiled into the binary, so a sideloaded
#     IPA can never miss ggml-metal.metal at runtime.
#   * iOS deployment target 17.0 (upstream: 16.4).
#
# Layout inside llama.cpp moves between releases (mtmd has lived in two places, headers in
# three), so every path is probed and reported rather than assumed. Whoever reads this log
# after a failure should be able to see what was there instead of guessing.
#
# Usage: scripts/build-llama-xcframework.sh [output-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$(tr -d '[:space:]' < "$ROOT/LLAMA_CPP_TAG")"
SRC="$ROOT/build/llama.cpp"
OUT_DIR="${1:-$ROOT/build}"
XCF="$OUT_DIR/LlamaCpp.xcframework"
IOS_MIN=17.0

echo "==> llama.cpp tag: $TAG"
mkdir -p "$OUT_DIR"

if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi
echo "==> source commit: $(git -C "$SRC" rev-parse HEAD)"

echo "==> top level:"
ls "$SRC"

# --- headers -----------------------------------------------------------------
# The bridge compiles against these. A missing one is reported with the directory listing
# that would answer "where did it go", because that is the question it raises.
stage_headers() {
  local hdr="$OUT_DIR/Headers"
  rm -rf "$hdr"; mkdir -p "$hdr/nlohmann"

  copy_first() {
    local label="$1"; shift
    for candidate in "$@"; do
      if [ -f "$candidate" ]; then
        cp "$candidate" "$hdr/"
        echo "    $label <- ${candidate#$SRC/}"
        return 0
      fi
    done
    echo "    !! $label not found; looked in:" >&2
    for candidate in "$@"; do echo "       ${candidate#$SRC/}" >&2; done
    return 1
  }

  echo "==> staging headers"
  cp "$SRC"/include/llama.h "$hdr/"
  cp "$SRC"/ggml/include/*.h "$hdr/"

  copy_first "mtmd.h"        "$SRC/tools/mtmd/mtmd.h" "$SRC/mtmd/mtmd.h" "$SRC/examples/llava/mtmd.h"
  copy_first "mtmd-helper.h" "$SRC/tools/mtmd/mtmd-helper.h" "$SRC/mtmd/mtmd-helper.h" || true
  copy_first "json-schema-to-grammar.h" "$SRC/common/json-schema-to-grammar.h"
  copy_first "common.h"      "$SRC/common/common.h"
  copy_first "sampling.h"    "$SRC/common/sampling.h" || true

  local json_found=0
  for candidate in "$SRC/vendor/nlohmann/json.hpp" "$SRC/common/json.hpp" "$SRC/examples/server/json.hpp"; do
    if [ -f "$candidate" ]; then
      cp "$candidate" "$hdr/nlohmann/"
      [ -f "${candidate%/*}/json_fwd.hpp" ] && cp "${candidate%/*}/json_fwd.hpp" "$hdr/nlohmann/"
      echo "    nlohmann/json.hpp <- ${candidate#$SRC/}"
      json_found=1
      break
    fi
  done
  if [ "$json_found" -eq 0 ]; then
    echo "    !! nlohmann/json.hpp not found; searching:" >&2
    find "$SRC" -name 'json.hpp' -maxdepth 4 >&2 || true
    exit 1
  fi

  echo "==> staged:"
  ls "$hdr" "$hdr/nlohmann"
}

# --- slices ------------------------------------------------------------------
build_slice() {
  local name="$1" sysroot="$2" archs="$3"
  local dir="$OUT_DIR/$name"
  echo "==> configuring $name ($sysroot, $archs)"
  cmake -S "$SRC" -B "$dir" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sysroot" \
    -DCMAKE_OSX_ARCHITECTURES="$archs" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_BLAS=OFF \
    -DMTMD_VIDEO=OFF

  # Only the libraries that go into the xcframework, never ALL_BUILD.
  #
  # ALL_BUILD also builds llama.cpp's own command-line targets, and those are not meant for
  # an iOS sysroot: `llama-app` includes a generated `build-info.h` that the Xcode generator
  # does not produce, so the build fails on a binary nothing here would ship anyway.
  #
  # `LLAMA_BUILD_TOOLS` stays at its default on purpose: mtmd lives under tools/, and
  # turning tools off takes the whole vision path with it.
  local project
  project=$(find "$dir" -maxdepth 1 -name '*.xcodeproj' | head -n 1)
  echo "==> generated project: ${project:-none found}"
  : > "$dir/targets.txt"
  [ -n "$project" ] && xcodebuild -list -project "$project" > "$dir/targets.txt" 2>&1 || true
  echo "==> targets:"
  cat "$dir/targets.txt"

  # Build the libraries by name rather than ALL_BUILD, and take the names from what was
  # actually generated. Hard-coding them cost a build once already: a guard asserting a
  # target called exactly `mtmd` failed the whole run on a naming assumption.
  local targets=""
  for candidate in llama common mtmd; do
    if grep -qE "^[[:space:]]*${candidate}[[:space:]]*$" "$dir/targets.txt"; then
      targets="$targets $candidate"
    else
      echo "    note: no target named exactly '$candidate'; near matches:"
      grep -i "$candidate" "$dir/targets.txt" | sed 's|^|      |' || echo "      (none)"
    fi
  done
  if [ -z "$targets" ]; then
    echo "!! none of llama, common or mtmd exist as targets; see the list above" >&2
    exit 1
  fi

  echo "==> building $name:$targets"
  # shellcheck disable=SC2086
  cmake --build "$dir" --config Release $(for t in $targets; do printf -- '--target %s ' "$t"; done)

  # One fat static library per slice: an xcframework will not take a pile of .a files.
  local libs
  libs=$(find "$dir" -name '*.a' | sort)
  if [ -z "$libs" ]; then
    echo "!! no static libraries were produced in $dir" >&2
    find "$dir" -maxdepth 3 -type d >&2
    exit 1
  fi
  echo "==> merging:"
  echo "$libs" | sed 's|^|    |'

  # Vision is not optional here: a build without mtmd would install happily and refuse every
  # request carrying an image. Better to fail now, with the target list above in the log.
  if ! echo "$libs" | grep -qi 'mtmd'; then
    echo "!! no mtmd library was produced; vision would be unavailable on the device" >&2
    echo "   targets that were generated:" >&2
    cat "$dir/targets.txt" >&2
    exit 1
  fi
  # shellcheck disable=SC2086
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
