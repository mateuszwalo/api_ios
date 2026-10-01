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

  # Whole directories, not a hand-picked list.
  #
  # Picking individual files cost a build: json-schema-to-grammar.h includes json-schema.h,
  # which was not on the list, and the failure surfaced two steps later as a missing header
  # during the Swift build. These headers include one another freely and the set changes
  # between releases, so copying the directories is both shorter and the only version that
  # stays correct across a tag bump.
  echo "==> staging headers"
  cp "$SRC"/include/*.h "$hdr/"
  cp "$SRC"/ggml/include/*.h "$hdr/"
  cp "$SRC"/common/*.h "$hdr/" 2>/dev/null || true
  cp "$SRC"/common/*.hpp "$hdr/" 2>/dev/null || true

  local mtmd_dir=""
  for candidate in "$SRC/tools/mtmd" "$SRC/mtmd" "$SRC/examples/llava"; do
    if [ -f "$candidate/mtmd.h" ]; then mtmd_dir="$candidate"; break; fi
  done
  if [ -z "$mtmd_dir" ]; then
    echo "!! mtmd.h not found anywhere; vision cannot be built" >&2
    find "$SRC" -name 'mtmd*.h' | head -n 20 >&2
    exit 1
  fi
  cp "$mtmd_dir"/*.h "$hdr/"
  echo "    mtmd headers <- ${mtmd_dir#$SRC/}"

  local json_dir=""
  for candidate in "$SRC/vendor/nlohmann" "$SRC/common" "$SRC/examples/server"; do
    if [ -f "$candidate/json.hpp" ]; then json_dir="$candidate"; break; fi
  done
  if [ -z "$json_dir" ]; then
    echo "!! nlohmann/json.hpp not found:" >&2
    find "$SRC" -name 'json.hpp' | head -n 20 >&2
    exit 1
  fi
  cp "$json_dir"/json*.hpp "$hdr/nlohmann/"
  echo "    nlohmann <- ${json_dir#$SRC/}"

  # Some headers reach for <nlohmann/json.hpp> and some for "json.hpp"; satisfy both.
  cp "$json_dir"/json*.hpp "$hdr/" 2>/dev/null || true

  echo "==> staged:"
  ls "$hdr"
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
  for candidate in llama mtmd; do
    if grep -qE "^[[:space:]]*${candidate}[[:space:]]*$" "$dir/targets.txt"; then
      targets="$targets $candidate"
    else
      echo "    note: no target named exactly '$candidate'; near matches:"
      grep -i "$candidate" "$dir/targets.txt" | sed 's|^|      |' || echo "      (none)"
    fi
  done

  # The library holding json_schema_to_grammar has been called both `common` and
  # `llama-common`. Guessing `common` cost a build: the target was silently skipped and the
  # miss only surfaced as undefined symbols at link time, two steps later.
  local common_target=""
  for alias in llama-common common; do
    if grep -qE "^[[:space:]]*${alias}[[:space:]]*$" "$dir/targets.txt"; then
      common_target="$alias"; break
    fi
  done
  if [ -z "$common_target" ]; then
    echo "!! no common library target found; json_schema_to_grammar would be missing" >&2
    cat "$dir/targets.txt" >&2
    exit 1
  fi
  targets="$targets $common_target"
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

  # Check the symbols this project actually calls, here rather than at link time.
  #
  # A missing library does not announce itself: the merge succeeds, the xcframework is
  # produced, and the absence turns up minutes later as undefined symbols in a log that
  # points at the caller rather than the cause. These three cover the parts of llama.cpp
  # that nothing else would reveal until the device refused a request.
  echo "==> checking symbols"
  local symbols
  symbols=$(nm -gU "$dir/libllamacpp.a" 2>/dev/null || true)
  local missing=""
  for symbol in json_schema_to_grammar llama_sampler_init_grammar mtmd_tokenize llama_model_chat_template; do
    if echo "$symbols" | grep -q "$symbol"; then
      echo "    ok: $symbol"
    else
      echo "    MISSING: $symbol"
      missing="$missing $symbol"
    fi
  done
  if [ -n "$missing" ]; then
    echo "!! the merged library is missing:$missing" >&2
    echo "   built targets:$targets" >&2
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
