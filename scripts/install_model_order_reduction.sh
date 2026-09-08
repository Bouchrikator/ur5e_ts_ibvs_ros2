#!/bin/bash
# Run INSIDE the container. Pinned official mapping/POD, without ECSW or contact.
# Usage: bash /ros2_ws/scripts/install_model_order_reduction.sh [--force]
set -euo pipefail

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa}"
THIRD_PARTY="${THIRD_PARTY:-/ros2_ws/third_party}"
BUILD_ROOT="${BUILD_ROOT:-/opt/build}"
JOBS="${JOBS:-2}"
MOR_SHA=d94dc49dff66d936ad11c33a8f195cc167c98b61
MOR_PREFIX="$SOFA_ROOT/plugins/ModelOrderReduction"
PATCH_DIR="$THIRD_PARTY/model-order-reduction-patches"
FORCE=0
if [ "${1:-}" = --force ] && [ "$#" = 1 ]; then
  FORCE=1
elif [ "$#" != 0 ]; then
  echo "usage: $0 [--force]" >&2; exit 2
fi

log() { echo "[install_model_order_reduction] $*"; }
# A changed patch invalidates the marker, unlike checking only for a .so file.
PATCH_HASH=$(cat "$PATCH_DIR"/*.patch | sha256sum | cut -d' ' -f1)
MARKER="$MOR_SHA $PATCH_HASH"
if [ "$FORCE" = 0 ] && [ -f "$MOR_PREFIX/lib/libModelOrderReduction.so" ] &&
    [ -f "$MOR_PREFIX/lib/python3/site-packages/mor/reduction/script/ReadStateFilesAndComputeModes.py" ] &&
    [ "$(cat "$MOR_PREFIX/.cable-mor-build" 2>/dev/null || true)" = "$MARKER" ]; then
  log "pinned mapping/POD already installed at $MOR_PREFIX"
  exit 0
fi

mkdir -p "$BUILD_ROOT"
MOR_SRC="$BUILD_ROOT/model-order-reduction-src"
MOR_BUILD="$BUILD_ROOT/model-order-reduction"
# Each source tree belongs exclusively to this installer; keep build outputs
# for incremental --force builds and refresh the pinned source before patching.
mkdir -p "$MOR_SRC"
log "fetching ModelOrderReduction @ $MOR_SHA"
curl -fsSL "https://github.com/SofaDefrost/ModelOrderReduction/archive/$MOR_SHA.tar.gz" \
  | tar -xz --strip-components=1 -C "$MOR_SRC"
for patch_file in "$PATCH_DIR"/*.patch; do
  log "applying $(basename "$patch_file")"
  patch -d "$MOR_SRC" -p1 --forward <"$patch_file" >/dev/null
done

log "building coordinate mapping (make -j$JOBS)"
cmake -S "$MOR_SRC" -B "$MOR_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$SOFA_ROOT" -DCMAKE_INSTALL_PREFIX="$MOR_PREFIX" \
  -DMOR_MAPPING_ONLY=ON -DSOFA_BUILD_TESTS=OFF -DMODELORDERREDUCTION_BUILD_TESTS=OFF \
  >"$MOR_BUILD.cmake.log" 2>&1 \
  || { tail -40 "$MOR_BUILD.cmake.log"; exit 1; }
cmake --build "$MOR_BUILD" --parallel "$JOBS" >"$MOR_BUILD.make.log" 2>&1 \
  || { tail -60 "$MOR_BUILD.make.log"; exit 1; }
cmake --install "$MOR_BUILD" >"$MOR_BUILD.install.log" 2>&1 \
  || { tail -40 "$MOR_BUILD.install.log"; exit 1; }
printf '%s\n' "$MARKER" >"$MOR_PREFIX/.cable-mor-build"
printf 'source: https://github.com/SofaDefrost/ModelOrderReduction @ %s\npatches_sha256: %s\n' \
  "$MOR_SHA" "$PATCH_HASH" >"$MOR_PREFIX/git-info.txt"
log "installed pinned mapping/POD at $MOR_PREFIX"
