#!/bin/bash
# =============================================================================
# Build the Optimus port (third_party/Optimus) and the patched Cosserat plugin
# into a running container's /opt/sofa. Runs INSIDE the container:
#
#     bash /ros2_ws/scripts/install_optimus.sh [--force] [cosserat|optimus]
#
# Idempotent: each step is skipped when its product is already installed.
#
# * Cosserat is rebuilt FROM SOURCE at the exact commit of the shipped v25.12
#   release (f64e029, see git-info.txt of the binary) with the internal-data
#   tracker patch of third_party/cosserat-patches: without it
#   BeamHookeLawForceField keeps its stiffness cache (m_K_section) when an
#   estimator changes EI/GI through a Data link, so every sigma point would run
#   with the same stiffness and the parameter gain would be identically zero.
#   Marker: the installed header declares doUpdateInternal().
# * Optimus (upstream cd41bf17, minimal core) is built against /opt/sofa and
#   installed to /opt/sofa/plugins/Optimus. Marker: lib/libOptimus.so.
#
# Memory: the host has ~7 GB, so both builds run with make -j2.
# =============================================================================

set -euo pipefail

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa}"
THIRD_PARTY="${THIRD_PARTY:-/ros2_ws/third_party}"
BUILD_ROOT="${BUILD_ROOT:-/opt/build}"
JOBS="${JOBS:-2}"

COSSERAT_REPO="${COSSERAT_REPO:-https://github.com/SofaDefrost/Cosserat.git}"
COSSERAT_SHA="${COSSERAT_SHA:-f64e029de61d320700bcb5f0ececc4efc0d8b1d9}"
COSSERAT_PREFIX="$SOFA_ROOT/plugins/Cosserat"
OPTIMUS_PREFIX="$SOFA_ROOT/plugins/Optimus"

FORCE=0
WHAT="all"
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    cosserat|optimus|all) WHAT="$arg" ;;
    *) echo "usage: $0 [--force] [cosserat|optimus|all]" >&2; exit 2 ;;
  esac
done

log() { echo "[install_optimus] $*"; }

install_cosserat() {
  if [ "$FORCE" = 0 ] && grep -q doUpdateInternal \
      "$COSSERAT_PREFIX/include/Cosserat/Cosserat/forcefield/BeamHookeLawForceField.h" 2>/dev/null; then
    log "patched Cosserat already installed at $COSSERAT_PREFIX"
    return
  fi

  local src="$BUILD_ROOT/cosserat-src" build="$BUILD_ROOT/cosserat"
  # Commit-pinned archive (git-over-https from the container is flaky against GitHub).
  rm -rf "$src" && mkdir -p "$src"
  log "fetching Cosserat @ $COSSERAT_SHA"
  curl -fsSL "${COSSERAT_REPO%.git}/archive/$COSSERAT_SHA.tar.gz" \
    | tar -xz --strip-components=1 -C "$src"
  for p in "$THIRD_PARTY"/cosserat-patches/*.patch; do
    log "applying $(basename "$p")"; patch -d "$src" -p1 --forward <"$p" >/dev/null
  done

  log "building Cosserat (make -j$JOBS, several minutes)"
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$SOFA_ROOT" -DCMAKE_INSTALL_PREFIX="$COSSERAT_PREFIX" \
    -DSOFA_BUILD_TESTS=OFF -DCOSSERAT_BUILD_TESTS=OFF \
    -DSofaPython3Tools="" -DCMAKE_DISABLE_FIND_PACKAGE_SofaPython3=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_SoftRobots=TRUE -DCMAKE_DISABLE_FIND_PACKAGE_STLIB=TRUE \
    -DCMAKE_INSTALL_RPATH='$ORIGIN;$ORIGIN/../../../lib' >"$build.cmake.log" 2>&1 \
    || { tail -30 "$build.cmake.log"; exit 1; }
  make -C "$build" -j"$JOBS" >"$build.make.log" 2>&1 || { tail -40 "$build.make.log"; exit 1; }
  # The shipped 21.12-soname binary and its dev-suffix shims are superseded by this build.
  rm -f "$COSSERAT_PREFIX"/lib/libCosserat.so* "$COSSERAT_PREFIX"/lib/libSofa*.so.25.12.99
  make -C "$build" install >/dev/null
  # SOFA's install macros force RUNPATH=$ORIGIN/../lib (= this very directory), so the
  # release layout resolves its SOFA dependencies through symlinks placed next to the plugin.
  for f in "$SOFA_ROOT"/lib/libSofa*.so.25.12.00; do ln -sf "$f" "$COSSERAT_PREFIX/lib/$(basename "$f")"; done
  printf 'source: %s @ %s\npatches: %s\n' "$COSSERAT_REPO" "$COSSERAT_SHA" \
    "$(ls "$THIRD_PARTY"/cosserat-patches/)" > "$COSSERAT_PREFIX/git-info.txt"
  log "patched Cosserat installed at $COSSERAT_PREFIX"
}

install_optimus() {
  if [ "$FORCE" = 0 ] && [ -f "$OPTIMUS_PREFIX/lib/libOptimus.so" ]; then
    log "Optimus already installed at $OPTIMUS_PREFIX"
    return
  fi
  local build="$BUILD_ROOT/optimus"
  log "building Optimus (make -j$JOBS)"
  cmake -S "$THIRD_PARTY/Optimus" -B "$build" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_PREFIX_PATH="$SOFA_ROOT" -DCMAKE_INSTALL_PREFIX="$OPTIMUS_PREFIX" \
    -DSTOCHASTIC_FILTERING=ON -DUSE_BLAS_FOR_OPTIMUS=OFF >"$build.cmake.log" 2>&1 \
    || { tail -30 "$build.cmake.log"; exit 1; }
  make -C "$build" -j"$JOBS" >"$build.make.log" 2>&1 || { grep -E "error" "$build.make.log" | head -40; exit 1; }
  make -C "$build" install >/dev/null
  log "Optimus installed at $OPTIMUS_PREFIX"
}

mkdir -p "$BUILD_ROOT"
case "$WHAT" in
  cosserat) install_cosserat ;;
  optimus) install_optimus ;;
  all) install_cosserat; install_optimus ;;
esac
