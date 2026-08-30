#!/bin/bash
# =============================================================================
# Install the Cosserat plugin into a running container's /opt/sofa.
#
# Runs INSIDE the container. Idempotent: exits immediately if the plugin is
# already loadable, so it is safe to call before every cable command.
#
# This mirrors the Dockerfile stage exactly (same pinned release, same
# checksum, same soname shims). It exists because `docker compose down`
# destroys the container overlay and the image has not always been rebuilt;
# without it the cable commands fail with "Plugin Cosserat not found" and the
# only recovery was a 7 GB image rebuild.
# =============================================================================

set -euo pipefail

SOFA_ROOT="${SOFA_ROOT:-/opt/sofa}"
PLUGIN_DIR="$SOFA_ROOT/plugins/Cosserat"
COSSERAT_URL="${COSSERAT_URL:-https://github.com/SofaDefrost/Cosserat/releases/download/release-v25.12/Cosserat_v25.12_for-SOFA-v25.12_Linux.zip}"
COSSERAT_SHA256="${COSSERAT_SHA256:-e6d9476e6543f0cff7a0819ddbbbf1b51401cee1f4147346d89a9ab02f19bb8c}"

if [ -f "$PLUGIN_DIR/lib/libCosserat.so" ]; then
  echo "[cosserat] already installed at $PLUGIN_DIR"
  exit 0
fi

echo "[cosserat] not present, installing the pinned v25.12 release..."
curl -fL -o /tmp/cosserat.zip "$COSSERAT_URL"
echo "${COSSERAT_SHA256}  /tmp/cosserat.zip" | sha256sum -c -

rm -rf /tmp/cosserat-extract
mkdir -p /tmp/cosserat-extract
unzip -q /tmp/cosserat.zip -d /tmp/cosserat-extract
SRC="$(find /tmp/cosserat-extract -mindepth 1 -maxdepth 1 -type d | head -n1)"
mkdir -p "$PLUGIN_DIR"
cp -a "$SRC"/. "$PLUGIN_DIR"/
rm -rf /tmp/cosserat.zip /tmp/cosserat-extract

# The release binary links against dev-suffix sonames (.so.25.12.99) while the
# SOFA release ships .so.25.12.00.
cd "$PLUGIN_DIR/lib"
for f in "$SOFA_ROOT"/lib/libSofa*.so.25.12.00; do
  b="$(basename "$f")"
  ln -sf "$f" "${b%25.12.00}25.12.99"
done

echo "[cosserat] installed at $PLUGIN_DIR"
