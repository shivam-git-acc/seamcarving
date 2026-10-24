#!/usr/bin/env bash
# Downloads the two single-header image libraries (stb, public domain / MIT) used for image I/O.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in stb_image.h stb_image_write.h; do
  [ -f "$f" ] || curl -fsSL -o "$f" "https://raw.githubusercontent.com/nothings/stb/master/$f"
done
echo "stb headers ready"
