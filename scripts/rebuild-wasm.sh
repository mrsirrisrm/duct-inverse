#!/usr/bin/env bash
# Rebuild the wall-duct-wasm pkg and copy it into the web app's wasm/ dir.
# Run this whenever Rust sources under rust/ change and you want the UI to
# pick up the updated physics.
set -euo pipefail

cd "$(dirname "$0")/.."

(cd rust/wall-duct-wasm && wasm-pack build --target web --release)

rm -rf wasm
mkdir -p wasm
cp rust/wall-duct-wasm/pkg/* wasm/

echo "Rebuilt wasm/ from rust/wall-duct-wasm"
