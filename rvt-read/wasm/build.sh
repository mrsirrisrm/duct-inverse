#!/usr/bin/env bash
# Build the browser module into web/pkg.
#
# Needs: rustup target add wasm32-unknown-unknown
#        cargo install wasm-bindgen-cli --version 0.2.118 --locked
# SIMD is on: rvt-rs's decompression runs about a third faster with it, and
# every current browser supports it.
set -euo pipefail
cd "$(dirname "$0")"
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --lib --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/duct_pipeline.wasm --target web --out-dir web/pkg
ls -l web/pkg/duct_pipeline_bg.wasm
