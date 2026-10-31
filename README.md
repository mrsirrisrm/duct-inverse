# duct-inverse — differentiable HVAC duct-noise explorer

A browser-side demo for exploring HVAC supply-path noise. You build a chain
of blocks (duct segments, bends, splitter attenuators, diffusers), the tool
runs a pure-Rust acoustic model compiled to WebAssembly, and it reports the
NC rating at the receiver, the octave bands pinning that rating, and the
design parameters most sensitive to change. Optionally, paste a Gemini API
key and the tool will turn those numbers into written design advice.

> ⚠️ **DEMO ONLY, FOR INDICATIVE PURPOSES — NUMBERS ARE NOT ACCURATE.**
> The physics uses published empirical fits with ASHRAE-shaped coefficients,
> not a consultancy database. Results are the right *shape* (regenerative
> noise, lining attenuation, end reflection, MAM resonance etc.) but should
> not be treated as engineering numbers.

## Quickstart

```
npm install
npm run dev
```

## What's in the repo

- `src/` — Solid + Vite + uPlot web app. Sliders + a dynamic chain editor on
  the left, NC chart + tables + advisor on the right.
- `wasm/` — the wasm-pack output (committed so Vercel can build without a
  Rust toolchain). Regenerate with `./scripts/rebuild-wasm.sh`.
- `rust/wall-duct-core/` — pure Rust physics + forward-mode autodiff
  (`Dual<N>`). Its elements follow published methods (ASHRAE duct, elbow and
  end reflection losses, Reynolds & Bevirt lined duct, TM 5-805-4 lined
  elbows and attenuators, Schultz's room equation, Woods' fitting regen, CIBSE
  diffuser noise); `cargo run --example reference` sets them beside their
  references, and `cargo run --example tm_classroom` runs TM 5-805-4's worked
  supply path (§7-6) element by element, which `tests/tm_classroom.rs` holds
  the engine to.
- `rust/wall-duct-wasm/` — `#[wasm_bindgen]` wrapper exposing a single
  `evaluate(params)` entry point returning a `DesignFacts` JSON object.
- `duct_poc/` — the original Python / PyTorch PoC, which the Rust core began
  as a port of. Its physics are the older placeholders.
- `scripts/rebuild-wasm.sh` — rebuild wasm and refresh `wasm/`.

## Deploying to Vercel

Vercel auto-detects Vite. Just point it at this repo and accept the
defaults. The committed `wasm/` dir means no Rust toolchain is needed on
the build host. `vercel.json` pins the cache headers on `.wasm` assets.

## Extracted from

This repo was split out from
[wall-insulation-inverse](https://github.com/mrsirrisrm/wall-insulation-inverse)
at revision `fd353e8`. The partition-transmission-loss half of that project
still lives there.
