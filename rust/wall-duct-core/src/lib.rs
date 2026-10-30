//! Differentiable HVAC duct-noise core.
//!
//! Began as a port of the Python PoC in `duct_poc/`; its elements now follow
//! ASHRAE's published methods (`examples/reference.rs`). Forward-mode autodiff via
//! `Dual<N>` (see [`autodiff`]), propagating a joint sensitivity vector
//! through source PWL, per-block insertion/regeneration, end reflection,
//! and room reception.
//!
//! Conventions:
//! - Frequencies: 8 octave bands (63 Hz .. 8 kHz) — see [`N_BANDS`].
//! - Lengths in metres, velocities in m/s, flows in m^3/s.
//! - All per-band spectra are stored as `[Dual<N>; N_BANDS]`, one sensitivity
//!   vector per band.
//!
//! The dual width `N` is chosen at the top level. For the first slice we run
//! a fixed 3-block chain and size `N` to exactly the number of design
//! parameters we're surfacing.

pub mod autodiff;
pub mod metrics;
pub mod physics;
pub mod model;

pub use metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};
