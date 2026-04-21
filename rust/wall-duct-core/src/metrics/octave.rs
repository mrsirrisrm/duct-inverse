//! Octave band centres used throughout the duct-noise PoC.
//!
//! NC is defined on 8 bands (63 Hz .. 8 kHz). Every spectrum in this crate
//! — source PWL, per-block losses/additions, receiver SPL — is an 8-long
//! array in this exact order.

pub const N_BANDS: usize = 8;

pub const OCTAVE_BANDS_HZ: [f64; N_BANDS] = [
    63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
];
