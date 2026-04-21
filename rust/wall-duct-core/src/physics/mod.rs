//! Duct-noise physics primitives, generic over the dual-number width `N`.
//!
//! Every function works on `Dual<N>` scalars and returns per-band spectra
//! as `[Dual<N>; N_BANDS]`. Spectra are combined with [`db_add`] (logarithmic
//! energy addition) to mix insertion losses with regenerated PWL.

pub mod source;
pub mod duct_attenuation;
pub mod attenuator;
pub mod end_reflection;
pub mod regenerative;
pub mod room;

use crate::autodiff::Dual;

use crate::metrics::octave::N_BANDS;

/// Logarithmic (energy) addition of two dB spectra, per band.
///     10 log10( 10^{a/10} + 10^{b/10} )
pub fn db_add<const N: usize>(
    a: &[Dual<N>; N_BANDS],
    b: &[Dual<N>; N_BANDS],
) -> [Dual<N>; N_BANDS] {
    let ten = Dual::<N>::constant(10.0);
    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        let ea = (a[i] / ten) * Dual::<N>::constant(std::f64::consts::LN_10);
        let eb = (b[i] / ten) * Dual::<N>::constant(std::f64::consts::LN_10);
        let s = ea.exp() + eb.exp();
        out[i] = s.log10() * ten;
    }
    out
}

/// Subtract a per-band loss spectrum from a per-band PWL (both dB).
pub fn sub_bands<const N: usize>(
    a: &[Dual<N>; N_BANDS],
    b: &[Dual<N>; N_BANDS],
) -> [Dual<N>; N_BANDS] {
    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        out[i] = a[i] - b[i];
    }
    out
}
