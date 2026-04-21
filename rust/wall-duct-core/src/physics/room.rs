//! Room reception: diffuser PWL → receiver SPL.
//!
//! Direct-plus-reverberant model:
//! ```text
//!     SPL = PWL + 10 log10( Q / (4 pi r^2) + 4 / R )
//! ```
//! with `R ≈ 0.161 V / RT60` (Sabine, small-absorption limit).

use std::f64::consts::PI;

use crate::autodiff::Dual;

use crate::metrics::octave::N_BANDS;

/// Sabine room constant A (m^2), clamped at RT60 ≥ 0.1 s.
pub fn room_constant_m2<const N: usize>(
    volume_m3: Dual<N>,
    rt60_s: Dual<N>,
) -> Dual<N> {
    let rt = rt60_s.max_f(0.1);
    volume_m3 * 0.161 / rt
}

/// SPL at the receiver for each band, given diffuser PWL, room geometry,
/// and receiver distance. Directivity `Q` defaults to 2 (ceiling diffuser).
pub fn receiver_spl<const N: usize>(
    pwl_bands_db: &[Dual<N>; N_BANDS],
    volume_m3: Dual<N>,
    rt60_s: Dual<N>,
    receiver_distance_m: Dual<N>,
    directivity_q: f64,
) -> [Dual<N>; N_BANDS] {
    let r = receiver_distance_m.max_f(0.3);
    let room_r = room_constant_m2(volume_m3, rt60_s).max_f(0.5);

    let direct = Dual::<N>::constant(directivity_q) / (r * r * (4.0 * PI));
    let reverberant = Dual::<N>::constant(4.0) / room_r;
    let correction = (direct + reverberant).log10() * 10.0;

    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        out[i] = pwl_bands_db[i] + correction;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Doubling distance in a purely reverberant room doesn't change SPL
    /// (reverberant field is distance-invariant). We approximate this by
    /// using a huge room so reverberant dominates direct.
    #[test]
    fn reverberant_dominated_room_is_distance_invariant() {
        let pwl = [Dual::<1>::constant(80.0); N_BANDS];
        let v = Dual::<1>::constant(1000.0);
        let rt = Dual::<1>::constant(2.0);
        let close = receiver_spl(&pwl, v, rt, Dual::<1>::constant(3.0), 2.0);
        let far = receiver_spl(&pwl, v, rt, Dual::<1>::constant(6.0), 2.0);
        for i in 0..N_BANDS {
            // Room constant A = 0.161 · 1000 / 2 = 80.5 m^2; reverberant is
            // ~4/80.5 = 0.05, direct (Q=2, r=3) = 2/113 ≈ 0.018, still
            // non-negligible. Allow 1 dB of wiggle.
            let diff = (far[i].value - close[i].value).abs();
            assert!(diff < 1.0, "band {i}: {diff} dB");
        }
    }

    /// Higher RT60 → larger reverberant term → higher SPL.
    #[test]
    fn longer_rt60_raises_spl() {
        let pwl = [Dual::<1>::constant(80.0); N_BANDS];
        let v = Dual::<1>::constant(50.0);
        let r = Dual::<1>::constant(2.0);
        let dry = receiver_spl(&pwl, v, Dual::<1>::constant(0.3), r, 2.0);
        let live = receiver_spl(&pwl, v, Dual::<1>::constant(1.5), r, 2.0);
        for i in 0..N_BANDS {
            assert!(live[i].value > dry[i].value);
        }
    }

    /// SPL gradient with respect to PWL is 1 in every band (linear passthrough).
    #[test]
    fn spl_gradient_wrt_pwl_is_one() {
        let mut pwl = [Dual::<1>::constant(80.0); N_BANDS];
        pwl[3] = Dual::<1>::var(80.0, 0); // 500 Hz band is the variable
        let v = Dual::<1>::constant(50.0);
        let rt = Dual::<1>::constant(0.6);
        let r = Dual::<1>::constant(2.0);
        let spl = receiver_spl(&pwl, v, rt, r, 2.0);
        for (i, band) in spl.iter().enumerate() {
            let expected = if i == 3 { 1.0 } else { 0.0 };
            assert_relative_eq!(band.grad[0], expected, max_relative = 1e-15);
        }
    }
}
