//! Room reception: diffuser PWL → receiver SPL, by Schultz's equation, which
//! ASHRAE uses for a single source in a normally furnished room:
//! ```text
//!     SPL = PWL − 10 log10 r − 5 log10 V − 3 log10 f + 25    (r ft, V ft³)
//! ```
//! It needs no absorption data: its fall with frequency is the furnishings'.

use crate::autodiff::Dual;

use crate::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};

const FT: f64 = 0.3048;

/// SPL at the receiver for each band, given diffuser PWL, room volume and
/// receiver distance.
pub fn receiver_spl<const N: usize>(
    pwl_bands_db: &[Dual<N>; N_BANDS],
    volume_m3: Dual<N>,
    receiver_distance_m: Dual<N>,
) -> [Dual<N>; N_BANDS] {
    let r_ft = receiver_distance_m.max_f(0.3) / FT;
    let v_ft3 = volume_m3 / FT.powi(3);
    let room = -(r_ft.log10() * 10.0) - v_ft3.log10() * 5.0 + 25.0;
    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        out[i] = pwl_bands_db[i] + room - 3.0 * OCTAVE_BANDS_HZ[i].log10();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 50 m³ (1766 ft³), 2 m (6.56 ft), 1 kHz: 25 − 8.17 − 16.23 − 9 = −8.40 dB.
    #[test]
    fn matches_schultz() {
        let pwl = [Dual::<1>::constant(0.0); N_BANDS];
        let spl = receiver_spl(&pwl, Dual::constant(50.0), Dual::constant(2.0));
        assert_relative_eq!(spl[4].value, -8.40, epsilon = 0.01);
        assert_relative_eq!(spl[3].value - spl[4].value, 3.0 * 2f64.log10(), max_relative = 1e-12);
    }

    /// Doubling distance lowers every band by 3 dB; doubling volume by 1.5.
    #[test]
    fn distance_and_volume() {
        let pwl = [Dual::<1>::constant(80.0); N_BANDS];
        let at = |v: f64, r: f64| receiver_spl(&pwl, Dual::constant(v), Dual::constant(r));
        let (base, far, big) = (at(50.0, 2.0), at(50.0, 4.0), at(100.0, 2.0));
        for i in 0..N_BANDS {
            assert_relative_eq!(base[i].value - far[i].value, 10.0 * 2f64.log10(), max_relative = 1e-12);
            assert_relative_eq!(base[i].value - big[i].value, 5.0 * 2f64.log10(), max_relative = 1e-12);
        }
    }

    /// SPL gradient with respect to PWL is 1 in every band (linear passthrough).
    #[test]
    fn spl_gradient_wrt_pwl_is_one() {
        let mut pwl = [Dual::<1>::constant(80.0); N_BANDS];
        pwl[3] = Dual::<1>::var(80.0, 0); // 500 Hz band is the variable
        let spl = receiver_spl(&pwl, Dual::constant(50.0), Dual::constant(2.0));
        for (i, band) in spl.iter().enumerate() {
            let expected = if i == 3 { 1.0 } else { 0.0 };
            assert_relative_eq!(band.grad[0], expected, max_relative = 1e-15);
        }
    }
}
