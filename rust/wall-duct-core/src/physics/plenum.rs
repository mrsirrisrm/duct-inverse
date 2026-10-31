//! Attenuation of a plenum chamber, by Wells' method as Arup's Strutt
//! documents it:
//! ```text
//!     A = −10 log10( S_o (cos θ / (2π d²) + 1 / R) ),  R = (S_o + S_i + S ᾱ) / (1 − ᾱ)
//! ```
//! `S_o`, `S_i` the outlet and inlet areas, `S` the walls' area less the
//! openings, `ᾱ` their absorption, `d` the inlet-to-outlet distance and `θ`
//! its angle to the inlet's axis. Below the inlet's cut-on, `c / 2a`, the
//! chamber is a plane-wave expansion the equation overstates: there the loss
//! is at most 20 dB, as ASHRAE (2011) advises.

use crate::autodiff::Dual;

use crate::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};

const C_AIR: f64 = 343.0;
const LOW_FREQUENCY_CAP_DB: f64 = 20.0;

/// The chamber's openings and the path between them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Openings {
    pub inlet_m2: f64,
    /// The inlet's larger dimension, which sets its cut-on.
    pub inlet_span_m: f64,
    pub outlet_m2: f64,
    pub distance_m: f64,
    pub cos_theta: f64,
}

/// Attenuation per band of a `length` x `width` x `height` chamber with wall
/// absorption `alpha`.
pub fn plenum_loss_db<const N: usize>(
    length_m: Dual<N>,
    width_m: Dual<N>,
    height_m: Dual<N>,
    openings: &Openings,
    alpha: &[f64; N_BANDS],
) -> [Dual<N>; N_BANDS] {
    let (s_i, s_o) = (openings.inlet_m2, openings.outlet_m2);
    let walls = ((length_m * width_m + length_m * height_m + width_m * height_m) * 2.0 - (s_i + s_o)).max_f(1e-3);
    let direct = openings.cos_theta.max(0.0) / (2.0 * std::f64::consts::PI * openings.distance_m.powi(2));
    let cut_on = C_AIR / (2.0 * openings.inlet_span_m);
    std::array::from_fn(|i| {
        let a = alpha[i].clamp(1e-3, 0.999);
        let room_constant = (walls * a + (s_i + s_o)) / (1.0 - a);
        let loss = -((Dual::<N>::constant(1.0) / room_constant + direct) * s_o).log10() * 10.0;
        if OCTAVE_BANDS_HZ[i] < cut_on && loss.value > LOW_FREQUENCY_CAP_DB {
            Dual::constant(LOW_FREQUENCY_CAP_DB)
        } else {
            loss
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const OPENINGS: Openings = Openings { inlet_m2: 0.25, inlet_span_m: 0.5, outlet_m2: 0.25, distance_m: 2.0, cos_theta: 1.0 };

    /// A 2 x 1.5 x 1.5 m chamber, 0.5 m openings 2 m apart in line, walls
    /// at α 0.5: S = 16.5 − 0.5 = 16 m², R = (8 + 0.5) / 0.5 = 17 m², so
    /// −10 lg(0.25 (1 / 8π + 1 / 17)) = 16.1 dB, under the cap below the
    /// 343 Hz cut-on too.
    #[test]
    fn wells_by_hand() {
        let c = Dual::<1>::constant;
        let loss = plenum_loss_db(c(2.0), c(1.5), c(1.5), &OPENINGS, &[0.5; N_BANDS]);
        let expected = -10.0 * (0.25 * (1.0 / (8.0 * std::f64::consts::PI) + 1.0 / 17.0)).log10();
        for band in loss {
            assert_relative_eq!(band.value, expected, max_relative = 1e-12);
        }
        assert_relative_eq!(expected, 16.1, epsilon = 0.05);
    }

    /// A well-lined chamber loses at most 20 dB below its inlet's cut-on,
    /// and more above it.
    #[test]
    fn low_frequency_cap() {
        let c = Dual::<1>::constant;
        let far = Openings { distance_m: 6.0, ..OPENINGS };
        let loss = plenum_loss_db(c(6.0), c(3.0), c(3.0), &far, &[0.95; N_BANDS]);
        assert!(loss[..3].iter().all(|l| l.value == LOW_FREQUENCY_CAP_DB));
        assert!(loss[3].value > LOW_FREQUENCY_CAP_DB);
    }

    /// More absorption, more loss; a bigger chamber too.
    #[test]
    fn absorption_and_size() {
        let c = Dual::<1>::constant;
        let at = |side: f64, alpha: f64| plenum_loss_db(c(side), c(side), c(side), &OPENINGS, &[alpha; N_BANDS])[5].value;
        assert!(at(1.5, 0.6) > at(1.5, 0.1));
        assert!(at(3.0, 0.1) > at(1.5, 0.1));
    }
}
