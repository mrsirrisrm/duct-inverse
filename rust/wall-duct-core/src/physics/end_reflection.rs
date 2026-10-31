//! End-reflection loss at the duct termination, ASHRAE's form:
//!
//! ```text
//!     ERL(f) = 10 log10( 1 + (a c / (pi · f · D))^2 )
//! ```
//! `a` is 0.7 for a duct ending flush with a hard wall or ceiling and 1.0
//! in free space, which TM 5-805-4 (table 7-5 part A) takes for a diffuser
//! in a suspended tile ceiling too. Above ~500 Hz for typical duct sizes the
//! loss approaches zero.

use crate::autodiff::Dual;

use crate::metrics::octave::{OCTAVE_BANDS_HZ, N_BANDS};

const C_AIR: f64 = 343.0;

/// Where a duct ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Termination {
    /// Flush with a hard ceiling, wall or floor.
    Flush,
    /// In a suspended tile ceiling, or free space.
    Suspended,
}

impl Termination {
    fn a(self) -> f64 {
        match self {
            Self::Flush => 0.7,
            Self::Suspended => 1.0,
        }
    }
}

/// ERL at each of the 8 octave bands, given the effective duct diameter.
pub fn end_reflection_loss_db<const N: usize>(
    effective_diameter_m: Dual<N>,
    termination: Termination,
) -> [Dual<N>; N_BANDS] {
    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        let f = OCTAVE_BANDS_HZ[i];
        let denom = effective_diameter_m * (std::f64::consts::PI * f);
        let ratio = Dual::<N>::constant(termination.a() * C_AIR) / denom;
        let inner = ratio * ratio + 1.0;
        out[i] = inner.log10() * 10.0;
    }
    out
}

/// Diameter of the circle with a rectangular termination's area.
pub fn effective_diameter_rectangular<const N: usize>(
    width_m: Dual<N>,
    height_m: Dual<N>,
) -> Dual<N> {
    (width_m * height_m * (4.0 / std::f64::consts::PI)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ERL rolls off toward 0 at high frequencies for typical duct sizes.
    #[test]
    fn high_band_loss_is_small() {
        let d = effective_diameter_rectangular(
            Dual::<1>::constant(0.4),
            Dual::<1>::constant(0.3),
        );
        let erl = end_reflection_loss_db(d, Termination::Flush);
        // 8 kHz band loss should be well under 1 dB.
        assert!(erl[N_BANDS - 1].value < 1.0);
        // 63 Hz band loss should be several dB.
        assert!(erl[0].value > 3.0);
    }

    /// Doubling the diameter drops the low-band loss (bigger aperture, less
    /// mismatch with the free-field).
    #[test]
    fn bigger_diameter_reduces_low_band_loss() {
        let small = end_reflection_loss_db(Dual::<1>::constant(0.2), Termination::Flush);
        let large = end_reflection_loss_db(Dual::<1>::constant(0.4), Termination::Flush);
        assert!(large[0].value < small[0].value);
    }

    /// Within 1.5 dB of ASHRAE's tabulated loss for a 250 mm square duct:
    /// 12, 8, 4, 1, 0 dB at 63 Hz to 1 kHz.
    #[test]
    fn matches_the_tabulated_loss() {
        let d = effective_diameter_rectangular(Dual::<1>::constant(0.25), Dual::<1>::constant(0.25));
        let erl = end_reflection_loss_db(d, Termination::Flush);
        for (i, table) in [12.0, 8.0, 4.0, 1.0, 0.0].into_iter().enumerate() {
            assert!((erl[i].value - table).abs() < 1.5, "band {i}: {} dB", erl[i].value);
        }
    }

    /// A 10 in round duct in a tile ceiling within 1.6 dB of TM table 7-5
    /// part A: 16, 10, 6, 2, 1 dB at 63 Hz to 1 kHz.
    #[test]
    fn suspended_matches_tm_part_a() {
        let erl = end_reflection_loss_db(Dual::<1>::constant(0.254), Termination::Suspended);
        for (i, table) in [16.0, 10.0, 6.0, 2.0, 1.0].into_iter().enumerate() {
            assert!((erl[i].value - table).abs() < 1.6, "band {i}: {} dB", erl[i].value);
        }
    }
}
