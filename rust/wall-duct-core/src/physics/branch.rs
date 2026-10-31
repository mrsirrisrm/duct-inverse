//! The change in sound power down one branch of a junction, ASHRAE's
//! *Algorithms for HVAC Acoustics* (1991) as Arup's Strutt gives it:
//! ```text
//!     ΔL = 10 log10[1 − ((r − 1) / (r + 1))²] + 10 log10(S_B / ΣS_B),  r = ΣS_B / S_M
//! ```
//! `S_B` the branch's area, `ΣS_B` all the outlets', the continuing main's
//! included, and `S_M` the main's before the junction. The first term is the
//! reflection where the outlets' area differs from the main's; it holds only
//! for plane waves, below the main's cut-on (`c / 2a`, or `0.586 c / d` for a
//! round duct).

use crate::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};

const C_AIR: f64 = 343.0;

/// The main before a junction and the outlets after it, areas in m².
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outlets {
    pub branch_m2: f64,
    pub total_m2: f64,
}

/// Cut-on of a duct `width` x `height`, or round of diameter `width`.
pub fn cut_on_hz(width_m: f64, height_m: f64, round: bool) -> f64 {
    if round { 0.586 * C_AIR / width_m } else { C_AIR / (2.0 * width_m.max(height_m)) }
}

/// ΔL per band, dB (negative: a loss), for a main of `main_m2` cutting on
/// at `main_cut_on_hz`.
pub fn branch_change_db(outlets: &Outlets, main_m2: f64, main_cut_on_hz: f64) -> [f64; N_BANDS] {
    let r = outlets.total_m2 / main_m2;
    let reflection = 10.0 * (1.0 - ((r - 1.0) / (r + 1.0)).powi(2)).log10();
    let division = 10.0 * (outlets.branch_m2 / outlets.total_m2).log10();
    OCTAVE_BANDS_HZ.map(|f| division + if f < main_cut_on_hz { reflection } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Outlets of the main's area split evenly: 3 dB, no reflection.
    #[test]
    fn even_split_of_equal_area() {
        let change = branch_change_db(&Outlets { branch_m2: 0.1, total_m2: 0.2 }, 0.2, 1000.0);
        change.iter().for_each(|&d| assert_relative_eq!(d, -10.0 * 2f64.log10(), max_relative = 1e-12));
    }

    /// Outlets twice the main's area reflect 10 lg(8 / 9) = 0.51 dB more,
    /// below the main's cut-on only.
    #[test]
    fn reflection_below_cut_on() {
        let change = branch_change_db(&Outlets { branch_m2: 0.2, total_m2: 0.4 }, 0.2, 400.0);
        let split = -10.0 * 2f64.log10();
        for (i, &d) in change.iter().enumerate() {
            let expected = split + if OCTAVE_BANDS_HZ[i] < 400.0 { 10.0 * (8.0f64 / 9.0).log10() } else { 0.0 };
            assert_relative_eq!(d, expected, max_relative = 1e-12);
        }
        assert_relative_eq!(10.0 * (8.0f64 / 9.0).log10(), -0.51, epsilon = 0.005);
    }

    /// A 600 x 300 mm main cuts on at 286 Hz; a 315 mm round one at 638 Hz.
    #[test]
    fn cut_on() {
        assert_relative_eq!(cut_on_hz(0.6, 0.3, false), 285.8, epsilon = 0.1);
        assert_relative_eq!(cut_on_hz(0.315, 0.315, true), 638.1, epsilon = 0.1);
    }
}
