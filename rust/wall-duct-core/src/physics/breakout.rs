//! Sound radiated through a duct's walls (breakout), and its way through a
//! ceiling below.
//!
//! The walls' transmission loss is ASHRAE's *Algorithms for HVAC Acoustics*
//! (1991), as Arup's Strutt gives it, `a`, `b` the section in inches, `q`
//! the walls' mass in lb/ft²:
//! ```text
//!     rectangular, below f_L = 24134 / √(ab):  TL = 10 log10(q² f / (a + b)) + 17
//!                  above:                      TL = 20 log10(q f) − 31
//!                  never below                 10 log10(24 L (1/a + 1/b)),  L in ft
//!     round, d in:  the larger of 17.6 lg q − 49.8 lg f − 55.3 lg d + 231.65
//!                             and 17.6 lg q − 6.6 lg f − 36.9 lg d + 97.4
//! ```
//! at most 50 dB. The power radiated from a run `L` long (ASHRAE 2007, after
//! Allen) is `Lw − TL + 10 log10(S* / A)`, `S* = P L*` its effective surface,
//! `L* = (γ^L − 1) / ln γ`, `γ = 10^(−α/10)` for the run's attenuation `α`
//! dB/m. Through a ceiling it loses TM 5-805-4's plenum/ceiling transfer
//! factor (table 7-1).

use crate::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};

const IN_PER_M: f64 = 39.3701;
const FT_PER_M: f64 = 3.28084;
const LB_FT2_PER_KG_M2: f64 = 0.204_816;
const TL_MAX_DB: f64 = 50.0;

/// TM 5-805-4 table 7-1, 63 Hz to 4 kHz, 8 kHz as 4 kHz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ceiling {
    /// Type 1: 1/2 in fibreglass tile, 6 lb/ft³.
    FibreglassTile,
    /// Type 2: 5/8 in mineral fibre tile, 35 lb/ft³.
    MineralFibreTile,
    /// Type 3: 5/8 in plasterboard.
    Plasterboard,
}

impl Ceiling {
    pub fn transfer_db(self) -> [f64; N_BANDS] {
        match self {
            Self::FibreglassTile => [4.0, 8.0, 8.0, 8.0, 10.0, 10.0, 14.0, 14.0],
            Self::MineralFibreTile => [5.0, 9.0, 10.0, 12.0, 14.0, 15.0, 15.0, 15.0],
            Self::Plasterboard => [10.0, 15.0, 21.0, 25.0, 27.0, 26.0, 27.0, 27.0],
        }
    }
}

/// A duct run: its section (round of diameter `width_m`), length and walls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Run {
    pub width_m: f64,
    pub height_m: f64,
    pub round: bool,
    pub length_m: f64,
    pub wall_kg_m2: f64,
}

/// The walls' breakout transmission loss per band, dB.
pub fn breakout_tl_db(run: &Run) -> [f64; N_BANDS] {
    let q = run.wall_kg_m2 * LB_FT2_PER_KG_M2;
    let (a, b) = (run.width_m.max(run.height_m) * IN_PER_M, run.width_m.min(run.height_m) * IN_PER_M);
    OCTAVE_BANDS_HZ.map(|f| {
        let tl = if run.round {
            let d = a;
            if d > 26.0 && f == 4000.0 {
                17.6 * q.log10() - 36.9 * d.log10() + 90.6
            } else {
                let low = 17.6 * q.log10() - 49.8 * f.log10() - 55.3 * d.log10() + 231.65;
                let high = 17.6 * q.log10() - 6.6 * f.log10() - 36.9 * d.log10() + 97.4;
                low.max(high)
            }
        } else {
            let f_l = 24134.0 / (a * b).sqrt();
            let tl = if f < f_l { 10.0 * (q * q * f / (a + b)).log10() + 17.0 } else { 20.0 * (q * f).log10() - 31.0 };
            let floor = 10.0 * (24.0 * run.length_m * FT_PER_M * (1.0 / a + 1.0 / b)).log10();
            tl.max(floor)
        };
        tl.min(TL_MAX_DB)
    })
}

/// Power radiated from the run's outside, given the power entering it and
/// its attenuation per metre.
pub fn radiated_pwl_db(lw_in: &[f64; N_BANDS], run: &Run, attenuation_db_per_m: &[f64; N_BANDS]) -> [f64; N_BANDS] {
    let (perimeter, area) = if run.round {
        (std::f64::consts::PI * run.width_m, std::f64::consts::FRAC_PI_4 * run.width_m.powi(2))
    } else {
        (2.0 * (run.width_m + run.height_m), run.width_m * run.height_m)
    };
    let tl = breakout_tl_db(run);
    std::array::from_fn(|i| {
        let ln_gamma = -attenuation_db_per_m[i].max(0.0) / 10.0 * std::f64::consts::LN_10;
        // (γ^L − 1) / ln γ, which is L as α goes to 0.
        let effective = if ln_gamma.abs() < 1e-9 { run.length_m } else { ((ln_gamma * run.length_m).exp() - 1.0) / ln_gamma };
        lw_in[i] - tl[i] + 10.0 * (perimeter * effective / area).log10()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A 24 x 12 in duct, 24 gauge (1 lb/ft²), 10 ft: f_L = 24134 / √288 =
    /// 1422 Hz; at 125 Hz 10 lg(125 / 36) + 17 = 22.4 dB, at 2 kHz
    /// 20 lg 2000 − 31 = 35.0 dB; at 63 Hz 19.4 dB, over its floor of
    /// 10 lg(240 (1/24 + 1/12)) = 14.8 dB.
    #[test]
    fn rectangular_by_hand() {
        let run = Run { width_m: 24.0 / IN_PER_M, height_m: 12.0 / IN_PER_M, round: false, length_m: 10.0 / FT_PER_M, wall_kg_m2: 1.0 / LB_FT2_PER_KG_M2 };
        let tl = breakout_tl_db(&run);
        assert_relative_eq!(tl[1], 10.0 * (125.0f64 / 36.0).log10() + 17.0, max_relative = 1e-9);
        assert_relative_eq!(tl[5], 20.0 * 2000f64.log10() - 31.0, max_relative = 1e-9);
        assert_relative_eq!(tl[0], 10.0 * (63.0f64 / 36.0).log10() + 17.0, max_relative = 1e-9);
        assert!(tl[0] > 10.0 * (240.0f64 * (1.0 / 24.0 + 1.0 / 12.0)).log10());
        assert!(tl.iter().all(|&t| t <= TL_MAX_DB));
    }

    /// A long thin duct's floor: 100 ft of 4 x 4 in, 10 lg(2400 · 0.5) = 30.8 dB.
    #[test]
    fn rectangular_floor() {
        let run = Run { width_m: 4.0 / IN_PER_M, height_m: 4.0 / IN_PER_M, round: false, length_m: 100.0 / FT_PER_M, wall_kg_m2: 0.5 / LB_FT2_PER_KG_M2 };
        assert_relative_eq!(breakout_tl_db(&run)[0], 10.0 * 1200f64.log10(), max_relative = 1e-9);
    }

    /// A round duct's walls are stiff: a 12 in one, 1 lb/ft², loses
    /// −6.6 lg 1000 − 36.9 lg 12 + 97.4 = 37.8 dB at 1 kHz, the larger form.
    #[test]
    fn round_is_stiff() {
        let run = Run { width_m: 12.0 / IN_PER_M, height_m: 12.0 / IN_PER_M, round: true, length_m: 3.0, wall_kg_m2: 1.0 / LB_FT2_PER_KG_M2 };
        let tl = breakout_tl_db(&run);
        let at_1k = (17.6 * 0.0 - 49.8 * 3.0 - 55.3 * 12f64.log10() + 231.65).max(-6.6 * 3.0 - 36.9 * 12f64.log10() + 97.4);
        assert_relative_eq!(tl[4], at_1k.min(50.0), max_relative = 1e-9);
        assert_relative_eq!(tl[4], 37.8, epsilon = 0.05);
        assert!(tl.iter().all(|&t| t <= TL_MAX_DB));
    }

    /// With no attenuation the effective length is the run's, and the
    /// radiated power is Lw − TL + 10 lg(P L / A); attenuation lowers it.
    #[test]
    fn radiated_power() {
        let run = Run { width_m: 0.6, height_m: 0.3, round: false, length_m: 4.0, wall_kg_m2: 6.3 };
        let lw = [80.0; N_BANDS];
        let tl = breakout_tl_db(&run);
        let bare = radiated_pwl_db(&lw, &run, &[0.0; N_BANDS]);
        for i in 0..N_BANDS {
            assert_relative_eq!(bare[i], 80.0 - tl[i] + 10.0 * (1.8 * 4.0 / 0.18f64).log10(), max_relative = 1e-9);
        }
        let damped = radiated_pwl_db(&lw, &run, &[3.0; N_BANDS]);
        assert!((0..N_BANDS).all(|i| damped[i] < bare[i]));
    }
}
