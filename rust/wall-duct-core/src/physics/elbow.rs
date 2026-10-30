//! Insertion loss of an unlined elbow.
//!
//! A square elbow without turning vanes: ASHRAE tabulates it by `f · w`
//! (kHz · in, `w` the duct width in the plane of the bend): 0, 1, 5, 8, 4 and
//! 3 dB over the ranges split at 1.9, 3.8, 7.5, 15 and 30. Interpolated
//! linearly in log `f · w` between the ranges' geometric centres, so the loss
//! has a gradient in `w`.
//!
//! A smooth one (turning vanes, radiused or round) loses far less: ASHRAE's
//! round elbow, 0, 1, 2 and 3 dB split at `f · D` 1.9, 3.8 and 7.5. TM 5-805-4
//! (table 7-6) gives vaned and round elbows the same loss.

use crate::autodiff::Dual;

use crate::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};

const IN_PER_M: f64 = 39.3701;

/// (f · w at a range's geometric centre, kHz · in; loss, dB).
const MITRED: &[(f64, f64)] = &[(1.34, 0.0), (2.69, 1.0), (5.34, 5.0), (10.6, 8.0), (21.2, 4.0), (42.4, 3.0)];
const SMOOTH: &[(f64, f64)] = &[(1.34, 0.0), (2.69, 1.0), (5.34, 2.0), (10.6, 3.0)];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElbowKind {
    /// Square, without turning vanes.
    Mitred,
    /// Square, with turning vanes.
    Vaned,
    /// Radiused, round or gored.
    Radiused,
}

impl ElbowKind {
    /// Datum level of its regenerated noise, Woods' table 5.2.
    pub fn regen_k_db(self) -> f64 {
        match self {
            Self::Mitred => 57.0,
            Self::Vaned => 56.0,
            Self::Radiused => 48.0,
        }
    }
}

/// Elbow loss at each of the 8 octave bands.
pub fn elbow_loss_db<const N: usize>(width_m: Dual<N>, kind: ElbowKind) -> [Dual<N>; N_BANDS] {
    let table = if kind == ElbowKind::Mitred { MITRED } else { SMOOTH };
    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        let fw = (width_m * (IN_PER_M * OCTAVE_BANDS_HZ[i] / 1000.0)).ln();
        let (first, last) = (table[0], table[table.len() - 1]);
        out[i] = if fw.value <= first.0.ln() {
            Dual::constant(first.1)
        } else if fw.value >= last.0.ln() {
            Dual::constant(last.1)
        } else {
            let k = table.windows(2).position(|p| fw.value < p[1].0.ln()).unwrap();
            let ((x0, y0), (x1, y1)) = (table[k], table[k + 1]);
            (fw - x0.ln()) * ((y1 - y0) / (x1 / x0).ln()) + y0
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A 250 mm (9.84 in) elbow sits at f · w = 4.9, 9.8, 19.7 at 500 Hz,
    /// 1 and 2 kHz: in the 5, 8 and 4 dB ranges, near their centres.
    #[test]
    fn follows_the_table() {
        let loss = elbow_loss_db(Dual::<1>::constant(0.25), ElbowKind::Mitred);
        assert!(loss[0].value < 0.5);
        for (i, table) in [(3, 5.0), (4, 8.0), (5, 4.0)] {
            assert!((loss[i].value - table).abs() < 1.5, "band {i}: {} dB", loss[i].value);
        }
    }

    /// A 160 mm (6.3 in) round elbow loses nothing at 63 and 125 Hz (f · D
    /// under 1.34) and 3 dB from 2 kHz (12.6, past the last range's centre).
    #[test]
    fn small_round_elbow() {
        let loss = elbow_loss_db(Dual::<1>::constant(0.16), ElbowKind::Radiused);
        assert!(loss[..2].iter().all(|l| l.value == 0.0));
        assert!(loss[5..].iter().all(|l| l.value == 3.0));
    }

    /// At a range's centre the loss is the tabulated value exactly.
    #[test]
    fn exact_at_range_centres() {
        let w = 10.6 / IN_PER_M; // f · w = 10.6 at 1 kHz
        assert_relative_eq!(elbow_loss_db(Dual::<1>::constant(w), ElbowKind::Mitred)[4].value, 8.0, max_relative = 1e-12);
    }
}
