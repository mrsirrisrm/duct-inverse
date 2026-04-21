//! NC rating (Noise Criterion, Beranek) — integer-valued, non-differentiable.
//!
//! The NC family is defined by tabulated 8-band reference curves labelled by
//! their value at 1 kHz. The rating of a spectrum is the lowest-label curve
//! that the spectrum does not exceed in any band.
//!
//! This slice implements the *exact* integer metric only. A smooth /
//! straight-through differentiable version (matching the Python PoC's
//! `nc_soft` + `_NCStraightThrough`) can be layered on later when needed —
//! the first milestone just needs to match the integer NC.

use super::octave::N_BANDS;

/// Tabulated NC reference curves for NC-15 .. NC-70 in steps of 5.
/// Rows: NC label. Columns: 63, 125, 250, 500, 1k, 2k, 4k, 8k Hz.
/// Source: Beranek 1957 / ASHRAE Fundamentals.
pub const NC_TABLE: [[f64; N_BANDS]; 12] = [
    // 63    125   250   500   1k    2k    4k    8k
    [47.0, 36.0, 29.0, 22.0, 17.0, 14.0, 12.0, 11.0], // NC-15
    [51.0, 40.0, 33.0, 26.0, 22.0, 19.0, 17.0, 16.0], // NC-20
    [54.0, 44.0, 37.0, 31.0, 27.0, 24.0, 22.0, 21.0], // NC-25
    [57.0, 48.0, 41.0, 35.0, 31.0, 29.0, 28.0, 27.0], // NC-30
    [60.0, 52.0, 45.0, 40.0, 36.0, 34.0, 33.0, 32.0], // NC-35
    [64.0, 56.0, 50.0, 45.0, 41.0, 39.0, 38.0, 37.0], // NC-40
    [67.0, 60.0, 54.0, 49.0, 46.0, 44.0, 43.0, 42.0], // NC-45
    [71.0, 64.0, 58.0, 54.0, 51.0, 49.0, 48.0, 47.0], // NC-50
    [74.0, 67.0, 62.0, 58.0, 56.0, 54.0, 53.0, 52.0], // NC-55
    [77.0, 71.0, 67.0, 63.0, 61.0, 59.0, 58.0, 57.0], // NC-60
    [80.0, 75.0, 71.0, 68.0, 66.0, 64.0, 63.0, 62.0], // NC-65
    [83.0, 79.0, 75.0, 72.0, 71.0, 69.0, 68.0, 67.0], // NC-70
];

pub const NC_MIN_LABEL: i32 = 15;
pub const NC_MAX_LABEL: i32 = 70;

/// NC curve at any (possibly fractional) label, linearly interpolating
/// between the tabulated 5-step rows. Saturates at the table endpoints.
pub fn curve_at(label: f64) -> [f64; N_BANDS] {
    let idx = (label - 15.0) / 5.0;
    let last = NC_TABLE.len() as i32 - 1;
    let lo = (idx.floor() as i32).clamp(0, last) as usize;
    let hi = (lo + 1).min(NC_TABLE.len() - 1);
    let t = (idx - lo as f64).clamp(0.0, 1.0);
    let mut out = [0.0; N_BANDS];
    for i in 0..N_BANDS {
        out[i] = (1.0 - t) * NC_TABLE[lo][i] + t * NC_TABLE[hi][i];
    }
    out
}

/// Lowest NC label whose reference curve is not exceeded at any band.
/// Searches at integer resolution over [15, 70]; returns 70 if the spectrum
/// exceeds every curve (treated as the rated ceiling for this PoC).
pub fn nc_exact(spl_bands: &[f64; N_BANDS]) -> i32 {
    for label in NC_MIN_LABEL..=NC_MAX_LABEL {
        let curve = curve_at(label as f64);
        let mut ok = true;
        for i in 0..N_BANDS {
            if spl_bands[i] > curve[i] {
                ok = false;
                break;
            }
        }
        if ok {
            return label;
        }
    }
    NC_MAX_LABEL
}

/// Per-band exceedance of the *next-better* NC curve (one below the rating).
/// These are the bands that kept the rating from being lower — the
/// "controlling" bands. Negative values are clamped to 0.
pub fn band_exceedances(spl_bands: &[f64; N_BANDS], nc_value: i32) -> [f64; N_BANDS] {
    let below = curve_at((nc_value - 1).max(NC_MIN_LABEL) as f64);
    let mut out = [0.0; N_BANDS];
    for i in 0..N_BANDS {
        let e = spl_bands[i] - below[i];
        out[i] = if e > 0.0 { e } else { 0.0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Integer labels fall exactly on the tabulated rows.
    #[test]
    fn curve_at_integer_labels_match_table() {
        for (row, label) in (15..=70).step_by(5).enumerate() {
            let c = curve_at(label as f64);
            for i in 0..N_BANDS {
                assert_relative_eq!(c[i], NC_TABLE[row][i], max_relative = 1e-15);
            }
        }
    }

    /// Off-grid labels interpolate linearly.
    #[test]
    fn curve_at_midpoint_is_average_of_neighbours() {
        let c = curve_at(22.5); // midway between NC-20 and NC-25
        for i in 0..N_BANDS {
            let expected = 0.5 * (NC_TABLE[1][i] + NC_TABLE[2][i]);
            assert_relative_eq!(c[i], expected, max_relative = 1e-15);
        }
    }

    /// A spectrum exactly at NC-40 must rate as 40.
    #[test]
    fn spectrum_at_curve_rates_as_label() {
        let spl = curve_at(40.0);
        assert_eq!(nc_exact(&spl), 40);
    }

    /// Bumping the controlling 250 Hz band 3 dB above NC-30 forces NC-35+.
    #[test]
    fn raising_a_band_raises_the_rating() {
        let mut spl = curve_at(30.0);
        spl[2] += 3.0; // 250 Hz
        assert!(nc_exact(&spl) > 30);
    }
}
