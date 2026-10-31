//! Attenuation along a straight duct, per metre and per band.
//!
//! - **Unlined rectangular** (ASHRAE 1991, P/A in 1/ft, dB/ft): up to 250 Hz
//!   `17 (P/A)^-0.25 f^-0.85` for P/A > 3, else `1.64 (P/A)^0.73 f^-0.58`;
//!   above, `0.02 (P/A)^0.8`.
//! - **Lined rectangular** (Reynolds & Bevirt 1994, as ASHRAE gives it):
//!   `B(f) (P/A)^C(f) t^D(f)` dB/ft, P/A in 1/ft and `t` the lining in
//!   inches, but never less than the unlined duct loses: at 63 Hz the lining
//!   adds little to the walls' own loss. TM 5-805-4's worked example (§7-6)
//!   is within ~3 dB for its 12 x 16 and 12 x 24 in runs.
//! - **Round** (ASHRAE's table for straight round duct, by diameter range):
//!   0.01 to 0.1 dB/ft, the stiff wall losing far less than a flat one at
//!   low frequency. A lined round duct is taken as unlined, short of its
//!   own fit: conservative.

use crate::autodiff::Dual;

use crate::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};

/// Reynolds & Bevirt lined rectangular duct constants, 63 Hz to 8 kHz.
const RB_B: [f64; N_BANDS] = [0.0133, 0.0574, 0.271, 1.0147, 1.770, 1.392, 1.518, 1.581];
const RB_C: [f64; N_BANDS] = [1.959, 1.410, 0.824, 0.500, 0.695, 0.802, 0.451, 0.219];
const RB_D: [f64; N_BANDS] = [0.917, 0.941, 1.079, 0.854, 0.602, 0.389, 0.208, 0.121];

/// ASHRAE straight round duct, dB/ft, 63 Hz to 8 kHz (8 kHz as 4 kHz), for
/// diameters up to 7, 15, 30 and 60 in.
const ROUND: [(f64, [f64; N_BANDS]); 4] = [
    (7.0, [0.03, 0.03, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10]),
    (15.0, [0.03, 0.03, 0.03, 0.05, 0.07, 0.07, 0.07, 0.07]),
    (30.0, [0.02, 0.02, 0.02, 0.03, 0.05, 0.05, 0.05, 0.05]),
    (60.0, [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02]),
];

const FT_PER_M: f64 = 3.28084;
const RAMP_IN: f64 = 0.25;
const IN_PER_M: f64 = 39.3701;

fn pow<const N: usize>(x: Dual<N>, p: f64) -> Dual<N> {
    (x.ln() * p).exp()
}

/// ASHRAE 1991 unlined rectangular duct loss, dB/ft, at band centre `f`.
fn unlined_db_per_ft<const N: usize>(pa_per_ft: Dual<N>, f: f64) -> Dual<N> {
    if f > 250.0 {
        pow(pa_per_ft, 0.8) * 0.02
    } else if pa_per_ft.value > 3.0 {
        pow(pa_per_ft, -0.25) * (17.0 * f.powf(-0.85))
    } else {
        pow(pa_per_ft, 0.73) * (1.64 * f.powf(-0.58))
    }
}

/// Perimeter-to-area ratio P/A for a rectangular duct (1/m).
pub fn rectangular_pa<const N: usize>(
    width_m: Dual<N>,
    height_m: Dual<N>,
) -> Dual<N> {
    (width_m + height_m) * 2.0 / (width_m * height_m)
}

/// Per-metre attenuation of a rectangular duct at each of the 8 octave bands.
pub fn attenuation_db_per_m<const N: usize>(
    width_m: Dual<N>,
    height_m: Dual<N>,
    lining_thickness_m: Dual<N>,
) -> [Dual<N>; N_BANDS] {
    let pa = rectangular_pa(width_m, height_m) / FT_PER_M;
    // The fit is for linings of 1 to 2 in, and t^D is steep at 0: below
    // RAMP_IN the lined loss falls linearly to 0, keeping a gradient, so a
    // bare duct is the unlined one.
    let t_in = (lining_thickness_m * IN_PER_M).max_f(0.0);
    std::array::from_fn(|i| {
        let lined = if t_in.value < RAMP_IN {
            pow(pa, RB_C[i]) * (RAMP_IN.powf(RB_D[i]) * RB_B[i] / RAMP_IN) * t_in
        } else {
            pow(pa, RB_C[i]) * pow(t_in, RB_D[i]) * RB_B[i]
        };
        let unlined = unlined_db_per_ft(pa, OCTAVE_BANDS_HZ[i]);
        (if lined.value > unlined.value { lined } else { unlined }) * FT_PER_M
    })
}

/// Per-metre attenuation of a round duct of diameter `d`.
pub fn round_db_per_m(diameter_m: f64) -> [f64; N_BANDS] {
    let d_in = diameter_m * IN_PER_M;
    let row = ROUND.iter().find(|r| d_in <= r.0).unwrap_or(&ROUND[ROUND.len() - 1]);
    row.1.map(|per_ft| per_ft * FT_PER_M)
}

/// Total attenuation of a duct segment of given length: rectangular
/// `width_m` x `height_m`, or round of diameter `width_m`.
pub fn segment_attenuation_db<const N: usize>(
    length_m: Dual<N>,
    width_m: Dual<N>,
    height_m: Dual<N>,
    lining_thickness_m: Dual<N>,
    round: bool,
) -> [Dual<N>; N_BANDS] {
    if round {
        return round_db_per_m(width_m.value).map(|per_m| length_m * per_m);
    }
    attenuation_db_per_m(width_m, height_m, lining_thickness_m).map(|per_m| per_m * length_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const IN: f64 = 0.0254;

    /// A 12 x 16 in duct with 1 in lining (P/A 3.5 /ft) above 125 Hz follows
    /// Reynolds & Bevirt: at 1 kHz 1.770 · 3.5^0.695 = 4.23 dB/ft.
    #[test]
    fn lined_regime_matches_formula() {
        let c = Dual::<1>::constant;
        let per_m = attenuation_db_per_m(c(12.0 * IN), c(16.0 * IN), c(IN));
        for i in 1..N_BANDS {
            let expected = RB_B[i] * 3.5f64.powf(RB_C[i]) * FT_PER_M;
            assert_relative_eq!(per_m[i].value, expected, max_relative = 1e-4);
        }
        assert_relative_eq!(per_m[4].value / FT_PER_M, 4.23, epsilon = 0.01);
    }

    /// At 63 Hz the lined duct loses what the unlined one does.
    #[test]
    fn lined_never_below_unlined() {
        let c = Dual::<1>::constant;
        let (w, h) = (c(12.0 * IN), c(16.0 * IN));
        let (lined, bare) = (attenuation_db_per_m(w, h, c(IN)), attenuation_db_per_m(w, h, c(0.0)));
        assert_eq!(lined[0].value, bare[0].value);
        assert!((0..N_BANDS).all(|i| lined[i].value >= bare[i].value));
    }

    /// With zero lining, ASHRAE's unlined equations hold. A 24 in square
    /// duct (P/A 2 /ft) loses 1.64 · 2^0.73 · 63^-0.58 dB/ft at 63 Hz and
    /// 0.02 · 2^0.8 above 250 Hz.
    #[test]
    fn unlined_regime_matches_ashrae() {
        let w = Dual::<1>::constant(0.6096);
        let t = Dual::<1>::constant(0.0);
        let per_m = attenuation_db_per_m(w, w, t);
        let expected = |i: usize| match i {
            0 => 1.64 * 2f64.powf(0.73) * 63f64.powf(-0.58),
            1 => 1.64 * 2f64.powf(0.73) * 125f64.powf(-0.58),
            2 => 1.64 * 2f64.powf(0.73) * 250f64.powf(-0.58),
            _ => 0.02 * 2f64.powf(0.8),
        } * FT_PER_M;
        for (i, band) in per_m.iter().enumerate() {
            assert_relative_eq!(band.value, expected(i), max_relative = 1e-4);
        }
    }

    /// Thicker lining must monotonically increase mid-band attenuation.
    #[test]
    fn thicker_lining_increases_mid_band_attenuation() {
        let w = Dual::<1>::constant(0.4);
        let h = Dual::<1>::constant(0.3);
        let t1 = Dual::<1>::constant(0.025);
        let t2 = Dual::<1>::constant(0.050);
        let a1 = attenuation_db_per_m(w, h, t1);
        let a2 = attenuation_db_per_m(w, h, t2);
        // 500 Hz, 1 kHz should rise.
        assert!(a2[3].value > a1[3].value);
        assert!(a2[4].value > a1[4].value);
    }

    /// A 10 in round duct loses ASHRAE's 0.03 dB/ft at 63 Hz, a tenth of
    /// what a 10 in square one does.
    #[test]
    fn round_duct_loses_little() {
        let c = Dual::<1>::constant;
        let d = c(10.0 * IN);
        let round = segment_attenuation_db(c(1.0 / FT_PER_M), d, d, c(0.0), true);
        let square = segment_attenuation_db(c(1.0 / FT_PER_M), d, d, c(0.0), false);
        assert_relative_eq!(round[0].value, 0.03, max_relative = 1e-9);
        assert!(square[0].value > 10.0 * round[0].value);
    }
}
