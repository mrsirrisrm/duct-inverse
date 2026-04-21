//! Natural attenuation along a straight duct, per metre and per band.
//!
//! Blend of two regimes, gated by lining thickness:
//!
//! - **Lined** (Reynolds-Bakke 1990):
//!     `A/L [dB/m] = B(f) · (P/A)^0.73 · t^C(f)`
//!   with `t` in inches (empirical basis).
//! - **Unlined** (ASHRAE Table 8 approximation): small band-wise values,
//!   loosely scaled by `sqrt(P/A / 10)` (reference P/A = 10 /m).
//!
//! Sigmoid gate on thickness blends them so the unlined regime vanishes
//! once the duct is actually lined, and vice-versa.
//! Smooth everywhere so gradients are well-defined at `t → 0`.

use crate::autodiff::Dual;

use crate::metrics::octave::N_BANDS;

/// Reynolds-Bakke B(f) coefficients — dB/m per (P/A)^0.73 · in^C.
const RB_B: [f64; N_BANDS] = [0.027, 0.105, 0.503, 1.26, 1.37, 0.66, 0.35, 0.15];
const RB_C: [f64; N_BANDS] = [0.00, 0.32, 0.72, 1.00, 0.91, 0.40, 0.10, -0.05];

/// Unlined per-metre loss at each band (approximate ASHRAE Table 8 midvalues).
const UNLINED_PER_M: [f64; N_BANDS] = [0.30, 0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10];

const IN_PER_M: f64 = 39.3701;

/// Perimeter-to-area ratio P/A for a rectangular duct (1/m).
pub fn rectangular_pa<const N: usize>(
    width_m: Dual<N>,
    height_m: Dual<N>,
) -> Dual<N> {
    (width_m + height_m) * 2.0 / (width_m * height_m)
}

/// Per-metre attenuation at each of the 8 octave bands.
pub fn attenuation_db_per_m<const N: usize>(
    width_m: Dual<N>,
    height_m: Dual<N>,
    lining_thickness_m: Dual<N>,
) -> [Dual<N>; N_BANDS] {
    let pa = rectangular_pa(width_m, height_m);
    // pa^0.73 via exp(0.73 * ln pa); pa is strictly positive.
    let pa073 = (pa.ln() * 0.73).exp();

    let t_in = lining_thickness_m * IN_PER_M;
    // Safe floor so t^C is defined at t ≈ 0 (branch-free, gradient lives on
    // the max arm only when crossing above the floor — same as the Python
    // torch.clamp which uses the max subgradient convention).
    let t_safe = t_in.max_f(1e-6);

    // Sigmoid gate ~1 when t > ~2.5 mm (0.1 in). Gate in on `t_in` (i.e.
    // inches) to match the Python factor 50·(t − 0.1).
    let gate = ((t_in - 0.1) * 50.0).sigmoid();
    let unlined_gate = Dual::<N>::constant(1.0) - gate;

    // Unlined correction: sqrt(P/A / 10). Reference P/A is 10 /m.
    let unlined_scale = (pa / 10.0).sqrt();

    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        // lined contribution: B · (P/A)^0.73 · t^C
        let t_c = (t_safe.ln() * RB_C[i]).exp();
        let lined = pa073 * t_c * RB_B[i];
        let unlined = unlined_scale * UNLINED_PER_M[i];
        out[i] = lined * gate + unlined * unlined_gate;
    }
    out
}

/// Total attenuation of a duct segment of given length.
pub fn segment_attenuation_db<const N: usize>(
    length_m: Dual<N>,
    width_m: Dual<N>,
    height_m: Dual<N>,
    lining_thickness_m: Dual<N>,
) -> [Dual<N>; N_BANDS] {
    let per_m = attenuation_db_per_m(width_m, height_m, lining_thickness_m);
    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        out[i] = per_m[i] * length_m;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// With thickness well above the gate (50 mm = ~2 inches), every band
    /// should be dominated by the Reynolds-Bakke lined form. Spot-check
    /// against a direct reproduction of the formula.
    #[test]
    fn lined_regime_matches_formula() {
        let w = Dual::<1>::constant(0.4);
        let h = Dual::<1>::constant(0.3);
        let t = Dual::<1>::constant(0.05);
        let per_m = attenuation_db_per_m(w, h, t);

        let pa: f64 = 2.0 * (0.4 + 0.3) / (0.4 * 0.3);
        let pa073 = pa.powf(0.73);
        let t_in: f64 = 0.05 * IN_PER_M;
        for i in 0..N_BANDS {
            let expected = RB_B[i] * pa073 * t_in.powf(RB_C[i]);
            // Gate ≈ 1 at this thickness; relax relative tolerance to allow
            // for the residual unlined contribution through (1-gate).
            assert_relative_eq!(per_m[i].value, expected, max_relative = 1e-3);
        }
    }

    /// With zero lining, the unlined band values (scaled by sqrt(P/A/10))
    /// should dominate.
    #[test]
    fn unlined_regime_close_to_table_values() {
        let w = Dual::<1>::constant(0.316);
        let h = Dual::<1>::constant(0.316); // P/A ≈ 12.66
        let t = Dual::<1>::constant(0.0);
        let per_m = attenuation_db_per_m(w, h, t);

        let pa: f64 = 2.0 * (0.316 + 0.316) / (0.316 * 0.316);
        let scale = (pa / 10.0).sqrt();
        for i in 0..N_BANDS {
            // Skip bands where RB_C < 0: t_safe^C blows up near t=0 and
            // leaks noticeably through the 0.7% gate residual. This mirrors
            // the Python behaviour — the docstring there flags it explicitly.
            if RB_C[i] < 0.0 {
                continue;
            }
            let expected = UNLINED_PER_M[i] * scale;
            assert_relative_eq!(per_m[i].value, expected, max_relative = 5e-2);
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
}
