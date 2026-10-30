//! Packed-splitter attenuator insertion loss.
//!
//! ```text
//!     IL(f) = length · per_m_peak(f) · blockage^1.5 · lining_factor(t)
//! ```
//! with `blockage = 1 − free_area_ratio`. Length = 0 → IL = 0 cleanly.

use crate::autodiff::Dual;

use crate::metrics::octave::N_BANDS;

/// TM 5-805-4 table 7-7: a 5 ft low pressure drop attenuator's IL, dB.
const TM_LOW_DROP_5FT: [f64; N_BANDS] = [3.0, 5.0, 11.0, 23.0, 31.0, 34.0, 19.0, 13.0];

/// Per-metre IL (dB/m) before the blockage factor, set so that the
/// reference design (50 % free, 100 mm splitters) is TM's 5 ft low pressure
/// drop unit. Its standard drop unit is ~1.5 times that, as blockage^1.5
/// gives at ~35 % free.
const PER_M_PEAK: [f64; N_BANDS] = {
    let mut out = [0.0; N_BANDS];
    let mut i = 0;
    while i < N_BANDS {
        // 1.524 m, and 0.5^1.5 = 0.353553...
        out[i] = TM_LOW_DROP_5FT[i] / 1.524 / 0.353_553_390_593_273_8;
        i += 1;
    }
    out
};

const REFERENCE_T_M: f64 = 0.1;

/// Smoothly-saturating lining factor: 0 at t=0, ≈1 at t=100 mm.
pub fn lining_factor<const N: usize>(thickness_m: Dual<N>) -> Dual<N> {
    Dual::<N>::constant(1.0) - (-thickness_m / 0.05).exp()
}

/// Insertion loss of a splitter attenuator, per octave band (8,).
pub fn attenuator_il<const N: usize>(
    length_m: Dual<N>,
    free_area_ratio: Dual<N>,
    splitter_thickness_m: Dual<N>,
) -> [Dual<N>; N_BANDS] {
    // Blockage in (ε, 1-ε) — clamp_f zeros out gradient outside the range
    // (sub-gradient convention), matching torch.clamp in the Python code.
    let blockage = (Dual::<N>::constant(1.0) - free_area_ratio).clamp_f(1e-3, 0.95);

    // blockage^1.5 via exp(1.5 · ln blockage).
    let blockage_15 = (blockage.ln() * 1.5).exp();

    // Relative lining factor — reference value is a plain f64 so `t_factor`
    // is 1.0 at the reference thickness and scales smoothly either side.
    let ref_factor = {
        // lining_factor on a Dual<0>-equivalent — evaluate as plain f64.
        1.0 - (-REFERENCE_T_M / 0.05f64).exp()
    };
    let t_factor = lining_factor(splitter_thickness_m) / ref_factor;

    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        out[i] = length_m * blockage_15 * t_factor * PER_M_PEAK[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The reference design, 5 ft long, is TM's low pressure drop unit.
    #[test]
    fn reference_design_is_tm_low_drop_5ft() {
        let c = Dual::<1>::constant;
        let il = attenuator_il(c(1.524), c(0.5), c(REFERENCE_T_M));
        for i in 0..N_BANDS {
            assert_relative_eq!(il[i].value, TM_LOW_DROP_5FT[i], max_relative = 1e-9);
        }
    }

    /// Length of 0 makes every band's IL exactly 0.
    #[test]
    fn zero_length_zero_il() {
        let l = Dual::<1>::constant(0.0);
        let r = Dual::<1>::constant(0.5);
        let t = Dual::<1>::constant(0.1);
        let il = attenuator_il(l, r, t);
        for i in 0..N_BANDS {
            assert_relative_eq!(il[i].value, 0.0, epsilon = 1e-15);
        }
    }

    /// At the reference (100 mm splitter, 50 % free, 1 m length), IL exactly
    /// equals PER_M_PEAK · 0.5^1.5.
    #[test]
    fn reference_design_matches_peak_table() {
        let l = Dual::<1>::constant(1.0);
        let r = Dual::<1>::constant(0.5);
        let t = Dual::<1>::constant(REFERENCE_T_M);
        let il = attenuator_il(l, r, t);
        for i in 0..N_BANDS {
            let expected = PER_M_PEAK[i] * 0.5f64.powf(1.5);
            assert_relative_eq!(il[i].value, expected, max_relative = 1e-12);
        }
    }

    /// Tighter free area (blockier) raises IL.
    #[test]
    fn more_blockage_raises_il() {
        let l = Dual::<1>::constant(1.0);
        let t = Dual::<1>::constant(REFERENCE_T_M);
        let open = attenuator_il(l, Dual::<1>::constant(0.7), t);
        let tight = attenuator_il(l, Dual::<1>::constant(0.3), t);
        for i in 0..N_BANDS {
            assert!(tight[i].value > open[i].value);
        }
    }
}
