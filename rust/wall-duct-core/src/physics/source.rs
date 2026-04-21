//! Source PWL spectra — fan / FCU / VAV box.
//!
//! Canonical shapes by source type, offset in dB by a user-controllable
//! overall level. The shape stays fixed; the offset is the design lever.

use crate::autodiff::Dual;

use crate::metrics::octave::N_BANDS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    FcuCeiling,
    VavBox,
    AxialFan,
}

impl SourceType {
    pub fn shape(self) -> [f64; N_BANDS] {
        match self {
            // 63    125   250   500   1k    2k    4k    8k
            Self::FcuCeiling => [62.0, 60.0, 58.0, 56.0, 54.0, 51.0, 48.0, 42.0],
            Self::VavBox => [70.0, 67.0, 63.0, 60.0, 57.0, 54.0, 50.0, 44.0],
            Self::AxialFan => [82.0, 80.0, 76.0, 72.0, 68.0, 64.0, 58.0, 50.0],
        }
    }
}

/// Source PWL spectrum: canonical shape + overall offset (dB).
pub fn source_pwl<const N: usize>(
    source_type: SourceType,
    overall_pwl_db: Dual<N>,
) -> [Dual<N>; N_BANDS] {
    let shape = source_type.shape();
    let mut out = [Dual::<N>::constant(0.0); N_BANDS];
    for i in 0..N_BANDS {
        out[i] = overall_pwl_db + shape[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn offset_shifts_every_band_equally() {
        let offset = Dual::<1>::var(3.0, 0);
        let pwl = source_pwl(SourceType::FcuCeiling, offset);
        let shape = SourceType::FcuCeiling.shape();
        for i in 0..N_BANDS {
            assert_relative_eq!(pwl[i].value, shape[i] + 3.0, max_relative = 1e-15);
            assert_relative_eq!(pwl[i].grad[0], 1.0, max_relative = 1e-15);
        }
    }
}
