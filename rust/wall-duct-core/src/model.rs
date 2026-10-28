//! Dynamic duct chain: Source → [Block; 1..=MAX_BLOCKS] → Room.
//!
//! Each `Block` variant owns its parameters as `Dual<N>` values tagged at
//! fixed slot indices — 8 system slots followed by up to `MAX_BLOCKS`
//! per-block groups of `MAX_BLOCK_SLOTS` each. The total dual width `N`
//! is sized at compile time so we never allocate on the hot path.
//!
//! Slot layout (`N = N_PARAMS`):
//!
//! ```text
//!   [0]  source_offset
//!   [1]  log_volume_flow
//!   [2]  log_room_volume
//!   [3]  log_rt60
//!   [4]  log_receiver_distance
//!   [5..8]   reserved for future system params
//!   [8..13]   block 0 (up to 5 params)
//!   [13..18]  block 1
//!   ...
//!   [8 + 5k .. 8 + 5(k+1)]  block k
//! ```
//!
//! Each block variant uses only the subset of its 5 slots that it needs;
//! unused slots remain zero-gradient and zero-valued (harmless).

use crate::autodiff::Dual;

use crate::metrics::nc;
use crate::metrics::octave::N_BANDS;
use crate::physics::{
    attenuator::attenuator_il,
    db_add,
    duct_attenuation::segment_attenuation_db,
    end_reflection::{effective_diameter_rectangular, end_reflection_loss_db},
    regenerative::regenerative_pwl_db,
    room::receiver_spl,
    source::{source_pwl, SourceType},
    sub_bands,
};

/// Compile-time maximum chain length. A path extracted from a model, with
/// each straight, bend and branch as its own block, runs to 20 or so. The
/// cost of being oversized is purely a larger sensitivity vector carried
/// through the forward pass.
pub const MAX_BLOCKS: usize = 64;

/// Slots reserved per block (largest variant is Attenuator with 5 params).
pub const MAX_BLOCK_SLOTS: usize = 5;

/// System-level parameter count (source_offset + 4 log-space vars + 3 reserved).
pub const SYSTEM_SLOTS: usize = 8;

/// Total dual width.
pub const N_PARAMS: usize = SYSTEM_SLOTS + MAX_BLOCKS * MAX_BLOCK_SLOTS;

// System slot indices.
pub const SLOT_SOURCE_OFFSET: usize = 0;
pub const SLOT_LOG_VOLUME_FLOW: usize = 1;
pub const SLOT_LOG_ROOM_VOLUME: usize = 2;
pub const SLOT_LOG_RT60: usize = 3;
pub const SLOT_LOG_RECEIVER_DISTANCE: usize = 4;

/// Slot index for the first parameter of block `k`.
pub const fn block_slot_base(k: usize) -> usize {
    SYSTEM_SLOTS + k * MAX_BLOCK_SLOTS
}

type DualN = Dual<N_PARAMS>;
type Band = [DualN; N_BANDS];

/// Linear-space parameters for a single block. Only fields relevant to the
/// variant are read; the others are ignored.
#[derive(Debug, Clone, Copy)]
pub enum BlockParams {
    Segment {
        length_m: f64,
        width_m: f64,
        height_m: f64,
        lining_m: f64,
    },
    Bend {
        width_m: f64,
        height_m: f64,
    },
    Attenuator {
        length_m: f64,
        width_m: f64,
        height_m: f64,
        free_area_ratio: f64,
        splitter_m: f64,
    },
    Diffuser {
        width_m: f64,
        height_m: f64,
    },
    /// Branch division: the path continues with `flow_ratio` of the flow
    /// arriving, and carries that share of the sound power.
    Branch {
        flow_ratio: f64,
    },
}

impl BlockParams {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Segment { .. } => "Segment",
            Self::Bend { .. } => "Bend",
            Self::Attenuator { .. } => "Attenuator",
            Self::Diffuser { .. } => "Diffuser",
            Self::Branch { .. } => "Branch",
        }
    }

    /// Human-readable names for each occupied slot in this variant, in the
    /// order they're tagged in `Block::from_params`. Used by the sensitivity
    /// report to label gradient entries.
    pub fn slot_names(&self) -> &'static [&'static str] {
        match self {
            Self::Segment { .. } => &["log_length", "log_width", "log_height", "lining_raw"],
            Self::Bend { .. } => &["log_width", "log_height"],
            Self::Attenuator { .. } => &[
                "length_raw",
                "log_width",
                "log_height",
                "free_area_raw",
                "splitter_raw",
            ],
            Self::Diffuser { .. } => &["log_width", "log_height"],
            Self::Branch { .. } => &["log_flow_ratio"],
        }
    }
}

/// A block with its parameters realised as `Dual<N>` values tagged at the
/// block's slot base. Forward passes are pure functions of these duals plus
/// the system-level flow `q`.
pub enum Block {
    Segment {
        log_length: DualN,
        log_width: DualN,
        log_height: DualN,
        lining_raw: DualN,
    },
    Bend {
        log_width: DualN,
        log_height: DualN,
    },
    Attenuator {
        length_raw: DualN,
        log_width: DualN,
        log_height: DualN,
        free_area_raw: DualN,
        splitter_raw: DualN,
    },
    Diffuser {
        log_width: DualN,
        log_height: DualN,
    },
    Branch {
        log_flow_ratio: DualN,
    },
}

impl Block {
    /// Build a `Block` from linear-space params, tagging each dual variable
    /// at its slot within the block's reserved `MAX_BLOCK_SLOTS`-wide range.
    pub fn from_params(params: &BlockParams, block_index: usize) -> Self {
        assert!(block_index < MAX_BLOCKS, "too many blocks");
        let base = block_slot_base(block_index);
        let v = |value: f64, offset: usize| DualN::var(value, base + offset);
        match *params {
            BlockParams::Segment {
                length_m,
                width_m,
                height_m,
                lining_m,
            } => Self::Segment {
                log_length: v(length_m.ln(), 0),
                log_width: v(width_m.ln(), 1),
                log_height: v(height_m.ln(), 2),
                lining_raw: v(lining_m, 3),
            },
            BlockParams::Bend { width_m, height_m } => Self::Bend {
                log_width: v(width_m.ln(), 0),
                log_height: v(height_m.ln(), 1),
            },
            BlockParams::Attenuator {
                length_m,
                width_m,
                height_m,
                free_area_ratio,
                splitter_m,
            } => Self::Attenuator {
                length_raw: v(length_m, 0),
                log_width: v(width_m.ln(), 1),
                log_height: v(height_m.ln(), 2),
                free_area_raw: v(free_area_ratio, 3),
                splitter_raw: v(splitter_m, 4),
            },
            BlockParams::Diffuser { width_m, height_m } => Self::Diffuser {
                log_width: v(width_m.ln(), 0),
                log_height: v(height_m.ln(), 1),
            },
            BlockParams::Branch { flow_ratio } => Self::Branch {
                log_flow_ratio: v(flow_ratio.ln(), 0),
            },
        }
    }

    /// Flow leaving this block given the flow arriving: changed only by a
    /// branch.
    pub fn flow_out(&self, q: DualN) -> DualN {
        match self {
            Self::Branch { log_flow_ratio } => q * log_flow_ratio.exp(),
            _ => q,
        }
    }

    /// Apply this block to an incoming PWL spectrum. `q` is the system-wide
    /// volume flow in m^3/s (a `Dual` so its sensitivity flows through).
    pub fn forward(&self, pwl_in: &Band, q: DualN) -> Band {
        match self {
            Self::Segment {
                log_length,
                log_width,
                log_height,
                lining_raw,
            } => {
                let length = log_length.exp();
                let width = log_width.exp();
                let height = log_height.exp();
                let lining = lining_raw.max_f(0.0);
                let loss = segment_attenuation_db(length, width, height, lining);
                sub_bands(pwl_in, &loss)
            }
            Self::Bend {
                log_width,
                log_height,
            } => {
                let width = log_width.exp();
                let height = log_height.exp();
                // Python `Bend.forward`: 1 m of unlined loss + regen at gross area.
                let natural = segment_attenuation_db(
                    DualN::constant(1.0),
                    width,
                    height,
                    DualN::constant(0.0),
                );
                let after_loss = sub_bands(pwl_in, &natural);
                let area = width * height;
                let regen = regenerative_pwl_db(q / area, area);
                db_add(&after_loss, &regen)
            }
            Self::Attenuator {
                length_raw,
                log_width,
                log_height,
                free_area_raw,
                splitter_raw,
            } => {
                let length = length_raw.max_f(0.0);
                let width = log_width.exp();
                let height = log_height.exp();
                let free_ratio = free_area_raw.clamp_f(0.1, 0.9);
                let splitter = splitter_raw.max_f(0.0);
                let il = attenuator_il(length, free_ratio, splitter);
                let after_il = sub_bands(pwl_in, &il);
                // Self-noise uses free-area velocity / free-area cross-section.
                let gross = width * height;
                let free = gross * free_ratio;
                let regen = regenerative_pwl_db(q / free, free);
                db_add(&after_il, &regen)
            }
            Self::Diffuser {
                log_width,
                log_height,
            } => {
                let width = log_width.exp();
                let height = log_height.exp();
                let d_eff = effective_diameter_rectangular(width, height);
                let erl = end_reflection_loss_db(d_eff);
                let area = width * height;
                let regen = regenerative_pwl_db(q / area, area);
                let after_erl = sub_bands(pwl_in, &erl);
                db_add(&after_erl, &regen)
            }
            Self::Branch { log_flow_ratio } => {
                // Power divides in proportion to flow: 10 log10(ratio), as
                // 10 / ln 10 · ln(ratio). Band-independent.
                let division = *log_flow_ratio * (10.0 / std::f64::consts::LN_10);
                let mut out = *pwl_in;
                for band in &mut out {
                    *band = *band + division;
                }
                out
            }
        }
    }
}

/// Full system description in linear units. The `Dual` version is produced
/// by [`DuctSystem::from_params`].
#[derive(Debug, Clone)]
pub struct Params {
    pub source_type: SourceType,
    pub source_offset_db: f64,
    pub volume_flow_m3s: f64,
    pub room_volume_m3: f64,
    pub rt60_s: f64,
    pub receiver_distance_m: f64,
    pub blocks: Vec<BlockParams>,
}

impl Params {
    /// Canonical 3-block demo (segment → attenuator → diffuser), matching
    /// the first-slice test case so the dynamic path can be checked against
    /// the same Python reference numbers.
    pub fn demo() -> Self {
        Self {
            source_type: SourceType::FcuCeiling,
            source_offset_db: 0.0,
            volume_flow_m3s: 0.25,
            room_volume_m3: 50.0,
            rt60_s: 0.6,
            receiver_distance_m: 2.0,
            blocks: vec![
                BlockParams::Segment {
                    length_m: 3.0,
                    width_m: 0.4,
                    height_m: 0.3,
                    lining_m: 0.025,
                },
                BlockParams::Attenuator {
                    length_m: 1.2,
                    width_m: 0.4,
                    height_m: 0.3,
                    free_area_ratio: 0.5,
                    splitter_m: 0.1,
                },
                BlockParams::Diffuser {
                    width_m: 0.4,
                    height_m: 0.3,
                },
            ],
        }
    }
}

/// Differentiable duct system — system-level parameters plus a chain of
/// `Block`s, all sharing a single `Dual<N_PARAMS>` sensitivity vector.
pub struct DuctSystem {
    pub source_type: SourceType,
    pub source_offset: DualN,
    pub log_volume_flow: DualN,
    pub log_room_volume: DualN,
    pub log_rt60: DualN,
    pub log_receiver_distance: DualN,
    pub blocks: Vec<Block>,
    /// Retained copy of the linear-space params so slot→name lookups can
    /// be done after the fact without plumbing labels through `Block`.
    pub block_params: Vec<BlockParams>,
}

impl DuctSystem {
    pub fn from_params(p: &Params) -> Self {
        assert!(
            p.blocks.len() <= MAX_BLOCKS,
            "chain has {} blocks, max is {}",
            p.blocks.len(),
            MAX_BLOCKS
        );
        let v = |value: f64, idx: usize| DualN::var(value, idx);
        let blocks = p
            .blocks
            .iter()
            .enumerate()
            .map(|(i, bp)| Block::from_params(bp, i))
            .collect();
        Self {
            source_type: p.source_type,
            source_offset: v(p.source_offset_db, SLOT_SOURCE_OFFSET),
            log_volume_flow: v(p.volume_flow_m3s.ln(), SLOT_LOG_VOLUME_FLOW),
            log_room_volume: v(p.room_volume_m3.ln(), SLOT_LOG_ROOM_VOLUME),
            log_rt60: v(p.rt60_s.ln(), SLOT_LOG_RT60),
            log_receiver_distance: v(p.receiver_distance_m.ln(), SLOT_LOG_RECEIVER_DISTANCE),
            blocks,
            block_params: p.blocks.clone(),
        }
    }

    pub fn volume_flow(&self) -> DualN { self.log_volume_flow.exp() }
    pub fn room_volume(&self) -> DualN { self.log_room_volume.exp() }
    pub fn rt60(&self) -> DualN { self.log_rt60.exp() }
    pub fn receiver_distance(&self) -> DualN { self.log_receiver_distance.exp() }

    /// `volume_flow` is the flow leaving the source; each block sees the flow
    /// left after the branches before it, and passes on its own `flow_out`.
    pub fn diffuser_pwl(&self) -> Band {
        let mut q = self.volume_flow();
        let mut pwl = source_pwl(self.source_type, self.source_offset);
        for block in &self.blocks {
            q = block.flow_out(q);
            pwl = block.forward(&pwl, q);
        }
        pwl
    }

    pub fn receiver_spl(&self) -> Band {
        receiver_spl(
            &self.diffuser_pwl(),
            self.room_volume(),
            self.rt60(),
            self.receiver_distance(),
            2.0,
        )
    }

    pub fn receiver_spl_values(&self) -> [f64; N_BANDS] {
        let spl = self.receiver_spl();
        let mut out = [0.0; N_BANDS];
        for i in 0..N_BANDS {
            out[i] = spl[i].value;
        }
        out
    }

    pub fn nc(&self) -> i32 {
        nc::nc_exact(&self.receiver_spl_values())
    }

    /// Human-readable label for a slot index: either a system-level name
    /// or `"block {k}.{param}"` using the block's own slot_names list.
    pub fn slot_label(&self, slot: usize) -> String {
        match slot {
            SLOT_SOURCE_OFFSET => "source_offset".into(),
            SLOT_LOG_VOLUME_FLOW => "log_volume_flow".into(),
            SLOT_LOG_ROOM_VOLUME => "log_room_volume".into(),
            SLOT_LOG_RT60 => "log_rt60".into(),
            SLOT_LOG_RECEIVER_DISTANCE => "log_receiver_distance".into(),
            _ if slot < SYSTEM_SLOTS => format!("system_reserved_{slot}"),
            _ => {
                let off = slot - SYSTEM_SLOTS;
                let block = off / MAX_BLOCK_SLOTS;
                let within = off % MAX_BLOCK_SLOTS;
                if let Some(bp) = self.block_params.get(block) {
                    let names = bp.slot_names();
                    let name = names.get(within).copied().unwrap_or("unused");
                    format!("block{block}.{name}")
                } else {
                    format!("block{block}.slot{within}")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Dynamic-chain parity against the Python PoC for the same 3-block
    /// demo case used in the first slice. Receiver SPL per band must match
    /// within 1e-6, and NC must come out as 57.
    #[test]
    fn matches_python_reference_demo_case() {
        let sys = DuctSystem::from_params(&Params::demo());
        let spl = sys.receiver_spl_values();
        let expected: [f64; N_BANDS] = [
            47.88240297989915,
            53.658029197414464,
            60.61023266584057,
            59.88291069919241,
            56.87429424812974,
            52.20217853004527,
            46.41449652581577,
            40.9136770112554,
        ];
        for i in 0..N_BANDS {
            assert_relative_eq!(spl[i], expected[i], max_relative = 1e-6);
        }
        assert_eq!(sys.nc(), 57);
    }

    /// Sanity: every *occupied* slot (system slots + the slots actually
    /// used by each block's variant) must contribute non-zero gradient to
    /// at least one band. Reserved / unused slots stay flat.
    #[test]
    fn occupied_slots_contribute_gradient() {
        let sys = DuctSystem::from_params(&Params::demo());
        let spl = sys.receiver_spl();
        // System slots 0..=4 are all live; 5..=7 are reserved.
        for slot in 0..=4 {
            let any = (0..N_BANDS).any(|b| spl[b].grad[slot].abs() > 1e-12);
            assert!(any, "system slot {slot} has no gradient");
        }
        // Per-block: check the first `slot_names().len()` offsets of each block.
        for (k, bp) in sys.block_params.iter().enumerate() {
            let base = block_slot_base(k);
            for off in 0..bp.slot_names().len() {
                let slot = base + off;
                let any = (0..N_BANDS).any(|b| spl[b].grad[slot].abs() > 1e-12);
                assert!(any, "block {k} slot {off} has no gradient");
            }
        }
    }

    /// Chain that exercises Bend (not used by the demo). Just check it
    /// runs and produces a finite, rated SPL — we don't have Python numbers
    /// pre-computed for this variant, so this is a smoke test.
    #[test]
    fn bend_chain_runs_and_rates() {
        let p = Params {
            source_type: SourceType::VavBox,
            source_offset_db: 0.0,
            volume_flow_m3s: 0.3,
            room_volume_m3: 40.0,
            rt60_s: 0.5,
            receiver_distance_m: 2.5,
            blocks: vec![
                BlockParams::Segment {
                    length_m: 2.0,
                    width_m: 0.35,
                    height_m: 0.25,
                    lining_m: 0.025,
                },
                BlockParams::Bend {
                    width_m: 0.35,
                    height_m: 0.25,
                },
                BlockParams::Attenuator {
                    length_m: 1.0,
                    width_m: 0.35,
                    height_m: 0.25,
                    free_area_ratio: 0.5,
                    splitter_m: 0.1,
                },
                BlockParams::Diffuser {
                    width_m: 0.35,
                    height_m: 0.25,
                },
            ],
        };
        let sys = DuctSystem::from_params(&p);
        let spl = sys.receiver_spl_values();
        for (i, &v) in spl.iter().enumerate() {
            assert!(v.is_finite(), "band {i} SPL is not finite: {v}");
            assert!(v > 0.0 && v < 120.0, "band {i} SPL {v} out of range");
        }
        let nc = sys.nc();
        assert!((15..=70).contains(&nc), "NC {nc} out of range");
    }

    /// A branch passing half the flow lowers every band by 10 log10 2 at the
    /// diffuser, less the change in diffuser regen from the halved velocity.
    /// With the diffuser's regen made negligible (a huge face, ~55 dB below
    /// the fan), the drop is 3.01 dB and the flow-ratio gradient 10 / ln 10,
    /// to within what that regen still contributes.
    #[test]
    fn branch_divides_power_by_flow_ratio() {
        let chain = |blocks: Vec<BlockParams>| Params {
            source_type: SourceType::AxialFan,
            source_offset_db: 0.0,
            volume_flow_m3s: 0.2,
            room_volume_m3: 50.0,
            rt60_s: 0.6,
            receiver_distance_m: 2.0,
            blocks,
        };
        let diffuser = BlockParams::Diffuser { width_m: 20.0, height_m: 20.0 };
        let whole = DuctSystem::from_params(&chain(vec![diffuser])).diffuser_pwl();
        let half = DuctSystem::from_params(&chain(vec![BlockParams::Branch { flow_ratio: 0.5 }, diffuser]))
            .diffuser_pwl();
        for i in 0..N_BANDS {
            assert_relative_eq!(whole[i].value - half[i].value, 10.0 * 2.0f64.log10(), max_relative = 1e-4);
            assert_relative_eq!(half[i].grad[block_slot_base(0)], 10.0 / std::f64::consts::LN_10, max_relative = 1e-4);
        }
    }

    /// Slot labels resolve to readable strings.
    #[test]
    fn slot_labels_resolve() {
        let sys = DuctSystem::from_params(&Params::demo());
        assert_eq!(sys.slot_label(SLOT_SOURCE_OFFSET), "source_offset");
        // Block 0 is a Segment — slot 8 is its log_length.
        assert_eq!(sys.slot_label(block_slot_base(0)), "block0.log_length");
        // Block 1 is an Attenuator — slot +3 is free_area_raw.
        assert_eq!(
            sys.slot_label(block_slot_base(1) + 3),
            "block1.free_area_raw"
        );
    }
}
