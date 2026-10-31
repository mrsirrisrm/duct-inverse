//! TM 5-805-4 (1995) §7-6, a classroom's supply: a fan's 24 x 60 in main,
//! lined 1 in throughout, a take-off to a 12 x 24 in branch at 1000 fpm, a
//! tee to 12 x 16 in, and a 10 in drop to each of four 500 cfm diffusers in
//! a tile ceiling. The room is 24 x 24 x 10 ft; the receiver 8 ft from a
//! diffuser. Figures are the manual's, 63 Hz to 4 kHz (it gives no 8 kHz).

#![allow(dead_code)]

use wall_duct_core::model::{BlockParams, Params};
use wall_duct_core::physics::elbow::ElbowKind;
use wall_duct_core::physics::end_reflection::Termination;
use wall_duct_core::physics::source::SourceType;

pub const IN: f64 = 0.0254;
pub const FT: f64 = 0.3048;
pub const CFM: f64 = 0.000_471_947;

pub type Bands7 = [f64; 7];

/// Fan Lw, eq 10-5 with table 10-13; 8 kHz on the table's slope.
pub const FAN_LW: [f64; 8] = [104.0, 100.0, 96.0, 95.0, 91.0, 89.0, 85.0, 81.0];

/// Lined runs: (label, length ft, width in, height in, round, TM's loss).
pub const LINED_RUNS: [(&str, f64, f64, f64, bool, Bands7); 4] = [
    ("20 ft 24x60", 20.0, 24.0, 60.0, false, [3.0, 4.0, 7.0, 18.0, 32.0, 30.0, 30.0]),
    ("9 ft 12x24", 9.0, 12.0, 24.0, false, [3.0, 2.0, 5.0, 12.0, 31.0, 27.0, 19.0]),
    ("10 ft 12x16", 10.0, 12.0, 16.0, false, [4.0, 3.0, 7.0, 16.0, 41.0, 41.0, 28.0]),
    ("3 ft 10 in round", 3.0, 10.0, 10.0, true, [0.0, 1.0, 2.0, 5.0, 6.0, 6.0, 5.0]),
];

/// The four lined elbows' sizes in the plane of the bend, in, and their total.
pub const ELBOWS_IN: [f64; 4] = [24.0, 60.0, 24.0, 16.0];
pub const ELBOWS_TM: Bands7 = [4.0, 8.0, 12.0, 16.0, 20.0, 26.0, 34.0];

/// Take-off by area 2 / 12, then the tee and the diffuser drop halving.
pub const BRANCH_RATIOS: [f64; 3] = [2.0 / 12.0, 0.5, 0.5];
pub const BRANCHES_TM: Bands7 = [14.0; 7];

/// End reflection, 10 in, table 7-5 part A (tile ceiling).
pub const ERL_TM: Bands7 = [16.0, 10.0, 6.0, 2.0, 1.0, 0.0, 0.0];

/// Total insertion loss; TM writes ">50" above 250 Hz.
pub const TOTAL_IL_TM: [Option<f64>; 7] = [Some(44.0), Some(42.0), Some(53.0), None, None, None, None];

/// The diffuser's catalogue Lw (none given at 63 Hz).
pub const DIFFUSER_LW_TM: [Option<f64>; 7] = [None, Some(43.0), Some(37.0), Some(35.0), Some(40.0), Some(40.0), Some(23.0)];

/// The catalogue Lw as an engine spectrum: 63 Hz taken as 125's and 8 kHz
/// as 4 kHz's, both well under the fan's there.
pub const DIFFUSER_LW: [f64; 8] = [43.0, 43.0, 37.0, 35.0, 40.0, 40.0, 23.0, 23.0];

/// Rel SPL at 8 ft in the 5760 ft³ room, from its room factor.
pub const REL_SPL_TM: Bands7 = [-9.0, -11.0, -9.0, -10.0, -11.0, -11.0, -11.0];

/// Room SPL from one diffuser, and from all four (+6 dB).
pub const LP_ONE_TM: Bands7 = [51.0, 47.0, 35.0, 25.0, 29.0, 29.0, 12.0];
pub const LP_FOUR_TM: Bands7 = [57.0, 53.0, 41.0, 31.0, 35.0, 35.0, 18.0];

pub const ROOM_FT3: f64 = 5760.0;
pub const RECEIVER_FT: f64 = 8.0;

/// The path from the fan to one diffuser as engine blocks, in order, the
/// diffuser's Lw `diffuser_lw` (TM's is [`DIFFUSER_LW`]) or else estimated.
/// The fan's flow is six branches' 2000 cfm, so the take-off's flow ratio is
/// TM's area ratio.
pub fn params(diffuser_lw: Option<[f64; 8]>) -> Params {
    let lined = |l_ft: f64, w_in: f64, h_in: f64, round: bool| BlockParams::Segment {
        length_m: l_ft * FT,
        width_m: w_in * IN,
        height_m: h_in * IN,
        lining_m: IN,
        round,
    };
    let elbow = |w_in: f64, h_in: f64| BlockParams::Bend {
        width_m: w_in * IN,
        height_m: h_in * IN,
        kind: ElbowKind::Vaned,
        lined: true,
        round: false,
    };
    let branch = |flow_ratio: f64, w_in: f64, h_in: f64| BlockParams::Branch {
        flow_ratio,
        width_m: w_in * IN,
        height_m: h_in * IN,
        round: false,
    };
    Params {
        source_type: SourceType::Given(FAN_LW),
        source_offset_db: 0.0,
        volume_flow_m3s: 12_000.0 * CFM,
        room_volume_m3: ROOM_FT3 * FT.powi(3),
        receiver_distance_m: RECEIVER_FT * FT,
        blocks: vec![
            elbow(24.0, 60.0),
            lined(20.0, 24.0, 60.0, false),
            elbow(60.0, 24.0),
            branch(BRANCH_RATIOS[0], 24.0, 60.0),
            lined(9.0, 12.0, 24.0, false),
            branch(BRANCH_RATIOS[1], 12.0, 24.0),
            elbow(24.0, 12.0),
            lined(10.0, 12.0, 16.0, false),
            elbow(16.0, 12.0),
            branch(BRANCH_RATIOS[2], 12.0, 16.0),
            lined(3.0, 10.0, 10.0, true),
            BlockParams::Diffuser {
                width_m: 10.0 * IN,
                height_m: 10.0 * IN,
                round: true,
                lw_db: diffuser_lw,
                termination: Termination::Suspended,
            },
        ],
    }
}
