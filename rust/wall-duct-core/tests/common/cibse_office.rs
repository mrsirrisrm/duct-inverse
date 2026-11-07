//! CIBSE Guide B (2005) example 5.A4.1: a fan's 900 x 750 mm main, 10 m to
//! a branch it continues from as 600 x 600 mm, 5 m to a second where a
//! 600 x 450 mm duct leaves, 3 m to a mitred elbow 600 mm across and 3 m to
//! a 600 x 450 mm termination in a 10 x 8 x 3 m room, the occupant 2 m from
//! it. All unlined. Figures are table 5.A4.1's, 63 Hz to 8 kHz.

#![allow(dead_code)]

use wall_duct_core::model::{BlockParams, Params};
use wall_duct_core::physics::branch::Outlets;
use wall_duct_core::physics::elbow::ElbowKind;
use wall_duct_core::physics::end_reflection::Termination;
use wall_duct_core::physics::source::SourceType;

pub type Bands = [f64; 8];

pub const FAN_LW: Bands = [97.0, 94.0, 99.0, 93.0, 85.0, 74.0, 68.0, 60.0];

/// Runs: (label, length m, width m, height m, the Guide's loss, table 5.4).
pub const RUNS: [(&str, f64, f64, f64, Bands); 3] = [
    ("10 m 900x750", 10.0, 0.9, 0.75, [6.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ("5 m 600x600", 5.0, 0.6, 0.6, [3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ("3 m 600x450", 3.0, 0.6, 0.45, [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
];

/// The branches by area, equation 5.2: 0.36 of 0.72 m², then 0.27 of 0.63.
pub const BRANCHES: [(f64, f64); 2] = [(0.36, 0.72), (0.27, 0.63)];
pub const BRANCHES_GUIDE: [f64; 2] = [3.0, 4.0];

/// Unlined mitred elbow 600 mm across, table 5.6.
pub const ELBOW_GUIDE: Bands = [0.0, 1.0, 5.0, 8.0, 4.0, 3.0, 3.0, 3.0];

/// End reflection, 600 x 450 mm as 600 mm diameter, table 5.9.
pub const ERL_GUIDE: Bands = [8.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Room, table 5.10 with 5.11 for 240 m³ at 2 m.
pub const ROOM_GUIDE: Bands = [-8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0];

pub const LP_GUIDE: Bands = [61.0, 65.0, 70.0, 65.0, 60.0, 49.0, 42.0, 33.0];

pub const ROOM_M3: f64 = 240.0;
pub const RECEIVER_M: f64 = 2.0;

/// The example gives no airflow and counts no regenerated noise: 2 m³/s
/// (3 m/s in the main) keeps the fittings' far below the fan's, and the
/// termination's own noise is left out.
pub const FLOW_M3S: f64 = 2.0;

pub fn params() -> Params {
    let duct = |(_, l, w, h, _): (&str, f64, f64, f64, Bands)| BlockParams::Segment {
        length_m: l,
        width_m: w,
        height_m: h,
        lining_m: 0.0,
        round: false,
        wrapped: false,
    };
    let branch = |(b, total): (f64, f64), w: f64, h: f64| BlockParams::Branch {
        flow_ratio: b / total,
        width_m: w,
        height_m: h,
        round: false,
        outlets: Some(Outlets { branch_m2: b, total_m2: total }),
        junction: wall_duct_core::physics::branch::Junction { side_m2: Some(b), ..Default::default() },
    };
    Params {
        source_type: SourceType::Given(FAN_LW),
        source_offset_db: 0.0,
        volume_flow_m3s: FLOW_M3S,
        room_volume_m3: ROOM_M3,
        receiver_distance_m: RECEIVER_M,
        lined_fit: Default::default(),
        blocks: vec![
            duct(RUNS[0]),
            branch(BRANCHES[0], 0.9, 0.75),
            duct(RUNS[1]),
            branch(BRANCHES[1], 0.6, 0.6),
            duct(RUNS[2]),
            BlockParams::Bend { width_m: 0.6, height_m: 0.45, kind: ElbowKind::Mitred, lined: false, round: false, zeta: None },
            duct(RUNS[2]),
            BlockParams::Diffuser {
                width_m: 0.6,
                height_m: 0.45,
                round: false,
                lw_db: Some([0.0; 8]),
                termination: Termination::Flush,
            },
        ],
    }
}
