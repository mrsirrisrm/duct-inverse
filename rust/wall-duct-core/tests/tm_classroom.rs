//! The engine against TM 5-805-4 §7-6's worked classroom supply
//! (`cargo run --example tm_classroom` prints the comparison).

mod common;

use common::tm_classroom::*;
use wall_duct_core::autodiff::Dual;
use wall_duct_core::metrics::octave::N_BANDS;
use wall_duct_core::model::DuctSystem;
use wall_duct_core::physics::{
    duct_attenuation::segment_attenuation_db,
    elbow::{elbow_loss_db, ElbowKind},
    room::receiver_spl,
};

type D = Dual<1>;

fn within(what: &str, got: f64, tm: f64, tol: f64) {
    assert!((got - tm).abs() <= tol, "{what}: engine {got:.1} dB, TM {tm} dB (tolerance {tol})");
}

/// The rectangular runs within 4 dB to 4 kHz, the 24 x 60 to 250 Hz only:
/// TM interpolates its table there, and the run is past 50 dB in all above.
#[test]
fn lined_runs() {
    let c = D::constant;
    for (name, l, w, h, round, tm) in LINED_RUNS.into_iter().filter(|r| !r.4) {
        let loss = segment_attenuation_db(c(l * FT), c(w * IN), c(h * IN), c(IN), round);
        let bands = if w * h > 1000.0 { 3 } else { 7 };
        (0..bands).for_each(|i| within(&format!("{name}, band {i}"), loss[i].value, tm[i], 4.0));
    }
}

#[test]
fn lined_elbows() {
    let c = D::constant;
    let elbows: Vec<_> = ELBOWS_IN.iter().map(|w| elbow_loss_db(c(w * IN), ElbowKind::Vaned, true)).collect();
    for i in 0..7 {
        within(&format!("elbows, band {i}"), elbows.iter().map(|e| e[i].value).sum(), ELBOWS_TM[i], 1.0);
    }
}

#[test]
fn branch_division() {
    let total: f64 = BRANCH_RATIOS.iter().map(|r| -10.0 * r.log10()).sum();
    within("branches", total, BRANCHES_TM[0], 0.5);
}

/// Schultz's equation stands in for TM's room factor: within 3 dB.
#[test]
fn room() {
    let c = D::constant;
    let rel = receiver_spl(&[c(0.0); N_BANDS], c(ROOM_FT3 * FT.powi(3)), c(RECEIVER_FT * FT));
    (0..7).for_each(|i| within(&format!("room, band {i}"), rel[i].value, REL_SPL_TM[i], 3.0));
}

/// The engine's room SPL from one diffuser, pinned. Where it differs from
/// TM's (51 47 35 25 29 29 12): its end reflection is the flush form, TM's
/// the tile ceiling's, ~2 dB less at 63 and 125 Hz; above 250 Hz it is the
/// CIBSE diffuser estimate, 6 to 13 dB over TM's catalogue unit. Change it
/// only for a change of method, with the reason.
#[test]
fn full_path_is_pinned() {
    let sys = DuctSystem::from_params(&params());
    let spl = sys.receiver_spl_values();
    let pinned = [54.1, 52.8, 38.0, 35.9, 34.6, 28.8, 22.8];
    (0..7).for_each(|i| within(&format!("Lp, band {i}"), spl[i], pinned[i], 0.1));
    // The fan-borne bands stay within 6 dB of TM's.
    (0..2).for_each(|i| within(&format!("Lp vs TM, band {i}"), spl[i], LP_ONE_TM[i], 6.0));
    assert_eq!(sys.nc(), 37);
}
