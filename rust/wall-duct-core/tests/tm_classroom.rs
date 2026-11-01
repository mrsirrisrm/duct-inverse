//! The engine against TM 5-805-4 §7-6's worked classroom supply
//! (`cargo run --example tm_classroom` prints the comparison).

mod common;

use common::tm_classroom::*;
use wall_duct_core::autodiff::Dual;
use wall_duct_core::metrics::octave::N_BANDS;
use wall_duct_core::model::DuctSystem;
use wall_duct_core::physics::{
    duct_attenuation::{segment_attenuation_db, LinedFit},
    elbow::{elbow_loss_db, ElbowKind},
    room::receiver_spl,
};

type D = Dual<1>;

fn within(what: &str, got: f64, tm: f64, tol: f64) {
    assert!((got - tm).abs() <= tol, "{what}: engine {got:.1} dB, TM {tm} dB (tolerance {tol})");
}

/// The runs within 4 dB to 4 kHz, the 24 x 60 to 250 Hz only: TM
/// interpolates its table there, and the run is past 50 dB in all above.
/// The 10 in round drop is CIBSE's table 5.5, within 1 dB of TM's.
#[test]
fn lined_runs() {
    let c = D::constant;
    for (name, l, w, h, round, tm) in LINED_RUNS {
        let loss = segment_attenuation_db(c(l * FT), c(w * IN), c(h * IN), c(IN), round, false, LinedFit::ReynoldsBevirt);
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

/// With TM's catalogue diffuser the engine's room SPL from one diffuser is
/// within 3.5 dB of TM's in every band (51 47 35 25 29 29 12). TM just meets
/// NC 30 there; the engine rates 31. Pinned: change it only for a change of
/// method, with the reason. (Was 51.2 50.1 34.6 24.5 28.3 27.4 10.5, NC 33,
/// before the lined round drop took CIBSE's table 5.5 for the bare duct's.)
#[test]
fn full_path_with_catalogue_diffuser() {
    let sys = DuctSystem::from_params(&params(Some(DIFFUSER_LW)));
    let spl = sys.receiver_spl_values();
    (0..7).for_each(|i| within(&format!("Lp vs TM, band {i}"), spl[i], LP_ONE_TM[i], 3.5));
    let pinned = [50.4, 48.4, 32.6, 24.2, 28.2, 27.3, 9.8];
    (0..7).for_each(|i| within(&format!("Lp, band {i}"), spl[i], pinned[i], 0.1));
    assert_eq!(sys.nc(), 31);
}

/// With the diffuser's sound estimated (CIBSE, pressure loss factor 4) it
/// runs 1 to 12 dB over the catalogue unit from 250 Hz up. Pinned.
#[test]
fn full_path_with_estimated_diffuser() {
    let sys = DuctSystem::from_params(&params(None));
    let spl = sys.receiver_spl_values();
    let pinned = [50.3, 48.4, 35.7, 35.9, 34.6, 28.7, 22.8];
    (0..7).for_each(|i| within(&format!("Lp, band {i}"), spl[i], pinned[i], 0.1));
    assert_eq!(sys.nc(), 34);
}

/// With ARI 885's lined-duct fit (CIBSE Guide B) for Reynolds & Bevirt's,
/// 125 Hz comes 1.2 dB nearer TM's 47, every band within 2.2 dB of TM's,
/// and the room rates NC 30, as TM. Pinned.
#[test]
fn full_path_with_ari_lining() {
    let mut p = params(Some(DIFFUSER_LW));
    p.lined_fit = LinedFit::Ari885;
    let sys = DuctSystem::from_params(&p);
    let spl = sys.receiver_spl_values();
    (0..7).for_each(|i| within(&format!("Lp vs TM, band {i}"), spl[i], LP_ONE_TM[i], 3.5));
    let pinned = [50.4, 47.2, 33.0, 24.2, 28.2, 27.3, 9.8];
    (0..7).for_each(|i| within(&format!("Lp, band {i}"), spl[i], pinned[i], 0.1));
    assert_eq!(sys.nc(), 30);
}
