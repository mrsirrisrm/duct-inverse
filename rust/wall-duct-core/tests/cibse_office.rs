//! The engine against CIBSE Guide B example 5.A4.1's office supply, element
//! by element and in all.

mod common;

use common::cibse_office::*;
use wall_duct_core::autodiff::Dual;
use wall_duct_core::metrics::octave::N_BANDS;
use wall_duct_core::model::DuctSystem;
use wall_duct_core::physics::{
    branch::{branch_change_db, cut_on_hz, Outlets},
    duct_attenuation::segment_attenuation_db,
    elbow::{elbow_loss_db, ElbowKind},
    end_reflection::{effective_diameter_rectangular, end_reflection_loss_db, Termination},
    room::receiver_spl,
};

type D = Dual<1>;

fn within(what: &str, got: f64, guide: f64, tol: f64) {
    assert!((got - guide).abs() <= tol, "{what}: engine {got:.1} dB, Guide {guide} dB (tolerance {tol})");
}

/// ASHRAE's unlined duct against the Guide's table 5.4: within 1.5 dB a run,
/// the engine losing a little more at 63 to 250 Hz in the smaller ducts.
#[test]
fn unlined_runs() {
    let c = D::constant;
    for (name, l, w, h, guide) in RUNS {
        let loss = segment_attenuation_db(c(l), c(w), c(h), c(0.0), false, false, Default::default());
        (0..N_BANDS).for_each(|i| within(&format!("{name}, band {i}"), loss[i].value, guide[i], 1.5));
    }
}

/// Division by area as equation 5.2, with ASHRAE's reflection (0.3 dB at
/// the second, below the 600 mm main's cut-on) on top.
#[test]
fn branches() {
    let mains = [(0.675, cut_on_hz(0.9, 0.75, false)), (0.36, cut_on_hz(0.6, 0.6, false))];
    for (k, ((b, total), (main, cut_on))) in BRANCHES.into_iter().zip(mains).enumerate() {
        let change = branch_change_db(&Outlets { branch_m2: b, total_m2: total }, main, cut_on);
        change.iter().for_each(|d| within(&format!("branch {k}"), -d, BRANCHES_GUIDE[k], 0.5));
    }
}

/// ASHRAE's table against the Guide's 5.6 (both NEBB's, in whole-dB steps
/// by frequency x width): within 1 dB.
#[test]
fn elbow() {
    let loss = elbow_loss_db(D::constant(0.6), ElbowKind::Mitred, false);
    (0..N_BANDS).for_each(|i| within(&format!("elbow, band {i}"), loss[i].value, ELBOW_GUIDE[i], 1.0));
}

/// The Guide rounds the 586 mm effective diameter to 600 for table 5.9.
#[test]
fn end_reflection() {
    let d = effective_diameter_rectangular(D::constant(0.6), D::constant(0.45));
    let erl = end_reflection_loss_db(d, Termination::Flush);
    (0..N_BANDS).for_each(|i| within(&format!("ERL, band {i}"), erl[i].value, ERL_GUIDE[i], 1.0));
}

/// Schultz's equation, which the Guide's tables 5.10 and 5.11 tabulate.
#[test]
fn room() {
    let c = D::constant;
    let rel = receiver_spl(&[c(0.0); N_BANDS], c(ROOM_M3), c(RECEIVER_M));
    (0..N_BANDS).for_each(|i| within(&format!("room, band {i}"), rel[i].value, ROOM_GUIDE[i], 0.6));
}

/// In all, within 3 dB of the Guide's 61 65 70 65 60 49 42 33 in every
/// band, 2.7 dB under at 63 to 125 Hz from the unlined runs and the end
/// reflection. Pinned: change it only for a change of method, with the
/// reason.
#[test]
fn full_path() {
    let sys = DuctSystem::from_params(&params());
    let spl = sys.receiver_spl_values();
    (0..N_BANDS).for_each(|i| within(&format!("Lp vs Guide, band {i}"), spl[i], LP_GUIDE[i], 3.0));
    let pinned = [58.3, 62.3, 68.3, 65.5, 60.3, 49.3, 42.4, 33.5];
    (0..N_BANDS).for_each(|i| within(&format!("Lp, band {i}"), spl[i], pinned[i], 0.1));
    assert_eq!(sys.nc(), 63);
}

/// At the chosen flow the fittings' own noise is nowhere within 0.1 dB of
/// mattering, as the example takes it.
#[test]
fn regen_negligible() {
    let mut quiet = params();
    quiet.volume_flow_m3s = 0.2;
    let (at_flow, without) = (DuctSystem::from_params(&params()).receiver_spl_values(), DuctSystem::from_params(&quiet).receiver_spl_values());
    (0..N_BANDS).for_each(|i| within(&format!("regen, band {i}"), at_flow[i], without[i], 0.1));
}
