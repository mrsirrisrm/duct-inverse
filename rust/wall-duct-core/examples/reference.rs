//! The engine's elements beside the published references they stand in for:
//! ASHRAE's unlined duct equations (1991, as Arup's Strutt documents them),
//! its elbow and end reflection tables, and Schultz's room equation.
//!
//! `cargo run --example reference`

use wall_duct_core::autodiff::Dual;
use wall_duct_core::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};
use wall_duct_core::physics::{
    duct_attenuation::segment_attenuation_db,
    elbow::{elbow_loss_db, ElbowKind},
    end_reflection::{effective_diameter_rectangular, end_reflection_loss_db},
    room::receiver_spl,
};

type D = Dual<1>;
const FT: f64 = 0.3048;
const IN: f64 = 0.0254;

fn row(label: &str, v: impl Fn(usize) -> f64) {
    print!("{label:<34}");
    (0..N_BANDS).for_each(|i| print!("{:>6.1}", v(i)));
    println!();
}

fn header(title: &str) {
    println!("\n{title}");
    print!("{:<34}", "");
    OCTAVE_BANDS_HZ.iter().for_each(|f| print!("{f:>6}"));
    println!();
}

/// ASHRAE 1991 unlined rectangular duct, dB/m.
fn ashrae_unlined(w: f64, h: f64, f: f64) -> f64 {
    let pa = 2.0 * (w + h) / (w * h) * FT; // 1/ft
    let per_ft = if f > 250.0 {
        0.02 * pa.powf(0.8)
    } else if pa > 3.0 {
        17.0 * pa.powf(-0.25) * f.powf(-0.85)
    } else {
        1.64 * pa.powf(0.73) * f.powf(-0.58)
    };
    per_ft / FT
}

/// ASHRAE unlined square elbow without turning vanes, by f·w in kHz·in.
fn ashrae_elbow(w: f64, f: f64) -> f64 {
    match f / 1000.0 * w / IN {
        x if x < 1.9 => 0.0,
        x if x < 3.8 => 1.0,
        x if x < 7.5 => 5.0,
        x if x < 15.0 => 8.0,
        x if x < 30.0 => 4.0,
        _ => 3.0,
    }
}

/// End reflection, sharp-ended duct, by √area in mm (63–1000 Hz).
const ERL_TABLE: [(f64, [f64; 5]); 5] = [
    (125.0, [17.0, 12.0, 8.0, 4.0, 1.0]),
    (250.0, [12.0, 8.0, 4.0, 1.0, 0.0]),
    (500.0, [8.0, 4.0, 1.0, 0.0, 0.0]),
    (1000.0, [4.0, 1.0, 0.0, 0.0, 0.0]),
    (2000.0, [1.0, 0.0, 0.0, 0.0, 0.0]),
];

fn main() {
    let c = D::constant;
    for (w, h) in [(0.3, 0.3), (0.6, 0.3), (0.6, 0.6), (1.2, 0.6)] {
        header(&format!("Unlined duct {w} x {h} m, dB/m"));
        let e = segment_attenuation_db(c(1.0), c(w), c(h), c(0.0), false);
        row("engine", |i| e[i].value);
        row("ASHRAE 1991 equations", |i| ashrae_unlined(w, h, OCTAVE_BANDS_HZ[i]));
    }

    for w in [0.15, 0.25, 0.5, 1.0] {
        header(&format!("Square elbow {w} x {w} m, no vanes: loss, dB"));
        let e = elbow_loss_db(c(w), ElbowKind::Mitred, false);
        row("engine", |i| e[i].value);
        row("ASHRAE elbow table", |i| ashrae_elbow(w, OCTAVE_BANDS_HZ[i]));
    }

    for (side, table) in ERL_TABLE {
        let s = side / 1000.0;
        for (w, h) in [(s, s), (2.0 * s, s / 2.0)] {
            header(&format!("End reflection, {:.0} x {:.0} mm, dB", w * 1000.0, h * 1000.0));
            let e = end_reflection_loss_db(effective_diameter_rectangular(c(w), c(h)));
            row("engine", |i| e[i].value);
            row("table (by sqrt area)", |i| table.get(i).copied().unwrap_or(f64::NAN));
        }
    }

    for (v, r) in [(50.0, 2.0), (50.0, 1.5), (200.0, 2.0), (500.0, 3.0)] {
        header(&format!("Room {v} m3, receiver {r} m: Lp - Lw, dB"));
        let e = receiver_spl(&[c(0.0); N_BANDS], c(v), c(r));
        row("engine", |i| e[i].value);
        let (v_ft3, r_ft) = (v / FT.powi(3), r / FT);
        row("Schultz: 25 - 10lg r - 5lg V - 3lg f", |i| {
            25.0 - 10.0 * r_ft.log10() - 5.0 * v_ft3.log10() - 3.0 * OCTAVE_BANDS_HZ[i].log10()
        });
    }
}
