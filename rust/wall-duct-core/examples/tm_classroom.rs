//! TM 5-805-4 §7-6's worked example, element by element and end to end,
//! beside the engine's own figures. The regression test of the same name
//! holds the engine to it.
//!
//! `cargo run --example tm_classroom`

#[path = "../tests/common/tm_classroom.rs"]
mod tm;

use tm::*;
use wall_duct_core::autodiff::Dual;
use wall_duct_core::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};
use wall_duct_core::model::DuctSystem;
use wall_duct_core::physics::{
    duct_attenuation::segment_attenuation_db,
    elbow::{elbow_loss_db, ElbowKind},
    end_reflection::{end_reflection_loss_db, Termination},
    regenerative::terminal_pwl_db,
    room::receiver_spl,
    source::source_pwl,
};

type D = Dual<1>;

fn row(label: &str, v: impl Fn(usize) -> f64) {
    print!("{label:<30}");
    (0..7).for_each(|i| print!("{:>6.1}", v(i)));
    println!();
}

fn opt(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

fn main() {
    let c = D::constant;
    print!("{:<30}", "");
    OCTAVE_BANDS_HZ[..7].iter().for_each(|f| print!("{f:>6}"));
    println!();

    let mut il = [0.0; N_BANDS];
    let mut add = |loss: &dyn Fn(usize) -> f64| (0..N_BANDS).for_each(|i| il[i] += loss(i));

    for (name, l, w, h, round, tm) in LINED_RUNS {
        let e = segment_attenuation_db(c(l * FT), c(w * IN), c(h * IN), c(IN), round);
        println!("{name}, 1 in lining{}", if round { " (engine: as unlined)" } else { "" });
        row("  TM", |i| tm[i]);
        row("  engine", |i| e[i].value);
        add(&|i| e[i].value);
    }

    println!("4 lined elbows");
    let elbows: Vec<_> = ELBOWS_IN.iter().map(|w| elbow_loss_db(c(w * IN), ElbowKind::Vaned, true)).collect();
    row("  TM", |i| ELBOWS_TM[i]);
    row("  engine", |i| elbows.iter().map(|e| e[i].value).sum());
    add(&|i| elbows.iter().map(|e| e[i].value).sum());

    println!("3 branches");
    let division: f64 = BRANCH_RATIOS.iter().map(|r| -10.0 * r.log10()).sum();
    row("  TM", |i| BRANCHES_TM[i]);
    row("  engine", |_| division);
    add(&|_| division);

    println!("end reflection, 10 in");
    let erl = end_reflection_loss_db(c(10.0 * IN), Termination::Suspended);
    row("  TM part A (tile ceiling)", |i| ERL_TM[i]);
    row("  engine (suspended)", |i| erl[i].value);
    add(&|i| erl[i].value);

    println!("total insertion loss");
    row("  TM", |i| opt(TOTAL_IL_TM[i]));
    row("  engine", |i| il[i]);

    println!("diffuser Lw, 10 in at 500 cfm");
    let a = std::f64::consts::PI / 4.0 * (10.0 * IN).powi(2);
    let lw = terminal_pwl_db(c(500.0 * CFM / a), c(a));
    row("  TM (catalogue)", |i| opt(DIFFUSER_LW_TM[i]));
    row("  engine (CIBSE)", |i| lw[i].value);

    println!("room: Lp - Lw at 8 ft");
    let rel = receiver_spl(&[c(0.0); N_BANDS], c(ROOM_FT3 * FT.powi(3)), c(RECEIVER_FT * FT));
    row("  TM Rel SPL", |i| REL_SPL_TM[i]);
    row("  engine (Schultz)", |i| rel[i].value);

    println!("room Lp, one diffuser");
    row("  TM", |i| LP_ONE_TM[i]);
    row("  engine, fan less engine IL", |i| FAN_LW[i] - il[i] + rel[i].value);
    for (label, lw) in [("TM's catalogue diffuser", Some(DIFFUSER_LW)), ("CIBSE diffuser estimate", None)] {
        let sys = DuctSystem::from_params(&params(lw));
        println!("engine Lw after each block, {label}");
        let (mut q, mut pwl) = (sys.volume_flow(), source_pwl(sys.source_type, sys.source_offset));
        for (block, p) in sys.blocks.iter().zip(&sys.block_params) {
            q = block.flow_out(q);
            pwl = block.forward(&pwl, q);
            row(&format!("  {}", p.type_name()), |i| pwl[i].value);
        }
        let spl = sys.receiver_spl_values();
        row("  room Lp (with regen)", |i| spl[i]);
        println!("  NC {} (TM: 30 from one diffuser)", sys.nc());
    }
}
