//! Lining and insulation elements in a Revit file: how many, their decoded
//! fields and saved parameters (`probe_lining <model.rvt>`).
use rvt::walker::{self, DecodedElement, WalkerLimits, PRODUCTION_ELEMENT_MIN_SCORE};
use rvt::RevitFile;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: probe_lining <model.rvt>");
    let mut file = RevitFile::open_bytes(std::fs::read(path).unwrap()).unwrap();
    let elements: Vec<DecodedElement> =
        walker::iter_elements_with_control(&mut file, PRODUCTION_ELEMENT_MIN_SCORE, WalkerLimits::default(), &rvt::control::WalkerControl::new())
            .unwrap()
            .collect();
    let mut classes: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &elements {
        *classes.entry(e.class.as_str()).or_default() += 1;
    }
    for (c, n) in &classes {
        if c.contains("Lining") || c.contains("Insulation") || c.starts_with("RbsDuct") || c.contains("Duct") {
            println!("{c}: {n}");
        }
    }
    let wanted: Vec<DecodedElement> = elements.into_iter().filter(|e| e.class.contains("Lining") || e.class.contains("Insulation")).take(3).collect();
    let saved = rvt::elements::parameters::saved_parameters_lossy(&mut file, &wanted).value;
    for e in &wanted {
        println!("--- {} {:?}", e.class, e.id);
        for (name, v) in &e.fields {
            println!("  field {name}: {v:?}");
        }
        for p in saved.entries_for(e) {
            println!("  param {p}");
        }
    }
}
