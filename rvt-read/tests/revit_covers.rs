//! Revit's insulation, read from the `.rvt`, against Revit's own IFC export
//! of the same model: the BIM4LCA housing's HVAC (Granlund). Needs the pair
//! in `DUCT_SAMPLES` (the repo's `samples/`, never committed); skips without.
#![cfg(feature = "rvt")]

use duct_heal::{ifc, rvt_native};

#[test]
fn insulation_matches_the_ifc() {
    let Ok(dir) = std::env::var("DUCT_SAMPLES") else {
        eprintln!("DUCT_SAMPLES unset: skipped");
        return;
    };
    let stem = format!("{dir}/HVAC_NordicLCA_Housing_Concrete_BuildingPermit");
    let (Ok(rvt), Ok(ifc_bytes)) = (std::fs::read(format!("{stem}.rvt")), std::fs::read(format!("{stem}.ifc"))) else {
        eprintln!("Granlund pair not in {dir}: skipped");
        return;
    };
    let mut net = rvt_native::network(rvt, &rvt::control::WalkerControl::new()).expect("network");
    ifc::merge(&mut net, &ifc::parse(&ifc_bytes));
    // Over the Revit elements the export tags: insulated in one, insulated in the other.
    let (mut both, mut differ) = (0, Vec::new());
    for n in net.nodes.iter().filter(|n| n.id > 0) {
        let Some(info) = &n.ifc else { continue };
        let covers = net.covers.get(&n.id).copied().unwrap_or_default();
        match (covers.insulated, info.insulated) {
            (true, true) => both += 1,
            (false, false) => {}
            _ => differ.push(n.id),
        }
    }
    assert!(differ.is_empty(), "insulated in one only: {differ:?}");
    assert!(both > 100, "{both} insulated");
    assert!(net.covers.values().all(|c| c.lining_m.is_none()), "no lining in this model");
}
