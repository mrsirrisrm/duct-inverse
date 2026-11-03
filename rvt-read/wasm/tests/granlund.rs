//! A Revit file rated on its own, against the same file with Revit's IFC
//! export of it: the BIM4LCA housing's HVAC (Granlund, Revit 2024). Needs the
//! pair in `DUCT_SAMPLES` (the repo's `samples/`, never committed); skips
//! without.

use serde_json::Value;

fn analyse(rvt: &[u8], ifc: Option<&[u8]>) -> Value {
    let control = rvt::control::WalkerControl::new();
    duct_pipeline::analyse_bytes(Some(rvt.to_vec()), ifc, None, Default::default(), &control).expect("analyse")
}

/// The worst path is the same, through the same blocks, and its receiver
/// within 1 dB in every band: the dampers, silencers and ducts the Revit file
/// gives are the ones the IFC adds nothing to.
#[test]
fn revit_alone_rates_as_with_its_ifc() {
    let Ok(dir) = std::env::var("DUCT_SAMPLES") else {
        eprintln!("DUCT_SAMPLES unset: skipped");
        return;
    };
    let stem = format!("{dir}/HVAC_NordicLCA_Housing_Concrete_BuildingPermit");
    let (Ok(rvt), Ok(ifc)) = (std::fs::read(format!("{stem}.rvt")), std::fs::read(format!("{stem}.ifc"))) else {
        eprintln!("Granlund pair not in {dir}: skipped");
        return;
    };
    let (alone, paired) = (analyse(&rvt, None), analyse(&rvt, Some(&ifc)));
    let (a, p) = (&alone["path"], &paired["path"]);
    assert_eq!(a["terminal"]["id"], p["terminal"]["id"]);
    assert_eq!(a["nc"], p["nc"]);
    let kinds = |v: &Value| v["blocks"].as_array().unwrap().iter().map(|b| b["type"].clone()).collect::<Vec<_>>();
    assert_eq!(kinds(a), kinds(p));
    assert_eq!(kinds(a).len(), 27);
    let bands = |v: &Value| v["receiver_spl_db"].as_array().unwrap().iter().map(|b| b.as_f64().unwrap()).collect::<Vec<_>>();
    for (x, y) in bands(a).iter().zip(bands(p)) {
        assert!((x - y).abs() < 1.0, "{:?} alone, {:?} with the IFC", bands(a), bands(p));
    }
}
