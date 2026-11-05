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
/// within 1 dB in every band; every room rates the same, with the same
/// issues: the dampers, silencers and ducts the Revit file gives are the
/// ones the IFC adds nothing to.
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
    assert_eq!(kinds(a).len(), 16);
    // Every room the 15 unit systems serve rates the same.
    let rooms = |v: &Value| v["rooms"].as_array().unwrap().iter().map(|r| (r["dominant"]["terminal"].clone(), r["nc"].clone())).collect::<Vec<_>>();
    assert_eq!(rooms(&alone), rooms(&paired));
    assert_eq!((a["units_rated"].as_u64(), a["terminals_rated"].as_u64()), (Some(15), Some(87)));
    let bands = |v: &Value| v["receiver_spl_db"].as_array().unwrap().iter().map(|b| b.as_f64().unwrap()).collect::<Vec<_>>();
    for (x, y) in bands(a).iter().zip(bands(p)) {
        assert!((x - y).abs() < 1.0, "{:?} alone, {:?} with the IFC", bands(a), bands(p));
    }
    // The same issues: the 15 terminals no unit feeds, and the unit's sound
    // power estimated.
    let issues = |v: &Value| v["issues"].as_array().unwrap().iter().map(|i| (i["kind"].clone(), i["elements"].as_array().unwrap().len())).collect::<Vec<_>>();
    assert_eq!(issues(&alone), issues(&paired));
    assert_eq!(issues(&alone), [("terminals_unserved".into(), 15), ("unit_sound_power_estimated".into(), 1)]);
}
