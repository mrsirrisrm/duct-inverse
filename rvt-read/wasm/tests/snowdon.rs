//! Autodesk's Snowdon Towers HVAC sample (Revit 2024), whose grilles sit on
//! its ducts' sides and whose diffusers hang from them on flex: tapped in,
//! neither records a connection at the duct. Needs the file in `DUCT_SAMPLES`
//! (the repo's `samples/`, never committed); skips without.

use serde_json::Value;

#[test]
fn terminals_on_ducts_and_on_flex_reach_their_units() {
    let Ok(dir) = std::env::var("DUCT_SAMPLES") else {
        eprintln!("DUCT_SAMPLES unset: skipped");
        return;
    };
    let Ok(rvt) = std::fs::read(format!("{dir}/Snowdon Towers Sample HVAC.rvt")) else {
        eprintln!("Snowdon not in {dir}: skipped");
        return;
    };
    let control = rvt::control::WalkerControl::new();
    let r = duct_pipeline::analyse_bytes(Some(rvt), None, None, Default::default(), &control).expect("analyse");
    assert_eq!(r["terminals_on_ducts"], 406);
    assert_eq!(r["runs_on_ducts"], 18);
    assert_eq!((r["healed"]["terminals"].as_u64(), r["healed"]["terminals_with_equipment"].as_u64()), (Some(509), Some(508)));
    // Every rated terminal, each with its room's rating; the worst room's among them.
    let terminals = r["terminals"].as_array().unwrap();
    assert_eq!(terminals.len() as u64, r["path"]["terminals_rated"].as_u64().unwrap());
    assert!(terminals.iter().all(|t| t["room_nc"].is_i64() && t["unit"].is_i64()));
    let worst = r["rooms"][0]["nc"].as_i64().unwrap();
    assert_eq!(terminals.iter().filter_map(|t| t["room_nc"].as_i64()).max(), Some(worst));
    // Its flows do not balance, and the worst room says so first.
    let issues = r["issues"].as_array().unwrap();
    let kinds: Vec<&str> = issues.iter().map(|i| i["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["worst_room_flow_mismatch", "terminals_unserved", "unit_flow_mismatch", "duct_velocity"]);
    let mismatch: &Value = &issues[2];
    assert_eq!(mismatch["details"].as_array().unwrap().len(), mismatch["elements"].as_array().unwrap().len());
}
