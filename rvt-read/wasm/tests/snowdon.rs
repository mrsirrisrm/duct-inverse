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
    let loudest = r["rooms"][0]["nc"].as_i64().unwrap();
    assert_eq!(terminals.iter().filter_map(|t| t["room_nc"].as_i64()).max(), Some(loudest));
    // Its flows do not balance: rooms rated through ducts over 20 m/s are
    // reported first, and the worst room is not one of them.
    let issues = r["issues"].as_array().unwrap();
    let kinds: Vec<&str> = issues.iter().map(|i| i["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds[0], "room_flow_implausible");
    let doubtful = r["rooms"].as_array().unwrap().iter().filter(|r| r["flows_implausible"] == true).count();
    assert!(doubtful > 0 && issues[0]["message"].as_str().unwrap().starts_with(&format!("{doubtful} rooms")));
    let room = r["rooms"].as_array().unwrap().iter().find(|r| r["flows_implausible"] == false).unwrap();
    assert_eq!(r["path"]["room"]["space"], room["space"]);
    let mismatch: &Value = issues.iter().find(|i| i["kind"] == "unit_flow_mismatch").unwrap();
    assert_eq!(mismatch["details"].as_array().unwrap().len(), mismatch["elements"].as_array().unwrap().len());
}
