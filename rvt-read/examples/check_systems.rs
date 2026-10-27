//! Agreement between takeoff-axis joins and the system names Revit recorded.
use duct_heal::{heal, rvt_json, EdgeSource};

fn main() {
    let path = std::env::args().nth(1).expect("usage: check_systems <elements.json>");
    let json = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut net = rvt_json::parse(&json);
    heal::join_takeoffs(&mut net);
    let (mut agree, mut differ, mut unknown) = (0, 0, 0);
    for e in net.edges.iter().filter(|e| matches!(e.source, EdgeSource::TakeoffAxis { .. })) {
        let (a, b) = (&net.nodes[e.a].systems, &net.nodes[e.b].systems);
        if a.is_empty() || b.is_empty() {
            unknown += 1;
        } else if a.iter().any(|s| b.contains(s)) {
            agree += 1;
        } else {
            differ += 1;
            println!("  differ: {:?} -> {:?}", a, b);
        }
    }
    println!("system names: {agree} agree, {differ} differ, {unknown} unnamed on one side");
}
