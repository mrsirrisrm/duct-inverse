//! `duct-heal <elements.json>`: heal an `rvt-elements` dump and report how
//! connected the duct network is before and after.

use std::collections::HashSet;

use duct_heal::{graph, heal, rvt_json, Network, NodeKind};

fn summary(label: &str, net: &Network) {
    let island = graph::islands(net);
    let count = island.iter().collect::<HashSet<_>>().len();
    let with_equipment: HashSet<usize> = net
        .nodes
        .iter()
        .zip(&island)
        .filter(|(n, _)| n.kind == NodeKind::Equipment)
        .map(|(_, &i)| i)
        .collect();
    let terminals: Vec<usize> = net
        .nodes
        .iter()
        .zip(&island)
        .filter(|(n, _)| n.kind == NodeKind::Terminal)
        .map(|(_, &i)| i)
        .collect();
    let served = terminals.iter().filter(|i| with_equipment.contains(i)).count();
    println!(
        "{label}: {} nodes, {} edges, {count} islands; {served} of {} terminals share an island with equipment",
        net.nodes.len(),
        net.edges.len(),
        terminals.len(),
    );
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: duct-heal <elements.json>");
    let json = serde_json::from_slice(&std::fs::read(&path).expect("read input")).expect("parse JSON");
    let mut net = rvt_json::parse(&json);
    let undecoded = net.nodes.iter().filter(|n| n.kind == NodeKind::Undecoded).count();
    println!("{undecoded} undecoded elements joined by id (flex ducts, mostly)");
    summary("recorded", &net);

    let takeoffs = heal::join_takeoffs(&mut net);
    println!(
        "takeoffs: {} found, {} joined to a main ({} with more than one candidate), {} unmatched",
        takeoffs.takeoffs,
        takeoffs.joined,
        takeoffs.ambiguous,
        takeoffs.unmatched.len(),
    );
    summary("healed", &net);

    let island = graph::islands(&net);
    for r in graph::equipment_reach(&net, &island).iter().filter(|r| r.terminals > 0) {
        let e = &net.nodes[r.equipment];
        println!(
            "  {} [{}]: {} terminals, {:.0} L/s",
            e.family,
            e.id,
            r.terminals,
            r.terminal_flow * 1000.0
        );
    }
}
