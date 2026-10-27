//! Print the path from a unit to its farthest terminal as a downduct table.
//!
//!     cargo run --example path_table -- <elements.json>
//!
//! Picks the unit and supply system serving the most terminals with a flow.
use duct_heal::{heal, path, rvt_json, NodeKind};

fn main() {
    let file = std::env::args().nth(1).expect("usage: path_table <elements.json>");
    let json = serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap();
    let mut net = rvt_json::parse(&json);
    heal::join_takeoffs(&mut net);

    let served = |tree: &path::Tree| -> Vec<usize> {
        tree.order
            .iter()
            .copied()
            .filter(|&n| net.nodes[n].kind == NodeKind::Terminal && net.nodes[n].flow.is_some_and(|q| q > 0.0))
            .collect()
    };
    let (tree, system) = net
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NodeKind::Equipment)
        .flat_map(|(e, n)| n.systems.iter().filter(|s| s.contains("Supply")).map(move |s| (e, s.clone())))
        .map(|(e, s)| (path::tree(&net, e, &s), s))
        .max_by_key(|(tree, _)| served(tree).len())
        .expect("no unit with a supply system");
    let terminals = served(&tree);
    let farthest = *terminals.iter().max_by_key(|&&t| tree.path_to(t).unwrap().len()).expect("no terminals");
    let flow = tree.flows(&net);
    let unit = &net.nodes[tree.root];
    println!(
        "{} [{}], {system}: {} terminals; unit Air Flow {} L/s, terminals sum to {:.0} L/s",
        unit.family,
        unit.id,
        terminals.len(),
        unit.flow.map_or("-".into(), |q| format!("{:.0}", q * 1000.0)),
        flow[tree.root] * 1000.0
    );
    println!("{:>4}  {:<10} {:<46} {:>11} {:>7} {:>7} {:>6}", "#", "kind", "family", "size mm", "len m", "L/s", "m/s");
    for (i, &n) in tree.path_to(farthest).unwrap().iter().enumerate() {
        let node = &net.nodes[n];
        let (size, len, v) = match node.kind {
            NodeKind::Duct { start, end, width, height } => {
                let l = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2) + (end[2] - start[2]).powi(2)).sqrt();
                (
                    format!("{:.0}x{:.0}", width * 1000.0, height * 1000.0),
                    format!("{l:.2}"),
                    format!("{:.1}", flow[n] / (width * height)),
                )
            }
            _ => (String::new(), String::new(), String::new()),
        };
        let kind = format!("{:?}", node.kind);
        let kind = kind.split([' ', '{']).next().unwrap();
        let family: String = node.family.chars().take(46).collect();
        println!("{i:>4}  {kind:<10} {family:<46} {size:>11} {len:>7} {:>7.0} {v:>6}", flow[n] * 1000.0);
    }
}
