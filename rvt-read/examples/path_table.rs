//! Print the critical supply path as a downduct table.
//!
//!     cargo run --example path_table -- <elements.json>
use duct_heal::{heal, path, rvt_json, NodeKind};

fn main() {
    let file = std::env::args().nth(1).expect("usage: path_table <elements.json>");
    let json = serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap();
    let mut net = rvt_json::parse(&json);
    heal::join_takeoffs(&mut net);

    let cp = path::critical_path(&net).expect("no unit with a supply system and terminal flows");
    let unit = &net.nodes[cp.tree.root];
    println!(
        "{} [{}], {}: unit Air Flow {} L/s, terminals sum to {:.0} L/s",
        unit.family,
        unit.id,
        cp.system,
        unit.flow.map_or("-".into(), |q| format!("{:.0}", q * 1000.0)),
        cp.flow[cp.tree.root] * 1000.0
    );
    println!("{:>4}  {:<10} {:<46} {:>11} {:>7} {:>7} {:>6}", "#", "kind", "family", "size mm", "len m", "L/s", "m/s");
    for (i, &n) in cp.path.iter().enumerate() {
        let node = &net.nodes[n];
        let (size, len, v) = match node.kind {
            NodeKind::Duct { start, end, width, height, .. } => {
                let l = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2) + (end[2] - start[2]).powi(2)).sqrt();
                (
                    format!("{:.0}x{:.0}", width * 1000.0, height * 1000.0),
                    format!("{l:.2}"),
                    format!("{:.1}", cp.flow[n] / (width * height)),
                )
            }
            _ => (String::new(), String::new(), String::new()),
        };
        let kind = format!("{:?}", node.kind);
        let kind = kind.split([' ', '{']).next().unwrap();
        let family: String = node.family.chars().take(46).collect();
        println!("{i:>4}  {kind:<10} {family:<46} {size:>11} {len:>7} {:>7.0} {v:>6}", cp.flow[n] * 1000.0);
    }
}
