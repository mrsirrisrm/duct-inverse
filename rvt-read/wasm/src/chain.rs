//! Map a path through the duct network onto `wall-duct-core` blocks.
//!
//! Element types come from family names by keyword here; in the tool that is
//! the classifier's job (`eval/`).

use duct_heal::{Network, NodeKind};
use wall_duct_core::model::BlockParams;

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Nearest decoded duct section along the path from `i`, looking downstream
/// first (a fitting's outlet sets its velocity).
fn section(net: &Network, path: &[usize], i: usize) -> Option<(f64, f64)> {
    let sized = |&n: &usize| match net.nodes[n].kind {
        NodeKind::Duct { width, height, .. } => Some((width, height)),
        _ => None,
    };
    path[i..].iter().find_map(sized).or_else(|| path[..i].iter().rev().find_map(sized))
}

/// A block and the element it came from, for the report.
pub struct Mapped {
    pub block: BlockParams,
    pub element: i64,
    pub note: String,
}

/// The chain for `path`. A branch is inserted wherever the flow drops between
/// neighbouring elements: takeoffs join mid-duct, so a branch can leave a
/// duct as well as a fitting.
pub fn chain(net: &Network, path: &[usize], flow: &[f64]) -> Vec<Mapped> {
    let mut out = Vec::new();
    for (i, &n) in path.iter().enumerate().skip(1) {
        let node = &net.nodes[n];
        let prev = path[i - 1];
        let mut push = |block, note: String| out.push(Mapped { block, element: node.id, note });
        if flow[n] < flow[prev] * (1.0 - 1e-9) {
            push(
                BlockParams::Branch { flow_ratio: flow[n] / flow[prev] },
                format!("{:.0} -> {:.0} L/s", flow[prev] * 1000.0, flow[n] * 1000.0),
            );
        }
        let family = node.family.to_ascii_lowercase();
        match node.kind {
            NodeKind::Duct { start, end, width, height } => push(
                BlockParams::Segment { length_m: dist(start, end), width_m: width, height_m: height, lining_m: 0.0 },
                "duct".into(),
            ),
            // Undecoded straights and flex: length from the elements either side.
            NodeKind::DuctNoGeometry | NodeKind::Undecoded => {
                let ends = (net.nodes[prev].location, path.get(i + 1).and_then(|&m| net.nodes[m].location));
                if let ((Some(a), Some(b)), Some((w, h))) = (ends, section(net, path, i)) {
                    push(
                        BlockParams::Segment { length_m: dist(a, b).max(0.01), width_m: w, height_m: h, lining_m: 0.0 },
                        "duct, length from neighbours".into(),
                    );
                }
            }
            NodeKind::Fitting if family.contains("elbow") || family.contains("bend") => {
                if let Some((w, h)) = section(net, path, i) {
                    push(BlockParams::Bend { width_m: w, height_m: h }, node.family.clone());
                }
            }
            NodeKind::Terminal => {
                if let Some((w, h)) = section(net, path, i) {
                    push(BlockParams::Diffuser { width_m: w, height_m: h }, format!("{}, neck from last duct", node.family));
                }
            }
            // Takeoffs, tees and transitions: division is the Branch above;
            // their own loss and regen are not modelled yet.
            _ => {}
        }
    }
    out
}
