//! Map a path through the duct network onto `wall-duct-core` blocks.
//!
//! Element types come from `duct_heal::classify`; a fitting it cannot name is
//! passed through, and the report lists it for the classifier.

use duct_heal::classify::{classify_node, ElementType};
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
        let ty = classify_node(node);
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
            NodeKind::Fitting
                if matches!(ty, Some(ElementType::ElbowMitred | ElementType::ElbowMitredVanes | ElementType::ElbowRadius)) =>
            {
                if let Some((w, h)) = section(net, path, i) {
                    push(BlockParams::Bend { width_m: w, height_m: h }, node.family.clone());
                }
            }
            // A silencer from the IFC: its length is its longest extent. The
            // engine's attenuator is a rectangular splitter unit; the free
            // area and splitter are stand-ins until the model carries data.
            NodeKind::Fitting if ty == Some(ElementType::Silencer) => {
                let length = node.ifc.as_ref().and_then(|i| i.size).map(|s| s[0].max(s[1]).max(s[2]));
                if let (Some(length_m), Some((w, h))) = (length, section(net, path, i)) {
                    push(
                        BlockParams::Attenuator { length_m, width_m: w, height_m: h, free_area_ratio: 0.5, splitter_m: 0.1 },
                        format!("{}, {:.2} m", node.family, length_m),
                    );
                }
            }
            NodeKind::Terminal => {
                if let Some((w, h)) = section(net, path, i) {
                    push(BlockParams::Diffuser { width_m: w, height_m: h }, format!("{}, neck from last duct", node.family));
                }
            }
            // Takeoffs, tees and transitions: division is the Branch above;
            // their own loss and regen are not modelled yet, nor are dampers
            // (the report warns).
            _ => {}
        }
    }
    out
}
