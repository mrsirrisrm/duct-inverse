//! Map a path through the duct network onto `wall-duct-core` blocks.
//!
//! Element types come from `duct_heal::classify`; a fitting it cannot name is
//! passed through, and the report lists it for the classifier.

use duct_heal::classify::{classify_node, ElementType};
use duct_heal::{Network, NodeKind};
use wall_duct_core::model::BlockParams;
use wall_duct_core::physics::elbow::ElbowKind;
use wall_duct_core::physics::end_reflection::Termination;
use wall_duct_core::physics::plenum::Openings;

/// A bare sheet-metal plenum's wall absorption: the model does not say how
/// one is lined, and bare is the least loss.
pub const PLENUM_ALPHA: f64 = 0.05;

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Nearest decoded duct section along the path from `i`, looking downstream
/// first (a fitting's outlet sets its velocity), or upstream first for the
/// main a branch leaves.
/// A duct section: width, height and whether it is round (of diameter
/// width).
type Section = (f64, f64, bool);

fn section_toward(net: &Network, path: &[usize], i: usize, downstream: bool) -> Option<Section> {
    let sized = |&n: &usize| match net.nodes[n].kind {
        NodeKind::Duct { width, height, round, .. } => Some((width, height, round)),
        _ => None,
    };
    let down = || path[i..].iter().find_map(sized);
    let up = || path[..i].iter().rev().find_map(sized);
    if downstream { down().or_else(up) } else { up().or_else(down) }
}

fn section(net: &Network, path: &[usize], i: usize) -> Option<Section> {
    section_toward(net, path, i, true)
}

fn area((w, h, round): Section) -> f64 {
    if round { std::f64::consts::FRAC_PI_4 * w * w } else { w * h }
}

/// Where duct `n` meets an element at `centre`: its nearer end, and the
/// direction along it toward that end.
fn duct_end_toward(net: &Network, n: usize, centre: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let NodeKind::Duct { start, end, .. } = net.nodes[n].kind else { return None };
    let (far, near) = if dist(start, centre) < dist(end, centre) { (end, start) } else { (start, end) };
    let l = dist(far, near).max(1e-9);
    Some((near, [(near[0] - far[0]) / l, (near[1] - far[1]) / l, (near[2] - far[2]) / l]))
}

/// A plenum's openings, from the sections either side and where those ducts
/// end. Without both ends, the outlet is taken across the chamber's length,
/// in line.
fn plenum_openings(net: &Network, path: &[usize], i: usize, size: [f64; 3]) -> Option<Openings> {
    let (inlet, outlet) = (section_toward(net, path, i, false)?, section_toward(net, path, i + 1, true)?);
    let centre = net.nodes[path[i]].location?;
    let ends = (duct_end_toward(net, path[i - 1], centre), path.get(i + 1).and_then(|&m| duct_end_toward(net, m, centre)));
    let (distance_m, cos_theta) = match ends {
        (Some((a, axis)), Some((b, _))) if dist(a, b) > 0.05 => {
            let d = dist(a, b);
            (d, ((b[0] - a[0]) * axis[0] + (b[1] - a[1]) * axis[1] + (b[2] - a[2]) * axis[2]) / d)
        }
        _ => (size[0].max(size[1]).max(size[2]), 1.0),
    };
    Some(Openings { inlet_m2: area(inlet), inlet_span_m: inlet.0.max(inlet.1), outlet_m2: area(outlet), distance_m, cos_theta })
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
            // A path with no duct to size the main has none to carry its noise either.
            let (width_m, height_m, round) = section_toward(net, path, i, false).unwrap_or((10.0, 10.0, false));
            push(
                BlockParams::Branch { flow_ratio: flow[n] / flow[prev], width_m, height_m, round },
                format!("{:.0} -> {:.0} L/s", flow[prev] * 1000.0, flow[n] * 1000.0),
            );
        }
        let ty = classify_node(node);
        match node.kind {
            NodeKind::Duct { start, end, width, height, round } => push(
                BlockParams::Segment { length_m: dist(start, end), width_m: width, height_m: height, lining_m: 0.0, round },
                "duct".into(),
            ),
            // Undecoded straights and flex: length from the elements either side.
            NodeKind::DuctNoGeometry | NodeKind::Undecoded => {
                let ends = (net.nodes[prev].location, path.get(i + 1).and_then(|&m| net.nodes[m].location));
                if let ((Some(a), Some(b)), Some((w, h, round))) = (ends, section(net, path, i)) {
                    push(
                        BlockParams::Segment { length_m: dist(a, b).max(0.01), width_m: w, height_m: h, lining_m: 0.0, round },
                        "duct, length from neighbours".into(),
                    );
                }
            }
            NodeKind::Fitting
                if matches!(ty, Some(ElementType::ElbowMitred | ElementType::ElbowMitredVanes | ElementType::ElbowRadius)) =>
            {
                if let Some((w, h, round)) = section(net, path, i) {
                    let kind = match ty {
                        Some(ElementType::ElbowMitred) => ElbowKind::Mitred,
                        Some(ElementType::ElbowMitredVanes) => ElbowKind::Vaned,
                        _ => ElbowKind::Radiused,
                    };
                    push(BlockParams::Bend { width_m: w, height_m: h, kind, lined: false, round }, node.family.clone());
                }
            }
            // Dampers as found, open; a VAV box's modulating, taken 15° closed.
            NodeKind::Fitting | NodeKind::Equipment
                if matches!(ty, Some(ElementType::VolumeDamper | ElementType::FireSmokeDamper | ElementType::VavTerminalUnit)) =>
            {
                if let Some((w, h, round)) = section(net, path, i) {
                    let closed_deg = if ty == Some(ElementType::VavTerminalUnit) { 15.0 } else { 0.0 };
                    push(BlockParams::Damper { width_m: w, height_m: h, closed_deg, round }, node.family.clone());
                }
            }
            // A silencer from the IFC: its length is its longest extent. The
            // engine's attenuator is a rectangular splitter unit; the free
            // area and splitter are stand-ins until the model carries data.
            NodeKind::Fitting if ty == Some(ElementType::Silencer) => {
                let length = node.ifc.as_ref().and_then(|i| i.size).map(|s| s[0].max(s[1]).max(s[2]));
                // A round section's square stands in for it.
                if let (Some(length_m), Some((w, h, _))) = (length, section(net, path, i)) {
                    push(
                        BlockParams::Attenuator { length_m, width_m: w, height_m: h, free_area_ratio: 0.5, splitter_m: 0.1 },
                        format!("{}, {:.2} m", node.family, length_m),
                    );
                }
            }
            // A plenum with its extent from the IFC; a Revit one alone has none.
            NodeKind::Fitting | NodeKind::Equipment if ty == Some(ElementType::Plenum) => {
                let size = node.ifc.as_ref().and_then(|i| i.size);
                if let Some(size) = size {
                    if let Some(openings) = plenum_openings(net, path, i, size) {
                        let mut s = size;
                        s.sort_by(|a, b| b.total_cmp(a));
                        push(
                            BlockParams::Plenum { length_m: s[0], width_m: s[1], height_m: s[2], openings, alpha: [PLENUM_ALPHA; 8] },
                            format!("{}, {:.2} m apart", node.family, openings.distance_m),
                        );
                    }
                }
            }
            NodeKind::Terminal => {
                if let Some((w, h, round)) = section(net, path, i) {
                    // Its ceiling is not in the model: flush, the smaller end
                    // reflection, is the conservative choice.
                    let block = BlockParams::Diffuser { width_m: w, height_m: h, round, lw_db: None, termination: Termination::Flush };
                    push(block, format!("{}, neck from last duct", node.family));
                }
            }
            // Takeoffs and tees are the Branch above; transitions are not
            // modelled yet.
            _ => {}
        }
    }
    out
}
