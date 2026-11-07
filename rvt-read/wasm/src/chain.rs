//! Map a path through the duct network onto `wall-duct-core` blocks.
//!
//! Element types come from `duct_heal::classify`; a fitting it cannot name is
//! passed through, and the report lists it for the classifier.

use duct_heal::classify::{classify_node, ElementType};
use duct_heal::{Network, NodeKind};
use wall_duct_core::model::BlockParams;
use wall_duct_core::physics::elbow::ElbowKind;
use wall_duct_core::physics::branch::{Junction, Outlets};
use wall_duct_core::physics::end_reflection::Termination;
use wall_duct_core::physics::plenum::Openings;

/// A bare sheet-metal plenum's wall absorption: the model does not say how
/// one is lined, and bare is the least loss.
pub const PLENUM_ALPHA: f64 = 0.05;

/// How a bend's regenerated noise is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BendRegen {
    /// CIBSE's detailed method where the model records a loss coefficient,
    /// else Woods'.
    #[default]
    Recorded,
    /// CIBSE's detailed method always, at a typical loss coefficient for the
    /// elbow's kind where none is recorded.
    Cibse,
    /// Woods' method always.
    Woods,
}

impl std::str::FromStr for BendRegen {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "recorded" => Ok(Self::Recorded),
            "cibse" => Ok(Self::Cibse),
            "woods" => Ok(Self::Woods),
            _ => Err(format!("bend regen: \"recorded\", \"cibse\" or \"woods\", not {s:?}")),
        }
    }
}

/// A 90° elbow's typical loss coefficient: rough textbook figures for a
/// square section, not any one table's.
pub fn typical_zeta(kind: ElbowKind) -> f64 {
    match kind {
        ElbowKind::Mitred => 1.2,
        ElbowKind::Vaned => 0.3,
        ElbowKind::Radiused => 0.25,
    }
}

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

/// The section of the first duct at or below `n` in the tree.
fn section_below(net: &Network, children: &[Vec<usize>], n: usize) -> Option<Section> {
    let mut queue = std::collections::VecDeque::from([n]);
    while let Some(m) = queue.pop_front() {
        if let NodeKind::Duct { width, height, round, .. } = net.nodes[m].kind {
            return Some((width, height, round));
        }
        queue.extend(&children[m]);
    }
    None
}

/// The outlets of `junction` that carry air, by the first duct below each,
/// `branch` the one the path takes: `None` if any has no duct to size it.
fn outlets(net: &Network, children: &[Vec<usize>], flow: &[f64], junction: usize, branch: usize) -> Option<Outlets> {
    let mut total_m2 = 0.0;
    for &c in children[junction].iter().filter(|&&c| flow[c] > 0.0) {
        total_m2 += area(section_below(net, children, c)?);
    }
    Some(Outlets { branch_m2: area(section_below(net, children, branch)?), total_m2 })
}

/// A damper, elbow or earlier take-off within 5 main diameters upstream of
/// `path[i]`, whose noise the junction there adds to (ASHRAE). The junction's
/// own tee, just before it, does not count.
fn turbulent_upstream(net: &Network, path: &[usize], flow: &[f64], i: usize, main_diameter_m: f64) -> bool {
    let mut walked = 0.0;
    for j in (1..i).rev() {
        let node = &net.nodes[path[j]];
        if flow[path[j]] < flow[path[j - 1]] * (1.0 - 1e-9) {
            return true;
        }
        match node.kind {
            NodeKind::Duct { start, end, .. } => walked += dist(start, end),
            NodeKind::Fitting => match classify_node(node) {
                Some(ElementType::TeeOrWye | ElementType::BranchTakeoff) if j == i - 1 => {}
                Some(
                    ElementType::ElbowMitred
                    | ElementType::ElbowMitredVanes
                    | ElementType::ElbowRadius
                    | ElementType::TeeOrWye
                    | ElementType::BranchTakeoff
                    | ElementType::VolumeDamper
                    | ElementType::FireSmokeDamper
                    | ElementType::VavTerminalUnit,
                ) => return true,
                _ => {}
            },
            _ => {}
        }
        if walked > 5.0 * main_diameter_m {
            return false;
        }
    }
    false
}

/// What sets the noise of the junction at `path[i]`. The outlet carrying the
/// most is the main carrying on; of the others the largest is its side
/// branch, and a path that takes neither is on a side branch of its own.
fn junction_at(net: &Network, path: &[usize], flow: &[f64], children: &[Vec<usize>], i: usize, main: Option<Section>) -> Junction {
    let (prev, n) = (path[i - 1], path[i]);
    let side = children[prev].iter().copied().filter(|&c| c != n && flow[c] > 0.0).max_by(|&a, &b| flow[a].total_cmp(&flow[b]));
    let (side_m2, side_flow_ratio) = match side.filter(|&s| flow[s] < flow[n]) {
        Some(s) => (section_below(net, children, s).map(area), Some(flow[s] / flow[prev])),
        None => (section_below(net, children, n).map(area), None),
    };
    let turbulent = main.is_some_and(|m| turbulent_upstream(net, path, flow, i, (4.0 * area(m) / std::f64::consts::PI).sqrt()));
    Junction { side_m2, side_flow_ratio, turbulent }
}

/// A block and the element it came from, for the report.
pub struct Mapped {
    pub block: BlockParams,
    pub element: i64,
    pub note: String,
}

/// The chain for `path`. A branch is inserted wherever the flow drops between
/// neighbouring elements: takeoffs join mid-duct, so a branch can leave a
/// duct as well as a fitting. `children` is the path's tree's, by node.
pub fn chain(net: &Network, path: &[usize], flow: &[f64], children: &[Vec<usize>], bend_regen: BendRegen) -> Vec<Mapped> {
    let mut out = Vec::new();
    for (i, &n) in path.iter().enumerate().skip(1) {
        let node = &net.nodes[n];
        let prev = path[i - 1];
        let mut push = |block, note: String| out.push(Mapped { block, element: node.id, note });
        if flow[n] < flow[prev] * (1.0 - 1e-9) {
            // A path with no duct to size the main has none to carry its noise either.
            let main = section_toward(net, path, i, false);
            let (width_m, height_m, round) = main.unwrap_or((10.0, 10.0, false));
            let outlets = main.and(outlets(net, children, flow, prev, n));
            let by = match outlets {
                Some(o) => format!("by area {:.0}% of {:.2} m2", 100.0 * o.branch_m2 / o.total_m2, o.total_m2),
                None => "by flow".into(),
            };
            push(
                BlockParams::Branch { flow_ratio: flow[n] / flow[prev], width_m, height_m, round, outlets, junction: junction_at(net, path, flow, children, i, main) },
                format!("{:.0} -> {:.0} L/s, {by}", flow[prev] * 1000.0, flow[n] * 1000.0),
            );
        }
        let ty = classify_node(node);
        let covers = net.covers.get(&node.id).copied().unwrap_or_default();
        let wrapped = covers.insulated || node.ifc.as_ref().is_some_and(|i| i.insulated);
        let lining_m = covers.lining_m.unwrap_or(0.0);
        let covered = |what: &str| match (lining_m > 0.0, wrapped) {
            (true, true) => format!("{what}, lined {:.0} mm, insulated", lining_m * 1000.0),
            (true, false) => format!("{what}, lined {:.0} mm", lining_m * 1000.0),
            (false, true) => format!("{what}, insulated"),
            (false, false) => what.into(),
        };
        match node.kind {
            NodeKind::Duct { start, end, width, height, round } => push(
                BlockParams::Segment { length_m: dist(start, end), width_m: width, height_m: height, lining_m, round, wrapped },
                covered("duct"),
            ),
            // Undecoded straights and flex: length from the elements either side.
            NodeKind::DuctNoGeometry | NodeKind::Undecoded => {
                let ends = (net.nodes[prev].location, path.get(i + 1).and_then(|&m| net.nodes[m].location));
                if let ((Some(a), Some(b)), Some((w, h, round))) = (ends, section(net, path, i)) {
                    push(
                        BlockParams::Segment { length_m: dist(a, b).max(0.01), width_m: w, height_m: h, lining_m, round, wrapped },
                        covered("duct, length from neighbours"),
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
                    let recorded = node.ifc.as_ref().and_then(|i| i.loss_coefficient);
                    let (zeta, note) = match (bend_regen, recorded) {
                        (BendRegen::Woods, _) | (BendRegen::Recorded, None) => (None, node.family.clone()),
                        (_, Some(z)) => (Some(z), format!("{}, ζ {z:.2}", node.family)),
                        (BendRegen::Cibse, None) => {
                            let z = typical_zeta(kind);
                            (Some(z), format!("{}, ζ {z:.2} typical", node.family))
                        }
                    };
                    let lined = lining_m > 0.0;
                    let note = if lined { format!("{note}, lined") } else { note };
                    push(BlockParams::Bend { width_m: w, height_m: h, kind, lined, round, zeta }, note);
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
            // A silencer: its length is its longest extent, the IFC's, else
            // its Revit box's (overstated for one off the model's axes). The
            // engine's attenuator is a rectangular splitter unit; the free
            // area and splitter are stand-ins until the model carries data.
            NodeKind::Fitting if ty == Some(ElementType::Silencer) => {
                let length = node.ifc.as_ref().and_then(|i| i.size).or(node.extent).map(|s| s[0].max(s[1]).max(s[2]));
                // A round section's square stands in for it.
                if let (Some(length_m), Some((w, h, _))) = (length, section(net, path, i)) {
                    push(
                        BlockParams::Attenuator { length_m, width_m: w, height_m: h, free_area_ratio: 0.5, splitter_m: 0.1 },
                        format!("{}, {:.2} m", node.family, length_m),
                    );
                }
            }
            // A plenum with its extent from the IFC, else its Revit box.
            NodeKind::Fitting | NodeKind::Equipment if ty == Some(ElementType::Plenum) => {
                let size = node.ifc.as_ref().and_then(|i| i.size).or(node.extent);
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

#[cfg(test)]
mod tests {
    use super::*;
    use duct_heal::{Covers, EdgeSource, Node};

    /// Revit's lining on a duct and a bend reaches their blocks; its
    /// insulation on the duct wraps it.
    #[test]
    fn revit_covers() {
        let node = |id, kind, family: &str| Node {
            id,
            kind,
            family: family.into(),
            type_name: String::new(),
            systems: vec!["Supply Air".into()],
            location: None,
            flow: Some(0.2),
            ifc: None,
            description: String::new(),
            extent: None,
        };
        let duct = |id, x: f64| {
            node(id, NodeKind::Duct { start: [x, 0.0, 0.0], end: [x + 2.0, 0.0, 0.0], width: 0.4, height: 0.3, round: false }, "")
        };
        let mut net = Network::default();
        let path = vec![
            net.add_node(node(1, NodeKind::Equipment, "AHU")),
            net.add_node(duct(2, 0.0)),
            net.add_node(node(3, NodeKind::Fitting, "Rectangular Elbow - Radius")),
            net.add_node(duct(4, 2.0)),
        ];
        for w in path.windows(2) {
            net.connect(w[0], w[1], EdgeSource::Recorded);
        }
        net.covers.insert(2, Covers { insulated: true, lining_m: Some(0.025) });
        net.covers.insert(3, Covers { insulated: false, lining_m: Some(0.025) });
        let mut children = vec![Vec::new(); 4];
        for w in path.windows(2) {
            children[w[0]].push(w[1]);
        }
        let mapped = chain(&net, &path, &[0.2; 4], &children, BendRegen::default());
        let blocks: Vec<_> = mapped.iter().map(|m| (&m.block, m.note.as_str())).collect();
        assert!(matches!(blocks[0], (BlockParams::Segment { lining_m: 0.025, wrapped: true, .. }, "duct, lined 25 mm, insulated")));
        assert!(matches!(blocks[1], (BlockParams::Bend { lined: true, .. }, _)), "{}", blocks[1].1);
        assert!(matches!(blocks[2], (BlockParams::Segment { lining_m: 0.0, wrapped: false, .. }, "duct")));
    }
}
