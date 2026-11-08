//! Add the joins a model's geometry implies but its connectors do not record.

use crate::network::{EdgeSource, Network, NodeKind, Point};

/// How close the branch axis must pass to the main's centreline, m.
const AXIS_TOL: f64 = 0.02;
/// Largest |cos| between branch and main axes still counted as a side branch.
const PERP_TOL: f64 = 0.1;
/// The takeoff must sit within this many main-duct sizes of the main's axis:
/// on rme's 287 joins, at most 0.91 (on its wall, its own depth out).
const REACH: f64 = 1.5;

pub fn is_takeoff(family: &str) -> bool {
    let f = family.to_ascii_lowercase();
    f.contains("takeoff") || f.contains("take-off") || f.contains("tap")
}

#[derive(Debug, Default, PartialEq)]
pub struct TakeoffReport {
    pub takeoffs: usize,
    pub joined: usize,
    pub ambiguous: usize,
    /// Takeoff ids no main duct could be found for.
    pub unmatched: Vec<i64>,
}

/// Join each takeoff to the main duct it sits on.
///
/// Revit records only a takeoff's branch connector; the tap onto the main is
/// implicit. A takeoff with more than one join already has its main. The main is the duct whose centreline the branch duct's axis runs
/// into, square to it, near the takeoff; between level ducts, anywhere a
/// branch flush with the main's top or bottom would put it. Where several qualify (stacked or
/// parallel mains), one sharing a system name wins, then the nearest.
pub fn join_takeoffs(net: &mut Network) -> TakeoffReport {
    let mut report = TakeoffReport::default();
    let adj = net.neighbours();
    let mut found = Vec::new();

    for (t, node) in net.nodes.iter().enumerate() {
        if node.kind != NodeKind::Fitting || !is_takeoff(&node.family) {
            continue;
        }
        report.takeoffs += 1;
        // Joined both ways already, as an IFC's ports join it.
        if adj[t].len() > 1 {
            continue;
        }
        let (Some(at), Some(branch)) = (
            node.location,
            adj[t].iter().find_map(|&b| match net.nodes[b].kind {
                NodeKind::Duct { start, end, height, .. } => Some((b, start, end, height)),
                _ => None,
            }),
        ) else {
            report.unmatched.push(node.id);
            continue;
        };
        let (b, b_start, b_end, b_height) = branch;
        let b_dir = sub(b_end, b_start);
        let level = |d: Point| d[2].abs() < PERP_TOL * norm(d);

        let mut candidates: Vec<(bool, f64, usize)> = net
            .nodes
            .iter()
            .enumerate()
            .filter(|&(m, _)| m != b && !adj[t].contains(&m))
            .filter_map(|(m, main)| {
                let NodeKind::Duct { start, end, width, height, .. } = main.kind else {
                    return None;
                };
                let m_dir = sub(end, start);
                let cos = dot(b_dir, m_dir).abs() / (norm(b_dir) * norm(m_dir)).max(1e-12);
                let (gap, s) = line_to_segment(b_start, b_dir, start, m_dir)?;
                let near = dist(at, lerp(start, m_dir, s));
                // Level ducts: a branch flush with a deeper main's top or
                // bottom runs half their heights' difference off its axis.
                let flush = if level(b_dir) && level(m_dir) { (height - b_height).max(0.0) / 2.0 } else { 0.0 };
                (cos < PERP_TOL && gap < AXIS_TOL + flush && near < REACH * width.max(height)).then(|| {
                    let shared = main.systems.iter().any(|s| node.systems.contains(s));
                    (!shared, near, m)
                })
            })
            .collect();
        candidates.sort_by(|x, y| x.partial_cmp(y).unwrap());

        match candidates.first() {
            Some(&(_, _, m)) => {
                if candidates.len() > 1 {
                    report.ambiguous += 1;
                }
                found.push((t, m, candidates.len()));
            }
            None => report.unmatched.push(node.id),
        }
    }

    for (t, m, candidates) in found {
        if net.connect(t, m, EdgeSource::TakeoffAxis { candidates }) {
            report.joined += 1;
        }
    }
    report
}

/// How far outside a duct's half-size a terminal on its side may stand, m:
/// its location is its face's, a grille's depth off the duct.
const FACE_TOL: f64 = 0.15;

/// Join each terminal nothing joins to the duct of its own system whose side
/// it is mounted on: its location within the duct's half-size and
/// [`FACE_TOL`] of the axis, between the duct's ends. Revit records an air
/// terminal on a duct as a tap on the duct's body, which is not read. The
/// nearest such duct wins; returns how many terminals were joined.
pub fn join_terminals_on_ducts(net: &mut Network) -> usize {
    let adj = net.neighbours();
    let mut found = Vec::new();
    for (t, node) in net.nodes.iter().enumerate() {
        let (NodeKind::Terminal, Some(at), true) = (&node.kind, node.location, adj[t].is_empty()) else {
            continue;
        };
        let best = net
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, duct)| duct.systems.iter().any(|s| node.systems.contains(s)))
            .filter_map(|(d, duct)| {
                let NodeKind::Duct { start, end, width, height, .. } = duct.kind else {
                    return None;
                };
                let axis = sub(end, start);
                let s = dot(sub(at, start), axis) / dot(axis, axis).max(1e-12);
                let off = dist(at, lerp(start, axis, s.clamp(0.0, 1.0)));
                ((0.0..=1.0).contains(&s) && off <= width.max(height) / 2.0 + FACE_TOL).then_some((off, d))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, d)) = best {
            found.push((t, d));
        }
    }
    found.into_iter().filter(|&(t, d)| net.connect(t, d, EdgeSource::OnDuct)).count()
}

/// Join each flex or other run with no decoded geometry, loose at one end, to
/// the duct of its own system its box reaches: within half the duct's size
/// plus [`FACE_TOL`] of its axis, and not already on the same island. A flex
/// tapped into a duct's side records nothing at that end, as a terminal on a
/// duct does ([`join_terminals_on_ducts`]). Returns how many joined.
pub fn join_runs_on_ducts(net: &mut Network) -> usize {
    let adj = net.neighbours();
    let island = crate::graph::islands(net);
    let mut found = Vec::new();
    for (r, run) in net.nodes.iter().enumerate() {
        let (NodeKind::DuctNoGeometry, Some(at), Some(extent), true) = (&run.kind, run.location, run.extent, adj[r].len() <= 1) else {
            continue;
        };
        let best = net
            .nodes
            .iter()
            .enumerate()
            .filter(|&(d, duct)| island[d] != island[r] && duct.systems.iter().any(|s| run.systems.contains(s)))
            .filter_map(|(d, duct)| {
                let NodeKind::Duct { start, end, width, height, .. } = duct.kind else {
                    return None;
                };
                let off = segment_to_box(start, end, at, extent);
                (off <= width.max(height) / 2.0 + FACE_TOL).then_some((off, d))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, d)) = best {
            found.push((r, d));
        }
    }
    found.into_iter().filter(|&(r, d)| net.connect(r, d, EdgeSource::OnDuct)).count()
}

/// The least distance from a segment to a box of the given centre and extent:
/// it is convex along the segment.
fn segment_to_box(start: Point, end: Point, centre: Point, extent: Point) -> f64 {
    let axis = sub(end, start);
    let off = |s: f64| {
        let p = lerp(start, axis, s);
        let outside: Point = std::array::from_fn(|k| ((p[k] - centre[k]).abs() - extent[k] / 2.0).max(0.0));
        norm(outside)
    };
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if off(a) <= off(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    off((lo + hi) / 2.0)
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

fn dist(a: Point, b: Point) -> f64 {
    norm(sub(a, b))
}

fn lerp(p: Point, d: Point, s: f64) -> Point {
    [p[0] + s * d[0], p[1] + s * d[1], p[2] + s * d[2]]
}

/// Closest approach between the infinite line `p + u·d` and the segment
/// `q + s·e`, `s` in [0, 1]. Returns the gap and `s`; `None` if the segment
/// is parallel to the line or the closest point is past either end.
fn line_to_segment(p: Point, d: Point, q: Point, e: Point) -> Option<(f64, f64)> {
    let (a, b, c) = (dot(d, d), dot(d, e), dot(e, e));
    let w = sub(p, q);
    let den = a * c - b * b;
    if den.abs() < 1e-12 * a * c {
        return None;
    }
    let u = (b * dot(e, w) - c * dot(d, w)) / den;
    let s = (a * dot(e, w) - b * dot(d, w)) / den;
    if !(-0.01..=1.01).contains(&s) {
        return None;
    }
    Some((dist(lerp(p, d, u), lerp(q, e, s)), s.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Node;

    fn node(id: i64, kind: NodeKind, family: &str, at: Point) -> Node {
        Node {
            id,
            kind,
            family: family.into(),
            type_name: String::new(),
            systems: vec!["Supply Air 1".into()],
            location: Some(at),
            flow: None,
            ifc: None,
            description: String::new(),
            extent: None,
        }
    }

    fn duct(id: i64, start: Point, end: Point) -> Node {
        node(id, NodeKind::Duct { start, end, width: 0.4, height: 0.2, round: false }, "", start)
    }

    /// A grille on a duct's side, 0.25 m off its axis (its half-height 0.1
    /// and the grille's depth), joins it; one of another system, one past
    /// the duct's end and one already joined do not.
    #[test]
    fn terminals_on_duct_sides() {
        let mut net = Network::default();
        let main = net.add_node(duct(1, [0.0, 0.0, 3.0], [10.0, 0.0, 3.0]));
        let on = net.add_node(node(2, NodeKind::Terminal, "Supply Grille - Curve Face", [4.0, 0.0, 2.75]));
        let mut other = node(3, NodeKind::Terminal, "Supply Grille - Curve Face", [6.0, 0.0, 2.75]);
        other.systems = vec!["Supply Air 2".into()];
        let other = net.add_node(other);
        let past = net.add_node(node(4, NodeKind::Terminal, "Supply Grille - Curve Face", [10.5, 0.0, 2.75]));
        let joined = net.add_node(node(5, NodeKind::Terminal, "Supply Grille - Curve Face", [8.0, 0.0, 2.75]));
        let elsewhere = net.add_node(duct(6, [8.0, 0.0, 2.0], [8.0, 0.0, 2.7]));
        net.connect(joined, elsewhere, EdgeSource::Recorded);
        assert_eq!(join_terminals_on_ducts(&mut net), 1);
        let adj = net.neighbours();
        assert_eq!(adj[on], [main]);
        assert!(adj[other].is_empty() && adj[past].is_empty());
        assert_eq!(adj[joined], [elsewhere]);
    }

    #[test]
    fn flex_runs_tapped_into_ducts() {
        let run = |id, location: Point, extent: Point| {
            let mut n = node(id, NodeKind::DuctNoGeometry, "", location);
            n.extent = Some(extent);
            n
        };
        let mut net = Network::default();
        let main = net.add_node(duct(1, [0.0, 0.0, 3.0], [10.0, 0.0, 3.0]));
        // Its box from the main's underside down to the diffuser below it.
        let flex = net.add_node(run(2, [4.0, 0.3, 2.3], [0.6, 0.6, 0.8]));
        let diffuser = net.add_node(node(3, NodeKind::Terminal, "Supply Diffuser", [4.0, 0.5, 1.9]));
        net.connect(flex, diffuser, EdgeSource::Recorded);
        // Too far below it, and of another system.
        let far = net.add_node(run(4, [6.0, 0.0, 1.0], [0.6, 0.6, 0.8]));
        let mut other = run(5, [8.0, 0.0, 2.5], [0.6, 0.6, 0.8]);
        other.systems = vec!["Supply Air 2".into()];
        let other = net.add_node(other);
        assert_eq!(join_runs_on_ducts(&mut net), 1);
        let adj = net.neighbours();
        assert!(adj[flex].contains(&main));
        assert!(adj[far].is_empty() && adj[other].is_empty());
    }

    /// Main along x, branch leaving along y from the main's side wall, and a
    /// parallel decoy main 1 m further on whose axis the branch line misses.
    fn tee() -> (Network, usize, usize) {
        let mut net = Network::default();
        let main = net.add_node(duct(1, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]));
        net.add_node(duct(2, [0.0, 0.0, 1.0], [10.0, 0.0, 1.0]));
        let branch = net.add_node(duct(3, [5.0, 0.3, 0.0], [5.0, 3.0, 0.0]));
        let takeoff = net.add_node(node(4, NodeKind::Fitting, "M_Rectangular Duct Takeoff", [5.0, 0.25, 0.0]));
        net.connect(takeoff, branch, EdgeSource::Recorded);
        (net, takeoff, main)
    }

    #[test]
    fn takeoff_joins_the_main_its_branch_runs_into() {
        let (mut net, takeoff, main) = tee();
        let report = join_takeoffs(&mut net);
        assert_eq!(report, TakeoffReport { takeoffs: 1, joined: 1, ambiguous: 0, unmatched: vec![] });
        let e = net.edges.last().unwrap();
        assert_eq!((e.a, e.b), (takeoff, main));
    }

    /// A 200 mm high branch flush with the top of a 300 mm main runs 50 mm
    /// above its axis, and joins it.
    #[test]
    fn takeoff_flush_with_a_deeper_mains_top() {
        let (mut net, takeoff, main) = tee();
        net.nodes[main].kind = NodeKind::Duct { start: [0.0; 3], end: [10.0, 0.0, 0.0], width: 0.4, height: 0.3, round: false };
        net.nodes[2].kind = NodeKind::Duct { start: [5.0, 0.3, 0.05], end: [5.0, 3.0, 0.05], width: 0.4, height: 0.2, round: false };
        let report = join_takeoffs(&mut net);
        assert_eq!((report.joined, report.unmatched), (1, vec![]));
        let e = net.edges.last().unwrap();
        assert_eq!((e.a, e.b), (takeoff, main));
    }

    #[test]
    fn takeoff_already_joined_to_a_main_is_left() {
        let (mut net, takeoff, _) = tee();
        net.connect(takeoff, 1, EdgeSource::Recorded);
        let report = join_takeoffs(&mut net);
        assert_eq!(report, TakeoffReport { takeoffs: 1, joined: 0, ambiguous: 0, unmatched: vec![] });
    }

    #[test]
    fn takeoff_off_the_end_of_every_main_is_reported() {
        let (mut net, _, _) = tee();
        net.nodes[0].kind = NodeKind::Duct { start: [0.0; 3], end: [4.0, 0.0, 0.0], width: 0.4, height: 0.2, round: false };
        let report = join_takeoffs(&mut net);
        assert_eq!((report.joined, report.unmatched), (0, vec![4]));
    }
}
