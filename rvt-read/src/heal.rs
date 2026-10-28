//! Add the joins a model's geometry implies but its connectors do not record.

use crate::network::{EdgeSource, Network, NodeKind, Point};

/// How close the branch axis must pass to the main's centreline, m.
const AXIS_TOL: f64 = 0.02;
/// Largest |cos| between branch and main axes still counted as a side branch.
const PERP_TOL: f64 = 0.1;
/// The takeoff must sit within this many main-duct sizes of the main's axis.
const REACH: f64 = 4.0;

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
/// implicit. The main is the duct whose centreline the branch duct's axis runs
/// into, square to it, near the takeoff. Where several qualify (stacked or
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
        let (Some(at), Some(branch)) = (
            node.location,
            adj[t].iter().find_map(|&b| match net.nodes[b].kind {
                NodeKind::Duct { start, end, .. } => Some((b, start, end)),
                _ => None,
            }),
        ) else {
            report.unmatched.push(node.id);
            continue;
        };
        let (b, b_start, b_end) = branch;
        let b_dir = sub(b_end, b_start);

        let mut candidates: Vec<(bool, f64, usize)> = net
            .nodes
            .iter()
            .enumerate()
            .filter(|&(m, _)| m != b && !adj[t].contains(&m))
            .filter_map(|(m, main)| {
                let NodeKind::Duct { start, end, width, height } = main.kind else {
                    return None;
                };
                let m_dir = sub(end, start);
                let cos = dot(b_dir, m_dir).abs() / (norm(b_dir) * norm(m_dir)).max(1e-12);
                let (gap, s) = line_to_segment(b_start, b_dir, start, m_dir)?;
                let near = dist(at, lerp(start, m_dir, s));
                (cos < PERP_TOL && gap < AXIS_TOL && near < REACH * width.max(height)).then(|| {
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
        }
    }

    fn duct(id: i64, start: Point, end: Point) -> Node {
        node(id, NodeKind::Duct { start, end, width: 0.4, height: 0.2 }, "", start)
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

    #[test]
    fn takeoff_off_the_end_of_every_main_is_reported() {
        let (mut net, _, _) = tee();
        net.nodes[0].kind = NodeKind::Duct { start: [0.0; 3], end: [4.0, 0.0, 0.0], width: 0.4, height: 0.2 };
        let report = join_takeoffs(&mut net);
        assert_eq!((report.joined, report.unmatched), (0, vec![4]));
    }
}
