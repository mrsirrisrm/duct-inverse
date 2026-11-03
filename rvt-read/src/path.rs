//! Air paths: the tree from a piece of equipment out to its terminals, and
//! the flow each element carries toward a given terminal.

use std::collections::VecDeque;

use crate::network::{EdgeSource, Network, NodeKind};

/// Breadth-first tree rooted at `root`, kept to one system. Other equipment
/// is a leaf: air does not pass through a second unit on its way to a
/// terminal.
pub struct Tree {
    pub root: usize,
    pub parent: Vec<Option<usize>>,
    /// Nodes in visiting order, root first.
    pub order: Vec<usize>,
}

/// The tree of `system` from `root`. Elements naming other systems are not
/// entered; elements naming none are, since many fittings carry no name. The
/// unit is left only through the elements naming `system`, where any do: its
/// other connectors are other systems' (an outdoor air inlet, a return).
pub fn tree(net: &Network, root: usize, system: &str) -> Tree {
    let adj = net.neighbours();
    let mut parent = vec![None; net.nodes.len()];
    let mut seen = vec![false; net.nodes.len()];
    let mut order = Vec::new();
    let mut queue = VecDeque::from([root]);
    seen[root] = true;
    let names = |m: usize| net.nodes[m].systems.iter().any(|s| s == system);
    let named_exit = adj[root].iter().any(|&m| names(m));
    while let Some(n) = queue.pop_front() {
        order.push(n);
        if n != root && net.nodes[n].kind == NodeKind::Equipment {
            continue;
        }
        for &m in &adj[n] {
            if n == root && named_exit && !names(m) {
                continue;
            }
            let systems = &net.nodes[m].systems;
            if !seen[m] && (systems.is_empty() || systems.iter().any(|s| s == system)) {
                seen[m] = true;
                parent[m] = Some(n);
                queue.push_back(m);
            }
        }
    }
    Tree { root, parent, order }
}

impl Tree {
    /// Root to `node`, inclusive; `None` if `node` is not in the tree.
    pub fn path_to(&self, node: usize) -> Option<Vec<usize>> {
        if node != self.root && self.parent[node].is_none() {
            return None;
        }
        let mut path = vec![node];
        while let Some(p) = self.parent[*path.last().unwrap()] {
            path.push(p);
        }
        path.reverse();
        Some(path)
    }

    /// Flow through every node: the sum of the design flows of the terminals
    /// beyond it.
    pub fn flows(&self, net: &Network) -> Vec<f64> {
        let mut flow = vec![0.0; net.nodes.len()];
        for &n in self.order.iter().rev() {
            let node = &net.nodes[n];
            if node.kind == NodeKind::Terminal {
                flow[n] += node.flow.unwrap_or(0.0);
            }
            if let Some(p) = self.parent[n] {
                flow[p] += flow[n];
            }
        }
        flow
    }
}

/// One supply path chosen for a first look at a model.
pub struct CriticalPath {
    pub tree: Tree,
    pub system: String,
    pub terminal: usize,
    /// Root to terminal, inclusive.
    pub path: Vec<usize>,
    /// Flow through each node of `tree`, m^3/s.
    pub flow: Vec<f64>,
}

fn has_flow(net: &Network, n: usize) -> bool {
    net.nodes[n].kind == NodeKind::Terminal && net.nodes[n].flow.is_some_and(|q| q > 0.0)
}

/// Terminals taking more than this times a unit's rated Air Flow: its tree
/// reaches air the unit cannot supply, most likely through a join to another
/// unit's network, and is not rated.
pub const OVERFED_RATIO: f64 = 1.5;

/// A unit whose tree's terminals take more than [`OVERFED_RATIO`] times its
/// rated flow.
pub struct Overfed {
    pub unit: usize,
    pub system: String,
    /// The unit's Air Flow and its terminals' sum, m^3/s.
    pub rated: f64,
    pub served: f64,
    /// Joins in the tree the model's connectors do not record (healed from
    /// geometry), each with the terminal flow beyond it, most first: where
    /// to look for the join that should not be there.
    pub healed_joins: Vec<(usize, usize, f64)>,
}

/// Every unit's tree, one per air system it names (one a duct, fitting or
/// terminal names too: not a unit's hydronic systems), with its flows.
fn unit_trees(net: &Network) -> Vec<(Tree, String, Vec<f64>)> {
    let air: std::collections::HashSet<&str> = net
        .nodes
        .iter()
        .filter(|n| n.kind != NodeKind::Equipment)
        .flat_map(|n| n.systems.iter().map(String::as_str))
        .collect();
    net.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NodeKind::Equipment)
        .flat_map(|(e, n)| n.systems.iter().filter(|s| air.contains(s.as_str())).map(move |s| (e, s)))
        .map(|(e, s)| {
            let tree = tree(net, e, s);
            let flow = tree.flows(net);
            (tree, s.clone(), flow)
        })
        .collect()
}

fn overfed_by(net: &Network, tree: &Tree, flow: &[f64]) -> Option<(f64, f64)> {
    let rated = net.nodes[tree.root].flow.filter(|&q| q > 0.0)?;
    let served = flow[tree.root];
    (served > OVERFED_RATIO * rated).then_some((rated, served))
}

/// The units [`critical_path`] passes over: their terminals take more air
/// than [`OVERFED_RATIO`] times their rating.
pub fn overfed(net: &Network) -> Vec<Overfed> {
    let healed: std::collections::HashSet<(usize, usize)> = net
        .edges
        .iter()
        .filter(|e| e.source != EdgeSource::Recorded)
        .flat_map(|e| [(e.a, e.b), (e.b, e.a)])
        .collect();
    unit_trees(net)
        .into_iter()
        .filter_map(|(tree, system, flow)| {
            let (rated, served) = overfed_by(net, &tree, &flow)?;
            let mut healed_joins: Vec<_> = tree
                .order
                .iter()
                .filter_map(|&n| Some((tree.parent[n]?, n)))
                .filter(|&(p, n)| healed.contains(&(p, n)) && flow[n] > 0.0)
                .map(|(p, n)| (p, n, flow[n]))
                .collect();
            healed_joins.sort_by(|a, b| b.2.total_cmp(&a.2));
            Some(Overfed { unit: tree.root, system, rated, served, healed_joins })
        })
        .collect()
}

/// Of every unit's systems, the one serving the most terminals with a design
/// flow, out to its farthest such terminal. Any system: fan noise reaches a
/// room down exhaust ducts too, and system names are in the model's language.
/// A unit [`overfed`] is passed over.
pub fn critical_path(net: &Network) -> Option<CriticalPath> {
    let (tree, system, flow) = unit_trees(net)
        .into_iter()
        .filter(|(tree, _, flow)| overfed_by(net, tree, flow).is_none())
        .max_by_key(|(tree, _, _)| tree.order.iter().filter(|&&n| has_flow(net, n)).count())?;
    let terminal = *tree
        .order
        .iter()
        .filter(|&&n| has_flow(net, n))
        .max_by_key(|&&t| tree.path_to(t).map_or(0, |p| p.len()))?;
    let path = tree.path_to(terminal)?;
    Some(CriticalPath { tree, system, terminal, path, flow })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{EdgeSource, Node};

    fn node(id: i64, kind: NodeKind, system: &str, flow: Option<f64>) -> Node {
        Node {
            id,
            kind,
            family: String::new(),
            type_name: String::new(),
            systems: vec![system.into()],
            location: None,
            flow,
            ifc: None,
            description: String::new(),
            extent: None,
        }
    }

    /// Unit → duct → tee → two supply terminals, plus a return grille on the
    /// unit's other side, which the supply tree must not enter.
    #[test]
    fn supply_tree_sums_supply_terminals_only() {
        let mut net = Network::default();
        let unit = net.add_node(node(1, NodeKind::Equipment, "SA", None));
        let duct = net.add_node(node(2, NodeKind::DuctNoGeometry, "SA", None));
        let tee = net.add_node(node(3, NodeKind::Fitting, "SA", None));
        let t1 = net.add_node(node(4, NodeKind::Terminal, "SA", Some(0.1)));
        let t2 = net.add_node(node(5, NodeKind::Terminal, "SA", Some(0.05)));
        let ret = net.add_node(node(6, NodeKind::Terminal, "RA", Some(0.15)));
        for (a, b) in [(unit, duct), (duct, tee), (tee, t1), (tee, t2), (unit, ret)] {
            net.connect(a, b, EdgeSource::Recorded);
        }
        let tree = tree(&net, unit, "SA");
        assert_eq!(tree.path_to(t2), Some(vec![unit, duct, tee, t2]));
        assert_eq!(tree.path_to(ret), None);
        let flow = tree.flows(&net);
        assert!((flow[unit] - 0.15).abs() < 1e-12);
        assert!((flow[tee] - 0.15).abs() < 1e-12);
        assert!((flow[t2] - 0.05).abs() < 1e-12);
    }

    /// A unit's outdoor-air inlet, on unnamed ductwork reaching another
    /// network, is not its supply: the unit is left through its supply duct
    /// alone. Its hydronic system, which no duct names, has no tree.
    #[test]
    fn unit_left_through_its_system_only() {
        let mut net = Network::default();
        let mut unit_node = node(1, NodeKind::Equipment, "SA", Some(0.1));
        unit_node.systems.push("Hydronic Supply".into());
        let unit = net.add_node(unit_node);
        let supply = net.add_node(node(2, NodeKind::DuctNoGeometry, "SA", None));
        let t1 = net.add_node(node(3, NodeKind::Terminal, "SA", Some(0.1)));
        let mut inlet = node(4, NodeKind::DuctNoGeometry, "", None);
        inlet.systems.clear();
        let inlet = net.add_node(inlet);
        let mut other = node(5, NodeKind::Terminal, "", Some(2.0));
        other.systems.clear();
        let other = net.add_node(other);
        for (a, b) in [(unit, supply), (supply, t1), (unit, inlet), (inlet, other)] {
            net.connect(a, b, EdgeSource::Recorded);
        }
        let tree = tree(&net, unit, "SA");
        assert_eq!(tree.path_to(other), None);
        let cp = critical_path(&net).expect("a path");
        assert_eq!((cp.system.as_str(), cp.terminal), ("SA", t1));
        assert!(unit_trees(&net).iter().all(|(_, s, _)| s == "SA"));
    }

    /// A unit whose terminals take three times its rated flow is not rated,
    /// and is reported with the healed join the extra air comes through.
    #[test]
    fn overfed_unit_reported_not_rated() {
        let mut net = Network::default();
        let unit = net.add_node(node(1, NodeKind::Equipment, "SA", Some(0.1)));
        let duct = net.add_node(node(2, NodeKind::DuctNoGeometry, "SA", None));
        let t1 = net.add_node(node(3, NodeKind::Terminal, "SA", Some(0.1)));
        let main = net.add_node(node(4, NodeKind::DuctNoGeometry, "SA", None));
        let t2 = net.add_node(node(5, NodeKind::Terminal, "SA", Some(0.2)));
        net.connect(unit, duct, EdgeSource::Recorded);
        net.connect(duct, t1, EdgeSource::Recorded);
        net.connect(duct, main, EdgeSource::TakeoffAxis { candidates: 1 });
        net.connect(main, t2, EdgeSource::Recorded);
        assert!(critical_path(&net).is_none());
        let [o] = overfed(&net).try_into().ok().expect("one overfed unit");
        assert_eq!(o.unit, unit);
        assert!((o.served - 0.3).abs() < 1e-12 && (o.rated - 0.1).abs() < 1e-12);
        assert_eq!(o.healed_joins.len(), 1);
        assert_eq!((o.healed_joins[0].0, o.healed_joins[0].1), (duct, main));
    }
}
