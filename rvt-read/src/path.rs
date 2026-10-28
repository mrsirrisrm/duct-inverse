//! Air paths: the tree from a piece of equipment out to its terminals, and
//! the flow each element carries toward a given terminal.

use std::collections::VecDeque;

use crate::network::{Network, NodeKind};

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
/// entered; elements naming none are, since many fittings carry no name.
pub fn tree(net: &Network, root: usize, system: &str) -> Tree {
    let adj = net.neighbours();
    let mut parent = vec![None; net.nodes.len()];
    let mut seen = vec![false; net.nodes.len()];
    let mut order = Vec::new();
    let mut queue = VecDeque::from([root]);
    seen[root] = true;
    while let Some(n) = queue.pop_front() {
        order.push(n);
        if n != root && net.nodes[n].kind == NodeKind::Equipment {
            continue;
        }
        for &m in &adj[n] {
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

/// Of every unit's systems, the one serving the most terminals with a design
/// flow, out to its farthest such terminal. Any system: fan noise reaches a
/// room down exhaust ducts too, and system names are in the model's language.
pub fn critical_path(net: &Network) -> Option<CriticalPath> {
    let (tree, system) = net
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NodeKind::Equipment)
        .flat_map(|(e, n)| n.systems.iter().map(move |s| (e, s)))
        .map(|(e, s)| (tree(net, e, s), s.clone()))
        .max_by_key(|(tree, _)| tree.order.iter().filter(|&&n| has_flow(net, n)).count())?;
    let terminal = *tree
        .order
        .iter()
        .filter(|&&n| has_flow(net, n))
        .max_by_key(|&&t| tree.path_to(t).map_or(0, |p| p.len()))?;
    let path = tree.path_to(terminal)?;
    let flow = tree.flows(net);
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
}
