//! Connectivity summaries: islands, and what each air handler reaches.

use crate::network::{Network, NodeKind};

/// Island (connected component) index of every node.
pub fn islands(net: &Network) -> Vec<usize> {
    let adj = net.neighbours();
    let mut island = vec![usize::MAX; net.nodes.len()];
    let mut next = 0;
    for start in 0..net.nodes.len() {
        if island[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        island[start] = next;
        while let Some(n) = stack.pop() {
            for &m in &adj[n] {
                if island[m] == usize::MAX {
                    island[m] = next;
                    stack.push(m);
                }
            }
        }
        next += 1;
    }
    island
}

#[derive(Debug)]
pub struct Reach {
    pub equipment: usize,
    pub terminals: usize,
    /// Sum of the reached terminals' design flows, m^3/s.
    pub terminal_flow: f64,
}

/// Terminals in the same island as each piece of equipment.
pub fn equipment_reach(net: &Network, island: &[usize]) -> Vec<Reach> {
    net.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NodeKind::Equipment)
        .map(|(e, _)| {
            let reached = net
                .nodes
                .iter()
                .enumerate()
                .filter(|&(t, n)| n.kind == NodeKind::Terminal && island[t] == island[e]);
            let (terminals, terminal_flow) =
                reached.fold((0, 0.0), |(c, f), (_, n)| (c + 1, f + n.flow.unwrap_or(0.0)));
            Reach { equipment: e, terminals, terminal_flow }
        })
        .collect()
}
