//! The duct network: elements as nodes, connections as edges.

use std::collections::HashMap;

pub type Point = [f64; 3];

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// Straight duct with a decoded centreline and section.
    Duct { start: Point, end: Point, width: f64, height: f64 },
    /// Straight duct whose geometry rvt-rs did not decode (bounding box only).
    DuctNoGeometry,
    Fitting,
    Terminal,
    Equipment,
    /// Referenced by a connector but not decoded (in practice, flex ducts).
    Undecoded,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: i64,
    pub kind: NodeKind,
    pub family: String,
    pub type_name: String,
    pub systems: Vec<String>,
    pub location: Option<Point>,
    /// Design airflow, m^3/s, where the family carries one.
    pub flow: Option<f64>,
    /// What an IFC export of the model says the element is, once merged.
    pub ifc: Option<IfcInfo>,
}

#[derive(Debug, Clone)]
pub struct IfcInfo {
    /// Specific class, upper case ("IFCDUCTSILENCER").
    pub class: String,
    /// Predefined type, "" if none.
    pub predefined: String,
    pub description: String,
    /// Extent in the element's own frame, metres, for non-straight elements.
    pub size: Option<Point>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeSource {
    /// A connector join stored in the model.
    Recorded,
    /// Takeoff to the main duct its branch axis runs into.
    TakeoffAxis { candidates: usize },
    /// A duct that ends inside an element no recorded join reaches (IFC).
    EndInside,
}

#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub source: EdgeSource,
}

#[derive(Debug, Default)]
pub struct Network {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    index: HashMap<i64, usize>,
}

impl Network {
    pub fn add_node(&mut self, node: Node) -> usize {
        let i = self.nodes.len();
        self.index.insert(node.id, i);
        self.nodes.push(node);
        i
    }

    pub fn find(&self, id: i64) -> Option<usize> {
        self.index.get(&id).copied()
    }

    /// Add an edge unless the pair is already joined.
    pub fn connect(&mut self, a: usize, b: usize, source: EdgeSource) -> bool {
        if a == b || self.edges.iter().any(|e| (e.a, e.b) == (a, b) || (e.a, e.b) == (b, a)) {
            return false;
        }
        self.edges.push(Edge { a, b, source });
        true
    }

    pub fn neighbours(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.nodes.len()];
        for e in &self.edges {
            adj[e.a].push(e.b);
            adj[e.b].push(e.a);
        }
        adj
    }
}
