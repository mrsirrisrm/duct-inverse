//! Read the element list `rvt-elements` writes (rvt-rs v0.10 JSON shape).
//!
//! rvt-rs reports Revit's internal units: feet, and ft^3/s for airflow.

use std::collections::HashSet;

use serde_json::Value;

use crate::network::{EdgeSource, Network, Node, NodeKind, Point};

const FT: f64 = 0.3048;
const CFS: f64 = 0.028_316_846_592;

fn kind_of(class: &str) -> Option<NodeKind> {
    Some(match class {
        "Duct" | "FlexDuct" => NodeKind::DuctNoGeometry,
        "DuctFitting" | "DuctAccessory" => NodeKind::Fitting,
        "DuctTerminal" => NodeKind::Terminal,
        "MechanicalEquipment" => NodeKind::Equipment,
        _ => return None,
    })
}

struct Fields<'a>(Vec<(&'a str, &'a Value)>);

impl<'a> Fields<'a> {
    fn new(element: &'a Value) -> Self {
        let fields = element["fields"].as_array().map(Vec::as_slice).unwrap_or_default();
        Self(
            fields
                .iter()
                .filter_map(|f| {
                    let value = f.get("value").or_else(|| f.get("id"))?;
                    Some((f["name"].as_str()?, value))
                })
                .collect(),
        )
    }

    fn get(&self, name: &str) -> Option<&'a Value> {
        self.0.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }

    fn f64(&self, name: &str) -> Option<f64> {
        self.get(name)?.as_f64()
    }

    fn str(&self, name: &str) -> String {
        self.get(name).and_then(Value::as_str).unwrap_or_default().to_string()
    }

    fn point(&self, [x, y, z]: [&str; 3]) -> Option<Point> {
        Some([self.f64(x)? * FT, self.f64(y)? * FT, self.f64(z)? * FT])
    }

    fn joined_ids(&self) -> impl Iterator<Item = i64> + '_ {
        self.0
            .iter()
            .filter(|(n, _)| n.starts_with("m_connector_") && n.ends_with("_element"))
            .filter_map(|(_, v)| v.as_i64())
    }
}

fn flow(element: &Value) -> Option<f64> {
    element["parameters"].as_array()?.iter().find_map(|p| match p["name"].as_str() {
        Some("Flow" | "Air Flow") => p["value"].as_f64().map(|v| v * CFS),
        _ => None,
    })
}

/// Build the network from `rvt-elements` output: duct-domain elements as
/// nodes, their recorded joins as edges. A join to an id the file never
/// decoded becomes an [`NodeKind::Undecoded`] node, so two elements that both
/// name the same missing flex duct end up connected through it.
pub fn parse(json: &Value) -> Network {
    let elements = json.as_array().map(Vec::as_slice).unwrap_or_default();
    let decoded: HashSet<i64> = elements.iter().filter_map(|e| e["id"].as_i64()).collect();
    let mut net = Network::default();
    let mut joins = Vec::new();

    for element in elements {
        let (Some(id), Some(mut kind)) = (
            element["id"].as_i64(),
            element["class_name"].as_str().and_then(kind_of),
        ) else {
            continue;
        };
        let f = Fields::new(element);
        if kind == NodeKind::DuctNoGeometry {
            if let (Some(start), Some(end), Some(w), Some(h)) = (
                f.point(["m_duct_start_x", "m_duct_start_y", "m_duct_start_z"]),
                f.point(["m_duct_end_x", "m_duct_end_y", "m_duct_end_z"]),
                f.f64("m_curve_width"),
                f.f64("m_curve_height"),
            ) {
                kind = NodeKind::Duct { start, end, width: w * FT, height: h * FT };
            }
        }
        let systems = f
            .0
            .iter()
            .filter(|(n, _)| n.starts_with("m_mep_system_"))
            .filter_map(|(_, v)| v.as_str().map(str::to_string))
            .collect();
        net.add_node(Node {
            id,
            kind,
            family: f.str("m_family_name"),
            type_name: f.str("m_type_name"),
            systems,
            location: f.point(["m_locationX", "m_locationY", "m_locationZ"]),
            flow: flow(element),
        });
        joins.extend(f.joined_ids().map(|other| (id, other)));
    }

    for (a, b) in joins {
        let b = match net.find(b) {
            Some(b) => b,
            // Decoded but outside the duct domain (pipes, electrical): not ours.
            None if decoded.contains(&b) => continue,
            None => net.add_node(Node {
                id: b,
                kind: NodeKind::Undecoded,
                family: String::new(),
                type_name: String::new(),
                systems: Vec::new(),
                location: None,
                flow: None,
            }),
        };
        let a = net.find(a).expect("join source is a parsed node");
        net.connect(a, b, EdgeSource::Recorded);
    }
    net
}
