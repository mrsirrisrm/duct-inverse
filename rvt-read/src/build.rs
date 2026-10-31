//! Build the network from decoded elements, whatever decoded them.
//!
//! rvt-rs reports Revit's internal units: feet, and ft^3/s for airflow.

use std::collections::HashSet;

use crate::network::{EdgeSource, Network, Node, NodeKind, Point};

const FT: f64 = 0.3048;
pub const CFS: f64 = 0.028_316_846_592;

/// A field value, as much of it as the network needs.
#[derive(Debug, Clone, Copy)]
pub enum Field<'a> {
    Num(f64),
    Id(i64),
    Str(&'a str),
}

/// One decoded element: its rvt-rs class, named fields, and design airflow
/// in ft^3/s where a parameter carries one.
pub struct Element<'a> {
    pub id: i64,
    pub class: &'a str,
    pub fields: Vec<(&'a str, Field<'a>)>,
    pub flow_cfs: Option<f64>,
}

fn kind_of(class: &str) -> Option<NodeKind> {
    Some(match class {
        "Duct" | "FlexDuct" => NodeKind::DuctNoGeometry,
        "DuctFitting" | "DuctAccessory" => NodeKind::Fitting,
        "DuctTerminal" => NodeKind::Terminal,
        "MechanicalEquipment" => NodeKind::Equipment,
        _ => return None,
    })
}

/// Whether `build` keeps elements of this class; callers can skip the rest.
pub fn is_duct_domain(class: &str) -> bool {
    kind_of(class).is_some()
}

impl<'a> Element<'a> {
    fn get(&self, name: &str) -> Option<Field<'a>> {
        self.fields.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }

    fn num(&self, name: &str) -> Option<f64> {
        match self.get(name)? {
            Field::Num(v) => Some(v),
            _ => None,
        }
    }

    fn str(&self, name: &str) -> String {
        match self.get(name) {
            Some(Field::Str(s)) => s.to_string(),
            _ => String::new(),
        }
    }

    fn point(&self, [x, y, z]: [&str; 3]) -> Option<Point> {
        Some([self.num(x)? * FT, self.num(y)? * FT, self.num(z)? * FT])
    }

    fn joined_ids(&self) -> impl Iterator<Item = i64> + '_ {
        self.fields.iter().filter_map(|(n, v)| match v {
            Field::Id(id) if n.starts_with("m_connector_") && n.ends_with("_element") => Some(*id),
            _ => None,
        })
    }
}

/// Duct-domain elements as nodes, their recorded joins as edges. `decoded`
/// is every id the decoder produced, in any class. A join to an id it never
/// decoded becomes an [`NodeKind::Undecoded`] node, so two elements that both
/// name the same missing flex duct end up connected through it.
pub fn build<'a>(elements: impl IntoIterator<Item = Element<'a>>, decoded: &HashSet<i64>) -> Network {
    let mut net = Network::default();
    let mut joins = Vec::new();

    for element in elements {
        let Some(mut kind) = kind_of(element.class) else {
            continue;
        };
        if kind == NodeKind::DuctNoGeometry {
            if let (Some(start), Some(end), Some(w), Some(h)) = (
                element.point(["m_duct_start_x", "m_duct_start_y", "m_duct_start_z"]),
                element.point(["m_duct_end_x", "m_duct_end_y", "m_duct_end_z"]),
                element.num("m_curve_width"),
                element.num("m_curve_height"),
            ) {
                let name = format!("{} {}", element.str("m_family_name"), element.str("m_type_name")).to_lowercase();
                let round = ["round", "circular", "spiral"].iter().any(|k| name.contains(k));
                // A round duct's one dimension is its diameter.
                let (w, h) = if round { (w.max(h), w.max(h)) } else { (w, h) };
                kind = NodeKind::Duct { start, end, width: w * FT, height: h * FT, round };
            }
        }
        let systems = element
            .fields
            .iter()
            .filter_map(|(n, v)| match v {
                Field::Str(s) if n.starts_with("m_mep_system_") => Some(s.to_string()),
                _ => None,
            })
            .collect();
        joins.extend(element.joined_ids().map(|other| (element.id, other)));
        net.add_node(Node {
            id: element.id,
            kind,
            family: element.str("m_family_name"),
            type_name: element.str("m_type_name"),
            systems,
            location: element.point(["m_locationX", "m_locationY", "m_locationZ"]),
            flow: element.flow_cfs.map(|q| q * CFS),
            ifc: None,
        });
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
                ifc: None,
            }),
        };
        let a = net.find(a).expect("join source is a built node");
        net.connect(a, b, EdgeSource::Recorded);
    }
    net
}

/// Design airflow from parameter entries in rvt-rs's JSON shape.
pub fn flow_from_parameters<'v>(parameters: impl IntoIterator<Item = &'v serde_json::Value>) -> Option<f64> {
    parameters.into_iter().find_map(|p| match p["name"].as_str() {
        Some("Flow" | "Air Flow") => p["value"].as_f64(),
        _ => None,
    })
}
