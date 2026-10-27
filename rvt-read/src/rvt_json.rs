//! Read the element list `rvt-elements` writes (rvt-rs v0.10 JSON shape).

use std::collections::HashSet;

use serde_json::Value;

use crate::build::{self, Element, Field};
use crate::network::Network;

fn field(f: &Value) -> Option<(&str, Field<'_>)> {
    let name = f["name"].as_str()?;
    let value = match f["kind"].as_str()? {
        "element_id" => Field::Id(f["id"].as_i64()?),
        "string" => Field::Str(f["value"].as_str()?),
        _ => Field::Num(f["value"].as_f64()?),
    };
    Some((name, value))
}

pub fn parse(json: &Value) -> Network {
    let elements = json.as_array().map(Vec::as_slice).unwrap_or_default();
    let decoded: HashSet<i64> = elements.iter().filter_map(|e| e["id"].as_i64()).collect();
    let duct_elements = elements.iter().filter_map(|e| {
        let class = e["class_name"].as_str().filter(|c| build::is_duct_domain(c))?;
        Some(Element {
            id: e["id"].as_i64()?,
            class,
            fields: e["fields"].as_array()?.iter().filter_map(field).collect(),
            flow_cfs: build::flow_from_parameters(e["parameters"].as_array().into_iter().flatten()),
        })
    });
    build::build(duct_elements, &decoded)
}
