//! Build the network straight from a Revit file with rvt-rs, with no JSON in
//! between (feature `rvt`).

use std::collections::HashSet;

use rvt::RevitFile;
use rvt::walker::{self, DecodedElement, InstanceField, WalkerLimits, PRODUCTION_ELEMENT_MIN_SCORE};

use crate::build::{self, Element, Field};
use crate::network::Network;

fn field<'a>((name, value): &'a (String, InstanceField)) -> Option<(&'a str, Field<'a>)> {
    let value = match value {
        InstanceField::ElementId { id, .. } => Field::Id(i64::from(*id)),
        InstanceField::Float { value, .. } => Field::Num(*value),
        InstanceField::Integer { value, .. } => Field::Num(*value as f64),
        InstanceField::String(s) => Field::Str(s),
        _ => return None,
    };
    Some((name, value))
}

/// The duct network of a Revit file. `progress` is rvt-rs's walker control,
/// for reporting and cancelling the element walk.
pub fn network(bytes: Vec<u8>, progress: &rvt::control::WalkerControl) -> Result<Network, rvt::Error> {
    network_with_laps(bytes, progress, &mut |_| {})
}

/// [`network`], calling `lap` with each stage's name as it finishes
/// ("walk", "parameters", "build"), for timing.
pub fn network_with_laps(
    bytes: Vec<u8>,
    progress: &rvt::control::WalkerControl,
    lap: &mut dyn FnMut(&'static str),
) -> Result<Network, rvt::Error> {
    let mut file = RevitFile::open_bytes(bytes)?;
    let elements: Vec<DecodedElement> =
        walker::iter_elements_with_control(&mut file, PRODUCTION_ELEMENT_MIN_SCORE, WalkerLimits::default(), progress)?
            .collect();
    lap("walk");
    let decoded: HashSet<i64> = elements.iter().filter_map(|e| e.id).map(i64::from).collect();
    // Keep only the duct elements, and read saved parameters for them alone:
    // they are all `build` uses, and the rest is most of the memory.
    let ducts: Vec<DecodedElement> =
        elements.into_iter().filter(|e| e.id.is_some() && build::is_duct_domain(&e.class)).collect();
    let saved = rvt::elements::parameters::saved_parameters_lossy(&mut file, &ducts).value;
    lap("parameters");
    let duct_elements = ducts.iter().map(|e| Element {
        id: i64::from(e.id.unwrap()),
        class: &e.class,
        fields: e.fields.iter().filter_map(field).collect(),
        flow_cfs: build::flow_from_parameters(&saved.entries_for(e)),
    });
    let net = build::build(duct_elements, &decoded);
    lap("build");
    Ok(net)
}
