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
/// ("walk", "parameters", "build", "covers", "spaces"), for timing.
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
        description: build::description_from_parameters(&saved.entries_for(e)),
    });
    let mut net = build::build(duct_elements, &decoded);
    lap("build");
    covers(&mut file, &mut net);
    lap("covers");
    net.spaces = spaces(&mut file);
    lap("spaces");
    Ok(net)
}

/// What covers each element, by its id. A lining with no thickness is left
/// out: the attenuation needs one.
fn covers(file: &mut RevitFile, net: &mut Network) {
    use rvt::elements::insulation::{coverings_lossy, CoveringKind};
    for c in coverings_lossy(file).value {
        let Some(host) = c.host_id else { continue };
        let covers = net.covers.entry(host.get()).or_default();
        match c.kind {
            CoveringKind::DuctInsulation => covers.insulated = true,
            CoveringKind::DuctLining => {
                if let Some(t) = c.thickness_feet.filter(|&t| t > 0.0) {
                    covers.lining_m = Some(t * build::FT);
                }
            }
            _ => {}
        }
    }
}

/// The MEP spaces, named by number and name.
fn spaces(file: &mut RevitFile) -> Vec<crate::ifc::IfcSpace> {
    let m = |v: f64| v * build::FT;
    rvt::elements::spaces::spaces_lossy(file)
        .value
        .into_iter()
        .map(|s| {
            let b = s.bbox_feet;
            let name = [s.number, s.name].into_iter().flatten().collect::<Vec<_>>().join(" ");
            crate::ifc::IfcSpace { name, min: [m(b[0]), m(b[1]), m(b[2])], max: [m(b[3]), m(b[4]), m(b[5])] }
        })
        .collect()
}
