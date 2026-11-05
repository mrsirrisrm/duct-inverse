//! The network read from a `.rvt` alone, against Revit's own IFC export of
//! the same model: the BIM4LCA housing's HVAC (Granlund, Revit 2024). Needs
//! the pair in `DUCT_SAMPLES` (the repo's `samples/`, never committed);
//! skips without.
#![cfg(feature = "rvt")]

use duct_heal::classify::{classify_node, ElementType};
use duct_heal::{ifc, rvt_native, Network, NodeKind};

/// The Revit network on its own, and the IFC's elements.
fn granlund() -> Option<(Network, Vec<ifc::IfcElement>)> {
    let Ok(dir) = std::env::var("DUCT_SAMPLES") else {
        eprintln!("DUCT_SAMPLES unset: skipped");
        return None;
    };
    let stem = format!("{dir}/HVAC_NordicLCA_Housing_Concrete_BuildingPermit");
    let (Ok(rvt), Ok(ifc_bytes)) = (std::fs::read(format!("{stem}.rvt")), std::fs::read(format!("{stem}.ifc"))) else {
        eprintln!("Granlund pair not in {dir}: skipped");
        return None;
    };
    let net = rvt_native::network(rvt, &rvt::control::WalkerControl::new()).expect("network");
    Some((net, ifc::parse(&ifc_bytes)))
}

/// Every element and join the IFC has, the Revit file has: merging adds none.
#[test]
fn network_is_the_ifcs() {
    let Some((mut net, elements)) = granlund() else { return };
    let edges = net.edges.len();
    let merged = ifc::merge(&mut net, &elements);
    assert_eq!((merged.added, merged.joined), (0, 0));
    assert_eq!(net.edges.len(), edges);
}

/// Over the Revit elements the export tags: insulated in one, insulated in
/// the other.
#[test]
fn insulation_matches_the_ifc() {
    let Some((mut net, elements)) = granlund() else { return };
    ifc::merge(&mut net, &elements);
    let (mut both, mut differ) = (0, Vec::new());
    for n in net.nodes.iter().filter(|n| n.id > 0) {
        let Some(info) = &n.ifc else { continue };
        let covers = net.covers.get(&n.id).copied().unwrap_or_default();
        match (covers.insulated, info.insulated) {
            (true, true) => both += 1,
            (false, false) => {}
            _ => differ.push(n.id),
        }
    }
    assert!(differ.is_empty(), "insulated in one only: {differ:?}");
    assert!(both > 100, "{both} insulated");
    assert!(net.covers.values().all(|c| c.lining_m.is_none()), "no lining in this model");
}

/// A straight Revit sizes is the IFC's extrusion: its section to 2 mm and
/// its length to 10 mm. 697 of the 701 are: along an axis, or round and
/// sloping; the 4 left are rectangular and off the axes.
#[test]
fn ducts_match_the_ifc() {
    let Some((net, elements)) = granlund() else { return };
    let d = |p: [f64; 3], q: [f64; 3]| (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>().sqrt();
    let (mut straights, mut sized) = (0, 0);
    for e in &elements {
        let Some(ifc::Shape::Axis { start, end, width, height, .. }) = e.shape else { continue };
        let Some(i) = e.tag.and_then(|t| net.find(t)) else { continue };
        straights += 1;
        let NodeKind::Duct { start: s, end: t, width: w, height: h, .. } = net.nodes[i].kind else { continue };
        sized += 1;
        let (mut ours, mut theirs) = ([w, h], [width, height]);
        ours.sort_by(f64::total_cmp);
        theirs.sort_by(f64::total_cmp);
        assert!((0..2).all(|k| (ours[k] - theirs[k]).abs() < 0.002), "{:?}: {ours:?} for {theirs:?}", e.tag);
        assert!((d(s, t) - d(start, end)).abs() < 0.01, "{:?}: {} m for {} m", e.tag, d(s, t), d(start, end));
    }
    assert_eq!((straights, sized), (701, 697));
}

/// From their Revit parameters alone, dampers and silencers are what the
/// export's classes say.
#[test]
fn accessories_classify_as_the_ifc() {
    let Some((net, elements)) = granlund() else { return };
    let mut checked = 0;
    for e in &elements {
        let want = match (e.class.as_str(), e.predefined.as_str()) {
            ("IFCDUCTSILENCER", _) => ElementType::Silencer,
            ("IFCDAMPER", "FIREDAMPER") => ElementType::FireSmokeDamper,
            ("IFCDAMPER", "BALANCINGDAMPER") => ElementType::VolumeDamper,
            _ => continue,
        };
        let node = &net.nodes[net.find(e.tag.expect("tag")).expect("in the Revit network")];
        assert!(node.ifc.is_none());
        assert_eq!(classify_node(node), Some(want), "{} {}", node.family, node.description);
        checked += 1;
    }
    assert_eq!(checked, 66);
}

/// Each air system's classification, from its Revit type, is what its
/// MagiCAD name says: tuloilma and ulkoilma or raitisilma (outdoor air) 1,
/// supply; poistoilma 2, extract, as the transfer system S1; jäteilma (JI)
/// 3, exhaust.
#[test]
fn systems_classified_as_named() {
    let Some((net, _)) = granlund() else { return };
    let air: Vec<_> = net.classifications.iter().filter(|(name, _)| !name.is_empty()).collect();
    assert_eq!(air.len(), 36);
    for (name, &c) in air {
        let want = match name.as_str() {
            n if n.starts_with("Tuloilma") || n.starts_with("Ulkoilma") || n.starts_with("Raitisilma") => 1,
            n if n.starts_with("Poistoilma") || n == "S1" => 2,
            n if n.starts_with("JI") => 3,
            n => panic!("{n}: {c}"),
        };
        assert_eq!(c, want, "{name}");
    }
}
