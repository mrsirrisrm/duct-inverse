//! Revit file in, downduct report out: rvt-rs decodes the elements,
//! duct-heal rebuilds the network and picks the critical supply path,
//! wall-duct-core rates it. All in one WASM module, nothing leaves the page.
//!
//! The report is JSON (see [`analyse`]); lengths in metres, flows in L/s.

pub mod chain;

use duct_heal::classify::{classify_node, ElementType};
use duct_heal::{graph, heal, ifc, path, rvt_native, Network, NodeKind};
use serde_json::{json, Value};
use wall_duct_core::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};
use wall_duct_core::model::{BlockParams, DuctSystem, Params, MAX_BLOCKS};
use wall_duct_core::physics::source::{source_pwl, SourceType};
use wasm_bindgen::prelude::*;

fn connectivity(net: &Network) -> Value {
    let island = graph::islands(net);
    let islands = island.iter().collect::<std::collections::HashSet<_>>().len();
    let reach = graph::equipment_reach(net, &island);
    let served: std::collections::HashSet<usize> = reach
        .iter()
        .filter(|r| r.terminals > 0)
        .map(|r| island[r.equipment])
        .collect();
    let terminals = net.nodes.iter().zip(&island).filter(|(n, _)| n.kind == NodeKind::Terminal);
    let (total, reached) = terminals.fold((0, 0), |(t, r), (_, i)| (t + 1, r + served.contains(i) as usize));
    json!({ "edges": net.edges.len(), "islands": islands, "terminals": total, "terminals_with_equipment": reached })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Date)]
    fn now() -> f64;
}

/// Milliseconds since an arbitrary origin.
fn clock_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    return now();
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static ORIGIN: OnceLock<std::time::Instant> = OnceLock::new();
        ORIGIN.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

/// Rule coverage over the fittings, terminals and equipment, by element, and
/// the distinct names the rules left: the classifier's queue.
fn classified(net: &Network) -> Value {
    let mut left = std::collections::BTreeMap::<(&str, &str), usize>::new();
    let (mut total, mut named) = (0, 0);
    for n in net.nodes.iter().filter(|n| matches!(n.kind, NodeKind::Fitting | NodeKind::Terminal | NodeKind::Equipment)) {
        total += 1;
        match classify_node(n) {
            Some(_) => named += 1,
            None => *left.entry((&n.family, &n.type_name)).or_default() += 1,
        }
    }
    let queue: Vec<Value> = left.iter().map(|((f, t), k)| json!({ "family": f, "type": t, "elements": k })).collect();
    json!({ "elements": total, "by_rule": named, "for_classifier": queue })
}

fn bands(spectrum: impl Iterator<Item = f64>) -> Vec<f64> {
    spectrum.map(|v| (v * 10.0).round() / 10.0).collect()
}

/// The report for one Revit file's bytes, with its IFC export's if given.
pub fn analyse_bytes(bytes: Vec<u8>, ifc_bytes: Option<&[u8]>, control: &rvt::control::WalkerControl) -> Result<Value, String> {
    let mut stage = clock_ms();
    let mut lap = || {
        let t = clock_ms();
        let ms = t - stage;
        stage = t;
        ms.round()
    };
    let mut decode = serde_json::Map::new();
    let mut net = rvt_native::network_with_laps(bytes, control, &mut |name| {
        decode.insert(name.into(), lap().into());
    })
    .map_err(|e| e.to_string())?;
    let mut warnings = Vec::new();
    let recorded = connectivity(&net);
    let ifc = ifc_bytes.map(|b| {
        let elements = ifc::parse(b);
        let parse_ms = lap();
        let m = ifc::merge(&mut net, &elements);
        json!({
            "parse_ms": parse_ms, "elements": m.elements, "matched": m.matched, "duct_geometry": m.geometry,
            "added": m.added, "joined_inside": m.joined, "connectivity": connectivity(&net),
        })
    });
    let takeoffs = heal::join_takeoffs(&mut net);
    let healed = connectivity(&net);
    let heal_ms = lap();

    let cp = path::critical_path(&net);
    let path_ms = lap();
    let Some(cp) = cp else {
        return Ok(json!({ "recorded": recorded, "ifc": ifc, "healed": healed, "classified": classified(&net), "path": null }));
    };
    let unit = &net.nodes[cp.tree.root];
    let q_unit = cp.flow[cp.tree.root];
    if let Some(rated) = unit.flow.filter(|&r| (r - q_unit).abs() > 0.05 * q_unit) {
        warnings.push(format!(
            "unit {} Air Flow is {:.0} L/s; its {} terminals sum to {:.0} L/s",
            unit.id,
            rated * 1000.0,
            cp.system,
            q_unit * 1000.0
        ));
    }
    for &n in &cp.path {
        let node = &net.nodes[n];
        let ty = classify_node(node);
        match ty {
            Some(ElementType::VolumeDamper | ElementType::FireSmokeDamper | ElementType::VavTerminalUnit | ElementType::Plenum) => {
                warnings.push(format!("{} {} ({}) is on the path but not modelled", ty.unwrap().label(), node.id, node.family))
            }
            None if node.kind == NodeKind::Fitting => {
                warnings.push(format!("fitting {} ({} | {}) is unclassified and passed through", node.id, node.family, node.type_name))
            }
            _ => {}
        }
    }
    let mapped = chain::chain(&net, &cp.path, &cp.flow);
    if mapped.len() > MAX_BLOCKS {
        return Err(format!("critical path maps to {} blocks; the engine takes {MAX_BLOCKS}", mapped.len()));
    }
    let params = Params {
        // No sound data in the model; the FCU shape stands in for the unit.
        source_type: SourceType::FcuCeiling,
        source_offset_db: 0.0,
        volume_flow_m3s: q_unit,
        room_volume_m3: 50.0,
        rt60_s: 0.6,
        receiver_distance_m: 2.0,
        blocks: mapped.iter().map(|m| m.block).collect(),
    };
    let sys = DuctSystem::from_params(&params);

    let mut q = sys.volume_flow();
    let mut pwl = source_pwl(sys.source_type, sys.source_offset);
    let mut blocks = Vec::new();
    for (k, (block, m)) in sys.blocks.iter().zip(&mapped).enumerate() {
        q = block.flow_out(q);
        pwl = block.forward(&pwl, q);
        let velocity = match m.block {
            BlockParams::Segment { width_m, height_m, .. }
            | BlockParams::Bend { width_m, height_m }
            | BlockParams::Attenuator { width_m, height_m, .. }
            | BlockParams::Diffuser { width_m, height_m } => Some(q.value / (width_m * height_m)),
            _ => None,
        };
        blocks.push(json!({
            "index": k,
            "type": m.block.type_name(),
            "element": m.element,
            "note": m.note,
            "params": format!("{:?}", m.block),
            "flow_l_s": q.value * 1000.0,
            "velocity_m_s": velocity,
            "lw_db": bands(pwl.iter().map(|b| b.value)),
        }));
    }

    let spl = sys.receiver_spl();
    let loudest = (0..N_BANDS).max_by(|&a, &b| spl[a].value.total_cmp(&spl[b].value)).unwrap();
    let mut grads: Vec<(usize, f64)> = spl[loudest].grad.iter().copied().enumerate().filter(|(_, g)| g.abs() > 1e-9).collect();
    grads.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));

    let engine_ms = lap();
    Ok(json!({
        "timings_ms": { "decode": decode, "heal": heal_ms, "path": path_ms, "engine": engine_ms },
        "recorded": recorded,
        "ifc": ifc,
        "takeoffs": { "found": takeoffs.takeoffs, "joined": takeoffs.joined, "ambiguous": takeoffs.ambiguous, "unmatched": takeoffs.unmatched },
        "healed": healed,
        "classified": classified(&net),
        "path": {
            "unit": { "id": unit.id, "family": unit.family, "air_flow_l_s": unit.flow.map(|q| q * 1000.0) },
            "system": cp.system,
            "terminal": { "id": net.nodes[cp.terminal].id, "family": net.nodes[cp.terminal].family },
            "elements": cp.path.len(),
            "flow_l_s": q_unit * 1000.0,
            "bands_hz": OCTAVE_BANDS_HZ,
            "source_lw_db": bands(source_pwl::<1>(sys.source_type, wall_duct_core::autodiff::Dual::constant(0.0)).iter().map(|b| b.value)),
            "blocks": blocks,
            "receiver_spl_db": bands(spl.iter().map(|b| b.value)),
            "nc": sys.nc(),
            "sensitivities": {
                "band_hz": OCTAVE_BANDS_HZ[loudest],
                "top": grads.iter().take(8).map(|(slot, g)| json!({ "param": sys.slot_label(*slot), "db_per_unit": g })).collect::<Vec<_>>(),
            },
        },
        "assumptions": [
            "source: generic FCU spectrum, 0 dB offset (no sound data in the model)",
            "room: 50 m3, RT60 0.6 s, receiver 2 m",
            "ducts unlined; takeoff, tee and transition regen and loss not modelled",
            "silencers: length from the IFC, section from the duct, free area 0.5 and 100 mm splitters assumed",
            "element types from family-name rules; names they leave are listed under classified",
        ],
        "warnings": warnings,
    }))
}

/// JS entry point: `analyse(rvt: Uint8Array, ifc?: Uint8Array) -> string`
/// (the JSON report). Run it in a Web Worker: a 40 MB model takes 10 to 20 s.
#[wasm_bindgen]
pub fn analyse(bytes: &[u8], ifc: Option<Vec<u8>>) -> Result<String, JsValue> {
    let report = analyse_bytes(bytes.to_vec(), ifc.as_deref(), &rvt::control::WalkerControl::new()).map_err(|e| JsValue::from_str(&e))?;
    Ok(report.to_string())
}
