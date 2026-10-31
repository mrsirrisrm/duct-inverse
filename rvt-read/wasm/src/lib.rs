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

/// A forward-curved fan's sound power from its flow, TM 5-805-4 (eq. 10-5,
/// table 10-13): Kw + 10 lg(cfm) + 20 lg(in. wg) + 2 dB blade frequency
/// increment at 500 Hz, at an assumed 250 Pa (1 in. wg) and peak efficiency.
/// The table stops at 4 kHz; 8 kHz carries on its slope.
fn fan_estimate(flow_m3s: f64) -> [f64; N_BANDS] {
    const KW: [f64; N_BANDS] = [47.0, 43.0, 39.0, 36.0, 34.0, 32.0, 28.0, 24.0];
    const BFI: [f64; N_BANDS] = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
    let cfm = flow_m3s / 0.000_471_947;
    std::array::from_fn(|i| KW[i] + 10.0 * cfm.log10() + BFI[i])
}

fn bands(spectrum: impl Iterator<Item = f64>) -> Vec<f64> {
    spectrum.map(|v| (v * 10.0).round() / 10.0).collect()
}

/// The report for one Revit file's bytes, with its IFC export's if given, or
/// for an IFC on its own. Without a Revit file, `recorded` is what the IFC's
/// ports and ends-inside joins give. `source` is the unit's discharge sound
/// power, 63 Hz to 8 kHz, where its data gives one.
pub fn analyse_bytes(
    bytes: Option<Vec<u8>>,
    ifc_bytes: Option<&[u8]>,
    source: Option<[f64; N_BANDS]>,
    control: &rvt::control::WalkerControl,
) -> Result<Value, String> {
    let mut stage = clock_ms();
    let mut lap = || {
        let t = clock_ms();
        let ms = t - stage;
        stage = t;
        ms.round()
    };
    if bytes.is_none() && ifc_bytes.is_none() {
        return Err("give a Revit file, an IFC, or both".into());
    }
    let from_rvt = bytes.is_some();
    let mut decode = serde_json::Map::new();
    let mut net = match bytes {
        Some(bytes) => rvt_native::network_with_laps(bytes, control, &mut |name| {
            decode.insert(name.into(), lap().into());
        })
        .map_err(|e| e.to_string())?,
        None => Network::default(),
    };
    let mut warnings = Vec::new();
    let mut recorded = connectivity(&net);
    let ifc = ifc_bytes.map(|b| {
        let elements = ifc::parse(b);
        let parse_ms = lap();
        let m = ifc::merge(&mut net, &elements);
        json!({
            "parse_ms": parse_ms, "elements": m.elements, "matched": m.matched, "duct_geometry": m.geometry,
            "added": m.added, "ports": m.ports, "joined_inside": m.joined, "connectivity": connectivity(&net),
        })
    });
    if !from_rvt {
        recorded = connectivity(&net);
    }
    let takeoffs = heal::join_takeoffs(&mut net);
    let healed = connectivity(&net);
    let heal_ms = lap();

    let Some(cp) = path::critical_path(&net) else {
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
    // No sound data in the model; without the unit's own, an estimate from its flow.
    let lw = source.unwrap_or_else(|| fan_estimate(q_unit));
    let params = |mapped: &[chain::Mapped]| Params {
        source_type: SourceType::Given(lw),
        source_offset_db: 0.0,
        volume_flow_m3s: q_unit,
        room_volume_m3: 50.0,
        receiver_distance_m: 2.0,
        blocks: mapped.iter().map(|m| m.block).collect(),
    };
    // The critical terminal is the loudest, not the farthest: each division
    // on the way out takes power off. Rate every one the unit serves.
    let rated = |path: &[usize]| {
        let mapped = chain::chain(&net, path, &cp.flow);
        let sys = (mapped.len() <= MAX_BLOCKS).then(|| DuctSystem::from_params(&params(&mapped)))?;
        let loudest = sys.receiver_spl_values().into_iter().fold(f64::MIN, f64::max);
        Some((sys.nc(), loudest, mapped))
    };
    let candidates: Vec<_> = cp
        .tree
        .order
        .iter()
        .filter(|&&t| net.nodes[t].kind == NodeKind::Terminal && net.nodes[t].flow.is_some_and(|q| q > 0.0))
        .filter_map(|&t| cp.tree.path_to(t).map(|p| (t, p)))
        .collect();
    let terminals_rated = candidates.len();
    let Some((terminal, path, (_, _, mapped))) = candidates
        .into_iter()
        .filter_map(|(t, p)| rated(&p).map(|r| (t, p, r)))
        .max_by(|(_, _, a), (_, _, b)| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap())
    else {
        return Err(format!("every path from the unit maps to more than the engine's {MAX_BLOCKS} blocks"));
    };
    let path_ms = lap();
    for &n in &path {
        let node = &net.nodes[n];
        let ty = classify_node(node);
        match ty {
            Some(ElementType::Plenum) => {
                warnings.push(format!("{} {} ({}) is on the path but not modelled", ty.unwrap().label(), node.id, node.family))
            }
            None if node.kind == NodeKind::Fitting => {
                warnings.push(format!("fitting {} ({} | {}) is unclassified and passed through", node.id, node.family, node.type_name))
            }
            _ => {}
        }
    }
    let sys = DuctSystem::from_params(&params(&mapped));

    let mut q = sys.volume_flow();
    let mut pwl = source_pwl(sys.source_type, sys.source_offset);
    let mut blocks = Vec::new();
    for (k, (block, m)) in sys.blocks.iter().zip(&mapped).enumerate() {
        q = block.flow_out(q);
        pwl = block.forward(&pwl, q);
        let velocity = match m.block {
            BlockParams::Segment { width_m, height_m, round, .. }
            | BlockParams::Bend { width_m, height_m, round, .. }
            | BlockParams::Diffuser { width_m, height_m, round, .. } => {
                Some(q.value / if round { std::f64::consts::FRAC_PI_4 * width_m * width_m } else { width_m * height_m })
            }
            BlockParams::Attenuator { width_m, height_m, .. } => Some(q.value / (width_m * height_m)),
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
            "terminal": { "id": net.nodes[terminal].id, "family": net.nodes[terminal].family },
            "terminals_rated": terminals_rated,
            "elements": path.len(),
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
            if source.is_some() { "source: the unit's sound power as given" } else { "source: a forward-curved fan at the unit's flow and 250 Pa (no sound data given)" },
            "room: 50 m3, receiver 2 m, Schultz's equation for a normally furnished room",
            "ducts unlined, round where the IFC profile or the type name says so; a tee's loss is its power division; transitions not modelled",
            "regen of bends, tees and dampers by Woods' method; of diffusers by CIBSE Guide B, pressure loss factor 4; silencers' own noise neglected",
            "end reflection: diffusers flush with a hard ceiling (a tile ceiling would lose ~2 dB more at 63-125 Hz)",
            "dampers open; a VAV box is its damper, 15° closed",
            "silencers: length from the IFC, section from the duct, free area 0.5 and 100 mm splitters assumed",
            "element types from family-name rules; names they leave are listed under classified",
        ],
        "warnings": warnings,
    }))
}

/// JS entry point: `analyse(rvt?: Uint8Array, ifc?: Uint8Array, lw?:
/// Float64Array) -> string` (the JSON report), a model file at least, `lw`
/// the unit's sound power in the 8 bands. Run it in a Web Worker: a 40 MB
/// model takes 10 to 20 s.
#[wasm_bindgen]
pub fn analyse(bytes: Option<Vec<u8>>, ifc: Option<Vec<u8>>, lw: Option<Vec<f64>>) -> Result<String, JsValue> {
    let source = lw.map(|v| <[f64; N_BANDS]>::try_from(v).map_err(|_| JsValue::from_str("give the unit's sound power in 8 bands, 63 Hz to 8 kHz"))).transpose()?;
    let report = analyse_bytes(bytes, ifc.as_deref(), source, &rvt::control::WalkerControl::new()).map_err(|e| JsValue::from_str(&e))?;
    Ok(report.to_string())
}

#[cfg(test)]
mod tests {
    /// TM 5-805-4's worked example: 20,000 cfm at 2.5 in. wg with a 6 dB
    /// efficiency correction gives 104 100 96 95 91 89 85; at 1 in. wg and
    /// peak efficiency that is 14 dB (20 lg 2.5 + 6) less.
    #[test]
    fn fan_estimate_reproduces_tm_example() {
        let lw = super::fan_estimate(20_000.0 * 0.000_471_947);
        for (got, tm) in lw.iter().zip([104.0, 100.0, 96.0, 95.0, 91.0, 89.0, 85.0]) {
            assert!((got - (tm - 20.0 * 2.5f64.log10() - 6.0)).abs() < 0.05, "{got} vs {tm}");
        }
    }
}
