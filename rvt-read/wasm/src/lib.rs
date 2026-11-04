//! Revit file in, downduct report out: rvt-rs decodes the elements,
//! duct-heal rebuilds the network and picks the critical supply path,
//! wall-duct-core rates it. All in one WASM module, nothing leaves the page.
//!
//! The report is JSON (see [`analyse`]); lengths in metres, flows in L/s.

pub mod chain;
pub mod rooms;

use duct_heal::classify::{classify_node, ElementType};
use duct_heal::{graph, heal, ifc, path, rvt_native, Network, NodeKind};
use serde_json::{json, Value};
use wall_duct_core::metrics::nc::curve_at;
use wall_duct_core::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};
use wall_duct_core::model::{BlockParams, DuctSystem, Params, MAX_BLOCKS};
use wall_duct_core::autodiff::Dual;
use wall_duct_core::physics::breakout::{radiated_pwl_db, Ceiling, Run};
use wall_duct_core::physics::duct_attenuation::segment_attenuation_db;
use wall_duct_core::physics::source::{source_pwl, SourceType};

/// Ducts' walls, 0.8 mm galvanised steel: the model gives no gauge.
const DUCT_WALL_KG_M2: f64 = 6.3;
/// The ceiling under a duct: the model does not say, and the lightest of
/// TM 5-805-4's lets the most through.
const CEILING: Ceiling = Ceiling::FibreglassTile;
use wasm_bindgen::prelude::*;

/// The network's islands, and its terminals, each with whether its island
/// holds a unit.
fn terminals_served(net: &Network) -> (usize, Vec<(usize, bool)>) {
    let island = graph::islands(net);
    let islands = island.iter().collect::<std::collections::HashSet<_>>().len();
    let reach = graph::equipment_reach(net, &island);
    let served: std::collections::HashSet<usize> = reach
        .iter()
        .filter(|r| r.terminals > 0)
        .map(|r| island[r.equipment])
        .collect();
    let terminals = (0..net.nodes.len()).filter(|&n| net.nodes[n].kind == NodeKind::Terminal);
    (islands, terminals.map(|n| (n, served.contains(&island[n]))).collect())
}

fn connectivity(net: &Network) -> Value {
    let (islands, terminals) = terminals_served(net);
    let reached = terminals.iter().filter(|(_, served)| *served).count();
    json!({ "edges": net.edges.len(), "islands": islands, "terminals": terminals.len(), "terminals_with_equipment": reached })
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

/// What in the model stops it being rated as it stands, for the engineer to
/// check: each issue's kind, the elements it is about, what is wrong and
/// what to look at.
fn issues(net: &Network) -> Vec<Value> {
    let (_, terminals) = terminals_served(net);
    let unserved: Vec<i64> = terminals.iter().filter(|(_, served)| !served).map(|&(n, _)| net.nodes[n].id).collect();
    let unserved = (!unserved.is_empty()).then(|| {
        json!({
            "kind": "terminals_unserved",
            "severity": "warning",
            "elements": unserved,
            "message": format!(
                "{} of the {} terminals connect to no unit, so the rooms they serve are not rated",
                unserved.len(), terminals.len()
            ),
            "check": "their ductwork for a gap, or for a unit not in this model (in a linked one)",
        })
    });
    path::overfed(net)
        .into_iter()
        .map(|o| {
            let unit = &net.nodes[o.unit];
            let joins: Vec<Value> = o
                .healed_joins
                .iter()
                .take(5)
                .map(|&(a, b, q)| json!({ "from": net.nodes[a].id, "to": net.nodes[b].id, "beyond_l_s": (q * 1000.0).round() }))
                .collect();
            json!({
                "kind": "unit_overfed",
                "severity": "error",
                "elements": [unit.id],
                "message": format!(
                    "unit {} ({}) is rated {:.0} L/s, but the {} terminals its ductwork reaches take {:.0} L/s: its network is joined to air it cannot supply, so it is not rated",
                    unit.id, unit.family, o.rated * 1000.0, o.system, o.served * 1000.0
                ),
                "check": if joins.is_empty() {
                    "the unit's Air Flow, and where its ductwork meets another unit's".to_string()
                } else {
                    "these joins, found from the geometry, not the model's connectors; the one carrying the extra air should not be there, or another unit should feed what lies beyond it".to_string()
                },
                "joins": joins,
            })
        })
        .chain(unserved)
        .collect()
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
/// power, 63 Hz to 8 kHz, where its data gives one; `bend_regen` how bends'
/// regenerated noise is found.
pub fn analyse_bytes(
    bytes: Option<Vec<u8>>,
    ifc_bytes: Option<&[u8]>,
    source: Option<[f64; N_BANDS]>,
    bend_regen: chain::BendRegen,
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
    let mut recorded = connectivity(&net);
    let mut spaces = Vec::new();
    let ifc = ifc_bytes.map(|b| {
        let elements;
        (elements, spaces) = ifc::parse_with_spaces(b);
        let parse_ms = lap();
        let m = ifc::merge(&mut net, &elements);
        json!({
            "parse_ms": parse_ms, "elements": m.elements, "matched": m.matched, "duct_geometry": m.geometry,
            "added": m.added, "ports": m.ports, "joined_inside": m.joined, "connectivity": connectivity(&net),
            "spaces": spaces.len(),
        })
    });
    if !from_rvt {
        recorded = connectivity(&net);
    }
    let takeoffs = heal::join_takeoffs(&mut net);
    let healed = connectivity(&net);
    let heal_ms = lap();
    // Rooms from the IFC's spaces, in its coordinates; else from the Revit
    // model's, in its own.
    let revit_rooms = spaces.is_empty() && !net.spaces.is_empty();
    if revit_rooms {
        spaces = net.spaces.clone();
    }
    let locate = |n: usize| {
        let node = &net.nodes[n];
        match node.kind {
            _ if !revit_rooms => node.ifc.as_ref().and_then(|i| i.location),
            NodeKind::Duct { start, end, .. } => Some(std::array::from_fn(|k| (start[k] + end[k]) / 2.0)),
            _ => node.location,
        }
    };

    let mut issues = issues(&net);
    let Some(cp) = path::critical_path(&net) else {
        return Ok(json!({ "recorded": recorded, "ifc": ifc, "healed": healed, "classified": classified(&net), "issues": issues, "path": null }));
    };
    let unit = &net.nodes[cp.tree.root];
    let q_unit = cp.flow[cp.tree.root];
    if let Some(rated) = unit.flow.filter(|&r| r > 0.0 && (r - q_unit).abs() > 0.05 * q_unit) {
        issues.push(json!({
            "kind": "unit_flow_mismatch",
            "severity": "warning",
            "elements": [unit.id],
            "message": format!(
                "unit {} Air Flow is {:.0} L/s; its {} terminals sum to {:.0} L/s, which the rating uses",
                unit.id, rated * 1000.0, cp.system, q_unit * 1000.0
            ),
            "check": "the unit's Air Flow and its terminals' design flows",
        }));
    }
    // No sound data in the model; without the unit's own, an estimate from its flow.
    let lw = source.unwrap_or_else(|| fan_estimate(q_unit));
    let params = |mapped: &[chain::Mapped], room_volume_m3: f64, receiver_distance_m: f64| Params {
        source_type: SourceType::Given(lw),
        source_offset_db: 0.0,
        volume_flow_m3s: q_unit,
        room_volume_m3,
        receiver_distance_m,
        lined_fit: Default::default(),
        blocks: mapped.iter().map(|m| m.block).collect(),
    };
    // Every terminal the unit serves, its power into its room: each division
    // on the way out takes power off, so the farthest is seldom the loudest.
    let candidates: Vec<_> = cp
        .tree
        .order
        .iter()
        .filter(|&&t| net.nodes[t].kind == NodeKind::Terminal && net.nodes[t].flow.is_some_and(|q| q > 0.0))
        .filter_map(|&t| cp.tree.path_to(t).map(|p| (t, p)))
        .collect();
    let terminals_rated = candidates.len();
    let mut children = vec![Vec::new(); net.nodes.len()];
    for &n in &cp.tree.order {
        if let Some(p) = cp.tree.parent[n] {
            children[p].push(n);
        }
    }
    let paths: Vec<_> = candidates
        .into_iter()
        .filter_map(|(t, p)| {
            let mapped = chain::chain(&net, &p, &cp.flow, &children, bend_regen);
            (mapped.len() <= MAX_BLOCKS).then_some((t, p, mapped))
        })
        .collect();
    if paths.is_empty() {
        return Err(format!("every path from the unit maps to more than the engine's {MAX_BLOCKS} blocks"));
    }
    // Each terminal's power by where it starts: the unit, then each element
    // on its path making its own noise.
    let mut by_source = Vec::new();
    let terminals: Vec<rooms::Terminal> = paths
        .iter()
        .map(|(t, _, mapped)| {
            let sys = DuctSystem::from_params(&params(mapped, rooms::DEFAULT_ROOM_M3, rooms::DEFAULT_RECEIVER_M));
            let origins = std::iter::once((unit.id, "unit")).chain(mapped.iter().map(|m| (m.element, m.block.type_name())));
            by_source.push(origins.zip(sys.diffuser_pwl_by_source()).collect::<Vec<_>>());
            let location = locate(*t);
            rooms::Terminal {
                pwl: sys.diffuser_pwl().map(|b| b.value),
                location,
                space: location.and_then(|p| ifc::space_at(&spaces, p)),
            }
        })
        .collect();
    // Breakout: each straight on a rated path, once, radiating what enters it
    // down through the ceiling of the room it is over.
    let mut seen = std::collections::HashSet::new();
    let (mut sources, mut source_elements) = (Vec::new(), Vec::new());
    for (_, _, mapped) in &paths {
        let sys = DuctSystem::from_params(&params(mapped, rooms::DEFAULT_ROOM_M3, rooms::DEFAULT_RECEIVER_M));
        let (mut q, mut pwl) = (sys.volume_flow(), source_pwl(sys.source_type, sys.source_offset));
        for (block, m) in sys.blocks.iter().zip(mapped) {
            q = block.flow_out(q);
            if let BlockParams::Segment { length_m, width_m, height_m, lining_m, round, wrapped } = m.block {
                let location = net.find(m.element).and_then(locate);
                let space = location.and_then(|p| ifc::space_at(&spaces, p));
                if let (Some(location), Some(space), true) = (location, space, seen.insert(m.element)) {
                    let c = Dual::<1>::constant;
                    let per_m = segment_attenuation_db(c(1.0), c(width_m), c(height_m), c(lining_m), round, wrapped, Default::default()).map(|b| b.value);
                    let run = Run { width_m, height_m, round, length_m, wall_kg_m2: DUCT_WALL_KG_M2 };
                    let radiated = radiated_pwl_db(&pwl.map(|b| b.value), &run, &per_m);
                    let ceiling = CEILING.transfer_db();
                    sources.push(rooms::Source { pwl: std::array::from_fn(|i| radiated[i] - ceiling[i]), location, space });
                    source_elements.push(m.element);
                }
            }
            pwl = block.forward(&pwl, q);
        }
    }
    let rated = rooms::rate(&terminals, &sources, &spaces);
    // What sets a room's rating: at the band nearest its NC curve, each
    // element's share of what its worst receiver hears there; and the runs
    // breaking out over it, together.
    let controls = |r: &rooms::RoomRating| {
        let curve = curve_at(r.nc as f64);
        let band = (0..N_BANDS).max_by(|&a, &b| (r.spl[a] - curve[a]).total_cmp(&(r.spl[b] - curve[b]))).unwrap();
        let e = |l: f64| 10f64.powf(l / 10.0);
        let mut shares: Vec<((i64, &str), f64)> = Vec::new();
        let mut add = |key: (i64, &'static str), energy: f64| match shares.iter_mut().find(|(k, _)| *k == key) {
            Some((_, sum)) => *sum += energy,
            None => shares.push((key, energy)),
        };
        for (part, spl) in &r.parts {
            match *part {
                rooms::Part::Terminal(t) => {
                    let whole: f64 = by_source[t].iter().map(|(_, pwl)| e(pwl[band])).sum();
                    for &(key, pwl) in &by_source[t] {
                        add(key, e(spl[band]) * e(pwl[band]) / whole);
                    }
                }
                rooms::Part::Source(k) => add((source_elements[k], "breakout"), e(spl[band])),
            }
        }
        let total: f64 = shares.iter().map(|(_, x)| x).sum();
        let breakout: f64 = shares.iter().filter(|((_, how), _)| *how == "breakout").map(|(_, x)| x / total).sum::<f64>() + 0.0;
        shares.sort_by(|a, b| b.1.total_cmp(&a.1));
        let parts = json!({
            "band_hz": OCTAVE_BANDS_HZ[band],
            "parts": shares
                .iter()
                .take(5)
                .filter(|(_, x)| x / total >= 0.005)
                .map(|((element, how), x)| json!({ "element": element, "as": how.to_lowercase(), "share": (x / total * 1000.0).round() / 1000.0 }))
                .collect::<Vec<_>>(),
        });
        (parts, breakout)
    };
    let room_json = |r: &rooms::RoomRating| {
        let id = |k: usize| net.nodes[paths[k].0].id;
        let (parts, breakout) = controls(r);
        json!({
            "space": r.space.map(|k| spaces[k].name.clone()),
            "volume_m3": (r.volume_m3 * 10.0).round() / 10.0,
            "terminals": r.terminals.len(),
            "nc": r.nc,
            "breakout": { "runs": r.breakout_runs, "share_at_band": (breakout * 1000.0).round() / 1000.0 },
            "receiver_under": id(r.receiver_under),
            "dominant": { "terminal": id(r.dominant), "distance_m": (r.dominant_distance_m * 100.0).round() / 100.0 },
            "controls": parts,
        })
    };
    let room = &rated[0];
    let (worst, _) = controls(room);
    let unit_share = worst["parts"].as_array().unwrap().iter().find(|p| p["as"] == "unit").and_then(|p| p["share"].as_f64()).unwrap_or(0.0);
    if source.is_none() && unit_share > 0.5 {
        issues.push(json!({
            "kind": "unit_sound_power_estimated",
            "severity": "warning",
            "elements": [unit.id],
            "message": format!(
                "the worst room, {}, is set at {:.0} Hz by unit {}'s own noise ({:.0}% of it), and the model gives no sound power for the unit: it is estimated from the unit's flow",
                room.space.map_or("a terminal in no space".to_string(), |k| spaces[k].name.clone()),
                worst["band_hz"].as_f64().unwrap(), unit.id, unit_share * 100.0
            ),
            "check": "the unit's sound power data, entered as its sound power",
        }));
    }
    let (terminal, path, mapped) = &paths[room.dominant];
    let terminal = *terminal;
    let path_ms = lap();
    for &n in path {
        let node = &net.nodes[n];
        let ty = classify_node(node);
        match ty {
            Some(ElementType::Plenum) if !mapped.iter().any(|m| m.element == node.id) => issues.push(json!({
                "kind": "plenum_unsized",
                "severity": "warning",
                "elements": [node.id],
                "message": format!("plenum {} ({}) on the worst room's path has no extent, so its loss is left out", node.id, node.family),
                "check": "its size: an IFC export gives it",
            })),
            None if node.kind == NodeKind::Fitting => issues.push(json!({
                "kind": "fitting_unclassified",
                "severity": "warning",
                "elements": [node.id],
                "message": format!(
                    "fitting {} ({} | {}) on the worst room's path is of no type the rules know, so it is passed through with neither loss nor noise",
                    node.id, node.family, node.type_name
                ),
                "check": "what the fitting is: its family and type names",
            })),
            _ => {}
        }
    }
    let sys = DuctSystem::from_params(&params(mapped, room.volume_m3, room.dominant_distance_m));

    let mut q = sys.volume_flow();
    let mut pwl = source_pwl(sys.source_type, sys.source_offset);
    let mut blocks = Vec::new();
    let last_branch = mapped.iter().rposition(|m| matches!(m.block, BlockParams::Branch { .. }));
    for (k, (block, m)) in sys.blocks.iter().zip(mapped).enumerate() {
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
        let grade = match (m.block, velocity) {
            (BlockParams::Segment { round, .. }, Some(v)) => {
                let grade = velocity_grade(v, round, last_branch.is_some_and(|b| k < b));
                if matches!(grade, "non-critical" | "none") {
                    issues.push(json!({
                        "kind": "duct_velocity",
                        "severity": "warning",
                        "elements": [m.element],
                        "message": format!(
                            "duct {} on the worst room's path runs at {v:.1} m/s, over CIBSE Guide B table 5.2's limit above a suspended ceiling for a normal space{}",
                            m.element,
                            if grade == "none" { ", and a non-critical one's" } else { "" }
                        ),
                        "check": "the duct's size for its flow",
                    }));
                }
                Some(grade)
            }
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
            "velocity_grade": grade,
            "lw_db": bands(pwl.iter().map(|b| b.value)),
        }));
    }

    // The room's worst receiver, from all its diffusers; the levers are the
    // dominant one's, at its distance from that receiver.
    let spl = sys.receiver_spl();
    let loudest = (0..N_BANDS).max_by(|&a, &b| room.spl[a].total_cmp(&room.spl[b])).unwrap();
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
        "issues": issues,
        "path": {
            "unit": { "id": unit.id, "family": unit.family, "air_flow_l_s": unit.flow.map(|q| q * 1000.0) },
            "system": cp.system,
            "terminal": { "id": net.nodes[terminal].id, "family": net.nodes[terminal].family },
            "terminals_rated": terminals_rated,
            "terminals_in_spaces": terminals.iter().filter(|t| t.space.is_some()).count(),
            "elements": path.len(),
            "flow_l_s": q_unit * 1000.0,
            "bands_hz": OCTAVE_BANDS_HZ,
            "source_lw_db": bands(source_pwl::<1>(sys.source_type, Dual::constant(0.0)).iter().map(|b| b.value)),
            "blocks": blocks,
            "receiver_spl_db": bands(room.spl.iter().copied()),
            "nc": room.nc,
            "room": room_json(room),
            "sensitivities": {
                "band_hz": OCTAVE_BANDS_HZ[loudest],
                "top": grads.iter().take(8).map(|(slot, g)| json!({ "param": sys.slot_label(*slot), "db_per_unit": g })).collect::<Vec<_>>(),
            },
        },
        "rooms": rated.iter().take(20).map(&room_json).collect::<Vec<_>>(),
        "assumptions": [
            if source.is_some() { "source: the unit's sound power as given" } else { "source: a forward-curved fan at the unit's flow and 250 Pa (no sound data given)" },
            if revit_rooms { "rooms from the Revit model's MEP spaces, their volume their extent's" } else { "rooms from the IFC's spaces, their volume their extent's" },
            "rooms: a receiver at 1.2 m under each diffuser, every diffuser in the room reaching it by Schultz's equation (a normally furnished room) at its distance; the room rated at its worst receiver",
            "a terminal in no space: rated alone, in 50 m3 at 2 m",
            "breakout: each straight on a rated path over a room radiates through its walls (ASHRAE 1991, 0.8 mm galvanised steel) and a fibreglass tile ceiling (TM 5-805-4 table 7-1), from its midpoint",
            "ducts lined where the Revit model lines them (Reynolds and Bevirt), bends too (TM 5-805-4 table 7-6), else unlined; round where the IFC profile or the type name says so; a rectangular one insulated outside (a Revit insulation or an IFC covering) loses twice the bare figures (TM 5-805-4); transitions not modelled",
            "branches: power divides by the outlets' areas, the first duct below each, with ASHRAE's reflection below the main's cut-on; by flow where an outlet has no duct",
            match bend_regen {
                chain::BendRegen::Recorded => "regen of bends by CIBSE Guide B's detailed method where the IFC records a loss coefficient, else by Woods' method; of tees and dampers by Woods'; of diffusers by CIBSE Guide B, pressure loss factor 4; silencers' own noise neglected",
                chain::BendRegen::Cibse => "regen of bends by CIBSE Guide B's detailed method, at the IFC's loss coefficient or else a typical one (mitred 1.2, vaned 0.3, radiused 0.25); of tees and dampers by Woods'; of diffusers by CIBSE Guide B, pressure loss factor 4; silencers' own noise neglected",
                chain::BendRegen::Woods => "regen of bends, tees and dampers by Woods' method; of diffusers by CIBSE Guide B, pressure loss factor 4; silencers' own noise neglected",
            },
            "velocity grades: CIBSE Guide B table 5.2 for ducts above a suspended ceiling (critical/normal/non-critical spaces), mains before the path's last branch 50% more; a warning past the normal space's",
            "end reflection: diffusers flush with a hard ceiling (a tile ceiling would lose ~2 dB more at 63-125 Hz)",
            "dampers open; a VAV box is its damper, 15° closed",
            "silencers: length from the IFC, else the Revit element's box; section from the duct, free area 0.5 and 100 mm splitters assumed",
            "plenums: Wells' method, their extent from the IFC or else the Revit element's box, openings the ducts either side; walls bare sheet metal (absorption 0.05), the least loss; at most 20 dB below the inlet's cut-on",
            "element types from rules on family and type names and Revit's descriptive parameters (IfcExportAs, MagiCAD's classification), then the IFC class; names they leave are listed under classified",
        ],
    }))
}

/// The strictest space whose duct velocity limit `v` m/s meets, CIBSE Guide
/// B table 5.2, ducts above a suspended ceiling: "critical" (a private
/// office, NR 25), "normal" (a general office, NR 35), "non-critical"
/// (circulation, NR 45) or "none". The table is for final runs to outlets;
/// a `main` (before the path's last branch) may run 50 % faster.
pub fn velocity_grade(v: f64, round: bool, main: bool) -> &'static str {
    let limits = if round { [5.0, 7.0, 10.0] } else { [3.0, 5.0, 6.0] };
    let scale = if main { 1.5 } else { 1.0 };
    match limits.iter().position(|&l| v <= l * scale) {
        Some(0) => "critical",
        Some(1) => "normal",
        Some(_) => "non-critical",
        None => "none",
    }
}

/// JS entry point: `analyse(rvt?: Uint8Array, ifc?: Uint8Array, lw?:
/// Float64Array, bendRegen?: "recorded" | "cibse" | "woods") -> string` (the
/// JSON report), a model file at least, `lw` the unit's sound power in the 8
/// bands. Run it in a Web Worker: a 40 MB model takes 10 to 20 s.
#[wasm_bindgen]
pub fn analyse(bytes: Option<Vec<u8>>, ifc: Option<Vec<u8>>, lw: Option<Vec<f64>>, bend_regen: Option<String>) -> Result<String, JsValue> {
    let source = lw.map(|v| <[f64; N_BANDS]>::try_from(v).map_err(|_| JsValue::from_str("give the unit's sound power in 8 bands, 63 Hz to 8 kHz"))).transpose()?;
    let bend_regen = bend_regen.map(|s| s.parse()).transpose().map_err(|e: String| JsValue::from_str(&e))?.unwrap_or_default();
    let report = analyse_bytes(bytes, ifc.as_deref(), source, bend_regen, &rvt::control::WalkerControl::new()).map_err(|e| JsValue::from_str(&e))?;
    Ok(report.to_string())
}

#[cfg(test)]
mod tests {
    /// Table 5.2 above a suspended ceiling: rectangular 3, 5 and 6 m/s, round
    /// 5, 7 and 10; a main 50 % more.
    #[test]
    fn velocity_grades() {
        use super::velocity_grade;
        assert_eq!(velocity_grade(4.0, false, false), "normal");
        assert_eq!(velocity_grade(4.0, false, true), "critical");
        assert_eq!(velocity_grade(6.5, false, false), "none");
        assert_eq!(velocity_grade(8.0, true, false), "non-critical");
        assert_eq!(velocity_grade(14.0, true, true), "non-critical");
    }

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
