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

/// Whether a terminal is an outdoor termination by its family or type name
/// (a cap, louvre, hood or intake): it serves no room.
fn outdoor_termination(node: &duct_heal::network::Node) -> bool {
    const WORDS: [&str; 5] = ["cap", "louver", "louvre", "hood", "intake"];
    let name = format!("{} {}", node.family, node.type_name).to_lowercase();
    name.split(|c: char| !c.is_alphanumeric()).any(|w| WORDS.contains(&w))
}

/// The terminals on a unit's island that no rated path reaches and nothing
/// else reports (outside air is not rated by design), as issues: no design flow (an outdoor termination needs
/// none); a path longer than the engine takes; or reaching a unit only
/// through ductwork in none of its systems, by the nearest unit. With them,
/// every terminal counted once under what became of it, `other` for none.
fn unrated_issues(net: &Network, rated: &std::collections::HashSet<usize>, too_long: &[usize]) -> (Vec<Value>, Value) {
    let (_, terminals) = terminals_served(net);
    let (mut on_tree, mut outside) = (vec![false; net.nodes.len()], vec![false; net.nodes.len()]);
    for (tree, system, _) in path::unit_trees(net) {
        for &n in &tree.order {
            on_tree[n] = true;
            outside[n] |= outside_air(&system);
        }
    }
    let adj = net.neighbours();
    let nearest_unit = |t: usize| {
        let mut seen = vec![false; net.nodes.len()];
        let mut queue = std::collections::VecDeque::from([t]);
        seen[t] = true;
        while let Some(n) = queue.pop_front() {
            if n != t && net.nodes[n].kind == NodeKind::Equipment {
                return Some(n);
            }
            for &m in &adj[n] {
                if !std::mem::replace(&mut seen[m], true) {
                    queue.push_back(m);
                }
            }
        }
        None
    };
    let (mut no_flow, mut long, mut off): (Vec<i64>, Vec<i64>, Vec<(usize, Vec<usize>)>) = Default::default();
    let adj_empty = |t: usize| adj[t].is_empty();
    let mut tally = std::collections::BTreeMap::<&str, usize>::new();
    for &(t, served) in &terminals {
        let node = &net.nodes[t];
        let bucket = if !served {
            if adj_empty(t) { "ductless" } else { "unserved" }
        } else if rated.contains(&t) {
            "rated"
        } else if outside[t] || node.systems.iter().any(|s| outside_air(s)) {
            "outside_air"
        } else if !node.flow.is_some_and(|q| q > 0.0) {
            if outdoor_termination(node) {
                "outdoor_termination"
            } else {
                no_flow.push(node.id);
                "no_flow"
            }
        } else if too_long.contains(&t) {
            long.push(node.id);
            "path_too_long"
        } else if let (false, Some(u)) = (on_tree[t], nearest_unit(t)) {
            match off.iter_mut().find(|(k, _)| *k == u) {
                Some((_, v)) => v.push(t),
                None => off.push((u, vec![t])),
            }
            "off_system"
        } else {
            "other"
        };
        *tally.entry(bucket).or_default() += 1;
    }
    let mut tally = json!(tally);
    tally["total"] = json!(terminals.len());
    let mut out: Vec<Value> = off
        .into_iter()
        .map(|(u, ts)| {
            let unit = &net.nodes[u];
            let q: f64 = ts.iter().filter_map(|&t| net.nodes[t].flow).sum();
            json!({
                "kind": "terminals_off_system",
                "severity": "warning",
                "elements": ts.iter().map(|&t| net.nodes[t].id).collect::<Vec<_>>(),
                "message": format!(
                    "{} terminals ({:.0} L/s) reach unit {} ({}) only through ductwork assigned to none of its air systems, so the rooms they serve are not rated",
                    ts.len(), q * 1000.0, unit.id, unit.family
                ),
                "check": "which system their ductwork belongs to, and assign it to the unit's",
            })
        })
        .collect();
    if !long.is_empty() {
        out.push(json!({
            "kind": "terminals_path_too_long",
            "severity": "warning",
            "message": format!("{} terminals' paths map to more than the engine's {MAX_BLOCKS} elements, so the rooms they serve are not rated", long.len()),
            "elements": long,
            "check": "these rooms by hand",
        }));
    }
    if !no_flow.is_empty() {
        out.push(json!({
            "kind": "terminals_no_flow",
            "severity": "warning",
            "message": format!("{} terminals connected to a unit have no design flow, so the rooms they serve are not rated", no_flow.len()),
            "elements": no_flow,
            "check": "their Flow parameter",
        }));
    }
    (out, tally)
}

/// Units that start no rated path but whose ductwork joins units that do (a
/// dedicated outdoor air unit feeding heat pumps): their noise reaches rooms
/// only through those, which is not rated. Units feeding the same ones are
/// one issue. Overfed units are reported apart; outside air's markers (a
/// MagiCAD connection node) are no source.
fn feeding_issues(net: &Network, rated_units: &std::collections::HashSet<usize>) -> Vec<Value> {
    let adj = net.neighbours();
    let overfed: std::collections::HashSet<usize> = path::overfed(net).into_iter().map(|o| o.unit).collect();
    let is_unit = |n: usize| net.nodes[n].kind == NodeKind::Equipment;
    let outside = |n: usize| {
        let s = &net.nodes[n].systems;
        !s.is_empty() && s.iter().all(|s| outside_air(s))
    };
    let mut groups: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
    for u in (0..net.nodes.len()).filter(|&u| is_unit(u) && !rated_units.contains(&u) && !overfed.contains(&u) && !outside(u) && !adj[u].is_empty()) {
        // Out along its ductwork, stopping at the next unit.
        let mut seen = vec![false; net.nodes.len()];
        let mut queue = std::collections::VecDeque::from([u]);
        seen[u] = true;
        let mut fed = Vec::new();
        while let Some(n) = queue.pop_front() {
            if n != u && is_unit(n) {
                if rated_units.contains(&n) {
                    fed.push(net.nodes[n].id);
                }
                continue;
            }
            for &m in &adj[n] {
                if !std::mem::replace(&mut seen[m], true) {
                    queue.push_back(m);
                }
            }
        }
        if fed.is_empty() {
            continue;
        }
        fed.sort_unstable();
        match groups.iter_mut().find(|(_, f)| *f == fed) {
            Some((units, _)) => units.push(net.nodes[u].id),
            None => groups.push((vec![net.nodes[u].id], fed)),
        }
    }
    groups
        .into_iter()
        .map(|(units, fed)| {
            let names: Vec<String> = units
                .iter()
                .map(|&id| format!("{} ({})", id, net.find(id).map_or("", |n| net.nodes[n].family.as_str())))
                .collect();
            json!({
                "kind": "unit_feeds_units",
                "severity": "warning",
                "elements": units.iter().chain(&fed).copied().collect::<Vec<_>>(),
                "message": if let [one] = names.as_slice() {
                    format!("unit {one} starts no rated path, but its ductwork joins {} rated units: where it supplies them, its noise reaches their rooms through them, and that is not rated", fed.len())
                } else {
                    format!("units {} start no rated path, but their ductwork joins {} rated units: where they supply them, their noise reaches those units' rooms through them, and that is not rated", names.join(", "), fed.len())
                },
                "check": "whether it feeds those units (then its noise into their rooms by hand), or its ductwork should carry one of its air systems",
            })
        })
        .collect()
}

/// What in the model stops it being rated as it stands, for the engineer to
/// check: each issue's kind, the elements it is about, what is wrong and
/// what to look at.
fn issues(net: &Network) -> Vec<Value> {
    let (_, terminals) = terminals_served(net);
    let adj = net.neighbours();
    // Those joined to nothing at all apart from those on ductwork that stops
    // short of a unit: most of the first return through a ceiling void.
    let (ductless, unserved): (Vec<usize>, Vec<usize>) =
        terminals.iter().filter(|(_, served)| !served).map(|&(n, _)| n).partition(|&n| adj[n].is_empty());
    let ids = |v: &[usize]| v.iter().map(|&n| net.nodes[n].id).collect::<Vec<_>>();
    let unserved = (!unserved.is_empty()).then(|| {
        json!({
            "kind": "terminals_unserved",
            "severity": "warning",
            "elements": ids(&unserved),
            "message": format!(
                "{} of the {} terminals are on ductwork that reaches no unit, so the rooms they serve are not rated",
                unserved.len(), terminals.len()
            ),
            "check": "their ductwork for a gap, or for a unit not in this model (in a linked one)",
        })
    });
    let ductless = (!ductless.is_empty()).then(|| {
        json!({
            "kind": "terminals_ductless",
            "severity": "warning",
            "elements": ids(&ductless),
            "message": format!(
                "{} of the {} terminals join no duct at all, so the rooms they serve are not rated",
                ductless.len(), terminals.len()
            ),
            "check": "whether they return through the ceiling void (no duct to model), or their ducts are missing or in a linked model",
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
        .chain(ductless)
        .collect()
}

/// Repeats of a kind as one issue, in the place of its first: what they share,
/// every element, and each one's own message under `details`.
fn grouped(issues: Vec<Value>) -> Vec<Value> {
    let headline = |kind: &str, n: usize| match kind {
        "unit_flow_mismatch" => Some(format!("{n} units' Air Flow differs from what their terminals sum to, which the ratings use")),
        "duct_velocity" => Some(format!("{n} ducts on the worst room's path run over CIBSE Guide B table 5.2's limit above a suspended ceiling for a normal space")),
        "unit_feeds_units" => Some(format!("{n} groups of units start no rated path but their ductwork joins rated units, so where they supply them their noise is not rated")),
        "terminals_off_system" => Some(format!("{n} units reach terminals only through ductwork assigned to none of their air systems, so the rooms those serve are not rated")),
        "plenum_unsized" => Some(format!("{n} plenums on the worst room's path have no extent, so their loss is left out")),
        "fitting_unclassified" => Some(format!("{n} fittings on the worst room's path are of no type the rules know, so they are passed through with neither loss nor noise")),
        _ => None,
    };
    let mut out: Vec<Value> = Vec::new();
    for issue in &issues {
        let kind = issue["kind"].as_str().unwrap_or_default();
        let same: Vec<&Value> = issues.iter().filter(|i| i["kind"] == issue["kind"]).collect();
        match headline(kind, same.len()).filter(|_| same.len() > 1) {
            Some(_) if out.iter().any(|i| i["kind"] == issue["kind"]) => {}
            Some(message) => {
                let elements: Vec<&Value> = same.iter().flat_map(|i| i["elements"].as_array().into_iter().flatten()).collect();
                let details: Vec<&Value> = same.iter().map(|i| &i["message"]).collect();
                out.push(json!({ "kind": kind, "severity": issue["severity"], "elements": elements, "message": message, "check": issue["check"], "details": details }));
            }
            None => out.push(issue.clone()),
        }
    }
    out
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
/// increment at 500 Hz, at peak efficiency against `pressure_pa`. The table
/// stops at 4 kHz; 8 kHz carries on its slope.
fn fan_estimate(flow_m3s: f64, pressure_pa: f64) -> [f64; N_BANDS] {
    const KW: [f64; N_BANDS] = [47.0, 43.0, 39.0, 36.0, 34.0, 32.0, 28.0, 24.0];
    const BFI: [f64; N_BANDS] = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
    let cfm = flow_m3s / 0.000_471_947;
    let in_wg = pressure_pa / 249.089;
    std::array::from_fn(|i| KW[i] + 10.0 * cfm.log10() + 20.0 * in_wg.log10() + BFI[i])
}

/// The fan pressure assumed for a supply, about 1 in. wg; and for a return
/// or extract, whose fan works against less ductwork and no terminal
/// devices, half that.
const SUPPLY_PA: f64 = 250.0;

/// A duct velocity no design runs at above occupied rooms (CIBSE Guide B
/// table 5.2 stops at 10 m/s, a main's 15): past it a path's flows and
/// sizes disagree.
const IMPLAUSIBLE_M_S: f64 = 20.0;
const RETURN_PA: f64 = 125.0;

/// Whether a system draws air from the rooms, by its name: English (Revit's
/// Exhaust Air is a room's extract), and MagiCAD's Finnish poistoilma.
fn returns_air(system: &str) -> bool {
    const WORDS: [&str; 4] = ["return", "exhaust", "extract", "poistoilma"];
    let name = system.to_lowercase();
    WORDS.iter().any(|w| name.contains(w))
}

/// Whether a system's terminals are outside: air discharged (MagiCAD's
/// jäteilma, coded JI; relief) or drawn in (ulkoilma, UI; outdoor air). It
/// serves no room.
fn outside_air(system: &str) -> bool {
    const WORDS: [&str; 5] = ["jäteilma", "ulkoilma", "relief", "outdoor air", "outside air"];
    let name = system.to_lowercase();
    let coded = |code: &str| name.strip_prefix(code).is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
    WORDS.iter().any(|w| name.contains(w)) || coded("ji") || coded("ui")
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
    let runs_on_ducts = heal::join_runs_on_ducts(&mut net);
    let on_ducts = heal::join_terminals_on_ducts(&mut net);
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
    let systems: Vec<_> = path::unit_systems(&net).into_iter().filter(|u| !outside_air(&u.system)).collect();
    if systems.is_empty() {
        return Ok(json!({ "recorded": recorded, "ifc": ifc, "healed": healed, "classified": classified(&net), "issues": issues, "path": null }));
    }
    // Each unit's flow, its terminals' sum, and its sound power: no sound
    // data in the model, so without the unit's own an estimate from its flow.
    let units: Vec<(f64, [f64; N_BANDS])> = systems
        .iter()
        .map(|u| {
            let q = u.flow[u.tree.root];
            // Revit's classification where its system type gives one (2
            // return air, 3 exhaust air), else the name.
            let from_rooms = match net.classifications.get(&u.system) {
                Some(&c) => c == 2 || c == 3,
                None => returns_air(&u.system),
            };
            let pressure = if from_rooms { RETURN_PA } else { SUPPLY_PA };
            (q, source.unwrap_or_else(|| fan_estimate(q, pressure)))
        })
        .collect();
    // A unit's Air Flow, where its terminals' sum is more than 5% off it.
    let mismatched = |u: usize| net.nodes[systems[u].tree.root].flow.filter(|&r| r > 0.0 && (r - units[u].0).abs() > 0.05 * units[u].0);
    // The furthest out first.
    let mut off: Vec<(usize, f64)> = (0..systems.len()).filter_map(|u| Some((u, mismatched(u)?))).collect();
    off.sort_by(|a, b| (units[b.0].0 / b.1).ln().abs().total_cmp(&(units[a.0].0 / a.1).ln().abs()));
    for (u, rated) in off {
        let (u, q_unit) = (&systems[u], units[u].0);
        let unit = &net.nodes[u.tree.root];
        issues.push(json!({
            "kind": "unit_flow_mismatch",
            "severity": "warning",
            "elements": [unit.id],
            "message": format!(
                "unit {} Air Flow is {:.0} L/s; its {} terminals sum to {:.0} L/s, which the rating uses",
                unit.id, rated * 1000.0, u.system, q_unit * 1000.0
            ),
            "check": "the unit's Air Flow and its terminals' design flows",
        }));
    }
    let params = |unit: usize, mapped: &[chain::Mapped], room_volume_m3: f64, receiver_distance_m: f64| Params {
        source_type: SourceType::Given(units[unit].1),
        source_offset_db: 0.0,
        volume_flow_m3s: units[unit].0,
        room_volume_m3,
        receiver_distance_m,
        lined_fit: Default::default(),
        blocks: mapped.iter().map(|m| m.block).collect(),
    };
    // Every terminal each unit serves, its power into its room: each division
    // on the way out takes power off, so the farthest is seldom the loudest.
    // A terminal two units' trees reach is rated under the first.
    let mut taken = vec![false; net.nodes.len()];
    let (mut paths, mut too_long) = (Vec::new(), Vec::new());
    for (u, unit) in systems.iter().enumerate() {
        let mut children = vec![Vec::new(); net.nodes.len()];
        for &n in &unit.tree.order {
            if let Some(p) = unit.tree.parent[n] {
                children[p].push(n);
            }
        }
        for &t in &unit.tree.order {
            if taken[t] || net.nodes[t].kind != NodeKind::Terminal || !net.nodes[t].flow.is_some_and(|q| q > 0.0) {
                continue;
            }
            let Some(p) = unit.tree.path_to(t) else { continue };
            taken[t] = true;
            let mapped = chain::chain(&net, &p, &unit.flow, &children, bend_regen);
            if mapped.len() <= MAX_BLOCKS {
                paths.push((u, t, p, mapped));
            } else {
                too_long.push(t);
            }
        }
    }
    let terminals_rated = paths.len();
    let (unrated, terminal_tally) = unrated_issues(&net, &paths.iter().map(|p| p.1).collect(), &too_long);
    issues.extend(unrated);
    issues.extend(feeding_issues(&net, &paths.iter().map(|p| systems[p.0].tree.root).collect()));
    if paths.is_empty() {
        return Err(format!("every path from the units maps to more than the engine's {MAX_BLOCKS} blocks"));
    }
    // Each terminal's power by where it starts: the unit, then each element
    // on its path making its own noise.
    let mut by_source = Vec::new();
    let terminals: Vec<rooms::Terminal> = paths
        .iter()
        .map(|&(u, t, _, ref mapped)| {
            let sys = DuctSystem::from_params(&params(u, mapped, rooms::DEFAULT_ROOM_M3, rooms::DEFAULT_RECEIVER_M));
            let origins = std::iter::once((net.nodes[systems[u].tree.root].id, "unit")).chain(mapped.iter().map(|m| (m.element, m.block.type_name())));
            by_source.push(origins.zip(sys.diffuser_pwl_by_source()).collect::<Vec<_>>());
            let location = locate(t);
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
    for &(u, _, _, ref mapped) in &paths {
        let sys = DuctSystem::from_params(&params(u, mapped, rooms::DEFAULT_ROOM_M3, rooms::DEFAULT_RECEIVER_M));
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
    // Each path's fastest duct: past IMPLAUSIBLE_M_S its sizes and its flows
    // disagree, and the regen correlations are far out of their range.
    let fastest = |k: usize| -> Option<(f64, i64)> {
        let (u, _, _, ref mapped) = paths[k];
        mapped
            .iter()
            .filter_map(|m| {
                let (BlockParams::Segment { width_m, height_m, round, .. } | BlockParams::Bend { width_m, height_m, round, .. }) = m.block else { return None };
                let q = systems[u].flow[net.find(m.element)?];
                Some((q / if round { std::f64::consts::FRAC_PI_4 * width_m * width_m } else { width_m * height_m }, m.element))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
    };
    let implausible: Vec<Option<(f64, i64)>> = rated
        .iter()
        .map(|r| r.terminals.iter().filter_map(|&k| fastest(k)).max_by(|a, b| a.0.total_cmp(&b.0)).filter(|f| f.0 > IMPLAUSIBLE_M_S))
        .collect();
    let doubtful: Vec<usize> = (0..rated.len()).filter(|&i| implausible[i].is_some()).collect();
    if !doubtful.is_empty() {
        let (v, duct) = doubtful.iter().filter_map(|&i| implausible[i]).max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
        let name = |r: &rooms::RoomRating| r.space.map_or("a terminal in no space".to_string(), |k| spaces[k].name.clone());
        issues.insert(0, json!({
            "kind": "room_flow_implausible",
            "severity": "error",
            "elements": doubtful.iter().filter_map(|&i| implausible[i]).map(|f| f.1).collect::<Vec<_>>(),
            "message": format!(
                "{} rooms are rated through ducts over {IMPLAUSIBLE_M_S:.0} m/s (duct {duct} at {v:.0} m/s): their terminals' flows and their ducts' sizes disagree, so their NC is not a rating ({}), and the worst room is taken from the rest",
                doubtful.len(),
                doubtful.iter().take(5).map(|&i| format!("{} NC {}", name(&rated[i]), rated[i].nc)).collect::<Vec<_>>().join(", ")
            ),
            "check": "the terminals' design flows and the duct sizes on these paths (the elements are each room's fastest duct)",
        }));
    }
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
        let id = |k: usize| net.nodes[paths[k].1].id;
        let (parts, breakout) = controls(r);
        json!({
            "space": r.space.map(|k| spaces[k].name.clone()),
            "volume_m3": (r.volume_m3 * 10.0).round() / 10.0,
            "terminals": r.terminals.len(),
            "nc": r.nc,
            "flows_implausible": rated.iter().position(|x| std::ptr::eq(x, r)).is_some_and(|i| implausible[i].is_some()),
            "breakout": { "runs": r.breakout_runs, "share_at_band": (breakout * 1000.0).round() / 1000.0 },
            "receiver_under": id(r.receiver_under),
            "dominant": { "terminal": id(r.dominant), "distance_m": (r.dominant_distance_m * 100.0).round() / 100.0 },
            "controls": parts,
        })
    };
    // Every rated terminal with its room's rating, for colouring a model by
    // NC: the rooms' list stops at 20.
    let mut room_nc = vec![None; paths.len()];
    for r in &rated {
        for &k in &r.terminals {
            room_nc[k] = Some(r.nc);
        }
    }
    let terminals_json: Vec<Value> = paths
        .iter()
        .zip(&terminals)
        .zip(&room_nc)
        .map(|((&(u, t, ..), terminal), nc)| {
            json!({
                "id": net.nodes[t].id,
                "unit": net.nodes[systems[u].tree.root].id,
                "system": systems[u].system,
                "space": terminal.space.map(|k| spaces[k].name.clone()),
                "room_nc": nc,
            })
        })
        .collect();
    let room = (0..rated.len()).find(|&i| implausible[i].is_none()).map_or(&rated[0], |i| &rated[i]);
    let (worst, _) = controls(room);
    let loudest_unit = worst["parts"].as_array().unwrap().iter().find(|p| p["as"] == "unit");
    let unit_share = loudest_unit.and_then(|p| p["share"].as_f64()).unwrap_or(0.0);
    if let (None, Some(loudest_unit), true) = (source, loudest_unit, unit_share > 0.5) {
        issues.push(json!({
            "kind": "unit_sound_power_estimated",
            "severity": "warning",
            "elements": [loudest_unit["element"]],
            "message": format!(
                "the worst room, {}, is set at {:.0} Hz by unit {}'s own noise ({:.0}% of it), and the model gives no sound power for the unit: it is estimated from the unit's flow",
                room.space.map_or("a terminal in no space".to_string(), |k| spaces[k].name.clone()),
                worst["band_hz"].as_f64().unwrap(), loudest_unit["element"], unit_share * 100.0
            ),
            "check": "the unit's sound power data, entered as its sound power",
        }));
    }
    let (u, terminal, path, mapped) = &paths[room.dominant];
    let (u, terminal) = (*u, *terminal);
    let unit = &net.nodes[systems[u].tree.root];
    let q_unit = units[u].0;
    // The rating rests on the flows: with the unit's own, regenerated noise
    // would be 55 lg of their ratio away.
    if let Some(rated) = mismatched(u) {
        let at = issues.iter().position(|i| i["severity"] != "error").unwrap_or(issues.len());
        issues.insert(at, json!({
            "kind": "worst_room_flow_mismatch",
            "severity": "warning",
            "elements": [unit.id],
            "message": format!(
                "the worst room, {}, is rated at the {:.0} L/s its terminals sum to, through unit {} whose Air Flow is {:.0} L/s: its rating depends on which is right",
                room.space.map_or("a terminal in no space".to_string(), |k| spaces[k].name.clone()),
                q_unit * 1000.0, unit.id, rated * 1000.0
            ),
            "check": "the system's terminal flows, the unit's Air Flow and the duct sizes they imply",
        }));
    }
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
    let sys = DuctSystem::from_params(&params(u, mapped, room.volume_m3, room.dominant_distance_m));

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
        "terminal_tally": terminal_tally,
        "terminals_on_ducts": on_ducts,
        "runs_on_ducts": runs_on_ducts,
        "healed": healed,
        "classified": classified(&net),
        "issues": grouped(issues),
        "path": {
            "unit": { "id": unit.id, "family": unit.family, "air_flow_l_s": unit.flow.map(|q| q * 1000.0) },
            "system": systems[u].system,
            "units_rated": systems.len(),
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
        "terminals": terminals_json,
        "assumptions": [
            if source.is_some() { "source: every unit's sound power as given" } else { "source: each unit's system a forward-curved fan at its flow, 250 Pa for a supply and 125 Pa for a return or extract, by the Revit system type's classification, else the system's name (no sound data given)" },
            "systems whose terminals are outside, discharging or drawing in outdoor air by their names, are not rated",
            "terminals with no design flow named as caps, louvres, hoods or intakes are outdoor terminations, serving no room",
            if revit_rooms { "rooms from the Revit model's MEP spaces, their volume their extent's" } else { "rooms from the IFC's spaces, their volume their extent's" },
            "rooms: a receiver at 1.2 m under each diffuser, every diffuser in the room reaching it by Schultz's equation (a normally furnished room) at its distance; the room rated at its worst receiver",
            "a terminal in no space: rated alone, in 50 m3 at 2 m",
            "breakout: each straight on a rated path over a room radiates through its walls (ASHRAE 1991, 0.8 mm galvanised steel) and a fibreglass tile ceiling (TM 5-805-4 table 7-1), from its midpoint",
            "ducts lined where the Revit model lines them (Reynolds and Bevirt), bends too (TM 5-805-4 table 7-6), else unlined; round where the IFC profile says so, else by the Revit duct's type name or MagiCAD's product code or outside area, else where its section is square (as Revit records a diameter); a rectangular one insulated outside (a Revit insulation or an IFC covering) loses twice the bare figures (TM 5-805-4); transitions not modelled",
            "branches: power divides by the outlets' areas, the first duct below each or a terminal's neck (its Duct Width and Height, or a metric type name's), with ASHRAE's reflection below the main's cut-on; by flow where an outlet has neither",
            match bend_regen {
                chain::BendRegen::Recorded => "regen of bends by CIBSE Guide B's detailed method where the IFC records a loss coefficient, else by Woods' method; of dampers by Woods'; of junctions by ASHRAE (Reynolds and Bledsoe), sharp edged, the velocity ratio held at 5; of diffusers by CIBSE Guide B, pressure loss factor 4; silencers' own noise neglected",
                chain::BendRegen::Cibse => "regen of bends by CIBSE Guide B's detailed method, at the IFC's loss coefficient or else a typical one (mitred 1.2, vaned 0.3, radiused 0.25); of dampers by Woods'; of junctions by ASHRAE (Reynolds and Bledsoe), sharp edged, the velocity ratio held at 5; of diffusers by CIBSE Guide B, pressure loss factor 4; silencers' own noise neglected",
                chain::BendRegen::Woods => "regen of bends and dampers by Woods' method; of junctions by ASHRAE (Reynolds and Bledsoe), sharp edged, the velocity ratio held at 5; of diffusers by CIBSE Guide B, pressure loss factor 4; silencers' own noise neglected",
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
    /// Repeats of a kind become one issue in its first's place, with every
    /// element and message; a lone one, and an overfed unit with its joins,
    /// stay as they are.
    #[test]
    fn repeated_issues_grouped() {
        let issue = |kind: &str, id: i64| serde_json::json!({ "kind": kind, "severity": "warning", "elements": [id], "message": format!("m{id}"), "check": "c" });
        let lone = issue("duct_velocity", 4);
        let overfed = [issue("unit_overfed", 5), issue("unit_overfed", 6)];
        let out = super::grouped(vec![overfed[0].clone(), issue("unit_flow_mismatch", 1), lone.clone(), issue("unit_flow_mismatch", 2), overfed[1].clone()]);
        assert_eq!(out.len(), 4);
        assert_eq!((&out[0], &out[2], &out[3]), (&overfed[0], &lone, &overfed[1]));
        assert_eq!(out[1]["elements"], serde_json::json!([1, 2]));
        assert_eq!(out[1]["details"], serde_json::json!(["m1", "m2"]));
        assert!(out[1]["message"].as_str().unwrap().starts_with("2 units'"));
    }

    /// Table 5.2 above a suspended ceiling: rectangular 3, 5 and 6 m/s, round
    /// 5, 7 and 10; a main 50 % more.
    /// A unit reaching a grille only through unnamed ductwork, and a terminal
    /// with no flow, are reported; an outdoor cap with none is not.
    #[test]
    fn unrated_terminals_reported() {
        use super::*;
        use duct_heal::{EdgeSource, Node};
        let mut net = Network::default();
        let node = |id, kind, systems: &[&str], flow, family: &str| Node {
            id, kind, family: family.into(), type_name: String::new(),
            systems: systems.iter().map(|s| s.to_string()).collect(),
            location: None, flow, ifc: None, description: String::new(), extent: None,
        };
        let unit = net.add_node(node(1, NodeKind::Equipment, &["SA"], None, "AHU"));
        let supply = net.add_node(node(2, NodeKind::DuctNoGeometry, &["SA"], None, ""));
        let diffuser = net.add_node(node(3, NodeKind::Terminal, &["SA"], Some(0.1), "Diffuser"));
        let idle = net.add_node(node(4, NodeKind::Terminal, &["SA"], None, "Diffuser"));
        let cap = net.add_node(node(5, NodeKind::Terminal, &["SA"], None, "Exhaust Cap-FB"));
        let back = net.add_node(node(6, NodeKind::DuctNoGeometry, &[], None, ""));
        let grille = net.add_node(node(7, NodeKind::Terminal, &[], Some(0.1), "Return Grille"));
        for (a, b) in [(unit, supply), (supply, diffuser), (supply, idle), (supply, cap), (unit, back), (back, grille)] {
            net.connect(a, b, EdgeSource::Recorded);
        }
        let rated = [diffuser].into_iter().collect();
        let (issues, tally) = unrated_issues(&net, &rated, &[]);
        let kinds: Vec<(&str, &Value)> = issues.iter().map(|i| (i["kind"].as_str().unwrap(), &i["elements"])).collect();
        assert_eq!(kinds, [("terminals_off_system", &serde_json::json!([7])), ("terminals_no_flow", &serde_json::json!([4]))]);
        // Every terminal under one head, the cap with no issue among them.
        assert_eq!(tally, serde_json::json!({ "rated": 1, "no_flow": 1, "outdoor_termination": 1, "off_system": 1, "total": 4 }));
    }

    /// A unit starting no rated path whose ductwork joins a rated one is
    /// reported, once with any other feeding the same; an outdoor air
    /// marker is not.
    #[test]
    fn unit_feeding_rated_units_reported() {
        use super::*;
        use duct_heal::{EdgeSource, Node};
        let mut net = Network::default();
        let node = |id, kind, systems: &[&str]| Node {
            id, kind, family: String::new(), type_name: String::new(),
            systems: systems.iter().map(|s| s.to_string()).collect(),
            location: None, flow: None, ifc: None, description: String::new(), extent: None,
        };
        let doas = net.add_node(node(1, NodeKind::Equipment, &["1,3,5"]));
        let duct = net.add_node(node(2, NodeKind::DuctNoGeometry, &[]));
        let pump = net.add_node(node(3, NodeKind::Equipment, &["SA 1"]));
        let idle = net.add_node(node(4, NodeKind::Equipment, &["2,4"]));
        let inlet = net.add_node(node(5, NodeKind::Equipment, &["Outdoor Air"]));
        for (a, b) in [(doas, duct), (duct, pump), (idle, duct), (inlet, duct)] {
            net.connect(a, b, EdgeSource::Recorded);
        }
        let issues = feeding_issues(&net, &[pump].into_iter().collect());
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0]["elements"], serde_json::json!([1, 4, 3]));
        assert!(feeding_issues(&net, &Default::default()).is_empty());
    }

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
        let lw = super::fan_estimate(20_000.0 * 0.000_471_947, 2.5 * 249.089);
        for (got, tm) in lw.iter().zip([104.0, 100.0, 96.0, 95.0, 91.0, 89.0, 85.0]) {
            assert!((got - (tm - 6.0)).abs() < 0.05, "{got} vs {tm}");
        }
        // A return's fan at half the pressure, 6 dB less.
        let (supply, ret) = (super::fan_estimate(1.0, super::SUPPLY_PA), super::fan_estimate(1.0, super::RETURN_PA));
        assert!(supply.iter().zip(ret).all(|(s, r)| (s - r - 20.0 * 2f64.log10()).abs() < 1e-9));
    }

    /// The samples' system names: supply, from the rooms, and outside.
    #[test]
    fn return_and_extract_systems() {
        for name in ["Mechanical Return Air 2", "Mechanical Exhaust Air 1", "Poistoilma, asunto 8 (P8)", "Extract 3"] {
            assert!(super::returns_air(name) && !super::outside_air(name), "{name}");
        }
        for name in ["JI6", "UI2", "Jäteilma", "Relief Air 1", "Mechanical Outdoor Air"] {
            assert!(super::outside_air(name), "{name}");
        }
        for name in ["Mechanical Supply Air 1", "Tuloilma, yleiset tilat (T1)", "JIT", "Supply"] {
            assert!(!super::returns_air(name) && !super::outside_air(name), "{name}");
        }
    }
}
