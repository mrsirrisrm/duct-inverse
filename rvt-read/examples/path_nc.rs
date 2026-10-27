//! Run one extracted supply path through `wall-duct-core`.
//!
//!     cargo run --example path_nc -- <elements.json>
//!
//! Same path as `path_table`: the unit and supply system serving the most
//! terminals with a flow, out to its farthest terminal. Element types come
//! from family names by keyword here; in the tool that is the classifier's
//! job (eval/).
use duct_heal::{heal, path, rvt_json, Network, NodeKind};
use wall_duct_core::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};
use wall_duct_core::model::{BlockParams, DuctSystem, Params};
use wall_duct_core::physics::source::{source_pwl, SourceType};

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Nearest decoded duct section along the path from `i`, looking downstream
/// first (a fitting's outlet sets its velocity).
fn section(net: &Network, path: &[usize], i: usize) -> Option<(f64, f64)> {
    let sized = |&n: &usize| match net.nodes[n].kind {
        NodeKind::Duct { width, height, .. } => Some((width, height)),
        _ => None,
    };
    path[i..].iter().find_map(sized).or_else(|| path[..i].iter().rev().find_map(sized))
}

/// The chain for `path`, with a note per block saying where it came from.
fn chain(net: &Network, path: &[usize], flow: &[f64]) -> (Vec<BlockParams>, Vec<String>) {
    let (mut blocks, mut notes) = (Vec::new(), Vec::new());
    for (i, &n) in path.iter().enumerate().skip(1) {
        let node = &net.nodes[n];
        let prev = path[i - 1];
        if flow[n] < flow[prev] * (1.0 - 1e-9) {
            blocks.push(BlockParams::Branch { flow_ratio: flow[n] / flow[prev] });
            notes.push(format!("{:.0} -> {:.0} L/s", flow[prev] * 1000.0, flow[n] * 1000.0));
        }
        let family = node.family.to_ascii_lowercase();
        match node.kind {
            NodeKind::Duct { start, end, width, height } => {
                blocks.push(BlockParams::Segment { length_m: dist(start, end), width_m: width, height_m: height, lining_m: 0.0 });
                notes.push(format!("duct {}", node.id));
            }
            // Undecoded straights and flex: length from the elements either side.
            NodeKind::DuctNoGeometry | NodeKind::Undecoded => {
                let ends = (net.nodes[prev].location, path.get(i + 1).and_then(|&m| net.nodes[m].location));
                if let ((Some(a), Some(b)), Some((w, h))) = (ends, section(net, path, i)) {
                    blocks.push(BlockParams::Segment { length_m: dist(a, b).max(0.01), width_m: w, height_m: h, lining_m: 0.0 });
                    notes.push(format!("duct {} (length from neighbours)", node.id));
                }
            }
            NodeKind::Fitting if family.contains("elbow") || family.contains("bend") => {
                let (w, h) = section(net, path, i).expect("a sized duct on the path");
                blocks.push(BlockParams::Bend { width_m: w, height_m: h });
                notes.push(node.family.clone());
            }
            NodeKind::Terminal => {
                let (w, h) = section(net, path, i).expect("a sized duct on the path");
                blocks.push(BlockParams::Diffuser { width_m: w, height_m: h });
                notes.push(format!("{} (neck from last duct)", node.family));
            }
            // Takeoffs, tees and transitions: division is the Branch above;
            // their own loss and regen are not modelled yet.
            _ => {}
        }
    }
    (blocks, notes)
}

fn main() {
    let file = std::env::args().nth(1).expect("usage: path_nc <elements.json>");
    let json = serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap();
    let mut net = rvt_json::parse(&json);
    heal::join_takeoffs(&mut net);

    let with_flow = |tree: &path::Tree| {
        tree.order
            .iter()
            .filter(|&&n| net.nodes[n].kind == NodeKind::Terminal && net.nodes[n].flow.is_some_and(|q| q > 0.0))
            .count()
    };
    let (tree, system) = net
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.kind == NodeKind::Equipment)
        .flat_map(|(e, n)| n.systems.iter().filter(|s| s.contains("Supply")).map(move |s| (e, s.clone())))
        .map(|(e, s)| (path::tree(&net, e, &s), s))
        .max_by_key(|(tree, _)| with_flow(tree))
        .expect("no unit with a supply system");
    let terminal = *tree
        .order
        .iter()
        .filter(|&&n| net.nodes[n].kind == NodeKind::Terminal && net.nodes[n].flow.is_some_and(|q| q > 0.0))
        .max_by_key(|&&t| tree.path_to(t).unwrap().len())
        .unwrap();
    let path = tree.path_to(terminal).unwrap();
    let flow = tree.flows(&net);
    let (blocks, notes) = chain(&net, &path, &flow);

    let params = Params {
        // No sound data in the model; the FCU shape stands in for the unit.
        source_type: SourceType::FcuCeiling,
        source_offset_db: 0.0,
        volume_flow_m3s: flow[tree.root],
        room_volume_m3: 50.0,
        rt60_s: 0.6,
        receiver_distance_m: 2.0,
        blocks,
    };
    println!(
        "{} [{}] {system} -> {} [{}]: {} path elements, {} blocks, {:.0} L/s at the unit",
        net.nodes[tree.root].family,
        net.nodes[tree.root].id,
        net.nodes[terminal].family,
        net.nodes[terminal].id,
        path.len(),
        params.blocks.len(),
        flow[tree.root] * 1000.0
    );

    let sys = DuctSystem::from_params(&params);
    let lw = |pwl: &[wall_duct_core::autodiff::Dual<{ wall_duct_core::model::N_PARAMS }>; N_BANDS]| {
        pwl.iter().map(|b| format!("{:5.1}", b.value)).collect::<Vec<_>>().join(" ")
    };
    let hz = OCTAVE_BANDS_HZ.iter().map(|f| format!("{:>5}", if *f >= 1000.0 { format!("{}k", f / 1000.0) } else { f.to_string() })).collect::<Vec<_>>().join(" ");
    println!("\n{:>3} {:<44} {:>6}   Lw dB re 1 pW: {hz}", "#", "block", "m/s");
    let mut q = sys.volume_flow();
    let mut pwl = source_pwl(sys.source_type, sys.source_offset);
    println!("{:>3} {:<44} {:>6}   {}", "", "source (FCU shape)", "", lw(&pwl));
    for (k, (block, bp)) in sys.blocks.iter().zip(&params.blocks).enumerate() {
        q = block.flow_out(q);
        pwl = block.forward(&pwl, q);
        let v = match *bp {
            BlockParams::Segment { width_m, height_m, .. }
            | BlockParams::Bend { width_m, height_m }
            | BlockParams::Diffuser { width_m, height_m } => format!("{:.1}", q.value / (width_m * height_m)),
            _ => String::new(),
        };
        let label = match *bp {
            BlockParams::Segment { length_m, width_m, height_m, .. } => {
                format!("Segment {length_m:.2} m {:.0}x{:.0}", width_m * 1e3, height_m * 1e3)
            }
            BlockParams::Bend { width_m, height_m } => format!("Bend {:.0}x{:.0}", width_m * 1e3, height_m * 1e3),
            BlockParams::Diffuser { width_m, height_m } => format!("Diffuser {:.0}x{:.0}", width_m * 1e3, height_m * 1e3),
            BlockParams::Branch { flow_ratio } => format!("Branch x{flow_ratio:.3} ({})", notes[k]),
            BlockParams::Attenuator { .. } => "Attenuator".into(),
        };
        println!("{k:>3} {label:<44} {v:>6}   {}", lw(&pwl));
    }

    let spl = sys.receiver_spl();
    println!("\nreceiver SPL  {}", spl.iter().map(|b| format!("{:5.1}", b.value)).collect::<Vec<_>>().join(" "));
    println!("NC {}", sys.nc());

    // Which parameters move the loudest band most.
    let worst = (0..N_BANDS).max_by(|&a, &b| spl[a].value.partial_cmp(&spl[b].value).unwrap()).unwrap();
    let mut grads: Vec<(usize, f64)> = spl[worst].grad.iter().copied().enumerate().filter(|(_, g)| g.abs() > 1e-9).collect();
    grads.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    println!("largest sensitivities of the {} Hz band (dB per unit of the log-space parameter):", OCTAVE_BANDS_HZ[worst]);
    for (slot, g) in grads.iter().take(8) {
        println!("  {:<28} {g:+.2}", sys.slot_label(*slot));
    }
}
