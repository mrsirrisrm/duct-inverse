//! WebAssembly surface for wall-duct-core.
//!
//! Single entry point `evaluate(params_js)` takes a JSON description of the
//! duct system (source + block chain + room) and returns a rich
//! `DesignFacts` object the UI can render directly: receiver SPL per band,
//! NC rating, controlling bands, top design-lever sensitivities, and
//! human-readable block labels.
//!
//! JSON boundary deliberately flat so the TS side can stay lightly typed.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use wall_duct_core::metrics::nc;
use wall_duct_core::metrics::octave::{N_BANDS, OCTAVE_BANDS_HZ};
use wall_duct_core::model::{
    self, block_slot_base, BlockParams, DuctSystem, Params, MAX_BLOCK_SLOTS, N_PARAMS,
    SLOT_LOG_RECEIVER_DISTANCE, SLOT_LOG_ROOM_VOLUME, SLOT_LOG_VOLUME_FLOW,
    SLOT_SOURCE_OFFSET, SYSTEM_SLOTS,
};
use wall_duct_core::physics::elbow::ElbowKind;
use wall_duct_core::physics::source::SourceType;

// ---- input shape ----

#[derive(Deserialize)]
pub struct InputParams {
    pub source_type: String,
    pub source_offset_db: f64,
    pub volume_flow_m3s: f64,
    pub room_volume_m3: f64,
    pub receiver_distance_m: f64,
    pub blocks: Vec<InputBlock>,
}

/// Tagged union for block input. `kind` discriminates the variant; irrelevant
/// fields are ignored. Optional fields default via serde when absent so the
/// TS side can send lean objects per variant.
#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum InputBlock {
    #[serde(rename = "segment")]
    Segment {
        length_m: f64,
        width_m: f64,
        height_m: f64,
        lining_m: f64,
    },
    #[serde(rename = "bend")]
    Bend { width_m: f64, height_m: f64 },
    #[serde(rename = "attenuator")]
    Attenuator {
        length_m: f64,
        width_m: f64,
        height_m: f64,
        free_area_ratio: f64,
        splitter_m: f64,
    },
    #[serde(rename = "diffuser")]
    Diffuser { width_m: f64, height_m: f64 },
}

fn parse_source_type(s: &str) -> Result<SourceType, String> {
    match s {
        "FCU (ceiling)" | "fcu" => Ok(SourceType::FcuCeiling),
        "VAV box" | "vav" => Ok(SourceType::VavBox),
        "Axial fan" | "axial_fan" => Ok(SourceType::AxialFan),
        other => Err(format!("unknown source_type: {other}")),
    }
}

impl InputBlock {
    fn into_params(self) -> BlockParams {
        match self {
            Self::Segment { length_m, width_m, height_m, lining_m } => {
                BlockParams::Segment { length_m, width_m, height_m, lining_m, round: false }
            }
            Self::Bend { width_m, height_m } => BlockParams::Bend { width_m, height_m, kind: ElbowKind::Mitred, lined: false, round: false },
            Self::Attenuator {
                length_m, width_m, height_m, free_area_ratio, splitter_m,
            } => BlockParams::Attenuator {
                length_m, width_m, height_m, free_area_ratio, splitter_m,
            },
            Self::Diffuser { width_m, height_m } => BlockParams::Diffuser { width_m, height_m, round: false },
        }
    }
}

// ---- output shape ----

#[derive(Serialize)]
pub struct BandFact {
    pub freq_hz: f64,
    pub spl_db: f64,
    pub nc_curve_db: f64,
    pub exceedance_db: f64,
}

#[derive(Serialize)]
pub struct Sensitivity {
    pub block: String,      // "system" or "block 2"
    pub block_type: String, // "-" for system, otherwise e.g. "Attenuator"
    pub param: String,      // e.g. "lining_raw", "log_volume_flow"
    pub value: f64,         // d(controlling-band SPL sum) / d(slot)
}

#[derive(Serialize)]
pub struct BlockSummary {
    pub index: usize,
    pub type_name: String,
    pub label: String,
}

#[derive(Serialize)]
pub struct DesignParams {
    pub source_type: String,
    pub source_offset_db: f64,
    pub volume_flow_ls: f64,
    pub room_volume_m3: f64,
    pub receiver_distance_m: f64,
}

#[derive(Serialize)]
pub struct DesignFacts {
    pub nc: i32,
    pub receiver_spl_bands: Vec<BandFact>,
    pub controlling_bands: Vec<BandFact>,
    pub top_sensitivities: Vec<Sensitivity>,
    pub all_sensitivities: Vec<Sensitivity>,
    pub blocks: Vec<BlockSummary>,
    pub design_params: DesignParams,
}

// ---- helpers ----

/// Slots that describe external constraints, not design levers. Excluded
/// from the top-sensitivities ranking but still included in all_sensitivities
/// for LLM context. Matches the Python `CONSTRAINT_PARAMS` set.
fn is_constraint_slot(slot: usize) -> bool {
    matches!(
        slot,
        SLOT_SOURCE_OFFSET
            | SLOT_LOG_VOLUME_FLOW
            | SLOT_LOG_ROOM_VOLUME
            | SLOT_LOG_RECEIVER_DISTANCE
    )
}

/// Describe a block in human-friendly form — mirrors `_describe_block` in
/// the Python analysis module.
fn describe_block(idx: usize, bp: &BlockParams, q_m3s: f64) -> BlockSummary {
    let mm = |x: f64| (x * 1000.0) as i64;
    let label = match *bp {
        BlockParams::Segment { length_m, width_m, height_m, lining_m, .. } => {
            let v = q_m3s / (width_m * height_m);
            let lining = if lining_m > 1e-4 {
                format!(", lined {} mm", mm(lining_m))
            } else {
                ", unlined".to_string()
            };
            format!(
                "Duct {length_m:.1} m, {}×{} mm{lining} · v ≈ {v:.1} m/s",
                mm(width_m), mm(height_m),
            )
        }
        BlockParams::Bend { width_m, height_m, kind, .. } => {
            let v = q_m3s / (width_m * height_m);
            format!(
                "Bend {}×{} mm, {kind:?} · v ≈ {v:.1} m/s",
                mm(width_m), mm(height_m),
            )
        }
        BlockParams::Damper { width_m, height_m, closed_deg, .. } => {
            let v = q_m3s / (width_m * height_m);
            format!(
                "Damper {}×{} mm, {closed_deg:.0}° closed · v ≈ {v:.1} m/s",
                mm(width_m), mm(height_m),
            )
        }
        BlockParams::Attenuator {
            length_m, width_m, height_m, free_area_ratio, splitter_m,
        } => {
            let gross = width_m * height_m;
            let free = gross * free_area_ratio;
            let v_face = q_m3s / gross;
            let v_free = q_m3s / free;
            format!(
                "Attenuator {} mm, free area {:.0}%, splitter {} mm · v_face {:.1} / v_free {:.1} m/s",
                mm(length_m), free_area_ratio * 100.0, mm(splitter_m), v_face, v_free,
            )
        }
        BlockParams::Diffuser { width_m, height_m, .. } => {
            let v = q_m3s / (width_m * height_m);
            format!(
                "Diffuser {}×{} mm · v ≈ {v:.1} m/s",
                mm(width_m), mm(height_m),
            )
        }
        BlockParams::Branch { flow_ratio, .. } => {
            format!("Branch, {:.0}% of flow continues", flow_ratio * 100.0)
        }
    };
    BlockSummary {
        index: idx,
        type_name: bp.type_name().to_string(),
        label,
    }
}

/// For each slot, a pair of (block label, param name) used in the output.
/// System slots get block="system", block_type="-".
fn slot_descriptor(sys: &DuctSystem, slot: usize) -> (String, String, String) {
    if slot < SYSTEM_SLOTS {
        let label = sys.slot_label(slot);
        ("system".into(), "-".into(), label)
    } else {
        let off = slot - SYSTEM_SLOTS;
        let block_idx = off / MAX_BLOCK_SLOTS;
        let within = off % MAX_BLOCK_SLOTS;
        if let Some(bp) = sys.block_params.get(block_idx) {
            let names = bp.slot_names();
            if within < names.len() {
                return (
                    format!("block {block_idx}"),
                    bp.type_name().to_string(),
                    names[within].to_string(),
                );
            }
        }
        (format!("block {block_idx}"), "-".into(), format!("slot{within}"))
    }
}

// ---- main entry ----

/// Core evaluation against a fully-parsed `Params` object. Pulled out of the
/// wasm entry so native tests can exercise the output shape without needing
/// a JS runtime.
pub fn evaluate_params(input_source_type: String, params: Params) -> DesignFacts {
    let sys = DuctSystem::from_params(&params);
    let spl = sys.receiver_spl();
    let spl_values = sys.receiver_spl_values();
    let nc_val = nc::nc_exact(&spl_values);
    let nc_curve = nc::curve_at(nc_val as f64);
    let exceedance = nc::band_exceedances(&spl_values, nc_val);

    // Band facts: per-band SPL, fitted NC curve, exceedance vs one curve below.
    let mut receiver_spl_bands = Vec::with_capacity(N_BANDS);
    for i in 0..N_BANDS {
        receiver_spl_bands.push(BandFact {
            freq_hz: OCTAVE_BANDS_HZ[i],
            spl_db: (spl_values[i] * 10.0).round() / 10.0,
            nc_curve_db: (nc_curve[i] * 10.0).round() / 10.0,
            exceedance_db: (exceedance[i] * 100.0).round() / 100.0,
        });
    }

    // Controlling bands: exceedance > 0.05 dB, sorted desc (matches Python).
    let mut controlling: Vec<BandFact> = receiver_spl_bands
        .iter()
        .filter(|b| b.exceedance_db > 0.05)
        .map(|b| BandFact {
            freq_hz: b.freq_hz,
            spl_db: b.spl_db,
            nc_curve_db: b.nc_curve_db,
            exceedance_db: b.exceedance_db,
        })
        .collect();
    controlling.sort_by(|a, b| b.exceedance_db.partial_cmp(&a.exceedance_db).unwrap());
    controlling.truncate(5);

    // Sensitivity per slot: sum of ∂SPL/∂slot across the *controlling* bands
    // (i.e. the ones pinning the NC rating). If no band exceeds the curve,
    // fall back to uniform weighting so the tool still reports sensitivities
    // at a quiet design. This is a proxy for dNC/dslot without implementing
    // a straight-through NC estimator in Rust — same actionable signal.
    let weights: [f64; N_BANDS] = {
        let mut w = [0.0f64; N_BANDS];
        let mut any = false;
        for i in 0..N_BANDS {
            if exceedance[i] > 0.05 {
                w[i] = exceedance[i];
                any = true;
            }
        }
        if !any {
            for i in 0..N_BANDS {
                w[i] = 1.0;
            }
        }
        w
    };
    let wsum: f64 = weights.iter().sum();
    let norm_w: [f64; N_BANDS] = core::array::from_fn(|i| weights[i] / wsum);

    let mut slot_grads = [0.0f64; N_PARAMS];
    for slot in 0..N_PARAMS {
        let mut g = 0.0;
        for band in 0..N_BANDS {
            g += norm_w[band] * spl[band].grad[slot];
        }
        slot_grads[slot] = g;
    }

    // Build the flat sensitivity list. Skip slots that are genuinely
    // unoccupied (past the last block, or gap within a variant) by checking
    // that either the slot is a live system slot, or within the block's
    // declared slot_names range.
    let occupied_slot = |slot: usize| -> bool {
        if slot < SYSTEM_SLOTS {
            return is_constraint_slot(slot);
        }
        let off = slot - SYSTEM_SLOTS;
        let block_idx = off / MAX_BLOCK_SLOTS;
        let within = off % MAX_BLOCK_SLOTS;
        sys.block_params
            .get(block_idx)
            .map(|bp| within < bp.slot_names().len())
            .unwrap_or(false)
    };

    let mut all_sens: Vec<Sensitivity> = (0..N_PARAMS)
        .filter(|&slot| occupied_slot(slot))
        .map(|slot| {
            let (block, block_type, param) = slot_descriptor(&sys, slot);
            Sensitivity {
                block, block_type, param,
                value: (slot_grads[slot] * 1000.0).round() / 1000.0,
            }
        })
        .collect();

    let mut ranked: Vec<Sensitivity> = all_sens
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            // Re-derive the slot index to test constraint-ness. Ranking
            // excludes external constraints (flow, room, distance,
            // source) — matches the Python CONSTRAINT_PARAMS filter.
            let occupied: Vec<usize> = (0..N_PARAMS).filter(|&s| occupied_slot(s)).collect();
            let slot = occupied[i];
            if is_constraint_slot(slot) {
                None
            } else {
                Some(Sensitivity {
                    block: s.block.clone(),
                    block_type: s.block_type.clone(),
                    param: s.param.clone(),
                    value: s.value,
                })
            }
        })
        .collect();
    ranked.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());
    ranked.truncate(6);
    // Sort all_sens by slot order (already in that order — no-op kept for clarity).
    // (We filtered by slot above, so `all_sens` is already in slot order.)
    let _ = &mut all_sens;

    // Block summaries for the UI.
    let blocks_out: Vec<BlockSummary> = sys
        .block_params
        .iter()
        .enumerate()
        .map(|(i, bp)| describe_block(i, bp, params.volume_flow_m3s))
        .collect();

    let design_params = DesignParams {
        source_type: input_source_type,
        source_offset_db: (params.source_offset_db * 10.0).round() / 10.0,
        volume_flow_ls: (params.volume_flow_m3s * 10000.0).round() / 10.0,
        room_volume_m3: (params.room_volume_m3 * 10.0).round() / 10.0,
        receiver_distance_m: (params.receiver_distance_m * 100.0).round() / 100.0,
    };

    // Silence unused-import warnings from the pull-ins above when compiling
    // the rlib half of cdylib + rlib.
    let _ = (block_slot_base, model::MAX_BLOCKS);

    DesignFacts {
        nc: nc_val,
        receiver_spl_bands,
        controlling_bands: controlling,
        top_sensitivities: ranked,
        all_sensitivities: all_sens,
        blocks: blocks_out,
        design_params,
    }
}

/// Evaluate a duct system and return the full design-facts JSON.
#[wasm_bindgen]
pub fn evaluate(params_js: JsValue) -> Result<JsValue, JsValue> {
    let input: InputParams = serde_wasm_bindgen::from_value(params_js)
        .map_err(|e| JsValue::from_str(&format!("bad input: {e}")))?;
    let source_type = parse_source_type(&input.source_type)
        .map_err(|e| JsValue::from_str(&e))?;
    let source_type_label = input.source_type;
    let params = Params {
        source_type,
        source_offset_db: input.source_offset_db,
        volume_flow_m3s: input.volume_flow_m3s,
        room_volume_m3: input.room_volume_m3,
        receiver_distance_m: input.receiver_distance_m,
        blocks: input.blocks.into_iter().map(InputBlock::into_params).collect(),
    };
    let out = evaluate_params(source_type_label, params);
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&format!("{e}")))
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wall_duct_core::model::Params as CoreParams;

    /// End-to-end: the demo case must rate NC=23 and hand back 8 band facts,
    /// some controlling bands, at least one design-lever sensitivity, and
    /// three block summaries.
    #[test]
    fn demo_case_produces_full_facts() {
        let params = CoreParams::demo();
        let out = evaluate_params("FCU (ceiling)".to_string(), params);

        assert_eq!(out.nc, 23);
        assert_eq!(out.receiver_spl_bands.len(), 8);
        assert_eq!(out.blocks.len(), 3);
        assert_eq!(out.blocks[0].type_name, "Segment");
        assert_eq!(out.blocks[1].type_name, "Attenuator");
        assert_eq!(out.blocks[2].type_name, "Diffuser");

        // Controlling bands should be non-empty at NC 24 for this demo.
        assert!(!out.controlling_bands.is_empty());

        // Top sensitivities exclude constraint params.
        assert!(!out.top_sensitivities.is_empty());
        for s in &out.top_sensitivities {
            assert!(s.param != "source_offset");
            assert!(s.param != "log_volume_flow");
            assert!(s.param != "log_room_volume");
            assert!(s.param != "log_receiver_distance");
        }

        // all_sensitivities should include the constraint params though.
        let has_flow = out
            .all_sensitivities
            .iter()
            .any(|s| s.param == "log_volume_flow");
        assert!(has_flow, "all_sensitivities missing log_volume_flow");

        // Block labels mention the physical parameters.
        assert!(out.blocks[0].label.contains("3.0 m"));
        assert!(out.blocks[1].label.contains("Attenuator"));
        assert!(out.blocks[1].label.contains("50%"));
    }
}
