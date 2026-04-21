// Node smoke test: evaluate the demo case through the compiled WASM and
// assert NC=57 with the expected SPL per band. Kept as a .mjs file so
// Node picks up ES-module conventions for the generated wrapper.
import { evaluate, version } from "./pkg-node/wall_duct_wasm.js";

const demo = {
  source_type: "FCU (ceiling)",
  source_offset_db: 0.0,
  volume_flow_m3s: 0.25,
  room_volume_m3: 50.0,
  rt60_s: 0.6,
  receiver_distance_m: 2.0,
  blocks: [
    { kind: "segment", length_m: 3.0, width_m: 0.4, height_m: 0.3, lining_m: 0.025 },
    { kind: "attenuator", length_m: 1.2, width_m: 0.4, height_m: 0.3, free_area_ratio: 0.5, splitter_m: 0.1 },
    { kind: "diffuser", width_m: 0.4, height_m: 0.3 },
  ],
};

const out = evaluate(demo);

console.log("wall-duct-wasm version:", version());
console.log("NC:", out.nc);
console.log("Bands:", out.receiver_spl_bands.map(b => b.spl_db));
console.log("Controlling:", out.controlling_bands.map(b => `${b.freq_hz}Hz +${b.exceedance_db}`));
console.log("Top sensitivities:");
for (const s of out.top_sensitivities) {
  console.log(`  ${s.block} ${s.block_type} ${s.param}: ${s.value}`);
}
console.log("Blocks:");
for (const b of out.blocks) {
  console.log(`  [${b.index}] ${b.type_name}: ${b.label}`);
}

if (out.nc !== 57) {
  console.error(`FAIL: expected NC=57, got ${out.nc}`);
  process.exit(1);
}
console.log("OK");
