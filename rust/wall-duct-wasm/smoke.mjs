// Node smoke test: evaluate the demo case through the compiled WASM and
// assert NC=23. Kept as a .mjs file so Node picks up ES-module conventions
// for the generated wrapper.
import { evaluate, version } from "./pkg-node/wall_duct_wasm.js";

const demo = {
  source_type: "FCU (ceiling)",
  source_offset_db: 0.0,
  volume_flow_m3s: 0.25,
  room_volume_m3: 50.0,
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

if (out.nc !== 23) {
  console.error(`FAIL: expected NC=23, got ${out.nc}`);
  process.exit(1);
}

// ARI 885's lined fit (CIBSE Guide B) changes the lined segment's loss; an
// unknown fit is refused.
const ari = evaluate({ ...demo, lined_fit: "ari885" });
console.log("NC with ARI 885 lining:", ari.nc, ari.receiver_spl_bands.map(b => b.spl_db));
if (ari.receiver_spl_bands.every((b, i) => b.spl_db === out.receiver_spl_bands[i].spl_db)) {
  console.error("FAIL: lined_fit ari885 changed nothing");
  process.exit(1);
}
try {
  evaluate({ ...demo, lined_fit: "nope" });
  console.error("FAIL: unknown lined_fit accepted");
  process.exit(1);
} catch (e) {}
console.log("OK");
