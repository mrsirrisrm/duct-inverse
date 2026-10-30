// Runs the analysis off the main thread: it takes 10 to 20 s on a 40 MB model.
import init, { analyse } from './pkg/duct_pipeline.js';

const ready = init();

self.onmessage = async ({ data: { rvt, ifc, lw } }) => {
  try {
    const wasm = await ready;
    const start = performance.now();
    const report = JSON.parse(analyse(rvt && new Uint8Array(rvt), ifc && new Uint8Array(ifc), lw && new Float64Array(lw)));
    self.postMessage({
      report,
      ms: Math.round(performance.now() - start),
      wasmMemoryMB: Math.round(wasm.memory.buffer.byteLength / 1e6),
    });
  } catch (e) {
    self.postMessage({ error: String(e) });
  }
};
