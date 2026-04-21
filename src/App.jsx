import { createSignal, createMemo, createEffect, onMount, Show, For } from "solid-js";
import uPlot from "uplot";
import init, { evaluate, version } from "../wasm/wall_duct_wasm.js";
import wasmUrl from "../wasm/wall_duct_wasm_bg.wasm?url";
import { adviceFor, getApiKey, setApiKey } from "./advisor.js";

// --- initial design ---------------------------------------------------------

const defaultSystem = {
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

const SOURCE_TYPES = ["FCU (ceiling)", "VAV box", "Axial fan"];

// Factory for a freshly-added block of each kind.
const newBlock = {
  segment: () => ({ kind: "segment", length_m: 2.0, width_m: 0.3, height_m: 0.3, lining_m: 0.025 }),
  bend: () => ({ kind: "bend", width_m: 0.3, height_m: 0.3 }),
  attenuator: () => ({ kind: "attenuator", length_m: 0.9, width_m: 0.3, height_m: 0.3, free_area_ratio: 0.5, splitter_m: 0.1 }),
  diffuser: () => ({ kind: "diffuser", width_m: 0.3, height_m: 0.3 }),
};

const typeLabels = {
  segment: "Segment",
  bend: "Bend",
  attenuator: "Attenuator",
  diffuser: "Diffuser",
};

// Human-friendly labels for the raw slot names the wasm emits. Mirrors the
// PARAM_LABELS dict in duct_poc/analysis.py.
const PARAM_LABELS = {
  source_offset: "source level offset",
  log_volume_flow: "volume flow rate",
  log_room_volume: "room volume",
  log_rt60: "room RT60",
  log_receiver_distance: "receiver distance",
  log_length: "length",
  length_raw: "length",
  log_width: "width",
  log_height: "height",
  lining_raw: "lining thickness",
  free_area_raw: "free-area ratio",
  splitter_raw: "splitter thickness",
};
const prettyParam = (name) => PARAM_LABELS[name] || name;

// --- small presentational helpers ------------------------------------------

function Slider(props) {
  const fmt = props.format || ((v) => v.toFixed(2));
  return (
    <div class="slider-row">
      <label>
        <span>{props.label}</span>
        <span>{fmt(props.value())}{props.unit ? ` ${props.unit}` : ""}</span>
      </label>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value()}
        onInput={(e) => props.set(Number(e.currentTarget.value))}
      />
    </div>
  );
}

function Metric(props) {
  return (
    <div class={`metric ${props.tone || ""}`}>
      <div class="label">{props.label}</div>
      <div class="value">{props.value}</div>
    </div>
  );
}

// Colour the NC metric against the user-selected target. Within 5 of the
// target is "warn" — close but not quite; anything above is "bad".
function ncTone(nc, target) {
  if (nc <= target) return "nc-good";
  if (nc <= target + 5) return "nc-warn";
  return "nc-bad";
}

// --- per-block parameter editor --------------------------------------------

function BlockEditor(props) {
  // props: block (accessor), index, update(partial), remove(), moveUp(), moveDown(),
  //        isFirst, isLast
  const b = props.block;
  const patch = (k) => (v) => props.update({ [k]: v });
  return (
    <div class="block-card">
      <div class="block-head">
        <span class="title">Block {props.index} — {typeLabels[b().kind]}</span>
        <button onClick={props.moveUp} disabled={props.isFirst} title="Move up">↑</button>
        <button onClick={props.moveDown} disabled={props.isLast} title="Move down">↓</button>
        <button class="remove" onClick={props.remove} title="Remove">✕</button>
      </div>
      <Show when={b().kind === "segment"}>
        <Slider label="Length" value={() => b().length_m} set={patch("length_m")}
          min="0.5" max="10" step="0.1"
          format={(v) => v.toFixed(1)} unit="m" />
        <Slider label="Width" value={() => b().width_m} set={patch("width_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
        <Slider label="Height" value={() => b().height_m} set={patch("height_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
        <Slider label="Lining thickness" value={() => b().lining_m} set={patch("lining_m")}
          min="0" max="0.1" step="0.005"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
      </Show>
      <Show when={b().kind === "bend"}>
        <Slider label="Width" value={() => b().width_m} set={patch("width_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
        <Slider label="Height" value={() => b().height_m} set={patch("height_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
      </Show>
      <Show when={b().kind === "attenuator"}>
        <Slider label="Length" value={() => b().length_m} set={patch("length_m")}
          min="0" max="3.0" step="0.1"
          format={(v) => v.toFixed(1)} unit="m" />
        <Slider label="Width" value={() => b().width_m} set={patch("width_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
        <Slider label="Height" value={() => b().height_m} set={patch("height_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
        <Slider label="Free area ratio" value={() => b().free_area_ratio} set={patch("free_area_ratio")}
          min="0.2" max="0.8" step="0.01"
          format={(v) => `${(v * 100).toFixed(0)}%`} />
        <Slider label="Splitter thickness" value={() => b().splitter_m} set={patch("splitter_m")}
          min="0.025" max="0.2" step="0.005"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
      </Show>
      <Show when={b().kind === "diffuser"}>
        <Slider label="Width" value={() => b().width_m} set={patch("width_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
        <Slider label="Height" value={() => b().height_m} set={patch("height_m")}
          min="0.15" max="1.0" step="0.01"
          format={(v) => (v * 1000).toFixed(0)} unit="mm" />
      </Show>
    </div>
  );
}

// ---------------------------------------------------------------------------

export default function App() {
  const [ready, setReady] = createSignal(false);
  const [system, setSystem] = createSignal(defaultSystem);
  const [ncTarget, setNcTarget] = createSignal(35);
  const [constraints, setConstraints] = createSignal("");
  const [apiKey, _setApiKey] = createSignal(getApiKey());
  const [advice, setAdvice] = createSignal("");
  const [advising, setAdvising] = createSignal(false);
  const [adviceError, setAdviceError] = createSignal("");

  onMount(async () => {
    await init(wasmUrl);
    console.log("wall-duct-wasm loaded:", version());
    setReady(true);
  });

  const S = () => system();
  const setS = (k) => (v) => setSystem({ ...system(), [k]: v });

  // --- block chain edits ---
  const updateBlock = (idx, partial) => {
    const cur = system();
    const blocks = cur.blocks.map((b, i) => i === idx ? { ...b, ...partial } : b);
    setSystem({ ...cur, blocks });
  };
  const removeBlock = (idx) => {
    const cur = system();
    setSystem({ ...cur, blocks: cur.blocks.filter((_, i) => i !== idx) });
  };
  const moveBlock = (idx, delta) => {
    const cur = system();
    const blocks = [...cur.blocks];
    const j = idx + delta;
    if (j < 0 || j >= blocks.length) return;
    [blocks[idx], blocks[j]] = [blocks[j], blocks[idx]];
    setSystem({ ...cur, blocks });
  };
  const addBlock = (kind) => {
    const cur = system();
    setSystem({ ...cur, blocks: [...cur.blocks, newBlock[kind]()] });
  };

  // --- facts, recomputed every edit ---
  const facts = createMemo(() => {
    if (!ready()) return null;
    try {
      return evaluate(system());
    } catch (e) {
      console.error("evaluate failed:", e);
      return null;
    }
  });

  // --- uPlot: per-band SPL + NC curve overlay ---
  let plotRef;
  let plot;
  createEffect(() => {
    const f = facts();
    if (!f || !plotRef) return;
    // Use octave-band indices 0..7 on the x axis (with band centres as tick
    // labels) — a log-freq axis with only 8 bands looks sparse.
    const xs = f.receiver_spl_bands.map((_, i) => i);
    const freqs = f.receiver_spl_bands.map((b) => b.freq_hz);
    const spl = f.receiver_spl_bands.map((b) => b.spl_db);
    const curve = f.receiver_spl_bands.map((b) => b.nc_curve_db);
    const ctrlSet = new Set(f.controlling_bands.map((b) => b.freq_hz));
    // A highlight series: only non-null at controlling bands.
    const ctrl = f.receiver_spl_bands.map((b) =>
      ctrlSet.has(b.freq_hz) ? b.spl_db : null,
    );

    if (plot) {
      plot.setData([xs, spl, curve, ctrl]);
      return;
    }
    plot = new uPlot(
      {
        width: plotRef.clientWidth,
        height: 320,
        padding: [8, 8, 8, 8],
        legend: { show: false },
        scales: {
          x: { time: false, range: [-0.3, xs.length - 0.7] },
          y: { auto: true },
        },
        axes: [
          {
            label: "octave band (Hz)",
            values: (_u, ticks) =>
              ticks.map((t) => {
                const i = Math.round(t);
                if (i < 0 || i >= freqs.length || i !== t) return "";
                const hz = freqs[i];
                return hz >= 1000 ? `${hz / 1000}k` : `${hz}`;
              }),
            splits: () => xs,
          },
          {
            label: "SPL (dB)",
            values: (_u, ticks) =>
              ticks.map((t) =>
                t == null || !Number.isFinite(t) ? "" : `${t.toFixed(0)}`,
              ),
          },
        ],
        series: [
          { label: "band" },
          {
            label: "SPL",
            stroke: "#3b6ea5",
            width: 2,
            points: { show: true, size: 7, fill: "#3b6ea5", stroke: "#3b6ea5" },
          },
          {
            label: `NC ${f.nc}`,
            stroke: "#999",
            width: 1,
            dash: [4, 3],
            points: { show: false },
          },
          {
            label: "controlling",
            stroke: "transparent",
            points: {
              show: true,
              size: 11,
              fill: "#d94545",
              stroke: "#d94545",
            },
          },
        ],
      },
      [xs, spl, curve, ctrl],
      plotRef,
    );
    const onResize = () => plot.setSize({ width: plotRef.clientWidth, height: 320 });
    window.addEventListener("resize", onResize);
  });

  // --- Gemini ---
  async function runAdvice() {
    const f = facts();
    if (!f) return;
    setAdvising(true);
    setAdviceError("");
    try {
      const out = await adviceFor(f, constraints());
      setAdvice(out);
    } catch (e) {
      setAdviceError(String(e.message || e));
    } finally {
      setAdvising(false);
    }
  }
  const updateKey = (k) => { setApiKey(k); _setApiKey(k); };

  return (
    <>
      <header>
        <h1>HVAC duct noise — differentiable design explorer</h1>
        <div class="subtitle">Pure-Rust forward model running in WASM · NC rating from source → chain → room, with LLM commentary</div>
        <div class="demo-banner">DEMO ONLY, FOR INDICATIVE PURPOSES — NUMBERS ARE NOT ACCURATE</div>
      </header>
      <main>
        <aside class="sidebar">
          <h2>Source & flow</h2>
          <div class="slider-row">
            <label><span>Source type</span></label>
            <select value={S().source_type}
              onChange={(e) => setS("source_type")(e.currentTarget.value)}>
              <For each={SOURCE_TYPES}>{(t) => <option value={t}>{t}</option>}</For>
            </select>
          </div>
          <Slider label="Source offset" value={() => S().source_offset_db} set={setS("source_offset_db")}
            min="-10" max="10" step="0.5"
            format={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`} unit="dB" />
          <Slider label="Volume flow" value={() => S().volume_flow_m3s} set={setS("volume_flow_m3s")}
            min="0.05" max="1.0" step="0.01"
            format={(v) => (v * 1000).toFixed(0)} unit="L/s" />

          <h2>Target</h2>
          <Slider label="NC target" value={ncTarget} set={setNcTarget}
            min="15" max="65" step="1"
            format={(v) => `NC ${v}`} />

          <h2>Room</h2>
          <Slider label="Volume" value={() => S().room_volume_m3} set={setS("room_volume_m3")}
            min="10" max="500" step="5"
            format={(v) => v.toFixed(0)} unit="m³" />
          <Slider label="RT60" value={() => S().rt60_s} set={setS("rt60_s")}
            min="0.2" max="2.0" step="0.05"
            format={(v) => v.toFixed(2)} unit="s" />
          <Slider label="Receiver distance" value={() => S().receiver_distance_m} set={setS("receiver_distance_m")}
            min="0.5" max="10" step="0.25"
            format={(v) => v.toFixed(2)} unit="m" />

          <h2>Chain</h2>
          <For each={S().blocks}>{(block, i) => (
            <BlockEditor
              block={() => S().blocks[i()]}
              index={i()}
              update={(partial) => updateBlock(i(), partial)}
              remove={() => removeBlock(i())}
              moveUp={() => moveBlock(i(), -1)}
              moveDown={() => moveBlock(i(), 1)}
              isFirst={i() === 0}
              isLast={i() === S().blocks.length - 1}
            />
          )}</For>
          <div class="add-block-row">
            <button onClick={() => addBlock("segment")}>+ Segment</button>
            <button onClick={() => addBlock("bend")}>+ Bend</button>
            <button onClick={() => addBlock("attenuator")}>+ Attenuator</button>
            <button onClick={() => addBlock("diffuser")}>+ Diffuser</button>
          </div>
        </aside>

        <section class="content">
          <Show when={facts()} fallback={<div class="card">Loading model…</div>}>
            {(f) => (
              <>
                <div class="metrics">
                  <Metric label="NC" value={f().nc} tone={ncTone(f().nc, ncTarget())} />
                  <Metric label="NC target" value={ncTarget()}
                    tone={f().nc <= ncTarget() ? "nc-good" : "nc-bad"} />
                  <Metric label="vs target"
                    value={f().nc <= ncTarget()
                      ? `${ncTarget() - f().nc} below`
                      : `+${f().nc - ncTarget()}`}
                    tone={f().nc <= ncTarget() ? "nc-good" : "nc-bad"} />
                  <Metric label="Volume flow" value={`${f().design_params.volume_flow_ls.toFixed(0)} L/s`} />
                  <Metric label="RT60" value={`${f().design_params.rt60_s.toFixed(2)} s`} />
                </div>

                <div class="card">
                  <h3>Receiver SPL per octave band vs NC {f().nc} curve</h3>
                  <div ref={plotRef} style={{ width: "100%", height: "320px" }} />
                </div>

                <div class="grid-2">
                  <div class="card">
                    <h3>Controlling bands</h3>
                    <Show when={f().controlling_bands.length > 0}
                          fallback={<div class="hint">No band exceeds the NC curve — the design sits inside the target.</div>}>
                      <table>
                        <thead>
                          <tr><th class="left">f (Hz)</th><th>SPL</th><th>NC curve</th><th>excess</th></tr>
                        </thead>
                        <tbody>
                          <For each={f().controlling_bands}>{(b) => (
                            <tr>
                              <td class="left">{b.freq_hz.toFixed(0)}</td>
                              <td>{b.spl_db.toFixed(1)}</td>
                              <td>{b.nc_curve_db.toFixed(1)}</td>
                              <td>+{b.exceedance_db.toFixed(2)}</td>
                            </tr>
                          )}</For>
                        </tbody>
                      </table>
                    </Show>
                  </div>

                  <div class="card">
                    <h3>Top design sensitivities</h3>
                    <table>
                      <thead>
                        <tr>
                          <th class="left">block</th>
                          <th class="left">type</th>
                          <th class="left">parameter</th>
                          <th>∂SPL / ∂param</th>
                        </tr>
                      </thead>
                      <tbody>
                        <For each={f().top_sensitivities}>{(s) => (
                          <tr>
                            <td class="left">{s.block}</td>
                            <td class="left">{s.block_type}</td>
                            <td class="left">{prettyParam(s.param)}</td>
                            <td>{s.value >= 0 ? "+" : ""}{s.value.toFixed(2)}</td>
                          </tr>
                        )}</For>
                      </tbody>
                    </table>
                    <div class="hint">Weighted toward the bands pinning NC. Negative ⇒ increasing this parameter lowers NC.</div>
                  </div>
                </div>

                <div class="card">
                  <h3>Block summaries</h3>
                  <table>
                    <thead>
                      <tr><th class="left">#</th><th class="left">type</th><th class="left">description</th></tr>
                    </thead>
                    <tbody>
                      <For each={f().blocks}>{(b) => (
                        <tr>
                          <td class="left">{b.index}</td>
                          <td class="left">{b.type_name}</td>
                          <td class="left">{b.label}</td>
                        </tr>
                      )}</For>
                    </tbody>
                  </table>
                </div>

                <div class="card">
                  <h3>AI design advice</h3>
                  <Show when={apiKey()}
                        fallback={(
                          <div>
                            <div class="hint">Paste a Gemini API key to enable LLM-generated design advice. Stored only in your browser's localStorage.</div>
                            <input
                              type="password"
                              placeholder="GEMINI_API_KEY"
                              style={{ width: "100%", padding: "8px", "margin-top": "8px", "font-family": "inherit" }}
                              onChange={(e) => updateKey(e.currentTarget.value.trim())}
                            />
                          </div>
                        )}>
                    <label class="constraints-label">
                      Additional constraints (optional)
                      <textarea
                        class="constraints-input"
                        placeholder="e.g. the diffuser is already as large as it can go; can't add more attenuator length"
                        value={constraints()}
                        onInput={(e) => setConstraints(e.currentTarget.value)}
                      />
                    </label>
                    <div style={{ display: "flex", gap: "8px", "align-items": "center", "margin-top": "8px" }}>
                      <button onClick={runAdvice} disabled={advising()}>
                        {advising() ? "Thinking…" : "Generate advice"}
                      </button>
                      <span class="hint" style={{ flex: 1 }}>
                        Sent to Gemini: current design + NC, controlling bands, top sensitivities, block summaries, and any constraints above.
                      </span>
                      <button
                        onClick={() => updateKey("")}
                        style={{ background: "transparent", color: "#666", border: "1px solid #ccc" }}>
                        Clear key
                      </button>
                    </div>
                    <Show when={adviceError()}>
                      <div class="error">{adviceError()}</div>
                    </Show>
                    <Show when={advice()}>
                      <div class="advice">{advice()}</div>
                    </Show>
                  </Show>
                </div>

                <details class="card">
                  <summary>Raw analysis (what the LLM sees)</summary>
                  <pre style={{ "font-size": "11px", "overflow": "auto" }}>{JSON.stringify(f(), null, 2)}</pre>
                </details>
              </>
            )}
          </Show>
        </section>
      </main>
    </>
  );
}
