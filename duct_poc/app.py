"""Streamlit UI — configurable duct chain with NC rating + AI advice.

Mirrors the style of insul_poc/app.py: sliders on the left, chart + ratings
+ sensitivities + LLM advice on the right.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv

from duct_poc.model import (
    DuctSystem, DuctSegment, Bend, Attenuator, Diffuser,
)
from duct_poc.analysis import analyse, pretty_param
from duct_poc.metrics.octave import OCTAVE_BANDS_HZ
from duct_poc.metrics.nc import _curve_at
from duct_poc.physics.source import SOURCE_SHAPES

load_dotenv()


st.set_page_config(page_title="Duct NC — inverse design PoC", layout="wide")
st.title("HVAC duct noise — differentiable chain model")


# -------------------- session-state chain --------------------

DEFAULT_CHAIN: list[dict] = [
    {"type": "segment", "length_m": 3.0, "width_m": 0.4, "height_m": 0.3,
     "lining_mm": 0.0},
    {"type": "bend", "width_m": 0.4, "height_m": 0.3},
    {"type": "attenuator", "length_m": 0.6, "width_m": 0.4, "height_m": 0.3,
     "free_area_ratio": 0.5, "splitter_mm": 100.0},
    {"type": "segment", "length_m": 2.0, "width_m": 0.3, "height_m": 0.2,
     "lining_mm": 25.0},
    {"type": "diffuser", "width_m": 0.3, "height_m": 0.3},
]

if "chain" not in st.session_state:
    st.session_state.chain = [dict(b) for b in DEFAULT_CHAIN]
if "counter" not in st.session_state:
    # monotonic id source so newly added blocks get unique widget keys
    st.session_state.counter = len(DEFAULT_CHAIN)


def _add_block(kind: str):
    st.session_state.counter += 1
    if kind == "segment":
        st.session_state.chain.append({
            "type": "segment", "length_m": 2.0,
            "width_m": 0.3, "height_m": 0.2,
            "lining_mm": 0.0,
        })
    elif kind == "bend":
        st.session_state.chain.append({
            "type": "bend",
            "width_m": 0.3, "height_m": 0.2,
        })
    elif kind == "attenuator":
        st.session_state.chain.append({
            "type": "attenuator", "length_m": 0.6,
            "width_m": 0.3, "height_m": 0.2,
            "free_area_ratio": 0.5, "splitter_mm": 100.0,
        })
    elif kind == "diffuser":
        st.session_state.chain.append({
            "type": "diffuser",
            "width_m": 0.3, "height_m": 0.3,
        })


def _remove_block(i: int):
    st.session_state.chain.pop(i)


def _move_block(i: int, delta: int):
    j = i + delta
    if 0 <= j < len(st.session_state.chain):
        st.session_state.chain[i], st.session_state.chain[j] = (
            st.session_state.chain[j], st.session_state.chain[i]
        )


# -------------------- sidebar: source + room --------------------

st.sidebar.header("Source & flow")
source_type = st.sidebar.selectbox("Source type", list(SOURCE_SHAPES.keys()))
source_offset = st.sidebar.slider("Source level offset (dB)", -20.0, 20.0, 0.0, 0.5)
# 0.25 m^3/s ≈ 540 L/s, a typical FCU supply rate.
volume_flow_ls = st.sidebar.slider(
    "Volume flow rate (L/s)", 20.0, 2000.0, 250.0, 10.0,
    help="Total flow rate through the chain. Each block's velocity is "
         "derived from this divided by the block's cross-section area.",
)
volume_flow_m3s = volume_flow_ls / 1000.0

st.sidebar.header("Room")
room_volume = st.sidebar.slider("Room volume (m³)", 10.0, 500.0, 60.0, 5.0)
rt60 = st.sidebar.slider("RT60 (s)", 0.1, 2.0, 0.6, 0.05)
receiver_dist = st.sidebar.slider("Receiver distance from diffuser (m)", 0.5, 8.0, 2.0, 0.1)

st.sidebar.header("NC target")
nc_target = st.sidebar.slider("NC target", 15, 60, 30, 1)


# -------------------- sidebar: chain builder --------------------

st.sidebar.header("Chain")
col_add1, col_add2 = st.sidebar.columns(2)
col_add1.button("+ Segment", on_click=_add_block, args=("segment",), use_container_width=True)
col_add2.button("+ Bend", on_click=_add_block, args=("bend",), use_container_width=True)
col_add3, col_add4 = st.sidebar.columns(2)
col_add3.button("+ Attenuator", on_click=_add_block, args=("attenuator",), use_container_width=True)
col_add4.button("+ Diffuser", on_click=_add_block, args=("diffuser",), use_container_width=True)

for i, blk in enumerate(st.session_state.chain):
    t = blk["type"]
    with st.sidebar.expander(f"Block {i} — {t.capitalize()}", expanded=False):
        col_up, col_down, col_rm = st.columns(3)
        col_up.button("↑", key=f"up_{i}", on_click=_move_block, args=(i, -1))
        col_down.button("↓", key=f"dn_{i}", on_click=_move_block, args=(i, +1))
        col_rm.button("✕", key=f"rm_{i}", on_click=_remove_block, args=(i,))

        if t == "segment":
            blk["length_m"] = st.slider("Length (m)", 0.1, 10.0, blk["length_m"], 0.1, key=f"seg_L_{i}")
            blk["width_m"] = st.slider("Width (mm)", 100.0, 1200.0, blk["width_m"]*1000, 10.0, key=f"seg_W_{i}") / 1000
            blk["height_m"] = st.slider("Height (mm)", 100.0, 1200.0, blk["height_m"]*1000, 10.0, key=f"seg_H_{i}") / 1000
            blk["lining_mm"] = st.slider("Lining thickness (mm)", 0.0, 100.0, blk["lining_mm"], 5.0, key=f"seg_T_{i}")
            area = blk["width_m"] * blk["height_m"]
            v = volume_flow_m3s / max(area, 1e-6)
            st.caption(f"Derived velocity: **{v:.2f} m/s** (area {area*1e4:.0f} cm²)"
                       + ("  ⚠ over 8 m/s — regenerative noise likely" if v > 8.0 else ""))
        elif t == "bend":
            blk["width_m"] = st.slider("Width (mm)", 100.0, 1200.0, blk["width_m"]*1000, 10.0, key=f"bnd_W_{i}") / 1000
            blk["height_m"] = st.slider("Height (mm)", 100.0, 1200.0, blk["height_m"]*1000, 10.0, key=f"bnd_H_{i}") / 1000
            area = blk["width_m"] * blk["height_m"]
            v = volume_flow_m3s / max(area, 1e-6)
            st.caption(f"Derived velocity: **{v:.2f} m/s**"
                       + ("  ⚠ over 7 m/s — bend self-noise" if v > 7.0 else ""))
        elif t == "attenuator":
            blk["length_m"] = st.slider("Length (m)", 0.0, 3.0, blk["length_m"], 0.05, key=f"att_L_{i}")
            blk["width_m"] = st.slider("Width (mm)", 100.0, 1200.0, blk["width_m"]*1000, 10.0, key=f"att_W_{i}") / 1000
            blk["height_m"] = st.slider("Height (mm)", 100.0, 1200.0, blk["height_m"]*1000, 10.0, key=f"att_H_{i}") / 1000
            blk["free_area_ratio"] = st.slider("Free-area ratio", 0.1, 0.9, blk["free_area_ratio"], 0.05, key=f"att_F_{i}")
            blk["splitter_mm"] = st.slider("Splitter thickness (mm)", 0.0, 300.0, blk["splitter_mm"], 10.0, key=f"att_T_{i}")
            gross = blk["width_m"] * blk["height_m"]
            free = gross * blk["free_area_ratio"]
            v_face = volume_flow_m3s / max(gross, 1e-6)
            v_free = volume_flow_m3s / max(free, 1e-6)
            st.caption(
                f"Face velocity: **{v_face:.2f} m/s**,  "
                f"free-area velocity: **{v_free:.2f} m/s**"
                + ("  ⚠ free-area > 10 m/s — self-noise will dominate"
                   if v_free > 10.0 else "")
            )
        elif t == "diffuser":
            blk["width_m"] = st.slider("Width (mm)", 100.0, 600.0, blk["width_m"]*1000, 10.0, key=f"dif_W_{i}") / 1000
            blk["height_m"] = st.slider("Height (mm)", 100.0, 600.0, blk["height_m"]*1000, 10.0, key=f"dif_H_{i}") / 1000
            area = blk["width_m"] * blk["height_m"]
            v = volume_flow_m3s / max(area, 1e-6)
            st.caption(f"Face velocity: **{v:.2f} m/s**"
                       + ("  ⚠ over 4 m/s — diffuser self-noise" if v > 4.0 else ""))


# -------------------- build the system --------------------

def build_system() -> DuctSystem:
    blocks = []
    for blk in st.session_state.chain:
        t = blk["type"]
        if t == "segment":
            blocks.append(DuctSegment(
                blk["length_m"], blk["width_m"], blk["height_m"],
                blk["lining_mm"],
            ))
        elif t == "bend":
            blocks.append(Bend(blk["width_m"], blk["height_m"]))
        elif t == "attenuator":
            blocks.append(Attenuator(
                blk["length_m"], blk["width_m"], blk["height_m"],
                blk["free_area_ratio"], blk["splitter_mm"],
            ))
        elif t == "diffuser":
            blocks.append(Diffuser(blk["width_m"], blk["height_m"]))
    return DuctSystem(
        source_type=source_type,
        source_offset_db=source_offset,
        volume_flow_m3s=volume_flow_m3s,
        blocks=blocks,
        room_volume_m3=room_volume,
        rt60_s=rt60,
        receiver_distance_m=receiver_dist,
    )


system = build_system()
facts = analyse(system)


# -------------------- top row: metrics --------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("NC rating", facts.nc)
m2.metric("NC target", nc_target, delta=int(facts.nc - nc_target),
          delta_color="inverse")
m3.metric("Blocks", len(facts.blocks))
m4.metric("Receiver distance", f"{receiver_dist:.1f} m")


# -------------------- chart + controlling bands --------------------

col_chart, col_side = st.columns([3, 2])

with col_chart:
    st.subheader("Octave-band receiver SPL vs. NC target")
    bands_hz = OCTAVE_BANDS_HZ.numpy()
    spl_vals = [b.spl_db for b in facts.receiver_spl_bands]
    target_curve = _curve_at(nc_target).numpy()
    fitted_curve = [b.nc_curve_db for b in facts.receiver_spl_bands]

    import altair as alt
    df = pd.DataFrame({
        "freq_hz": bands_hz,
        "SPL_dB": spl_vals,
        "NC_target_dB": target_curve,
        "NC_fitted_dB": fitted_curve,
    })
    ctrl_freqs = [b.freq_hz for b in facts.controlling_bands]
    df["controlling"] = df["freq_hz"].isin(ctrl_freqs)

    base = alt.Chart(df).encode(
        x=alt.X("freq_hz:O", sort=list(bands_hz), title="frequency (Hz, octave)"),
    )
    bars = base.mark_bar().encode(
        y=alt.Y("SPL_dB:Q", title="SPL (dB)"),
        color=alt.condition(
            "datum.controlling",
            alt.value("#e45756"),
            alt.value("#4c78a8"),
        ),
        tooltip=["freq_hz", "SPL_dB"],
    )
    fitted = base.mark_line(point=True, color="black", strokeDash=[4, 2]).encode(
        y="NC_fitted_dB:Q", tooltip=["freq_hz", "NC_fitted_dB"],
    )
    target = base.mark_line(point=False, color="#888", strokeDash=[2, 2]).encode(
        y="NC_target_dB:Q", tooltip=["freq_hz", "NC_target_dB"],
    )
    st.altair_chart(bars + target + fitted, use_container_width=True)
    st.caption(
        "Blue bars: receiver SPL in each octave band. Red bars: bands "
        "exceeding the next-lower NC curve (those are pinning the rating). "
        "Dashed black line: fitted NC curve. Grey dashed: your NC target."
    )

with col_side:
    st.subheader("Controlling bands")
    if facts.controlling_bands:
        rows = [
            {"f (Hz)": int(b.freq_hz),
             "SPL (dB)": b.spl_db,
             "above curve": b.exceedance_db}
            for b in facts.controlling_bands
        ]
        st.table(pd.DataFrame(rows))
    else:
        st.info("No band exceeds the rated NC curve's next step — the design "
                "is sitting comfortably inside its NC rating.")

    st.subheader("Top sensitivities")
    sens_rows = [
        {"location": s.block, "type": s.block_type,
         "parameter": pretty_param(s.param), "dNC/dp": s.value}
        for s in facts.top_sensitivities
    ]
    st.table(pd.DataFrame(sens_rows))
    st.caption(
        "Linear gradient of NC with respect to each parameter. Negative "
        "means increasing the parameter lowers NC (the goal)."
    )


# -------------------- LLM advice --------------------

st.subheader("Design advice")

@st.cache_data(show_spinner=False)
def _cached_advice(facts_json: str, model_name: str = "gemini-3.1-flash-lite-preview") -> str:
    from duct_poc.advisor import advice_for
    import json as _json
    class _Shim:
        def __init__(self, d): self._d = d
        def to_json(self): return self._d
    return advice_for(_Shim(_json.loads(facts_json)), model_name=model_name)


if not os.environ.get("GEMINI_API_KEY"):
    st.info(
        "Set GEMINI_API_KEY in a .env file at the project root to enable "
        "AI-generated design advice."
    )
else:
    if st.button("Generate advice", type="primary"):
        with st.spinner("Consulting Gemini..."):
            try:
                import json
                advice = _cached_advice(
                    json.dumps(facts.to_json(), sort_keys=True)
                )
                st.markdown(advice)
            except Exception as e:
                st.error(f"Could not generate advice: {e}")


with st.expander("Raw analysis JSON (what the LLM sees)"):
    st.json(facts.to_json())
