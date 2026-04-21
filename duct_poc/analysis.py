"""Turn a DuctSystem into a structured design-analysis facts object.

Same pattern as the insul_poc analysis module: physics and rating
decisions stay in Python; the LLM only handles phrasing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import torch

from duct_poc.model import DuctSystem
from duct_poc.metrics.octave import OCTAVE_BANDS_HZ
from duct_poc.metrics.nc import nc_exact, nc_band_exceedances, _curve_at


# Human-friendly labels for internal parameter names. Keys are the raw
# attribute names used on the nn.Module (log_*/…_raw). The display layer
# uses this mapping; raw names remain as `param` for machine use.
PARAM_LABELS: dict[str, str] = {
    # System-level parameters
    "source_offset": "source level offset",
    "log_volume_flow": "volume flow rate",
    "log_room_volume": "room volume",
    "log_rt60": "room RT60",
    "log_receiver_distance": "receiver distance",
    # Per-block parameters (same key on every block class)
    "log_length": "length",
    "length_raw": "length",
    "log_width": "width",
    "log_height": "height",
    "lining_raw": "lining thickness",
    "free_area_raw": "free-area ratio",
    "splitter_raw": "splitter thickness",
}


def pretty_param(name: str) -> str:
    """Convert an internal parameter name (e.g. 'log_volume_flow') to a
    human-friendly label (e.g. 'volume flow rate'). Falls back to the raw
    name if unmapped so new parameters remain visible rather than silent."""
    return PARAM_LABELS.get(name, name)


# System-level parameters that are *external constraints*, not design
# levers: the user doesn't get to change the room they're in or the
# supply rate they have to move. Excluded from the top-sensitivities
# ranking but kept in the raw JSON so they're still visible if needed.
CONSTRAINT_PARAMS: set[str] = {
    "log_volume_flow",
    "log_room_volume",
    "log_rt60",
    "log_receiver_distance",
    "source_offset",
}


@dataclass
class BandFact:
    freq_hz: float
    spl_db: float
    nc_curve_db: float
    exceedance_db: float


@dataclass
class SensitivityFact:
    block: str        # e.g. "block 3" or "system"
    block_type: str   # e.g. "Attenuator", "DuctSegment", "-" for system
    param: str        # e.g. "lining_raw"
    value: float      # dNC / dparam (linear gradient)


@dataclass
class BlockSummary:
    index: int
    type_name: str
    label: str        # human-friendly e.g. "Duct 3.0m 400x300 lined 25mm"


@dataclass
class DesignFacts:
    nc: int
    receiver_spl_bands: list[BandFact]
    controlling_bands: list[BandFact]
    top_sensitivities: list[SensitivityFact]
    blocks: list[BlockSummary]
    design_params: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "nc": self.nc,
            "receiver_spl_bands": [asdict(b) for b in self.receiver_spl_bands],
            "controlling_bands": [asdict(b) for b in self.controlling_bands],
            "top_sensitivities": [asdict(s) for s in self.top_sensitivities],
            "blocks": [asdict(b) for b in self.blocks],
            "design_params": self.design_params,
        }


def _describe_block(idx: int, blk, q_m3s: float) -> BlockSummary:
    t = type(blk).__name__
    if t == "DuctSegment":
        v = q_m3s / float(blk.area)
        label = (f"Duct {float(blk.length):.1f} m, "
                 f"{int(float(blk.width)*1000)}×{int(float(blk.height)*1000)} mm")
        if float(blk.lining_thickness) > 1e-4:
            label += f", lined {int(float(blk.lining_thickness)*1000)} mm"
        else:
            label += ", unlined"
        label += f" · v ≈ {v:.1f} m/s"
    elif t == "Bend":
        v = q_m3s / float(blk.area)
        label = (f"Bend {int(float(blk.width)*1000)}×{int(float(blk.height)*1000)} mm · "
                 f"v ≈ {v:.1f} m/s")
    elif t == "Attenuator":
        length_mm = int(float(blk.length) * 1000)
        v_face = q_m3s / float(blk.gross_area)
        v_free = q_m3s / float(blk.free_area)
        label = (f"Attenuator {length_mm} mm, free area "
                 f"{float(blk.free_area_ratio):.0%}, "
                 f"splitter {int(float(blk.splitter_thickness)*1000)} mm · "
                 f"v_face {v_face:.1f} / v_free {v_free:.1f} m/s")
    elif t == "Diffuser":
        v = q_m3s / float(blk.area)
        label = (f"Diffuser {int(float(blk.width)*1000)}×{int(float(blk.height)*1000)} mm · "
                 f"v ≈ {v:.1f} m/s")
    else:
        label = t
    return BlockSummary(index=idx, type_name=t, label=label)


def analyse(system: DuctSystem, top_k: int = 6) -> DesignFacts:
    # Forward + backward for sensitivities
    system.zero_grad(set_to_none=True)
    spl = system.receiver_spl()
    nc_val = int(nc_exact(spl))
    nc_for_grad = system.nc()
    nc_for_grad.backward()

    # Band facts
    curve = _curve_at(nc_val)
    exceedance = nc_band_exceedances(spl.detach(), nc_val)
    bands: list[BandFact] = []
    for i, f in enumerate(OCTAVE_BANDS_HZ.tolist()):
        bands.append(BandFact(
            freq_hz=float(f),
            spl_db=round(float(spl[i]), 1),
            nc_curve_db=round(float(curve[i]), 1),
            exceedance_db=round(float(exceedance[i]), 2),
        ))

    # Controlling bands — those with exceedance > 0.05 dB (the bands
    # pressing against the rated curve). We keep the threshold low because
    # NC is a worst-band metric: any band within ~1 dB of the curve is
    # genuinely pinning the rating.
    controlling = sorted(
        [b for b in bands if b.exceedance_db > 0.05],
        key=lambda b: -b.exceedance_db,
    )[:5]

    # Sensitivities — flatten named_parameters into (block, block_type,
    # param, grad). For system-level parameters the block_type is "-".
    # External constraints (flow rate, room properties) are excluded from
    # the top-sensitivities ranking: they're knobs the user generally
    # can't turn, and showing them drowns out the actionable levers.
    sens: list[SensitivityFact] = []
    blocks_list = list(system.blocks)
    for name, p in system.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        if parts[0] == "blocks":
            idx = int(parts[1])
            block_label = f"block {idx}"
            block_type = type(blocks_list[idx]).__name__
            param = ".".join(parts[2:])
        else:
            if name in CONSTRAINT_PARAMS:
                continue
            block_label = "system"
            block_type = "-"
            param = name
        sens.append(SensitivityFact(
            block=block_label, block_type=block_type, param=param,
            value=round(float(p.grad.detach()), 3),
        ))
    top_sens = sorted(sens, key=lambda s: -abs(s.value))[:top_k]

    # Block summaries (use the current volume flow for derived velocity)
    q = float(system.volume_flow.detach())
    block_summaries = [
        _describe_block(i, blk, q) for i, blk in enumerate(system.blocks)
    ]

    # High-level design params (the ones we'll surface to the LLM)
    design_params = {
        "source_type": system.source_type,
        "source_offset_db": round(float(system.source_offset.detach()), 1),
        "volume_flow_Ls": round(q * 1000.0, 1),
        "room_volume_m3": round(float(system.room_volume.detach()), 1),
        "rt60_s": round(float(system.rt60.detach()), 2),
        "receiver_distance_m": round(float(system.receiver_distance.detach()), 2),
    }

    return DesignFacts(
        nc=nc_val,
        receiver_spl_bands=bands,
        controlling_bands=controlling,
        top_sensitivities=top_sens,
        blocks=block_summaries,
        design_params=design_params,
    )
