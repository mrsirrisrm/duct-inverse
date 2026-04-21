"""Packed-splitter attenuator insertion loss (IL).

Illustrative fit in the spirit of catalogue data for a standard splitter
attenuator with mineral-wool splitters and perforated-sheet facings.

Shape factors (for a differentiable, demo-quality model):

    IL(f) = length_m * per_m_peak(f) * blockage^1.5 * lining_factor(t)

where
    per_m_peak(f) = a band-dependent IL/m for a nominal 50% free-area,
                   100 mm splitter packing, peaking around 500 Hz
    blockage      = 1 - free_area_ratio  (0 = empty tube, 1 = fully packed)
    lining_factor = smooth-saturating function of splitter thickness

Length of 0 -> IL of 0 with zero gradient distortion, so the optimiser can
freely push a "slot" attenuator to zero length if it is not needed.
"""
from __future__ import annotations

import torch

# Nominal per-metre IL for a 50%-free, 100mm-splitter attenuator (dB/m).
# Peaks in the 500 Hz - 2 kHz region where splitter attenuators are most
# effective. Values are typical of 1.2 m long units in catalogues (~20-30
# dB at peak, which works out to ~20 dB/m for the assumed reference).
PER_M_PEAK = torch.tensor(
    # 63   125   250   500   1k    2k    4k    8k
    [2.0, 5.0, 12.0, 22.0, 28.0, 24.0, 14.0, 6.0],
    dtype=torch.float64,
)

# Reference lining thickness the PER_M_PEAK values are calibrated for.
REFERENCE_T_M = 0.1  # 100 mm


def lining_factor(thickness_m: torch.Tensor) -> torch.Tensor:
    """Smoothly-saturating function: 0 at t=0, ~1 at t=100mm, ~1.2 at 200mm."""
    return 1.0 - torch.exp(-thickness_m / 0.05)


def attenuator_il(
    length_m: torch.Tensor,
    free_area_ratio: torch.Tensor,
    splitter_thickness_m: torch.Tensor,
) -> torch.Tensor:
    """Insertion loss of a splitter attenuator, per octave band (8,).

    Returns a (8,) tensor. Differentiable in all three parameters. Length
    of 0 -> IL of 0 cleanly."""
    blockage = torch.clamp(1.0 - free_area_ratio, min=1e-3, max=0.95)
    t_factor = lining_factor(splitter_thickness_m) / lining_factor(torch.tensor(REFERENCE_T_M, dtype=splitter_thickness_m.dtype))
    return length_m * PER_M_PEAK * blockage.pow(1.5) * t_factor
