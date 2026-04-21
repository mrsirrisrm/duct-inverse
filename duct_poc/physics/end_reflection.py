"""End-reflection loss at the duct termination.

Low-frequency sound radiated from a duct end experiences a reflection loss
because the outlet area is small compared to the wavelength. ASHRAE gives
a standard closed form:

    ERL(dB) ≈ 10 log10(1 + (c / (pi * f * D))^1.88)

where D is the effective diameter of the duct opening (4A/P for rectangular,
with A=area and P=perimeter). Smooth everywhere, differentiable in area.
Above ~500 Hz for typical duct sizes the loss goes to zero.
"""
from __future__ import annotations

import torch

from duct_poc.metrics.octave import OCTAVE_BANDS_HZ

C_AIR = 343.0  # m/s


def end_reflection_loss_db(effective_diameter_m: torch.Tensor) -> torch.Tensor:
    """ERL at each of the 8 octave bands. Returns (8,)."""
    f = OCTAVE_BANDS_HZ.to(effective_diameter_m.dtype)
    ratio = C_AIR / (torch.pi * f * effective_diameter_m)
    return 10.0 * torch.log10(1.0 + ratio.pow(1.88))


def effective_diameter_rectangular(
    width_m: torch.Tensor, height_m: torch.Tensor
) -> torch.Tensor:
    """Hydraulic equivalent diameter for a rectangular duct end."""
    return 4.0 * (width_m * height_m) / (2.0 * (width_m + height_m))
