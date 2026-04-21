"""Room reception: PWL at diffuser face -> SPL at receiver position.

Standard direct-plus-reverberant model:

    SPL = PWL + 10 log10( Q/(4 pi r^2) + 4/R )

where
    Q is the diffuser directivity (2 for a ceiling diffuser in a corner-
      mounted Q=2 convention; we default to Q=2)
    r is the receiver distance from the diffuser (m)
    R = S * alpha_bar / (1 - alpha_bar) is the room constant (m^2),
      related to reverberation time via the Sabine formula:
          RT60 = 0.161 * V / A  =>  A = 0.161 * V / RT60
      and R = A / (1 - A/S) ~= A for reasonably live rooms.

For a PoC we parameterise by V and RT60 and assume R ~= 0.161*V/RT60
(valid when alpha_bar is small, which is typical for offices). Smooth
in both parameters.
"""
from __future__ import annotations

import math
import torch


def room_constant_m2(volume_m3: torch.Tensor, rt60_s: torch.Tensor) -> torch.Tensor:
    """Sabine room constant A (m^2). For diffuse-field SPL this is
    effectively the room absorption."""
    return 0.161 * volume_m3 / torch.clamp(rt60_s, min=0.1)


def receiver_spl(
    pwl_bands_db: torch.Tensor,
    volume_m3: torch.Tensor,
    rt60_s: torch.Tensor,
    receiver_distance_m: torch.Tensor,
    directivity_Q: float = 2.0,
) -> torch.Tensor:
    """SPL at the receiver (8,) given the PWL emitted from the diffuser."""
    R = room_constant_m2(volume_m3, rt60_s)
    r = torch.clamp(receiver_distance_m, min=0.3)
    direct = directivity_Q / (4.0 * math.pi * r * r)
    reverberant = 4.0 / torch.clamp(R, min=0.5)
    correction_db = 10.0 * torch.log10(direct + reverberant)
    return pwl_bands_db + correction_db
