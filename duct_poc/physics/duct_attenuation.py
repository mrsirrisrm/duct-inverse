"""Natural attenuation of sound propagating along a duct (per metre).

Two regimes:

1. UNLINED rectangular ducts — small losses (~0.03..0.3 dB/m), dominated
   by breakout radiation. We use the ASHRAE HVAC Applications (2019) Table
   8, implemented as a closed-form fit so it's smooth in duct dimensions.

2. LINED rectangular ducts — much larger losses (~0.3..30 dB/m), the
   primary way engineers control mid/high-frequency noise. We use the
   Reynolds-Bakke (1990) empirical form:

       A/L [dB/m] = B(f) * (P/A)^0.73 * t^C(f)

   where P is lining perimeter, A is free-area cross-section, t is lining
   thickness in inches (the original fit), B and C are band-dependent
   coefficients.

The two regimes are blended by the lining thickness parameter — at t=0
the lined form -> 0 and the unlined form takes over, at t > 25mm the
lined form dominates. Smooth transition via a sigmoid gate on thickness.

All inputs/outputs are metric (SI). Lining thickness is in metres.
"""
from __future__ import annotations

import torch

from duct_poc.metrics.octave import OCTAVE_BANDS_HZ

# --- Reynolds-Bakke lined-duct coefficients, per octave band (63..8k Hz) ---
# The published fit is in inches for t; we convert inputs from metres to
# inches internally to reuse the tabulated constants.
# Source: ASHRAE HVAC Applications ch. 48 (1995), after Reynolds & Bakke.
# These are illustrative mid-values — production code would use the full
# tabulated B,C per band range; this captures the right shape.
RB_B = torch.tensor(
    # 63    125    250   500   1k    2k    4k    8k  (dB/m per (P/A)^0.73 in^C)
    [0.027, 0.105, 0.503, 1.26, 1.37, 0.66, 0.35, 0.15],
    dtype=torch.float64,
)
RB_C = torch.tensor(
    [0.00, 0.32, 0.72, 1.00, 0.91, 0.40, 0.10, -0.05],
    dtype=torch.float64,
)

# --- Unlined-duct loss per metre, ASHRAE Table 8 (mid-sized ducts) ---
# Roughly inversely proportional to sqrt(P/A); we again use the nominal
# values and a P/A correction. These are small — most of the action is
# in lined sections.
UNLINED_PER_M = torch.tensor(
    # 63    125    250   500   1k    2k    4k    8k
    [0.30, 0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10],
    dtype=torch.float64,
)

IN_PER_M = 39.3701


def rectangular_PA(width_m: torch.Tensor, height_m: torch.Tensor) -> torch.Tensor:
    """Perimeter-to-area ratio P/A for a rectangular duct (1/m)."""
    P = 2.0 * (width_m + height_m)
    A = width_m * height_m
    return P / A


def duct_attenuation_db_per_m(
    width_m: torch.Tensor,
    height_m: torch.Tensor,
    lining_thickness_m: torch.Tensor,
) -> torch.Tensor:
    """Attenuation per metre at each of the 8 octave bands.

    Returns a (8,) tensor. Differentiable in width, height, and thickness.

    lining_thickness_m = 0 gives unlined behaviour with continuous limit.
    """
    pa = rectangular_PA(width_m, height_m)  # scalar tensor, 1/m

    # Lined contribution (Reynolds-Bakke)
    # thickness in inches (empirical fit basis)
    t_in = lining_thickness_m * IN_PER_M
    # Safe floor so pow is defined at t=0
    t_safe = torch.clamp(t_in, min=1e-6)
    # Per-band lined loss: B * (P/A)^0.73 * t^C
    lined_per_m = RB_B * (pa ** 0.73) * (t_safe.unsqueeze(-1) ** RB_C).squeeze(0)
    # When thickness is 0, lined_per_m collapses to B*(P/A)^0.73*1e-6^C which
    # is near-zero for C>0 and small for C<0. Gate it off smoothly so the
    # unlined-only limit is clean:
    gate = torch.sigmoid(50.0 * (t_in - 0.1))  # ~1 for t > 0.1 in = 2.5 mm
    lined_per_m = lined_per_m * gate

    # Unlined contribution (loose P/A correction — reference P/A is ~10/m)
    unlined_per_m = UNLINED_PER_M * (pa / 10.0).pow(0.5)

    return lined_per_m + unlined_per_m * (1.0 - gate)


def duct_attenuation_db(
    length_m: torch.Tensor,
    width_m: torch.Tensor,
    height_m: torch.Tensor,
    lining_thickness_m: torch.Tensor,
) -> torch.Tensor:
    """Total attenuation of a duct segment, per band (8,)."""
    return length_m * duct_attenuation_db_per_m(width_m, height_m, lining_thickness_m)
