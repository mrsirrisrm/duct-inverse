"""Source PWL spectra — fan / FCU / VAV box / etc.

Illustrative broadband spectra parameterised by overall PWL and a 'type'
selector. For a real tool these would come from vendor data or ASHRAE
tables indexed by size, flow, and static pressure. For the PoC we give
three canonical spectra and let the user scale the overall level.
"""
from __future__ import annotations

import torch

# Canonical shapes, normalised to 0 dB overall (sum is arbitrary -- the user
# scales with the 'overall_pwl_db' parameter to set the operating level).
# Row order corresponds to the user-visible type labels.

SOURCE_SHAPES = {
    # 63    125   250   500   1k    2k    4k    8k
    "FCU (ceiling)":   [62.0, 60.0, 58.0, 56.0, 54.0, 51.0, 48.0, 42.0],
    "VAV box":         [70.0, 67.0, 63.0, 60.0, 57.0, 54.0, 50.0, 44.0],
    "Axial fan":       [82.0, 80.0, 76.0, 72.0, 68.0, 64.0, 58.0, 50.0],
}


def source_pwl(source_type: str, overall_pwl_db: torch.Tensor) -> torch.Tensor:
    """Return the source PWL spectrum (8,).

    overall_pwl_db is the *offset* applied to the canonical shape — i.e.
    the user says 'I have a +3 dB FCU' and we add 3 dB to every band.
    This preserves the shape but lets the user scale the source magnitude.
    """
    if source_type not in SOURCE_SHAPES:
        raise ValueError(f"Unknown source type: {source_type}")
    shape = torch.tensor(SOURCE_SHAPES[source_type], dtype=overall_pwl_db.dtype)
    return shape + overall_pwl_db
