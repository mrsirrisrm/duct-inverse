"""Octave-band centres used throughout the duct-noise PoC.

NC is defined on these 8 bands. All spectra in this project are 8-long
tensors in this order — source PWLs, per-block losses, receiver SPLs.
"""
from __future__ import annotations

import torch

OCTAVE_BANDS_HZ = torch.tensor(
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    dtype=torch.float64,
)

N_BANDS = 8
