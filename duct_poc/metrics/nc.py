"""NC rating (Noise Criterion, Beranek) — differentiable via straight-through.

The NC family is defined by tabulated 8-band reference curves. Each curve is
labelled by its value at 1000 Hz (approximately). The rating of a spectrum
is the lowest-label curve that the spectrum does not exceed in any band.

Same pattern as STC: exact integer in the forward pass, smooth relaxation
in the backward pass.

Reference curves from Beranek 1957; tabulated over NC-15 through NC-70 in
steps of 5. We implement at integer resolution by linearly interpolating
between the tabulated curves — physically reasonable and keeps the metric
monotone and smooth.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Tabulated NC reference curves for NC-15 through NC-70 (step 5).
# Rows: NC label. Columns: 63, 125, 250, 500, 1k, 2k, 4k, 8k Hz.
# Source: Beranek / ASHRAE Fundamentals.
NC_LABELS = list(range(15, 71, 5))
NC_TABLE = torch.tensor(
    [
        # 63   125  250  500  1k   2k   4k   8k
        [47.0, 36.0, 29.0, 22.0, 17.0, 14.0, 12.0, 11.0],  # NC-15
        [51.0, 40.0, 33.0, 26.0, 22.0, 19.0, 17.0, 16.0],  # NC-20
        [54.0, 44.0, 37.0, 31.0, 27.0, 24.0, 22.0, 21.0],  # NC-25
        [57.0, 48.0, 41.0, 35.0, 31.0, 29.0, 28.0, 27.0],  # NC-30
        [60.0, 52.0, 45.0, 40.0, 36.0, 34.0, 33.0, 32.0],  # NC-35
        [64.0, 56.0, 50.0, 45.0, 41.0, 39.0, 38.0, 37.0],  # NC-40
        [67.0, 60.0, 54.0, 49.0, 46.0, 44.0, 43.0, 42.0],  # NC-45
        [71.0, 64.0, 58.0, 54.0, 51.0, 49.0, 48.0, 47.0],  # NC-50
        [74.0, 67.0, 62.0, 58.0, 56.0, 54.0, 53.0, 52.0],  # NC-55
        [77.0, 71.0, 67.0, 63.0, 61.0, 59.0, 58.0, 57.0],  # NC-60
        [80.0, 75.0, 71.0, 68.0, 66.0, 64.0, 63.0, 62.0],  # NC-65
        [83.0, 79.0, 75.0, 72.0, 71.0, 69.0, 68.0, 67.0],  # NC-70
    ],
    dtype=torch.float64,
)


def _curve_at(label: float) -> torch.Tensor:
    """Return the NC curve at any (possibly fractional) label by linear
    interpolation between the 5-step tabulated curves. Extrapolation past
    the tabulated range saturates to the end-points; at the PoC scale we
    never need to extrapolate far."""
    label = torch.as_tensor(float(label), dtype=torch.float64)
    # Map label to a fractional index into NC_TABLE rows (0 -> NC-15, step 5).
    idx = (label - 15.0) / 5.0
    lo = int(torch.clamp(torch.floor(idx), 0, NC_TABLE.shape[0] - 1).item())
    hi = min(lo + 1, NC_TABLE.shape[0] - 1)
    t = float(torch.clamp(idx - lo, 0.0, 1.0))
    return (1.0 - t) * NC_TABLE[lo] + t * NC_TABLE[hi]


def nc_exact(spl_bands: torch.Tensor) -> int:
    """Lowest NC label whose reference curve is not exceeded at any of the
    8 octave bands. Returns an integer NC value.

    Search range: 15..70, step 1 (finer than the tabulated 5-step grid —
    the interpolated curves are smooth and monotone in label so this is
    well-defined).
    """
    spl = spl_bands.detach().to(torch.float64)
    best = 70
    for label in range(15, 71):
        curve = _curve_at(label)
        if bool((spl <= curve).all()):
            best = label
            break
    else:
        best = 70
    return best


def nc_soft(spl_bands: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    """Smooth relaxation used for the backward pass.

    For each candidate label we compute a soft "satisfaction" weight =
    sigmoid(min_slack * beta) where min_slack = min(curve - spl). Then take
    a softargmin over labels weighted toward smaller labels that still
    satisfy. This gives a scalar approximating NC but with non-zero gradient
    everywhere.
    """
    dtype = spl_bands.dtype
    labels = torch.tensor(list(range(15, 71)), dtype=dtype, device=spl_bands.device)
    # Build (L, 8) curves.
    curves = torch.stack([_curve_at(float(l)).to(dtype=dtype, device=spl_bands.device)
                          for l in labels])
    # slack = curve - spl (positive means satisfied)
    slacks = curves - spl_bands.unsqueeze(0)
    # min across bands (softmin via logsumexp for stability)
    min_slack = -(1.0 / beta) * torch.logsumexp(-beta * slacks, dim=1)
    satisfaction = torch.sigmoid(beta * min_slack)

    # Soft-argmin over labels: prefer smallest satisfying label.
    # Weight = satisfaction / exp(label).  Equivalent to: inverse-label
    # preference multiplied by satisfaction probability.
    weights = satisfaction * torch.exp(-0.1 * labels)
    weights = weights / (weights.sum() + 1e-12)
    return (labels * weights).sum()


class _NCStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spl_bands: torch.Tensor, soft_val: torch.Tensor) -> torch.Tensor:
        exact = torch.tensor(
            float(nc_exact(spl_bands)),
            dtype=spl_bands.dtype,
            device=spl_bands.device,
        )
        return exact + (soft_val - soft_val.detach())

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


def nc(spl_bands: torch.Tensor) -> torch.Tensor:
    """Differentiable NC: exact integer value with smooth gradient."""
    return _NCStraightThrough.apply(spl_bands, nc_soft(spl_bands))


def nc_band_exceedances(spl_bands: torch.Tensor, nc_value: int) -> torch.Tensor:
    """Per-band exceedance of the fitted NC curve (how much each band pokes
    above the curve — positive means controlling). The controlling bands
    are the ones with the largest exceedance at `nc_value - 1` (one below
    the rating), since those are what kept the rating from being lower.
    """
    curve_above = _curve_at(nc_value)
    curve_below = _curve_at(max(nc_value - 1, 15))
    # Bands that exceed the next-better (lower) curve are the ones pinning
    # the rating.
    return (spl_bands - curve_below).clamp(min=0.0)
