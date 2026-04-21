"""User-configurable duct chain -> receiver SPL spectrum.

The user builds a linear chain of blocks in the UI; we walk it, accumulate
the PWL at each node, apply end reflection, then the room correction. All
per-block parameters are torch.Tensor so gradients flow through the whole
chain to every parameter the user is sliding.

Block types:
    DuctSegment       natural attenuation over a straight length
    Bend              natural attenuation (short) + regenerative PWL
    Attenuator        insertion loss of a splitter attenuator; length=0 off
    Diffuser          end reflection + regenerative; becomes the PWL at the
                      duct termination

Then the Room takes the diffuser PWL and produces SPL at the receiver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from duct_poc.metrics.octave import OCTAVE_BANDS_HZ, N_BANDS
from duct_poc.metrics.nc import nc as nc_metric, nc_exact
from duct_poc.physics.source import source_pwl, SOURCE_SHAPES
from duct_poc.physics.duct_attenuation import duct_attenuation_db
from duct_poc.physics.attenuator import attenuator_il
from duct_poc.physics.end_reflection import (
    end_reflection_loss_db, effective_diameter_rectangular,
)
from duct_poc.physics.regenerative import regenerative_pwl_db
from duct_poc.physics.room import receiver_spl


BlockType = Literal["segment", "bend", "attenuator", "diffuser"]


def db_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Logarithmic (energy) addition of two dB spectra, per band."""
    return 10.0 * torch.log10(
        torch.pow(10.0, a / 10.0) + torch.pow(10.0, b / 10.0)
    )


class DuctSegment(nn.Module):
    """Straight duct segment (rectangular). Velocity is derived from
    system flow rate Q divided by the segment's cross-section area."""
    def __init__(self, length_m: float, width_m: float, height_m: float,
                 lining_thickness_mm: float = 0.0):
        super().__init__()
        self.log_length = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(max(length_m, 1e-3))))))
        self.log_width = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(width_m)))))
        self.log_height = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(height_m)))))
        self.lining_raw = nn.Parameter(torch.tensor(float(lining_thickness_mm) / 1000.0))

    @property
    def length(self): return torch.exp(self.log_length)
    @property
    def width(self): return torch.exp(self.log_width)
    @property
    def height(self): return torch.exp(self.log_height)
    @property
    def lining_thickness(self): return torch.clamp(self.lining_raw, min=0.0)
    @property
    def area(self): return self.width * self.height

    def flow_velocity(self, q_m3s: torch.Tensor) -> torch.Tensor:
        return q_m3s / self.area

    def forward(self, pwl_in: torch.Tensor, q_m3s: torch.Tensor) -> torch.Tensor:
        atten = duct_attenuation_db(self.length, self.width, self.height, self.lining_thickness)
        return pwl_in - atten


class Bend(nn.Module):
    """90-degree bend: small natural loss + regenerative noise.
    Velocity derived from Q and block area."""
    def __init__(self, width_m: float, height_m: float):
        super().__init__()
        self.log_width = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(width_m)))))
        self.log_height = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(height_m)))))

    @property
    def width(self): return torch.exp(self.log_width)
    @property
    def height(self): return torch.exp(self.log_height)
    @property
    def area(self): return self.width * self.height

    def flow_velocity(self, q_m3s: torch.Tensor) -> torch.Tensor:
        return q_m3s / self.area

    def forward(self, pwl_in: torch.Tensor, q_m3s: torch.Tensor) -> torch.Tensor:
        natural = duct_attenuation_db(
            torch.tensor(1.0, dtype=pwl_in.dtype),
            self.width, self.height,
            torch.tensor(0.0, dtype=pwl_in.dtype),
        )
        regen = regenerative_pwl_db(self.flow_velocity(q_m3s), self.area)
        return db_add(pwl_in - natural, regen)


class Attenuator(nn.Module):
    """Splitter attenuator block. Face velocity derived from Q and
    gross area; regenerative velocity uses the free-area cross-section
    (Q / (w * h * free_area_ratio)) — this is the classic self-noise
    trap: tight attenuators regenerate aggressively."""
    def __init__(self, length_m: float, width_m: float, height_m: float,
                 free_area_ratio: float = 0.5,
                 splitter_thickness_mm: float = 100.0):
        super().__init__()
        self.length_raw = nn.Parameter(torch.tensor(float(length_m)))
        self.log_width = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(width_m)))))
        self.log_height = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(height_m)))))
        self.free_area_raw = nn.Parameter(torch.tensor(float(free_area_ratio)))
        self.splitter_raw = nn.Parameter(torch.tensor(float(splitter_thickness_mm)/1000.0))

    @property
    def length(self): return torch.clamp(self.length_raw, min=0.0)
    @property
    def width(self): return torch.exp(self.log_width)
    @property
    def height(self): return torch.exp(self.log_height)
    @property
    def free_area_ratio(self): return torch.clamp(self.free_area_raw, 0.1, 0.9)
    @property
    def splitter_thickness(self): return torch.clamp(self.splitter_raw, min=0.0)
    @property
    def gross_area(self): return self.width * self.height
    @property
    def free_area(self): return self.gross_area * self.free_area_ratio

    def flow_velocity(self, q_m3s: torch.Tensor) -> torch.Tensor:
        """Face (gross-area) velocity, shown in the UI as 'entering' velocity."""
        return q_m3s / self.gross_area

    def free_area_velocity(self, q_m3s: torch.Tensor) -> torch.Tensor:
        """Velocity through the narrowed free-area section — drives self-noise."""
        return q_m3s / self.free_area

    def forward(self, pwl_in: torch.Tensor, q_m3s: torch.Tensor) -> torch.Tensor:
        il = attenuator_il(self.length, self.free_area_ratio, self.splitter_thickness)
        # Regenerative self-noise uses free-area velocity and area.
        regen = regenerative_pwl_db(self.free_area_velocity(q_m3s), self.free_area)
        return db_add(pwl_in - il, regen)


class Diffuser(nn.Module):
    """Diffuser: end reflection + regenerative noise. Velocity derived
    from Q and face area."""
    def __init__(self, width_m: float, height_m: float):
        super().__init__()
        self.log_width = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(width_m)))))
        self.log_height = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(height_m)))))

    @property
    def width(self): return torch.exp(self.log_width)
    @property
    def height(self): return torch.exp(self.log_height)
    @property
    def area(self): return self.width * self.height

    def flow_velocity(self, q_m3s: torch.Tensor) -> torch.Tensor:
        return q_m3s / self.area

    def forward(self, pwl_in: torch.Tensor, q_m3s: torch.Tensor) -> torch.Tensor:
        d_eff = effective_diameter_rectangular(self.width, self.height)
        erl = end_reflection_loss_db(d_eff)
        regen = regenerative_pwl_db(self.flow_velocity(q_m3s), self.area)
        return db_add(pwl_in - erl, regen)


class DuctSystem(nn.Module):
    """Full system: source -> user-defined chain -> diffuser -> room.

    Volume flow rate Q is a *single* system-level design parameter.
    Each block's velocity is derived as Q / A_block (except attenuators,
    which use the free-area cross-section for self-noise). This matches
    how HVAC is actually sized: the designer picks a flow and lets
    velocities fall out of the geometry."""

    def __init__(
        self,
        source_type: str = "FCU (ceiling)",
        source_offset_db: float = 0.0,
        volume_flow_m3s: float = 0.25,
        blocks: list[nn.Module] | None = None,
        room_volume_m3: float = 50.0,
        rt60_s: float = 0.6,
        receiver_distance_m: float = 2.0,
    ):
        super().__init__()
        self.source_type = source_type
        self.source_offset = nn.Parameter(torch.tensor(float(source_offset_db)))
        self.log_volume_flow = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(volume_flow_m3s))))
        )
        self.blocks = nn.ModuleList(blocks or [])
        self.log_room_volume = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(room_volume_m3))))
        )
        self.log_rt60 = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(rt60_s))))
        )
        self.log_receiver_distance = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(receiver_distance_m))))
        )

    @property
    def volume_flow(self): return torch.exp(self.log_volume_flow)
    @property
    def room_volume(self): return torch.exp(self.log_room_volume)
    @property
    def rt60(self): return torch.exp(self.log_rt60)
    @property
    def receiver_distance(self): return torch.exp(self.log_receiver_distance)

    def diffuser_pwl(self) -> torch.Tensor:
        q = self.volume_flow
        pwl = source_pwl(self.source_type, self.source_offset)
        for blk in self.blocks:
            pwl = blk(pwl, q)
        return pwl

    def receiver_spl(self) -> torch.Tensor:
        return receiver_spl(
            self.diffuser_pwl(),
            self.room_volume, self.rt60,
            self.receiver_distance,
        )

    def nc(self) -> torch.Tensor:
        return nc_metric(self.receiver_spl())
