# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure tensor helpers for platform traversal shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def platform_terrain_type_start_index(fraction: float, num_cols: int) -> int:
    """Map a terrain fraction to Isaac Lab's first curriculum column."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1).")
    if num_cols <= 0:
        raise ValueError("num_cols must be positive.")
    return max(0, math.ceil((fraction - 0.001) * num_cols))


def platform_terrain_type_masks(
    terrain_types: torch.Tensor,
    num_cols: int,
    platform_start: float,
    descent_start: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify ordinary, platform-ascent, and platform-descent columns."""
    platform_min = platform_terrain_type_start_index(platform_start, num_cols)
    platform = terrain_types >= platform_min
    if descent_start is None:
        return platform, torch.zeros_like(platform)
    if descent_start <= platform_start:
        raise ValueError("descent_start must follow platform_start.")
    descent_min = platform_terrain_type_start_index(descent_start, num_cols)
    return platform, terrain_types >= descent_min


def platform_positive_reward_clip(
    total_reward: torch.Tensor,
    termination_reward: torch.Tensor,
) -> torch.Tensor:
    """Clip ordinary reward to zero before restoring the termination term."""
    return torch.clamp(total_reward - termination_reward, min=0.0) + termination_reward


@dataclass
class PlatformTraversalSettings:
    """All platform geometry, reward, and safety tuning in one place."""

    wheel_radius: float = 0.087
    probe_forward: float = 0.14
    min_step_height: float = 0.08
    surface_tolerance: float = 0.04
    support_force_threshold: float = 10.0
    wall_force_threshold: float = 20.0
    top_horizontal_force_ratio: float = 2.0
    command_threshold: float = 0.05
    support_history_steps: int = 10
    leading_ascent_speed: float = 0.60
    trailing_ascent_speed: float = 0.60
    ascent_slowdown_distance: float = 0.05
    ascent_min_speed: float = 0.20
    ascent_overshoot_margin: float = 0.04
    ascent_overshoot_ramp: float = 0.10
    ascent_overshoot_penalty_scale: float = 0.20
    leading_axle_sync_tolerance: float = 0.03
    leading_axle_sync_ramp: float = 0.07
    leading_descent_speed: float = 0.22
    trailing_descent_speed: float = 0.45
    speed_quadratic_blend: float = 0.5
    leading_ascent_contact_floor: float = 0.05
    trailing_ascent_contact_floor: float = 0.20
    backslide_penalty_scale: float = 0.50
    trailing_descent_wall_penalty_scale: float = 0.35
    trailing_descent_phase_penalty: float = 0.12
    air_time_grace: float = 0.04
    air_time_ramp: float = 0.08
    min_support_count: int = 2
    vertical_speed_scale: float = 0.35
    max_leading_descent_speed: float = 0.30
    max_trailing_descent_speed: float = 0.55
    max_descent_ang_vel: float = 1.0
    max_descent_tilt: float = math.radians(45.0)
    trailing_descent_air_time_grace: float = 0.08
    trailing_descent_air_time_ramp: float = 0.15
    min_trailing_descent_progress_speed: float = 0.25

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject settings that would invert gates or divide by zero."""
        positive = (
            "wheel_radius",
            "min_step_height",
            "surface_tolerance",
            "support_force_threshold",
            "wall_force_threshold",
            "top_horizontal_force_ratio",
            "support_history_steps",
            "leading_ascent_speed",
            "trailing_ascent_speed",
            "ascent_slowdown_distance",
            "ascent_min_speed",
            "ascent_overshoot_ramp",
            "leading_axle_sync_tolerance",
            "leading_axle_sync_ramp",
            "leading_descent_speed",
            "trailing_descent_speed",
            "air_time_ramp",
            "min_support_count",
            "vertical_speed_scale",
            "max_leading_descent_speed",
            "max_trailing_descent_speed",
            "max_descent_ang_vel",
            "trailing_descent_air_time_ramp",
            "min_trailing_descent_progress_speed",
        )
        nonnegative = (
            "probe_forward",
            "command_threshold",
            "backslide_penalty_scale",
            "ascent_overshoot_margin",
            "ascent_overshoot_penalty_scale",
            "trailing_descent_wall_penalty_scale",
            "trailing_descent_phase_penalty",
            "air_time_grace",
            "trailing_descent_air_time_grace",
        )
        unit_interval = (
            "speed_quadratic_blend",
            "leading_ascent_contact_floor",
            "trailing_ascent_contact_floor",
        )
        if (
            any(getattr(self, name) <= 0.0 for name in positive)
            or any(getattr(self, name) < 0.0 for name in nonnegative)
            or any(not 0.0 <= getattr(self, name) <= 1.0 for name in unit_interval)
            or self.ascent_min_speed > min(self.leading_ascent_speed, self.trailing_ascent_speed)
            or not 0.0 < self.max_descent_tilt < math.pi / 2
        ):
            raise ValueError("Invalid platform traversal settings.")


def _platform_sample_heights(
    ray_hits_w: torch.Tensor,
    query_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nearest query heights, scan validity, and total height span."""
    valid_rays = torch.isfinite(ray_hits_w).all(dim=2)
    valid_rays &= (torch.abs(ray_hits_w) < 1.0e6).all(dim=2)

    distance_xy_sq = torch.sum(
        torch.square(ray_hits_w[:, None, :, :2] - query_xy[:, :, None, :]),
        dim=-1,
    ).masked_fill(~valid_rays[:, None, :], torch.inf)
    nearest_ray = torch.argmin(distance_xy_sq, dim=2)
    nearest_height = torch.gather(ray_hits_w[:, :, 2], 1, nearest_ray)

    valid_scan = torch.any(valid_rays, dim=1)
    nearest_height = torch.where(valid_scan[:, None], nearest_height, 0.0)
    ray_heights = ray_hits_w[..., 2]
    height_min = torch.amin(
        torch.where(valid_rays, ray_heights, torch.full_like(ray_heights, torch.inf)),
        dim=1,
    )
    height_max = torch.amax(
        torch.where(valid_rays, ray_heights, torch.full_like(ray_heights, -torch.inf)),
        dim=1,
    )
    return nearest_height, valid_scan, height_max - height_min


def _platform_travel_order(
    values: torch.Tensor,
    forward_travel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return leading/trailing axle values for positive or negative X travel."""
    front, rear = values[:, :2], values[:, 2:]
    selector = forward_travel.reshape((-1,) + (1,) * (front.ndim - 1))
    return torch.where(selector, front, rear), torch.where(selector, rear, front)


def _platform_step_direction(
    surface_delta: torch.Tensor,
    min_step_height: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify a left/right axle target as ascent, descent, or flat."""
    ascent = torch.amin(surface_delta, dim=1) > min_step_height
    descent = torch.amax(surface_delta, dim=1) < -min_step_height
    direction = torch.where(
        ascent,
        torch.ones_like(surface_delta[:, 0]),
        torch.where(
            descent,
            -torch.ones_like(surface_delta[:, 0]),
            torch.zeros_like(surface_delta[:, 0]),
        ),
    )[:, None].expand_as(surface_delta)
    return ascent | descent, descent, direction


def _platform_axle_on_destination(
    wheel_height: torch.Tensor,
    surface_here: torch.Tensor,
    surface_ahead: torch.Tensor,
    destination_surface: torch.Tensor,
    stable_support: torch.Tensor,
    wheel_radius: float,
    tolerance: float,
) -> torch.Tensor:
    """Require stable support on the leading axle's destination surface."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    supported = (
        (torch.abs(wheel_height - surface_here - wheel_radius) < tolerance)
        & (torch.abs(surface_here - surface_ahead) < tolerance)
        & (torch.abs(surface_here - destination_surface) < tolerance)
        & stable_support
    )
    return torch.all(supported, dim=-1)


def _platform_resolve_trailing_state(
    local_direction: torch.Tensor,
    local_transition_detected: torch.Tensor,
    scan_spans_transition: torch.Tensor,
    leading_transition_detected: torch.Tensor,
    leading_is_descent: torch.Tensor,
    height_delta: torch.Tensor,
    air_time: torch.Tensor,
    target_surface_error: torch.Tensor,
    surface_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve trailing direction, descent phase, and exposed-wall mask."""
    if surface_tolerance <= 0.0:
        raise ValueError("surface_tolerance must be positive.")

    meaningful_target_error = torch.abs(height_delta) > surface_tolerance
    inferred_after_crossing = torch.where(
        meaningful_target_error,
        torch.sign(-height_delta),
        torch.zeros_like(height_delta),
    )
    can_infer_after_crossing = scan_spans_transition & (~leading_transition_detected | ~leading_is_descent)
    direction = torch.where(
        local_transition_detected[:, None],
        local_direction,
        torch.where(
            can_infer_after_crossing[:, None],
            inferred_after_crossing,
            torch.zeros_like(local_direction),
        ),
    )

    airborne_above_target = torch.any((height_delta > 0.0) & (air_time > 0.0), dim=-1)
    is_descent = (
        torch.any(direction < 0.0, dim=-1)
        | (scan_spans_transition & leading_is_descent)
        | (scan_spans_transition & airborne_above_target)
    )
    exposed_wall = (target_surface_error < surface_tolerance) & (height_delta > surface_tolerance)
    return direction, is_descent, exposed_wall


def _platform_phase_masks(
    active: torch.Tensor,
    edge_visible: torch.Tensor,
    leading_transition_detected: torch.Tensor,
    leading_completed: torch.Tensor,
    trailing_on_target: torch.Tensor,
    leading_is_descent: torch.Tensor,
    trailing_is_descent: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Build current-frame traversal phases without a persistent latch."""
    leading = active & leading_transition_detected & ~leading_completed
    trailing = active & leading_completed & ~trailing_on_target
    sequence = edge_visible & ~leading_completed
    leading_descent = leading & leading_is_descent
    trailing_descent = trailing & trailing_is_descent
    return sequence, leading, trailing, leading_descent, trailing_descent


def _platform_wall_contact_scores(
    contact_forces_xy: torch.Tensor,
    travel_xy: torch.Tensor,
    force_threshold: float,
    along_contact_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project wall force opposite to ascent travel and along descent travel."""
    if force_threshold <= 0.0:
        raise ValueError("force_threshold must be positive.")

    force_along_travel = torch.sum(contact_forces_xy * travel_xy, dim=-1)
    opposing_travel = torch.clamp(-force_along_travel / force_threshold, 0.0, 1.0)
    along_travel = torch.clamp(force_along_travel / force_threshold, 0.0, 1.0)
    along_travel *= along_contact_mask.to(force_along_travel.dtype)
    return opposing_travel, along_travel


def _platform_landing_force_penalty(
    net_forces_w_history: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Sum peak upward contact force above a threshold."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    forces_z = torch.clamp(net_forces_w_history[..., 2], min=0.0)
    peak_force_z = torch.amax(forces_z, dim=1)
    return torch.sum(torch.clamp(peak_force_z - threshold, min=0.0), dim=1)


def _platform_mean_scan_height(
    ray_hits_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean scan height and an all-rays-valid mask."""
    valid = torch.isfinite(ray_hits_w).all(dim=(1, 2))
    valid &= (torch.abs(ray_hits_w) < 1.0e6).all(dim=(1, 2))
    safe_heights = torch.where(
        valid[:, None],
        ray_hits_w[..., 2],
        torch.zeros_like(ray_hits_w[..., 2]),
    )
    return torch.mean(safe_heights, dim=1, keepdim=True), valid


def _platform_descent_stall_cost(
    air_time: torch.Tensor,
    vertical_velocity: torch.Tensor,
    grace: float,
    ramp: float,
    min_downward_speed: float,
) -> torch.Tensor:
    """Penalize prolonged trailing-wheel air time only when descent has stalled."""
    if grace < 0.0 or ramp <= 0.0 or min_downward_speed <= 0.0:
        raise ValueError("Invalid descent stall parameters.")

    air_gate = torch.clamp((air_time - grace) / ramp, 0.0, 1.0)
    downward_speed = torch.clamp(-vertical_velocity, min=0.0)
    stall_ratio = torch.clamp(
        (min_downward_speed - downward_speed) / min_downward_speed,
        0.0,
        1.0,
    )
    return torch.mean(air_gate * stall_ratio, dim=-1)


def _platform_toward_target_progress(
    vertical_velocity: torch.Tensor,
    direction: torch.Tensor,
    height_delta: torch.Tensor,
    wall_score: torch.Tensor,
    ascent_speed: float,
    descent_speed: float,
    ascent_slowdown_distance: float,
    ascent_min_speed: float,
    speed_quadratic_blend: float,
    ascent_contact_floor: float,
    backslide_penalty_scale: float,
) -> torch.Tensor:
    """Return signed per-wheel progress toward the selected surface."""
    remaining_distance = torch.clamp(-direction * height_delta, min=0.0)
    ascent_distance_ratio = torch.clamp(
        remaining_distance / ascent_slowdown_distance,
        min=0.0,
        max=1.0,
    )
    ascent_speed_limit = ascent_min_speed + (ascent_speed - ascent_min_speed) * ascent_distance_ratio
    speed_limit = torch.where(
        direction > 0.0,
        ascent_speed_limit,
        torch.full_like(direction, descent_speed),
    )
    raw_speed_ratio = direction * vertical_velocity / speed_limit
    speed_ratio = torch.clamp(raw_speed_ratio, min=-1.0, max=1.0)
    toward_ratio = torch.clamp(speed_ratio, min=0.0)
    backslide_ratio = torch.clamp(-speed_ratio, min=0.0)
    toward_progress = (1.0 - speed_quadratic_blend) * toward_ratio + speed_quadratic_blend * torch.square(toward_ratio)
    ascent_overspeed = torch.clamp(raw_speed_ratio - 1.0, min=0.0, max=1.0)
    toward_progress = torch.where(
        direction > 0.0,
        torch.clamp(toward_progress - torch.square(ascent_overspeed), min=0.0),
        toward_progress,
    )
    backslide_progress = (1.0 - speed_quadratic_blend) * backslide_ratio + speed_quadratic_blend * torch.square(
        backslide_ratio
    )
    ascent_gate = ascent_contact_floor + (1.0 - ascent_contact_floor) * wall_score
    contact_gate = torch.where(
        direction > 0.0,
        ascent_gate,
        torch.ones_like(wall_score),
    )
    return contact_gate * toward_progress - backslide_penalty_scale * backslide_progress


def _platform_ascent_overshoot(
    height_delta: torch.Tensor,
    ascent_mask: torch.Tensor,
    margin: float,
    ramp: float,
) -> torch.Tensor:
    """Return mean axle over-height while moving upward."""
    overshoot = torch.square(torch.clamp((height_delta - margin) / ramp, min=0.0, max=1.0))
    return torch.mean(overshoot * ascent_mask.float(), dim=1)


def _squared_excess(
    value: torch.Tensor,
    threshold: float,
    scale: float,
) -> torch.Tensor:
    """Return a squared, unit-clamped excess above a threshold."""
    return torch.square(torch.clamp((value - threshold) / scale, min=0.0, max=1.0))
