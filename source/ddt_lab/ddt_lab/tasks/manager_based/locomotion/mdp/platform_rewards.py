# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small reward and diagnostic terms used by the platform locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from .rewards import _fit_height_scan_plane_yaw, _terrain_type_mask, feet_stumble

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ASCENT_DIAGNOSTIC_TERM = "ascent_diagnostics"


def _stable_horizontal_support(
    contact_sensor: ContactSensor,
    body_ids: list[int],
    support_force_threshold: float,
    top_horizontal_force_ratio: float,
    support_history_steps: int,
    stable_history_fraction: float,
) -> torch.Tensor:
    """Return wheels with sustained, predominantly upward contact force."""
    if support_force_threshold <= 0.0:
        raise ValueError("support_force_threshold must be positive.")
    if top_horizontal_force_ratio <= 0.0:
        raise ValueError("top_horizontal_force_ratio must be positive.")
    if support_history_steps <= 0:
        raise ValueError("support_history_steps must be positive.")
    if not 0.0 < stable_history_fraction <= 1.0:
        raise ValueError("stable_history_fraction must be in (0, 1].")

    history_steps = min(support_history_steps, contact_sensor.data.net_forces_w_history.shape[1])
    forces = contact_sensor.data.net_forces_w_history[:, :history_steps, body_ids]
    force_z = torch.clamp(forces[..., 2], min=0.0)
    force_xy = torch.linalg.norm(forces[..., :2], dim=-1)
    horizontal_support = (force_z > support_force_threshold) & (
        force_xy < top_horizontal_force_ratio * force_z
    )
    required_steps = max(1, int(stable_history_fraction * history_steps + 0.999))
    return horizontal_support[:, 0] & (horizontal_support.sum(dim=1) >= required_steps)


def _height_scan_transition(
    asset: RigidObject,
    terrain_sensor: RayCaster,
    plane_residual_gate_std: float,
    transition_gate_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return discontinuity mask plus the lowest and highest scanned tread."""
    if plane_residual_gate_std <= 0.0:
        raise ValueError("plane_residual_gate_std must be positive.")
    if not 0.0 < transition_gate_threshold < 1.0:
        raise ValueError("transition_gate_threshold must be in (0, 1).")

    ray_hits_z = terrain_sensor.data.ray_hits_w[..., 2]
    finite_hits = torch.isfinite(ray_hits_z).all(dim=1)
    _, plane_residual, plane_valid = _fit_height_scan_plane_yaw(asset, terrain_sensor)
    planar_gate = torch.exp(-torch.square(plane_residual / plane_residual_gate_std))
    transition = finite_hits & plane_valid & (planar_gate < transition_gate_threshold)
    safe_hits_z = torch.where(
        finite_hits[:, None],
        ray_hits_z,
        torch.zeros_like(ray_hits_z),
    )
    return transition, safe_hits_z.amin(dim=1), safe_hits_z.amax(dim=1)


def _first_episode_event(
    env: ManagerBasedRLEnv, name: str, condition: torch.Tensor
) -> torch.Tensor:
    """Return only the first occurrence of a condition in each episode."""
    occurred = getattr(env, name, None)
    if occurred is None or occurred.shape != condition.shape:
        occurred = torch.zeros_like(condition)
    just_reset = env.episode_length_buf <= 1
    occurred = occurred & ~just_reset
    event = condition & ~occurred
    setattr(env, name, (occurred | condition).detach())
    return event


def _log_ascent_events(env: ManagerBasedRLEnv, events: dict[str, torch.Tensor]) -> None:
    """Expose event counts through the standard episodic TensorBoard logger."""
    reward_manager = env.reward_manager
    parent = reward_manager._episode_sums[_ASCENT_DIAGNOSTIC_TERM]
    episode_scale = float(env.max_episode_length_s)
    for name, event in events.items():
        key = f"{_ASCENT_DIAGNOSTIC_TERM}/{name}"
        if key not in reward_manager._episode_sums:
            reward_manager._episode_sums[key] = torch.zeros_like(parent)
        reward_manager._episode_sums[key].add_(
            event.detach().to(device=parent.device, dtype=parent.dtype),
            alpha=episode_scale,
        )


def platform_feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    terrain_type_start: float,
    force_ratio: float = 5.0,
    upright_gate: bool = True,
) -> torch.Tensor:
    """Apply the generic stumble cost only outside platform columns."""
    cost = feet_stumble(env, sensor_cfg, force_ratio, upright_gate)
    return cost * (~_terrain_type_mask(env, terrain_type_start)).float()


def platform_landing_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize only excessive upward wheel force, leaving wall-normal force free."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.clamp(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2], min=0.0
    )
    peak_force_z = torch.amax(forces_z, dim=1)
    return torch.sum(torch.clamp(peak_force_z - threshold, min=0.0), dim=1)


def platform_descent_front_impact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    terrain_type_start: float,
    command_threshold: float = 0.10,
) -> torch.Tensor:
    """Penalize excessive front-wheel landing force only during forward descents."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if command_threshold < 0.0:
        raise ValueError("command_threshold must be non-negative.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.clamp(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2], min=0.0
    )
    peak_force_z = torch.amax(forces_z, dim=1)
    excess_force = torch.sum(torch.clamp(peak_force_z - threshold, min=0.0), dim=1)
    forward_command = env.command_manager.get_command(command_name)[:, 0] > command_threshold
    forward_descent = _terrain_type_mask(env, terrain_type_start) & forward_command
    return excess_force * forward_descent.float()


def platform_ascent_rear_lateral_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    terrain_type_start: float,
    terrain_type_end: float,
) -> torch.Tensor:
    """Penalize only excessive rear-wheel horizontal force on ascent platforms."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_xy = torch.linalg.norm(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :2], dim=-1
    )
    peak_force_xy = torch.amax(forces_xy, dim=1)
    excess_force = torch.sum(torch.clamp(peak_force_xy - threshold, min=0.0), dim=1)
    ascent_terrain = _terrain_type_mask(env, terrain_type_start) & ~_terrain_type_mask(
        env, terrain_type_end
    )
    return excess_force * ascent_terrain.float()


def platform_ascent_diagnostics(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg,
    wheel_sensor_cfg: SceneEntityCfg,
    wheel_asset_cfg: SceneEntityCfg,
    command_name: str,
    terrain_type_start: float,
    terrain_type_end: float,
    wheel_radius: float = 0.087,
    command_threshold: float = 0.10,
    lift_height: float = 0.04,
    surface_tolerance: float = 0.04,
    support_force_threshold: float = 10.0,
    top_horizontal_force_ratio: float = 2.0,
    support_history_steps: int = 10,
    stable_history_fraction: float = 0.8,
    plane_residual_gate_std: float = 0.015,
    transition_gate_threshold: float = 0.10,
) -> torch.Tensor:
    """Log whether exploration reaches each ascent milestone without shaping reward."""
    if not 0.0 <= terrain_type_start < terrain_type_end < 1.0:
        raise ValueError("terrain type bounds must satisfy 0 <= start < end < 1.")
    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive.")
    if lift_height <= 0.0:
        raise ValueError("lift_height must be positive.")
    if surface_tolerance <= 0.0:
        raise ValueError("surface_tolerance must be positive.")

    asset: Articulation = env.scene[wheel_asset_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[wheel_sensor_cfg.name]
    transition, lower_surface_z, _ = _height_scan_transition(
        asset,
        terrain_sensor,
        plane_residual_gate_std,
        transition_gate_threshold,
    )
    ascent_terrain = _terrain_type_mask(env, terrain_type_start) & ~_terrain_type_mask(
        env, terrain_type_end
    )
    command_active = env.command_manager.get_command(command_name)[:, 0] > command_threshold
    opportunity = ascent_terrain & command_active & transition

    stable_support = _stable_horizontal_support(
        contact_sensor,
        wheel_sensor_cfg.body_ids,
        support_force_threshold,
        top_horizontal_force_ratio,
        support_history_steps,
        stable_history_fraction,
    )
    wheel_pos_z = asset.data.body_pos_w[:, wheel_asset_cfg.body_ids, 2]
    wheel_bottom_z = wheel_pos_z - wheel_radius
    front_lifted = torch.any(
        wheel_bottom_z[:, :2] > lower_surface_z[:, None] + lift_height,
        dim=1,
    )
    ray_hits_w = terrain_sensor.data.ray_hits_w
    wheel_xy = asset.data.body_pos_w[:, wheel_asset_cfg.body_ids, :2]
    ray_distance_sq = torch.sum(
        torch.square(ray_hits_w[:, None, :, :2] - wheel_xy[:, :, None, :]),
        dim=-1,
    )
    nearest_ray = torch.argmin(ray_distance_sq, dim=-1)
    local_surface_z = torch.gather(ray_hits_w[..., 2], 1, nearest_ray)
    on_local_surface = torch.abs(wheel_bottom_z - local_surface_z) < surface_tolerance
    above_lower_tread = local_surface_z > lower_surface_z[:, None] + lift_height
    stable_upper = stable_support & on_local_surface & above_lower_tread
    front_axle_upper = torch.all(stable_upper[:, :2], dim=1)
    full_ascent = torch.all(stable_upper, dim=1)

    _log_ascent_events(
        env,
        {
            "edge_opportunity": _first_episode_event(
                env, "_platform_ascent_opportunity_event", opportunity
            ),
            "front_lift": _first_episode_event(
                env, "_platform_ascent_front_lift_event", opportunity & front_lifted
            ),
            "front_axle_upper": _first_episode_event(
                env,
                "_platform_ascent_front_axle_upper_event",
                opportunity & front_axle_upper,
            ),
            "full_ascent": _first_episode_event(
                env, "_platform_ascent_full_event", opportunity & full_ascent
            ),
        },
    )
    return torch.zeros(env.num_envs, device=env.device)


def platform_descent_axle_support_penalty(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg,
    wheel_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    terrain_type_start: float,
    command_threshold: float = 0.10,
    support_force_threshold: float = 10.0,
    top_horizontal_force_ratio: float = 2.0,
    support_history_steps: int = 10,
    stable_history_fraction: float = 0.8,
    plane_residual_gate_std: float = 0.015,
    transition_gate_threshold: float = 0.10,
) -> torch.Tensor:
    """Penalize descent transitions with neither complete axle stably supported."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[wheel_sensor_cfg.name]
    transition, _, _ = _height_scan_transition(
        asset,
        terrain_sensor,
        plane_residual_gate_std,
        transition_gate_threshold,
    )
    stable_support = _stable_horizontal_support(
        contact_sensor,
        wheel_sensor_cfg.body_ids,
        support_force_threshold,
        top_horizontal_force_ratio,
        support_history_steps,
        stable_history_fraction,
    )
    front_axle_supported = torch.all(stable_support[:, :2], dim=1)
    rear_axle_supported = torch.all(stable_support[:, 2:], dim=1)
    command_active = torch.abs(env.command_manager.get_command(command_name)[:, 0]) > command_threshold
    active = _terrain_type_mask(env, terrain_type_start) & command_active & transition
    return (active & ~front_axle_supported & ~rear_axle_supported).float()
