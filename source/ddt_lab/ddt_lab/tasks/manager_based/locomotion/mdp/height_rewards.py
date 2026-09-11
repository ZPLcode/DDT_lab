# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms shared only by commanded-height all-terrain tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _terrain_type_mask(
    env: ManagerBasedRLEnv,
    terrain_type_start: float,
) -> torch.Tensor:
    """Select the final fraction of terrain columns."""
    if not 0.0 <= terrain_type_start < 1.0:
        raise ValueError(
            f"terrain_type_start must be in [0, 1); received {terrain_type_start}."
        )
    terrain = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    first_type = math.ceil((terrain_type_start - 0.001) * terrain_generator.num_cols)
    return terrain.terrain_types >= first_type


def _fit_height_scan_plane_yaw(
    asset: RigidObject,
    sensor: RayCaster,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a plane to a yaw-aligned scan and return normal, residual, and validity."""
    raw_hits = sensor.data.ray_hits_w
    valid = torch.isfinite(raw_hits).all(dim=(1, 2))
    valid &= (torch.abs(raw_hits) < 1.0e6).all(dim=(1, 2))
    valid &= torch.isfinite(sensor.data.pos_w).all(dim=1)
    valid &= torch.isfinite(asset.data.root_quat_w).all(dim=1)

    hits = torch.where(valid[:, None, None], raw_hits, torch.zeros_like(raw_hits))
    sensor_pos_w = torch.where(valid[:, None], sensor.data.pos_w, torch.zeros_like(sensor.data.pos_w))
    root_quat_w = torch.where(valid[:, None], asset.data.root_quat_w, torch.zeros_like(asset.data.root_quat_w))
    root_quat_w[:, 0] = torch.where(valid, root_quat_w[:, 0], torch.ones_like(root_quat_w[:, 0]))

    relative_hits = hits - sensor_pos_w.unsqueeze(1)
    num_rays = relative_hits.shape[1]
    yaw = yaw_quat(root_quat_w)
    yaw_expanded = yaw.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4)
    hits_yaw = quat_apply_inverse(yaw_expanded, relative_hits.reshape(-1, 3)).reshape_as(relative_hits)

    centered = hits_yaw - hits_yaw.mean(dim=1, keepdim=True)
    x, y, z = centered.unbind(dim=2)
    # Solve z = slope_x * x + slope_y * y with 2-D normal equations.
    xx = torch.mean(x * x, dim=1)
    yy = torch.mean(y * y, dim=1)
    xy = torch.mean(x * y, dim=1)
    xz = torch.mean(x * z, dim=1)
    yz = torch.mean(y * z, dim=1)
    determinant = torch.clamp(xx * yy - xy * xy, min=1.0e-8)
    slope_x = (xz * yy - yz * xy) / determinant
    slope_y = (yz * xx - xz * xy) / determinant

    residual = z - slope_x.unsqueeze(1) * x - slope_y.unsqueeze(1) * y
    residual_rms = torch.sqrt(torch.mean(torch.square(residual), dim=1) + 1.0e-8)
    normal_yaw = torch.stack((-slope_x, -slope_y, torch.ones_like(slope_x)), dim=1)
    normal_yaw = torch.nn.functional.normalize(normal_yaw, dim=1)
    normal_w = math_utils.quat_apply(yaw, normal_yaw)
    return normal_w, residual_rms, valid


def _flat_terrain_gate(
    asset: RigidObject,
    sensor: RayCaster,
    slope_std: float | None,
    residual_std: float | None,
) -> torch.Tensor:
    """Fade shaping on sloped or non-planar terrain."""
    if slope_std is not None and slope_std <= 0.0:
        raise ValueError("slope_std must be positive.")
    if residual_std is not None and residual_std <= 0.0:
        raise ValueError("residual_std must be positive.")

    terrain_normal_w, residual_rms, valid = _fit_height_scan_plane_yaw(asset, sensor)
    gate = torch.ones_like(residual_rms)
    if slope_std is not None:
        slope = torch.linalg.norm(terrain_normal_w[:, :2], dim=1) / torch.clamp(terrain_normal_w[:, 2].abs(), min=0.2)
        gate *= torch.exp(-torch.square(slope / slope_std))
    if residual_std is not None:
        gate *= torch.exp(-torch.square(residual_rms / residual_std))
    return torch.where(valid, gate, torch.zeros_like(gate))


def terrain_track_base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    plane_residual_sensor_cfg: SceneEntityCfg | None = None,
    plane_residual_gate_std: float | None = None,
) -> torch.Tensor:
    """Track commanded clearance above local terrain and relax it at discontinuities."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    if plane_residual_gate_std is not None and plane_residual_gate_std <= 0.0:
        raise ValueError("plane_residual_gate_std must be positive.")

    asset: RigidObject = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 0]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ray_heights = sensor.data.ray_hits_w[..., 2]
        valid = torch.isfinite(ray_heights).all(dim=1)
        valid &= (torch.abs(ray_heights) < 1.0e6).all(dim=1)
        sanitized = torch.where(valid[:, None], ray_heights, torch.zeros_like(ray_heights))
        local_height = asset.data.root_pos_w[:, 2] - torch.mean(sanitized, dim=1)
        height = torch.where(valid, local_height, height)

    reward = torch.exp(-torch.square(target - height) / std**2)
    if plane_residual_sensor_cfg is not None and plane_residual_gate_std is not None:
        residual_sensor: RayCaster = env.scene.sensors[plane_residual_sensor_cfg.name]
        _, residual_rms, valid = _fit_height_scan_plane_yaw(asset, residual_sensor)
        residual_gate = torch.exp(-torch.square(residual_rms / plane_residual_gate_std))
        residual_gate = torch.where(valid, residual_gate, torch.ones_like(residual_gate))
        reward = residual_gate * reward + (1.0 - residual_gate)

    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward * upright_gate


def smooth_ground_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    height_spread_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame tilt only where the footprint scan is flat."""
    if height_spread_std <= 0.0:
        raise ValueError("height_spread_std must be positive.")

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raw_hits = sensor.data.ray_hits_w
    valid = torch.isfinite(raw_hits).all(dim=(1, 2))
    valid &= (torch.abs(raw_hits) < 1.0e6).all(dim=(1, 2))
    valid &= torch.isfinite(asset.data.projected_gravity_b).all(dim=1)
    heights = torch.where(valid[:, None], raw_hits[..., 2], torch.zeros_like(raw_hits[..., 2]))
    height_spread = torch.amax(heights, dim=1) - torch.amin(heights, dim=1)
    flat_gate = torch.exp(-torch.square(height_spread / height_spread_std))
    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.where(valid, tilt_error * flat_gate, torch.zeros_like(tilt_error))


def wheel_thigh_x_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    thigh_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    terrain_slope_gate_std: float | None = None,
    plane_residual_gate_std: float | None = None,
) -> torch.Tensor:
    """Penalize wheel-axle displacement from its thigh joint along root-frame X."""
    if std <= 0.0:
        raise ValueError("std must be positive.")

    asset: Articulation = env.scene[thigh_cfg.name]
    wheel_from_thigh_w = (
        asset.data.body_link_pos_w[:, wheel_cfg.body_ids, :] - asset.data.body_link_pos_w[:, thigh_cfg.body_ids, :]
    )
    num_feet = wheel_from_thigh_w.shape[1]
    root_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    wheel_from_thigh_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), wheel_from_thigh_w.reshape(-1, 3)).reshape(
        env.num_envs, num_feet, 3
    )
    penalty = torch.mean(torch.square(wheel_from_thigh_b[:, :, 0] / std), dim=1)

    if terrain_sensor_cfg is not None:
        terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        penalty *= _flat_terrain_gate(
            asset,
            terrain_sensor,
            terrain_slope_gate_std,
            plane_residual_gate_std,
        )
    return penalty


def flat_yaw_hip_pos_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    yaw_threshold: float = 0.10,
    tolerance: float = 0.05,
    terrain_slope_gate_std: float = 0.05,
    plane_residual_gate_std: float = 0.015,
) -> torch.Tensor:
    """Penalize excessive hip deflection during flat-ground yaw commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.square(torch.clamp(hip_pos.abs() - tolerance, min=0.0)).sum(dim=-1)
    command = env.command_manager.get_command(command_name)
    yaw_gate = (torch.abs(command[:, 2]) > yaw_threshold).float()
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    flat_gate = _flat_terrain_gate(
        asset,
        terrain_sensor,
        terrain_slope_gate_std,
        plane_residual_gate_std,
    )
    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * yaw_gate * flat_gate * upright_gate


def landing_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize peak upward wheel force above a threshold."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.clamp(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2], min=0.0)
    peak_force_z = torch.amax(forces_z, dim=1)
    return torch.sum(torch.clamp(peak_force_z - threshold, min=0.0), dim=1)


def height_feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_ratio: float = 4.0,
) -> torch.Tensor:
    """Penalize feet whose horizontal contact force dominates vertical support."""
    if force_ratio < 0.0:
        raise ValueError("force_ratio must be non-negative.")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    forces_z = torch.abs(forces[..., 2])
    forces_xy = torch.linalg.norm(forces[..., :2], dim=2)
    penalty = torch.any(forces_xy > force_ratio * forces_z, dim=1).float()
    upright_gate = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * upright_gate


def stair_edge_drop_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_air_time: float = 0.02,
    min_contact_time: float = 0.005,
    settling_grace_time: float = 0.04,
    stable_contact_time: float = 0.10,
    downward_speed_threshold: float = 0.05,
    command_threshold: float = 0.10,
    lateral_command_threshold: float = 0.05,
    terrain_type_start: float = 0.0,
    terrain_type_end: float | None = None,
) -> torch.Tensor:
    """Penalize unstable contacts at stair edges."""
    if min_air_time < 0.0:
        raise ValueError("min_air_time must be non-negative.")
    if min_contact_time < 0.0:
        raise ValueError("min_contact_time must be non-negative.")
    if not min_contact_time <= settling_grace_time < stable_contact_time:
        raise ValueError(
            "settling_grace_time must be at least min_contact_time and below stable_contact_time."
        )
    if downward_speed_threshold < 0.0:
        raise ValueError("downward_speed_threshold must be non-negative.")
    if terrain_type_end is not None and not terrain_type_start < terrain_type_end < 1.0:
        raise ValueError("terrain_type_end must be greater than terrain_type_start and below 1.")

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    first_air = sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    current_contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    foot_vertical_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    falling = foot_vertical_velocity < -downward_speed_threshold
    # A contact is unstable if the foot leaves again before settling.
    short_contact_drop = (
        first_air
        & (last_air_time > min_air_time)
        & (last_contact_time > min_contact_time)
        & (last_contact_time < stable_contact_time + env.physics_dt)
        & falling
    )
    edge_slide = (
        (last_air_time > min_air_time)
        & (current_contact_time > settling_grace_time)
        & (current_contact_time < stable_contact_time + env.physics_dt)
        & falling
    )
    false_touchdown = short_contact_drop | edge_slide
    penalty = torch.sum(false_touchdown.float(), dim=1)

    command = env.command_manager.get_command(command_name)
    pure_x = (torch.abs(command[:, 0]) > command_threshold) & (
        torch.abs(command[:, 1]) < lateral_command_threshold
    )
    terrain_gate = _terrain_type_mask(env, terrain_type_start)
    if terrain_type_end is not None:
        terrain_gate &= ~_terrain_type_mask(env, terrain_type_end)

    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * pure_x * terrain_gate * upright_gate


def stair_descent_touchdown_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    min_air_time: float = 0.02,
    command_name: str = "base_velocity",
    command_threshold: float = 0.10,
    lateral_command_threshold: float = 0.05,
    terrain_type_start: float = 0.70,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize hard touchdowns while descending stairs."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if min_air_time < 0.0:
        raise ValueError("min_air_time must be non-negative.")
    if command_threshold < 0.0 or lateral_command_threshold < 0.0:
        raise ValueError("command thresholds must be non-negative.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    touchdown = first_contact & (last_air_time > min_air_time)

    forces_z = torch.clamp(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2],
        min=0.0,
    )
    peak_force_z = torch.amax(forces_z, dim=1)
    excess_force = torch.clamp(peak_force_z - threshold, min=0.0)
    penalty = torch.sum(excess_force * touchdown.float(), dim=1)

    command = env.command_manager.get_command(command_name)
    rolling_x = (torch.abs(command[:, 0]) > command_threshold) & (
        torch.abs(command[:, 1]) < lateral_command_threshold
    )
    gate = rolling_x & _terrain_type_mask(env, terrain_type_start)
    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * gate.float() * upright_gate


def body_obstacle_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_bottom_offset: float,
    safe_clearance: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize terrain approaching a rigid body footprint from below."""
    if body_bottom_offset <= 0.0 or std <= 0.0:
        raise ValueError("body_bottom_offset and std must be positive.")
    if safe_clearance < 0.0:
        raise ValueError("safe_clearance must be non-negative.")

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raw_hits = sensor.data.ray_hits_w
    valid = torch.isfinite(raw_hits).all(dim=(1, 2))
    valid &= (torch.abs(raw_hits) < 1.0e6).all(dim=(1, 2))
    valid &= torch.isfinite(asset.data.root_pos_w).all(dim=1)
    valid &= torch.isfinite(asset.data.root_quat_w).all(dim=1)

    hits = torch.where(valid[:, None, None], raw_hits, torch.zeros_like(raw_hits))
    root_pos_w = torch.where(valid[:, None], asset.data.root_pos_w, torch.zeros_like(asset.data.root_pos_w))
    root_quat_w = torch.where(valid[:, None], asset.data.root_quat_w, torch.zeros_like(asset.data.root_quat_w))
    root_quat_w[:, 0] = torch.where(valid, root_quat_w[:, 0], torch.ones_like(root_quat_w[:, 0]))

    local_bottom = torch.zeros_like(root_pos_w)
    local_bottom[:, 2] = -body_bottom_offset
    body_bottom_center_w = root_pos_w + math_utils.quat_apply(root_quat_w, local_bottom)
    body_up = torch.zeros_like(root_pos_w)
    body_up[:, 2] = 1.0
    body_normal_w = math_utils.quat_apply(root_quat_w, body_up)
    normal_z = torch.clamp(body_normal_w[:, 2], min=0.2)
    ray_offset_xy = hits[..., :2] - body_bottom_center_w[:, None, :2]
    # Project the tilted body underside onto every terrain ray.
    body_bottom_z = (
        body_bottom_center_w[:, None, 2]
        - (body_normal_w[:, None, 0] * ray_offset_xy[..., 0] + body_normal_w[:, None, 1] * ray_offset_xy[..., 1])
        / normal_z[:, None]
    )

    minimum_clearance = torch.amin(body_bottom_z - hits[..., 2], dim=1)
    violation = torch.clamp(safe_clearance - minimum_clearance, min=0.0)
    penalty = 1.0 - torch.exp(-torch.square(violation / std))
    return torch.where(valid, penalty, torch.zeros_like(penalty))


def _update_debounced_air_time(
    carried_air_time: torch.Tensor,
    first_contact: torch.Tensor,
    last_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    current_air_time: torch.Tensor,
    landing_confirm_time: float,
    max_tracked_air_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carry air time across brief contacts until a stable landing is confirmed."""
    if landing_confirm_time <= 0.0:
        raise ValueError("landing_confirm_time must be positive.")
    if max_tracked_air_time <= 0.0:
        raise ValueError("max_tracked_air_time must be positive.")

    carried_candidate = carried_air_time + first_contact.float() * last_air_time
    stable_contact = current_contact_time >= landing_confirm_time
    next_carried = torch.where(stable_contact, torch.zeros_like(carried_candidate), carried_candidate)
    next_carried = torch.clamp(next_carried, max=max_tracked_air_time)
    effective_air_time = torch.clamp(next_carried + current_air_time, max=max_tracked_air_time)
    unconfirmed_air_phase = effective_air_time > 0.0
    return next_carried, effective_air_time, unconfirmed_air_phase


def height_foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    target_height: float = 0.08,
    std: float = 0.04,
    wheel_radius: float = 0.09,
    command_threshold: float = 0.05,
    max_air_time: float = 0.4,
    min_contact: int = 2,
    lift_penalty_scale: float = 1.0,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_target_height: float | None = None,
    hover_penalty_scale: float = 0.0,
    target_centered: bool = False,
    plane_residual_gate_std: float | None = None,
    upright_gate: bool = True,
    landing_confirm_time: float = 0.0,
) -> torch.Tensor:
    """Shape wheel clearance with terrain and landing confirmation gates."""
    if std <= 0.0 or max_air_time <= 0.0:
        raise ValueError("std and max_air_time must be positive.")
    if landing_confirm_time < 0.0:
        raise ValueError("landing_confirm_time must be non-negative.")
    if hover_penalty_scale < 0.0 or lift_penalty_scale < 0.0:
        raise ValueError("clearance penalty scales must be non-negative.")
    if target_height < 0.0 or (yaw_target_height is not None and yaw_target_height < 0.0):
        raise ValueError("clearance targets must be non-negative.")
    if plane_residual_gate_std is not None and plane_residual_gate_std <= 0.0:
        raise ValueError("plane_residual_gate_std must be positive.")

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    in_air = ~in_contact
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    planar_gate = torch.ones(env.num_envs, device=env.device)
    if terrain_sensor_cfg is not None and terrain_sensor_cfg.name in env.scene.sensors:
        terrain_sensor: RayCaster = env.scene[terrain_sensor_cfg.name]
        ray_heights = terrain_sensor.data.ray_hits_w[..., 2]
        finite = torch.isfinite(ray_heights).all(dim=1)
        finite &= (torch.abs(ray_heights) < 1.0e6).all(dim=1)
        fallback_z = env.scene.env_origins[:, 2:3]
        sanitized = torch.where(finite[:, None], ray_heights, fallback_z)
        terrain_z = torch.mean(sanitized, dim=1, keepdim=True)
        terrain_z = torch.where(finite[:, None], terrain_z, fallback_z)
        if plane_residual_gate_std is not None:
            _, residual, valid = _fit_height_scan_plane_yaw(asset, terrain_sensor)
            residual_gate = torch.exp(-torch.square(residual / plane_residual_gate_std))
            planar_gate = torch.where(valid, residual_gate, torch.ones_like(residual_gate))
    else:
        terrain_z = torch.zeros_like(foot_pos_z)

    clearance = foot_pos_z - terrain_z - wheel_radius
    command = env.command_manager.get_command(command_name)
    lateral_or_rot = torch.norm(command[:, 1:3], dim=1)
    active = (lateral_or_rot > command_threshold).unsqueeze(-1)

    clearance_target = torch.full_like(clearance, target_height)
    if yaw_target_height is not None:
        yaw_active = (torch.abs(command[:, 2]) > command_threshold).unsqueeze(-1)
        clearance_target = torch.where(
            yaw_active,
            torch.full_like(clearance_target, yaw_target_height),
            clearance_target,
        )
    height_error = clearance_target - clearance
    if not target_centered:
        height_error = torch.clamp(height_error, min=0.0)
    height_reward = torch.exp(-height_error.pow(2) / std**2)

    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    active_air_phase = in_air
    if landing_confirm_time > 0.0:
        # Carry swing time across brief contacts caused by stair edges.
        carried_air_time = getattr(env, "_height_foot_clearance_carried_air_time", None)
        if carried_air_time is None or carried_air_time.shape != air_time.shape:
            carried_air_time = torch.zeros_like(air_time)
        just_reset = (env.episode_length_buf <= 1).unsqueeze(-1)
        carried_air_time = torch.where(just_reset, torch.zeros_like(carried_air_time), carried_air_time)
        carried_air_time, air_time, active_air_phase = _update_debounced_air_time(
            carried_air_time,
            sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids],
            sensor.data.last_air_time[:, sensor_cfg.body_ids],
            sensor.data.current_contact_time[:, sensor_cfg.body_ids],
            air_time,
            landing_confirm_time,
            2.0 * max_air_time,
        )
        env._height_foot_clearance_carried_air_time = carried_air_time

    swing_decay = torch.exp(-(air_time / max_air_time).pow(2))
    hover_excess = torch.clamp((air_time - max_air_time) / max_air_time, min=0.0, max=1.0)
    hover_penalty = hover_penalty_scale * torch.square(hover_excess)
    enough_support = (in_contact.float().sum(dim=-1, keepdim=True) >= min_contact).float()
    positive_active_term = in_air.float() * height_reward * swing_decay * enough_support
    active_term = positive_active_term - active_air_phase.float() * hover_penalty
    inactive_term = -in_air.float() * lift_penalty_scale * torch.clamp(clearance, min=0.0).pow(2)

    reward = torch.where(active, active_term, inactive_term)
    if upright_gate:
        reward *= (torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7).unsqueeze(-1)
    return reward.sum(dim=-1) * planar_gate


def zero_command_base_motion_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    yaw_scale: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize chassis planar and yaw speed for a zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    standing = torch.linalg.norm(command, dim=1) < command_threshold
    planar_speed = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
    yaw_speed = torch.abs(asset.data.root_com_ang_vel_b[:, 2])
    return standing * (planar_speed + yaw_scale * yaw_speed)


def zero_command_position_l1(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return horizontal distance from the active stop anchor."""
    term = env.command_manager.get_term(command_name)
    delta = term.robot.data.root_pos_w[:, :2] - term.hold_position_w
    distance = torch.linalg.vector_norm(delta, dim=1)
    return torch.where(term.hold_active, distance, 0.0)
