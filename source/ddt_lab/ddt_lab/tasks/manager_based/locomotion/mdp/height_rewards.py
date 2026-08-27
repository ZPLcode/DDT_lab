# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms shared only by commanded-height all-terrain tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse

from .rewards import _fit_height_scan_plane_yaw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    # Fall back to origin-relative height when the local scan is invalid.
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
        # Remove sharp height pressure where stair edges violate the plane model.
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
    # Footprint height spread acts as a cheap flat-ground detector.
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
    # Root-frame X is fore/aft regardless of the robot's world heading.
    num_feet = wheel_from_thigh_w.shape[1]
    root_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    wheel_from_thigh_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), wheel_from_thigh_w.reshape(-1, 3)).reshape(
        env.num_envs, num_feet, 3
    )
    penalty = torch.mean(torch.square(wheel_from_thigh_b[:, :, 0] / std), dim=1)

    if terrain_sensor_cfg is not None:
        terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        # Relax this posture prior near non-planar terrain transitions.
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
    # Small hip motion inside the dead band remains free.
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
    # Use the history peak so brief impacts between policy steps are retained.
    forces_z = torch.clamp(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2], min=0.0)
    peak_force_z = torch.amax(forces_z, dim=1)
    return torch.sum(torch.clamp(peak_force_z - threshold, min=0.0), dim=1)


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

    # Sanitize invalid rays before constructing the tilted chassis-bottom plane.
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
    # Evaluate the underside plane at every footprint ray location.
    body_bottom_z = (
        body_bottom_center_w[:, None, 2]
        - (body_normal_w[:, None, 0] * ray_offset_xy[..., 0] + body_normal_w[:, None, 1] * ray_offset_xy[..., 1])
        / normal_z[:, None]
    )

    minimum_clearance = torch.amin(body_bottom_z - hits[..., 2], dim=1)
    violation = torch.clamp(safe_clearance - minimum_clearance, min=0.0)
    # Bound each environment's penalty to [0, 1].
    penalty = 1.0 - torch.exp(-torch.square(violation / std))
    return torch.where(valid, penalty, torch.zeros_like(penalty))


def zero_command_base_motion_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    yaw_scale: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize chassis planar and yaw drift only for a zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    # Do not oppose motion requested by a non-zero command.
    standing = torch.linalg.norm(command, dim=1) < command_threshold
    planar_drift = torch.sum(torch.square(asset.data.root_com_lin_vel_b[:, :2]), dim=1)
    yaw_drift = torch.square(asset.data.root_com_ang_vel_b[:, 2])
    return standing * (planar_drift + yaw_scale * yaw_drift)


def zero_command_wheel_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel rotation only for a zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    # Measured wheel speed closes the stationary-policy spin loophole.
    standing = torch.linalg.norm(command, dim=1) < command_threshold
    wheel_speed = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return standing * torch.mean(torch.square(wheel_speed), dim=1)
