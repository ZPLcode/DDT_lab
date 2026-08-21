# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateless reward and safety terms for platform ascent and descent."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .platform_utils import (
    PlatformTraversalSettings,
    _platform_ascent_overshoot,
    _platform_axle_on_destination,
    _platform_descent_stall_cost,
    _platform_landing_force_penalty,
    _platform_mean_scan_height,
    _platform_phase_masks,
    _platform_resolve_trailing_state,
    _platform_sample_heights,
    _platform_step_direction,
    _platform_toward_target_progress,
    _platform_travel_order,
    _platform_wall_contact_scores,
    _squared_excess,
)
from .rewards import _terrain_type_mask, feet_stumble

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _fit_height_scan_plane_yaw(
    asset: RigidObject,
    sensor: RayCaster,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a plane to a yaw-aligned height scan."""
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
    hits_yaw = quat_apply_inverse(yaw_expanded, relative_hits.reshape(-1, 3)).reshape(relative_hits.shape)

    centered = hits_yaw - hits_yaw.mean(dim=1, keepdim=True)
    x, y, z = centered.unbind(dim=2)
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
    """Return a smooth gate that fades on slopes and discontinuities."""
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


def platform_discontinuity_gated_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    discontinuity_height: float,
    discontinuity_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Penalize vertical base speed except while crossing a ledge."""
    if discontinuity_height < 0.0:
        raise ValueError("discontinuity_height must be non-negative.")
    if discontinuity_std <= 0.0:
        raise ValueError("discontinuity_std must be positive.")

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raw_hits = sensor.data.ray_hits_w
    valid = torch.isfinite(raw_hits).all(dim=(1, 2))
    valid &= (torch.abs(raw_hits) < 1.0e6).all(dim=(1, 2))
    safe_heights = torch.where(valid[:, None], raw_hits[..., 2], torch.zeros_like(raw_hits[..., 2]))
    height_spread = torch.amax(safe_heights, dim=1) - torch.amin(safe_heights, dim=1)
    excess = torch.clamp(height_spread - discontinuity_height, min=0.0)
    smooth_gate = torch.exp(-torch.square(excess / discontinuity_std))
    smooth_gate = torch.where(valid, smooth_gate, torch.ones_like(smooth_gate))
    penalty = torch.square(asset.data.root_lin_vel_b[:, 2]) * smooth_gate
    if upright_gate:
        penalty *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


def platform_terrain_gated_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    plane_residual_gate_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Track terrain-relative base height only on a locally smooth surface."""
    if plane_residual_gate_std <= 0.0:
        raise ValueError("plane_residual_gate_std must be positive.")
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raw_hits = sensor.data.ray_hits_w
    valid_hits = torch.isfinite(raw_hits).all(dim=(1, 2))
    valid_hits &= (torch.abs(raw_hits) < 1.0e6).all(dim=(1, 2))
    safe_height = torch.where(valid_hits[:, None], raw_hits[..., 2], torch.zeros_like(raw_hits[..., 2]))
    terrain_height = torch.mean(safe_height, dim=1)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - terrain_height - target_height)
    _, residual_rms, plane_valid = _fit_height_scan_plane_yaw(asset, sensor)
    residual_gate = torch.exp(-torch.square(residual_rms / plane_residual_gate_std))
    cost = torch.where(
        valid_hits & plane_valid,
        height_error * residual_gate,
        torch.zeros_like(height_error),
    )
    if upright_gate:
        cost *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return cost


def platform_terrain_or_world_orientation_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    roughness_std: float,
    discontinuity_height: float | None = None,
    discontinuity_std: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Follow smooth slopes, remain world-level on rough ground, and fade at ledges."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    terrain_normal_w, residual_rms, valid = _fit_height_scan_plane_yaw(asset, sensor)

    body_up = torch.zeros_like(terrain_normal_w)
    body_up[:, 2] = 1.0
    body_up_w = math_utils.quat_apply(asset.data.root_quat_w, body_up)
    terrain_cosine = torch.clamp(torch.sum(body_up_w * terrain_normal_w, dim=1), -1.0, 1.0)
    terrain_error = 2.0 * (1.0 - terrain_cosine)
    world_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    smooth_plane_gate = torch.exp(-torch.square(residual_rms / roughness_std))
    rough_world_gate = torch.ones_like(smooth_plane_gate)
    if discontinuity_height is not None:
        raw_heights = sensor.data.ray_hits_w[..., 2]
        heights = torch.where(valid[:, None], raw_heights, torch.zeros_like(raw_heights))
        height_spread = torch.amax(heights, dim=1) - torch.amin(heights, dim=1)
        excess = torch.clamp(height_spread - discontinuity_height, min=0.0)
        rough_world_gate = torch.exp(-torch.square(excess / discontinuity_std))

    cost = smooth_plane_gate * terrain_error + (1.0 - smooth_plane_gate) * rough_world_gate * world_error
    return torch.where(valid, cost, world_error)


def platform_flat_yaw_hip_pos_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    yaw_threshold: float = 0.10,
    tolerance: float = 0.05,
    terrain_slope_gate_std: float = 0.05,
    plane_residual_gate_std: float = 0.015,
) -> torch.Tensor:
    """Penalize hip deflection during flat-ground yaw commands."""
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


def platform_wheel_thigh_x_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    thigh_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    terrain_slope_gate_std: float | None = None,
    plane_residual_gate_std: float | None = None,
    upright_gate: bool = False,
) -> torch.Tensor:
    """Align each wheel axle below its corresponding thigh-pitch joint."""
    asset: Articulation = env.scene[thigh_cfg.name]
    thigh_pos_w = asset.data.body_link_pos_w[:, thigh_cfg.body_ids, :]
    wheel_pos_w = asset.data.body_link_pos_w[:, wheel_cfg.body_ids, :]
    wheel_from_thigh_w = wheel_pos_w - thigh_pos_w
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
    if upright_gate:
        penalty *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty


def platform_run_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    lateral_or_rot_threshold: float | None = None,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    plane_residual_gate_std: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize leg-joint deviation while translating on locally planar terrain."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    command = env.command_manager.get_command(command_name)
    active = torch.norm(command[:, :2], dim=1) > command_threshold
    if lateral_or_rot_threshold is not None:
        active &= torch.norm(command[:, 1:3], dim=1) < lateral_or_rot_threshold
    penalty *= active
    if terrain_sensor_cfg is not None and plane_residual_gate_std is not None:
        sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        _, residual_rms, valid = _fit_height_scan_plane_yaw(asset, sensor)
        planar_gate = torch.exp(-torch.square(residual_rms / plane_residual_gate_std))
        penalty *= torch.where(valid, planar_gate, torch.zeros_like(planar_gate))
    return penalty


def platform_foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    target_height: float = 0.08,
    yaw_target_height: float | None = None,
    std: float = 0.04,
    wheel_radius: float = 0.09,
    command_threshold: float = 0.10,
    max_air_time: float = 0.4,
    hover_penalty_scale: float = 0.0,
    min_contact: int = 2,
    lift_penalty_scale: float = 1.0,
    target_centered: bool = False,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    plane_residual_gate_std: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shape swing clearance while suppressing unwanted lift during rolling."""
    if max_air_time <= 0.0:
        raise ValueError("max_air_time must be positive.")
    if hover_penalty_scale < 0.0:
        raise ValueError("hover_penalty_scale must be non-negative.")
    if target_height < 0.0 or (yaw_target_height is not None and yaw_target_height < 0.0):
        raise ValueError("clearance targets must be non-negative.")

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    in_air = ~in_contact
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    planar_gate = torch.ones(env.num_envs, device=env.device)
    if terrain_sensor_cfg is not None and terrain_sensor_cfg.name in env.scene.sensors:
        terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        terrain_z, scan_valid = _platform_mean_scan_height(terrain_sensor.data.ray_hits_w)
        planar_gate = scan_valid.float()
        if plane_residual_gate_std is not None:
            if plane_residual_gate_std <= 0.0:
                raise ValueError("plane_residual_gate_std must be positive.")
            _, plane_residual, plane_valid = _fit_height_scan_plane_yaw(asset, terrain_sensor)
            fitted_gate = torch.exp(-torch.square(plane_residual / plane_residual_gate_std))
            planar_gate = torch.where(
                scan_valid & plane_valid,
                fitted_gate,
                torch.zeros_like(fitted_gate),
            )
    else:
        terrain_z = torch.zeros_like(foot_pos_z)

    clearance = foot_pos_z - terrain_z - wheel_radius
    command = env.command_manager.get_command(command_name)
    active = (torch.norm(command[:, 1:3], dim=1) > command_threshold).unsqueeze(-1)
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
    swing_decay = torch.exp(-(air_time / max_air_time).pow(2))
    hover_excess = torch.clamp((air_time - max_air_time) / max_air_time, min=0.0, max=1.0)
    hover_penalty = hover_penalty_scale * torch.square(hover_excess)
    has_other_contact = (in_contact.float().sum(dim=-1, keepdim=True) >= min_contact).float()
    active_term = height_reward * swing_decay * has_other_contact - hover_penalty
    inactive_term = -lift_penalty_scale * torch.clamp(clearance, min=0.0).pow(2)
    reward = in_air.float() * torch.where(active, active_term, inactive_term)
    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward.sum(dim=-1) * planar_gate * upright_gate


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
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return _platform_landing_force_penalty(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids],
        threshold,
    )


def _platform_contact_sequence_state(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg,
    wheel_sensor_cfg: SceneEntityCfg,
    wheel_asset_cfg: SceneEntityCfg,
    command_name: str,
    terrain_type_start: float,
    settings: PlatformTraversalSettings,
) -> dict[str, torch.Tensor | Articulation]:
    """Measure current leading/trailing geometry in commanded-X order."""
    settings.validate()
    asset: Articulation = env.scene[wheel_asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[wheel_sensor_cfg.name]
    terrain_sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]

    wheel_pos_w = asset.data.body_pos_w[:, wheel_asset_cfg.body_ids]
    wheel_vel_z = asset.data.body_lin_vel_w[:, wheel_asset_cfg.body_ids, 2]
    wheel_forces = contact_sensor.data.net_forces_w[:, wheel_sensor_cfg.body_ids]
    wheel_force_history = contact_sensor.data.net_forces_w_history[:, :, wheel_sensor_cfg.body_ids]
    wheel_air_time = contact_sensor.data.current_air_time[:, wheel_sensor_cfg.body_ids]

    command_x = env.command_manager.get_command(command_name)[:, 0]
    motion_active = torch.abs(command_x) > settings.command_threshold
    forward_travel = command_x >= 0.0

    leading_pos_w, trailing_pos_w = _platform_travel_order(wheel_pos_w, forward_travel)
    leading_vel_z, trailing_vel_z = _platform_travel_order(wheel_vel_z, forward_travel)
    leading_forces, trailing_forces = _platform_travel_order(wheel_forces, forward_travel)
    _, trailing_air_time = _platform_travel_order(wheel_air_time, forward_travel)

    travel_sign = torch.where(
        motion_active,
        torch.where(forward_travel, torch.ones_like(command_x), -torch.ones_like(command_x)),
        torch.zeros_like(command_x),
    )
    forward_b = torch.zeros(env.num_envs, 3, device=env.device)
    forward_b[:, 0] = 1.0
    travel_w = quat_apply(yaw_quat(asset.data.root_quat_w), forward_b) * travel_sign[:, None]

    leading_probe_xy = leading_pos_w[..., :2] + settings.probe_forward * travel_w[:, None, :2]
    trailing_probe_xy = trailing_pos_w[..., :2] + settings.probe_forward * travel_w[:, None, :2]
    query_xy = torch.cat(
        (leading_probe_xy, trailing_pos_w[..., :2], trailing_probe_xy),
        dim=1,
    )
    query_heights, scan_valid, scan_height_span = _platform_sample_heights(
        terrain_sensor.data.ray_hits_w,
        query_xy,
    )
    leading_next_surface_z, trailing_surface_z, trailing_next_surface_z = torch.chunk(query_heights, 3, dim=1)
    scan_spans_transition = scan_valid & (scan_height_span > settings.min_step_height)

    (
        leading_transition_detected,
        leading_is_descent,
        leading_direction,
    ) = _platform_step_direction(
        leading_next_surface_z - trailing_surface_z,
        settings.min_step_height,
    )
    trailing_transition_detected, _, trailing_direction = _platform_step_direction(
        trailing_next_surface_z - trailing_surface_z,
        settings.min_step_height,
    )

    leading_fz = torch.clamp(leading_forces[..., 2], min=0.0)
    trailing_fz = torch.clamp(trailing_forces[..., 2], min=0.0)
    history_steps = min(settings.support_history_steps, wheel_force_history.shape[1])
    force_history = wheel_force_history[:, :history_steps]
    history_fz = torch.clamp(force_history[..., 2], min=0.0)
    history_fxy = torch.linalg.norm(force_history[..., :2], dim=-1)
    support_history = (history_fz > settings.support_force_threshold) & (
        history_fxy < settings.top_horizontal_force_ratio * history_fz
    )
    required_history_steps = math.ceil(0.8 * history_steps)
    stable_support = support_history[:, 0] & (torch.sum(support_history, dim=1) >= required_history_steps)
    leading_stable_support, trailing_stable_support = _platform_travel_order(stable_support, forward_travel)

    leading_height_delta = leading_pos_w[..., 2] - (leading_next_surface_z + settings.wheel_radius)
    leading_on_target = scan_spans_transition & torch.all(
        (torch.abs(leading_height_delta) < settings.surface_tolerance) & leading_stable_support,
        dim=1,
    )

    trailing_height_delta = trailing_pos_w[..., 2] - (trailing_next_surface_z + settings.wheel_radius)
    trailing_target_surface_error = torch.abs(trailing_surface_z - trailing_next_surface_z)
    trailing_on_target = _platform_axle_on_destination(
        trailing_pos_w[..., 2],
        trailing_surface_z,
        trailing_next_surface_z,
        leading_next_surface_z,
        trailing_stable_support,
        settings.wheel_radius,
        settings.surface_tolerance,
    )

    leading_completed = leading_on_target
    (
        trailing_direction,
        trailing_is_descent,
        trailing_exposed_wall,
    ) = _platform_resolve_trailing_state(
        trailing_direction,
        trailing_transition_detected,
        scan_spans_transition,
        leading_transition_detected,
        leading_is_descent,
        trailing_height_delta,
        trailing_air_time,
        trailing_target_surface_error,
        settings.surface_tolerance,
    )

    travel_xy = travel_w[:, None, :2]
    leading_ascent_wall_score, _ = _platform_wall_contact_scores(
        leading_forces[..., :2],
        travel_xy,
        settings.wall_force_threshold,
        torch.zeros_like(leading_fz, dtype=torch.bool),
    )
    trailing_ascent_wall_score, trailing_descent_wall_score = _platform_wall_contact_scores(
        trailing_forces[..., :2],
        travel_xy,
        settings.wall_force_threshold,
        trailing_exposed_wall,
    )
    trailing_descent_wall_contact = torch.mean(
        trailing_descent_wall_score * (trailing_direction < 0.0).float(),
        dim=1,
    )

    motion_on_platform = _terrain_type_mask(env, terrain_type_start) & motion_active
    active = motion_on_platform & scan_valid
    phases = _platform_phase_masks(
        active,
        active & scan_spans_transition,
        leading_transition_detected,
        leading_completed,
        trailing_on_target,
        leading_is_descent,
        trailing_is_descent,
    )
    (
        sequence_phase,
        leading_phase,
        trailing_phase,
        leading_descent_phase,
        trailing_descent_phase,
    ) = phases

    return {
        "asset": asset,
        "motion_on_platform": motion_on_platform,
        "sequence_phase": sequence_phase,
        "leading_phase": leading_phase,
        "trailing_phase": trailing_phase,
        "leading_descent_phase": leading_descent_phase,
        "trailing_descent_phase": trailing_descent_phase,
        "trailing_supported": torch.all(trailing_fz > settings.support_force_threshold, dim=1),
        "support_count": torch.sum(
            torch.clamp(wheel_forces[..., 2], min=0.0) > settings.support_force_threshold,
            dim=1,
        ),
        "leading_direction": leading_direction,
        "trailing_direction": trailing_direction,
        "leading_height_delta": leading_height_delta,
        "trailing_height_delta": trailing_height_delta,
        "leading_vel_z": leading_vel_z,
        "trailing_vel_z": trailing_vel_z,
        "leading_height_spread": torch.abs(leading_pos_w[:, 0, 2] - leading_pos_w[:, 1, 2]),
        "trailing_air_time": trailing_air_time,
        "leading_ascent_wall_score": leading_ascent_wall_score,
        "trailing_ascent_wall_score": trailing_ascent_wall_score,
        "trailing_descent_wall_contact": trailing_descent_wall_contact,
        "leading_needs_progress": leading_direction * leading_height_delta < 0.0,
        "trailing_needs_progress": trailing_direction * trailing_height_delta < 0.0,
    }


def platform_traversal_reward(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg,
    wheel_sensor_cfg: SceneEntityCfg,
    wheel_asset_cfg: SceneEntityCfg,
    command_name: str,
    terrain_type_start: float,
    settings: PlatformTraversalSettings,
) -> torch.Tensor:
    """Reward supported leading-first ascent or descent."""
    state = _platform_contact_sequence_state(
        env,
        terrain_sensor_cfg,
        wheel_sensor_cfg,
        wheel_asset_cfg,
        command_name,
        terrain_type_start,
        settings,
    )
    leading_progress = _platform_toward_target_progress(
        state["leading_vel_z"],
        state["leading_direction"],
        state["leading_height_delta"],
        state["leading_ascent_wall_score"],
        settings.leading_ascent_speed,
        settings.leading_descent_speed,
        settings.ascent_slowdown_distance,
        settings.ascent_min_speed,
        settings.speed_quadratic_blend,
        settings.leading_ascent_contact_floor,
        settings.backslide_penalty_scale,
    )
    trailing_progress = _platform_toward_target_progress(
        state["trailing_vel_z"],
        state["trailing_direction"],
        state["trailing_height_delta"],
        state["trailing_ascent_wall_score"],
        settings.trailing_ascent_speed,
        settings.trailing_descent_speed,
        settings.ascent_slowdown_distance,
        settings.ascent_min_speed,
        settings.speed_quadratic_blend,
        settings.trailing_ascent_contact_floor,
        settings.backslide_penalty_scale,
    )

    leading_phase = state["leading_phase"].float()
    leading_overshoot = _platform_ascent_overshoot(
        state["leading_height_delta"],
        state["leading_direction"] > 0.0,
        settings.ascent_overshoot_margin,
        settings.ascent_overshoot_ramp,
    )
    leading_reward = leading_phase * (
        state["trailing_supported"].float()
        * torch.mean(
            leading_progress * state["leading_needs_progress"].float(),
            dim=1,
        )
        - settings.ascent_overshoot_penalty_scale * leading_overshoot
    )
    trailing_phase = state["trailing_phase"].float()
    trailing_overshoot = _platform_ascent_overshoot(
        state["trailing_height_delta"],
        (state["leading_direction"] > 0.0) | (state["trailing_direction"] > 0.0),
        settings.ascent_overshoot_margin,
        settings.ascent_overshoot_ramp,
    )
    trailing_reward = trailing_phase * (
        torch.mean(
            trailing_progress * state["trailing_needs_progress"].float(),
            dim=1,
        )
        - settings.ascent_overshoot_penalty_scale * trailing_overshoot
    )
    trailing_penalty = state["trailing_descent_phase"].float() * (
        settings.trailing_descent_wall_penalty_scale * state["trailing_descent_wall_contact"]
        + settings.trailing_descent_phase_penalty
    )
    return leading_reward + trailing_reward - trailing_penalty


def platform_safety_cost(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg,
    wheel_sensor_cfg: SceneEntityCfg,
    wheel_asset_cfg: SceneEntityCfg,
    command_name: str,
    terrain_type_start: float,
    settings: PlatformTraversalSettings,
) -> torch.Tensor:
    """Return the maximum active platform safety violation."""
    state = _platform_contact_sequence_state(
        env,
        terrain_sensor_cfg,
        wheel_sensor_cfg,
        wheel_asset_cfg,
        command_name,
        terrain_type_start,
        settings,
    )
    asset: Articulation = state["asset"]

    sequence_cost = state["sequence_phase"].float() * torch.clamp(
        (state["trailing_air_time"] - settings.air_time_grace) / settings.air_time_ramp,
        min=0.0,
        max=1.0,
    ).mean(dim=1)

    support_deficit = torch.clamp(
        (settings.min_support_count - state["support_count"]).float() / float(settings.min_support_count),
        min=0.0,
        max=1.0,
    )
    root_vertical_speed = torch.clamp(
        torch.abs(asset.data.root_lin_vel_w[:, 2]) / settings.vertical_speed_scale,
        max=1.0,
    )
    ballistic_cost = state["motion_on_platform"].float() * support_deficit * torch.square(root_vertical_speed)

    leading_ascent_phase = state["leading_phase"] & ~state["leading_descent_phase"]
    leading_axle_sync_cost = leading_ascent_phase.float() * _squared_excess(
        state["leading_height_spread"],
        settings.leading_axle_sync_tolerance,
        settings.leading_axle_sync_ramp,
    )

    leading_fast_cost = torch.mean(
        _squared_excess(
            torch.clamp(-state["leading_vel_z"], min=0.0),
            settings.max_leading_descent_speed,
            settings.max_leading_descent_speed,
        ),
        dim=1,
    )
    trailing_fast_cost = torch.mean(
        _squared_excess(
            torch.clamp(-state["trailing_vel_z"], min=0.0),
            settings.max_trailing_descent_speed,
            settings.max_trailing_descent_speed,
        ),
        dim=1,
    )
    fast_descent_cost = (
        state["leading_descent_phase"].float() * leading_fast_cost
        + state["trailing_descent_phase"].float() * trailing_fast_cost
    )

    trailing_hang_cost = state["trailing_descent_phase"].float() * (
        _platform_descent_stall_cost(
            state["trailing_air_time"],
            state["trailing_vel_z"],
            settings.trailing_descent_air_time_grace,
            settings.trailing_descent_air_time_ramp,
            settings.min_trailing_descent_progress_speed,
        )
    )
    tilt_angle = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], min=-1.0, max=1.0))
    tilt_cost = _squared_excess(
        tilt_angle,
        settings.max_descent_tilt,
        math.pi / 2 - settings.max_descent_tilt,
    )
    angular_cost = _squared_excess(
        torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=1),
        settings.max_descent_ang_vel,
        settings.max_descent_ang_vel,
    )
    descent_phase = state["leading_descent_phase"] | state["trailing_descent_phase"]
    stability_cost = descent_phase.float() * torch.maximum(tilt_cost, angular_cost)

    base_cost = torch.maximum(
        torch.maximum(sequence_cost, ballistic_cost),
        torch.maximum(fast_descent_cost, stability_cost),
    )
    base_cost = torch.maximum(base_cost, leading_axle_sync_cost)
    return torch.maximum(base_cost, trailing_hang_cost)
