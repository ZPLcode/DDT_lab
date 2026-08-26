# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and safety terms for platform ascent and descent."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, yaw_quat

from .platform_utils import (
    PlatformTraversalSettings,
    _platform_ascent_overshoot,
    _platform_axle_on_destination,
    _platform_descent_stall_cost,
    _platform_landing_force_penalty,
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


_PLATFORM_REWARD_TERM = "platform_traversal"


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
    commanded_forward = command_x >= 0.0
    just_reset = env.episode_length_buf <= 1

    descent_travel_sign = getattr(env, "_platform_trailing_descent_travel_sign", None)
    if descent_travel_sign is None:
        descent_travel_sign = torch.zeros_like(command_x)
    descent_travel_sign = torch.where(just_reset, torch.zeros_like(descent_travel_sign), descent_travel_sign)
    descent_latch = descent_travel_sign != 0.0
    forward_travel = torch.where(descent_latch, descent_travel_sign > 0.0, commanded_forward)

    leading_pos_w, trailing_pos_w = _platform_travel_order(wheel_pos_w, forward_travel)
    leading_vel_z, trailing_vel_z = _platform_travel_order(wheel_vel_z, forward_travel)
    leading_forces, trailing_forces = _platform_travel_order(wheel_forces, forward_travel)
    _, trailing_air_time = _platform_travel_order(wheel_air_time, forward_travel)

    travel_sign = torch.where(
        motion_active | descent_latch,
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

    platform_column = _terrain_type_mask(env, terrain_type_start)
    descent_handoff = (
        platform_column
        & motion_active
        & leading_is_descent
        & leading_on_target
        & ~trailing_on_target
    )
    handoff_travel_sign = torch.where(commanded_forward, torch.ones_like(command_x), -torch.ones_like(command_x))
    descent_travel_sign = torch.where(descent_handoff, handoff_travel_sign, descent_travel_sign)
    descent_travel_sign = torch.where(
        trailing_on_target | just_reset,
        torch.zeros_like(descent_travel_sign),
        descent_travel_sign,
    )
    env._platform_trailing_descent_travel_sign = descent_travel_sign
    descent_latch = descent_travel_sign != 0.0

    leading_completed = leading_completed | descent_latch
    trailing_is_descent = trailing_is_descent | descent_latch
    trailing_direction = torch.where(
        descent_latch[:, None] & (trailing_height_delta > 0.0),
        -torch.ones_like(trailing_direction),
        trailing_direction,
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

    motion_on_platform = platform_column & motion_active
    active = platform_column & (motion_active | descent_latch) & scan_valid
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
        trailing_wall_phase,
    ) = phases
    trailing_descent_phase = trailing_descent_phase & descent_latch
    trailing_wall_phase = trailing_wall_phase & descent_latch

    return {
        "asset": asset,
        "motion_on_platform": motion_on_platform,
        "sequence_phase": sequence_phase,
        "leading_phase": leading_phase,
        "trailing_phase": trailing_phase,
        "leading_descent_phase": leading_descent_phase,
        "trailing_descent_phase": trailing_descent_phase,
        "trailing_wall_phase": trailing_wall_phase,
        "trailing_supported": torch.all(trailing_fz > settings.support_force_threshold, dim=1),
        "support_count": torch.sum(
            torch.clamp(wheel_forces[..., 2], min=0.0) > settings.support_force_threshold,
            dim=1,
        ),
        "travel_sign": travel_sign,
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


def _record_platform_reward_components(
    env: ManagerBasedRLEnv,
    components: dict[str, torch.Tensor],
) -> None:
    """Accumulate platform phase rewards for training diagnostics."""
    reward_manager = env.reward_manager
    weight = reward_manager.get_term_cfg(_PLATFORM_REWARD_TERM).weight
    scale = float(weight) * float(env.step_dt)
    template = reward_manager._episode_sums[_PLATFORM_REWARD_TERM]

    for name, component in components.items():
        key = f"{_PLATFORM_REWARD_TERM}/{name}"
        buffer = reward_manager._episode_sums.setdefault(key, torch.zeros_like(template))
        buffer.add_(component.detach().to(buffer), alpha=scale)


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
    leading_descent = state["leading_descent_phase"].float()
    trailing_descent = state["trailing_descent_phase"].float()
    _record_platform_reward_components(
        env,
        {
            "ascent_leading": leading_reward * (1.0 - leading_descent),
            "ascent_trailing": trailing_reward * (1.0 - trailing_descent),
            "descent": (
                leading_reward * leading_descent
                + trailing_reward * trailing_descent
                - trailing_penalty
            ),
        },
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
    contact_sensor: ContactSensor = env.scene.sensors[wheel_sensor_cfg.name]
    base_body_ids = getattr(env, "_platform_base_contact_body_ids", None)
    if base_body_ids is None:
        base_body_ids = contact_sensor.find_bodies("base_link")[0]
        env._platform_base_contact_body_ids = base_body_ids
    base_peak_force = torch.linalg.norm(
        contact_sensor.data.net_forces_w_history[:, :, base_body_ids], dim=-1
    ).amax(dim=(1, 2))

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
    trailing_wall_cost = state["trailing_wall_phase"].float() * torch.clamp(
        settings.trailing_descent_wall_cost_scale
        * torch.square(state["trailing_descent_wall_contact"]),
        max=1.0,
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
    leading_pitch_cost = descent_phase.float() * _squared_excess(
        torch.clamp(state["travel_sign"] * asset.data.root_ang_vel_b[:, 1], min=0.0),
        settings.max_leading_descent_pitch_rate,
        settings.max_leading_descent_pitch_rate,
    )
    base_impact_cost = descent_phase.float() * _squared_excess(
        base_peak_force,
        settings.base_impact_force_threshold,
        settings.base_impact_force_ramp,
    )

    base_cost = torch.maximum(
        torch.maximum(sequence_cost, ballistic_cost),
        torch.maximum(fast_descent_cost, stability_cost),
    )
    base_cost = torch.maximum(
        base_cost,
        torch.maximum(
            leading_axle_sync_cost,
            torch.maximum(leading_pitch_cost, base_impact_cost),
        ),
    )
    return torch.maximum(base_cost, torch.maximum(trailing_hang_cost, trailing_wall_cost))
