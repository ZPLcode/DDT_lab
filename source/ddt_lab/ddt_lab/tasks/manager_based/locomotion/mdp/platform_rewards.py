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
    _platform_axle_on_destination,
    _platform_descent_stall_cost,
    _platform_phase_masks,
    _platform_resolve_trailing_state,
    _platform_sample_heights,
    _platform_step_direction,
    _platform_travel_order,
    _platform_wall_contact_scores,
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
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.clamp(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2], min=0.0
    )
    peak_force_z = torch.amax(forces_z, dim=1)
    return torch.sum(torch.clamp(peak_force_z - threshold, min=0.0), dim=1)


def _platform_update_trailing_unsupported_time(
    env: ManagerBasedRLEnv,
    unsupported: torch.Tensor,
    just_reset: torch.Tensor,
) -> torch.Tensor:
    """Track time without horizontal-tread support, updating once per env step."""
    timer = getattr(env, "_platform_trailing_unsupported_time", None)
    last_step = getattr(env, "_platform_trailing_unsupported_last_step", None)
    if timer is None or last_step is None:
        timer = torch.zeros_like(unsupported, dtype=torch.float)
        last_step = torch.full_like(env.episode_length_buf, -1)

    current_step = env.episode_length_buf
    advance = current_step != last_step
    advanced_timer = torch.where(
        unsupported,
        timer + float(env.step_dt),
        torch.zeros_like(timer),
    )
    timer = torch.where(advance[:, None], advanced_timer, timer)
    timer = torch.where(just_reset[:, None], torch.zeros_like(timer), timer)
    last_step = torch.where(advance | just_reset, current_step, last_step)

    env._platform_trailing_unsupported_time = timer
    env._platform_trailing_unsupported_last_step = last_step
    return timer


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
    descent_latch = getattr(env, "_platform_trailing_descent_latch", None)
    descent_forward = getattr(env, "_platform_trailing_descent_forward", None)
    if descent_latch is None or descent_forward is None:
        descent_latch = torch.zeros_like(motion_active)
        descent_forward = commanded_forward.clone()
    descent_latch = descent_latch & ~just_reset
    forward_travel = torch.where(descent_latch, descent_forward, commanded_forward)

    leading_pos_w, trailing_pos_w = _platform_travel_order(wheel_pos_w, forward_travel)
    leading_vel_z, trailing_vel_z = _platform_travel_order(wheel_vel_z, forward_travel)
    leading_forces, trailing_forces = _platform_travel_order(wheel_forces, forward_travel)
    _, trailing_air_time = _platform_travel_order(wheel_air_time, forward_travel)

    travel_active = motion_active | descent_latch
    travel_direction = torch.where(
        forward_travel,
        torch.ones_like(command_x),
        -torch.ones_like(command_x),
    )
    travel_sign = torch.where(
        travel_active,
        travel_direction,
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
    leading_next_surface_z, trailing_surface_z, trailing_next_surface_z = torch.chunk(
        query_heights, 3, dim=1
    )
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
    support_history = (
        (history_fz > settings.support_force_threshold)
        & (history_fxy < settings.top_horizontal_force_ratio * history_fz)
    )
    required_history_steps = math.ceil(0.8 * history_steps)
    stable_support = support_history[:, 0] & (
        torch.sum(support_history, dim=1) >= required_history_steps
    )
    leading_stable_support, trailing_stable_support = _platform_travel_order(
        stable_support, forward_travel
    )

    leading_height_delta = leading_pos_w[..., 2] - (
        leading_next_surface_z + settings.wheel_radius
    )
    leading_on_target = scan_spans_transition & torch.all(
        (torch.abs(leading_height_delta) < settings.surface_tolerance)
        & leading_stable_support,
        dim=1,
    )

    trailing_height_delta = trailing_pos_w[..., 2] - (
        trailing_next_surface_z + settings.wheel_radius
    )
    trailing_target_surface_error = torch.abs(
        trailing_surface_z - trailing_next_surface_z
    )
    trailing_on_target = _platform_axle_on_destination(
        trailing_pos_w[..., 2],
        trailing_surface_z,
        trailing_next_surface_z,
        leading_next_surface_z,
        trailing_stable_support,
        settings.wheel_radius,
        settings.surface_tolerance,
    )

    # Stable leading support confirms the handoff; only descent remains latched.
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
    confirmed_descent_handoff = (
        platform_column
        & motion_active
        & leading_is_descent
        & leading_on_target
        & ~trailing_on_target
    )
    descent_latch = (
        (descent_latch | confirmed_descent_handoff)
        & ~trailing_on_target
        & ~just_reset
    )
    descent_forward = torch.where(
        confirmed_descent_handoff, commanded_forward, descent_forward
    )
    env._platform_trailing_descent_latch = descent_latch
    env._platform_trailing_descent_forward = descent_forward
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
    trailing_ascent_wall_score, trailing_descent_wall_score = (
        _platform_wall_contact_scores(
            trailing_forces[..., :2],
            travel_xy,
            settings.wall_force_threshold,
            trailing_exposed_wall,
        )
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

    trailing_fxy = torch.linalg.norm(trailing_forces[..., :2], dim=-1)
    trailing_force_support = (
        (trailing_fz > settings.support_force_threshold)
        & (trailing_fxy < settings.top_horizontal_force_ratio * trailing_fz)
    )
    trailing_surface_height_error = torch.minimum(
        torch.abs(
            trailing_pos_w[..., 2]
            - (trailing_surface_z + settings.wheel_radius)
        ),
        torch.abs(
            trailing_pos_w[..., 2]
            - (trailing_next_surface_z + settings.wheel_radius)
        ),
    )
    trailing_tread_support = trailing_force_support & (
        trailing_surface_height_error < settings.surface_tolerance
    )
    trailing_unsupported_time = _platform_update_trailing_unsupported_time(
        env,
        trailing_descent_phase[:, None] & ~trailing_tread_support,
        just_reset,
    )

    return {
        "asset": asset,
        "motion_on_platform": motion_on_platform,
        "sequence_phase": sequence_phase,
        "leading_phase": leading_phase,
        "trailing_phase": trailing_phase,
        "leading_descent_phase": leading_descent_phase,
        "trailing_descent_phase": trailing_descent_phase,
        "trailing_wall_phase": trailing_wall_phase,
        "trailing_supported": torch.all(
            trailing_fz > settings.support_force_threshold, dim=1
        ),
        "support_count": torch.sum(
            torch.clamp(wheel_forces[..., 2], min=0.0)
            > settings.support_force_threshold,
            dim=1,
        ),
        "travel_sign": travel_sign,
        "leading_direction": leading_direction,
        "trailing_direction": trailing_direction,
        "leading_height_delta": leading_height_delta,
        "trailing_height_delta": trailing_height_delta,
        "leading_vel_z": leading_vel_z,
        "trailing_vel_z": trailing_vel_z,
        "leading_height_spread": torch.abs(
            leading_pos_w[:, 0, 2] - leading_pos_w[:, 1, 2]
        ),
        "trailing_air_time": trailing_air_time,
        "trailing_unsupported_time": trailing_unsupported_time,
        "leading_ascent_wall_score": leading_ascent_wall_score,
        "trailing_ascent_wall_score": trailing_ascent_wall_score,
        "trailing_descent_wall_contact": trailing_descent_wall_contact,
        "leading_needs_progress": leading_direction * leading_height_delta < 0.0,
        "trailing_needs_progress": trailing_direction * trailing_height_delta < 0.0,
    }


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
    """Return signed per-wheel progress toward the selected tread."""
    remaining_distance = torch.clamp(-direction * height_delta, min=0.0)
    ascent_distance_ratio = torch.clamp(
        remaining_distance / ascent_slowdown_distance,
        min=0.0,
        max=1.0,
    )
    ascent_speed_limit = ascent_min_speed + (
        ascent_speed - ascent_min_speed
    ) * ascent_distance_ratio
    speed_limit = torch.where(
        direction > 0.0,
        ascent_speed_limit,
        torch.full_like(direction, descent_speed),
    )
    raw_speed_ratio = direction * vertical_velocity / speed_limit
    speed_ratio = torch.clamp(
        raw_speed_ratio,
        min=-1.0,
        max=1.0,
    )
    toward_ratio = torch.clamp(speed_ratio, min=0.0)
    backslide_ratio = torch.clamp(-speed_ratio, min=0.0)
    toward_progress = (
        (1.0 - speed_quadratic_blend) * toward_ratio
        + speed_quadratic_blend * torch.square(toward_ratio)
    )
    ascent_overspeed = torch.clamp(raw_speed_ratio - 1.0, min=0.0, max=1.0)
    backslide_progress = (
        (1.0 - speed_quadratic_blend) * backslide_ratio
        + speed_quadratic_blend * torch.square(backslide_ratio)
    )
    ascent_gate = ascent_contact_floor + (1.0 - ascent_contact_floor) * wall_score
    contact_gate = torch.where(
        direction > 0.0,
        ascent_gate,
        torch.ones_like(wall_score),
    )
    return (
        contact_gate * toward_progress
        - 2.0 * torch.square(ascent_overspeed) * (direction > 0.0).float()
        - backslide_penalty_scale * backslide_progress
    )


def _platform_ascent_overshoot(
    height_delta: torch.Tensor,
    ascent_mask: torch.Tensor,
    margin: float,
    ramp: float,
) -> torch.Tensor:
    """Return mean axle over-height, active only while moving upward."""
    overshoot = torch.square(
        torch.clamp((height_delta - margin) / ramp, min=0.0, max=1.0)
    )
    return torch.mean(overshoot * ascent_mask.float(), dim=1)


def _log_platform_reward_components(
    env: ManagerBasedRLEnv,
    components: dict[str, torch.Tensor],
) -> None:
    """Accumulate diagnostic components without adding reward terms."""
    reward_manager = env.reward_manager
    parent_buffer = reward_manager._episode_sums[_PLATFORM_REWARD_TERM]
    scale = (
        float(reward_manager.get_term_cfg(_PLATFORM_REWARD_TERM).weight)
        * float(env.step_dt)
    )
    for name, component in components.items():
        key = f"{_PLATFORM_REWARD_TERM}/{name}"
        if key not in reward_manager._episode_sums:
            reward_manager._episode_sums[key] = torch.zeros_like(parent_buffer)
        buffer = reward_manager._episode_sums[key]
        buffer.add_(
            component.detach().to(device=buffer.device, dtype=buffer.dtype),
            alpha=scale,
        )


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
        state["trailing_supported"].float() * torch.mean(
            leading_progress * state["leading_needs_progress"].float(),
            dim=1,
        )
        - settings.ascent_overshoot_penalty_scale * leading_overshoot
    )
    trailing_phase = state["trailing_phase"].float()
    trailing_overshoot = _platform_ascent_overshoot(
        state["trailing_height_delta"],
        (state["leading_direction"] > 0.0)
        | (state["trailing_direction"] > 0.0),
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
        settings.trailing_descent_wall_penalty_scale
        * state["trailing_descent_wall_contact"]
        + settings.trailing_descent_phase_penalty
    )
    leading_descent = state["leading_descent_phase"].float()
    trailing_descent = state["trailing_descent_phase"].float()
    _log_platform_reward_components(
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


def _squared_excess(
    value: torch.Tensor,
    threshold: float,
    scale: float,
) -> torch.Tensor:
    return torch.square(
        torch.clamp((value - threshold) / scale, min=0.0, max=1.0)
    )


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
        (state["trailing_air_time"] - settings.air_time_grace)
        / settings.air_time_ramp,
        min=0.0,
        max=1.0,
    ).mean(dim=1)

    support_deficit = torch.clamp(
        (settings.min_support_count - state["support_count"]).float()
        / float(settings.min_support_count),
        min=0.0,
        max=1.0,
    )
    root_vertical_speed = torch.clamp(
        torch.abs(asset.data.root_lin_vel_w[:, 2]) / settings.vertical_speed_scale,
        max=1.0,
    )
    ballistic_cost = (
        state["motion_on_platform"].float()
        * support_deficit
        * torch.square(root_vertical_speed)
    )

    leading_ascent_phase = (
        state["leading_phase"] & ~state["leading_descent_phase"]
    )
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
            state["trailing_unsupported_time"],
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

    tilt_angle = torch.acos(
        torch.clamp(-asset.data.projected_gravity_b[:, 2], min=-1.0, max=1.0)
    )
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
    descent_phase = (
        state["leading_descent_phase"] | state["trailing_descent_phase"]
    )
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
        torch.maximum(leading_axle_sync_cost, torch.maximum(leading_pitch_cost, base_impact_cost)),
    )
    return torch.maximum(
        base_cost,
        torch.maximum(trailing_hang_cost, trailing_wall_cost),
    )
