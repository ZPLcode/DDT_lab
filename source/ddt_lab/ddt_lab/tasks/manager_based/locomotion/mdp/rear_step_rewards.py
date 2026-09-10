# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward rear wheels clearing a riser before landing on its upper tread."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .platform_rewards import _stable_horizontal_support
from .rewards import _terrain_type_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _nearest_tread(
    rays: torch.Tensor, valid_rays: torch.Tensor, points_xy: torch.Tensor, max_distance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample nearby ground without treating a missing ray as zero-height terrain."""
    distance_sq = torch.square(rays[:, None, :, :2] - points_xy[:, :, None, :]).sum(dim=-1)
    distance_sq = distance_sq.masked_fill(~valid_rays[:, None, :], float("inf"))
    distance, index = distance_sq.min(dim=-1)
    return torch.gather(rays[..., 2], 1, index), distance <= max_distance**2


class RearWheelStepReward(ManagerTermBase):
    """Per-wheel lift progress, clean landing bonuses, and wall-contact penalties.

    Targets are latched in world coordinates from the existing height scanner.
    Each rear wheel may lift independently once the front axle supports the body.
    Lift progress pays only for new height, and landing pays once per rising tread;
    neither hovering nor repeatedly hopping on the same tread earns more reward.
    Free lift after a wall contact can still earn progress, while that attempt
    remains ineligible for the clean landing bonus.

    Wall contact is a geometry-gated estimate from net wheel forces, not a contact
    pair measurement. Scan resolution and force thresholds limit its accuracy.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        shape = (env.num_envs, 2)
        self.active = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.wall_touched = torch.zeros_like(self.active)
        self.cleared = torch.zeros_like(self.active)
        # Retained across target changes until reset, so curriculum can judge the
        # whole episode before RewardManager clears this term's state.
        self.episode_clean_landed = torch.zeros_like(self.active)
        self.episode_wall_touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.floor_z = torch.zeros(shape, device=env.device)
        self.target_z = torch.zeros_like(self.floor_z)
        self.best_lift = torch.zeros_like(self.floor_z)
        self.completed_z = torch.full_like(self.floor_z, -1.0e6)
        self.edge_xy = torch.zeros((*shape, 2), device=env.device)
        self.direction_xy = torch.zeros_like(self.edge_xy)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.active[env_ids] = False
        self.wall_touched[env_ids] = False
        self.cleared[env_ids] = False
        self.episode_clean_landed[env_ids] = False
        self.episode_wall_touched[env_ids] = False
        self.best_lift[env_ids] = 0.0
        self.completed_z[env_ids] = -1.0e6

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        terrain_sensor_cfg: SceneEntityCfg,
        wheel_sensor_cfg: SceneEntityCfg,
        wheel_asset_cfg: SceneEntityCfg,
        command_name: str,
        terrain_type_start: float,
        terrain_type_end: float,
        wheel_radius: float = 0.087,
        lookahead: float = 0.55,
        min_step_height: float = 0.04,
        clearance_margin: float = 0.06,
        surface_tolerance: float = 0.05,
        landing_margin: float = 0.05,
        scan_tolerance: float = 0.15,
        wall_force_threshold: float = 20.0,
        wall_force_ratio: float = 0.5,
        clearance_bonus: float = 0.25,
        landing_bonus: float = 1.0,
        wall_penalty_scale: float = 0.5,
        approach_distance: float = 0.30,
        approach_penalty_scale: float = 2.0,
    ) -> torch.Tensor:
        if not 0.0 <= terrain_type_start < terrain_type_end < 1.0:
            raise ValueError("Rear-step terrain bounds must satisfy 0 <= start < end < 1.")
        if (
            min(
                wheel_radius,
                lookahead,
                min_step_height,
                clearance_margin,
                surface_tolerance,
                scan_tolerance,
                approach_distance,
            )
            <= 0
        ):
            raise ValueError("Rear-step geometry scales must be positive.")
        if (
            min(
                landing_margin,
                wall_force_threshold,
                wall_force_ratio,
                clearance_bonus,
                landing_bonus,
                wall_penalty_scale,
                approach_penalty_scale,
            )
            < 0
        ):
            raise ValueError("Rear-step margins, force thresholds and reward scales must be non-negative.")
        if len(wheel_asset_cfg.body_ids) != 4 or len(wheel_sensor_cfg.body_ids) != 4:
            raise ValueError("Rear-step wheels must be ordered FL, FR, RL, RR.")

        asset = env.scene[wheel_asset_cfg.name]
        sensor = env.scene.sensors[wheel_sensor_cfg.name]
        raw_rays = env.scene.sensors[terrain_sensor_cfg.name].data.ray_hits_w
        valid_rays = torch.isfinite(raw_rays).all(dim=-1)
        rays = torch.where(valid_rays[..., None], raw_rays, torch.zeros_like(raw_rays))
        wheels = asset.data.body_pos_w[:, wheel_asset_cfg.body_ids]
        bottom_z = wheels[..., 2] - wheel_radius
        # Project the body's forward axis onto the horizontal plane (quaternion is wxyz).
        w, x, y, z = asset.data.root_quat_w.unbind(dim=-1)
        heading = torch.stack((1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z)), dim=-1)
        heading = heading / torch.linalg.norm(heading, dim=-1, keepdim=True).clamp_min(1.0e-6)
        rear_xy = wheels[:, 2:, :2]
        local_z, local_valid = _nearest_tread(rays, valid_rays, wheels[..., :2], scan_tolerance)
        lower_z, lower_valid = _nearest_tread(
            rays, valid_rays, rear_xy - scan_tolerance * heading[:, None, :], scan_tolerance
        )

        delta = rays[:, None, :, :2] - rear_xy[:, :, None, :]
        forward = (delta * heading[:, None, None, :]).sum(dim=-1)
        lateral = delta[..., 0] * heading[:, None, None, 1] - delta[..., 1] * heading[:, None, None, 0]
        candidates = (
            valid_rays[:, None, :]
            & lower_valid[:, :, None]
            & (forward >= 0.0)
            & (forward <= lookahead)
            & (lateral.abs() <= 0.10)
            & (rays[:, None, :, 2] > lower_z[:, :, None] + min_step_height)
        )
        distance, index = forward.masked_fill(~candidates, float("inf")).min(dim=-1)
        candidate_z = torch.gather(rays[..., 2], 1, index)
        candidate_xy = torch.gather(rays[..., :2], 1, index[..., None].expand(-1, -1, 2))
        ascent = _terrain_type_mask(env, terrain_type_start) & ~_terrain_type_mask(env, terrain_type_end)
        forward_command = env.command_manager.get_command(command_name)[:, 0] > 0.10
        enabled = ascent & forward_command & (-asset.data.projected_gravity_b[:, 2] > 0.3)
        arm = (
            enabled[:, None]
            & ~self.active
            & torch.isfinite(distance)
            & (candidate_z > self.completed_z + min_step_height)
            & (bottom_z[:, 2:] < candidate_z + clearance_margin)
        )
        self.floor_z = torch.where(arm, lower_z, self.floor_z)
        self.target_z = torch.where(arm, candidate_z, self.target_z)
        self.edge_xy = torch.where(arm[..., None], candidate_xy, self.edge_xy)
        self.direction_xy = torch.where(arm[..., None], heading[:, None, :], self.direction_xy)
        self.wall_touched &= ~arm
        self.cleared &= ~arm
        self.active |= arm

        valid = self.active & enabled[:, None] & local_valid[:, 2:]
        edge_progress = ((rear_xy - self.edge_xy) * self.direction_xy).sum(dim=-1)
        below_top = bottom_z[:, 2:] < self.target_z + 0.01
        near_wall = (edge_progress >= -wheel_radius - scan_tolerance) & (edge_progress <= scan_tolerance)
        history = sensor.data.net_forces_w_history[:, :, wheel_sensor_cfg.body_ids[2:]]
        force_xy = torch.linalg.norm(history[..., :2], dim=-1)
        force_z = history[..., 2].clamp_min(0.0)
        excess = (force_xy - wall_force_ratio * force_z - wall_force_threshold).clamp_min(0.0).amax(dim=1)
        wall_contact = valid & near_wall & below_top & (excess > 0.0)
        self.wall_touched |= wall_contact
        self.episode_wall_touched |= wall_contact.any(dim=1)
        wall_cost = torch.where(wall_contact, (excess / 100.0).clamp(max=3.0), 0.0)

        support = _stable_horizontal_support(sensor, wheel_sensor_cfg.body_ids, 10.0, 2.0, 10, 0.8)
        on_surface = local_valid & ((bottom_z - local_z).abs() < surface_tolerance)
        front_support = (support[:, :2] & on_surface[:, :2]).all(dim=1)
        front_ready = front_support[:, None] & (
            local_z[:, :2].amin(dim=1)[:, None] >= self.target_z - surface_tolerance
        )
        lift = (
            (bottom_z[:, 2:] - self.floor_z)
            / (self.target_z + clearance_margin - self.floor_z).clamp_min(min_step_height)
        ).clamp(0.0, 1.0)
        self.best_lift = torch.where(arm, lift, self.best_lift)
        new_lift = (lift - self.best_lift).clamp_min(0.0)
        self.best_lift = torch.where(self.active, torch.maximum(self.best_lift, lift), self.best_lift)
        airborne = sensor.data.net_forces_w[:, wheel_sensor_cfg.body_ids[2:]].norm(dim=-1) < 10.0
        # A past hit must not cut off learning to lift away from the wall. Keep
        # the height high-water mark above, including contact-assisted progress,
        # so only further free lift pays and repeated recovery cannot farm it.
        free_swing = valid & front_ready & airborne & ~wall_contact
        lift_reward = torch.where(free_swing, new_lift, 0.0)
        clean_swing = free_swing & ~self.wall_touched
        # Require clearance while still behind the lip, before moving onto the tread.
        self.cleared |= clean_swing & (lift >= 1.0 - 1.0e-5) & (edge_progress <= -wheel_radius)
        # Discourage advancing toward the lip before clearance, using wheel
        # translation rather than wheel spin or the commanded base velocity.
        # Once the cleared wheel has crossed the lip, touchdown is allowed.
        proximity = ((edge_progress + wheel_radius + approach_distance) / approach_distance).clamp(0.0, 1.0)
        approaching_speed = (
            (asset.data.body_lin_vel_w[:, wheel_asset_cfg.body_ids[2:], :2] * self.direction_xy)
            .sum(dim=-1)
            .clamp(0.0, 1.0)
        )
        safely_crossed = self.cleared & (edge_progress >= wheel_radius)
        unsafe_approach = valid & front_ready & ~safely_crossed & (edge_progress <= wheel_radius + landing_margin)
        approach_cost = torch.where(unsafe_approach, proximity * (1.0 - lift) * approaching_speed, 0.0)
        landed = (
            valid
            & front_ready
            & support[:, 2:]
            & on_surface[:, 2:]
            & (local_z[:, 2:] >= self.target_z - surface_tolerance)
            & (local_z[:, 2:] > self.floor_z + min_step_height)
            & (edge_progress >= wheel_radius + landing_margin)
        )
        clean_landing = landed & self.cleared & ~self.wall_touched
        self.episode_clean_landed |= clean_landing
        self.completed_z = torch.where(landed, self.target_z, self.completed_z)
        self.active &= ~landed

        # Progress and landing are event bonuses: divide by dt so RewardManager's
        # integration leaves their configured total value independent of control rate.
        components = {
            "clearance_progress": clearance_bonus * lift_reward.sum(dim=1) / env.step_dt,
            "clean_landing": landing_bonus * clean_landing.float().sum(dim=1) / env.step_dt,
            "wall_contact": -wall_penalty_scale * wall_cost.sum(dim=1),
            "unsafe_approach": -approach_penalty_scale * approach_cost.sum(dim=1),
        }
        # Recovery is a diagnostic subset of clearance_progress, not extra reward.
        logged_components = {
            **components,
            "recovery_progress": (
                clearance_bonus * torch.where(self.wall_touched, lift_reward, 0.0).sum(dim=1) / env.step_dt
            ),
        }
        for name, value in logged_components.items():
            key = f"ascent_rear_step/{name}"
            sums = env.reward_manager._episode_sums
            if key not in sums:
                sums[key] = torch.zeros(env.num_envs, device=env.device)
            sums[key].add_(value.detach() * self.cfg.weight * env.step_dt)
        # A 0/1 episode diagnostic; overwrite so a later hit revokes eligibility.
        # This is only the cleanliness prerequisite, not traversal success.
        clean_pair = self.episode_clean_landed.all(dim=1) & ~self.episode_wall_touched
        sums["ascent_rear_step/wall_free_clean_pair"] = clean_pair.float() * env.max_episode_length_s
        return sum(components.values())
