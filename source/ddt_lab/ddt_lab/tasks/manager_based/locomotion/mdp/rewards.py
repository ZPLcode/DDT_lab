# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .platform_utils import platform_terrain_type_start_index

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _terrain_type_mask(
    env: ManagerBasedRLEnv,
    terrain_type_start: float,
) -> torch.Tensor:
    """Select the final fraction of generated terrain columns."""
    if not 0.0 <= terrain_type_start < 1.0:
        raise ValueError("terrain_type_start must be in [0, 1).")
    terrain = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    first_type = platform_terrain_type_start_index(terrain_type_start, terrain_generator.num_cols)
    return terrain.terrain_types >= first_type


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def heading_command_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    terrain_type_start: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared world-yaw error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "heading_target"):
        raise AttributeError(f"Command term '{command_name}' does not expose a heading target.")
    error = torch.square(math_utils.wrap_to_pi(command.heading_target - asset.data.heading_w))
    if terrain_type_start is not None:
        error *= _terrain_type_mask(env, terrain_type_start)
    return error


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_rate_l2(env: ManagerBasedRLEnv, upright_gate: bool = True) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    planar_command_only: bool = False,
    upright_gate: bool = True,
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    if planar_command_only:
        command = command[:, :2]
    reward *= torch.norm(command, dim=1) < command_threshold
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def run_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    lateral_or_rot_threshold: float | None = None,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    plane_residual_gate_std: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize leg deviation during forward rolling on locally planar terrain."""
    penalty = mdp.joint_deviation_l1(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    active = torch.norm(command[:, :2], dim=1) > command_threshold
    if lateral_or_rot_threshold is not None:
        active &= torch.norm(command[:, 1:3], dim=1) < lateral_or_rot_threshold
    penalty *= active

    if terrain_sensor_cfg is not None and plane_residual_gate_std is not None:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        _, residual_rms, valid = _fit_height_scan_plane_yaw(asset, sensor)
        planar_gate = torch.exp(-torch.square(residual_rms / plane_residual_gate_std))
        penalty *= torch.where(valid, planar_gate, torch.zeros_like(planar_gate))
    return penalty


def zero_command_base_motion_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    yaw_scale: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize chassis drift only while the velocity command is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
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
    """Penalize wheel rotation only while the velocity command is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    standing = torch.linalg.norm(command, dim=1) < command_threshold
    wheel_speed = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return standing * torch.mean(torch.square(wheel_speed), dim=1)


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.lateral_only: bool = cfg.params.get("lateral_only", False)
        self.penalize_forward: bool = cfg.params.get("penalize_forward", False)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        lateral_only: bool = False,
        penalize_forward: bool = False,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # gate: enforce gait only when commanded
        cmd_raw = env.command_manager.get_command(self.command_name)
        if self.lateral_only:
            lat_cmd = torch.norm(cmd_raw[:, 1:3], dim=1)
            if self.penalize_forward:
                # three-state signed gate:
                #   lin_y/ang_z active → +1 (reward trot)
                #   pure lin_x active  → -1 (penalise trot / encourage rolling)
                #   neither            →  0
                fwd_cmd = torch.abs(cmd_raw[:, 0])
                lat_active = lat_cmd > self.command_threshold
                pure_fwd = (fwd_cmd > self.command_threshold) & (lat_cmd < self.command_threshold)
                sign = torch.where(
                    lat_active,
                    torch.ones_like(lat_cmd),
                    torch.where(pure_fwd, -torch.ones_like(lat_cmd), torch.zeros_like(lat_cmd)),
                )
                active = sign  # use directly as multiplier below
                reward = sign * sync_reward * async_reward
            else:
                active = lat_cmd > self.command_threshold
                reward = torch.where(active, sync_reward * async_reward, 0.0)
        else:
            cmd = torch.linalg.norm(cmd_raw, dim=1)
            body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
            active = torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold)
            reward = torch.where(active, sync_reward * async_reward, 0.0)
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_ratio: float = 4.0,
    upright_gate: bool = True,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > force_ratio * forces_z, dim=1).float()
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    exclude_terrain_type_start: float | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    if exclude_terrain_type_start is not None:
        reward = torch.where(_terrain_type_mask(env, exclude_terrain_type_start), 0.0, reward)
    return reward


def terrain_gated_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    plane_residual_gate_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Track terrain-relative height on locally planar surfaces."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raw_hits = sensor.data.ray_hits_w
    valid_hits = torch.isfinite(raw_hits).all(dim=(1, 2))
    valid_hits &= (torch.abs(raw_hits) < 1.0e6).all(dim=(1, 2))
    safe_heights = torch.where(valid_hits[:, None], raw_hits[..., 2], torch.zeros_like(raw_hits[..., 2]))
    terrain_height = torch.mean(safe_heights, dim=1)
    height_error = torch.square(asset.data.root_pos_w[:, 2] - terrain_height - target_height)

    _, residual_rms, plane_valid = _fit_height_scan_plane_yaw(asset, sensor)
    residual_gate = torch.exp(-torch.square(residual_rms / plane_residual_gate_std))
    cost = torch.where(valid_hits & plane_valid, height_error * residual_gate, torch.zeros_like(height_error))
    if upright_gate:
        cost *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return cost


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
    exclude_terrain_type_start: float | None = None,
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    if exclude_terrain_type_start is not None:
        reward = torch.where(_terrain_type_mask(env, exclude_terrain_type_start), 0.0, reward)
    return reward


def discontinuity_gated_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    discontinuity_height: float,
    discontinuity_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Fade the vertical-speed penalty while the scanner crosses a ledge."""
    if discontinuity_height < 0.0:
        raise ValueError("discontinuity_height must be non-negative.")
    if discontinuity_std <= 0.0:
        raise ValueError("discontinuity_std must be positive.")

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    raw_hits = sensor.data.ray_hits_w
    valid = torch.isfinite(raw_hits).all(dim=(1, 2))
    valid &= (torch.abs(raw_hits) < 1.0e6).all(dim=(1, 2))
    heights = torch.where(valid[:, None], raw_hits[..., 2], torch.zeros_like(raw_hits[..., 2]))
    height_spread = torch.amax(heights, dim=1) - torch.amin(heights, dim=1)
    excess = torch.clamp(height_spread - discontinuity_height, min=0.0)
    smooth_gate = torch.exp(-torch.square(excess / discontinuity_std))
    smooth_gate = torch.where(valid, smooth_gate, torch.ones_like(smooth_gate))

    penalty = torch.square(asset.data.root_lin_vel_b[:, 2]) * smooth_gate
    if upright_gate:
        penalty *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    upright_gate: bool = True,
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


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
    normal_w = math_utils.quat_apply(yaw, torch.nn.functional.normalize(normal_yaw, dim=1))
    return normal_w, residual_rms, valid


def _flat_terrain_gate(
    asset: RigidObject,
    sensor: RayCaster,
    slope_std: float | None,
    residual_std: float | None,
) -> torch.Tensor:
    if slope_std is not None and slope_std <= 0.0:
        raise ValueError("slope_std must be positive.")
    if residual_std is not None and residual_std <= 0.0:
        raise ValueError("residual_std must be positive.")

    terrain_normal_w, residual_rms, valid = _fit_height_scan_plane_yaw(asset, sensor)
    gate = torch.ones_like(residual_rms)
    if slope_std is not None:
        slope = torch.linalg.norm(terrain_normal_w[:, :2], dim=1) / torch.clamp(
            terrain_normal_w[:, 2].abs(), min=0.2
        )
        gate *= torch.exp(-torch.square(slope / slope_std))
    if residual_std is not None:
        gate *= torch.exp(-torch.square(residual_rms / residual_std))
    return torch.where(valid, gate, torch.zeros_like(gate))


def terrain_or_world_orientation_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    roughness_std: float,
    discontinuity_height: float | None = None,
    discontinuity_std: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Follow smooth slopes, remain level on rough ground, and relax at ledges."""
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


def flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    if upright_gate:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_thigh_x_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    thigh_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    terrain_slope_gate_std: float | None = None,
    plane_residual_gate_std: float | None = None,
    upright_gate: bool = False,
) -> torch.Tensor:
    """Penalize fore-aft offsets between corresponding wheel and thigh frames."""
    asset: Articulation = env.scene[thigh_cfg.name]
    offset_w = asset.data.body_link_pos_w[:, wheel_cfg.body_ids] - asset.data.body_link_pos_w[:, thigh_cfg.body_ids]
    num_feet = offset_w.shape[1]
    root_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    offset_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), offset_w.reshape(-1, 3)).reshape(
        env.num_envs, num_feet, 3
    )
    penalty = torch.mean(torch.square(offset_b[:, :, 0] / std), dim=1)

    if terrain_sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        penalty *= _flat_terrain_gate(asset, sensor, terrain_slope_gate_std, plane_residual_gate_std)
    if upright_gate:
        penalty *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty


def default_joint_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_gate: bool = True,
) -> torch.Tensor:
    """Penalize joint position deviation from the default pose (sum of squares).

    Port of ``LocomotionWithNP3O._reward_default_joint``. Includes the same
    upright filter (``clamp(-grav_z, 0, 0.7) / 0.7``) so the penalty fades when
    the robot is not roughly upright.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(q - q_default), dim=1)
    if upright_gate:
        reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def power_distribution_var(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize uneven power distribution across joints using variance.

    Computes the variance of per-joint mechanical power ``(τ · θ̇)`` across
    the selected joints.  A high variance means some joints are working much
    harder than others; penalising it encourages balanced actuation.

    Reference reward: ``var(τ · θ̇)²``, weight ``-1e-5``.

    Shape: ``(num_envs,)``
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # per-joint power (B, n_joints)
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    reward = torch.var(power, dim=1)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    # variance across joints for each environment
    return reward


def wheel_scrub_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise wheel scrubbing during lateral / rotational commands.

    Scrubbing occurs when a foot is on the ground and moving sideways —
    the typical outcome of tank-style differential steering.  Penalising
    ``contact_weight × foot_lateral_speed`` pushes the policy toward
    lifting and re-planting legs instead.

    Physics basis: scrub_power ∝ F_normal · v_lateral (Coulomb friction loss).
    Reference: Bjelonic et al. "Keep Rollin'" RA-L/ICRA 2019.
    Recommended weight: -0.5 ~ -2.0
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # contact force magnitude as proxy for normal force.
    # norm() ≈ F_normal × sqrt(1+μ²), the constant factor is absorbed by the 50N normalization.
    # More robust than z-component alone on non-flat terrain (slopes reduce z projection).
    contact_force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)  # (B, K)
    contact_w = torch.clamp(contact_force / 50.0, 0.0, 1.0)

    # foot velocity projected onto robot body-frame y-axis (true lateral direction).
    # This avoids penalising forward wheel rolling (body-x) which is desirable.
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (B, K, 3)
    B, K = foot_vel_w.shape[:2]
    root_quat = asset.data.root_quat_w  # (B, 4)
    root_quat_exp = root_quat.unsqueeze(1).expand(-1, K, -1).reshape(B * K, 4)
    foot_vel_body = quat_apply_inverse(root_quat_exp, foot_vel_w.reshape(B * K, 3)).reshape(B, K, 3)
    foot_lateral_speed = foot_vel_body[:, :, 1].abs()  # body-y = sideways  (B, K)

    # gate: only penalise when vy or wz command is active
    cmd = env.command_manager.get_command(command_name)
    lateral_or_rot = torch.norm(cmd[:, 1:3], dim=1)
    cmd_gate = torch.clamp(lateral_or_rot / command_threshold, 0.0, 1.0).unsqueeze(-1)  # (B, 1)

    scrub = contact_w * foot_lateral_speed * cmd_gate  # (B, K)
    return scrub.sum(dim=-1) * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def _update_debounced_air_time(
    carried_air_time: torch.Tensor,
    first_contact: torch.Tensor,
    last_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    current_air_time: torch.Tensor,
    landing_confirm_time: float,
    max_tracked_air_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carry air time across brief contacts and reset it only after a stable landing."""
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


def foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    target_height: float = 0.08,
    yaw_target_height: float | None = None,
    std: float = 0.04,
    wheel_radius: float = 0.09,
    command_threshold: float = 0.10,
    max_air_time: float = 0.4,
    landing_confirm_time: float = 0.0,
    hover_penalty_scale: float = 0.0,
    min_contact: int = 2,
    min_stance_time: float = 0.0,
    rearm_penalty_scale: float = 0.0,
    lift_penalty_scale: float = 1.0,
    target_centered: bool = False,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    plane_residual_gate_std: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shape swing clearance for lateral/yaw motion and suppress it while rolling."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    if max_air_time <= 0.0:
        raise ValueError("max_air_time must be positive.")
    if landing_confirm_time < 0.0:
        raise ValueError("landing_confirm_time must be non-negative.")
    if hover_penalty_scale < 0.0:
        raise ValueError("hover_penalty_scale must be non-negative.")
    if min_stance_time < 0.0:
        raise ValueError("min_stance_time must be non-negative.")
    if rearm_penalty_scale < 0.0:
        raise ValueError("rearm_penalty_scale must be non-negative.")
    if target_height < 0.0 or (yaw_target_height is not None and yaw_target_height < 0.0):
        raise ValueError("clearance targets must be non-negative.")

    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0  # (B, K)
    in_air = ~in_contact
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (B, K)

    planar_gate = torch.ones(env.num_envs, device=env.device)
    if terrain_sensor_cfg is not None and terrain_sensor_cfg.name in env.scene.sensors:
        terrain_sensor: RayCaster = env.scene[terrain_sensor_cfg.name]
        ray_hits_z = terrain_sensor.data.ray_hits_w[..., 2]
        terrain_z = torch.mean(ray_hits_z, dim=1, keepdim=True)
        if plane_residual_gate_std is not None:
            if plane_residual_gate_std <= 0.0:
                raise ValueError("plane_residual_gate_std must be positive.")
            _, plane_residual, plane_valid = _fit_height_scan_plane_yaw(asset, terrain_sensor)
            planar_gate = torch.exp(-torch.square(plane_residual / plane_residual_gate_std))
            planar_gate = torch.where(plane_valid, planar_gate, torch.ones_like(planar_gate))
    else:
        terrain_z = torch.zeros_like(foot_pos_z)
    clearance = foot_pos_z - terrain_z - wheel_radius

    cmd = env.command_manager.get_command(command_name)
    lateral_or_rot = torch.norm(cmd[:, 1:3], dim=1)
    active = (lateral_or_rot > command_threshold).unsqueeze(-1)
    yaw_active = (torch.abs(cmd[:, 2]) > command_threshold).unsqueeze(-1)

    clearance_target = torch.full_like(clearance, target_height)
    if yaw_target_height is not None:
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
        carried_air_time = getattr(env, "_foot_clearance_carried_air_time", None)
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
        env._foot_clearance_carried_air_time = carried_air_time
    swing_decay = torch.exp(-(air_time / max_air_time).pow(2))
    hover_excess = torch.clamp((air_time - max_air_time) / max_air_time, min=0.0, max=1.0)
    hover_penalty = hover_penalty_scale * torch.square(hover_excess)
    num_contact = in_contact.float().sum(dim=-1, keepdim=True)
    has_other_contact = (num_contact >= min_contact).float()
    last_contact_time = sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    stable_takeoff = (last_contact_time >= min_stance_time).float()
    rearm_penalty = rearm_penalty_scale * (1.0 - stable_takeoff)

    positive_active_term = in_air.float() * height_reward * swing_decay * has_other_contact * stable_takeoff
    active_term = positive_active_term - active_air_phase.float() * (hover_penalty + rearm_penalty)

    inactive_term = -lift_penalty_scale * torch.clamp(clearance, min=0.0).pow(2)
    reward = torch.where(active, active_term, in_air.float() * inactive_term)
    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward.sum(dim=-1) * planar_gate * upright_gate


def wheel_roll_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward grounded feet with forward velocity whenever lin_x is active.

    Gate: only requires vx > threshold (no lat_inactive condition).
    This creates a constant incentive to keep feet on the ground and roll
    forward, which the policy will override only when scrubbing forces it to lift.
    Recommended weight: +0.5 ~ +2.0
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0  # (B, K)
    foot_vel_x = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0].abs()  # (B, K)

    cmd = env.command_manager.get_command(command_name)
    fwd_active = (torch.abs(cmd[:, 0]) > command_threshold).float().unsqueeze(-1)  # (B, 1)

    return (in_contact.float() * foot_vel_x * fwd_active).sum(dim=-1)


def feet_air_time_wheeled(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.3,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Reward long swing steps during lateral/rotational commands (wheel-legged variant).

    Mirrors ``feet_air_time`` logic: reward = (last_air_time - threshold) × first_contact.
    Gate: only active when lin_y or ang_z command exceeds ``command_threshold``.
    Recommended weight: +1.0 ~ +2.0
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]

    cmd = env.command_manager.get_command(command_name)
    lateral_or_rot = torch.norm(cmd[:, 1:3], dim=1)
    cmd_gate = (lateral_or_rot > command_threshold).float()

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * cmd_gate


def no_step_forward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalise feet in the air during pure forward (lin_x) motion.

    Gate: lin_x active AND norm(vy, wz) quiet.
    Pairs with ``gait_trot`` (lateral_only): the two rewards together teach
    - wheel rolling during lin_x  (no_step_forward penalises stepping)
    - stepping gait during lin_y/ang_z (gait_trot rewards trot)
    Recommended weight: -1.0 ~ -3.0
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0  # (B, K)
    in_air = ~in_contact  # (B, K)

    cmd = env.command_manager.get_command(command_name)
    fwd_active = torch.abs(cmd[:, 0]) > command_threshold
    lat_quiet = torch.norm(cmd[:, 1:3], dim=1) < command_threshold
    pure_fwd = (fwd_active & lat_quiet).float()  # (B,)

    # penalise number of airborne feet during pure forward motion
    return in_air.float().sum(dim=-1) * pure_fwd


def hip_zero_on_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    hip_asset_cfg: SceneEntityCfg,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Dead-band hip constraint active only when the foot is in contact.

    During swing phase the hip is free to move for repositioning.
    Once the foot touches down the hip must return to 0 to prevent lateral
    ground forces and scrubbing.

    ``sensor_cfg.body_ids`` and ``hip_asset_cfg.joint_ids`` must share the
    same leg order (FL → FR → RL → RR).
    Recommended weight: -3.0 ~ -10.0
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[hip_asset_cfg.name]

    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0  # (B, K)
    hip_pos = asset.data.joint_pos[:, hip_asset_cfg.joint_ids]  # (B, K)
    excess = torch.clamp(hip_pos.abs() - tolerance, min=0.0)
    return (in_contact.float() * excess.pow(2)).sum(dim=-1)


def hip_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    tolerance: float = 0.05,
    loose_ratio: float = 0.2,
    exclude_terrain_type_start: float | None = None,
) -> torch.Tensor:
    """Hip constraint with two penalty levels based on command direction.

    - lat_quiet (standing / pure lin_x): full penalty × 1.0
    - lat_active (lin_y / ang_z stepping): reduced penalty × ``loose_ratio``

    ``loose_ratio=0`` → no penalty during stepping (original behaviour).
    ``loose_ratio=0.2`` → 20% penalty during stepping, 100% when still/lin_x.

    Recommended weight: -5.0 ~ -20.0
    """
    asset: Articulation = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (B, K)

    cmd = env.command_manager.get_command(command_name)
    lat_quiet = (torch.norm(cmd[:, 1:3], dim=1) < command_threshold).float()  # (B,) 0 or 1

    # smooth blend: 1.0 when quiet, loose_ratio when lateral/rot active
    scale = lat_quiet + loose_ratio * (1.0 - lat_quiet)  # (B,)

    excess = torch.clamp(hip_pos.abs() - tolerance, min=0.0)  # (B, K)
    reward = excess.pow(2).sum(dim=-1) * scale
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    if exclude_terrain_type_start is not None:
        reward = torch.where(_terrain_type_mask(env, exclude_terrain_type_start), 0.0, reward)
    return reward


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
    """Penalize hip deflection during flat-ground yaw commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.square(torch.clamp(hip_pos.abs() - tolerance, min=0.0)).sum(dim=-1)
    command = env.command_manager.get_command(command_name)
    yaw_gate = (torch.abs(command[:, 2]) > yaw_threshold).float()
    sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    flat_gate = _flat_terrain_gate(asset, sensor, terrain_slope_gate_std, plane_residual_gate_std)
    upright_gate = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * yaw_gate * flat_gate * upright_gate


class PositionHoldAtRest(ManagerTermBase):
    """Penalise xy displacement from the position saved when cmd last became zero.

    When the command magnitude transitions to zero (or just after reset),
    the robot's current xy world position is saved.  While cmd remains zero,
    any deviation from that saved position is penalised as ``||Δpos||²``.

    Recommended weight: -5.0 ~ -20.0
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._target_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._prev_zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        cmd = torch.norm(env.command_manager.get_command(command_name), dim=1)
        cmd_zero = cmd < command_threshold  # (B,) bool

        # After episode reset (episode_length_buf == 1), clear saved position
        just_reset = env.episode_length_buf == 1
        self._initialized[just_reset] = False

        # Save position when: just transitioned to zero, or not yet initialised
        save = cmd_zero & (~self._prev_zero | ~self._initialized)
        if save.any():
            self._target_xy[save] = asset.data.root_pos_w[save, :2]
            self._initialized[save] = True

        self._prev_zero = cmd_zero.clone()

        # Penalise xy drift from saved position while cmd is zero
        active = cmd_zero & self._initialized
        pos_error = (asset.data.root_pos_w[:, :2] - self._target_xy).norm(dim=-1)
        return pos_error.pow(2) * active.float() * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def hip_mirror(
    env: ManagerBasedRLEnv,
    left_asset_cfg: SceneEntityCfg,
    right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise left-right hip asymmetry: enforce q_left + q_right ≈ 0.

    For a wheel-legged robot, the hip joints on the left and right sides should
    be mirror images (opposite signs) when viewed from above:
        FL_hip ≈ -FR_hip,  RL_hip ≈ -RR_hip

    Penalising ``(q_left + q_right)²`` pushes both directions of rotation to
    produce the same *magnitude* of hip angle, removing the asymmetry where
    one rotation direction produces large hip deviations and the other does not.

    Recommended weight: -0.5 ~ -2.0
    """
    asset: Articulation = env.scene[left_asset_cfg.name]
    q_left = asset.data.joint_pos[:, left_asset_cfg.joint_ids]  # (B, K_L)
    q_right = asset.data.joint_pos[:, right_asset_cfg.joint_ids]  # (B, K_R)
    # symmetric ↔ sum ≈ 0
    return (q_left + q_right).pow(2).sum(dim=-1)


# ------------------------------------------------------------------
# D1H-specific reward functions
# ------------------------------------------------------------------


def keep_upright_only_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, upright_gz_threshold: float, time_coeff: float
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    g_z = asset.data.projected_gravity_b[:, 2]

    if not hasattr(env, "upright_timer_buf") or env.upright_timer_buf is None:
        env.upright_timer_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    fully_upright_mask = (g_z <= upright_gz_threshold).float()

    env.upright_timer_buf = torch.where(
        fully_upright_mask > 0.5, env.upright_timer_buf + env.step_dt, torch.zeros_like(env.upright_timer_buf)
    )

    base_score = fully_upright_mask * 1.0
    time_bonus = fully_upright_mask * env.upright_timer_buf * time_coeff
    total_reward = base_score + time_bonus

    return total_reward


def reward_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
    command_name: str = "base_velocity",
    min_air_t: float = 0.5,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    air_time = contact_sensor.data.air_time[:, sensor_cfg.body_ids]

    landed = contact & (air_time > min_air_t)
    rew_airTime = torch.sum(landed.float(), dim=-1)

    cmd = env.command_manager.get_command(command_name)
    moving_mask = torch.norm(cmd[:, :2], dim=1) > 0.1
    rew_airTime *= moving_mask

    return rew_airTime


def reward_collision_head(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["base_link"]),
    threshold: float = 10.0,
) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.sum((forces > threshold).float(), dim=-1)


def reward_dof_thigh_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_thigh_joint"])
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def reward_body_pos_to_feet_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sigma: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_quat_w

    rel_w = feet_pos_w - base_pos_w
    n_envs, n_feet = rel_w.shape[:2]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    rel_w_flat = rel_w.reshape(-1, 3)
    rel_b = quat_apply_inverse(base_quat_w_expanded, rel_w_flat).reshape(n_envs, n_feet, 3)

    x_mean_abs = torch.abs(torch.mean(rel_b[:, :, 0], dim=1))
    reward = torch.exp(-x_mean_abs / sigma)
    return reward


def reward_body_feet_distance_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sigma: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_quat_w = asset.data.root_quat_w

    foot_diff_w = feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :]
    foot_diff_b = quat_apply_inverse(base_quat_w, foot_diff_w)

    x_err = torch.abs(foot_diff_b[:, 0]) / sigma
    reward = torch.square(x_err)
    return reward


def reward_body_feet_distance_y(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sigma: float, desired_feet_distance: float
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_quat_w = asset.data.root_quat_w

    foot_diff_w = feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :]
    foot_diff_b = quat_apply_inverse(base_quat_w, foot_diff_w)

    y_abs = torch.abs(foot_diff_b[:, 1])
    y_err = torch.abs(y_abs - desired_feet_distance) / sigma
    reward = torch.square(y_err)
    return reward


def reward_body_symmetry_y(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sigma: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_quat_w

    rel_w = feet_pos_w - base_pos_w
    rel_b_0 = quat_apply_inverse(base_quat_w, rel_w[:, 0, :])
    rel_b_1 = quat_apply_inverse(base_quat_w, rel_w[:, 1, :])

    y1_abs = torch.abs(rel_b_0[:, 1])
    y2_abs = torch.abs(rel_b_1[:, 1])
    sym_err = torch.abs(y1_abs - y2_abs)

    reward = torch.exp(-sym_err / sigma)
    return reward


def reward_body_symmetry_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sigma: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_body_names = ["FL_foot", "FR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_body_names]

    feet_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    base_quat_w = asset.data.root_quat_w

    rel_w = feet_pos_w - base_pos_w
    rel_b_0 = quat_apply_inverse(base_quat_w, rel_w[:, 0, :])
    rel_b_1 = quat_apply_inverse(base_quat_w, rel_w[:, 1, :])

    z1_abs = torch.abs(rel_b_0[:, 2])
    z2_abs = torch.abs(rel_b_1[:, 2])
    sym_err = torch.abs(z1_abs - z2_abs)

    reward = torch.exp(-sym_err / sigma)
    return reward
