# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stock D1 NP3O task for curriculum-based platform ascent and descent."""

import math
from collections.abc import Sequence

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
import torch
from ddt_lab.assets.terrains.platform import (
    D1_HIGH_PLATFORM_TERRAINS_CFG,
    D1_PLATFORM_DESCENT_TERRAIN_START,
    D1_PLATFORM_TERRAIN_START,
    D1_SLOPE_TERRAIN_START,
    D1_STAIRS_TERRAIN_START,
)
from ddt_lab.managers import CostTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardManager
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..base_env_cfg import CommandsCfg, SceneCfg, TerminationsCfg
from .rough_env_cfg import CostsCfg, D1RoughNP3OEnvCfg, RoughRewardsCfg


class PlatformRewardManager(RewardManager):
    """Apply the platform reward clip before recording and episode reset."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._termination_index = self._term_names.index("is_terminated")
        self._clipped_episode_sum = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )

    def compute(self, dt: float) -> torch.Tensor:
        reward = super().compute(dt)
        termination_reward = self._step_reward[:, self._termination_index] * dt
        clipped_reward = mdp.platform_positive_reward_clip(reward, termination_reward)
        self._reward_buf[:] = clipped_reward
        self._clipped_episode_sum += clipped_reward
        return self._reward_buf

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        clipped_sum = torch.mean(self._clipped_episode_sum[env_ids])
        extras = super().reset(env_ids)
        extras["Episode_Reward/total_clipped"] = clipped_sum / self._env.max_episode_length_s
        self._clipped_episode_sum[env_ids] = 0.0
        return extras


class D1PlatformRLEnv(ManagerBasedRLEnv):
    """D1 platform environment with non-negative ordinary rewards."""

    def load_managers(self):
        super().load_managers()
        self.reward_manager = PlatformRewardManager(self.cfg.rewards, self)


@configclass
class PlatformCommandsCfg(CommandsCfg):
    """General commands with dedicated ascent and bidirectional descent ranges."""

    base_velocity = mdp.TerrainAwareVelocityCommandCfg(
        asset_name="robot",
        # One command per 20 s episode. In particular, a descent command must
        # not reverse sign halfway through a leading/trailing transition.
        resampling_time_range=(20.0, 20.0),
        # Standing samples remain on ordinary columns; platform commands clear them.
        rel_standing_envs=0.15,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        # Training must not depend on the remote velocity-arrow USD asset.
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.2),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        platform_terrain_start=D1_PLATFORM_TERRAIN_START,
        planar_deadzone=0.2,
        platform_ranges=mdp.TerrainAwareVelocityCommandCfg.PlatformRanges(
            lin_vel_x=(0.20, 0.60),
            lin_vel_y=(0.0, 0.0),
            heading=(-math.radians(10.0), math.radians(10.0)),
        ),
        platform_descent_terrain_start=D1_PLATFORM_DESCENT_TERRAIN_START,
        platform_descent_ranges=mdp.TerrainAwareVelocityCommandCfg.PlatformRanges(
            # Magnitude range; TerrainAwareVelocityCommand assigns +/- signs.
            lin_vel_x=(0.20, 0.32),
            lin_vel_y=(0.0, 0.0),
            heading=(-math.radians(10.0), math.radians(10.0)),
        ),
        platform_descent_bidirectional=True,
    )


def _platform_term_params() -> dict[str, object]:
    """Create independently resolvable inputs shared by reward and cost terms."""
    return {
        "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
        "wheel_sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        ),
        "wheel_asset_cfg": SceneEntityCfg(
            "robot",
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            preserve_order=True,
        ),
        "command_name": "base_velocity",
        "terrain_type_start": D1_PLATFORM_TERRAIN_START,
        "settings": mdp.PlatformTraversalSettings(),
    }


@configclass
class PlatformRewardsCfg(RoughRewardsCfg):
    """Reward terms for supported platform traversal."""

    is_terminated = RewTerm(func=mdp.is_terminated, weight=-0.8)
    highplatform_yaw = RewTerm(
        func=mdp.heading_command_error_l2,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "terrain_type_start": D1_PLATFORM_TERRAIN_START,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.platform_feet_stumble,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "terrain_type_start": D1_PLATFORM_TERRAIN_START,
        },
    )
    contact_forces = RewTerm(
        func=mdp.platform_landing_force_penalty,
        weight=-2.0e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 500.0,
        },
    )
    platform_traversal = RewTerm(
        func=mdp.platform_traversal_reward,
        weight=1.5,
        params=_platform_term_params(),
    )
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "exclude_terrain_type_start": D1_STAIRS_TERRAIN_START,
        },
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={
            "target_height": 0.52,
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "exclude_terrain_type_start": D1_PLATFORM_TERRAIN_START,
        },
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-8.0,
        params={
            "exclude_terrain_type_start": D1_SLOPE_TERRAIN_START,
        },
    )
    flat_wheel_thigh_alignment = RewTerm(
        func=mdp.wheel_thigh_x_alignment,
        weight=-0.5,
        params={
            "std": 0.05,
            "thigh_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"],
                preserve_order=True,
            ),
            "wheel_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "exclude_terrain_type_start": D1_SLOPE_TERRAIN_START,
            "upright_gate": True,
        },
    )
    wheel_scrub_penalty = RewTerm(
        func=mdp.wheel_scrub_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "command_threshold": 0.10,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=1.50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "target_height": 0.04,
            "std": 0.03,
            "wheel_radius": 0.087,
            "command_threshold": 0.10,
            "max_air_time": 0.20,
            "min_contact": 2,
            "lift_penalty_scale": 20.0,
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "exclude_terrain_type_start": D1_STAIRS_TERRAIN_START,
        },
    )
    zero_command_base_motion_l2 = RewTerm(
        func=mdp.zero_command_base_motion_l2,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "yaw_scale": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    zero_command_wheel_vel_l2 = RewTerm(
        func=mdp.zero_command_wheel_vel_l2,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            # Include yaw so turning is not classified as standing.
            "planar_command_only": False,
            "upright_gate": False,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
        },
    )

    joint_default = RewTerm(
        func=mdp.default_joint_l2,
        weight=-1.0,
        params={
            "upright_gate": False,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
        },
    )

    hip_pos = RewTerm(
        func=mdp.hip_pos,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            "command_name": "base_velocity",
            "command_threshold": 0.10,
            "tolerance": 0.20,
            "loose_ratio": 1.0,
            "exclude_terrain_type_start": D1_PLATFORM_TERRAIN_START,
        },
    )


@configclass
class PlatformCostsCfg(CostsCfg):
    """One bounded safety constraint for supported platform traversal."""

    platform_safety = CostTermCfg(
        func=mdp.platform_safety_cost,
        scale=1.0,
        d_value=0.0,
        k_value=0.05,
        params=_platform_term_params(),
    )


@configclass
class PlatformTerminationsCfg(TerminationsCfg):
    """Allow aggressive edge contact, but reset once the robot overturns."""

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 2})


@configclass
class D1PlatformNP3OEnvCfg(D1RoughNP3OEnvCfg):
    """Platform curriculum with supported ascent and descent shaping.

    Actor observations and actions remain unchanged; the platform task adds
    one bounded NP3O safety cost and is intended for a fresh training run.
    """

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: PlatformCommandsCfg = PlatformCommandsCfg()
    rewards: PlatformRewardsCfg = PlatformRewardsCfg()
    costs: PlatformCostsCfg = PlatformCostsCfg()
    terminations: PlatformTerminationsCfg = PlatformTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.physx.gpu_collision_stack_size = 2**27

        # The rough-task override disables this term; the platform task keeps it on flat terrain.
        self.rewards.flat_orientation_l2.weight = -10.0

        # Keep a 10-sample (50 ms) contact window. The traversal state accepts
        # 8/10 valid samples while requiring the latest sample to be valid.
        self.scene.contact_forces.history_length = 10

        # The final 60% of columns are split between ascent and descent.
        self.scene.terrain.terrain_generator = D1_HIGH_PLATFORM_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 5

        # Slow descent commands use a reachable promotion distance.
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel,
            params={
                "move_up_terrain_type_start": D1_PLATFORM_DESCENT_TERRAIN_START,
                "move_up_distance_override": 3.0,
            },
        )
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # Do not let a temporary pitch erase tracking and contact feedback.
        for reward_name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "ang_vel_xy_l2",
            "action_rate_l2",
            "undesired_contacts",
            "feet_stumble",
        ):
            getattr(self.rewards, reward_name).params["upright_gate"] = False
        self.rewards.power_distribution_var = None
        self.rewards.joint_mirror = None
        self.rewards.gait_trot = None
        # Restore command-gated stepping terms changed by the rough base cfg.
        self.rewards.foot_clearance.weight = 0.50
        self.rewards.wheel_scrub_penalty.weight = -0.5
        # The upright bonus would discourage the required nose-up transition.
        self.rewards.upward = None

        # The inherited ``.*_base_link`` regex does not match D1's ``base_link``.
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.add_base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # A persistent external wrench masks platform-contact behavior.
        self.events.base_external_force_torque = None

        # Avoid vertical pushes that resemble platform-edge impacts.
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
        }

        # Keep enough tangential wheel force for supported climbing.
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.25)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.add_base_com.params["com_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.85, 1.15)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.85, 1.15)

        self.events.reset_base.params["pose_range"].update({
            "x": (-0.50, 0.50),
            "y": (-0.50, 0.50),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        })
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.05, 0.05),
        }

        if self.__class__.__name__ == "D1PlatformNP3OEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class D1PlatformNP3OEnvCfg_PLAY(D1PlatformNP3OEnvCfg):
    """Small evaluation variant of the stock D1 platform framework."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            # Preserve the deterministic 8/6/6 ordinary/ascent/descent layout.
            self.scene.terrain.terrain_generator.num_cols = 20
            self.scene.terrain.terrain_generator.curriculum = True
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        if self.__class__.__name__ == "D1PlatformNP3OEnvCfg_PLAY":
            self.disable_zero_weight_rewards()
