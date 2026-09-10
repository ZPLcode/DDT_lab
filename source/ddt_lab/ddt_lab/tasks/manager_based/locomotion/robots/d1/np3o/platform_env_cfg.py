# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1 NP3O platform ascent and descent task."""

import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
import torch
from ddt_lab.assets.terrains.platform import (
    D1_FLAT_X_TERRAIN_END,
    D1_FLAT_X_TERRAIN_START,
    D1_PLATFORM_DESCENT_TERRAIN_START,
    D1_PLATFORM_TERRAIN_START,
    D1_PLATFORM_TERRAINS_CFG,
)
from ddt_lab.managers import CostTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from ..base_env_cfg import CommandsCfg, SceneCfg, TerminationsCfg
from .rough_env_cfg import CostsCfg, D1RoughNP3OEnvCfg, RoughRewardsCfg


@configclass
class PlatformSceneCfg(SceneCfg):
    """Platform scene with a reward-only chassis footprint scanner."""

    height_scanner_base = None
    body_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.75, 0.23)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


class D1PlatformRLEnv(ManagerBasedRLEnv):
    """Clip ordinary rewards before adding traversal, safety and termination terms."""

    _UNCLIPPED_REWARD_TERMS = (
        "contact_forces",
        "descent_front_impact",
        "ascent_rear_lateral_force",
        "descent_axle_support",
        "ascent_rear_step",
    )

    def step(self, action: torch.Tensor):
        observations, reward, terminated, time_out, extras = super().step(action)
        term_indices = getattr(self, "_unclipped_reward_term_indices", None)
        if term_indices is None:
            active_terms = self.reward_manager.active_terms
            missing_terms = [name for name in self._UNCLIPPED_REWARD_TERMS if name not in active_terms]
            if missing_terms:
                raise RuntimeError(f"Missing unclipped platform reward terms: {missing_terms}")
            term_indices = [active_terms.index(name) for name in self._UNCLIPPED_REWARD_TERMS]
            self._unclipped_reward_term_indices = term_indices

        # RewardManager stores per-term weighted rewards before dt scaling.
        unclipped_reward = self.reward_manager._step_reward[:, term_indices].sum(dim=1) * self.step_dt
        termination_cfg = self.reward_manager.get_term_cfg("is_terminated")
        termination_reward = terminated.float() * termination_cfg.weight * self.step_dt
        regular_reward = reward - termination_reward - unclipped_reward
        reward = torch.clamp(regular_reward, min=0.0) + unclipped_reward + termination_reward
        self.reward_buf = reward
        return observations, reward, terminated, time_out, extras


@configclass
class PlatformCommandsCfg(CommandsCfg):
    """Velocity commands for ordinary, ascent, and descent terrain columns."""

    base_velocity = mdp.TerrainAwareVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.15,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.2),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        platform_terrain_start=D1_PLATFORM_TERRAIN_START,
        planar_deadzone=0.2,
        flat_x_terrain_start=D1_FLAT_X_TERRAIN_START,
        flat_x_terrain_end=D1_FLAT_X_TERRAIN_END,
        flat_x_min_speed=0.20,
        platform_ranges=mdp.TerrainAwareVelocityCommandCfg.PlatformRanges(
            lin_vel_x=(0.20, 0.60),
            lin_vel_y=(0.0, 0.0),
            heading=(-math.radians(10.0), math.radians(10.0)),
        ),
        platform_descent_terrain_start=D1_PLATFORM_DESCENT_TERRAIN_START,
        platform_descent_ranges=mdp.TerrainAwareVelocityCommandCfg.PlatformRanges(
            lin_vel_x=(0.20, 0.32),
            lin_vel_y=(0.0, 0.0),
            heading=(-math.radians(10.0), math.radians(10.0)),
        ),
        platform_descent_bidirectional=True,
    )


@configclass
class PlatformRewardsCfg(RoughRewardsCfg):
    """Reward terms for command-conditioned platform locomotion."""

    is_terminated = RewTerm(func=mdp.is_terminated, weight=-0.8)
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
        weight=-4.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 400.0,
        },
    )
    descent_front_impact = RewTerm(
        func=mdp.platform_descent_front_impact_penalty,
        weight=-8.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "threshold": 400.0,
            "terrain_type_start": D1_PLATFORM_DESCENT_TERRAIN_START,
            "command_threshold": 0.10,
        },
    )
    ascent_rear_lateral_force = RewTerm(
        func=mdp.platform_ascent_rear_lateral_force_penalty,
        weight=-4.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "threshold": 200.0,
            "terrain_type_start": D1_PLATFORM_TERRAIN_START,
            "terrain_type_end": D1_PLATFORM_DESCENT_TERRAIN_START,
        },
    )
    ascent_diagnostics = RewTerm(
        func=mdp.platform_ascent_diagnostics,
        weight=1.0,
        params={
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
            "terrain_type_end": D1_PLATFORM_DESCENT_TERRAIN_START,
            "wheel_radius": 0.087,
            "lift_height": 0.04,
            "surface_tolerance": 0.04,
            "support_force_threshold": 10.0,
            "top_horizontal_force_ratio": 2.0,
            "support_history_steps": 10,
            "stable_history_fraction": 0.8,
            "plane_residual_gate_std": 0.015,
            "transition_gate_threshold": 0.10,
        },
    )
    ascent_rear_step = RewTerm(
        func=mdp.RearWheelStepReward,
        weight=1.0,
        params={
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "wheel_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "wheel_asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "terrain_type_start": D1_PLATFORM_TERRAIN_START,
            "terrain_type_end": D1_PLATFORM_DESCENT_TERRAIN_START,
            "clearance_margin": 0.06,
            "clearance_bonus": 0.25,
            "landing_bonus": 1.0,
            "wall_force_threshold": 20.0,
            "wall_penalty_scale": 0.5,
            "approach_distance": 0.30,
            "approach_penalty_scale": 2.0,
        },
    )
    descent_axle_support = RewTerm(
        func=mdp.platform_descent_axle_support_penalty,
        weight=-0.1,
        params={
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "wheel_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "terrain_type_start": D1_PLATFORM_DESCENT_TERRAIN_START,
            "support_force_threshold": 10.0,
            "top_horizontal_force_ratio": 2.0,
            "support_history_steps": 10,
            "stable_history_fraction": 0.8,
            "plane_residual_gate_std": 0.015,
            "transition_gate_threshold": 0.10,
        },
    )
    lin_vel_z_l2 = RewTerm(
        func=mdp.discontinuity_gated_lin_vel_z_l2,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("body_scanner"),
            "discontinuity_height": 0.06,
            "discontinuity_std": 0.015,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_height_l2 = RewTerm(
        func=mdp.terrain_gated_base_height_l2,
        weight=-10.0,
        params={
            "target_height": 0.52,
            "sensor_cfg": SceneEntityCfg("body_scanner"),
            "plane_residual_gate_std": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    terrain_or_world_orientation_l2 = RewTerm(
        func=mdp.terrain_or_world_orientation_l2,
        weight=-8.0,
        params={
            "sensor_cfg": SceneEntityCfg("body_scanner"),
            "roughness_std": 0.015,
            "discontinuity_height": 0.06,
            "discontinuity_std": 0.015,
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
            "hover_penalty_scale": 2.0,
            "min_contact": 2,
            "lift_penalty_scale": 100.0,
            "target_centered": True,
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "plane_residual_gate_std": 0.015,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "upright_gate": False,
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
            "planar_command_only": False,
            "upright_gate": False,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
        },
    )
    hip_default = RewTerm(
        func=mdp.default_joint_l2,
        weight=-1.0,
        params={
            "upright_gate": False,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )
    flat_yaw_hip_pos = RewTerm(
        func=mdp.flat_yaw_hip_pos_l2,
        weight=-15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "command_name": "base_velocity",
            "yaw_threshold": 0.10,
            "tolerance": 0.20,
            "terrain_slope_gate_std": 0.05,
            "plane_residual_gate_std": 0.015,
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
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "terrain_slope_gate_std": 0.05,
            "plane_residual_gate_std": 0.015,
            "upright_gate": True,
        },
    )
    run_still = RewTerm(
        func=mdp.run_still,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "lateral_or_rot_threshold": 0.1,
            "terrain_sensor_cfg": SceneEntityCfg("body_scanner"),
            "plane_residual_gate_std": 0.03,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
        },
    )


@configclass
class PlatformCostsCfg(CostsCfg):
    """Generic joint-limit costs used by the platform task."""

    joint_pos_limit = CostTermCfg(
        func=mdp.joint_pos_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_vel_limit = CostTermCfg(
        func=mdp.joint_vel_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_torque_limit = CostTermCfg(
        func=mdp.joint_torque_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "safe_limit": 60.0,
        },
    )


@configclass
class PlatformTerminationsCfg(TerminationsCfg):
    """Terminate only after the robot overturns."""

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.pi / 2})


@configclass
class D1PlatformNP3OEnvCfg(D1RoughNP3OEnvCfg):
    """Platform curriculum driven by velocity-command tracking."""

    scene: PlatformSceneCfg = PlatformSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: PlatformCommandsCfg = PlatformCommandsCfg()
    rewards: PlatformRewardsCfg = PlatformRewardsCfg()
    costs: PlatformCostsCfg = PlatformCostsCfg()
    terminations: PlatformTerminationsCfg = PlatformTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.sim.physx.gpu_collision_stack_size = 2**27
        legs = self.scene.robot.actuators["legs"]
        legs.effort_limit = 90.0
        legs.stiffness = 60.0
        legs.damping = 2.0

        self.scene.body_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.history_length = 10
        self.scene.terrain.terrain_generator = D1_PLATFORM_TERRAINS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 5

        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel,
            params={
                "move_up_terrain_type_start": D1_PLATFORM_DESCENT_TERRAIN_START,
                "move_up_distance_override": 3.0,
                "clean_ascent_reward_term": "ascent_rear_step",
                "clean_ascent_terrain_range": (D1_PLATFORM_TERRAIN_START, D1_PLATFORM_DESCENT_TERRAIN_START),
            },
        )
        self.curriculum.highplatform_levels = CurrTerm(
            func=mdp.terrain_level_statistics,
            params={
                "terrain_type_groups": {
                    "up": ("highplatform_up",),
                    "down": ("highplatform_down",),
                },
                "statistic_names": ("mean",),
            },
        )
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.flat_orientation_l2 = None
        self.rewards.action_rate_l2.weight = -0.02
        for name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "base_height_l2",
            "action_rate_l2",
            "undesired_contacts",
            "feet_stumble",
        ):
            getattr(self.rewards, name).params["upright_gate"] = False
        self.rewards.power_distribution_var = None
        self.rewards.hip_pos = None
        self.rewards.joint_mirror = None
        self.rewards.gait_trot = None
        self.rewards.foot_clearance.weight = 1.50
        self.rewards.wheel_scrub_penalty.weight = -0.5
        self.rewards.upward = None

        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.add_base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque = None
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
        }
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
        self.disable_zero_weight_rewards()


@configclass
class D1PlatformNP3OEnvCfg_PLAY(D1PlatformNP3OEnvCfg):
    """Small deterministic evaluation variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 20
            self.scene.terrain.terrain_generator.curriculum = True
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
