# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""All-terrain commanded-height NP3O task for the stock D1 robot."""

import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
import isaaclab.terrains as terrain_gen
from ddt_lab.assets.ddt_robot import DDT_D1_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ImuCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from ..base_env_cfg import CommandsCfg, SceneCfg
from .height_env_cfg import HEIGHT_RANGE
from .height_observations_cfg import HeightPrivilegedObservationsCfg
from .rough_env_cfg import D1RoughNP3OEnvCfg


@configclass
class D1AllTerrainSceneCfg(SceneCfg):
    """Stock D1 scene with base-height and body-footprint terrain scanners."""

    robot: ArticulationCfg = DDT_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=ImuCfg.OffsetCfg(rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=False,
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    body_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        # The footprint reaches beyond both wheel axles before a riser contacts the chassis.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.90, 0.23)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class AllTerrainCommandsCfg(CommandsCfg):
    """Velocity commands with explicit forward/reverse pure-X coverage."""

    base_velocity = mdp.ModeBalancedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.15,
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        rel_pure_x_envs=0.30,
        pure_x_min_speed=0.15,
    )


@configclass
class D1HeightAllTerrainNP3OEnvCfg(D1RoughNP3OEnvCfg):
    """Variable-height locomotion over flat, rough, obstacle, and stair terrain."""

    scene: D1AllTerrainSceneCfg = D1AllTerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: AllTerrainCommandsCfg = AllTerrainCommandsCfg()
    observations: HeightPrivilegedObservationsCfg = HeightPrivilegedObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self._configure_robot_and_sensors()
        self._configure_commands_and_observations()
        self._configure_events()
        self._configure_rewards()
        self._configure_curricula_and_terrain()

    def _configure_robot_and_sensors(self):
        self.scene.robot = DDT_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        policy_period = self.decimation * self.sim.dt
        self.scene.height_scanner.update_period = policy_period
        self.scene.height_scanner_base.update_period = policy_period
        self.scene.body_scanner.update_period = policy_period
        # Ten physics samples retain short landing-force peaks between policy steps.
        self.scene.contact_forces.history_length = 10

    def _configure_commands_and_observations(self):
        self.commands.base_height = mdp.UniformHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            ranges=mdp.UniformHeightCommandCfg.Ranges(height=(0.17, HEIGHT_RANGE[1])),
        )
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = False

    def _configure_events(self):
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 6.0)
        self.events.add_base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.push_robot.interval_range_s = (4.0, 6.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (-0.5, 0.5),
            "roll": (-1.0, 1.0),
            "pitch": (-1.0, 1.0),
            "yaw": (-0.5, 0.5),
        }

    def _configure_rewards(self):
        rewards = self.rewards
        for name in (
            "flat_orientation_l2",
            "base_height_l2",
            "joint_torques_l2",
            "joint_vel_l2",
            "joint_pos_limits",
            "joint_vel_limits",
            "joint_power",
            "power_distribution_var",
            "contact_forces",
            "upward",
            "default_joint_l2",
            "gait_trot",
            "joint_mirror",
        ):
            setattr(rewards, name, None)

        rewards.track_lin_vel_xy_exp.weight = 3.5
        rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        rewards.track_ang_vel_z_exp.weight = 1.0
        rewards.track_ang_vel_z_exp.params["std"] = 0.5
        rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
        rewards.ang_vel_xy_l2.weight = -0.20
        rewards.joint_acc_l2.weight = -2.5e-7
        rewards.action_rate_l2.weight = -0.03

        rewards.track_base_height_exp = RewTerm(
            func=mdp.terrain_track_base_height_exp,
            weight=5.0,
            params={
                "command_name": "base_height",
                "std": 0.05,
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
                "plane_residual_sensor_cfg": SceneEntityCfg("height_scanner"),
                "plane_residual_gate_std": 0.03,
            },
        )
        rewards.terrain_or_world_orientation_l2 = None
        rewards.flat_orientation_l2 = RewTerm(
            func=mdp.smooth_ground_flat_orientation_l2,
            weight=-16.0,
            params={
                "sensor_cfg": SceneEntityCfg("body_scanner"),
                "height_spread_std": 0.02,
            },
        )
        rewards.hip_pos = None
        rewards.hip_default = RewTerm(
            func=mdp.default_joint_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
        )
        rewards.flat_yaw_hip_pos = RewTerm(
            func=mdp.flat_yaw_hip_pos_l2,
            weight=-15.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
                "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
                "command_name": "base_velocity",
                "yaw_threshold": 0.10,
                "tolerance": 0.18,
                "terrain_slope_gate_std": 0.05,
                "plane_residual_gate_std": 0.015,
            },
        )
        rewards.wheel_thigh_x_alignment = RewTerm(
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
                "plane_residual_gate_std": 0.015,
            },
        )
        rewards.wheel_scrub_penalty = RewTerm(
            func=mdp.wheel_scrub_penalty,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "command_threshold": 0.10,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        rewards.foot_clearance = RewTerm(
            func=mdp.height_foot_clearance,
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
                "upright_gate": False,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        rewards.contact_forces = RewTerm(
            func=mdp.landing_force_penalty,
            weight=-2.0e-3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 450.0,
            },
        )
        rewards.feet_stumble = RewTerm(
            func=mdp.height_feet_stumble,
            weight=-2.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "force_ratio": 2.0,
            },
        )
        rewards.undesired_contacts.weight = -1.0
        rewards.undesired_contacts.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!base_link$)(?!.*_foot$).*"]),
            "threshold": 1.0,
        }
        rewards.belly_contact = RewTerm(
            func=mdp.undesired_contacts,
            weight=-8.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
                "threshold": 1.0,
            },
        )
        rewards.body_obstacle_clearance = RewTerm(
            func=mdp.body_obstacle_clearance,
            weight=-4.0,
            params={
                "sensor_cfg": SceneEntityCfg("body_scanner"),
                "body_bottom_offset": 0.149,
                "safe_clearance": 0.01,
                "std": 0.02,
            },
        )
        rewards.stand_still = RewTerm(
            func=mdp.stand_still,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            },
        )
        rewards.zero_command_base_motion_l2 = RewTerm(
            func=mdp.zero_command_base_motion_l2,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "yaw_scale": 0.25,
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            },
        )
        rewards.zero_command_wheel_vel_l2 = RewTerm(
            func=mdp.zero_command_wheel_vel_l2,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
        )

    def _configure_curricula_and_terrain(self):
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.base_height_cmd = CurrTerm(
            func=mdp.base_height_command_curriculum,
            params={
                "command_name": "base_height",
                "start_floor": 0.35,
                "end_floor": 0.17,
                "full_steps": 144000,
            },
        )

        terrain_generator = self.scene.terrain.terrain_generator
        terrain_generator.num_rows = 16
        terrain_generator.sub_terrains = {
            "flat_grass": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
            "gravel": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.15,
                noise_range=(-0.03, 0.03),
                noise_step=0.01,
                downsampled_scale=0.20,
                border_width=0.25,
            ),
            "grass_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
            "grass_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.0, 0.4),
                platform_width=2.0,
                border_width=0.25,
            ),
            "brick_rubble": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                obstacle_width_range=(0.20, 0.80),
                obstacle_height_range=(0.015, 0.06),
                obstacle_height_mode="fixed",
                num_obstacles=48,
                platform_width=2.0,
                border_width=0.25,
            ),
            "ledges_10cm": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                obstacle_width_range=(0.40, 1.60),
                obstacle_height_range=(0.04, 0.10),
                obstacle_height_mode="fixed",
                num_obstacles=14,
                platform_width=2.0,
                border_width=0.25,
            ),
            "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.05, 0.17),
                step_width=0.30,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.05, 0.17),
                step_width=0.30,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        }
        self.scene.terrain.max_init_terrain_level = 2


def _configure_play(cfg):
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.scene.terrain.max_init_terrain_level = None
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.events.add_base_com = None
    cfg.events.randomize_actuator_gains = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.command_levels_lin_vel = None
    cfg.curriculum.base_height_cmd = None


@configclass
class D1HeightAllTerrainNP3OEnvCfg_PLAY(D1HeightAllTerrainNP3OEnvCfg):
    """Small deterministic evaluation variant of the all-terrain task."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)
