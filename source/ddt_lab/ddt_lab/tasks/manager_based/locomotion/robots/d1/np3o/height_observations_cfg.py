# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation contract shared by D1 commanded-height NP3O tasks."""

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .rough_env_cfg import PrivilegedObservationsCfg

HEIGHT_POLICY_OBSERVATION_DIM = 58
HEIGHT_SCALE = 1.0

# Joint order is part of the exported 58-D policy ABI.
LEG_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_foot_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_foot_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RL_foot_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RR_foot_joint",
)


@configclass
class HeightPolicyCfg(ObsGroup):
    """58-D actor observation used by training and deployment export."""

    base_ang_vel = ObsTerm(
        func=mdp.imu_ang_vel,
        params={"asset_cfg": SceneEntityCfg("imu")},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        clip=(-100.0, 100.0),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=mdp.imu_projected_gravity,
        params={"asset_cfg": SceneEntityCfg("imu")},
        noise=Unoise(n_min=-0.05, n_max=0.05),
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=(2.0, 2.0, 0.25),
    )
    height_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_height"},
        clip=(-100.0, 100.0),
        scale=HEIGHT_SCALE,
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel_without_wheel,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_ORDER), preserve_order=True),
            "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
        },
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_ORDER), preserve_order=True)},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        clip=(-100.0, 100.0),
        scale=0.05,
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class HeightCriticCfg(ObsGroup):
    """Critic observation with privileged base velocity and height command."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=(2.0, 2.0, 0.25),
    )
    height_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_height"},
        clip=(-100.0, 100.0),
        scale=HEIGHT_SCALE,
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel_without_wheel,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_ORDER), preserve_order=True),
            "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(LEG_ORDER), preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=0.05,
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)


@configclass
class HeightPrivilegedObservationsCfg(PrivilegedObservationsCfg):
    """Height-policy actor, critic, and critic-only terrain observations."""

    policy: HeightPolicyCfg = HeightPolicyCfg()
    critic: HeightCriticCfg = HeightCriticCfg()
    # Height scans remain critic-only so the actor/export ABI stays 58-D.
    scanner: PrivilegedObservationsCfg.ScannerCfg | None = PrivilegedObservationsCfg.ScannerCfg()
