# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1 platform task using the D1Box robot assembly."""

from ddt_lab.assets.ddt_robot import DDT_D1BOX_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from ..d1.np3o.platform_env_cfg import (
    D1PlatformNP3OEnvCfg,
    D1PlatformNP3OEnvCfg_PLAY,
    PlatformSceneCfg,
)


@configclass
class D1BoxPlatformSceneCfg(PlatformSceneCfg):
    """Platform scene with the D1Box robot and wider body scanner."""

    robot: ArticulationCfg = DDT_D1BOX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    body_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(1.15, 0.23)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class D1BoxPlatformNP3OEnvCfg(D1PlatformNP3OEnvCfg):
    """D1Box platform training task."""

    scene: D1BoxPlatformSceneCfg = D1BoxPlatformSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.02
        self.disable_zero_weight_rewards()


@configclass
class D1BoxPlatformNP3OEnvCfg_PLAY(D1PlatformNP3OEnvCfg_PLAY):
    """D1Box platform evaluation task."""

    scene: D1BoxPlatformSceneCfg = D1BoxPlatformSceneCfg(num_envs=50, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.02
        self.disable_zero_weight_rewards()
