# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for D1Box platform tasks."""

import gymnasium as gym

from ..d1.np3o.platform_env_cfg import D1PlatformRLEnv
from . import agents
from .platform_env_cfg import D1BoxPlatformNP3OEnvCfg, D1BoxPlatformNP3OEnvCfg_PLAY

gym.register(
    id="DDT-Velocity-Platform-D1Box-NP3O-v0",
    entry_point=D1PlatformRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": D1BoxPlatformNP3OEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1box_platform_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Platform-D1Box-NP3O-Play-v0",
    entry_point=D1PlatformRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": D1BoxPlatformNP3OEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1box_platform_np3o_runner_cfg",
    },
)
