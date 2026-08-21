# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O runner configuration for the D1Box platform task."""

from ddt_lab.tasks.manager_based.locomotion.robots.d1.np3o.agents.np3o_cfg import (
    d1_platform_np3o_runner_cfg,
)


def d1box_platform_np3o_runner_cfg() -> dict:
    """Use a separate experiment namespace for D1Box checkpoints."""
    cfg = d1_platform_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1box_platform"
    return cfg
