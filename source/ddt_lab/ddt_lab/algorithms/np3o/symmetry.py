# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left/right and fore/aft transforms for D1 policy observations and actions."""

# fmt: off
# 58-D height policy: ang_vel[0:3], gravity[3:6], velocity_cmd[6:9],
# height_cmd[9], joint_pos[10:26], joint_vel[26:42], last_action[42:58].
OBS_PERM_HEIGHT_LR = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    14, 15, 16, 17, 10, 11, 12, 13, 22, 23, 24, 25, 18, 19, 20, 21,
    30, 31, 32, 33, 26, 27, 28, 29, 38, 39, 40, 41, 34, 35, 36, 37,
    46, 47, 48, 49, 42, 43, 44, 45, 54, 55, 56, 57, 50, 51, 52, 53,
]
OBS_SIGN_HEIGHT_LR = [
    -1, 1, -1, 1, -1, 1, 1, -1, -1, 1,
    -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1,
    -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1,
    -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1,
]

# Each action leg block is [hip, thigh, calf, wheel].
ACT_PERM_LR = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
ACT_SIGN_LR = [-1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1]

OBS_PERM_HEIGHT_FA = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    18, 19, 20, 21, 22, 23, 24, 25, 10, 11, 12, 13, 14, 15, 16, 17,
    34, 35, 36, 37, 38, 39, 40, 41, 26, 27, 28, 29, 30, 31, 32, 33,
    50, 51, 52, 53, 54, 55, 56, 57, 42, 43, 44, 45, 46, 47, 48, 49,
]
OBS_SIGN_HEIGHT_FA = [-1, -1, 1, -1, -1, 1, -1, -1, 1, 1] + [1] * 48
ACT_PERM_FA = [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
ACT_SIGN_FA = [1] * 16
# fmt: on


def _remove_height_command(perm: list[int], sign: list[int]) -> tuple[list[int], list[int]]:
    height_index = 9
    velocity_perm = [index - int(index > height_index) for index in perm if index != height_index]
    velocity_sign = sign[:height_index] + sign[height_index + 1 :]
    return velocity_perm, velocity_sign


OBS_PERM_VELOCITY_LR, OBS_SIGN_VELOCITY_LR = _remove_height_command(OBS_PERM_HEIGHT_LR, OBS_SIGN_HEIGHT_LR)
OBS_PERM_VELOCITY_FA, OBS_SIGN_VELOCITY_FA = _remove_height_command(OBS_PERM_HEIGHT_FA, OBS_SIGN_HEIGHT_FA)


def _validate_involution(perm: list[int], sign: list[int]) -> None:
    if len(perm) != len(sign):
        raise ValueError("symmetry permutation and sign lengths differ")
    if sorted(perm) != list(range(len(perm))):
        raise ValueError("symmetry permutation is not a bijection")
    if any(value not in (-1, 1) for value in sign):
        raise ValueError("symmetry signs must be -1 or 1")
    if any(perm[perm[index]] != index or sign[index] * sign[perm[index]] != 1 for index in range(len(perm))):
        raise ValueError("symmetry transform is not an involution")


def _build(device, obs_perm, obs_sign, action_perm, action_sign):
    import torch

    _validate_involution(obs_perm, obs_sign)
    _validate_involution(action_perm, action_sign)
    return (
        torch.tensor(obs_perm, dtype=torch.long, device=device),
        torch.tensor(obs_sign, dtype=torch.float, device=device),
        torch.tensor(action_perm, dtype=torch.long, device=device),
        torch.tensor(action_sign, dtype=torch.float, device=device),
    )


def build_d1_mirror(device, obs_dim: int = 58):
    """Build the D1 left/right observation and action transform."""
    if obs_dim == 58:
        obs_perm, obs_sign = OBS_PERM_HEIGHT_LR, OBS_SIGN_HEIGHT_LR
    elif obs_dim == 57:
        obs_perm, obs_sign = OBS_PERM_VELOCITY_LR, OBS_SIGN_VELOCITY_LR
    else:
        raise ValueError(f"D1 symmetry requires 57 or 58 policy features, received {obs_dim}")
    return _build(device, obs_perm, obs_sign, ACT_PERM_LR, ACT_SIGN_LR)


def build_d1_mirror_fa(device, obs_dim: int = 58):
    """Build the D1 fore/aft observation and action transform."""
    if obs_dim == 58:
        obs_perm, obs_sign = OBS_PERM_HEIGHT_FA, OBS_SIGN_HEIGHT_FA
    elif obs_dim == 57:
        obs_perm, obs_sign = OBS_PERM_VELOCITY_FA, OBS_SIGN_VELOCITY_FA
    else:
        raise ValueError(f"D1 symmetry requires 57 or 58 policy features, received {obs_dim}")
    return _build(device, obs_perm, obs_sign, ACT_PERM_FA, ACT_SIGN_FA)
