# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left/right transforms for D1 policy observations and actions."""

# fmt: off
# Observation layout: IMU(6), velocity(3), height(1), joints(32), actions(16).
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

# Each leg contributes [hip, thigh, calf, wheel] in FL, FR, RL, RR order.
ACT_PERM_LR = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
ACT_SIGN_LR = [-1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1]

# fmt: on


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


def build_d1_mirror(device):
    """Build the D1 left/right observation and action transform.

    Args:
        device: Torch device used for the transform tensors.

    Returns:
        Observation and action permutations followed by their sign tensors.
    """
    return _build(device, OBS_PERM_HEIGHT_LR, OBS_SIGN_HEIGHT_LR, ACT_PERM_LR, ACT_SIGN_LR)
