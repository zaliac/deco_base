"""Small, dependency-free rotation conversions used by dataset preprocessors."""

from __future__ import annotations

import numpy as np
import torch


def batch_rodrigues(axis_angle: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Convert ``(..., 3)`` axis-angle rotations to ``(..., 3, 3)`` matrices."""
    if not torch.is_tensor(axis_angle):
        raise TypeError("axis_angle must be a torch.Tensor")
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected axis_angle[..., 3], got {tuple(axis_angle.shape)}")
    if not axis_angle.is_floating_point():
        axis_angle = axis_angle.float()

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(epsilon)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(axis.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(skew)
    angle = angle[..., None]
    return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def batch_rot2aa(rotation_matrices: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Convert ``(..., 3, 3)`` rotation matrices to axis-angle vectors.

    Quaternion extraction is stable both near the identity and near a 180-degree
    rotation, cases where the common trace-only formula is numerically fragile.
    """
    if not torch.is_tensor(rotation_matrices):
        raise TypeError("rotation_matrices must be a torch.Tensor")
    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected rotation_matrices[..., 3, 3], got "
            f"{tuple(rotation_matrices.shape)}"
        )
    matrix = rotation_matrices.float() if not rotation_matrices.is_floating_point() else rotation_matrices
    flat = matrix.reshape(-1, 3, 3)
    result = torch.empty((flat.shape[0], 4), dtype=matrix.dtype, device=matrix.device)
    trace = flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2]

    positive = trace > 0
    scale = torch.sqrt((trace[positive] + 1.0).clamp_min(epsilon)) * 2.0
    result[positive, 0] = 0.25 * scale
    result[positive, 1] = (flat[positive, 2, 1] - flat[positive, 1, 2]) / scale
    result[positive, 2] = (flat[positive, 0, 2] - flat[positive, 2, 0]) / scale
    result[positive, 3] = (flat[positive, 1, 0] - flat[positive, 0, 1]) / scale

    for diagonal in range(3):
        choose = (~positive) & (flat[:, diagonal, diagonal] >= flat[:, (diagonal + 1) % 3, (diagonal + 1) % 3]) & (flat[:, diagonal, diagonal] >= flat[:, (diagonal + 2) % 3, (diagonal + 2) % 3])
        if not choose.any():
            continue
        i, j, k = diagonal, (diagonal + 1) % 3, (diagonal + 2) % 3
        scale = torch.sqrt((1.0 + flat[choose, i, i] - flat[choose, j, j] - flat[choose, k, k]).clamp_min(epsilon)) * 2.0
        result[choose, 0] = (flat[choose, k, j] - flat[choose, j, k]) / scale
        result[choose, i + 1] = 0.25 * scale
        result[choose, j + 1] = (flat[choose, j, i] + flat[choose, i, j]) / scale
        result[choose, k + 1] = (flat[choose, k, i] + flat[choose, i, k]) / scale

    result = result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(epsilon)
    # q and -q represent the same rotation.  Choosing w >= 0 keeps the angle in
    # [0, pi], the conventional axis-angle representation.
    result = torch.where(result[:, :1] < 0, -result, result)
    vector = result[:, 1:]
    sin_half = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, result[:, :1])
    axis_angle = vector * (angle / sin_half.clamp_min(epsilon))
    axis_angle = torch.where(sin_half < epsilon, 2.0 * vector, axis_angle)
    return axis_angle.reshape(matrix.shape[:-2] + (3,))


def ea2rm(euler_angles, *, degrees: bool = False):
    """Convert XYZ Euler angles to matrices (active rotations, ``Rz @ Ry @ Rx``).

    Accepts either a torch tensor or a NumPy array and returns the same type.
    Angles are radians unless ``degrees=True``.
    """
    is_numpy = isinstance(euler_angles, np.ndarray)
    angles = torch.as_tensor(euler_angles)
    if angles.shape[-1] != 3:
        raise ValueError(f"Expected euler_angles[..., 3], got {tuple(angles.shape)}")
    if not angles.is_floating_point():
        angles = angles.float()
    if degrees:
        angles = torch.deg2rad(angles)
    x, y, z = angles.unbind(dim=-1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    rows = (
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx,
    )
    matrix = torch.stack(rows, dim=-1).reshape(angles.shape[:-1] + (3, 3))
    return matrix.cpu().numpy() if is_numpy else matrix
