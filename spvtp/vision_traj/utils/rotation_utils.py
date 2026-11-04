"""SE(3) / 6D rotation helpers (Zhou et al. 2019 continuous representation)."""

import torch


def rotmat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6), taking first two columns."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt."""
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def se3_relative(T_from: torch.Tensor, T_to: torch.Tensor) -> torch.Tensor:
    """Returns T_from^{-1} @ T_to. Inputs (..., 4, 4)."""
    return torch.linalg.solve(T_from, T_to)


def decompose_se3(T: torch.Tensor):
    """(..., 4, 4) -> (pos(...,3), rot6d(...,6))."""
    pos = T[..., :3, 3]
    rot6d = rotmat_to_rot6d(T[..., :3, :3])
    return pos, rot6d


def compose_se3(pos: torch.Tensor, rot6d: torch.Tensor) -> torch.Tensor:
    """(pos(...,3), rot6d(...,6)) -> (..., 4, 4)."""
    R = rot6d_to_rotmat(rot6d)
    shape = pos.shape[:-1]
    T = torch.zeros(*shape, 4, 4, device=pos.device, dtype=pos.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T
