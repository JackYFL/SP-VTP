from pathlib import Path

import torch

from spvtp.vision_traj.utils.viz import save_trajectory_comparison


def test_save_trajectory_comparison(tmp_path: Path) -> None:
    first_frame = torch.rand(3, 224, 224)
    current_frame = torch.rand(3, 224, 224)
    prompt_object = torch.tensor([0.1, 0.2, 0.35, 0.5])
    prompt_target = torch.tensor([0.55, 0.45, 0.8, 0.9])

    pred = torch.zeros(16, 10)
    gt = torch.zeros(16, 10)
    pred[:, 0] = torch.linspace(0.0, 0.15, 16)
    pred[:, 1] = torch.linspace(0.0, -0.05, 16)
    pred[:, 3] = 1.0
    pred[:, 7] = 1.0
    gt[:, 0] = torch.linspace(0.0, 0.12, 16)
    gt[:, 2] = torch.linspace(0.0, 0.08, 16)
    gt[:, 3] = 1.0
    gt[:, 7] = 1.0

    out = tmp_path / "traj.png"
    save_trajectory_comparison(
        out,
        first_frame=first_frame,
        current_frame=current_frame,
        prompt_object=prompt_object,
        prompt_target=prompt_target,
        prompt_form="bbox",
        pred_actions=pred,
        gt_actions=gt,
        title="test",
        subtitle="episode=test",
    )

    assert out.exists()
    assert out.stat().st_size > 0
