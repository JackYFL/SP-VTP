"""Fine-tune and evaluate GR00T N1.6 on the EgoSPT trajectory protocol.

This is an external-policy baseline, not a component transplanted into SP-VTP.
It keeps the paper's task split, 16 x 10-D relative action target, and four-step
history. Two task-specification protocols are supported:

* ``visual_prompt`` (legacy): a rendered first-frame prompt, current RGB, and a
  generic instruction; and
* ``language_only``: current RGB and the task-specific instruction
  ``pick {object} and place to {target}``, with no first frame or box input.

For scene3, the anonymous task folders are converted to deterministic local
object/target IDs offline. The annotated boxes are used only to construct this
language map; neither rendered boxes nor box coordinates enter the model.

Run this module with NVIDIA's official Isaac-GR00T N1.6 source tree on
``PYTHONPATH``.  The checkpoint's native 50-step/128-D output head is retained;
the official processor pads and masks our 16-step/10-D supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from spvtp.vision_traj.data.dataset import (
    EpisodeRecord,
    VisionTrajDataset,
    build_episode_index,
    filter_episodes_by_scenes,
    split_episodes,
    worker_init_fn,
)


INSTRUCTION = "Move the object highlighted in green to the target highlighted in red."
VISUAL_PROMPT_PROTOCOL = "visual_prompt"
LANGUAGE_ONLY_PROTOCOL = "language_only"


class GrootEgoSPTDataset(Dataset):
    """Expose :class:`VisionTrajDataset` with stable external-policy keys."""

    def __init__(
        self,
        base: VisionTrajDataset,
        prompt_protocol: str = VISUAL_PROMPT_PROTOCOL,
        language_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.base = base
        self.prompt_protocol = prompt_protocol
        self.language_map = language_map or {}
        if prompt_protocol not in (VISUAL_PROMPT_PROTOCOL, LANGUAGE_ONLY_PROTOCOL):
            raise ValueError(f"unknown prompt protocol: {prompt_protocol!r}")
        if prompt_protocol == LANGUAGE_ONLY_PROTOCOL:
            missing = [episode.key for episode in base.episodes if episode.key not in self.language_map]
            if missing:
                raise KeyError(
                    f"language map is missing {len(missing)} dataset episodes; first={missing[0]!r}"
                )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        episode_index, _ = self.base._samples[index]
        episode = self.base.episodes[episode_index]
        task = (
            INSTRUCTION
            if self.prompt_protocol == VISUAL_PROMPT_PROTOCOL
            else self.language_map[episode.key]["instruction"]
        )
        result = {
            "observation.images.current_frame": sample["current_frame"],
            "observation.state": sample["action_history"].reshape(-1),
            "action": sample["future_actions"],
            "task": task,
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.prompt_protocol == VISUAL_PROMPT_PROTOCOL:
            result["observation.images.first_frame"] = sample["first_frame"]
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="spvtp/EgoSPT")
    parser.add_argument("--annotations", default="annotations_merged.json")
    parser.add_argument(
        "--frame-cache-root",
        default=None,
        help="Optional node-local predecoded frame cache used by VisionTrajDataset.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompt-protocol",
        choices=(VISUAL_PROMPT_PROTOCOL, LANGUAGE_ONLY_PROTOCOL),
        default=VISUAL_PROMPT_PROTOCOL,
        help="Visual boxes in the first frame, or current RGB plus episode language only.",
    )
    parser.add_argument(
        "--language-map",
        default=None,
        help="Episode-level map from build_language_prompt_map.py; required for language_only.",
    )
    parser.add_argument(
        "--split-manifest",
        default=(
            "runs/vision_traj_experiments_final/"
            "baseline_vision_bbox_and_bbox_all_scenes/split_manifest.json"
        ),
        help="Reuse the exact completed-run split and avoid a redundant zarr scan.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/vision_traj_experiments_final/baseline_groot_n1_6_all_scenes",
    )
    parser.add_argument("--scenes", default="scene1,scene2,scene3")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--history-horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional staged-pilot cap; zero runs the complete epoch schedule.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-max-batches", type=int, default=64)
    parser.add_argument(
        "--final-eval-max-batches",
        type=int,
        default=0,
        help="Optional cap for pilot final evaluation; zero evaluates the full validation split.",
    )
    parser.add_argument("--stats-samples", type=int, default=4096)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--eval-checkpoint",
        default=None,
        help="Load a saved model.safetensors checkpoint before evaluation.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-step", type=int, default=0)
    return parser.parse_args()


def _trajectory_arrays(dataset, index):
    history, future = dataset.action_arrays(index)
    return history.reshape(-1), future


def _array_stats(array: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).clip(min=1.0e-8).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def compute_min_max_stats(dataset: VisionTrajDataset, n_samples: int, seed: int) -> dict:
    """Compute the nested statistics schema consumed by official GR00T N1.6."""
    rng = np.random.default_rng(seed)
    count = min(n_samples, len(dataset))
    indices = rng.choice(len(dataset), size=count, replace=False)
    states, actions = [], []
    for index in indices:
        state, action = _trajectory_arrays(dataset, int(index))
        states.append(state)
        actions.append(action)
    state_array = np.stack(states)
    action_array = np.concatenate(actions, axis=0)
    return {
        "new_embodiment": {
            "state": {"history": _array_stats(state_array)},
            "action": {"trajectory": _array_stats(action_array)},
        }
    }


def _serializable_stats(stats: dict) -> dict:
    return stats


def episodes_from_split_manifest(
    manifest_path: str | os.PathLike,
    annotations_path: str | os.PathLike,
    data_root: str | os.PathLike,
    split_seed: int,
) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
    """Rebuild episode records from a prior split without opening every zarr twice."""
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if int(manifest["seed"]) != split_seed:
        raise ValueError(
            f"split manifest seed {manifest['seed']} does not match requested {split_seed}"
        )
    with open(annotations_path) as handle:
        raw_annotations = json.load(handle)
    annotations = {key.replace("\\", "/"): value for key, value in raw_annotations.items()}
    root = Path(data_root)

    def convert(rows: list[dict]) -> list[EpisodeRecord]:
        records = []
        for row in rows:
            key = row["episode_key"].replace("\\", "/")
            parts = key.split("/")
            scene, task, episode = parts[0], parts[1], parts[2]
            annotation = annotations[key]["0"]
            episode_root = root / scene / task / "recording_output_processed" / episode
            records.append(
                EpisodeRecord(
                    key=key,
                    video_path=episode_root / "camera_1.mp4",
                    zarr_root=episode_root,
                    object_box=np.asarray(annotation["object"], dtype=np.float32),
                    target_box=np.asarray(annotation["target"], dtype=np.float32),
                    # VisionTrajDataset recomputes the usable length while loading.
                    n_frames=0,
                    scene=scene,
                    task=task,
                )
            )
        return records

    return convert(manifest["train_episodes"]), convert(manifest["val_episodes"])


class OfficialGrootBatchProcessor:
    """Convert a PyTorch batch to official N1.6 ``VLAStepData`` records."""

    def __init__(self, processor, prompt_protocol: str) -> None:
        self.processor = processor
        self.prompt_protocol = prompt_protocol

    def train(self) -> None:
        self.processor.train()

    def eval(self) -> None:
        self.processor.eval()

    @staticmethod
    def _uint8_hwc(image: torch.Tensor) -> np.ndarray:
        image = image.detach().cpu().clamp(0.0, 1.0)
        return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import VLAStepData

        features = []
        batch_size = int(batch["action"].shape[0])
        for index in range(batch_size):
            actions = {}
            if self.processor.training:
                actions["trajectory"] = batch["action"][index].detach().cpu().numpy()
            images = {
                "current_frame": [
                    self._uint8_hwc(batch["observation.images.current_frame"][index])
                ]
            }
            if self.prompt_protocol == VISUAL_PROMPT_PROTOCOL:
                images["first_frame"] = [
                    self._uint8_hwc(batch["observation.images.first_frame"][index])
                ]
            step_data = VLAStepData(
                images=images,
                states={
                    "history": batch["observation.state"][index]
                    .detach()
                    .cpu()
                    .numpy()[None, :]
                },
                actions=actions,
                text=batch["task"][index],
                embodiment=EmbodimentTag.NEW_EMBODIMENT,
            )
            features.append(self.processor([{"content": step_data}]))
        return self.processor.collator(features)["inputs"]


def _make_policy_and_processor(args: argparse.Namespace, stats: dict, device: torch.device):
    # Importing the model module registers the exact N1.6 architecture with
    # Hugging Face AutoModel.  This deliberately avoids LeRobot's N1.5 wrapper.
    from gr00t.configs.data.embodiment_configs import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
    )
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6  # noqa: F401
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
    from transformers import AutoModel

    video_keys = (
        ["first_frame", "current_frame"]
        if args.prompt_protocol == VISUAL_PROMPT_PROTOCOL
        else ["current_frame"]
    )
    modality_configs = {
        "new_embodiment": {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=video_keys,
            ),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["history"]),
            "action": ModalityConfig(
                delta_indices=list(range(args.action_horizon)),
                modality_keys=["trajectory"],
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation.ABSOLUTE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["instruction"]),
        }
    }
    loading_kwargs = {"trust_remote_code": True, "local_files_only": True}
    policy, loading_info = AutoModel.from_pretrained(
        args.model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        backbone_trainable_params_fp32=True,
        transformers_loading_kwargs=loading_kwargs,
        output_loading_info=True,
        **loading_kwargs,
    )
    if loading_info.get("missing_keys"):
        raise RuntimeError(
            "Official N1.6 checkpoint did not load cleanly; missing keys: "
            + ", ".join(loading_info["missing_keys"][:10])
        )
    config = policy.config
    if config.action_horizon < args.action_horizon or config.max_action_dim < 10:
        raise ValueError(
            f"checkpoint head {config.action_horizon}x{config.max_action_dim} cannot cover "
            f"requested {args.action_horizon}x10"
        )
    processor = Gr00tN1d6Processor(
        modality_configs=modality_configs,
        statistics=stats,
        use_percentiles=False,
        clip_outliers=True,
        image_crop_size=config.image_crop_size,
        image_target_size=config.image_target_size,
        shortest_image_edge=config.shortest_image_edge,
        crop_fraction=config.crop_fraction,
        random_rotation_angle=None,
        color_jitter_params=None,
        formalize_language=config.formalize_language,
        model_name=config.model_name,
        model_type=config.backbone_model_type,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        apply_sincos_state_encoding=False,
        max_action_horizon=config.action_horizon,
        use_albumentations=True,
        extra_augmentation_config=None,
        use_relative_action=False,
        transformers_loading_kwargs=loading_kwargs,
    )
    return policy, OfficialGrootBatchProcessor(processor, args.prompt_protocol), config


def _unnormalize_action(action: torch.Tensor, stats: dict) -> torch.Tensor:
    action_stats = stats["new_embodiment"]["action"]["trajectory"]
    minimum = torch.as_tensor(action_stats["min"], device=action.device, dtype=action.dtype)
    maximum = torch.as_tensor(action_stats["max"], device=action.device, dtype=action.dtype)
    return (action + 1.0) * 0.5 * (maximum - minimum) + minimum


def evaluate(accelerator, policy, preprocessor, loader, stats, output_dir, step, max_batches):
    from .evaluation import evaluate_vla
    model = accelerator.unwrap_model(policy)
    was_training = preprocessor.training
    preprocessor.eval()
    def predict(batch):
        with accelerator.autocast():
            prediction = model.get_action(preprocessor(batch))["action_pred"]
        return _unnormalize_action(prediction.float(), stats)
    try:
        return evaluate_vla(accelerator, policy, loader, output_dir, step, max_batches, predict)
    finally:
        preprocessor.train(was_training)


def _save_model(accelerator: Accelerator, policy, output_dir: Path, step: int) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = accelerator.get_state_dict(policy)
        state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
        save_file(state, checkpoint_dir / "model.safetensors")
        with open(checkpoint_dir / "training_state.json", "w") as handle:
            json.dump({"step": step}, handle, indent=2)
    accelerator.wait_for_everyone()


def load_eval_checkpoint(policy, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GR00T eval checkpoint not found: {checkpoint_path}")
    state = load_file(checkpoint_path, device="cpu")
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "GR00T eval checkpoint did not load cleanly: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    print(
        f"[GR00T eval] loaded {len(state)} tensors from {checkpoint_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    language_map: dict[str, dict[str, Any]] = {}
    language_map_metadata: dict[str, Any] = {}
    if args.prompt_protocol == LANGUAGE_ONLY_PROTOCOL:
        if not args.language_map:
            raise ValueError("--language-map is required for --prompt-protocol language_only")
        with open(args.language_map) as handle:
            language_payload = json.load(handle)
        if language_payload.get("protocol") != LANGUAGE_ONLY_PROTOCOL:
            raise ValueError(f"invalid language-map protocol in {args.language_map!r}")
        if language_payload.get("boxes_enter_model") is not False:
            raise ValueError("language map must explicitly record boxes_enter_model=false")
        language_map = language_payload["episodes"]
        language_map_metadata = {
            key: value
            for key, value in language_payload.items()
            if key not in ("episodes", "scene3_cluster_diagnostics")
        }
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
        # Our cosine schedule is defined in global optimizer updates.  The
        # default AcceleratedScheduler advances once per process (6x under
        # six-GPU DDP), so step it explicitly only on synchronized updates.
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp],
    )
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "experiment_config.json", "w") as handle:
            json.dump(vars(args), handle, indent=2)
    accelerator.wait_for_everyone()

    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    if args.split_manifest and Path(args.split_manifest).exists():
        train_episodes, val_episodes = episodes_from_split_manifest(
            args.split_manifest,
            args.annotations,
            args.data_root,
            args.split_seed,
        )
        train_episodes = filter_episodes_by_scenes(train_episodes, scenes)
        val_episodes = filter_episodes_by_scenes(val_episodes, scenes)
    else:
        episodes = build_episode_index(args.data_root, args.annotations)
        episodes = filter_episodes_by_scenes(episodes, scenes)
        train_episodes, val_episodes = split_episodes(episodes, args.val_ratio, args.split_seed)
    if args.smoke_steps:
        # Interface validation does not need to scan the full NFS-backed corpus.
        train_episodes = train_episodes[:4]
        val_episodes = val_episodes[:4]

    def make_base(selected_episodes) -> VisionTrajDataset:
        return VisionTrajDataset(
            selected_episodes,
            action_horizon=args.action_horizon,
            history_horizon=args.history_horizon,
            image_size=args.image_size,
            stride=args.stride,
            prompt_form=(
                "vision_bbox_and_bbox"
                if args.prompt_protocol == VISUAL_PROMPT_PROTOCOL
                else "none"
            ),
            frame_cache_root=args.frame_cache_root,
        )

    train_base = make_base(train_episodes)
    val_base = make_base(val_episodes)
    train_dataset = GrootEgoSPTDataset(
        train_base, args.prompt_protocol, language_map=language_map
    )
    val_dataset = GrootEgoSPTDataset(
        val_base, args.prompt_protocol, language_map=language_map
    )
    if args.prompt_protocol == LANGUAGE_ONLY_PROTOCOL:
        input_audit = train_dataset[0]
        if "observation.images.first_frame" in input_audit:
            raise AssertionError("language-only model input unexpectedly contains first_frame")
        expected_keys = {
            "observation.images.current_frame",
            "observation.state",
            "action",
            "task",
            "index",
        }
        if set(input_audit) != expected_keys:
            raise AssertionError(
                f"language-only input keys differ: {sorted(input_audit)}"
            )
    stats = compute_min_max_stats(train_base, args.stats_samples, args.split_seed)

    if accelerator.is_main_process:
        if args.prompt_protocol == VISUAL_PROMPT_PROTOCOL:
            prompt_protocol = "generic language + first-frame rendered BBox + current RGB"
            instruction_summary: str | dict[str, Any] = INSTRUCTION
            input_modalities = ["first_frame_visual_prompt", "current_rgb", "history", "language"]
        else:
            instructions = sorted(
                {language_map[episode.key]["instruction"] for episode in train_base.episodes}
            )
            prompt_protocol = "task-specific language + current RGB; no first frame or boxes"
            instruction_summary = {
                "template": "pick {object} and place to {target}",
                "unique_train_instructions": len(instructions),
                "examples": instructions[:10],
            }
            input_modalities = ["current_rgb", "history", "task_specific_language"]
        metadata = {
            "method": "GR00T N1.6 external VLA baseline",
            "prompt_protocol": prompt_protocol,
            "instruction": instruction_summary,
            "input_modalities": input_modalities,
            "model_input_contains_first_frame": args.prompt_protocol == VISUAL_PROMPT_PROTOCOL,
            "model_input_contains_visual_boxes": args.prompt_protocol == VISUAL_PROMPT_PROTOCOL,
            "model_input_contains_bbox_coordinates": False,
            "language_map": language_map_metadata,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_episodes": len(train_base.episodes),
            "val_episodes": len(val_base.episodes),
            "split_seed": args.split_seed,
            "training_seed": args.seed,
            "stats": _serializable_stats(stats),
        }
        with open(output_dir / "experiment_meta.json", "w") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"[GR00T data] train={len(train_dataset)} val={len(val_dataset)}", flush=True)

    multiprocessing_context = "forkserver" if args.num_workers else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    policy, preprocessor, config = _make_policy_and_processor(args, stats, accelerator.device)
    if args.eval_checkpoint:
        load_eval_checkpoint(policy, args.eval_checkpoint)
    if args.eval_only:
        if not args.eval_checkpoint:
            raise ValueError("--eval-only requires --eval-checkpoint")
        policy, val_loader = accelerator.prepare(policy, val_loader)
        started = time.time()
        evaluate(
            accelerator,
            policy,
            preprocessor,
            val_loader,
            stats,
            output_dir,
            args.eval_step,
            args.final_eval_max_batches or None,
        )
        if accelerator.is_main_process:
            with open(output_dir / "completed.json", "w") as handle:
                json.dump(
                    {
                        "step": args.eval_step,
                        "elapsed_seconds": time.time() - started,
                        "eval_only": True,
                        "eval_checkpoint": args.eval_checkpoint,
                    },
                    handle,
                    indent=2,
                )
        return
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(
        math.ceil(len(train_loader) / accelerator.num_processes) / args.grad_accum_steps
    )
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if args.smoke_steps:
        total_steps = args.smoke_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    policy, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        policy, optimizer, train_loader, val_loader, scheduler
    )

    if args.resume:
        accelerator.load_state(args.resume)
    if accelerator.is_main_process:
        trainable_count = sum(parameter.numel() for parameter in trainable)
        print(
            f"[GR00T train] world={accelerator.num_processes} batch={args.batch_size} "
            f"accum={args.grad_accum_steps} effective_batch="
            f"{args.batch_size * accelerator.num_processes * args.grad_accum_steps} "
            f"updates={total_steps} trainable={trainable_count:,}",
            flush=True,
        )

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    loss_sum = 0.0
    loss_count = 0
    started = time.time()
    while update_step < total_steps:
        for batch in train_loader:
            with accelerator.accumulate(policy):
                processed = preprocessor(batch)
                loss = policy(processed)["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 10.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            micro_step += 1
            loss_sum += float(loss.detach())
            loss_count += 1
            if not accelerator.sync_gradients:
                continue

            update_step += 1
            if update_step % args.log_every == 0 and accelerator.is_main_process:
                elapsed = time.time() - started
                record = {
                    "step": update_step,
                    "loss": loss_sum / max(1, loss_count),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": elapsed,
                }
                print(f"[GR00T train] {json.dumps(record)}", flush=True)
                with open(output_dir / "train_log.jsonl", "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                loss_sum = 0.0
                loss_count = 0

            if args.eval_every and update_step % args.eval_every == 0:
                evaluate(
                    accelerator,
                    policy,
                    preprocessor,
                    val_loader,
                    stats,
                    output_dir,
                    update_step,
                    args.eval_max_batches,
                )
            if update_step >= total_steps:
                break

    evaluate(
        accelerator,
        policy,
        preprocessor,
        val_loader,
        stats,
        output_dir,
        update_step,
        1
        if args.smoke_steps
        else (args.final_eval_max_batches or None),
    )
    if not args.smoke_steps:
        _save_model(accelerator, policy, output_dir, update_step)
    if accelerator.is_main_process:
        with open(output_dir / "completed.json", "w") as handle:
            json.dump({"step": update_step, "elapsed_seconds": time.time() - started}, handle, indent=2)


if __name__ == "__main__":
    main()
