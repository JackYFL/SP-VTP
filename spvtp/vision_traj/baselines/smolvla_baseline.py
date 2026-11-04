"""Fine-tune and evaluate SmolVLA with task-specific language only.

The model input is strictly current RGB, four previous 10-D relative actions,
and the task-specific pick/place instruction. First-frame images, rendered
visual prompts, and bounding-box coordinates are forbidden.

SmolVLA's pretrained state projection accepts 32 values, while the controlled
EgoSPT protocol contains a 4 x 10-D history. We expand that one projection to
40 inputs, copy all 32 pretrained columns, zero-initialize the eight new
columns, and train the projection with the action expert. All other checkpoint
tensors must load strictly.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from spvtp.vision_traj.data.dataset import (
    VisionTrajDataset,
    build_episode_index,
    filter_episodes_by_scenes,
    split_episodes,
    worker_init_fn,
)
from spvtp.vision_traj.baselines.groot_baseline import (
    GrootEgoSPTDataset,
    LANGUAGE_ONLY_PROTOCOL,
    _trajectory_arrays,
    episodes_from_split_manifest,
)
from spvtp.vision_traj.baselines.pi05_baseline import _save_trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="spvtp/EgoSPT")
    parser.add_argument("--annotations", default="annotations_merged.json")
    parser.add_argument("--frame-cache-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--language-map", required=True)
    parser.add_argument(
        "--split-manifest",
        default=(
            "runs/vision_traj_experiments_final/"
            "baseline_vision_bbox_and_bbox_all_scenes/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "runs/vision_traj_experiments_final/"
            "baseline_smolvla_language_only_all_scenes"
        ),
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
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-10)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-max-batches", type=int, default=32)
    parser.add_argument("--final-eval-max-batches", type=int, default=0)
    parser.add_argument("--stats-samples", type=int, default=4096)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--eval-checkpoint",
        default=None,
        help="Load a saved trainable.safetensors checkpoint before evaluation.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-step", type=int, default=0)
    parser.add_argument("--full-finetune", action="store_true")
    return parser.parse_args()


def _array_stats(array: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "min": torch.from_numpy(array.min(axis=0)).float(),
        "max": torch.from_numpy(array.max(axis=0)).float(),
        "mean": torch.from_numpy(array.mean(axis=0)).float(),
        "std": torch.from_numpy(array.std(axis=0).clip(min=1.0e-8)).float(),
        "q01": torch.from_numpy(np.quantile(array, 0.01, axis=0)).float(),
        "q99": torch.from_numpy(np.quantile(array, 0.99, axis=0)).float(),
    }


def compute_stats(dataset: VisionTrajDataset, n_samples: int, seed: int) -> dict:
    """Compute state/action statistics without decoding video frames."""
    rng = np.random.default_rng(seed)
    count = min(n_samples, len(dataset))
    indices = rng.choice(len(dataset), size=count, replace=False)
    states, actions = [], []
    for index in indices:
        state, action = _trajectory_arrays(dataset, int(index))
        states.append(state)
        actions.append(action)
    return {
        "observation.state": _array_stats(np.stack(states)),
        "action": _array_stats(np.concatenate(actions, axis=0)),
    }


def _stats_to_json(stats: dict) -> dict:
    return {
        key: {name: value.tolist() for name, value in values.items()}
        for key, values in stats.items()
    }


def load_language_map(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("protocol") != LANGUAGE_ONLY_PROTOCOL:
        raise ValueError(f"invalid language-map protocol in {path!s}")
    if payload.get("boxes_enter_model") is not False:
        raise ValueError("language map must explicitly record boxes_enter_model=false")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("episodes", "scene3_cluster_diagnostics")
    }
    return payload["episodes"], metadata


class SmolVLABatchProcessor:
    """Apply official mean/std normalization and SmolVLM tokenization."""

    def __init__(self, stats: dict, config, device: torch.device) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
        self.stats = stats
        self.device = device
        self.tokenizer_max_length = config.tokenizer_max_length
        self.pad_language_to = config.pad_language_to

    @staticmethod
    def _normalize(value: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        mean = stats["mean"].to(device=value.device, dtype=value.dtype)
        std = stats["std"].to(device=value.device, dtype=value.dtype).clamp_min(1.0e-8)
        return (value - mean) / std

    def __call__(self, batch: dict[str, Any], include_action: bool = True) -> dict[str, Any]:
        state = batch["observation.state"].to(self.device, non_blocking=True).float()
        state = self._normalize(state, self.stats["observation.state"])
        prompts = [
            task.strip().replace("_", " ").replace("\n", " ") + "\n"
            for task in batch["task"]
        ]
        tokens = self.tokenizer(
            prompts,
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding=self.pad_language_to,
            padding_side="right",
            return_tensors="pt",
        )
        processed = {
            "observation.images.current_frame": batch[
                "observation.images.current_frame"
            ].to(self.device, non_blocking=True),
            "observation.state": state,
            "observation.language.tokens": tokens["input_ids"].to(self.device),
            "observation.language.attention_mask": tokens["attention_mask"].to(
                self.device, dtype=torch.bool
            ),
        }
        if include_action:
            action = batch["action"].to(self.device, non_blocking=True).float()
            processed["action"] = self._normalize(action, self.stats["action"])
        return processed

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        stats = self.stats["action"]
        mean = stats["mean"].to(device=action.device, dtype=action.dtype)
        std = stats["std"].to(device=action.device, dtype=action.dtype)
        return action * std + mean


def make_policy(args: argparse.Namespace, device: torch.device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(args.model_path, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(
            f"checkpoint config is {type(config).__name__}, expected SmolVLAConfig"
        )
    history_dim = args.history_horizon * 10
    config.input_features = {
        "observation.images.current_frame": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(history_dim,)
        ),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))
    }
    config.chunk_size = args.action_horizon
    config.n_action_steps = args.action_horizon
    config.max_state_dim = history_dim
    config.max_action_dim = 32
    config.num_steps = args.num_inference_steps
    config.device = str(device)
    config.load_vlm_weights = False
    config.freeze_vision_encoder = not args.full_finetune
    config.train_expert_only = not args.full_finetune
    config.train_state_proj = True

    policy = SmolVLAPolicy(config)
    checkpoint_path = Path(args.model_path) / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SmolVLA checkpoint not found: {checkpoint_path}")
    checkpoint = load_file(checkpoint_path, device="cpu")
    state_key = "model.state_proj.weight"
    old_state_width = int(checkpoint[state_key].shape[1])
    if history_dim < old_state_width:
        raise ValueError(
            f"history width {history_dim} is smaller than pretrained width {old_state_width}"
        )
    if history_dim != old_state_width:
        expanded = checkpoint[state_key].new_zeros(
            checkpoint[state_key].shape[0], history_dim
        )
        expanded[:, :old_state_width] = checkpoint[state_key]
        checkpoint[state_key] = expanded
    missing, unexpected = policy.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "SmolVLA checkpoint did not load cleanly: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    loaded_count = len(checkpoint)
    del checkpoint
    gc.collect()
    print(
        f"[SmolVLA model] all {loaded_count} tensors loaded; "
        f"state projection {old_state_width}->{history_dim}, new columns zero initialized",
        flush=True,
    )
    return policy, config, old_state_width


def load_trainable_checkpoint(policy, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SmolVLA eval checkpoint not found: {checkpoint_path}")
    state = load_file(checkpoint_path, device="cpu")
    missing, unexpected = policy.load_state_dict(state, strict=False)
    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(trainable_names.difference(state))
    if unexpected or missing_trainable:
        raise RuntimeError(
            "SmolVLA eval checkpoint did not load cleanly: "
            f"missing_trainable={missing_trainable[:8]}, unexpected={unexpected[:8]}"
        )
    print(
        f"[SmolVLA eval] loaded {len(state)} trainable tensors from {checkpoint_path}",
        flush=True,
    )


def evaluate(accelerator, policy, processor, loader, output_dir, step, max_batches):
    from .evaluation import evaluate_vla
    model = accelerator.unwrap_model(policy)
    def predict(batch):
        with accelerator.autocast():
            prediction = model.predict_action_chunk(processor(batch, include_action=False))
        return processor.unnormalize_action(prediction.float())
    return evaluate_vla(accelerator, policy, loader, output_dir, step, max_batches, predict)


def main() -> None:
    args = parse_args()
    language_map, language_metadata = load_language_map(args.language_map)
    ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum_steps,
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
            args.split_manifest, args.annotations, args.data_root, args.split_seed
        )
        train_episodes = filter_episodes_by_scenes(train_episodes, scenes)
        val_episodes = filter_episodes_by_scenes(val_episodes, scenes)
    else:
        episodes = build_episode_index(args.data_root, args.annotations)
        episodes = filter_episodes_by_scenes(episodes, scenes)
        train_episodes, val_episodes = split_episodes(
            episodes, args.val_ratio, args.split_seed
        )
    if args.smoke_steps:
        train_episodes = train_episodes[:4]
        val_episodes = val_episodes[:4]

    def make_base(selected_episodes) -> VisionTrajDataset:
        return VisionTrajDataset(
            selected_episodes,
            action_horizon=args.action_horizon,
            history_horizon=args.history_horizon,
            image_size=args.image_size,
            stride=args.stride,
            prompt_form="none",
            frame_cache_root=args.frame_cache_root,
        )

    train_base = make_base(train_episodes)
    val_base = make_base(val_episodes)
    train_dataset = GrootEgoSPTDataset(
        train_base, LANGUAGE_ONLY_PROTOCOL, language_map=language_map
    )
    val_dataset = GrootEgoSPTDataset(
        val_base, LANGUAGE_ONLY_PROTOCOL, language_map=language_map
    )
    input_audit = train_dataset[0]
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
    stats = compute_stats(train_base, args.stats_samples, args.split_seed)

    if accelerator.is_main_process:
        instructions = sorted(
            {language_map[episode.key]["instruction"] for episode in train_base.episodes}
        )
        metadata = {
            "method": "SmolVLA-450M external VLA baseline",
            "prompt_protocol": (
                "task-specific language + current RGB; no first frame or boxes"
            ),
            "instruction": {
                "template": "pick {object} and place to {target}",
                "unique_train_instructions": len(instructions),
                "examples": instructions[:10],
            },
            "input_modalities": [
                "current_rgb",
                "history",
                "task_specific_language",
            ],
            "model_input_contains_first_frame": False,
            "model_input_contains_visual_boxes": False,
            "model_input_contains_bbox_coordinates": False,
            "language_map": language_metadata,
            "action_target": [args.action_horizon, 10],
            "history_state_dim": args.history_horizon * 10,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_episodes": len(train_base.episodes),
            "val_episodes": len(val_base.episodes),
            "split_seed": args.split_seed,
            "training_seed": args.seed,
            "stats": _stats_to_json(stats),
        }
        with open(output_dir / "experiment_meta.json", "w") as handle:
            json.dump(metadata, handle, indent=2)
        print(
            f"[SmolVLA data] train={len(train_dataset)} val={len(val_dataset)}",
            flush=True,
        )

    context = "forkserver" if args.num_workers else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=context,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    policy, config, old_state_width = make_policy(args, accelerator.device)
    processor = SmolVLABatchProcessor(stats, config, accelerator.device)
    if args.eval_checkpoint:
        load_trainable_checkpoint(policy, args.eval_checkpoint)
    if args.eval_only:
        if not args.eval_checkpoint:
            raise ValueError("--eval-only requires --eval-checkpoint")
        policy, val_loader = accelerator.prepare(policy, val_loader)
        started = time.time()
        evaluate(
            accelerator,
            policy,
            processor,
            val_loader,
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
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        math.ceil(len(train_loader) / accelerator.num_processes)
        / args.grad_accum_steps
    )
    total_steps = min(updates_per_epoch * args.epochs, args.max_steps)
    if args.smoke_steps:
        total_steps = args.smoke_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    policy, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        policy, optimizer, train_loader, val_loader, scheduler
    )
    if args.resume:
        accelerator.load_state(args.resume)
    if accelerator.is_main_process:
        trainable_count = sum(parameter.numel() for parameter in trainable)
        total_count = sum(
            parameter.numel()
            for parameter in accelerator.unwrap_model(policy).parameters()
        )
        print(
            f"[SmolVLA train] world={accelerator.num_processes} "
            f"batch={args.batch_size} accum={args.grad_accum_steps} "
            f"effective_batch={args.batch_size * accelerator.num_processes * args.grad_accum_steps} "
            f"updates={total_steps} trainable={trainable_count:,}/{total_count:,} "
            f"state_projection={old_state_width}->{args.history_horizon * 10}",
            flush=True,
        )

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    loss_sum = 0.0
    loss_count = 0
    started = time.time()
    while update_step < total_steps:
        for batch in train_loader:
            with accelerator.accumulate(policy):
                processed = processor(batch, include_action=True)
                with accelerator.autocast():
                    loss, _ = policy(processed)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 10.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            loss_count += 1
            if not accelerator.sync_gradients:
                continue
            update_step += 1
            if accelerator.is_main_process and (
                update_step == 1 or update_step % args.log_every == 0
            ):
                elapsed = time.time() - started
                record = {
                    "step": update_step,
                    "loss": loss_sum / max(1, loss_count),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": elapsed,
                }
                print(f"[SmolVLA train] {json.dumps(record)}", flush=True)
                with open(output_dir / "train_log.jsonl", "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                loss_sum = 0.0
                loss_count = 0
            if args.eval_every and update_step % args.eval_every == 0:
                evaluate(
                    accelerator,
                    policy,
                    processor,
                    val_loader,
                    output_dir,
                    update_step,
                    args.eval_max_batches,
                )
            if args.save_every and update_step % args.save_every == 0:
                _save_trainable(
                    accelerator, policy, output_dir, update_step
                )
            if update_step >= total_steps:
                break

    final_cap = args.final_eval_max_batches or None
    if args.smoke_steps and final_cap is None:
        final_cap = 1
    evaluate(
        accelerator,
        policy,
        processor,
        val_loader,
        output_dir,
        update_step,
        final_cap,
    )
    if not args.smoke_steps:
        _save_trainable(accelerator, policy, output_dir, update_step)
    if accelerator.is_main_process:
        with open(output_dir / "completed.json", "w") as handle:
            json.dump(
                {
                    "step": update_step,
                    "elapsed_seconds": time.time() - started,
                    "smoke": bool(args.smoke_steps),
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
