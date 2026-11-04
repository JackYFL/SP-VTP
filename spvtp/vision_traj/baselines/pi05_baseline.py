"""Fine-tune and evaluate pi0.5 on the EgoSPT trajectory protocol.

This adapter uses the official LeRobot port of Physical Intelligence's pi0.5
flow-matching checkpoint.  It deliberately keeps the same task interface used
by the GR00T baseline and the paper's main method:

* first-frame RGB with the object/target boxes rendered in green/red;
* current RGB observation;
* four previous 10-D relative actions, flattened to a 40-D state;
* a generic instruction which does not reveal scene/task identity; and
* a 16 x 10-D relative-action target.

The checkpoint has no horizon-dependent learned parameters, so its runtime
chunk size is changed from 50 to 16 rather than padding the supervision with 34
artificial zero actions.  The 40-D history is padded to 64 dimensions before
pi0.5 discretizes it into the language prefix.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
from safetensors.torch import save_file
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
    INSTRUCTION,
    _trajectory_arrays,
    episodes_from_split_manifest,
)


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
        "--lerobot-revision",
        default="f0d2b37bebddf7e8852cb624712da0ad564601b6",
        help="Pinned upstream LeRobot source revision used for the pi0.5 architecture.",
    )
    parser.add_argument(
        "--tokenizer-model",
        required=True,
    )
    parser.add_argument(
        "--split-manifest",
        default=(
            "runs/vision_traj_experiments_final/"
            "baseline_vision_bbox_and_bbox_all_scenes/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/vision_traj_experiments_final/baseline_pi05_all_scenes",
    )
    parser.add_argument("--scenes", default="scene1,scene2,scene3")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--history-horizon", type=int, default=4)
    parser.add_argument("--max-state-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-max-batches", type=int, default=8)
    parser.add_argument("--final-eval-max-batches", type=int, default=0)
    parser.add_argument("--stats-samples", type=int, default=4096)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Train the full VLM too. This generally needs more than 70 GB/GPU.",
    )
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


def compute_quantile_stats(dataset: VisionTrajDataset, n_samples: int, seed: int) -> dict:
    """Compute the LeRobot statistics schema without decoding video frames."""
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


class Pi05BatchProcessor:
    """Apply pi0.5 quantile normalization and state-aware tokenization."""

    def __init__(
        self,
        stats: dict,
        tokenizer_model: str | os.PathLike,
        device: torch.device,
        max_state_dim: int,
        tokenizer_max_length: int,
    ) -> None:
        from transformers import GemmaTokenizer

        tokenizer_model = Path(tokenizer_model)
        if not tokenizer_model.is_file():
            raise FileNotFoundError(f"PaliGemma tokenizer model not found: {tokenizer_model}")
        self.tokenizer = GemmaTokenizer(vocab_file=str(tokenizer_model))
        self.device = device
        self.max_state_dim = max_state_dim
        self.tokenizer_max_length = tokenizer_max_length
        self.stats = stats

    @staticmethod
    def _quantile_normalize(value: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        q01 = stats["q01"].to(device=value.device, dtype=value.dtype)
        q99 = stats["q99"].to(device=value.device, dtype=value.dtype)
        denominator = (q99 - q01).clamp_min(1.0e-8)
        return 2.0 * (value - q01) / denominator - 1.0

    def __call__(self, batch: dict[str, Any], include_action: bool = True) -> dict[str, Any]:
        state = batch["observation.state"].to(self.device, non_blocking=True).float()
        state = self._quantile_normalize(state, self.stats["observation.state"])
        if state.shape[-1] > self.max_state_dim:
            raise ValueError(
                f"state dimension {state.shape[-1]} exceeds max_state_dim={self.max_state_dim}"
            )
        state = torch.nn.functional.pad(state, (0, self.max_state_dim - state.shape[-1]))

        # Match LeRobot's Pi05PrepareStateTokenizerProcessorStep exactly.
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        discretized = np.digitize(state.detach().cpu().numpy(), bins=bins) - 1
        prompts = []
        for task, values in zip(batch["task"], discretized, strict=True):
            task = task.strip().replace("_", " ").replace("\n", " ")
            state_string = " ".join(map(str, values))
            prompts.append(f"Task: {task}, State: {state_string};\nAction: ")
        tokens = self.tokenizer(
            prompts,
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )

        processed = {
            "observation.images.first_frame": batch["observation.images.first_frame"].to(
                self.device, non_blocking=True
            ),
            "observation.images.current_frame": batch["observation.images.current_frame"].to(
                self.device, non_blocking=True
            ),
            "observation.state": state,
            "observation.language.tokens": tokens["input_ids"].to(self.device),
            "observation.language.attention_mask": tokens["attention_mask"].to(
                self.device, dtype=torch.bool
            ),
        }
        if include_action:
            action = batch["action"].to(self.device, non_blocking=True).float()
            processed["action"] = self._quantile_normalize(action, self.stats["action"])
        return processed

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        q01 = self.stats["action"]["q01"].to(device=action.device, dtype=action.dtype)
        q99 = self.stats["action"]["q99"].to(device=action.device, dtype=action.dtype)
        return (action + 1.0) * (q99 - q01) / 2.0 + q01


def _make_policy(args: argparse.Namespace, device: torch.device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    # Load through the registered base class so draccus consumes the checkpoint's
    # ``type: pi05`` discriminator before decoding PI05Config.
    config = PreTrainedConfig.from_pretrained(args.model_path, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError(f"checkpoint config is {type(config).__name__}, expected PI05Config")
    config.input_features = {
        "observation.images.first_frame": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)
        ),
        "observation.images.current_frame": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(40,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
    }
    config.chunk_size = args.action_horizon
    config.n_action_steps = args.action_horizon
    config.max_state_dim = args.max_state_dim
    config.max_action_dim = 32
    config.num_inference_steps = args.num_inference_steps
    config.device = str(device)
    config.dtype = "bfloat16"
    config.gradient_checkpointing = True
    config.freeze_vision_encoder = not args.full_finetune
    config.train_expert_only = not args.full_finetune

    # Load explicitly because the bundled loader catches strict-load failures
    # and can otherwise return a partially initialized policy. The upstream
    # architecture retains the checkpoint's namespaces; PaliGemma ties its LM
    # head and token embedding, so materialize the second state-dict key too.
    policy = PI05Policy(config)
    checkpoint_path = Path(args.model_path) / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"pi0.5 checkpoint not found: {checkpoint_path}")
    from safetensors.torch import load_file

    checkpoint = load_file(checkpoint_path, device="cpu")
    remapped = {}
    for name, value in checkpoint.items():
        remapped[f"model.{name}"] = value
        if name == "paligemma_with_expert.paligemma.lm_head.weight":
            remapped[
                "model.paligemma_with_expert.paligemma.model."
                "language_model.embed_tokens.weight"
            ] = value.clone()
    missing, unexpected = policy.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "pi0.5 checkpoint did not load cleanly: "
            f"missing={len(missing)} {missing[:8]}, "
            f"unexpected={len(unexpected)} {unexpected[:8]}"
        )
    loaded_count = len(remapped)
    del checkpoint, remapped
    gc.collect()
    # Flow matching calls gemma_expert.model directly; its 257k-token LM head
    # is never on the action path. Freeze it so DDP does not track a 263M-param
    # unused projection as part of the action expert fine-tune.
    for parameter in policy.model.paligemma_with_expert.gemma_expert.lm_head.parameters():
        parameter.requires_grad = False
    print(f"[pi0.5 model] all {loaded_count} checkpoint tensors loaded cleanly", flush=True)
    return policy, config


def evaluate(accelerator, policy, processor, loader, output_dir, step, max_batches):
    from .evaluation import evaluate_vla
    model = accelerator.unwrap_model(policy)
    def predict(batch):
        with accelerator.autocast():
            prediction = model.predict_action_chunk(processor(batch, include_action=False))
        return processor.unnormalize_action(prediction.float())
    return evaluate_vla(accelerator, policy, loader, output_dir, step, max_batches, predict)


def _save_trainable(accelerator: Accelerator, policy, output_dir: Path, step: int) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(policy)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        state = accelerator.get_state_dict(policy)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in state.items()
            if name in trainable_names
        }
        checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, checkpoint_dir / "trainable.safetensors")
        with open(checkpoint_dir / "training_state.json", "w") as handle:
            json.dump({"step": step, "trainable_tensors": len(state)}, handle, indent=2)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if args.history_horizon * 10 > args.max_state_dim:
        raise ValueError("max_state_dim is too small for the flattened action history")
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
        train_episodes = train_episodes[:4]
        val_episodes = val_episodes[:4]

    def make_base(selected_episodes) -> VisionTrajDataset:
        return VisionTrajDataset(
            selected_episodes,
            action_horizon=args.action_horizon,
            history_horizon=args.history_horizon,
            image_size=args.image_size,
            stride=args.stride,
            prompt_form="vision_bbox_and_bbox",
            frame_cache_root=args.frame_cache_root,
        )

    train_base = make_base(train_episodes)
    val_base = make_base(val_episodes)
    train_dataset = GrootEgoSPTDataset(train_base)
    val_dataset = GrootEgoSPTDataset(val_base)
    stats = compute_quantile_stats(train_base, args.stats_samples, args.split_seed)

    if accelerator.is_main_process:
        metadata = {
            "method": "pi0.5 external VLA baseline (official LeRobot checkpoint)",
            "prompt_protocol": "generic language + first-frame rendered BBox + current RGB",
            "instruction": INSTRUCTION,
            "action_target": [args.action_horizon, 10],
            "history_state_dim": args.history_horizon * 10,
            "max_state_dim": args.max_state_dim,
            "finetuning": "full" if args.full_finetune else "action expert only",
            "unused_expert_lm_head_frozen": True,
            "lerobot_revision": args.lerobot_revision,
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
        print(f"[pi0.5 data] train={len(train_dataset)} val={len(val_dataset)}", flush=True)

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

    policy, config = _make_policy(args, accelerator.device)
    processor = Pi05BatchProcessor(
        stats,
        args.tokenizer_model,
        accelerator.device,
        args.max_state_dim,
        config.tokenizer_max_length,
    )
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
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
        total_count = sum(parameter.numel() for parameter in accelerator.unwrap_model(policy).parameters())
        print(
            f"[pi0.5 train] world={accelerator.num_processes} batch={args.batch_size} "
            f"accum={args.grad_accum_steps} effective_batch="
            f"{args.batch_size * accelerator.num_processes * args.grad_accum_steps} "
            f"updates={total_steps} trainable={trainable_count:,}/{total_count:,}",
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
                    accelerator.clip_grad_norm_(trainable, 1.0)
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
                print(
                    f"[pi0.5 step {update_step}/{total_steps}] "
                    f"loss={loss_sum / max(1, loss_count):.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed:.1f}s",
                    flush=True,
                )
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
                _save_trainable(accelerator, policy, output_dir, update_step)
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
    _save_trainable(accelerator, policy, output_dir, update_step)
    if accelerator.is_main_process:
        with open(output_dir / "completed.json", "w") as handle:
            json.dump({"step": update_step, "elapsed_seconds": time.time() - started}, handle, indent=2)


if __name__ == "__main__":
    main()
