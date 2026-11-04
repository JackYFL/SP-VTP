"""Fine-tune OpenVLA-7B with the official OFT continuous-action recipe.

The controlled model input is current RGB, optional action history, and a
task-specific pick/place instruction. No first frame, rendered visual prompt,
or bounding-box coordinate is exposed to the policy. Setting
``--history-horizon 0`` removes the proprio/history token entirely.

This adapter uses the OpenVLA-7B pretrained backbone, rank-32 LoRA, the OFT L1
continuous action head, parallel action decoding, and one proprio/history
token. The action protocol is set directly to 16 x 10-D instead of loading a
task-specific LIBERO head with a mismatched action space.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
import types
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from PIL import Image
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
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
    episodes_from_split_manifest,
)
from spvtp.vision_traj.baselines.pi05_baseline import _save_trainable
from spvtp.vision_traj.baselines.smolvla_baseline import (
    _stats_to_json,
    compute_stats,
    load_language_map,
)


IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="spvtp/EgoSPT")
    parser.add_argument("--annotations", default="annotations_merged.json")
    parser.add_argument("--frame-cache-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--language-map", required=True)
    parser.add_argument(
        "--oft-source",
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
        default=(
            "runs/vision_traj_experiments_final/"
            "baseline_openvla_oft_language_only_all_scenes"
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
    parser.add_argument("--sample-start-step", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-max-batches", type=int, default=16)
    parser.add_argument("--final-eval-max-batches", type=int, default=0)
    parser.add_argument("--stats-samples", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=32)
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
    return parser.parse_args()


def _get_current_action_mask(token_ids: torch.Tensor, action_dim: int) -> torch.Tensor:
    labeled = token_ids != IGNORE_INDEX
    cumulative = torch.cumsum(labeled, dim=1)
    return (1 <= cumulative) & (cumulative <= action_dim) & (
        token_ids > ACTION_TOKEN_BEGIN_IDX
    )


def _get_next_actions_mask(token_ids: torch.Tensor, action_dim: int) -> torch.Tensor:
    labeled = token_ids != IGNORE_INDEX
    cumulative = torch.cumsum(labeled, dim=1)
    return (cumulative > action_dim) & (token_ids > ACTION_TOKEN_BEGIN_IDX)


def _install_prismatic_shims(action_horizon: int, action_dim: int, history_dim: int) -> None:
    """Provide only the OFT modules needed by the standalone HF model."""

    class NormalizationType(str, Enum):
        NORMAL = "normal"
        BOUNDS = "bounds"
        BOUNDS_Q99 = "bounds_q99"

    prismatic = types.ModuleType("prismatic")
    prismatic.__path__ = []
    training = types.ModuleType("prismatic.training")
    training.__path__ = []
    train_utils = types.ModuleType("prismatic.training.train_utils")
    train_utils.get_current_action_mask = lambda token_ids: _get_current_action_mask(
        token_ids, action_dim
    )
    train_utils.get_next_actions_mask = lambda token_ids: _get_next_actions_mask(
        token_ids, action_dim
    )
    vla = types.ModuleType("prismatic.vla")
    vla.__path__ = []
    constants = types.ModuleType("prismatic.vla.constants")
    constants.IGNORE_INDEX = IGNORE_INDEX
    constants.ACTION_TOKEN_BEGIN_IDX = ACTION_TOKEN_BEGIN_IDX
    constants.STOP_INDEX = STOP_INDEX
    constants.NUM_ACTIONS_CHUNK = action_horizon
    constants.ACTION_DIM = action_dim
    constants.PROPRIO_DIM = history_dim
    constants.NormalizationType = NormalizationType
    constants.ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
    sys.modules.update(
        {
            "prismatic": prismatic,
            "prismatic.training": training,
            "prismatic.training.train_utils": train_utils,
            "prismatic.vla": vla,
            "prismatic.vla.constants": constants,
        }
    )


def _load_oft_runtime(source: str | Path, action_horizon: int, history_dim: int):
    """Load the official standalone OFT HF classes without the RLDS stack."""
    _install_prismatic_shims(action_horizon, 10, history_dim)
    source = Path(source)
    if not (source / "modeling_prismatic.py").is_file():
        raise FileNotFoundError(f"OpenVLA-OFT HF source not found: {source}")
    package_name = "egospt_openvla_oft_runtime"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    configuration = importlib.import_module(
        f"{package_name}.configuration_prismatic"
    )
    processing = importlib.import_module(f"{package_name}.processing_prismatic")
    modeling = importlib.import_module(f"{package_name}.modeling_prismatic")
    return configuration, processing, modeling


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.ffn(value)


class MLPResNet(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(2)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.fc1(self.layer_norm1(value)))
        for block in self.blocks:
            value = block(value)
        return self.fc2(self.layer_norm2(value))


class OFTL1ActionHead(nn.Module):
    """Official OFT L1 head with runtime-configurable action dimensions."""

    def __init__(
        self, llm_dim: int, action_horizon: int, action_dim: int
    ) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.model = MLPResNet(
            input_dim=llm_dim * action_dim,
            hidden_dim=llm_dim,
            output_dim=action_dim,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            batch_size, self.action_horizon, -1
        )
        return self.model(hidden_states)

    def predict_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward(hidden_states)


class ProprioProjector(nn.Module):
    def __init__(self, llm_dim: int, history_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(history_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(state)))


class OpenVLAOFTPolicy(nn.Module):
    def __init__(
        self,
        vla,
        action_head: OFTL1ActionHead,
        proprio_projector: ProprioProjector | None,
        action_horizon: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.vla = vla
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    @property
    def num_patches(self) -> int:
        return (
            self.vla.vision_backbone.get_num_patches()
            * self.vla.vision_backbone.get_num_images_in_input()
            + int(self.proprio_projector is not None)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.vla(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"].to(torch.bfloat16),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch.get("observation.state"),
            proprio_projector=self.proprio_projector,
            use_film=False,
        )
        token_labels = batch["labels"][:, 1:]
        action_mask = _get_current_action_mask(
            token_labels, self.action_dim
        ) | _get_next_actions_mask(token_labels, self.action_dim)
        text_hidden = output.hidden_states[-1][:, self.num_patches : -1]
        expected = self.action_horizon * self.action_dim
        if not torch.all(action_mask.sum(dim=1) == expected):
            raise AssertionError(
                f"OpenVLA action mask must contain {expected} positions"
            )
        hidden = text_hidden[action_mask].reshape(
            batch["input_ids"].shape[0], expected, -1
        )
        prediction = self.action_head(hidden.to(torch.bfloat16))
        loss = F.l1_loss(prediction.float(), batch["action"].float())
        return loss, prediction

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Batched parallel decoding equivalent to OFT predict_action."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, prefix_length = input_ids.shape
        count = self.action_horizon * self.action_dim
        placeholder = torch.ones(
            batch_size, count, dtype=input_ids.dtype, device=input_ids.device
        )
        stop = torch.full(
            (batch_size, 1),
            STOP_INDEX,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, placeholder, stop], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    batch_size,
                    count + 1,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[:, prefix_length : prefix_length + count] = (
            ACTION_TOKEN_BEGIN_IDX + 1
        )
        labels[:, -1] = STOP_INDEX
        input_embeddings = self.vla.get_input_embeddings()(input_ids)
        action_mask = _get_current_action_mask(
            labels, self.action_dim
        ) | _get_next_actions_mask(labels, self.action_dim)
        language_embeddings = input_embeddings[~action_mask].reshape(
            batch_size, -1, input_embeddings.shape[-1]
        )
        projected_patches = self.vla._process_vision_features(
            batch["pixel_values"].to(torch.bfloat16),
            language_embeddings,
            False,
        )
        if self.proprio_projector is not None:
            projected_patches = self.vla._process_proprio_features(
                projected_patches,
                batch["observation.state"],
                self.proprio_projector,
            )
        input_embeddings = input_embeddings * ~action_mask.unsqueeze(-1)
        embeddings, multimodal_mask = self.vla._build_multimodal_attention(
            input_embeddings, projected_patches, attention_mask
        )
        output = self.vla.language_model(
            input_ids=None,
            attention_mask=multimodal_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        start = self.num_patches + prefix_length - 1
        hidden = output.hidden_states[-1][:, start : start + count]
        return self.action_head(hidden.to(torch.bfloat16))


def make_policy_and_processor(args: argparse.Namespace, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    configuration, processing, modeling = _load_oft_runtime(
        args.oft_source,
        args.action_horizon,
        args.history_horizon * 10,
    )
    config = configuration.OpenVLAConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    image_processor = processing.PrismaticImageProcessor.from_pretrained(
        args.model_path
    )
    processor = processing.PrismaticProcessor(image_processor, tokenizer)
    vla = modeling.OpenVLAForActionPrediction.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    vla.vision_backbone.set_num_images_in_input(1)
    vla.config.use_cache = False
    # The OFT-compatible Transformers source used on this cluster also carries
    # optional token-pruning hooks.  Vanilla OpenVLA configs predate those
    # fields; explicitly disabling the hook preserves the stock Llama path.
    vla.language_model.config.proportion_attn_var = None
    if hasattr(vla, "gradient_checkpointing_enable"):
        # Reentrant checkpointing fires the same LoRA parameter hook more than
        # once under DDP.  The non-reentrant implementation has identical
        # activation-memory intent and is supported by current Transformers.
        vla.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=min(args.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    if hasattr(vla, "enable_input_require_grads"):
        vla.enable_input_require_grads()
    llm_dim = int(vla.llm_dim)
    action_head = OFTL1ActionHead(llm_dim, args.action_horizon, 10)
    history_dim = args.history_horizon * 10
    proprio_projector = (
        ProprioProjector(llm_dim, history_dim) if history_dim else None
    )
    policy = OpenVLAOFTPolicy(
        vla,
        action_head,
        proprio_projector,
        args.action_horizon,
        10,
    ).to(device)
    policy.action_head.to(torch.bfloat16)
    if policy.proprio_projector is not None:
        policy.proprio_projector.to(torch.bfloat16)
    return policy, processor


def load_trainable_checkpoint(policy, checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OpenVLA-OFT eval checkpoint not found: {checkpoint_path}")
    state = load_file(checkpoint_path, device="cpu")
    missing, unexpected = policy.load_state_dict(state, strict=False)
    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(trainable_names.difference(state))
    if unexpected or missing_trainable:
        raise RuntimeError(
            "OpenVLA-OFT eval checkpoint did not load cleanly: "
            f"missing_trainable={missing_trainable[:8]}, unexpected={unexpected[:8]}"
        )
    print(
        f"[OpenVLA-OFT eval] loaded {len(state)} trainable tensors from {checkpoint_path}",
        flush=True,
    )


class OpenVLAOFTBatchProcessor:
    def __init__(
        self,
        stats: dict,
        processor,
        device: torch.device,
        use_history: bool = True,
    ) -> None:
        self.stats = stats
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.device = device
        self.use_history = use_history
        self.action_bins = np.linspace(-1, 1, 256)

    @staticmethod
    def _normalize(value: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        low = stats["q01"].to(value.device, value.dtype)
        high = stats["q99"].to(value.device, value.dtype)
        return (2.0 * (value - low) / (high - low).clamp_min(1.0e-8) - 1.0).clamp(
            -1.0, 1.0
        )

    def _action_ids(self, normalized_action: np.ndarray) -> list[int]:
        discretized = np.digitize(normalized_action, self.action_bins)
        return (
            self.tokenizer.vocab_size - discretized.reshape(-1)
        ).astype(np.int64).tolist()

    def _pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        output = []
        for image in images:
            array = (
                image.detach()
                .cpu()
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            output.append(
                self.image_processor.apply_transform(Image.fromarray(array))
            )
        return torch.stack(output).to(self.device, non_blocking=True)

    def __call__(
        self, batch: dict[str, Any], include_action: bool = True
    ) -> dict[str, torch.Tensor]:
        prefixes = [
            f"In: What action should the robot take to {task.strip()}?\nOut: "
            for task in batch["task"]
        ]
        prefix_ids = [
            self.tokenizer(prefix, add_special_tokens=True).input_ids
            for prefix in prefixes
        ]
        if include_action:
            action = batch["action"].to(self.device).float()
            normalized_action = self._normalize(action, self.stats["action"])
            sequences, labels = [], []
            for prefix, normalized in zip(
                prefix_ids,
                normalized_action.detach().cpu().numpy(),
                strict=True,
            ):
                action_ids = self._action_ids(normalized)
                sequences.append(
                    torch.tensor(
                        prefix + action_ids + [self.tokenizer.eos_token_id],
                        dtype=torch.long,
                    )
                )
                labels.append(
                    torch.tensor(
                        [IGNORE_INDEX] * len(prefix)
                        + action_ids
                        + [self.tokenizer.eos_token_id],
                        dtype=torch.long,
                    )
                )
            input_ids = pad_sequence(
                sequences,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            label_ids = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
        else:
            input_ids = pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in prefix_ids],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            label_ids = None
            normalized_action = None
        processed = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).to(
                self.device
            ),
            "pixel_values": self._pixel_values(
                batch["observation.images.current_frame"]
            ),
        }
        if self.use_history:
            state = batch["observation.state"].to(self.device).float()
            state = self._normalize(state, self.stats["observation.state"])
            processed["observation.state"] = state.to(torch.bfloat16)
        if include_action:
            processed["labels"] = label_ids.to(self.device)
            processed["action"] = normalized_action.to(self.device)
        return processed

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        stats = self.stats["action"]
        low = stats["q01"].to(action.device, action.dtype)
        high = stats["q99"].to(action.device, action.dtype)
        return (action + 1.0) * (high - low) / 2.0 + low


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
    history_dim = args.history_horizon * 10
    if history_dim not in (0, 40) or args.action_horizon != 16:
        raise ValueError(
            "controlled OpenVLA-OFT protocol requires history=0 or 40 and horizon=16"
        )
    if args.sample_start_step < args.history_horizon:
        raise ValueError("sample-start-step must cover history-horizon")
    language_map, language_metadata = load_language_map(args.language_map)
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
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
            sample_start_step=args.sample_start_step,
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
    if history_dim == 0 and input_audit["observation.state"].numel() != 0:
        raise AssertionError("history-free dataset unexpectedly contains state")
    stats = compute_stats(train_base, args.stats_samples, args.split_seed)

    if accelerator.is_main_process:
        instructions = sorted(
            {language_map[episode.key]["instruction"] for episode in train_base.episodes}
        )
        metadata = {
            "method": "OpenVLA-7B + official OFT L1 continuous head",
            "prompt_protocol": (
                "task-specific language + current RGB"
                + (" + action history" if history_dim else " + no history")
                + "; no first frame or boxes"
            ),
            "instruction": {
                "template": "pick {object} and place to {target}",
                "unique_train_instructions": len(instructions),
                "examples": instructions[:10],
            },
            "input_modalities": ["current_rgb"]
            + (["history"] if history_dim else [])
            + ["task_specific_language"],
            "model_input_contains_first_frame": False,
            "model_input_contains_visual_boxes": False,
            "model_input_contains_bbox_coordinates": False,
            "model_input_contains_action_history": bool(history_dim),
            "language_map": language_metadata,
            "action_target": [16, 10],
            "history_state_dim": history_dim,
            "sample_start_step": args.sample_start_step,
            "continuous_action_head": "OFT L1 parallel decoding",
            "lora_rank": args.lora_rank,
            "libero_task_specific_head_loaded": False,
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
            f"[OpenVLA-OFT data] train={len(train_dataset)} val={len(val_dataset)}",
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
    policy, hf_processor = make_policy_and_processor(args, accelerator.device)
    processor = OpenVLAOFTBatchProcessor(
        stats,
        hf_processor,
        accelerator.device,
        use_history=bool(history_dim),
    )
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
            return 0.1 + 0.9 * step / warmup_steps
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
            f"[OpenVLA-OFT train] world={accelerator.num_processes} "
            f"batch={args.batch_size} accum={args.grad_accum_steps} "
            f"effective_batch={args.batch_size * accelerator.num_processes * args.grad_accum_steps} "
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
                record = {
                    "step": update_step,
                    "loss": loss_sum / max(1, loss_count),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": elapsed,
                }
                print(f"[OpenVLA-OFT train] {json.dumps(record)}", flush=True)
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
