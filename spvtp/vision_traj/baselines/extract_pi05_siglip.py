"""Extract the pi0.5 SigLIP vision tower into strict timm-format weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_PREFIX = (
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="official pi0.5 model.safetensors")
    parser.add_argument("--output", required=True, help="output timm-format safetensors")
    return parser.parse_args()


def extract(source: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())

        def tensor(name: str) -> torch.Tensor:
            source_name = SOURCE_PREFIX + name
            if source_name not in available:
                raise KeyError(f"missing pi0.5 vision tensor: {source_name}")
            return checkpoint.get_tensor(source_name).contiguous()

        output = {
            "pos_embed": tensor("embeddings.position_embedding.weight").unsqueeze(0),
            "patch_embed.proj.weight": tensor("embeddings.patch_embedding.weight"),
            "patch_embed.proj.bias": tensor("embeddings.patch_embedding.bias"),
        }
        for layer in range(27):
            source_layer = f"encoder.layers.{layer}."
            target_layer = f"blocks.{layer}."
            for source_name, target_name in (
                ("layer_norm1.weight", "norm1.weight"),
                ("layer_norm1.bias", "norm1.bias"),
                ("self_attn.out_proj.weight", "attn.proj.weight"),
                ("self_attn.out_proj.bias", "attn.proj.bias"),
                ("layer_norm2.weight", "norm2.weight"),
                ("layer_norm2.bias", "norm2.bias"),
                ("mlp.fc1.weight", "mlp.fc1.weight"),
                ("mlp.fc1.bias", "mlp.fc1.bias"),
                ("mlp.fc2.weight", "mlp.fc2.weight"),
                ("mlp.fc2.bias", "mlp.fc2.bias"),
            ):
                output[target_layer + target_name] = tensor(source_layer + source_name)
            output[target_layer + "attn.qkv.weight"] = torch.cat(
                [
                    tensor(source_layer + f"self_attn.{name}_proj.weight")
                    for name in ("q", "k", "v")
                ],
                dim=0,
            )
            output[target_layer + "attn.qkv.bias"] = torch.cat(
                [
                    tensor(source_layer + f"self_attn.{name}_proj.bias")
                    for name in ("q", "k", "v")
                ],
                dim=0,
            )
        output["norm.weight"] = tensor("post_layernorm.weight")
        output["norm.bias"] = tensor("post_layernorm.bias")
    return output


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = extract(source)
    save_file(
        state,
        str(output),
        metadata={
            "source": str(source),
            "source_prefix": SOURCE_PREFIX,
            "architecture": "pi0.5 PaliGemma SigLIP-So400M/14 224",
            "format": "timm vit_so400m_patch14_siglip_224 global_pool=''",
        },
    )
    manifest = {
        "source": str(source),
        "source_size_bytes": source.stat().st_size,
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "tensor_count": len(state),
        "architecture": "SigLIP-So400M/14, 224px, 27 layers, 1152 hidden",
    }
    with output.with_suffix(".json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
