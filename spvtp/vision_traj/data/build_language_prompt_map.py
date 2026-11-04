"""Build episode-level language prompts for language-only VLA baselines.

Scene 1/2 expose semantic task names such as ``put_fork3_to_plate2``. Scene 3
only exposes ``subsceneN`` even though each subscene contains the full 5 x 9
object/target design. For Scene 3, first-frame boxes are used offline to recover
stable local instance IDs with capacity-constrained clustering. The boxes are
never part of the VLA input.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


TASK_PATTERN = re.compile(r"^put_(?P<object>.+)_to_(?P<target>.+)$")
PROMPT_TEMPLATE = "pick {object} and place to {target}"


def normalize_episode_key(key: str) -> str:
    return key.replace("\\", "/")


def named_task_instruction(task: str) -> tuple[str, str, str]:
    """Return object, target, and instruction parsed from a semantic task name."""
    match = TASK_PATTERN.fullmatch(task)
    if match is None:
        raise ValueError(f"task has no object/target semantics: {task!r}")
    object_name = match.group("object").replace("_", " ")
    target_name = match.group("target").replace("_", " ")
    return (
        object_name,
        target_name,
        PROMPT_TEMPLATE.format(object=object_name, target=target_name),
    )


def drop_instance_id(name: str) -> str:
    """Remove a trailing numeric instance ID while preserving the category name."""
    return re.sub(r"(?<=\D)\d+$", "", name).rstrip()


def _box_center(box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def _kmeans_plus_plus(
    points: np.ndarray, clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = [points[int(rng.integers(len(points)))]]
    while len(centers) < clusters:
        distance_sq = np.min(
            np.sum((points[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        total = float(distance_sq.sum())
        candidate = (
            int(rng.integers(len(points)))
            if total <= 0.0
            else int(rng.choice(len(points), p=distance_sq / total))
        )
        centers.append(points[candidate])
    return np.asarray(centers)


def balanced_kmeans(
    points: np.ndarray,
    clusters: int,
    *,
    seed: int = 0,
    restarts: int = 64,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cluster 2-D points while enforcing the known balanced task design."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected [N,2] points, got {points.shape}")
    if len(points) < clusters:
        raise ValueError(f"cannot form {clusters} clusters from {len(points)} points")

    scale = points.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    normalized = (points - points.mean(axis=0)) / scale
    capacities = np.full(clusters, len(points) // clusters, dtype=np.int64)
    capacities[: len(points) % clusters] += 1
    slot_to_cluster = np.repeat(np.arange(clusters), capacities)
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for _ in range(restarts):
        centers = _kmeans_plus_plus(normalized, clusters, rng)
        labels = np.full(len(points), -1, dtype=np.int64)
        for _ in range(max_iterations):
            slot_centers = centers[slot_to_cluster]
            costs = np.sum(
                (normalized[:, None, :] - slot_centers[None, :, :]) ** 2, axis=-1
            )
            rows, slots = linear_sum_assignment(costs)
            next_labels = np.empty_like(labels)
            next_labels[rows] = slot_to_cluster[slots]
            next_centers = np.stack(
                [normalized[next_labels == index].mean(axis=0) for index in range(clusters)]
            )
            if np.array_equal(next_labels, labels):
                centers, labels = next_centers, next_labels
                break
            centers, labels = next_centers, next_labels

        inertia = float(np.sum((normalized - centers[labels]) ** 2))
        if best is None or inertia < best[0]:
            best = inertia, labels.copy(), centers.copy()

    assert best is not None
    inertia, labels, normalized_centers = best
    centers = normalized_centers * scale + points.mean(axis=0)
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    old_to_new = np.empty(clusters, dtype=np.int64)
    old_to_new[order] = np.arange(clusters)
    labels = old_to_new[labels]
    centers = centers[order]

    all_distances = np.linalg.norm(
        normalized[:, None, :] - normalized_centers[None, :, :], axis=-1
    )
    nearest = np.partition(all_distances, 1, axis=1)[:, :2]
    diagnostics = {
        "samples": int(len(points)),
        "clusters": int(clusters),
        "counts": np.bincount(labels, minlength=clusters).tolist(),
        "centers_xy": centers.tolist(),
        "normalized_inertia": inertia,
        "mean_nearest_to_second_margin": float(np.mean(nearest[:, 1] - nearest[:, 0])),
    }
    return labels, centers, diagnostics


def build_prompt_map(
    annotations_path: str | Path, *, drop_object_id: bool = False
) -> dict[str, Any]:
    annotations_path = Path(annotations_path)
    with annotations_path.open() as handle:
        raw_annotations = json.load(handle)
    annotations = {
        normalize_episode_key(key): value["0"]
        for key, value in raw_annotations.items()
        if "0" in value
    }

    episodes: dict[str, dict[str, Any]] = {}
    scene3_groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, annotation in annotations.items():
        parts = key.split("/")
        if len(parts) < 4:
            continue
        scene, task = parts[0], parts[1]
        if scene == "scene3":
            scene3_groups[task].append((key, annotation))
            continue
        object_name, target_name, _ = named_task_instruction(task)
        prompt_object = drop_instance_id(object_name) if drop_object_id else object_name
        episodes[key] = {
            "scene": scene,
            "task": task,
            "object": object_name,
            "prompt_object": prompt_object,
            "target": target_name,
            "instruction": PROMPT_TEMPLATE.format(
                object=prompt_object, target=target_name
            ),
            "label_source": "semantic_task_directory",
        }

    subscene_diagnostics: dict[str, dict[str, Any]] = {}
    for task in sorted(
        scene3_groups, key=lambda name: int(name.removeprefix("subscene"))
    ):
        rows = sorted(scene3_groups[task])
        object_points = np.stack([_box_center(item["object"]) for _, item in rows])
        target_points = np.stack([_box_center(item["target"]) for _, item in rows])
        object_labels, _, object_diagnostics = balanced_kmeans(object_points, 5, seed=0)
        target_labels, _, target_diagnostics = balanced_kmeans(target_points, 9, seed=1)
        subscene_diagnostics[task] = {
            "object": object_diagnostics,
            "target": target_diagnostics,
        }
        for row_index, (key, _) in enumerate(rows):
            object_name = f"fork{int(object_labels[row_index]) + 1}"
            target_name = f"target{int(target_labels[row_index]) + 1}"
            prompt_object = drop_instance_id(object_name) if drop_object_id else object_name
            episodes[key] = {
                "scene": "scene3",
                "task": task,
                "object": object_name,
                "prompt_object": prompt_object,
                "target": target_name,
                "instruction": PROMPT_TEMPLATE.format(
                    object=prompt_object, target=target_name
                ),
                "label_source": "balanced_box_clustering_for_language_annotation_only",
            }

    return {
        "protocol": "language_only",
        "template": PROMPT_TEMPLATE,
        "boxes_enter_model": False,
        "object_instance_id_in_prompt": not drop_object_id,
        "target_instance_id_in_prompt": True,
        "language_ablation": (
            "drop_trailing_object_instance_id"
            if drop_object_id
            else "none"
        ),
        "scene3_id_scope": "local_to_each_subscene",
        "scene3_design_constraint": "5 objects x 9 targets",
        "annotations": str(annotations_path),
        "episodes": episodes,
        "scene3_cluster_diagnostics": subscene_diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="annotations_merged.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--drop-object-id",
        action="store_true",
        help="Render object categories without their trailing numeric instance ID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_prompt_map(args.annotations, drop_object_id=args.drop_object_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(
        f"[language-map] episodes={len(result['episodes'])} "
        f"scene3_subscenes={len(result['scene3_cluster_diagnostics'])} output={output}"
    )


if __name__ == "__main__":
    main()
