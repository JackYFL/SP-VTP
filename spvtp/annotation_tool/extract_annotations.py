"""
Extract all *_annotations.json files under a root directory and merge them.

Each entry records the video path relative from the scenex folder (scene1, scene2, ...).

Usage:
    python extract_annotations.py --root /path/to/data --output /path/to/output.json
"""

import argparse
import json
import re
from pathlib import Path


def find_scene_root(path: Path) -> Path | None:
    """Return the part of the path up to and including the scenex folder."""
    for part_idx, part in enumerate(path.parts):
        if re.fullmatch(r"scene\d+", part, re.IGNORECASE):
            return Path(*path.parts[: part_idx + 1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Merge annotation JSON files")
    parser.add_argument("--root",   required=True, help="Root directory to search")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    merged = {}
    json_files = sorted(root.rglob("*_annotations.json"))

    if not json_files:
        print(f"No annotation files found under {root}")
        return

    for json_file in json_files:
        scene_root = find_scene_root(json_file)
        if scene_root is None:
            print(f"  [skip] no scenex folder in path: {json_file}")
            continue

        # video path: same as json but strip _annotations.json -> .mp4
        video_path = json_file.with_name(
            json_file.name.replace("_annotations.json", ".mp4")
        )
        # relative path from scene root's parent (keeps scenex in the key)
        rel_video = video_path.relative_to(scene_root.parent)

        with open(json_file) as f:
            data = json.load(f)

        merged[str(rel_video)] = data
        print(f"  [ok] {rel_video}")

    with open(output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nDone. {len(merged)} files -> {output}")


if __name__ == "__main__":
    main()
