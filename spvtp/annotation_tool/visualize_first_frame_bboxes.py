"""
Visualize frame-0 bounding boxes from annotations_merged.json.

For each annotated video, this script reads the first frame from the video file,
draws the frame-0 object/target boxes, and saves the result under an output
directory while preserving the relative directory structure.

Usage:
    python visualize_first_frame_bboxes.py
    python visualize_first_frame_bboxes.py \
        --annotations annotations_merged.json \
        --video-root video_data \
        --output-root bounding_box_vis
"""

import argparse
import json
from pathlib import Path

import cv2


BOX_COLORS = {
    "object": (0, 255, 0),
    "target": (0, 165, 255),
}


def normalize_video_key(video_key: str) -> Path:
    return Path(video_key.replace("\\", "/"))


def draw_box(image, box, label: str, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    text_x = max(0, x1)
    text_y = max(text_h + baseline + 4, y1)
    bg_tl = (text_x, text_y - text_h - baseline - 6)
    bg_br = (text_x + text_w + 10, text_y)
    cv2.rectangle(image, bg_tl, bg_br, color, -1)
    cv2.putText(
        image,
        label,
        (text_x + 5, text_y - baseline - 3),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def extract_boxes(frame0_data: dict) -> list[tuple[str, list[int]]]:
    boxes: list[tuple[str, list[int]]] = []

    for label in ("object", "target"):
        box = frame0_data.get(label)
        if isinstance(box, list) and len(box) == 4:
            boxes.append((label, box))

    if boxes:
        return boxes

    generic_boxes = frame0_data.get("boxes", [])
    if isinstance(generic_boxes, list):
        labels = ["object", "target"]
        for idx, box in enumerate(generic_boxes):
            if isinstance(box, list) and len(box) == 4:
                label = labels[idx] if idx < len(labels) else f"box_{idx}"
                boxes.append((label, box))

    return boxes


def process_video(video_key: str, annotations: dict, video_root: Path, output_root: Path):
    frame0_data = annotations.get("0")
    if not isinstance(frame0_data, dict):
        return "skip_no_frame0"

    rel_video_path = normalize_video_key(video_key)
    video_path = video_root / rel_video_path
    if not video_path.exists():
        return f"missing_video:{video_path}"

    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        return f"failed_read:{video_path}"

    for label, box in extract_boxes(frame0_data):
        color = BOX_COLORS.get(label, (255, 0, 0))
        draw_box(frame, box, label, color)

    output_path = output_root / rel_video_path.with_suffix(".jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        return f"failed_write:{output_path}"

    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Draw frame-0 bounding boxes on the first frame of each video."
    )
    parser.add_argument(
        "--annotations",
        default="annotations_merged.json",
        help="Merged annotations JSON path.",
    )
    parser.add_argument(
        "--video-root",
        default="video_data",
        help="Root directory containing video files.",
    )
    parser.add_argument(
        "--output-root",
        default="bounding_box_vis",
        help="Directory to save visualization images.",
    )
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)

    with annotations_path.open() as f:
        all_annotations = json.load(f)

    counts = {
        "ok": 0,
        "skip_no_frame0": 0,
        "skip_duplicate": 0,
        "missing_video": 0,
        "failed_read": 0,
        "failed_write": 0,
    }
    errors = []
    seen_output_paths: set[Path] = set()

    for video_key, annotations in all_annotations.items():
        if not isinstance(annotations, dict):
            errors.append(f"invalid_annotations:{video_key}")
            continue

        rel_video_path = normalize_video_key(video_key)
        output_path = output_root / rel_video_path.with_suffix(".jpg")
        if output_path in seen_output_paths:
            counts["skip_duplicate"] += 1
            errors.append(f"duplicate_output:{video_key} -> {output_path}")
            continue
        seen_output_paths.add(output_path)

        result = process_video(video_key, annotations, video_root, output_root)
        if result == "ok":
            counts["ok"] += 1
        elif result == "skip_no_frame0":
            counts["skip_no_frame0"] += 1
        elif result == "skip_duplicate":
            counts["skip_duplicate"] += 1
        elif result.startswith("missing_video:"):
            counts["missing_video"] += 1
            errors.append(result)
        elif result.startswith("failed_read:"):
            counts["failed_read"] += 1
            errors.append(result)
        elif result.startswith("failed_write:"):
            counts["failed_write"] += 1
            errors.append(result)
        else:
            errors.append(result)

    print(
        "Done. "
        f"saved={counts['ok']} "
        f"skip_no_frame0={counts['skip_no_frame0']} "
        f"skip_duplicate={counts['skip_duplicate']} "
        f"missing_video={counts['missing_video']} "
        f"failed_read={counts['failed_read']} "
        f"failed_write={counts['failed_write']}"
    )

    if errors:
        print("\nIssues:")
        for item in errors[:20]:
            print(f"  - {item}")
        if len(errors) > 20:
            print(f"  - ... {len(errors) - 20} more")


if __name__ == "__main__":
    main()
