#!/usr/bin/env python3
"""Collect vision_traj experiment summaries into one xlsx workbook.

This script intentionally avoids pandas/openpyxl so it can run in the current
project environment. It reads:

    runs/vision_traj_experiments/<run>/experiment_meta.json
    runs/vision_traj_experiments/<run>/eval_metrics/<scene>/summary.json

and writes:

    runs/vision_traj_experiments/experiment_results.xlsx
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


RESULT_HEADERS = [
    "experiment_group",
    "experiment_name",
    "run_name",
    "scene",
    "final_epoch",
    "final_displacement_error",
    "pos_L2",
    "rot6d_L2",
    "grip_L1",
    "checkpoint_path",
    "wandb_url",
    "notes",
]

VIS_HEADERS = [
    "run_name",
    "scene",
    "episode_id",
    "checkpoint_path",
    "full_trajectory_pdf_path",
    "first_chunk_pdf_path",
    "pos_L2",
    "rot6d_L2",
    "grip_L1",
    "final_displacement_error",
    "notes",
]

SHEET_ORDER = [
    "baseline",
    "ablation_vision_encoder_size",
    "ablation_dinov2_base_tuning",
    "ablation_vision_encoder_type",
    "ablation_head",
    "ablation_fusion",
    "ablation_history",
    "visualization_index",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/vision_traj_experiments")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _metric_rows(root: Path) -> dict[str, list[list[object]]]:
    rows_by_sheet = {sheet: [RESULT_HEADERS] for sheet in SHEET_ORDER if sheet != "visualization_index"}
    for meta_path in sorted(root.glob("*/experiment_meta.json")):
        run_dir = meta_path.parent
        meta = _read_json(meta_path)
        group = str(meta.get("experiment_group", "unknown"))
        if group not in rows_by_sheet:
            rows_by_sheet[group] = [RESULT_HEADERS]
        ckpt = run_dir / "ckpt_final.pt"
        for scene_dir in sorted((run_dir / "eval_metrics").glob("*")):
            summary_path = scene_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = _read_json(summary_path)
            rows_by_sheet[group].append(
                [
                    group,
                    meta.get("experiment_name", ""),
                    meta.get("run_name", run_dir.name),
                    scene_dir.name,
                    summary.get("epoch", ""),
                    summary.get("mean_final_displacement_error", ""),
                    summary.get("mean_pos_l2", ""),
                    summary.get("mean_rot6d_l2", ""),
                    summary.get("mean_grip_l1", ""),
                    str(ckpt),
                    meta.get("wandb_url", ""),
                    "",
                ]
            )
    return rows_by_sheet


def _scene_from_path(path: Path) -> str:
    for part in path.parts:
        if part in {"scene1", "scene2", "scene3"}:
            return part
    return ""


def _episode_from_name(path: Path) -> str:
    match = re.search(r"episode[_-](\d+)|ep(\d+)", path.stem)
    if not match:
        return ""
    return next(g for g in match.groups() if g is not None)


def _visualization_rows(root: Path) -> list[list[object]]:
    rows = [VIS_HEADERS]
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "experiment_meta.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        run_name = meta.get("run_name", run_dir.name)
        ckpt = run_dir / "ckpt_final.pt"
        full_pdfs = sorted((run_dir / "visualizations").glob("**/*stitched.pdf"))
        first_chunk_pdfs = sorted((run_dir / "visualizations").glob("**/*.pdf"))
        first_chunk_pdfs = [p for p in first_chunk_pdfs if "stitched" not in p.name]
        for pdf in full_pdfs:
            scene = _scene_from_path(pdf)
            row = [
                run_name,
                scene,
                _episode_from_name(pdf),
                str(ckpt),
                str(pdf),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
            rows.append(row)
        for pdf in first_chunk_pdfs:
            scene = _scene_from_path(pdf)
            row = [
                run_name,
                scene,
                _episode_from_name(pdf),
                str(ckpt),
                "",
                str(pdf),
                "",
                "",
                "",
                "",
                "",
            ]
            rows.append(row)
    return rows


def _col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell_xml(value: object, row: int, col: int) -> str:
    ref = f"{_col_name(col)}{row}"
    if value is None:
        value = ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(rows: list[list[object]]) -> str:
    row_xml = []
    for r_idx, row in enumerate(rows, start=1):
        cells = "".join(_cell_xml(value, r_idx, c_idx) for c_idx, value in enumerate(row))
        row_xml.append(f'<row r="{r_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(row_xml)
        + '</sheetData></worksheet>'
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        + "".join(sheets)
        + '</sheets></workbook>'
    )


def _workbook_rels_xml(sheet_names: list[str]) -> str:
    rels = []
    for idx, _ in enumerate(sheet_names, start=1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + '</Relationships>'
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def _content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + '</Types>'
    )


def write_xlsx(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    sheet_names = list(sheets)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(sheet_names)))
        zf.writestr("_rels/.rels", _root_rels_xml())
        zf.writestr("xl/workbook.xml", _workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(sheet_names))
        for idx, name in enumerate(sheet_names, start=1):
            zf.writestr(posixpath.join("xl", "worksheets", f"sheet{idx}.xml"), _sheet_xml(sheets[name]))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "experiment_results.xlsx"
    sheets = _metric_rows(root)
    sheets["visualization_index"] = _visualization_rows(root)
    ordered = {}
    for name in SHEET_ORDER:
        if name in sheets:
            ordered[name] = sheets[name]
    for name, rows in sheets.items():
        if name not in ordered:
            ordered[name] = rows
    write_xlsx(out, ordered)
    print(f"[collect_results_xlsx] wrote {out}")


if __name__ == "__main__":
    main()
