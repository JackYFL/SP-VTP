# Video Bounding Box Annotator

A lightweight GUI tool for annotating **object** and **target** bounding boxes on video frames. Annotations are saved as JSON files alongside the video.

## Installation

```bash
cd /path/to/SP-VTP
pip install -e ".[annotation]"
```


## Usage

```bash
# Open with a specific video file
python -m spvtp.annotation_tool.video_bbox_annotator --video path/to/video.mp4

# Open with a file dialog (no arguments)
python -m spvtp.annotation_tool.video_bbox_annotator
```

## Interface

### Buttons

| Button | Action |
|--------|--------|
| **📂 Open Video** | Open a new video file via file dialog |
| **✏️ Annotate** | Jump to frame 0 to start annotating |
| **◀ Prev / Next ▶** | Step one frame backward / forward |
| **▶ Play / ⏸ Pause** | Toggle video playback |
| **💾 Save (S)** | Save annotations to JSON |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next frame |
| `Space` | Play / pause |
| `S` | Save annotations |
| `Q` / `Esc` | Quit |

### Timeline

Click or drag anywhere on the blue seek bar to jump to that point in the video.

### Drawing Boxes

Bounding boxes can only be drawn on **frame 0**.

- **Left-click and drag** — draw a bounding box
- **Right-click** on an existing box — delete it

Each frame supports exactly **two** boxes:

| Order | Role | Color |
|-------|------|-------|
| 1st box | **Object** | Green `#00E676` |
| 2nd box | **Target** | Orange `#FF9800` |

The status bar shows `Object: ✓/✗  Target: ✓/✗` to indicate which boxes have been drawn.

## Extracting Annotations

After annotating, merge all `*_annotations.json` files under a root directory into a single JSON:

```bash
python -m spvtp.annotation_tool.extract_annotations \
  --root /path/to/data \
  --output /path/to/output.json
```

The root directory is expected to contain `scenex` folders (e.g. `scene1`, `scene2`):

```text
<root>/scene1/putxtox/episode_1/xxx.mp4
<root>/scene1/putxtox/episode_1/xxx_annotations.json
```

The merged output uses the path from `scenex` as the key:

```json
{
  "scene1/putxtox/episode_1/xxx.mp4": {
    "0": {
      "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1]],
      "object": [x0, y0, x1, y1],
      "target": [x0, y0, x1, y1]
    }
  }
}
```

## Modifying Merged Annotations

Use `modify_first_frame_bbox.py` to edit the frame-0 bounding boxes directly inside `annotations_merged.json`.

```bash
# Open with the default merged annotations file and video root
python -m spvtp.annotation_tool.modify_first_frame_bbox

# Open with explicit paths
python -m spvtp.annotation_tool.modify_first_frame_bbox \
  --annotations annotations_merged.json \
  --video-root video_data
```

This tool:

- loads videos from `annotations_merged.json`
- displays the video and allows frame browsing / playback
- allows editing bounding boxes only on frame `0`
- writes changes back to the original `annotations_merged.json` in place

### Modify Interface

| Control | Action |
|--------|--------|
| **Left sidebar** | Jump directly to a directory from `annotations_merged.json` |
| **◀ Prev Video / Next Video ▶** | Switch to previous / next video |
| **▶ Play / ⏸ Pause** | Play or pause the current video |
| **⤒ Frame 0** | Jump back to frame 0 |
| **💾 Save (S)** | Save the current merged annotations file |
| **Timeline** | Click or drag to jump to any frame |
| **－ / Fit / ＋** | Zoom out / fit to window / zoom in |

### Modify Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / next frame |
| `↑` / `↓` | Previous / next video |
| `Space` | Play / pause |
| `+` / `-` / `0` | Zoom in / zoom out / reset view |
| `S` | Save merged annotations |
| `Q` / `Esc` | Quit |

### Modify Boxes

Bounding boxes can only be edited on **frame 0**.

- **Left-click and drag** — create a box
- **Drag inside a box** — move the box
- **Drag box handles** — resize the box
- **Right-click a box** — delete the box
- **Mouse wheel** — zoom the canvas view
- **Shift + drag** — pan when zoomed in

The edited data is written back to the existing merged entry:

```json
{
  "scene1/putxtox/episode_1/xxx.mp4": {
    "0": {
      "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1]],
      "object": [x0, y0, x1, y1],
      "target": [x0, y0, x1, y1]
    }
  }
}
```

## Output Format

Annotations are saved to `<video_name>_annotations.json` in the same directory as the video.

```json
{
  "0": {
    "boxes": [[x0, y0, x1, y1], [x0, y0, x1, y1]],
    "object": [x0, y0, x1, y1],
    "target": [x0, y0, x1, y1]
  }
}
```

- `boxes` — list of all boxes in draw order
- `object` — bounding box of the object (first drawn box)
- `target` — bounding box of the target location (second drawn box)

All coordinates are in **pixel space** relative to the original video resolution.

If an annotation file already exists for the loaded video, it is loaded automatically on startup.
