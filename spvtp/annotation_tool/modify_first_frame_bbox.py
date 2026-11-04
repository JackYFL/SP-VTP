"""
Modify frame-0 bounding boxes directly in annotations_merged.json.

This tool loads merged annotations, shows the video frames for each entry, and
lets you create, move, resize, or delete the frame-0 object/target boxes.
Edits are written back to the original merged JSON file in place.

Usage:
    python modify_first_frame_bbox.py
    python modify_first_frame_bbox.py \
        --annotations annotations_merged.json \
        --video-root video_data
"""

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk


ROLE_STYLE = {
    "object": ("#00E676", "Object"),
    "target": ("#FF9800", "Target"),
}

STATE_PATH = Path(".modify_first_frame_bbox_state.json")
DEFAULT_VIDEO_ROOT = Path("video_data")


def normalize_video_key(video_key: str) -> Path:
    return Path(video_key.replace("\\", "/"))


def load_last_video_root() -> Path | None:
    state = load_state()
    last_root = state.get("last_video_root")
    if not isinstance(last_root, str) or not last_root.strip():
        return None
    return Path(last_root)


def load_last_selected_dir() -> str | None:
    state = load_state()
    last_dir = state.get("last_selected_dir")
    if not isinstance(last_dir, str) or not last_dir.strip():
        return None
    try:
        return str(Path(last_dir))
    except (OSError, json.JSONDecodeError):
        return None

def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with STATE_PATH.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(video_root: Path, selected_dir: str | None = None):
    with STATE_PATH.open("w") as f:
        json.dump(
            {
                "last_video_root": str(video_root),
                "last_selected_dir": selected_dir or "",
            },
            f,
            indent=2,
        )


def save_last_video_root(video_root: Path):
    state = load_state()
    save_state(video_root, state.get("last_selected_dir"))


def save_last_selected_dir(video_root: Path, selected_dir: str):
    save_state(video_root, selected_dir)


class FirstFrameBBoxEditor:
    def __init__(self, root: tk.Tk, annotations_path: Path, video_root: Path, start_key: str | None):
        self.root = root
        self.annotations_path = annotations_path
        self.video_root = video_root.resolve()

        with self.annotations_path.open() as f:
            self.all_annotations = json.load(f)

        self.video_keys: list[str] = []
        self.directory_keys: list[str] = []
        self.current_video_idx = 0
        self._refresh_video_keys(start_key)
        if not self.video_keys:
            raise ValueError(
                f"No matching videos found under {self.video_root} for {self.annotations_path}"
            )

        self.cap = None
        self._frame_image = None
        self._tk_img = None
        self._base_scale = 1.0
        self._scale = 1.0
        self._offset = (0.0, 0.0)
        self._zoom_factor = 1.0
        self._pan_offset = (0.0, 0.0)
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.playing = False
        self._play_after_id = None
        self._current_boxes: list[tuple[int, int, int, int]] = []
        self._status_after_id = None
        self._updating_dir_list = False

        self._drag_start = None
        self._drag_rect_id = None
        self._active_box_idx = None
        self._active_handle = None
        self._interaction_mode = None
        self._interaction_box = None
        self._move_offset = None
        self._handle_size = 6
        self._panning = False
        self._pan_start = None
        self._pan_origin = (0.0, 0.0)

        self._build_ui()
        self.root.update_idletasks()
        self._load_current_video()

    def _build_ui(self):
        self.root.title("Modify First Frame BBox")
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        self.root.bind("<Up>", lambda e: self._step_video(-1))
        self.root.bind("<Down>", lambda e: self._step_video(1))
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("+", lambda e: self._zoom_in())
        self.root.bind("=", lambda e: self._zoom_in())
        self.root.bind("-", lambda e: self._zoom_out())
        self.root.bind("0", lambda e: self._reset_view())
        self.root.bind("s", lambda e: self._save_annotations(show_message=True))
        self.root.bind("S", lambda e: self._save_annotations(show_message=True))
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.bind("q", lambda e: self._quit())

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        sidebar = tk.Frame(main_frame, width=320)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="Directories",
            anchor="w",
            font=("Helvetica", 11, "bold"),
        ).pack(fill="x", padx=8, pady=(8, 4))

        dir_list_frame = tk.Frame(sidebar)
        dir_list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.dir_listbox = tk.Listbox(dir_list_frame, exportselection=False)
        self.dir_listbox.pack(side="left", fill="both", expand=True)
        self.dir_listbox.bind("<<ListboxSelect>>", self._on_directory_selected)
        dir_scrollbar = tk.Scrollbar(dir_list_frame, command=self.dir_listbox.yview)
        dir_scrollbar.pack(side="right", fill="y")
        self.dir_listbox.config(yscrollcommand=dir_scrollbar.set)

        content = tk.Frame(main_frame)
        content.pack(side="left", fill="both", expand=True)

        self.info_var = tk.StringVar()
        tk.Label(content, textvariable=self.info_var, anchor="w", justify="left").pack(
            fill="x", padx=4, pady=2
        )

        self.root_var = tk.StringVar()
        tk.Label(
            content,
            textvariable=self.root_var,
            anchor="w",
            justify="left",
            fg="#444444",
        ).pack(fill="x", padx=4, pady=(0, 2))

        self.path_var = tk.StringVar()
        tk.Label(
            content,
            textvariable=self.path_var,
            anchor="w",
            justify="left",
            fg="#555555",
        ).pack(fill="x", padx=4, pady=(0, 2))

        controls_frame = tk.Frame(content)
        controls_frame.pack(side="bottom", fill="x")

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            controls_frame,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            fg="#1b5e20",
        )
        self.status_label.pack(fill="x", padx=4, pady=(0, 4))

        btn_frame = tk.Frame(controls_frame)
        btn_frame.pack(fill="x", pady=(0, 4))
        tk.Button(btn_frame, text="◀ Prev Video", width=14, command=lambda: self._step_video(-1)).pack(
            side="left", padx=4
        )
        tk.Button(btn_frame, text="Next Video ▶", width=14, command=lambda: self._step_video(1)).pack(
            side="left", padx=4
        )
        tk.Button(btn_frame, text="－", width=5, command=self._zoom_out).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Fit", width=6, command=self._reset_view).pack(side="left", padx=2)
        tk.Button(btn_frame, text="＋", width=5, command=self._zoom_in).pack(side="left", padx=2)
        self.play_btn = tk.Button(btn_frame, text="▶ Play", width=12, command=self._toggle_play)
        self.play_btn.pack(side="left", padx=4)
        tk.Button(btn_frame, text="⤒ Frame 0", width=12, command=lambda: self._show_frame(0)).pack(
            side="left", padx=4
        )
        tk.Button(btn_frame, text="💾 Save (S)", width=14, command=lambda: self._save_annotations(True)).pack(
            side="left", padx=12
        )

        self.seek_canvas = tk.Canvas(controls_frame, height=18, bg="#333333", cursor="hand2")
        self.seek_canvas.pack(fill="x", padx=4, pady=(0, 4))
        self.seek_canvas.bind("<ButtonPress-1>", self._on_seek_click)
        self.seek_canvas.bind("<B1-Motion>", self._on_seek_click)
        self.seek_canvas.bind("<Configure>", lambda e: self._draw_seekbar())

        self.canvas = tk.Canvas(content, cursor="crosshair", bg="black", width=1280, height=720)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Button-2>", self._on_right_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Shift-ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<Shift-ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_mousewheel_linux(e, zoom_in=True))
        self.canvas.bind("<Button-5>", lambda e: self._on_mousewheel_linux(e, zoom_in=False))

    def _refresh_video_keys(self, preferred_key: str | None = None):
        matching_keys = []
        for key in self.all_annotations.keys():
            if (self.video_root / normalize_video_key(key)).exists():
                matching_keys.append(key)

        matching_keys.sort(key=lambda key: str(normalize_video_key(key)).lower())
        if not matching_keys:
            raise ValueError(
                f"No matching videos found under {self.video_root} for {self.annotations_path}"
            )

        self.video_keys = matching_keys
        self.directory_keys = sorted(
            {self._directory_for_key(key) for key in self.video_keys},
            key=str.lower,
        )
        self.current_video_idx = self._find_video_index(preferred_key)

    def _find_video_index(self, preferred_key: str | None) -> int:
        if not preferred_key:
            return 0
        if preferred_key in self.video_keys:
            return self.video_keys.index(preferred_key)

        normalized_target = str(normalize_video_key(preferred_key))
        for idx, key in enumerate(self.video_keys):
            if str(normalize_video_key(key)) == normalized_target:
                return idx
        return 0

    def _set_status(self, message: str, duration_ms: int = 0, color: str = "#1b5e20"):
        self.status_var.set(message)
        self.status_label.configure(fg=color)

        if self._status_after_id:
            self.root.after_cancel(self._status_after_id)
            self._status_after_id = None
        if duration_ms > 0:
            self._status_after_id = self.root.after(duration_ms, lambda: self.status_var.set(""))

    def _directory_for_key(self, key: str) -> str:
        return str(normalize_video_key(key).parent)

    def _sync_directory_list_selection(self, directory: str):
        self._updating_dir_list = True
        try:
            self.dir_listbox.selection_clear(0, tk.END)
            for idx, value in enumerate(self.directory_keys):
                if value == directory:
                    self.dir_listbox.selection_set(idx)
                    self.dir_listbox.see(idx)
                    break
        finally:
            self._updating_dir_list = False

    def _jump_to_directory(self, directory: str):
        for idx, key in enumerate(self.video_keys):
            if self._directory_for_key(key) == directory:
                self.current_video_idx = idx
                save_last_selected_dir(self.video_root, directory)
                self._load_current_video()
                self._set_status(f"Jumped to: {directory}", duration_ms=1500)
                return
        raise ValueError(f"Directory not found in current video list: {directory}")

    def _on_directory_selected(self, _event=None):
        if self._updating_dir_list:
            return
        selection = self.dir_listbox.curselection()
        if not selection:
            return
        directory = self.directory_keys[selection[0]]
        current_directory = self._directory_for_key(self._current_video_key())
        if directory == current_directory:
            return
        try:
            self._jump_to_directory(directory)
        except Exception as exc:
            messagebox.showerror("Directory Error", str(exc))

    def _current_video_key(self) -> str:
        return self.video_keys[self.current_video_idx]

    def _current_video_path(self) -> Path:
        return self.video_root / normalize_video_key(self._current_video_key())

    def _current_entry(self):
        key = self._current_video_key()
        entry = self.all_annotations.get(key)
        if not isinstance(entry, dict):
            entry = {}
            self.all_annotations[key] = entry
        return entry

    def _current_frame0(self):
        entry = self._current_entry()
        frame0 = entry.get("0")
        if not isinstance(frame0, dict):
            frame0 = {}
            entry["0"] = frame0
        return frame0

    def _load_boxes_for_current_video(self):
        frame0 = self._current_frame0()
        boxes = []

        for name in ("object", "target"):
            box = frame0.get(name)
            if isinstance(box, list) and len(box) == 4:
                boxes.append(tuple(int(v) for v in box))

        if not boxes:
            raw_boxes = frame0.get("boxes", [])
            if isinstance(raw_boxes, list):
                for box in raw_boxes[:2]:
                    if isinstance(box, list) and len(box) == 4:
                        boxes.append(tuple(int(v) for v in box))

        self._current_boxes = boxes

    def _write_current_boxes_back(self):
        frame0 = self._current_frame0()
        boxes = [list(box) for box in self._current_boxes]
        frame0["boxes"] = boxes
        if boxes:
            frame0["object"] = boxes[0]
        else:
            frame0.pop("object", None)
        if len(boxes) > 1:
            frame0["target"] = boxes[1]
        else:
            frame0.pop("target", None)

    def _save_annotations(self, show_message: bool = False):
        with self.annotations_path.open("w") as f:
            json.dump(self.all_annotations, f, indent=2)
        if show_message:
            self._set_status(f"Saved to: {self.annotations_path}", duration_ms=1800)

    def _open_video_capture(self, video_path: Path):
        self._stop_play()
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _load_current_video(self):
        video_key = self._current_video_key()
        video_path = self._current_video_path()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found for key {video_key}: {video_path}")

        self._open_video_capture(video_path)
        self._load_boxes_for_current_video()
        self._reset_view_state()
        self._reset_interaction()
        self._show_frame(0)

    def _read_frame(self, frame_idx: int) -> Image.Image:
        if self.cap is None:
            raise IOError("No video opened")
        frame_idx = max(0, min(frame_idx, max(self.total_frames - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            raise IOError(f"Cannot read frame {frame_idx} from {self._current_video_path()}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _show_frame(self, frame_idx: int):
        if self.total_frames <= 0:
            raise IOError(f"Video has no frames: {self._current_video_path()}")
        self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self._frame_image = self._read_frame(self.current_frame_idx)
        self._reset_interaction()
        self._render_current_video()

    def _draw_seekbar(self):
        self.seek_canvas.delete("all")
        w = self.seek_canvas.winfo_width()
        h = self.seek_canvas.winfo_height()
        if w <= 1 or self.total_frames <= 1:
            return
        self.seek_canvas.create_rectangle(0, 0, w, h, fill="#444444", outline="")
        ratio = self.current_frame_idx / max(self.total_frames - 1, 1)
        filled_x = int(ratio * w)
        self.seek_canvas.create_rectangle(0, 0, filled_x, h, fill="#2196F3", outline="")
        tx = max(4, min(filled_x, w - 4))
        self.seek_canvas.create_oval(tx - 5, 1, tx + 5, h - 1, fill="white", outline="")

    def _on_seek_click(self, event):
        self._stop_play()
        w = self.seek_canvas.winfo_width()
        if w <= 1 or self.total_frames <= 1:
            return
        ratio = max(0.0, min(1.0, event.x / w))
        frame_idx = int(ratio * (self.total_frames - 1))
        self._show_frame(frame_idx)

    def _reset_view_state(self):
        self._zoom_factor = 1.0
        self._pan_offset = (0.0, 0.0)

    def _clamp_pan(self, scaled_w: float, scaled_h: float, canvas_w: float, canvas_h: float):
        pan_x, pan_y = self._pan_offset
        if scaled_w <= canvas_w:
            pan_x = 0.0
        else:
            max_pan_x = (scaled_w - canvas_w) / 2
            pan_x = min(max(pan_x, -max_pan_x), max_pan_x)

        if scaled_h <= canvas_h:
            pan_y = 0.0
        else:
            max_pan_y = (scaled_h - canvas_h) / 2
            pan_y = min(max(pan_y, -max_pan_y), max_pan_y)

        self._pan_offset = (pan_x, pan_y)

    def _set_zoom(self, new_zoom: float, anchor_canvas: tuple[float, float] | None = None):
        if self._frame_image is None:
            return

        min_zoom, max_zoom = 1.0, 8.0
        new_zoom = min(max(new_zoom, min_zoom), max_zoom)
        if abs(new_zoom - self._zoom_factor) < 1e-6:
            return

        anchor_img = None
        if anchor_canvas is not None:
            anchor_img = self._canvas_to_img(*anchor_canvas)

        self._zoom_factor = new_zoom
        self._render_current_video()

        if anchor_img is not None:
            new_cx, new_cy = self._img_to_canvas(*anchor_img)
            dx = anchor_canvas[0] - new_cx
            dy = anchor_canvas[1] - new_cy
            self._pan_offset = (self._pan_offset[0] + dx, self._pan_offset[1] + dy)
            self._render_current_video()

    def _zoom_in(self):
        canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 1280
        canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 720
        self._set_zoom(self._zoom_factor * 1.25, anchor_canvas=(canvas_w / 2, canvas_h / 2))

    def _zoom_out(self):
        canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 1280
        canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 720
        self._set_zoom(self._zoom_factor / 1.25, anchor_canvas=(canvas_w / 2, canvas_h / 2))

    def _reset_view(self):
        self._reset_view_state()
        self._render_current_video()

    def _on_mousewheel(self, event):
        if self._frame_image is None:
            return
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self._set_zoom(self._zoom_factor * factor, anchor_canvas=(event.x, event.y))

    def _on_mousewheel_linux(self, event, zoom_in: bool):
        factor = 1.1 if zoom_in else 1 / 1.1
        self._set_zoom(self._zoom_factor * factor, anchor_canvas=(event.x, event.y))

    def _on_pan_start(self, event):
        if self._frame_image is None or self._zoom_factor <= 1.0:
            return
        self._panning = True
        self._pan_start = (event.x, event.y)
        self._pan_origin = self._pan_offset
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event):
        if not self._panning or self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_offset = (self._pan_origin[0] + dx, self._pan_origin[1] + dy)
        self._render_current_video()

    def _on_pan_end(self, _event):
        self._panning = False
        self._pan_start = None
        self._set_canvas_cursor(*self._canvas_to_img(
            self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(),
            self.canvas.winfo_pointery() - self.canvas.winfo_rooty(),
        ))

    def _render_current_video(self):
        img = self._frame_image
        if img is None:
            return

        cw = self.canvas.winfo_width() if self.canvas.winfo_width() > 10 else 1280
        ch = self.canvas.winfo_height() if self.canvas.winfo_height() > 10 else 720
        self._base_scale, _ = self._fit(img.size, (cw, ch))
        self._scale = self._base_scale * self._zoom_factor
        scaled_w = img.width * self._scale
        scaled_h = img.height * self._scale
        self._clamp_pan(scaled_w, scaled_h, cw, ch)
        self._offset = (
            (cw - scaled_w) / 2 + self._pan_offset[0],
            (ch - scaled_h) / 2 + self._pan_offset[1],
        )

        disp = img.resize((max(1, int(scaled_w)), max(1, int(scaled_h))), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        ox, oy = self._offset
        self.canvas.create_image(ox, oy, anchor="nw", image=self._tk_img)

        roles = ["object", "target"]
        if self.current_frame_idx == 0:
            for idx, box in enumerate(self._current_boxes):
                role = roles[idx] if idx < len(roles) else None
                self._draw_box(box, role)

        key = self._current_video_key()
        self.root.title(f"Modify First Frame BBox — {key}")
        current_directory = self._directory_for_key(key)
        self.root_var.set(f"Video Root: {self.video_root}")
        self.dir_listbox.delete(0, tk.END)
        for directory in self.directory_keys:
            self.dir_listbox.insert(tk.END, directory)
        self._sync_directory_list_selection(current_directory)
        save_last_selected_dir(self.video_root, current_directory)
        self.path_var.set(f"{self.current_video_idx + 1}/{len(self.video_keys)}  {key}")
        hints = ["Draw object box", "Draw target box", ""][len(self._current_boxes)] if len(self._current_boxes) < 2 else ""
        edit_hint = "Editing enabled" if self.current_frame_idx == 0 else "Editing disabled outside frame 0"
        self.info_var.set(
            f"Frame {self.current_frame_idx + 1}/{self.total_frames}  |  "
            f"Object: {'✓' if len(self._current_boxes) > 0 else '✗'}  "
            f"Target: {'✓' if len(self._current_boxes) > 1 else '✗'}  |  "
            f"Zoom: {self._zoom_factor:.2f}x  |  "
            f"{edit_hint}  |  "
            + (f"Next: {hints}  |  " if hints and self.current_frame_idx == 0 else "")
            + "[Drag] draw/move  [Handle] resize  [Wheel] zoom  [Shift+Drag] pan  [←/→] prev/next frame  "
            + "[↑/↓] prev/next video  [Space] play/pause  [S] save  [RClick] delete"
        )
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        self._draw_seekbar()

    @staticmethod
    def _fit(img_size, canvas_size):
        iw, ih = img_size
        cw, ch = canvas_size
        scale = min(cw / iw, ch / ih, 1.0)
        ox = (cw - iw * scale) / 2
        oy = (ch - ih * scale) / 2
        return scale, (ox, oy)

    def _step_video(self, delta: int):
        next_idx = self.current_video_idx + delta
        if next_idx < 0 or next_idx >= len(self.video_keys):
            return
        self.current_video_idx = next_idx
        self._load_current_video()

    def _step_frame(self, delta: int):
        self._stop_play()
        if self.total_frames <= 0:
            return
        self._show_frame(self.current_frame_idx + delta)

    def _toggle_play(self):
        if self.playing:
            self._stop_play()
            self._render_current_video()
            return
        if self.total_frames <= 1:
            return
        self.playing = True
        self.play_btn.config(text="⏸ Pause")
        self._play_tick()

    def _play_tick(self):
        if not self.playing:
            return
        next_idx = self.current_frame_idx + 1
        if next_idx >= self.total_frames:
            self._stop_play()
            self._render_current_video()
            return
        self._show_frame(next_idx)
        delay = max(1, int(1000 / self.fps))
        self._play_after_id = self.root.after(delay, self._play_tick)

    def _stop_play(self):
        self.playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.config(text="▶ Play")
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _canvas_to_img(self, cx, cy):
        ox, oy = self._offset
        s = self._scale
        return (cx - ox) / s, (cy - oy) / s

    def _img_to_canvas(self, ix, iy):
        ox, oy = self._offset
        s = self._scale
        return ix * s + ox, iy * s + oy

    def _clamp_point_to_image(self, ix, iy):
        iw, ih = self._frame_image.size
        return min(max(ix, 0), iw), min(max(iy, 0), ih)

    def _normalize_box(self, box):
        x0, y0, x1, y1 = box
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = self._clamp_point_to_image(x0, y0)
        x1, y1 = self._clamp_point_to_image(x1, y1)
        return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

    def _find_box_hit(self, ix, iy):
        tol = max(6, 10 / max(self._scale, 1e-6))
        handles = {
            "nw": lambda x0, y0, x1, y1: (x0, y0),
            "n": lambda x0, y0, x1, y1: ((x0 + x1) / 2, y0),
            "ne": lambda x0, y0, x1, y1: (x1, y0),
            "e": lambda x0, y0, x1, y1: (x1, (y0 + y1) / 2),
            "se": lambda x0, y0, x1, y1: (x1, y1),
            "s": lambda x0, y0, x1, y1: ((x0 + x1) / 2, y1),
            "sw": lambda x0, y0, x1, y1: (x0, y1),
            "w": lambda x0, y0, x1, y1: (x0, (y0 + y1) / 2),
        }

        for idx in range(len(self._current_boxes) - 1, -1, -1):
            x0, y0, x1, y1 = self._current_boxes[idx]
            for handle, getter in handles.items():
                hx, hy = getter(x0, y0, x1, y1)
                if abs(ix - hx) <= tol and abs(iy - hy) <= tol:
                    return idx, "resize", handle
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                return idx, "move", None

        return None, None, None

    def _resize_box(self, original_box, handle, ix, iy):
        x0, y0, x1, y1 = original_box
        if "w" in handle:
            x0 = ix
        if "e" in handle:
            x1 = ix
        if "n" in handle:
            y0 = iy
        if "s" in handle:
            y1 = iy
        return self._normalize_box((x0, y0, x1, y1))

    def _move_box(self, original_box, ix, iy):
        if self._move_offset is None:
            return self._normalize_box(original_box)

        iw, ih = self._frame_image.size
        x0, y0, x1, y1 = original_box
        width = x1 - x0
        height = y1 - y0
        new_x0 = ix - self._move_offset[0]
        new_y0 = iy - self._move_offset[1]
        new_x0 = min(max(new_x0, 0), max(iw - width, 0))
        new_y0 = min(max(new_y0, 0), max(ih - height, 0))
        return self._normalize_box((new_x0, new_y0, new_x0 + width, new_y0 + height))

    def _preview_box(self, box):
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
        x0, y0, x1, y1 = box
        cx0, cy0 = self._img_to_canvas(x0, y0)
        cx1, cy1 = self._img_to_canvas(x1, y1)
        self._drag_rect_id = self.canvas.create_rectangle(
            cx0, cy0, cx1, cy1, outline="cyan", width=2, dash=(4, 2), tags="drag"
        )

    def _reset_interaction(self):
        self._drag_start = None
        self._active_box_idx = None
        self._active_handle = None
        self._interaction_mode = None
        self._interaction_box = None
        self._move_offset = None
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _cursor_for_handle(self, handle):
        return {
            "nw": "crosshair",
            "ne": "crosshair",
            "se": "crosshair",
            "sw": "crosshair",
            "n": "sb_v_double_arrow",
            "s": "sb_v_double_arrow",
            "e": "sb_h_double_arrow",
            "w": "sb_h_double_arrow",
        }.get(handle, "crosshair")

    def _set_canvas_cursor(self, ix, iy):
        _, mode, handle = self._find_box_hit(ix, iy)
        if mode == "resize":
            self.canvas.config(cursor=self._cursor_for_handle(handle))
        elif mode == "move":
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="crosshair")

    def _persist_and_refresh(self):
        self._write_current_boxes_back()
        self._save_annotations(show_message=False)
        self._reset_interaction()
        self._render_current_video()

    def _on_mouse_press(self, event):
        if self._panning or (event.state & 0x0001):
            return
        if self.current_frame_idx != 0:
            return
        ix, iy = self._clamp_point_to_image(*self._canvas_to_img(event.x, event.y))
        hit_idx, mode, handle = self._find_box_hit(ix, iy)
        self._reset_interaction()

        if hit_idx is not None:
            self._active_box_idx = hit_idx
            self._interaction_mode = mode
            self._interaction_box = self._current_boxes[hit_idx]
            if mode == "resize":
                self._active_handle = handle
            else:
                x0, y0, _, _ = self._interaction_box
                self._move_offset = (ix - x0, iy - y0)
            self._preview_box(self._interaction_box)
            return

        if len(self._current_boxes) >= 2:
            return

        self._interaction_mode = "draw"
        self._drag_start = (ix, iy)
        self._preview_box((ix, iy, ix, iy))

    def _on_mouse_drag(self, event):
        if self._panning:
            return
        if self._interaction_mode is None:
            return

        ix, iy = self._clamp_point_to_image(*self._canvas_to_img(event.x, event.y))
        if self._interaction_mode == "draw" and self._drag_start is not None:
            x0, y0 = self._drag_start
            self._preview_box(self._normalize_box((x0, y0, ix, iy)))
            return

        if self._interaction_box is None or self._active_box_idx is None:
            return

        if self._interaction_mode == "resize":
            preview_box = self._resize_box(self._interaction_box, self._active_handle, ix, iy)
        else:
            preview_box = self._move_box(self._interaction_box, ix, iy)
        self._preview_box(preview_box)

    def _on_mouse_release(self, event):
        if self._panning:
            return
        if self._interaction_mode is None:
            return

        ix, iy = self._clamp_point_to_image(*self._canvas_to_img(event.x, event.y))
        if self._interaction_mode == "draw" and self._drag_start is not None:
            x0, y0 = self._drag_start
            updated_box = self._normalize_box((x0, y0, ix, iy))
            if (updated_box[2] - updated_box[0]) >= 5 and (updated_box[3] - updated_box[1]) >= 5:
                self._current_boxes.append(updated_box)
                self._persist_and_refresh()
                return
        elif self._interaction_mode == "resize" and self._interaction_box is not None:
            updated_box = self._resize_box(self._interaction_box, self._active_handle, ix, iy)
            if (updated_box[2] - updated_box[0]) >= 5 and (updated_box[3] - updated_box[1]) >= 5:
                self._current_boxes[self._active_box_idx] = updated_box
                self._persist_and_refresh()
                return
        elif self._interaction_mode == "move" and self._interaction_box is not None:
            self._current_boxes[self._active_box_idx] = self._move_box(self._interaction_box, ix, iy)
            self._persist_and_refresh()
            return

        self._reset_interaction()
        self._render_current_video()

    def _on_mouse_move(self, event):
        if self._panning:
            self.canvas.config(cursor="fleur")
            return
        if self.current_frame_idx != 0:
            self.canvas.config(cursor="arrow")
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        self._set_canvas_cursor(ix, iy)

    def _on_right_click(self, event):
        if self.current_frame_idx != 0:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        for idx, (x0, y0, x1, y1) in enumerate(self._current_boxes):
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                self._current_boxes.pop(idx)
                self._persist_and_refresh()
                return

    def _draw_box(self, box, role: str | None = None):
        x0, y0, x1, y1 = box
        cx0, cy0 = self._img_to_canvas(x0, y0)
        cx1, cy1 = self._img_to_canvas(x1, y1)
        color, role_label = ROLE_STYLE.get(role, ("#FFFFFF", ""))
        self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline=color, width=2)

        w, h = x1 - x0, y1 - y0
        prefix = f"[{role_label}] " if role_label else ""
        label = f"{prefix}({x0}, {y0})  {w}x{h}"
        bg_w = len(label) * 6 + 4
        self.canvas.create_rectangle(cx0, cy0 - 18, cx0 + bg_w, cy0, fill="#000000", outline="")
        self.canvas.create_text(
            cx0 + 4,
            cy0 - 9,
            anchor="w",
            text=label,
            fill=color,
            font=("Helvetica", 10, "bold"),
        )

        handle_half = self._handle_size / 2
        handles = [
            (cx0, cy0),
            ((cx0 + cx1) / 2, cy0),
            (cx1, cy0),
            (cx1, (cy0 + cy1) / 2),
            (cx1, cy1),
            ((cx0 + cx1) / 2, cy1),
            (cx0, cy1),
            (cx0, (cy0 + cy1) / 2),
        ]
        for hx, hy in handles:
            self.canvas.create_rectangle(
                hx - handle_half,
                hy - handle_half,
                hx + handle_half,
                hy + handle_half,
                fill=color,
                outline="#000000",
                width=1,
            )

    def _quit(self):
        self._stop_play()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Modify frame-0 bounding boxes directly in a merged annotations JSON."
    )
    parser.add_argument(
        "--annotations",
        default="annotations_merged.json",
        help="Merged annotations JSON path.",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Root directory containing video files. Defaults to the last loaded directory.",
    )
    parser.add_argument(
        "--start-key",
        default=None,
        help="Optional video key to open first.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    try:
        remembered_root = load_last_video_root()
        video_root = Path(args.video_root) if args.video_root else remembered_root or DEFAULT_VIDEO_ROOT
        remembered_dir = load_last_selected_dir()
        editor = FirstFrameBBoxEditor(
            root=root,
            annotations_path=Path(args.annotations),
            video_root=video_root,
            start_key=args.start_key,
        )
        save_last_video_root(video_root.resolve())
        if remembered_dir and not args.start_key:
            try:
                editor._jump_to_directory(remembered_dir)
            except Exception:
                pass
    except Exception as exc:
        root.destroy()
        raise SystemExit(str(exc))

    root.mainloop()


if __name__ == "__main__":
    main()
