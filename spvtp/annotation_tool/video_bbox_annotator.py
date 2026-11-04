"""
Video Bounding Box Annotator
Usage:
    python video_bbox_annotator.py --video <path_to_video>

Controls:
    Left/Right arrow : prev/next frame
    Space            : play / pause
    S                : save annotations
    Q / Esc          : quit

Mouse:
    Drag to draw a bounding box on any frame.
    Drag box handles to resize existing boxes.
    Drag inside a box to move it.
    Right-click a box to delete it.
"""

import argparse
import json
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk


class VideoAnnotator:
    def __init__(self, root: tk.Tk, video_path: str):
        self.root = root
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_idx = 0
        self.frames_cache: dict[int, Image.Image] = {}  # small LRU-style cache
        self._cache_lock = threading.Lock()

        # annotations: {frame_idx: [(x1,y1,x2,y2), ...]}
        self.annotations: dict[int, list] = {}
        self._load_existing_annotations()

        # playback state
        self.playing = False
        self._play_after_id = None

        # mouse drag state
        self._drag_start = None
        self._drag_rect_id = None
        self._active_box_idx = None
        self._active_handle = None
        self._interaction_mode = None
        self._interaction_anchor = None
        self._interaction_box = None
        self._move_offset = None
        self._handle_size = 6

        self._build_ui()
        self.root.update_idletasks()
        self._show_frame(0)

    # ------------------------------------------------------------------ #
    #  UI                                                                  #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self.root.title(f"Annotator — {self.video_path}")
        self.root.bind("<Left>",  lambda e: self._step(-1))
        self.root.bind("<Right>", lambda e: self._step(1))
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("s",       lambda e: self._save())
        self.root.bind("S",       lambda e: self._save())
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.bind("q",        lambda e: self._quit())

        # ---- top label ----
        self.info_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.info_var, anchor="w").pack(
            fill="x", padx=4, pady=2)

        # ---- canvas ----
        self.canvas = tk.Canvas(self.root, cursor="crosshair",
                                bg="black", width=960, height=540)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Motion>",          self._on_mouse_move)
        self.canvas.bind("<Button-2>",        self._on_right_click)
        self.canvas.bind("<Button-3>",        self._on_right_click)

        # ---- seek bar (custom canvas for click-to-seek) ----
        self.seek_canvas = tk.Canvas(self.root, height=18, bg="#333", cursor="hand2")
        self.seek_canvas.pack(fill="x", padx=4, pady=2)
        self.seek_canvas.bind("<ButtonPress-1>",   self._on_seek_click)
        self.seek_canvas.bind("<B1-Motion>",       self._on_seek_click)
        self.seek_canvas.bind("<Configure>",       lambda e: self._draw_seekbar())

        # ---- buttons ----
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="📂 Open Video", width=14,
                  command=self._open_video).pack(side="left", padx=4)
        tk.Button(btn_frame, text="✏️ Annotate", width=12,
                  command=self._goto_annotate).pack(side="left", padx=4)
        tk.Frame(btn_frame, width=20).pack(side="left")  # spacer
        tk.Button(btn_frame, text="◀ Prev",  width=10,
                  command=lambda: self._step(-1)).pack(side="left", padx=4)
        self.play_btn = tk.Button(btn_frame, text="▶ Play", width=10,
                                  command=self._toggle_play)
        self.play_btn.pack(side="left", padx=4)
        tk.Button(btn_frame, text="Next ▶",  width=10,
                  command=lambda: self._step(1)).pack(side="left", padx=4)
        tk.Button(btn_frame, text="💾 Save (S)", width=14,
                  command=self._save).pack(side="left", padx=12)

    # ------------------------------------------------------------------ #
    #  Frame helpers                                                       #
    # ------------------------------------------------------------------ #
    def _read_frame(self, idx: int) -> Image.Image:
        with self._cache_lock:
            if idx in self.frames_cache:
                return self.frames_cache[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        if not ok:
            bgr = cv2.imread.__func__ if False else \
                  (lambda: None)()  # dummy fallback
            h, w = 540, 960
            bgr = __import__("numpy").zeros((h, w, 3), dtype="uint8")
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self._cache_lock:
            if len(self.frames_cache) > 60:
                oldest = next(iter(self.frames_cache))
                del self.frames_cache[oldest]
            self.frames_cache[idx] = img
        return img

    def _prefetch_frames(self, center: int, radius: int = 5):
        def _worker():
            cap = cv2.VideoCapture(self.video_path)
            try:
                for offset in range(1, radius + 1):
                    for idx in (center + offset, center - offset):
                        if 0 <= idx < self.total_frames:
                            with self._cache_lock:
                                if idx in self.frames_cache:
                                    continue
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ok, bgr = cap.read()
                            if not ok:
                                continue
                            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                            with self._cache_lock:
                                if len(self.frames_cache) > 60:
                                    oldest = next(iter(self.frames_cache))
                                    del self.frames_cache[oldest]
                                self.frames_cache[idx] = img
            finally:
                cap.release()
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _show_frame(self, idx: int):
        idx = max(0, min(idx, self.total_frames - 1))
        self.frame_idx = idx

        img = self._read_frame(idx)
        cw = self.canvas.winfo_width()  if self.canvas.winfo_width()  > 10 else 960
        ch = self.canvas.winfo_height() if self.canvas.winfo_height() > 10 else 540
        self._scale, self._offset = self._fit(img.size, (cw, ch))

        disp = img.resize(
            (int(img.width * self._scale), int(img.height * self._scale)),
            Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        ox, oy = self._offset
        self.canvas.create_image(ox, oy, anchor="nw", image=self._tk_img)

        self._prefetch_frames(idx)

        # draw existing boxes for this frame
        roles = ["object", "target"]
        for i, box in enumerate(self.annotations.get(idx, [])):
            role = roles[i] if i < len(roles) else None
            self._draw_box(box, role)

        self._draw_seekbar()

        boxes = self.annotations.get(idx, [])
        hints = ["Draw object box", "Draw target box", ""][len(boxes)] if len(boxes) < 2 else ""
        self.info_var.set(
            f"Frame {idx+1}/{self.total_frames}  |  "
            f"Object: {'✓' if len(boxes) > 0 else '✗'}  "
            f"Target: {'✓' if len(boxes) > 1 else '✗'}  |  "
            + (f"Next: {hints}  |  " if hints else "")
            + "[Drag] draw/move  [Handle] resize  [←/→] browse  "
            + "[Space] play/pause  [S] save  [RClick] delete")

    @staticmethod
    def _fit(img_size, canvas_size):
        iw, ih = img_size
        cw, ch = canvas_size
        scale = min(cw / iw, ch / ih, 1.0)
        ox = (cw - iw * scale) / 2
        oy = (ch - ih * scale) / 2
        return scale, (ox, oy)

    # ------------------------------------------------------------------ #
    #  Navigation & playback                                               #
    # ------------------------------------------------------------------ #
    def _step(self, delta: int):
        self._stop_play()
        self._show_frame(self.frame_idx + delta)

    def _draw_seekbar(self):
        self.seek_canvas.delete("all")
        w = self.seek_canvas.winfo_width()
        h = self.seek_canvas.winfo_height()
        if w <= 1 or self.total_frames <= 1:
            return
        # track
        self.seek_canvas.create_rectangle(0, 0, w, h, fill="#444", outline="")
        # filled portion
        ratio = self.frame_idx / max(self.total_frames - 1, 1)
        filled_x = int(ratio * w)
        self.seek_canvas.create_rectangle(0, 0, filled_x, h, fill="#2196F3", outline="")
        # thumb
        tx = max(4, min(filled_x, w - 4))
        self.seek_canvas.create_oval(tx - 5, 1, tx + 5, h - 1, fill="white", outline="")

    def _on_seek_click(self, event):
        self._stop_play()
        w = self.seek_canvas.winfo_width()
        if w <= 1:
            return
        ratio = max(0.0, min(1.0, event.x / w))
        idx = int(ratio * (self.total_frames - 1))
        self._show_frame(idx)

    def _toggle_play(self):
        if self.playing:
            self._stop_play()
        else:
            self.playing = True
            self.play_btn.config(text="⏸ Pause")
            self._play_tick()

    def _play_tick(self):
        if not self.playing:
            return
        next_idx = self.frame_idx + 1
        if next_idx >= self.total_frames:
            self._stop_play()
            return
        self._show_frame(next_idx)
        delay = max(1, int(1000 / self.fps))
        self._play_after_id = self.root.after(delay, self._play_tick)

    def _stop_play(self):
        self.playing = False
        self.play_btn.config(text="▶ Play")
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    # ------------------------------------------------------------------ #
    #  Mouse events                                                        #
    # ------------------------------------------------------------------ #
    def _canvas_to_img(self, cx, cy):
        ox, oy = self._offset
        s = self._scale
        return (cx - ox) / s, (cy - oy) / s

    def _img_to_canvas(self, ix, iy):
        ox, oy = self._offset
        s = self._scale
        return ix * s + ox, iy * s + oy

    def _clamp_point_to_image(self, ix, iy):
        img = self._read_frame(self.frame_idx)
        iw, ih = img.size
        return min(max(ix, 0), iw), min(max(iy, 0), ih)

    def _normalize_box(self, box):
        x0, y0, x1, y1 = box
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = self._clamp_point_to_image(x0, y0)
        x1, y1 = self._clamp_point_to_image(x1, y1)
        return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

    def _find_box_hit(self, ix, iy):
        boxes = self.annotations.get(self.frame_idx, [])
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

        for idx in range(len(boxes) - 1, -1, -1):
            x0, y0, x1, y1 = boxes[idx]
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

        img = self._read_frame(self.frame_idx)
        iw, ih = img.size
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
            cx0, cy0, cx1, cy1, outline="cyan", width=2, dash=(4, 2), tags="drag")

    def _reset_interaction(self):
        self._drag_start = None
        self._active_box_idx = None
        self._active_handle = None
        self._interaction_mode = None
        self._interaction_anchor = None
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
        if self.frame_idx != 0:
            self.canvas.config(cursor="arrow")
            return
        _, mode, handle = self._find_box_hit(ix, iy)
        if mode == "resize":
            self.canvas.config(cursor=self._cursor_for_handle(handle))
        elif mode == "move":
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="crosshair")

    def _on_mouse_press(self, event):
        if self.frame_idx != 0:
            return
        ix, iy = self._clamp_point_to_image(*self._canvas_to_img(event.x, event.y))
        hit_idx, mode, handle = self._find_box_hit(ix, iy)
        self._reset_interaction()

        if hit_idx is not None:
            self._active_box_idx = hit_idx
            self._interaction_mode = mode
            self._interaction_box = self.annotations[self.frame_idx][hit_idx]
            if mode == "resize":
                self._active_handle = handle
            elif mode == "move":
                x0, y0, _, _ = self._interaction_box
                self._move_offset = (ix - x0, iy - y0)
            self._preview_box(self._interaction_box)
            return

        self._interaction_mode = "draw"
        self._drag_start = (ix, iy)
        self._interaction_anchor = (ix, iy)
        self._preview_box((ix, iy, ix, iy))

    def _on_mouse_drag(self, event):
        if self._interaction_mode is None:
            return
        ix, iy = self._clamp_point_to_image(*self._canvas_to_img(event.x, event.y))

        if self._interaction_mode == "draw" and self._drag_start is not None:
            x0, y0 = self._drag_start
            preview_box = self._normalize_box((x0, y0, ix, iy))
            self._preview_box(preview_box)
            return

        if self._interaction_box is None or self._active_box_idx is None:
            return

        if self._interaction_mode == "resize":
            preview_box = self._resize_box(self._interaction_box, self._active_handle, ix, iy)
        else:
            preview_box = self._move_box(self._interaction_box, ix, iy)
        self._preview_box(preview_box)

    def _on_mouse_release(self, event):
        if self._interaction_mode is None:
            return
        ix, iy = self._clamp_point_to_image(*self._canvas_to_img(event.x, event.y))
        existing = self.annotations.setdefault(self.frame_idx, [])
        updated_box = None

        if self._interaction_mode == "draw" and self._drag_start is not None:
            x0, y0 = self._drag_start
            updated_box = self._normalize_box((x0, y0, ix, iy))
            if self._active_box_idx is None and len(existing) >= 2:
                self._reset_interaction()
                self._show_frame(self.frame_idx)
                return
            if (updated_box[2] - updated_box[0]) >= 5 and (updated_box[3] - updated_box[1]) >= 5:
                existing.append(updated_box)
        elif self._interaction_mode == "resize" and self._interaction_box is not None:
            updated_box = self._resize_box(self._interaction_box, self._active_handle, ix, iy)
            if (updated_box[2] - updated_box[0]) >= 5 and (updated_box[3] - updated_box[1]) >= 5:
                existing[self._active_box_idx] = updated_box
        elif self._interaction_mode == "move" and self._interaction_box is not None:
            updated_box = self._move_box(self._interaction_box, ix, iy)
            existing[self._active_box_idx] = updated_box

        self._reset_interaction()
        self._show_frame(self.frame_idx)

    def _on_mouse_move(self, event):
        ix, iy = self._canvas_to_img(event.x, event.y)
        self._set_canvas_cursor(ix, iy)

    def _on_right_click(self, event):
        if self.frame_idx != 0:
            return
        cx, cy = event.x, event.y
        ix, iy = self._canvas_to_img(cx, cy)
        boxes = self.annotations.get(self.frame_idx, [])
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                boxes.pop(i)
                self._show_frame(self.frame_idx)
                return

    # role: "object" -> green, "target" -> orange, None -> white
    _ROLE_STYLE = {
        "object": ("#00E676", "Object"),
        "target": ("#FF9800", "Target"),
    }

    def _draw_box(self, box, role: str = None):
        x0, y0, x1, y1 = box
        cx0, cy0 = self._img_to_canvas(x0, y0)
        cx1, cy1 = self._img_to_canvas(x1, y1)
        color, role_label = self._ROLE_STYLE.get(role, ("#FFFFFF", ""))
        # box outline
        self.canvas.create_rectangle(
            cx0, cy0, cx1, cy1, outline=color, width=2)
        # label text
        w, h = x1 - x0, y1 - y0
        prefix = f"[{role_label}] " if role_label else ""
        label = f"{prefix}({x0}, {y0})  {w}×{h}"
        bg_w = len(label) * 5 + 2
        self.canvas.create_rectangle(
            cx0, cy0 - 18, cx0 + bg_w, cy0,
            fill="#000000", outline="")
        self.canvas.create_text(
            cx0 + 4, cy0 - 9, anchor="w",
            text=label,
            fill=color, font=("Helvetica", 10, "bold"))

        handle_half = self._handle_size / 2
        handles = [
            (cx0, cy0), ((cx0 + cx1) / 2, cy0), (cx1, cy0),
            (cx1, (cy0 + cy1) / 2), (cx1, cy1), ((cx0 + cx1) / 2, cy1),
            (cx0, cy1), (cx0, (cy0 + cy1) / 2),
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

    # ------------------------------------------------------------------ #
    #  Save / load                                                         #
    # ------------------------------------------------------------------ #
    def _annotation_path(self):
        stem = Path(self.video_path).stem
        return Path(self.video_path).parent / f"{stem}_annotations.json"

    def _save(self):
        out = {}
        for k, boxes in self.annotations.items():
            entry = {"boxes": [list(b) for b in boxes]}
            if len(boxes) > 0:
                entry["object"] = list(boxes[0])
            if len(boxes) > 1:
                entry["target"] = list(boxes[1])
            out[str(k)] = entry
        path = self._annotation_path()
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        messagebox.showinfo("Saved", f"Annotations saved to:\n{path}")

    def _load_existing_annotations(self):
        path = self._annotation_path()
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            self.annotations = {}
            for k, v in raw.items():
                # support both old list format and new dict format
                if isinstance(v, list):
                    boxes = [tuple(b) for b in v]
                else:
                    boxes = [tuple(b) for b in v.get("boxes", [])]
                self.annotations[int(k)] = boxes

    # ------------------------------------------------------------------ #
    #  Open video / output dir                                            #
    # ------------------------------------------------------------------ #
    def _goto_annotate(self):
        self._stop_play()
        self._show_frame(0)

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        if not path:
            return
        self._stop_play()
        self.cap.release()
        self.frames_cache.clear()
        self.annotations.clear()
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{path}")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.root.title(f"Annotator — {path}")
        self._load_existing_annotations()
        self.root.update_idletasks()
        self._show_frame(0)

    def _quit(self):
        self._stop_play()
        self.cap.release()
        self.root.destroy()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Video bounding box annotator")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file")
    args = parser.parse_args()

    root = tk.Tk()

    video_path = args.video
    if not video_path:
        video_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not video_path:
            root.destroy()
            return

    app = VideoAnnotator(root, video_path)
    root.mainloop()


if __name__ == "__main__":
    main()
