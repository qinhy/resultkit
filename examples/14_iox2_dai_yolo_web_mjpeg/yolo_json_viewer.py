#!/usr/bin/env python3
"""
YOLO JSON Viewer

A desktop GUI for viewing YOLO detection/segmentation JSON files.

Features
--------
- Open a YOLO JSON file and an optional source image
- Render segmentation polygons, bounding boxes, labels, and confidence
- Filter by confidence and class
- Zoom, pan, fit-to-window, and 100% view
- Select detections from a table
- Export the annotated image as PNG

Expected JSON structure
-----------------------
{
  "image_width": 4032,
  "image_height": 3040,
  "input_jpg_path": "...",
  "detections": [
    {
      "class_id": 39,
      "class_name": "bottle",
      "confidence": 0.95,
      "bbox_xyxy": [x1, y1, x2, y2],
      "mask": {
        "format": "polygon",
        "polygons": [
          {
            "points_xy": [[x, y], ...],
            "is_hole": false,
            "parent_index": 0
          }
        ]
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Iterable

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Pillow is required.\nInstall it with:\n\n    pip install pillow\n"
    ) from exc


APP_TITLE = "YOLO JSON Viewer"
DEFAULT_BG = (235, 235, 235)
SELECTED_COLOR = (255, 255, 255, 255)


@dataclass
class Detection:
    index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    polygons: list[dict[str, Any]]
    mask_area: float | None

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def deterministic_color(name: str) -> tuple[int, int, int]:
    """Generate a stable, visually distinct RGB color from a class name."""
    value = 0
    for i, ch in enumerate(name):
        value = (value * 131 + ord(ch) + i * 17) & 0xFFFFFFFF
    hue = (value % 360) / 360.0
    saturation = 0.72
    lightness = 0.50
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return int(r * 255), int(g * 255), int(b * 255)


def safe_font(size: int) -> ImageFont.ImageFont:
    """Load a common TrueType font, falling back to Pillow's default font."""
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


class YoloViewer(tk.Tk):
    def __init__(self, json_path: str | None = None, image_path: str | None = None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1450x900")
        self.minsize(1000, 650)

        self.json_path: Path | None = None
        self.image_path: Path | None = None
        self.raw_data: dict[str, Any] = {}
        self.detections: list[Detection] = []
        self.class_names: list[str] = []
        self.class_enabled: dict[str, tk.BooleanVar] = {}
        self.original_image: Image.Image | None = None
        self.rendered_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.image_width = 1280
        self.image_height = 720

        self.zoom = 1.0
        self.fit_mode = True
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pan_start: tuple[int, int] | None = None
        self.pan_origin: tuple[float, float] | None = None
        self.selected_detection: int | None = None

        self.conf_threshold = tk.DoubleVar(value=0.0)
        self.mask_opacity = tk.IntVar(value=35)
        self.show_masks = tk.BooleanVar(value=True)
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=True)
        self.show_only_selected = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Open a YOLO JSON file to begin.")
        self.threshold_text = tk.StringVar(value="0.00")
        self.opacity_text = tk.StringVar(value="35%")

        self._build_menu()
        self._build_layout()
        self._bind_events()

        if json_path:
            self.load_json(Path(json_path))
        if image_path:
            self.load_image(Path(image_path))
        elif self.raw_data:
            self._try_auto_load_image()

        self.after(100, self.fit_to_window)

    # --------------------------- UI construction ---------------------------

    def _build_menu(self) -> None:
        menu = tk.Menu(self)

        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open JSON…", accelerator="Ctrl+O", command=self.open_json_dialog)
        file_menu.add_command(label="Open image…", accelerator="Ctrl+I", command=self.open_image_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export annotated PNG…", accelerator="Ctrl+E", command=self.export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_command(label="Fit to window", accelerator="F", command=self.fit_to_window)
        view_menu.add_command(label="100%", accelerator="1", command=self.zoom_100)
        view_menu.add_command(label="Zoom in", accelerator="+", command=lambda: self.change_zoom(1.2))
        view_menu.add_command(label="Zoom out", accelerator="-", command=lambda: self.change_zoom(1 / 1.2))
        menu.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="Controls", command=self.show_controls)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menu)

    def _build_layout(self) -> None:
        root_pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        root_pane.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(root_pane, padding=8)
        main = ttk.Frame(root_pane)
        root_pane.add(sidebar, weight=0)
        root_pane.add(main, weight=1)

        sidebar.configure(width=330)

        files_frame = ttk.LabelFrame(sidebar, text="Files", padding=8)
        files_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(files_frame, text="Open JSON", command=self.open_json_dialog).pack(fill=tk.X)
        ttk.Button(files_frame, text="Open source image", command=self.open_image_dialog).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(files_frame, text="Export annotated PNG", command=self.export_dialog).pack(fill=tk.X, pady=(6, 0))

        self.json_label = ttk.Label(files_frame, text="JSON: not loaded", wraplength=290)
        self.json_label.pack(fill=tk.X, pady=(8, 0))
        self.image_label = ttk.Label(files_frame, text="Image: not loaded", wraplength=290)
        self.image_label.pack(fill=tk.X, pady=(4, 0))

        display_frame = ttk.LabelFrame(sidebar, text="Display", padding=8)
        display_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Checkbutton(display_frame, text="Segmentation masks", variable=self.show_masks, command=self.request_render).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Bounding boxes", variable=self.show_boxes, command=self.request_render).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Labels", variable=self.show_labels, command=self.request_render).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Confidence values", variable=self.show_confidence, command=self.request_render).pack(anchor=tk.W)
        ttk.Checkbutton(
            display_frame,
            text="Show only selected detection",
            variable=self.show_only_selected,
            command=self.request_render,
        ).pack(anchor=tk.W, pady=(3, 0))

        threshold_row = ttk.Frame(display_frame)
        threshold_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(threshold_row, text="Confidence threshold").pack(side=tk.LEFT)
        ttk.Label(threshold_row, textvariable=self.threshold_text, width=5).pack(side=tk.RIGHT)

        threshold_scale = ttk.Scale(
            display_frame,
            from_=0.0,
            to=1.0,
            variable=self.conf_threshold,
            command=self._on_threshold_changed,
        )
        threshold_scale.pack(fill=tk.X)

        opacity_row = ttk.Frame(display_frame)
        opacity_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(opacity_row, text="Mask opacity").pack(side=tk.LEFT)
        ttk.Label(opacity_row, textvariable=self.opacity_text, width=5).pack(side=tk.RIGHT)

        opacity_scale = ttk.Scale(
            display_frame,
            from_=0,
            to=100,
            variable=self.mask_opacity,
            command=self._on_opacity_changed,
        )
        opacity_scale.pack(fill=tk.X)

        class_frame = ttk.LabelFrame(sidebar, text="Classes", padding=8)
        class_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        class_buttons = ttk.Frame(class_frame)
        class_buttons.pack(fill=tk.X)
        ttk.Button(class_buttons, text="All", width=8, command=lambda: self.set_all_classes(True)).pack(side=tk.LEFT)
        ttk.Button(class_buttons, text="None", width=8, command=lambda: self.set_all_classes(False)).pack(side=tk.LEFT, padx=(5, 0))

        self.class_canvas = tk.Canvas(class_frame, width=280, height=190, highlightthickness=0)
        self.class_scrollbar = ttk.Scrollbar(class_frame, orient=tk.VERTICAL, command=self.class_canvas.yview)
        self.class_inner = ttk.Frame(self.class_canvas)

        self.class_inner.bind(
            "<Configure>",
            lambda event: self.class_canvas.configure(scrollregion=self.class_canvas.bbox("all")),
        )
        self.class_canvas.create_window((0, 0), window=self.class_inner, anchor="nw")
        self.class_canvas.configure(yscrollcommand=self.class_scrollbar.set)

        self.class_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(7, 0))
        self.class_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(7, 0))

        zoom_frame = ttk.LabelFrame(sidebar, text="View", padding=8)
        zoom_frame.pack(fill=tk.X)
        zoom_buttons = ttk.Frame(zoom_frame)
        zoom_buttons.pack(fill=tk.X)
        ttk.Button(zoom_buttons, text="Fit", command=self.fit_to_window).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(zoom_buttons, text="100%", command=self.zoom_100).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(zoom_buttons, text="−", width=4, command=lambda: self.change_zoom(1 / 1.2)).pack(side=tk.LEFT)
        ttk.Button(zoom_buttons, text="+", width=4, command=lambda: self.change_zoom(1.2)).pack(side=tk.LEFT, padx=(5, 0))

        vertical_pane = ttk.Panedwindow(main, orient=tk.VERTICAL)
        vertical_pane.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(vertical_pane)
        table_frame = ttk.Frame(vertical_pane, padding=(4, 4, 4, 0))
        vertical_pane.add(canvas_frame, weight=4)
        vertical_pane.add(table_frame, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#202124",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        columns = ("index", "class", "confidence", "bbox", "mask_area")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("index", text="#")
        self.tree.heading("class", text="Class")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("bbox", text="BBox [x1, y1, x2, y2]")
        self.tree.heading("mask_area", text="Mask area")

        self.tree.column("index", width=45, anchor=tk.CENTER, stretch=False)
        self.tree.column("class", width=140)
        self.tree.column("confidence", width=95, anchor=tk.CENTER)
        self.tree.column("bbox", width=300)
        self.tree.column("mask_area", width=110, anchor=tk.E)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(6, 3))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_events(self) -> None:
        self.bind("<Control-o>", lambda event: self.open_json_dialog())
        self.bind("<Control-i>", lambda event: self.open_image_dialog())
        self.bind("<Control-e>", lambda event: self.export_dialog())
        self.bind("<Key-f>", lambda event: self.fit_to_window())
        self.bind("<Key-1>", lambda event: self.zoom_100())
        self.bind("<Key-plus>", lambda event: self.change_zoom(1.2))
        self.bind("<Key-equal>", lambda event: self.change_zoom(1.2))
        self.bind("<Key-minus>", lambda event: self.change_zoom(1 / 1.2))

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1.2))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1 / 1.2))
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-3>", self._end_pan)
        self.canvas.bind("<Button-1>", self._select_from_canvas)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # ----------------------------- Data loading ----------------------------

    def open_json_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            title="Open YOLO JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.load_json(Path(filename))
            self._try_auto_load_image()

    def open_image_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            title="Open source image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.load_image(Path(filename))

    def load_json(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Could not open JSON", str(exc))
            return

        try:
            width = int(data["image_width"])
            height = int(data["image_height"])
            raw_detections = data.get("detections", [])
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror(
                "Unsupported JSON",
                "The file must contain image_width, image_height, and a detections list.\n\n"
                f"Details: {exc}",
            )
            return

        detections: list[Detection] = []
        for index, item in enumerate(raw_detections, start=1):
            try:
                bbox_values = tuple(float(v) for v in item["bbox_xyxy"])
                if len(bbox_values) != 4:
                    raise ValueError("bbox_xyxy must have four values")
                mask = item.get("mask") or {}
                detections.append(
                    Detection(
                        index=index,
                        class_id=int(item.get("class_id", -1)),
                        class_name=str(item.get("class_name", f"class_{item.get('class_id', -1)}")),
                        confidence=float(item.get("confidence", 0.0)),
                        bbox=bbox_values,  # type: ignore[arg-type]
                        polygons=list(mask.get("polygons") or []),
                        mask_area=float(mask["area"]) if mask.get("area") is not None else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Skipping invalid detection #{index}: {exc}", file=sys.stderr)

        self.raw_data = data
        self.json_path = path
        self.image_width = width
        self.image_height = height
        self.detections = detections
        self.class_names = sorted({det.class_name for det in detections})
        self.selected_detection = None

        self.json_label.configure(text=f"JSON: {path.name}")
        self.status_var.set(
            f"Loaded {len(detections)} detections, {len(self.class_names)} classes, "
            f"image size {width}×{height}."
        )

        self._rebuild_class_filters()
        self._populate_table()
        self.request_render()
        self.after(50, self.fit_to_window)

    def load_image(self, path: Path) -> None:
        try:
            image = Image.open(path)
            image.load()
            image = image.convert("RGB")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not open image", str(exc))
            return

        self.image_path = path
        self.original_image = image
        self.image_label.configure(text=f"Image: {path.name}")

        if self.raw_data and image.size != (self.image_width, self.image_height):
            proceed = messagebox.askyesno(
                "Image size differs",
                f"The JSON expects {self.image_width}×{self.image_height}, "
                f"but the image is {image.width}×{image.height}.\n\n"
                "Display it anyway?",
            )
            if not proceed:
                self.original_image = None
                self.image_path = None
                self.image_label.configure(text="Image: not loaded")
                return

        if not self.raw_data:
            self.image_width, self.image_height = image.size

        self.status_var.set(f"Loaded image: {path.name} ({image.width}×{image.height}).")
        self.request_render()
        self.after(50, self.fit_to_window)

    def _try_auto_load_image(self) -> None:
        if not self.raw_data or self.original_image is not None:
            return

        candidate_text = self.raw_data.get("input_jpg_path")
        if not candidate_text:
            return

        candidates: list[Path] = []
        raw_candidate = Path(str(candidate_text))
        candidates.append(raw_candidate)

        if self.json_path:
            candidates.append(self.json_path.parent / raw_candidate.name)
            normalized_parts = str(candidate_text).replace("\\", "/").split("/")
            candidates.append(self.json_path.parent / normalized_parts[-1])

        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    self.load_image(candidate)
                    return
            except OSError:
                continue

    # ------------------------------- Filters -------------------------------

    def _rebuild_class_filters(self) -> None:
        for child in self.class_inner.winfo_children():
            child.destroy()
        self.class_enabled.clear()

        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        for class_name in self.class_names:
            var = tk.BooleanVar(value=True)
            self.class_enabled[class_name] = var
            color = deterministic_color(class_name)
            swatch = tk.Canvas(self.class_inner, width=14, height=14, highlightthickness=0)
            swatch.create_rectangle(1, 1, 13, 13, fill=self._rgb_to_hex(color), outline="")
            swatch.grid(row=len(self.class_enabled) - 1, column=0, padx=(0, 4), pady=2)
            checkbox = ttk.Checkbutton(
                self.class_inner,
                text=f"{class_name} ({counts[class_name]})",
                variable=var,
                command=self._on_filter_changed,
            )
            checkbox.grid(row=len(self.class_enabled) - 1, column=1, sticky="w", pady=1)

    def set_all_classes(self, enabled: bool) -> None:
        for var in self.class_enabled.values():
            var.set(enabled)
        self._on_filter_changed()

    def _on_threshold_changed(self, value: str) -> None:
        try:
            threshold = float(value)
        except ValueError:
            threshold = self.conf_threshold.get()
        self.threshold_text.set(f"{threshold:.2f}")
        self._on_filter_changed()

    def _on_opacity_changed(self, value: str) -> None:
        try:
            opacity = int(float(value))
        except ValueError:
            opacity = self.mask_opacity.get()
        self.opacity_text.set(f"{opacity}%")
        self.request_render()

    def _on_filter_changed(self) -> None:
        self._populate_table()
        self.request_render()

    def visible_detections(self) -> list[Detection]:
        threshold = self.conf_threshold.get()
        result = [
            det
            for det in self.detections
            if det.confidence >= threshold
            and self.class_enabled.get(det.class_name, tk.BooleanVar(value=True)).get()
        ]
        if self.show_only_selected.get() and self.selected_detection is not None:
            result = [det for det in result if det.index == self.selected_detection]
        return result

    # ------------------------------- Rendering -----------------------------

    def request_render(self) -> None:
        self.after_idle(self.render)

    def render(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            return

        if self.original_image is not None:
            base = self.original_image.copy().convert("RGBA")
            if base.size != (self.image_width, self.image_height):
                base = base.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        else:
            base = Image.new("RGBA", (self.image_width, self.image_height), DEFAULT_BG + (255,))
            self._draw_background_grid(base)

        visible = self.visible_detections()
        self._draw_detections(base, visible)
        self.rendered_image = base.convert("RGB")
        self._display_rendered_image()

        self.status_var.set(
            f"Showing {len(visible)} of {len(self.detections)} detections | "
            f"zoom {self.zoom * 100:.1f}%"
        )

    def _draw_background_grid(self, image: Image.Image) -> None:
        draw = ImageDraw.Draw(image)
        spacing = max(80, int(min(self.image_width, self.image_height) / 15))
        line = (215, 215, 215, 255)
        for x in range(0, self.image_width, spacing):
            draw.line((x, 0, x, self.image_height), fill=line, width=1)
        for y in range(0, self.image_height, spacing):
            draw.line((0, y, self.image_width, y), fill=line, width=1)

    def _draw_detections(self, base: Image.Image, detections: Iterable[Detection]) -> None:
        line_width = max(2, int(round(min(self.image_width, self.image_height) / 850)))
        font_size = max(13, int(round(min(self.image_width, self.image_height) / 105)))
        font = safe_font(font_size)
        opacity = int(255 * self.mask_opacity.get() / 100)

        for det in detections:
            color_rgb = deterministic_color(det.class_name)
            selected = det.index == self.selected_detection

            if self.show_masks.get() and det.polygons:
                self._draw_mask(base, det, color_rgb, opacity)

            draw = ImageDraw.Draw(base, "RGBA")
            x1, y1, x2, y2 = det.bbox

            box_color = SELECTED_COLOR if selected else color_rgb + (255,)
            box_width = line_width * (3 if selected else 1)

            if self.show_boxes.get():
                for inset in range(box_width):
                    draw.rectangle(
                        (x1 - inset, y1 - inset, x2 + inset, y2 + inset),
                        outline=box_color,
                        width=1,
                    )

            if self.show_labels.get():
                label = det.class_name
                if self.show_confidence.get():
                    label += f" {det.confidence:.3f}"
                if selected:
                    label = f"★ {label}"

                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                pad = max(3, font_size // 5)

                label_x = max(0, min(int(x1), self.image_width - text_w - 2 * pad))
                label_y = int(y1) - text_h - 2 * pad
                if label_y < 0:
                    label_y = min(self.image_height - text_h - 2 * pad, int(y1) + 2)

                bg_color = color_rgb + (225,)
                draw.rounded_rectangle(
                    (
                        label_x,
                        label_y,
                        label_x + text_w + 2 * pad,
                        label_y + text_h + 2 * pad,
                    ),
                    radius=max(2, pad),
                    fill=bg_color,
                    outline=SELECTED_COLOR if selected else color_rgb + (255,),
                    width=max(1, line_width),
                )

                brightness = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
                text_color = (0, 0, 0, 255) if brightness > 150 else (255, 255, 255, 255)
                draw.text(
                    (label_x + pad, label_y + pad),
                    label,
                    font=font,
                    fill=text_color,
                )

    def _draw_mask(
        self,
        base: Image.Image,
        det: Detection,
        color_rgb: tuple[int, int, int],
        opacity: int,
    ) -> None:
        mask = Image.new("L", (self.image_width, self.image_height), 0)
        mask_draw = ImageDraw.Draw(mask)

        outer_polygons: list[tuple[int, list[tuple[float, float]]]] = []
        hole_polygons: list[tuple[int | None, list[tuple[float, float]]]] = []

        for poly_index, polygon in enumerate(det.polygons):
            points_raw = polygon.get("points_xy") or []
            points = []
            for point in points_raw:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        points.append((float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        pass
            if len(points) < 3:
                continue

            if polygon.get("is_hole", False):
                parent = polygon.get("parent_index")
                try:
                    parent = int(parent) if parent is not None else None
                except (TypeError, ValueError):
                    parent = None
                hole_polygons.append((parent, points))
            else:
                outer_polygons.append((poly_index, points))

        for _, points in outer_polygons:
            mask_draw.polygon(points, fill=opacity)

        # Holes are removed from the mask. Parent references are accepted but not required.
        for _, points in hole_polygons:
            mask_draw.polygon(points, fill=0)

        overlay = Image.new("RGBA", base.size, color_rgb + (255,))
        base.alpha_composite(Image.composite(overlay, Image.new("RGBA", base.size), mask))

    def _display_rendered_image(self) -> None:
        if self.rendered_image is None:
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        if self.fit_mode:
            self.zoom = min(canvas_w / self.image_width, canvas_h / self.image_height)
            self.zoom = max(self.zoom, 0.01)
            self.offset_x = (canvas_w - self.image_width * self.zoom) / 2
            self.offset_y = (canvas_h - self.image_height * self.zoom) / 2

        display_w = max(1, int(round(self.image_width * self.zoom)))
        display_h = max(1, int(round(self.image_height * self.zoom)))
        resized = self.rendered_image.resize((display_w, display_h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor=tk.NW,
            image=self.tk_image,
            tags=("rendered_image",),
        )

    # --------------------------- Selection and table -----------------------

    def _populate_table(self) -> None:
        selected_iid = str(self.selected_detection) if self.selected_detection is not None else None
        self.tree.delete(*self.tree.get_children())

        threshold = self.conf_threshold.get()
        for det in self.detections:
            if det.confidence < threshold:
                continue
            class_var = self.class_enabled.get(det.class_name)
            if class_var is not None and not class_var.get():
                continue

            x1, y1, x2, y2 = det.bbox
            bbox_text = f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            area_text = f"{det.mask_area:,.0f}" if det.mask_area is not None else "—"
            iid = str(det.index)
            self.tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(det.index, det.class_name, f"{det.confidence:.4f}", bbox_text, area_text),
            )

        if selected_iid and self.tree.exists(selected_iid):
            self.tree.selection_set(selected_iid)
            self.tree.see(selected_iid)

    def _on_tree_select(self, event: tk.Event | None = None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        try:
            self.selected_detection = int(selection[0])
        except ValueError:
            return
        self.request_render()

    def _select_from_canvas(self, event: tk.Event) -> None:
        if self.zoom <= 0:
            return
        image_x = (event.x - self.offset_x) / self.zoom
        image_y = (event.y - self.offset_y) / self.zoom

        candidates = []
        for det in self.visible_detections():
            x1, y1, x2, y2 = det.bbox
            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                candidates.append(det)

        if not candidates:
            self.selected_detection = None
            self.tree.selection_remove(self.tree.selection())
            self.request_render()
            return

        # Prefer the smallest containing box, then the highest confidence.
        selected = min(candidates, key=lambda det: (det.bbox_area, -det.confidence))
        self.selected_detection = selected.index

        iid = str(selected.index)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)
        self.request_render()

    # ------------------------------ Zoom/pan -------------------------------

    def _on_canvas_resize(self, event: tk.Event) -> None:
        if self.fit_mode:
            self._display_rendered_image()

    def fit_to_window(self) -> None:
        self.fit_mode = True
        self._display_rendered_image()

    def zoom_100(self) -> None:
        self.fit_mode = False
        self.zoom = 1.0
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self.offset_x = (canvas_w - self.image_width) / 2
        self.offset_y = (canvas_h - self.image_height) / 2
        self._display_rendered_image()

    def change_zoom(self, factor: float) -> None:
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self._zoom_at(canvas_w / 2, canvas_h / 2, factor)

    def _on_mousewheel(self, event: tk.Event) -> None:
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        if self.rendered_image is None:
            return

        old_zoom = self.zoom
        new_zoom = max(0.01, min(20.0, old_zoom * factor))
        if math.isclose(new_zoom, old_zoom):
            return

        image_x = (canvas_x - self.offset_x) / old_zoom
        image_y = (canvas_y - self.offset_y) / old_zoom

        self.fit_mode = False
        self.zoom = new_zoom
        self.offset_x = canvas_x - image_x * new_zoom
        self.offset_y = canvas_y - image_y * new_zoom
        self._display_rendered_image()

    def _start_pan(self, event: tk.Event) -> None:
        self.fit_mode = False
        self.pan_start = (event.x, event.y)
        self.pan_origin = (self.offset_x, self.offset_y)
        self.canvas.configure(cursor="fleur")

    def _pan(self, event: tk.Event) -> None:
        if self.pan_start is None or self.pan_origin is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.offset_x = self.pan_origin[0] + dx
        self.offset_y = self.pan_origin[1] + dy
        self._display_rendered_image()

    def _end_pan(self, event: tk.Event) -> None:
        self.pan_start = None
        self.pan_origin = None
        self.canvas.configure(cursor="crosshair")

    # ------------------------------- Export --------------------------------

    def export_dialog(self) -> None:
        if not self.raw_data:
            messagebox.showinfo("Nothing to export", "Open a YOLO JSON file first.")
            return

        default_name = "annotated.png"
        if self.json_path:
            default_name = f"{self.json_path.stem}_annotated.png"

        filename = filedialog.asksaveasfilename(
            title="Export annotated image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png")],
        )
        if filename:
            self.export_annotated(Path(filename))

    def export_annotated(self, path: Path) -> None:
        if self.original_image is not None:
            output = self.original_image.copy().convert("RGBA")
            if output.size != (self.image_width, self.image_height):
                output = output.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        else:
            output = Image.new("RGBA", (self.image_width, self.image_height), DEFAULT_BG + (255,))
            self._draw_background_grid(output)

        self._draw_detections(output, self.visible_detections())

        try:
            output.convert("RGB").save(path, format="PNG")
        except OSError as exc:
            messagebox.showerror("Could not export image", str(exc))
            return

        self.status_var.set(f"Exported: {path}")
        messagebox.showinfo("Export complete", f"Saved:\n{path}")

    # ------------------------------- Helpers -------------------------------

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def show_controls(self) -> None:
        messagebox.showinfo(
            "Controls",
            "Mouse wheel: zoom\n"
            "Middle or right mouse drag: pan\n"
            "Left click a box: select detection\n"
            "F: fit to window\n"
            "1: 100% zoom\n"
            "+ / -: zoom\n"
            "Ctrl+O: open JSON\n"
            "Ctrl+I: open image\n"
            "Ctrl+E: export PNG",
        )

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "YOLO JSON Viewer\n\n"
            "A lightweight Tkinter/Pillow viewer for detection and polygon segmentation output.",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View YOLO detection/segmentation JSON files.")
    parser.add_argument("json", nargs="?", help="Path to the YOLO JSON file")
    parser.add_argument("--image", help="Optional source image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = YoloViewer(json_path=args.json, image_path=args.image)
    app.mainloop()


if __name__ == "__main__":
    main()
