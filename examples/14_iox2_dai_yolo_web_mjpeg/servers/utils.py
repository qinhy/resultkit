"""Small CUDA bbox + GNU Unifont 16px bitmap label drawer.

This module keeps the old public function style:

    draw_boxes_gpu_with_bitmap_labels(img_uint8, boxes_xyxy=..., conf=..., cls=..., names=...)

Font handling is explicit but cached. Call set_default_font_hex_path(...) once, or set
UNIFONT_HEX_PATH, before drawing labels.

Input image: HWC RGB uint8 torch.Tensor on CUDA.
Font file: GNU Unifont-style .hex. ASCII glyphs are usually 8x16; CJK glyphs
are usually 16x16. Both are drawn as bitmap masks on GPU tensors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
from collections.abc import Mapping, Sequence

import torch


Glyph = tuple[torch.Tensor, int]      # bool[16, width], width is 8 or 16
GlyphMap = dict[int, Glyph]           # codepoint -> glyph
_FONT_CACHE: dict[tuple[str, tuple[str, int | None]], GlyphMap] = {}
_MISS_CACHE: dict[str, set[int]] = {}
_DEFAULT_FONT_HEX_PATH: str | None = None


DEFAULT_COLOR_PALETTE_RGB = [
    (0, 255, 0), (255, 128, 0), (255, 0, 0), (255, 0, 255),
    (128, 0, 255), (0, 128, 255), (0, 255, 255), (255, 255, 0),
    (128, 255, 0), (255, 128, 128), (255, 64, 64), (255, 255, 255),
    (192, 192, 192), (128, 128, 128), (0, 200, 255), (255, 200, 0),
    (255, 100, 0), (160, 82, 45), (220, 220, 220), (139, 69, 19),
    (128, 128, 0), (165, 42, 42), (255, 255, 128), (255, 180, 0),
]


def set_default_font_hex_path(path: str) -> None:
    """Set the GNU Unifont .hex path used by the old-style draw function."""
    global _DEFAULT_FONT_HEX_PATH
    _DEFAULT_FONT_HEX_PATH = str(Path(path).expanduser().resolve())


def clear_font_cache() -> None:
    """Clear cached glyphs and remembered missing codepoints."""
    _FONT_CACHE.clear()
    _MISS_CACHE.clear()


def _font_path() -> str:
    path = _DEFAULT_FONT_HEX_PATH or os.environ.get("UNIFONT_HEX_PATH")
    if not path:
        raise ValueError(
            "No Unifont .hex path set. Call set_default_font_hex_path('/path/to/unifont.hex') "
            "or set UNIFONT_HEX_PATH before drawing labels."
        )
    return str(Path(path).expanduser().resolve())


def _device_key(device) -> tuple[str, int | None]:
    d = torch.device(device)
    idx = d.index
    if d.type == "cuda" and idx is None and torch.cuda.is_available():
        idx = torch.cuda.current_device()
    return d.type, idx


def _positive_int(value, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _glyph_from_hex(hex_bits: str) -> Glyph:
    """Parse one GNU Unifont .hex glyph into bool[16, 8] or bool[16, 16]."""
    raw = bytes.fromhex(hex_bits.strip())
    if len(raw) == 16:
        width = 8       # one byte per row
    elif len(raw) == 32:
        width = 16      # two bytes per row
    else:
        raise ValueError(f"expected 16 or 32 glyph bytes, got {len(raw)}")

    g = torch.zeros((16, width), dtype=torch.bool)
    row_bytes = width // 8
    k = 0
    for y in range(16):
        row = int.from_bytes(raw[k:k + row_bytes], "big")
        k += row_bytes
        for x in range(width):
            g[y, x] = bool(row & (1 << (width - 1 - x)))
    return g, width


def cached_hex16(path: str, chars: Iterable[str], *, device=None) -> GlyphMap:
    """Load requested glyphs from .hex once, then reuse them on later calls."""
    path = str(Path(path).expanduser().resolve())
    device = torch.device("cuda" if device is None else device)
    key = (path, _device_key(device))
    glyphs = _FONT_CACHE.setdefault(key, {})
    misses = _MISS_CACHE.setdefault(path, set())

    need = {ord(c) for c in chars}
    missing = need - set(glyphs) - misses
    if not missing:
        return glyphs

    found: set[int] = set()
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            cp_s, bits = line.strip().split(":", 1)
            cp = int(cp_s, 16)
            if cp not in missing:
                continue
            mask, width = _glyph_from_hex(bits)
            glyphs[cp] = (mask.to(device=device), width)
            found.add(cp)
            if len(found) == len(missing):
                break

    misses.update(missing - found)
    return glyphs


def _missing_glyph(device=None) -> Glyph:
    """Simple 16x16 square fallback glyph."""
    g = torch.zeros((16, 16), dtype=torch.bool, device=device)
    g[1, 1:15] = True
    g[14, 1:15] = True
    g[1:15, 1] = True
    g[1:15, 14] = True
    return g, 16


def _text_mask16(text: str, font: GlyphMap, *, spacing: int, scale: int, device) -> torch.Tensor:
    text = str(text)
    if not text:
        return torch.empty((0, 0), dtype=torch.bool, device=device)

    spacing = max(0, int(spacing))
    scale = _positive_int(scale, "font_scale")
    fallback = _missing_glyph(device=device)

    parts: list[torch.Tensor] = []
    widths: list[int] = []
    for ch in text:
        glyph, width = font.get(ord(ch), fallback)
        if glyph.device != device:
            glyph = glyph.to(device=device)
        parts.append(glyph)
        widths.append(width)

    total_w = sum(widths) + spacing * max(0, len(parts) - 1)
    mask = torch.zeros((16, total_w), dtype=torch.bool, device=device)

    x = 0
    for glyph, width in zip(parts, widths):
        mask[:, x:x + width] = glyph
        x += width + spacing

    if scale > 1:
        mask = mask.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
    return mask


def chars_from_labels(labels: Iterable[str]) -> str:
    """Unique chars helper for cache preloading."""
    return "".join(sorted({c for s in labels for c in str(s)}))


def label_for_class(class_id: int, names=None) -> str:
    if names is None:
        return str(class_id)
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return str(class_id)


def _lookup_color_in_mapping(color_map: Mapping, class_id: int, *, names=None):
    label = label_for_class(class_id, names)
    candidates = (class_id, str(class_id), label, label.upper(), label.lower())
    for key in candidates:
        if key in color_map:
            return color_map[key]
    for key in ("default", "DEFAULT", None):
        if key in color_map:
            return color_map[key]
    return DEFAULT_COLOR_PALETTE_RGB[class_id % len(DEFAULT_COLOR_PALETTE_RGB)]


def _resolve_color_tensor(
    color_rgb,
    *,
    device,
    n: int,
    cls_cpu: torch.Tensor | None,
    names=None,
    color_mode: str,
    name: str,
) -> torch.Tensor:
    """Resolve [3], [N,3], class palette [K,3], or class/name mapping."""
    color_mode = str(color_mode).lower()
    if color_mode not in {"class", "box"}:
        raise ValueError("color_mode must be 'class' or 'box'")

    if isinstance(color_rgb, Mapping):
        if cls_cpu is None:
            raise ValueError(f"{name} mappings require cls to be provided")
        rows = [_lookup_color_in_mapping(color_rgb, int(cls_cpu[i]), names=names) for i in range(n)]
        return torch.as_tensor(rows, dtype=torch.uint8, device=device).reshape(n, 3)

    colors = torch.as_tensor(color_rgb, dtype=torch.uint8, device=device)
    if colors.ndim == 1 and colors.numel() == 3:
        return colors.reshape(3)

    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] == 0:
        raise ValueError(f"{name} must be RGB [3], [N,3], class palette [K,3], or mapping")

    if color_mode == "box":
        if colors.shape[0] < n:
            raise ValueError(f"{name} has {colors.shape[0]} rows, but {n} boxes need colors")
        return colors[:n].contiguous()

    if cls_cpu is None:
        raise ValueError(f"{name} class palettes require cls, or use color_mode='box'")
    idx = torch.remainder(cls_cpu[:n].to(dtype=torch.long, device=device), colors.shape[0])
    return colors.index_select(0, idx).contiguous()


def _color_at(colors: torch.Tensor, i: int) -> torch.Tensor:
    return colors if colors.ndim == 1 else colors[i]


def _clip_xyxy(box, *, width: int, height: int):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _boxes_to_cpu_xyxy(boxes_xyxy: torch.Tensor | None) -> torch.Tensor | None:
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return None
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must have shape [N, 4]")
    return boxes_xyxy.detach().round().to(dtype=torch.long, device="cpu")


def _put_mask(out: torch.Tensor, mask: torch.Tensor, x: int, y: int, fg, bg=None) -> None:
    h, w = out.shape[:2]
    mh, mw = mask.shape
    if mh == 0 or mw == 0:
        return

    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x) + mw), min(h, int(y) + mh)
    if x1 <= x0 or y1 <= y0:
        return

    mx0, my0 = x0 - int(x), y0 - int(y)
    region = out[y0:y1, x0:x1]
    m = mask[my0:my0 + y1 - y0, mx0:mx0 + x1 - x0]
    if bg is not None:
        region[:] = bg
    region[m] = fg


def _draw_box(out: torch.Tensor, box, color: torch.Tensor, thickness: int) -> tuple[int, int, int, int] | None:
    h, w = out.shape[:2]
    clipped = _clip_xyxy(box, width=w, height=h)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    t = min(_positive_int(thickness, "box_thickness"), x2 - x1, y2 - y1)
    out[y1:y1 + t, x1:x2] = color
    out[y2 - t:y2, x1:x2] = color
    out[y1:y2, x1:x1 + t] = color
    out[y1:y2, x2 - t:x2] = color
    return clipped


def draw_boxes_gpu_with_bitmap_labels(
    img_uint8: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor | None,
    conf: torch.Tensor | None,
    cls: torch.Tensor | None,
    names=None,
    color_rgb=(0, 255, 0),
    text_bg_rgb=(0, 0, 0),
    box_thickness: int = 2,
    font_scale: int = 2,
    draw_scores: bool = True,
    color_mode: str = "class",
) -> torch.Tensor:
    """Draw YOLO-style boxes and 16px bitmap labels on a CUDA HWC uint8 RGB image.

    This keeps the old function signature while using a GNU Unifont .hex bitmap
    font internally. Set the font path once with set_default_font_hex_path(...),
    or set the UNIFONT_HEX_PATH environment variable.

    color_rgb and text_bg_rgb accept:
      - (r, g, b): one color for all boxes
      - [N, 3] with color_mode='box': one color per box
      - [K, 3] with color_mode='class': palette indexed by cls
      - {class_id/name: (r, g, b), 'default': (...)} mappings

    font_scale is kept for compatibility. With this 16px font path,
    font_scale=2 means native 16px text, font_scale=4 means 32px text, etc.
    """
    if img_uint8.ndim != 3 or img_uint8.shape[-1] != 3:
        raise ValueError("img_uint8 must be HWC RGB [H, W, 3]")
    if img_uint8.dtype != torch.uint8:
        raise ValueError("img_uint8 must be uint8")
    if not img_uint8.is_cuda:
        raise ValueError("img_uint8 must be CUDA")

    box_thickness = _positive_int(box_thickness, "box_thickness")
    font_scale = _positive_int(font_scale, "font_scale")

    boxes_cpu = _boxes_to_cpu_xyxy(boxes_xyxy)
    if boxes_cpu is None:
        return img_uint8.clone().contiguous()

    out = img_uint8.clone().contiguous()
    device = out.device
    n_boxes = int(boxes_cpu.shape[0])

    cls_cpu = None if cls is None else cls.detach().round().to(dtype=torch.long, device="cpu")
    conf_cpu = None
    if conf is not None and draw_scores:
        conf_cpu = conf.detach().float().to(device="cpu")

    n_color = n_boxes
    if cls_cpu is not None:
        n_color = min(n_color, int(cls_cpu.shape[0])) if str(color_mode).lower() == "class" else n_color

    box_colors = _resolve_color_tensor(
        color_rgb,
        device=device,
        n=n_boxes,
        cls_cpu=cls_cpu,
        names=names,
        color_mode=color_mode,
        name="color_rgb",
    )

    bg_colors = None
    if text_bg_rgb is not None:
        bg_colors = _resolve_color_tensor(
            text_bg_rgb,
            device=device,
            n=n_boxes,
            cls_cpu=cls_cpu,
            names=names,
            color_mode=color_mode,
            name="text_bg_rgb",
        )

    # Draw boxes first.
    clipped_boxes: list[tuple[int, int, int, int] | None] = []
    for i, box in enumerate(boxes_cpu.tolist()):
        clipped_boxes.append(_draw_box(out, box, _color_at(box_colors, i), box_thickness))

    # Labels require cls. This mirrors the old behavior.
    if cls_cpu is None:
        return out.contiguous()

    n_labels = min(n_boxes, int(cls_cpu.shape[0]))
    if conf_cpu is not None:
        n_labels = min(n_labels, int(conf_cpu.shape[0]))
    if n_labels == 0:
        return out.contiguous()

    labels: list[str] = []
    for i in range(n_labels):
        label = label_for_class(int(cls_cpu[i]), names).upper()
        if conf_cpu is not None:
            label = f"{label} {float(conf_cpu[i]):.2f}"
        labels.append(label)

    font = cached_hex16(_font_path(), chars_from_labels(labels), device=img_uint8.device)

    # Compatibility mapping: old 5x7 font_scale=2 roughly means native 16px here.
    render_scale = max(1, int(round(font_scale / 2)))
    spacing = max(1, render_scale)

    for i, label in enumerate(labels):
        clipped = clipped_boxes[i]
        if clipped is None or not label:
            continue
        x1, y1, _x2, _y2 = clipped
        fg = _color_at(box_colors, i)
        bg = None if bg_colors is None else _color_at(bg_colors, i)
        mask = _text_mask16(label, font, spacing=spacing, scale=render_scale, device=device)
        label_y = y1 - int(mask.shape[0]) - 1 if y1 > int(mask.shape[0]) else y1 + box_thickness + 1
        _put_mask(out, mask, x1, label_y, fg, bg)

    return out.contiguous()
