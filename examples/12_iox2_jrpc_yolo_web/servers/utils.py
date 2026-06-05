import torch
from collections import OrderedDict
from collections.abc import Mapping, Sequence


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],

    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],

    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
}


TEXT_BITMAP_CACHE_MAX_SIZE = 4096
BITMAP_FONT_CACHE: dict[tuple, tuple[torch.Tensor, dict[str, int], dict[str, tuple[torch.Tensor, torch.Tensor]]]] = {}
TEXT_BITMAP_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()


DEFAULT_COLOR_PALETTE_RGB = [
            (0, 255, 0),          # person , 0
            (255, 128, 0),        # bicycle , 1
            (255, 0, 0),          # car , 2
            (255, 0, 255),        # motorcycle , 3
            (128, 0, 255),        # airplane , 4
            (0, 128, 255),        # bus , 5
            (0, 255, 255),        # train , 6
            (255, 255, 0),        # truck , 7
            (128, 255, 0),        # boat , 8
            (255, 128, 128),      # traffic light , 9
            (255, 64, 64),       # fire hydrant , 10
            (255, 255, 255),     # stop sign , 11
            (192, 192, 192),     # parking meter , 12
            (128, 128, 128),     # bench , 13

            (0, 200, 255),       # bird , 14
            (255, 200, 0),       # cat , 15
            (255, 100, 0),       # dog , 16
            (160, 82, 45),       # horse , 17
            (220, 220, 220),     # sheep , 18
            (139, 69, 19),       # cow , 19
            (128, 128, 0),       # elephant , 20
            (165, 42, 42),       # bear , 21
            (255, 255, 128),     # zebra , 22
            (255, 180, 0),       # giraffe , 23

            (64, 128, 255),      # backpack , 24
            (255, 64, 128),      # umbrella , 25
            (192, 64, 255),      # handbag , 26
            (64, 255, 128),      # tie , 27
            (128, 64, 255),      # suitcase , 28

            (0, 255, 128),       # frisbee , 29
            (128, 255, 255),     # skis , 30
            (0, 128, 128),       # snowboard , 31
            (255, 128, 255),     # sports ball , 32
            (128, 0, 128),       # kite , 33
            (255, 200, 128),     # baseball bat , 34
            (200, 128, 64),      # baseball glove , 35
            (128, 255, 128),     # skateboard , 36
            (64, 200, 255),      # surfboard , 37
            (200, 255, 64),      # tennis racket , 38

            (0, 180, 255),       # bottle , 39
            (180, 0, 255),       # wine glass , 40
            (255, 180, 180),     # cup , 41
            (180, 255, 180),     # fork , 42
            (255, 255, 180),     # knife , 43
            (180, 180, 255),     # spoon , 44
            (255, 128, 64),      # bowl , 45

            (255, 255, 64),      # banana , 46
            (255, 0, 64),        # apple , 47
            (255, 180, 64),      # sandwich , 48
            (255, 100, 64),      # orange , 49
            (0, 200, 0),         # broccoli , 50
            (255, 140, 0),       # carrot , 51
            (255, 80, 80),       # hot dog , 52
            (255, 120, 0),       # pizza , 53
            (210, 105, 30),      # donut , 54
            (255, 192, 203),     # cake , 55

            (128, 80, 40),       # chair , 56
            (160, 120, 80),      # couch , 57
            (0, 160, 80),        # potted plant , 58
            (180, 120, 255),     # bed , 59
            (120, 80, 40),       # dining table , 60
            (220, 220, 255),     # toilet , 61
            (64, 64, 255),       # tv , 62
            (0, 64, 255),        # laptop , 63
            (128, 128, 255),     # mouse , 64
            (64, 128, 128),      # remote , 65
            (128, 64, 64),       # keyboard , 66
            (0, 200, 200),       # cell phone , 67

            (200, 200, 200),     # microwave , 68
            (180, 180, 180),     # oven , 69
            (255, 220, 128),     # toaster , 70
            (128, 200, 255),     # sink , 71
            (180, 220, 255),     # refrigerator , 72

            (100, 100, 255),     # book , 73
            (255, 255, 200),     # clock , 74
            (200, 128, 255),     # vase , 75
            (255, 100, 200),     # scissors , 76
            (160, 100, 60),      # teddy bear , 77
            (220, 180, 140),     # hair drier , 78
            (180, 255, 220),     # toothbrush , 79            
]


def validate_positive_int(value, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value


def torch_device_cache_key(device):
    d = torch.device(device)
    index = d.index

    # torch.device("cuda") has no explicit index. Resolve it so cache entries
    # do not accidentally get shared across active CUDA devices.
    if d.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()

    return d.type, index


def lookup_color_in_mapping(
    color_map: Mapping,
    class_id: int,
    *,
    names=None,
):
    label = label_for_class(class_id, names)
    candidates = (class_id, str(class_id), label, label.upper(), label.lower())

    for key in candidates:
        if key in color_map:
            return color_map[key]

    for key in ("default", "DEFAULT", None):
        if key in color_map:
            return color_map[key]

    return DEFAULT_COLOR_PALETTE_RGB[class_id % len(DEFAULT_COLOR_PALETTE_RGB)]


def resolve_color_tensor(
    color_rgb,
    *,
    device,
    n: int | None = None,
    cls_cpu: torch.Tensor | None = None,
    names=None,
    color_mode: str = "class",
    dtype=torch.uint8,
    name: str = "color_rgb",
) -> torch.Tensor:
    """Resolve a single RGB color or one RGB color per box.

    Accepted inputs:
      - ``(r, g, b)``: one color for all boxes
      - ``[(r, g, b), ...]`` / ``Tensor[K, 3]`` with ``color_mode='class'``:
        palette indexed by ``cls``
      - ``[(r, g, b), ...]`` / ``Tensor[N, 3]`` with ``color_mode='box'``:
        one color per box
      - ``{class_id/name: (r, g, b), ...}``: class/name keyed colors
    """
    color_mode = str(color_mode).lower()
    if color_mode not in {"class", "box"}:
        raise ValueError("color_mode must be 'class' or 'box'.")

    if isinstance(color_rgb, Mapping):
        if n is None:
            raise ValueError(f"n is required when resolving mapped {name}.")
        if cls_cpu is None:
            raise ValueError(f"{name} mappings require cls to be provided.")

        rows = [
            lookup_color_in_mapping(color_rgb, int(cls_cpu[i]), names=names)
            for i in range(min(n, int(cls_cpu.shape[0])))
        ]
        if len(rows) != n:
            raise ValueError(f"{name} mapping could only resolve {len(rows)} colors for {n} boxes.")
        return torch.as_tensor(rows, dtype=dtype, device=device).reshape(n, 3)

    colors = torch.as_tensor(color_rgb, dtype=dtype, device=device)

    if colors.ndim == 1 and colors.numel() == 3:
        return colors.reshape(3)

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(
            f"{name} must be an RGB triplet, an [N, 3] per-box tensor/list, "
            "a [K, 3] class palette, or a mapping."
        )

    if colors.shape[0] == 0:
        raise ValueError(f"{name} palette/per-box colors cannot be empty.")

    if n is None:
        return colors.contiguous()

    if color_mode == "box":
        if colors.shape[0] < n:
            raise ValueError(f"{name} has {colors.shape[0]} colors but {n} boxes need to be drawn.")
        return colors[:n].contiguous()

    if cls_cpu is None:
        raise ValueError(f"{name} class palettes require cls to be provided, or use color_mode='box'.")

    cls_idx = cls_cpu[:n].to(dtype=torch.long, device=device)
    cls_idx = torch.remainder(cls_idx, colors.shape[0])
    return colors.index_select(0, cls_idx).contiguous()


def select_color_for_item(colors: torch.Tensor, index: int) -> torch.Tensor:
    if colors.ndim == 1:
        return colors
    return colors[int(index)]


def clip_box_xyxy(box, *, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Round and clip an XYXY box for Python slicing with x2/y2 exclusive."""
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def boxes_to_cpu_xyxy(boxes_xyxy: torch.Tensor | None) -> torch.Tensor | None:
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return None

    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must have shape [N, 4].")

    # Python slicing needs Python ints, so box metadata must be on CPU.
    return boxes_xyxy.detach().round().to(dtype=torch.long, device="cpu")


def label_for_class(class_id: int, names=None) -> str:
    if names is None:
        return str(class_id)

    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))

    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    return str(class_id)


def make_bitmap_font_5x7(device, scale: int = 2):
    """Build a 5x7 bitmap font atlas on ``device``.

    Returns ``(atlas, char_to_idx, glyph_offsets)`` for compatibility with the
    original implementation. ``glyph_offsets`` can be useful for custom sparse
    drawing routines, even though the fast path below uses dense text masks.
    """
    scale = validate_positive_int(scale, "scale")
    chars = list(FONT_5X7.keys())

    glyphs = []
    for ch in chars:
        rows = FONT_5X7[ch]
        glyph = [[c == "1" for c in row] for row in rows]
        glyphs.append(torch.tensor(glyph, dtype=torch.bool))

    atlas = torch.stack(glyphs, dim=0).to(device=device, non_blocking=True)

    if scale > 1:
        atlas = atlas.repeat_interleave(scale, dim=1)
        atlas = atlas.repeat_interleave(scale, dim=2)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    glyph_offsets = {}
    for ch, idx in char_to_idx.items():
        gy, gx = torch.nonzero(atlas[idx], as_tuple=True)
        glyph_offsets[ch] = (gy, gx)

    return atlas.contiguous(), char_to_idx, glyph_offsets


def get_bitmap_font_5x7_cached(device, scale: int = 2):
    scale = validate_positive_int(scale, "scale")
    key = (torch_device_cache_key(device), scale)

    cached = BITMAP_FONT_CACHE.get(key)
    if cached is None:
        cached = make_bitmap_font_5x7(device=device, scale=scale)
        BITMAP_FONT_CACHE[key] = cached

    return cached


def clear_bitmap_caches() -> None:
    """Clear cached font atlases and rendered text masks."""
    BITMAP_FONT_CACHE.clear()
    TEXT_BITMAP_CACHE.clear()


def get_text_bitmap_mask_cached(
    text: str,
    *,
    atlas: torch.Tensor,
    char_to_idx: dict,
    spacing: int,
    padding: int,
) -> torch.Tensor:
    text = str(text).upper()
    if not text:
        return torch.empty((0, 0), dtype=torch.bool, device=atlas.device)

    spacing = max(0, int(spacing))
    padding = max(0, int(padding))

    glyph_h = int(atlas.shape[1])
    glyph_w = int(atlas.shape[2])

    key = (
        torch_device_cache_key(atlas.device),
        glyph_h,
        glyph_w,
        spacing,
        padding,
        text,
    )

    cached = TEXT_BITMAP_CACHE.get(key)
    if cached is not None:
        TEXT_BITMAP_CACHE.move_to_end(key)
        return cached

    text_w = len(text) * glyph_w + max(0, len(text) - 1) * spacing
    total_w = text_w + padding * 2
    total_h = glyph_h + padding * 2

    mask = torch.zeros((total_h, total_w), dtype=torch.bool, device=atlas.device)

    unknown_idx = char_to_idx.get("?", char_to_idx.get(" ", 0))

    cursor_x = padding
    for ch in text:
        idx = char_to_idx.get(ch, unknown_idx)
        mask[padding:padding + glyph_h, cursor_x:cursor_x + glyph_w] |= atlas[idx]
        cursor_x += glyph_w + spacing

    if len(TEXT_BITMAP_CACHE) >= TEXT_BITMAP_CACHE_MAX_SIZE:
        TEXT_BITMAP_CACHE.popitem(last=False)

    TEXT_BITMAP_CACHE[key] = mask
    return mask


def draw_text_bitmap_gpu_fast(
    out: torch.Tensor,
    text: str,
    x: int,
    y: int,
    *,
    atlas: torch.Tensor,
    char_to_idx: dict,
    fg: torch.Tensor,
    bg: torch.Tensor | None = None,
    spacing: int = 2,
    padding: int = 2,
    enabled: bool = True,
) -> None:
    """Draw one clipped bitmap label using a cached full-text mask.

    This clips once, optionally fills the background once, then writes all glyph
    pixels in one boolean-indexed assignment. ``out`` is modified in place.
    """
    if not enabled or not text:
        return

    h, w = out.shape[:2]

    mask = get_text_bitmap_mask_cached(
        text,
        atlas=atlas,
        char_to_idx=char_to_idx,
        spacing=spacing,
        padding=padding,
    )

    mh, mw = mask.shape
    if mh == 0 or mw == 0:
        return

    x = int(x)
    y = int(y)

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + mw)
    y1 = min(h, y + mh)

    if x1 <= x0 or y1 <= y0:
        return

    mx0 = x0 - x
    my0 = y0 - y
    mx1 = mx0 + (x1 - x0)
    my1 = my0 + (y1 - y0)

    region = out[y0:y1, x0:x1]
    mask_view = mask[my0:my1, mx0:mx1]

    if bg is not None:
        region[:] = bg

    region[mask_view] = fg


def draw_box_borders_inplace(
    out: torch.Tensor,
    boxes_cpu: torch.Tensor,
    *,
    color: torch.Tensor,
    thickness: int,
) -> None:
    h, w = out.shape[:2]
    thickness = validate_positive_int(thickness, "thickness")

    if color.ndim == 1:
        if color.numel() != 3:
            raise ValueError("color must have shape [3] or [N, 3].")
    elif color.ndim == 2:
        if color.shape[1] != 3 or color.shape[0] < int(boxes_cpu.shape[0]):
            raise ValueError("color must have shape [3] or [N, 3] with one row per box.")
    else:
        raise ValueError("color must have shape [3] or [N, 3].")

    for i, box in enumerate(boxes_cpu.tolist()):
        clipped = clip_box_xyxy(box, width=w, height=h)
        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        t = min(thickness, x2 - x1, y2 - y1)
        color_i = select_color_for_item(color, i)

        out[y1:y1 + t, x1:x2] = color_i
        out[y2 - t:y2, x1:x2] = color_i
        out[y1:y2, x1:x1 + t] = color_i
        out[y1:y2, x2 - t:x2] = color_i


def darken_box_regions_inplace(
    out: torch.Tensor,
    boxes_cpu: torch.Tensor,
    *,
    dim_shift: int,
) -> None:
    h, w = out.shape[:2]
    dim_shift = max(0, int(dim_shift))

    for box in boxes_cpu.tolist():
        clipped = clip_box_xyxy(box, width=w, height=h)
        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        out[y1:y2, x1:x2].bitwise_right_shift_(dim_shift)


def draw_boxes_gpu_simple(
    img_uint8: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor | None,
    dim_shift: int = 1,
    color_rgb=None,
    thickness: int = 2,
) -> torch.Tensor:
    """Draw a quick box overlay on a cloned HWC uint8 RGB image.

    By default this preserves the original behavior and darkens each full box
    region. Pass ``color_rgb=(r, g, b)`` to draw one border color, or pass
    ``color_rgb`` as ``[N, 3]`` / ``Tensor[N, 3]`` to use one RGB color per
    box.
    """
    if img_uint8.ndim != 3 or img_uint8.shape[-1] != 3:
        raise ValueError("img_uint8 must be HWC RGB [H, W, 3].")

    if img_uint8.dtype != torch.uint8:
        raise ValueError("img_uint8 must be uint8.")

    out = img_uint8.clone().contiguous()
    boxes_cpu = boxes_to_cpu_xyxy(boxes_xyxy)
    if boxes_cpu is None:
        return out

    if color_rgb is None:
        darken_box_regions_inplace(out, boxes_cpu, dim_shift=dim_shift)
    else:
        color = resolve_color_tensor(
            color_rgb,
            device=out.device,
            n=int(boxes_cpu.shape[0]),
            color_mode="box",
        )
        draw_box_borders_inplace(out, boxes_cpu, color=color, thickness=thickness)

    return out.contiguous()


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
    """Draw YOLO-style boxes and bitmap labels on a CUDA HWC uint8 RGB image.

    Key properties:
      - cached font atlas and cached full-label bitmap masks
      - one CPU metadata transfer for boxes/classes/scores
      - colored box borders honor ``color_rgb`` and ``box_thickness``
      - ``color_rgb`` accepts one RGB triplet, a class palette, per-box colors,
        or a class/name keyed mapping
      - labels support ``names`` as either a dict or a sequence

    Multi-color examples:
      - ``color_rgb=[(255, 0, 0), (0, 255, 0), ...]`` with the default
        ``color_mode='class'`` indexes the palette by ``cls``.
      - ``color_rgb=colors_per_detection`` with ``color_mode='box'`` uses one
        ``[r, g, b]`` row per box.
      - ``color_rgb={0: (255, 0, 0), "person": (0, 255, 0)}`` supports class
        ids or names.
    """
    if img_uint8.ndim != 3 or img_uint8.shape[-1] != 3:
        raise ValueError("img_uint8 must be HWC RGB [H, W, 3].")

    if img_uint8.dtype != torch.uint8:
        raise ValueError("img_uint8 must be uint8.")

    if not img_uint8.is_cuda:
        raise ValueError("img_uint8 must be CUDA.")

    font_scale = validate_positive_int(font_scale, "font_scale")
    box_thickness = validate_positive_int(box_thickness, "box_thickness")

    boxes_cpu = boxes_to_cpu_xyxy(boxes_xyxy)
    if boxes_cpu is None:
        return img_uint8.clone().contiguous()

    device = img_uint8.device
    h, w = img_uint8.shape[:2]

    # Labels require Python strings, so small metadata has to come to CPU.
    # The box metadata is reused for both box drawing and label placement.
    cls_cpu = None if cls is None else cls.detach().round().to(dtype=torch.long, device="cpu")
    conf_cpu = None
    if conf is not None and draw_scores:
        conf_cpu = conf.detach().float().to(device="cpu")

    out = img_uint8.clone().contiguous()
    n_boxes = int(boxes_cpu.shape[0])
    box_color = resolve_color_tensor(
        color_rgb,
        device=device,
        n=n_boxes,
        cls_cpu=cls_cpu,
        names=names,
        color_mode=color_mode,
        name="color_rgb",
    )
    draw_box_borders_inplace(
        out,
        boxes_cpu,
        color=box_color,
        thickness=box_thickness,
    )

    if cls_cpu is None:
        return out.contiguous()

    n = min(int(boxes_cpu.shape[0]), int(cls_cpu.shape[0]))
    if conf_cpu is not None:
        n = min(n, int(conf_cpu.shape[0]))

    if n == 0:
        return out.contiguous()

    atlas, char_to_idx, _ = get_bitmap_font_5x7_cached(
        device=device,
        scale=font_scale,
    )

    spacing = max(1, int(font_scale))
    padding = 2
    label_h = int(atlas.shape[1]) + padding * 2

    bg_color = None
    if text_bg_rgb is not None:
        bg_color = resolve_color_tensor(
            text_bg_rgb,
            device=device,
            n=n,
            cls_cpu=cls_cpu,
            names=names,
            color_mode=color_mode,
            name="text_bg_rgb",
        )

    for i in range(n):
        clipped = clip_box_xyxy(boxes_cpu[i].tolist(), width=w, height=h)
        if clipped is None:
            continue

        x1, y1, _x2, _y2 = clipped
        class_id = int(cls_cpu[i])
        label = label_for_class(class_id, names).upper()

        text = f"{label} {float(conf_cpu[i]):.2f}" if conf_cpu is not None else label

        text_x = x1
        y_above = y1 - label_h - 1
        text_y = y_above if y_above >= 0 else y1 + 1

        draw_text_bitmap_gpu_fast(
            out,
            text,
            text_x,
            text_y,
            atlas=atlas,
            char_to_idx=char_to_idx,
            fg=select_color_for_item(box_color, i),
            bg=None if bg_color is None else select_color_for_item(bg_color, i),
            spacing=spacing,
            padding=padding,
            enabled=True,
        )

    return out.contiguous()


