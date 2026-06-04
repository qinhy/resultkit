import torch


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


def make_bitmap_font_5x7(device, scale: int = 2):
    chars = list(FONT_5X7.keys())

    glyphs = []
    for ch in chars:
        rows = FONT_5X7[ch]
        glyph = [[c == "1" for c in row] for row in rows]
        glyphs.append(torch.tensor(glyph, dtype=torch.bool))

    atlas = torch.stack(glyphs, dim=0).to(device)

    if scale > 1:
        atlas = atlas.repeat_interleave(scale, dim=1)
        atlas = atlas.repeat_interleave(scale, dim=2)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    # Precompute foreground pixel offsets for each glyph.
    glyph_offsets = {}
    for ch, idx in char_to_idx.items():
        gy, gx = torch.nonzero(atlas[idx], as_tuple=True)
        glyph_offsets[ch] = (gy, gx)

    return atlas, char_to_idx, glyph_offsets


def draw_text_bitmap_gpu(
    out: torch.Tensor,
    text: str,
    x,
    y,
    *,
    atlas: torch.Tensor,
    char_to_idx: dict,
    glyph_offsets: dict,
    fg_rgb=(0, 255, 0),
    bg_rgb=(0, 0, 0),
    spacing: int = 2,
    padding: int = 2,
    enabled=None,
):
    """
    Draw bitmap text into HWC RGB uint8 CUDA tensor.

    x, y may be Python ints or CUDA scalar tensors.
    """
    if not text:
        return

    device = out.device
    h, w = out.shape[:2]
    out_flat = out.view(-1, 3)

    x = torch.as_tensor(x, device=device, dtype=torch.long)
    y = torch.as_tensor(y, device=device, dtype=torch.long)

    if enabled is None:
        enabled = torch.tensor(True, device=device, dtype=torch.bool)
    else:
        enabled = torch.as_tensor(enabled, device=device, dtype=torch.bool)

    text = text.upper()

    glyph_h = atlas.shape[1]
    glyph_w = atlas.shape[2]

    text_w = len(text) * glyph_w + max(0, len(text) - 1) * spacing
    text_h = glyph_h

    total_w = text_w + padding * 2
    total_h = text_h + padding * 2

    fg = torch.tensor(fg_rgb, dtype=torch.uint8, device=device)
    bg = torch.tensor(bg_rgb, dtype=torch.uint8, device=device)

    # Background rectangle.
    if bg_rgb is not None:
        rows = torch.arange(total_h, device=device).view(-1, 1)
        cols = torch.arange(total_w, device=device).view(1, -1)

        yy = y + rows
        xx = x + cols

        valid = (
            enabled
            & (yy >= 0)
            & (yy < h)
            & (xx >= 0)
            & (xx < w)
        )

        lin = yy * w + xx
        out_flat[lin[valid]] = bg

    # Foreground glyph pixels.
    base_y = y + padding
    cursor_x = x + padding

    unknown = "?"

    for ch in text:
        ch = ch.upper()
        if ch not in char_to_idx:
            ch = unknown

        gy, gx = glyph_offsets[ch]

        yy = base_y + gy
        xx = cursor_x + gx

        valid = (
            enabled
            & (yy >= 0)
            & (yy < h)
            & (xx >= 0)
            & (xx < w)
        )

        lin = yy * w + xx
        out_flat[lin[valid]] = fg

        cursor_x = cursor_x + glyph_w + spacing


def draw_boxes_gpu_simple(
    img_uint8: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor | None,
    color_rgb=(0, 255, 0),
    thickness: int = 2,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Draw boxes on GPU using PyTorch only.
    """
    out = img_uint8.clone()

    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return out.contiguous()

    device = img_uint8.device
    h, w = img_uint8.shape[:2]

    boxes = boxes_xyxy.to(device=device).round().to(torch.long)

    x1 = boxes[:, 0].clamp(0, w - 1)
    y1 = boxes[:, 1].clamp(0, h - 1)
    x2 = boxes[:, 2].clamp(0, w - 1)
    y2 = boxes[:, 3].clamp(0, h - 1)

    valid = (x2 > x1) & (y2 > y1)

    yy = torch.arange(h, device=device).view(1, h, 1)
    xx = torch.arange(w, device=device).view(1, 1, w)

    final_mask = torch.zeros((h, w), dtype=torch.bool, device=device)

    n = boxes.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        x1c = x1[start:end].view(-1, 1, 1)
        y1c = y1[start:end].view(-1, 1, 1)
        x2c = x2[start:end].view(-1, 1, 1)
        y2c = y2[start:end].view(-1, 1, 1)
        vc = valid[start:end].view(-1, 1, 1)

        inside = (
            vc
            & (xx >= x1c)
            & (xx <= x2c)
            & (yy >= y1c)
            & (yy <= y2c)
        )

        border = inside & (
            ((xx - x1c) < thickness)
            | ((x2c - xx) < thickness)
            | ((yy - y1c) < thickness)
            | ((y2c - yy) < thickness)
        )

        final_mask |= border.any(dim=0)

    color = torch.tensor(color_rgb, dtype=torch.uint8, device=device)
    out[final_mask] = color

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
) -> torch.Tensor:
    """
    Draw YOLO boxes + simple bitmap labels.

    Input/output:
        CUDA HWC RGB uint8 tensor.

    Image pixels stay on GPU.
    Label string creation is Python-side.
    """
    if img_uint8.ndim != 3 or img_uint8.shape[-1] != 3:
        raise ValueError("img_uint8 must be HWC RGB [H, W, 3].")

    if img_uint8.dtype != torch.uint8:
        raise ValueError("img_uint8 must be uint8.")

    if not img_uint8.is_cuda:
        raise ValueError("img_uint8 must be CUDA.")

    out = draw_boxes_gpu_simple(
        img_uint8,
        boxes_xyxy=boxes_xyxy,
        color_rgb=color_rgb,
        thickness=box_thickness,
    )

    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return out.contiguous()

    if cls is None:
        return out.contiguous()

    device = img_uint8.device
    h, w = img_uint8.shape[:2]

    boxes = boxes_xyxy.to(device=device).round().to(torch.long)

    x1 = boxes[:, 0].clamp(0, w - 1)
    y1 = boxes[:, 1].clamp(0, h - 1)
    x2 = boxes[:, 2].clamp(0, w - 1)
    y2 = boxes[:, 3].clamp(0, h - 1)

    valid = (x2 > x1) & (y2 > y1)

    atlas, char_to_idx, glyph_offsets = make_bitmap_font_5x7(
        device=device,
        scale=font_scale,
    )

    glyph_h = atlas.shape[1]
    label_h = glyph_h + 4

    # Small metadata read only. The image does not move to CPU.
    cls_list = cls.detach().round().to(torch.long).cpu().tolist()

    if conf is not None and draw_scores:
        conf_list = conf.detach().float().cpu().tolist()
    else:
        conf_list = [None] * len(cls_list)

    for i, class_id in enumerate(cls_list):
        if isinstance(names, dict):
            label = names.get(int(class_id), str(int(class_id)))
        else:
            label = str(int(class_id))

        # This bitmap font maps lowercase to uppercase.
        label = str(label).upper()

        if draw_scores and conf_list[i] is not None:
            text = f"{label} {conf_list[i]:.2f}"
        else:
            text = label

        # Prefer above the box; if not enough room, draw inside/below top edge.
        y_above = y1[i] - label_h - 1
        text_y = torch.where(y_above >= 0, y_above, y1[i] + 1)
        text_x = x1[i]

        draw_text_bitmap_gpu(
            out,
            text,
            text_x,
            text_y,
            atlas=atlas,
            char_to_idx=char_to_idx,
            glyph_offsets=glyph_offsets,
            fg_rgb=color_rgb,
            bg_rgb=text_bg_rgb,
            spacing=max(1, font_scale),
            padding=2,
            enabled=valid[i],
        )

    return out.contiguous()