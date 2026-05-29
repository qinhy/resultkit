import cv2
import numpy as np
from pydantic import BaseModel, PositiveFloat

from .MatModel import Model4Mat


class FrameGenerator(BaseModel):
    img:Model4Mat.ImageMat

    name: str = "gen" 
    fps: PositiveFloat = 30.0
    draw_overlay: bool = True
    overlay_prefix: str = "Steam SIM"
    enable_stereo_offset: bool = True
    stereo_offset_px: int = 12
    is_mono: bool = False
    frame_idx: int = 0

    @property
    def color(self):
        return self.img.color_format
    
    def model_post_init(self, context):
        self.is_mono = (self.color in [Model4Mat.ImageMat.ColorFormat.GRAY, Model4Mat.ImageMat.ColorFormat.BAYER])
        return super().model_post_init(context)
    
    def read(self) -> Model4Mat.ImageMat:
        frame = self._make_synthetic_frame()

        if self.draw_overlay:
            self._draw_overlay(frame)

        self.frame_idx += 1
        self.img.unsafe_update_data(frame)
        return self.img
    
    def as_generator(self):
        while True:
            yield self.read()
    
    def _make_synthetic_frame(self) -> np.ndarray:
        t = self.frame_idx / max(self.fps, 1.0)

        if self.is_mono:
            frame = self._make_gray_frame(t)
            frame = self._apply_stereo_offset(frame)
            return frame

        return self._make_bgr_frame(t)

    def _make_gray_frame(self, t: float) -> np.ndarray:
        yy, xx = np.indices(
            (self.img.BCHW[-2], self.img.BCHW[-1]),
            dtype=np.int32,
        )

        phase = int(self.frame_idx * 4)
        frame = ((xx * 2 + yy + phase) % 256).astype(np.uint8)

        cx, cy = self._moving_center(t)

        cv2.circle(
            frame,
            (cx, cy),
            max(12, self.img.BCHW[-2] // 14),
            235,
            -1,
        )

        cv2.rectangle(
            frame,
            (max(0, cx - 90), max(0, cy - 30)),
            (min(self.img.BCHW[-1] - 1, cx + 90), min(self.img.BCHW[-2] - 1, cy + 30)),
            80,
            3,
        )

        return frame

    def _make_bgr_frame(self, t: float) -> np.ndarray:
        x = np.linspace(0, 255, self.img.BCHW[-1], dtype=np.uint8)
        y = np.linspace(0, 255, self.img.BCHW[-2], dtype=np.uint8)

        xv = np.tile(x, (self.img.BCHW[-2], 1))
        yv = np.tile(y[:, None], (1, self.img.BCHW[-1]))

        frame = np.empty(
            (self.img.BCHW[-2], self.img.BCHW[-1], 3),
            dtype=np.uint8,
        )

        frame[..., 0] = (xv.astype(np.uint16) + self.frame_idx * 3) % 256
        frame[..., 1] = (yv.astype(np.uint16) + self.frame_idx * 2) % 256
        frame[..., 2] = (
            xv.astype(np.uint16) // 2
            + yv.astype(np.uint16) // 2
            + self.frame_idx * 5
        ) % 256

        cx, cy = self._moving_center(t)
        radius = max(20, min(self.img.BCHW[-1], self.img.BCHW[-2]) // 10)

        cv2.circle(
            frame,
            (cx, cy),
            radius,
            (0, 0, 255),
            -1,
        )

        cv2.rectangle(
            frame,
            (max(0, cx - radius * 2), max(0, cy - radius)),
            (min(self.img.BCHW[-1] - 1, cx + radius * 2), min(self.img.BCHW[-2] - 1, cy + radius)),
            (255, 0, 0),
            4,
        )

        cv2.line(
            frame,
            (0, cy),
            (self.img.BCHW[-1] - 1, self.img.BCHW[-2] - cy - 1),
            (255, 255, 255),
            2,
        )

        return frame

    def _moving_center(self, t: float) -> tuple[int, int]:
        cx = int(
            self.img.BCHW[-1] * 0.5
            + np.sin(t * 1.2) * self.img.BCHW[-1] * 0.30
        )

        cy = int(
            self.img.BCHW[-2] * 0.5
            + np.cos(t * 0.9) * self.img.BCHW[-2] * 0.28
        )

        return cx, cy

    def _apply_stereo_offset(self, gray: np.ndarray) -> np.ndarray:
        if not self.enable_stereo_offset:
            return gray

        if self.name == "left":
            dx = -self.stereo_offset_px
        elif self.name == "right":
            dx = self.stereo_offset_px
        else:
            dx = 0

        if dx == 0:
            return gray

        matrix = np.float32([[1, 0, dx], [0, 1, 0]])

        return cv2.warpAffine(
            gray,
            matrix,
            (self.img.BCHW[-1], self.img.BCHW[-2]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _draw_overlay(self, frame: np.ndarray) -> None:
        msg = (
            f"{self.overlay_prefix} "
            f"{self.name.upper()} "
            f"frame={self.frame_idx:06d}"
        )

        if self.is_mono:
            cv2.putText(
                frame,
                msg,
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                0,
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                msg,
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                255,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                msg,
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                msg,
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
