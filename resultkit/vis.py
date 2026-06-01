from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from .MatModel import Model4Mat
from .gen import FrameGenerator


class FrameGeneratorVisualizer(BaseModel):
    generator: Union[FrameGenerator, Model4Mat.ImageMatPubSub]
    window_name: Optional[str] = None
    delay_ms: PositiveInt = 1
    batch_index: NonNegativeInt = 0
    convert_rgb_to_bgr: bool = True
    _is_gen: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context):
        self._is_gen =  isinstance(self.generator, FrameGenerator)
        return super().model_post_init(context)

    @property
    def title(self) -> str:
        return self.window_name or self.generator.name

    def read(self) -> np.ndarray:
        if self._is_gen:
           res = self.generator.read()
        else:
            img:Model4Mat.ImageMatPubSub = self.generator
            res = img.sub()
        return self.image_to_cv_frame(res, self.batch_index, self.convert_rgb_to_bgr)

    def show_once(self) -> int:
        frame = self.read()
        cv2.imshow(self.title, frame)
        return cv2.waitKey(self.delay_ms) & 0xFF

    def run(self, max_frames: Optional[int] = None) -> None:
        frames = 0
        try:
            while max_frames is None or frames < max_frames:
                key = self.show_once()
                frames += 1
                if key in (ord("q"), 27):
                    break
        finally:
            self.close()

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass

    @staticmethod
    def image_to_cv_frame(
        img: Model4Mat.ImageMat,
        batch_index: int = 0,
        convert_rgb_to_bgr: bool = True,
    ) -> np.ndarray:
        data = img.get_data()
        if hasattr(data, "detach"):
            data = data.detach().cpu()
            if data.dtype.is_floating_point:
                data = data.mul(255.0).clamp(0, 255).byte()
            data = data.numpy()

        frame = FrameGeneratorVisualizer._select_frame(np.asarray(data), img.shape_type, batch_index)

        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]

        if frame.ndim == 3:
            if img.color_format == Model4Mat.ImageMat.ColorFormat.RGB and convert_rgb_to_bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif img.color_format == Model4Mat.ImageMat.ColorFormat.BGR:
                frame = frame
            else:
                frame = frame[..., :3]

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(frame)

    @staticmethod
    def _select_frame(
        data: np.ndarray,
        shape_type: Model4Mat.ImageMat.ShapeType,
        batch_index: int,
    ) -> np.ndarray:
        shape_type = Model4Mat.ImageMat.ShapeType(shape_type)

        if shape_type == Model4Mat.ImageMat.ShapeType.HW:
            return data
        if shape_type == Model4Mat.ImageMat.ShapeType.HWC:
            return data
        if shape_type == Model4Mat.ImageMat.ShapeType.BHW:
            return data[batch_index]
        if shape_type == Model4Mat.ImageMat.ShapeType.BHWC:
            return data[batch_index]
        if shape_type == Model4Mat.ImageMat.ShapeType.BCHW:
            frame = data[batch_index]
            if frame.shape[0] == 1:
                return frame[0]
            return np.transpose(frame, (1, 2, 0))

        raise ValueError(f"Unsupported image shape_type: {shape_type}")
