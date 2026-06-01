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
           res = res.get_data()
        else:
            img:Model4Mat.ImageMatPubSub = self.generator
            res = img.sub().get_data()
        return res
    
    def show_once(self) -> int:
        frame = self.read()
        try:
            cv2.imshow(self.title, frame)
        except cv2.error:
            pass
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





