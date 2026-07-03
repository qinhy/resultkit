import json
import multiprocessing
import os
import sys
import glob
import time
import uuid
import platform
from enum import IntEnum
from typing import Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..MatModel import CodecFormat, ColorFormat, ImageShapeType, Model4Mat
ImageMat = Model4Mat.ImageMat
from ..logger import logger

class ImageMatGenerator(BaseModel):
    sources: List[str]
    color_types: List['ColorFormat']
    uuid: str = ''
    shmIO_mode: Literal[False,'writer','reader'] = False
    fps:int = -1
    _min_frame_time:float = 0.0

    _resources: list = []
    _frame_generators: list = []
    ouput_mats:List[ImageMat] = []

    def model_post_init(self, context):
        self._min_frame_time = 1.0 / self.fps if self.fps != 0 else 0
        if not self.uuid:
            self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'
        if len(self.sources)==0:raise ValueError('empty sources.')
        self._frame_generators = [self.create_frame_generator(i,src) for i,src in enumerate(self.sources)]

        if len(self._frame_generators)==0:raise ValueError('empty frame_generators.')
        if len(self.color_types)==0:raise ValueError('empty color_types.')
        if len(self.ouput_mats)==0:
            self.ouput_mats = [ImageMat(color_type=color_type)
                        for gen,color_type in zip(self._frame_generators, self.color_types)]            
        return super().model_post_init(context)        

    def register_resource(self, resource):
        self._resources.append(resource)
        return resource

    @staticmethod
    def has_func(obj, name):
        return callable(getattr(obj, name, None))

    def release_resources(self):
        cleanup_methods = [
            "exit", "end", "teardown",
            "stop", "shutdown", "terminate",
            "join", "cleanup", "deactivate",
            "release", "close", "disconnect",
            "destroy",
        ]

        for res in self._resources:
            for method in cleanup_methods:
                if self.has_func(res, method):
                    try:
                        getattr(res, method)()
                    except Exception as e:
                        logger(f"Error during {method} on {res}: {e}")

        self._resources.clear()

    def create_frame_generator(self, idx,source):
        raise NotImplementedError("Subclasses must implement `create_frame_generator`")

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        try:
            frames = [next(frame_gen) for frame_gen in self._frame_generators]
            if not frames or any(f is None for f in frames):
                raise StopIteration
            for frame, mat in zip(frames, self.ouput_mats):
                mat.unsafe_update_data(frame)

            if self.fps:
                # Enforce FPS limit
                elapsed = time.time() - start_time
                sleep_time = self._min_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            return self.ouput_mats
        except StopIteration:
            raise StopIteration
    
    def reset_generators(self):
        self.release_resources()
        self._frame_generators = [self.create_frame_generator(i,src) for i,src in enumerate(self.sources)]

    def release(self):
        self.release_resources()

    def __del__(self):
        self.release()

    def __len__(self):
        return None

class InvalidSourceFrameGenerator(ImageMatGenerator):
    # Same style as your CvVideoFrameGenerator
    color_types: List['ColorFormat'] = []
    frame_size: Tuple[int, int] = (480, 640)
    fallback_text: str = "NOT VALID"

    def _make_placeholder_frame(self, source) -> np.ndarray:
        h, w = self.frame_size

        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Main message
        cv2.putText(
            frame,
            self.fallback_text,
            (20, int(h * 0.4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Show (part of) the source string under it
        src_txt = str(source)
        if len(src_txt) > 40:
            src_txt = src_txt[:37] + "..."

        cv2.putText(
            frame,
            src_txt,
            (20, int(h * 0.4) + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return np.ascontiguousarray(frame)

    def create_frame_generator(self, idx, source):
        # Match your color_types handling style, but per index
        if idx >= len(self.color_types):
            self.color_types.append(ColorFormat.BGR)
        else:
            self.color_types[idx] = ColorFormat.BGR

        frame = self._make_placeholder_frame(source)

        def gen(frame=frame):
            # Infinite stream of the same "NOT VALID" frame
            while True:
                yield frame

        return gen()
    
class CvVideoFrameGenerator(ImageMatGenerator):    
    color_types: List['ColorFormat'] = []
    
    def create_frame_generator(self, idx,source):
        if idx>=len(self.color_types):
            self.color_types.append(ColorFormat.BGR)
        else:
            self.color_types[0] = ColorFormat.BGR
        cap = self.register_resource(cv2.VideoCapture(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        def gen(cap=cap):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                yield np.ascontiguousarray(frame)
        return gen()
    
class NumpyRawFrameFileGenerator(ImageMatGenerator):
    color_types: List['ColorFormat']
    def create_frame_generator(self, idx,source):
        arr = np.load(source)
        def gen(arr=arr):
            cnt=-1
            while True:
                # idx = np.random.choice(len(arr))
                cnt += 1
                yield np.ascontiguousarray(arr[cnt%len(arr)])
        return gen()    

