#!/usr/bin/env python3
"""Build an RGB-indexed Fast-FoundationStereo cloud and split it by YOLO masks.

The per-frame heavy path stays in Torch/CUDA:

1. stereo rectification with ``grid_sample``;
2. Fast-FoundationStereo inference;
3. disparity reprojection into the rectified-left frame;
4. transformation into the original-left and RGB-camera frames;
5. RGB lens projection and color sampling.

Only the final compact point, color, and RGB-pixel arrays are copied to CPU for
``split_cloud``. Unlike the older depth-image round trip, this implementation
directly projects points into the RGB image and therefore does not use
``splat_px`` or an RGB-space z-buffer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from pcd_dnn_utils import (
    FastFoundationStereoDisparity,
    TorchCudaStereoRectifier,
    TorchDnnColoredCloudBuilder,
)
from pcd_utils import StereoRectification, StereoRgbCalibration, read_image
from pcd_yolo_utils import split_cloud

ColorOrder = Literal["RGB", "BGR"]
ImageLayout = Literal["HWC", "CHW"]
ValueRange = Literal["auto", "0_1", "0_255"]


@dataclass
class RgbIndexedCloudDnnCuda:
    """Reusable CUDA implementation of RGB-indexed DNN cloud generation."""

    calibration: StereoRgbCalibration
    predictor: FastFoundationStereoDisparity
    min_disparity: float = 0.5
    max_depth_m: float | None = 5.0
    stride: int = 1
    alpha: float = 0.0
    model_scale: float = 1.0
    rgb_image_is_undistorted: bool = False
    stereo_input_color_order: ColorOrder = "BGR"
    align_corners: bool = False

    rectification: StereoRectification | None = field(init=False, default=None)
    rectifier: TorchCudaStereoRectifier | None = field(init=False, default=None)
    cloud_builder: TorchDnnColoredCloudBuilder | None = field(init=False, default=None)
    _setup_key: tuple[object, ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.predictor.device_type != "cuda":
            raise ValueError(
                "RgbIndexedCloudDnnCuda requires predictor.device on CUDA"
            )
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.model_scale <= 0:
            raise ValueError("model_scale must be positive")
        if self.stereo_input_color_order not in ("RGB", "BGR"):
            raise ValueError("stereo_input_color_order must be RGB or BGR")

    @property
    def device(self) -> torch.device:
        return self.predictor.device

    @staticmethod
    def _image_hw(
        image: np.ndarray | torch.Tensor,
        layout: ImageLayout | None,
    ) -> tuple[int, int]:
        shape = tuple(map(int, image.shape))
        if len(shape) == 2:
            return shape[0], shape[1]
        if len(shape) != 3:
            raise ValueError(f"Expected a 2D or 3D image, got shape {shape}")

        if layout == "CHW":
            return shape[1], shape[2]
        if layout == "HWC":
            return shape[0], shape[1]

        if shape[0] <= 4 and shape[2] > 4:
            return shape[1], shape[2]
        if shape[2] <= 4:
            return shape[0], shape[1]
        raise ValueError(f"Cannot infer image layout from shape {shape}")

    def _ensure_setup(
        self,
        *,
        stereo_hw: tuple[int, int],
        rgb_shape: tuple[int, ...],
        rgb_layout: ImageLayout,
        rgb_input_color_order: ColorOrder,
        rgb_value_range: ValueRange,
    ) -> None:
        setup_key = (
            stereo_hw,
            rgb_shape,
            rgb_layout,
            rgb_input_color_order,
            rgb_value_range,
        )
        if self._setup_key == setup_key:
            return

        stereo_h, stereo_w = stereo_hw
        self.rectification = self.calibration.get_rectifier(self.alpha).make(
            (stereo_w, stereo_h)
        )
        self.rectifier = TorchCudaStereoRectifier(
            self.rectification,
            device=self.device,
            align_corners=self.align_corners,
        )
        self.cloud_builder = TorchDnnColoredCloudBuilder(
            self.calibration,
            self.rectification,
            rgb_shape,
            device=self.device,
            rgb_layout=rgb_layout,
            min_disparity=max(0.5, float(self.min_disparity)),
            max_depth_m=self.max_depth_m,
            stride=self.stride,
            output_frame="left",
            input_color_order=rgb_input_color_order,
            rgb_image_is_undistorted=self.rgb_image_is_undistorted,
            rgb_value_range=rgb_value_range,
        )
        self._setup_key = setup_key

    def process(
        self,
        left_image: np.ndarray | torch.Tensor,
        right_image: np.ndarray | torch.Tensor,
        rgb_image: np.ndarray | torch.Tensor,
        *,
        stereo_layout: ImageLayout | None = None,
        stereo_value_range: ValueRange = "auto",
        rgb_layout: ImageLayout = "HWC",
        rgb_value_range: ValueRange = "auto",
        rgb_input_color_order: ColorOrder | None = None,
        remove_invisible: bool = True,
        download_disparity: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return original-left points, RGB colors, RGB pixels, and disparity."""
        
        if self.rectifier is None or self.cloud_builder is None:
            color_order = rgb_input_color_order or self.stereo_input_color_order
            left_hw = self._image_hw(left_image, stereo_layout)
            right_hw = self._image_hw(right_image, stereo_layout)
            if left_hw != right_hw:
                raise ValueError(f"dimensions differ: left={left_hw}, right={right_hw}")

            self._ensure_setup(
                stereo_hw=left_hw,
                rgb_shape=tuple(map(int, rgb_image.shape)),
                rgb_layout=rgb_layout,
                rgb_input_color_order=color_order,
                rgb_value_range=rgb_value_range,
            )

        left_rectified, right_rectified = self.rectifier.rectify(
            left_image,
            right_image,
            input_layout=stereo_layout,
            input_value_range=stereo_value_range,
        )
        disparity_cuda = self.predictor.predict_cuda(
            left_rectified,
            right_rectified,
            input_color_order="RGB",
            input_layout=None,
            input_value_range="0_255",
            model_scale=self.model_scale,
            remove_invisible=remove_invisible,
        )

        points_cuda, colors_cuda, pixel_xy_cuda, valid_count_cuda = (
            self.cloud_builder.build_cuda(
                disparity_cuda,
                rgb_image,
                return_pixel_xy=True,
            )
        )

        if int(valid_count_cuda.item()) == 0:
            raise RuntimeError("DNN disparity produced no valid 3D points")
        if points_cuda.shape[0] == 0:
            raise RuntimeError("No DNN points projected into the RGB image")

        def to_numpy(x,dt,cp):
            return x.detach().cpu().numpy().astype(dt, copy=cp)

        points_left = to_numpy(points_cuda, np.float32, False)
        colors_rgb = to_numpy(colors_cuda, np.uint8, False)
        pixel_xy = to_numpy(pixel_xy_cuda, np.int64, False)
        disparity = None
        if download_disparity:
            disparity = to_numpy(disparity_cuda, np.float32, False)
        return points_left, colors_rgb, pixel_xy, disparity


def build_rgb_indexed_cloud_dnn(
    left_image: np.ndarray | torch.Tensor,
    right_image: np.ndarray | torch.Tensor,
    rgb_image: np.ndarray | torch.Tensor,
    calibration: StereoRgbCalibration,
    predictor: FastFoundationStereoDisparity,
    *,
    processor: RgbIndexedCloudDnnCuda | None = None,
    min_disparity: float = 0.0,
    max_depth_m: float | None = 5.0,
    stride: int = 1,
    alpha: float = 0.0,
    splat_px: int = 0,
    rgb_image_is_undistorted: bool = False,
    stereo_input_color_order: ColorOrder = "BGR",
    model_scale: float = 1.0,
    download_disparity: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compatibility wrapper around :class:`RgbIndexedCloudDnnCuda`.

    ``splat_px`` is accepted to ease migration but is not used. The CUDA path
    directly projects each valid point into the RGB image instead of producing
    and re-reading a rasterized RGB depth image.
    """
    if splat_px != 0:
        raise ValueError(
            "The direct CUDA projection path does not implement splat_px; "
            "use splat_px=0"
        )

    processor = processor or RgbIndexedCloudDnnCuda(
        calibration=calibration,
        predictor=predictor,
        min_disparity=min_disparity,
        max_depth_m=max_depth_m,
        stride=stride,
        alpha=alpha,
        model_scale=model_scale,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
        stereo_input_color_order=stereo_input_color_order,
    )
    return processor.process(
        left_image,
        right_image,
        rgb_image,
        stereo_layout=None,
        stereo_value_range="0_255",
        rgb_layout="HWC",
        rgb_value_range="0_255",
        rgb_input_color_order=stereo_input_color_order,
        download_disparity=download_disparity,
    )


if __name__ == "__main__":
    root = "recording/rgb_stereo/2026-07-22/field_all/111737.603022000JST/"
    left = read_image(root+"imgs/rgbd_left/left.jpg", color=False)
    right = read_image(root+"imgs/rgbd_left/right.jpg", color=False)
    rgb = read_image(root+"imgs/rgbd_left/rgb.jpg", color=True)  # cv2 gives BGR
    with open(root+"calib/rgbd_left.json") as f:
        calibration = StereoRgbCalibration.from_dict(json.load(f))

    detections = json.loads(Path(root+"yolo/rgbd_left/rgb.json").read_text(encoding="utf-8"))
    from yolo_utils import show_yolo_results_pil
    show_yolo_results_pil(detections)

    predictor = FastFoundationStereoDisparity(
        repo_dir="./examples/14_iox2_dai_yolo_web_mjpeg/fast-foundationstereo",
        model_path="weights/23-36-37/model_best_bp2_serialize.pth",
        device="cuda:0",
        valid_iters=8,
        max_disp=192,
    )

    processor = RgbIndexedCloudDnnCuda(
        calibration=calibration,
        predictor=predictor,
        min_disparity=0.5,
        max_depth_m=5.0,
        stride=1,
        alpha=0.0,
        model_scale=1.0,
        rgb_image_is_undistorted=False,
        stereo_input_color_order="BGR",
    )
    points_left, colors_rgb, pixel_xy, _disparity = processor.process(
        left,
        right,
        rgb,
        stereo_value_range="0_255",
        rgb_layout="HWC",
        rgb_value_range="0_255",
        rgb_input_color_order="BGR",
        download_disparity=True,
    )

    output_dir = Path(root+"tmp/")
    manifest = split_cloud(
        points_left,
        colors_rgb,
        pixel_xy,
        detections,
        output_dir,
        min_points=1,
        erode_pixels=0,
        exclusive=False,
        save_background=True,
        save_full_cloud=True,
        binary_pcd=False,
    )
