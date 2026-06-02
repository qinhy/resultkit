# tests/test_model4mat.py
import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resultkit.MatModel import ColorFormat, Model4Mat, MatLib, DataType, MatStore, ImageShapeType


def add_to_store(model):
    store = MatStore()
    return store.add_new_obj(model)


# -------------------------
# ImageMat enum helpers
# -------------------------

def test_color_format_channels():
    C = ColorFormat

    assert C.channels(C.RGB) == 3
    assert C.channels("RGB") == 3
    assert C.channels(C.BGR) == 3
    assert C.channels(C.GRAY) == 1
    assert C.channels(C.BAYER) == 1

    with pytest.raises(ValueError):
        C.channels(C.UNKNOWN)


def test_shape_type_to_bchw():
    S = ImageShapeType

    assert S.to_bchw(S.HW, np.zeros((10, 20))) == (1, 1, 10, 20)
    assert S.to_bchw(S.BHW, np.zeros((4, 10, 20))) == (4, 1, 10, 20)
    assert S.to_bchw(S.HWC, np.zeros((10, 20, 3))) == (1, 3, 10, 20)
    assert S.to_bchw(S.BHWC, np.zeros((4, 10, 20, 3))) == (4, 3, 10, 20)
    assert S.to_bchw(S.BCHW, torch.zeros((4, 3, 10, 20))) == (4, 3, 10, 20)


def test_shape_type_to_bchw_rejects_bad_dims():
    S = ImageShapeType

    with pytest.raises(ValueError, match="expects"):
        S.to_bchw(S.HWC, np.zeros((10, 20)))

    with pytest.raises(ValueError, match="Cannot convert unknown"):
        S.to_bchw(S.UNKNOWN, np.zeros((10, 20)))


# -------------------------
# NumPy ImageMat validation
# -------------------------

def test_numpy_gray_hw_image_is_valid():
    img = Model4Mat.ImageMat(
        color_format=ColorFormat.GRAY,
        data=np.zeros((10, 20), dtype=np.uint8),
    )

    assert img.lib == MatLib.NUMPY
    assert img.dtype == DataType.UINT8
    assert img.shape_type == ImageShapeType.HW
    assert img.BCHW == (1, 1, 10, 20)
    assert img.size() == (20, 10)


def test_numpy_rgb_hwc_image_is_valid():
    img = Model4Mat.ImageMat(
        color_format=ColorFormat.RGB,
        data=np.zeros((10, 20, 3), dtype=np.uint8),
    )

    assert img.lib == MatLib.NUMPY
    assert img.shape_type == ImageShapeType.HWC
    assert img.BCHW == (1, 3, 10, 20)
    assert img.size() == (20, 10)


def test_numpy_gray_bhw_batch_is_valid():
    img = Model4Mat.ImageMat(
        color_format=ColorFormat.GRAY,
        data=np.zeros((4, 10, 20), dtype=np.uint8),
    )

    assert img.shape_type == ImageShapeType.BHW
    assert img.BCHW == (4, 1, 10, 20)


def test_numpy_unknown_3d_layout_rejected():
    with pytest.raises(ValueError, match="Cannot infer 3-D NumPy image layout"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.UNKNOWN,
            data=np.zeros((10, 20, 3), dtype=np.uint8),
        )


def test_numpy_channel_mismatch_rejected():
    with pytest.raises(TypeError, match="Expected 3 channels"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.RGB,
            shape_type=ImageShapeType.HWC,
            data=np.zeros((10, 20, 1), dtype=np.uint8),
        )


def test_numpy_non_uint8_rejected():
    with pytest.raises(TypeError, match="Expected uint8 dtype"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            data=np.zeros((10, 20), dtype=np.float32),
        )


def test_numpy_bchw_layout_rejected():
    with pytest.raises(TypeError, match="Expected HW, BHW, HWC, or BHWC"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            shape_type=ImageShapeType.BCHW,
            data=np.zeros((1, 1, 10, 20), dtype=np.uint8),
        )


# -------------------------
# Torch ImageMat validation
# -------------------------

def test_torch_bchw_float_image_is_valid():
    img = Model4Mat.ImageMat(
        color_format=ColorFormat.RGB,
        data=torch.zeros((2, 3, 10, 20), dtype=torch.float32),
    )

    assert img.lib == MatLib.TORCH
    assert img.dtype == DataType.FLOAT32
    assert img.shape_type == ImageShapeType.BCHW
    assert img.BCHW == (2, 3, 10, 20)
    assert img.size() == (20, 10)


def test_torch_uint8_rejected():
    with pytest.raises(TypeError, match="Expected float dtype"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            data=torch.zeros((1, 1, 10, 20), dtype=torch.uint8),
        )


def test_torch_non_bchw_rejected():
    with pytest.raises(TypeError, match="Expected BCHW shape"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            data=torch.zeros((10, 20), dtype=torch.float32),
        )


def test_torch_nan_rejected():
    data = torch.zeros((1, 1, 10, 20), dtype=torch.float32)
    data[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            data=data,
        )


def test_torch_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            data=torch.full((1, 1, 10, 20), 2.0, dtype=torch.float32),
        )


# -------------------------
# ImageMat conversions
# -------------------------

def test_numpy_gray_to_torch():
    img = add_to_store(
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            data=np.full((2, 3), 255, dtype=np.uint8),
        )
    )

    out = img.to_torch()

    assert out.lib == MatLib.TORCH
    assert out.shape_type == ImageShapeType.BCHW
    assert out.BCHW == (1, 1, 2, 3)
    assert out.data.shape == torch.Size((1, 1, 2, 3))
    assert out.data.dtype == torch.float32
    assert torch.allclose(out.data, torch.ones((1, 1, 2, 3)))


def test_numpy_rgb_hwc_to_torch():
    img = add_to_store(
        Model4Mat.ImageMat(
            color_format=ColorFormat.RGB,
            data=np.full((2, 3, 3), 255, dtype=np.uint8),
        )
    )

    out = img.to_torch()

    assert out.lib == MatLib.TORCH
    assert out.shape_type == ImageShapeType.BCHW
    assert out.BCHW == (1, 3, 2, 3)
    assert out.data.shape == torch.Size((1, 3, 2, 3))
    assert torch.allclose(out.data, torch.ones((1, 3, 2, 3)))


def test_torch_gray_to_numpy():
    img = add_to_store(
        Model4Mat.ImageMat(
            color_format=ColorFormat.GRAY,
            data=torch.ones((1, 1, 2, 3), dtype=torch.float32),
        )
    )

    out = img.to_numpy()

    assert out.lib == MatLib.NUMPY
    assert out.shape_type == ImageShapeType.BHW
    assert out.BCHW == (1, 1, 2, 3)
    assert out.data.shape == (1, 2, 3)
    assert out.data.dtype == np.uint8
    np.testing.assert_array_equal(out.data, np.full((1, 2, 3), 255, dtype=np.uint8))


def test_torch_rgb_to_numpy():
    img = add_to_store(
        Model4Mat.ImageMat(
            color_format=ColorFormat.RGB,
            data=torch.ones((1, 3, 2, 4), dtype=torch.float32),
        )
    )

    out = img.to_numpy()

    assert out.lib == MatLib.NUMPY
    assert out.shape_type == ImageShapeType.BHWC
    assert out.BCHW == (1, 3, 2, 4)
    assert out.data.shape == (1, 2, 4, 3)
    assert out.data.dtype == np.uint8
    np.testing.assert_array_equal(out.data, np.full((1, 2, 4, 3), 255, dtype=np.uint8))


# -------------------------
# BoundingBox validation
# -------------------------

def test_bounding_box_zero_one_is_valid():
    box = Model4Mat.BoundingBox(
        data=np.array([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32),
        scale=Model4Mat.BoundingBox.ScaleFormat.ZERO_ONE,
        format=Model4Mat.BoundingBox.AxisFormat.XYXY,
    )

    assert box.lib == MatLib.NUMPY
    assert box.shape() == (1, 4)


def test_bounding_box_rejects_wrong_rank():
    with pytest.raises(ValueError):
        Model4Mat.BoundingBox(
            data=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        )


def test_bounding_box_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        Model4Mat.BoundingBox(
            data=np.array([[0.1, 0.2, 0.8]], dtype=np.float32),
        )


def test_bounding_box_zero_one_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Model4Mat.BoundingBox(
            data=np.array([[-0.1, 0.2, 1.2, 0.9]], dtype=np.float32),
            scale=Model4Mat.BoundingBox.ScaleFormat.ZERO_ONE,
        )


def test_bounding_box_raw_rejects_normalized_values():
    with pytest.raises(ValueError, match="raw pixels"):
        Model4Mat.BoundingBox(
            data=np.array([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32),
            scale=Model4Mat.BoundingBox.ScaleFormat.RAW,
        )


# -------------------------
# Known current-code issues
# -------------------------

# @pytest.mark.xfail(
#     reason=(
#         "Current BoundingBox.to_abcd builds method names with Enum objects. "
#         "It should use self.format.value and type.value."
#     )
# )
def test_bounding_box_xyxy_to_xywh_conversion():
    box = add_to_store(
        Model4Mat.BoundingBox(
            data=np.array([[10.0, 20.0, 30.0, 50.0]], dtype=np.float32),
            scale=Model4Mat.BoundingBox.ScaleFormat.RAW,
            format=Model4Mat.BoundingBox.AxisFormat.XYXY,
            image_size=(100, 100),
        )
    )

    out:Model4Mat.BoundingBox = box.to_xywh()

    assert out.format == Model4Mat.BoundingBox.AxisFormat.XYWH
    np.testing.assert_allclose(
        out.data,
        np.array([[10.0, 20.0, 20.0, 30.0]], dtype=np.float32),
    )


# @pytest.mark.xfail(
#     reason=(
#         "Current BoundingBox.to_scale calls self.to_xyxy() without the required "
#         "data argument."
#     )
# )
def test_bounding_box_to_scale_raw_to_zero_one():
    box = add_to_store(
        Model4Mat.BoundingBox(
            data=np.array([[10.0, 20.0, 30.0, 50.0]], dtype=np.float32),
            scale=Model4Mat.BoundingBox.ScaleFormat.RAW,
            format=Model4Mat.BoundingBox.AxisFormat.XYXY,
            image_size=(100, 100),
        )
    )

    out:Model4Mat.BoundingBox = box.to_scale(Model4Mat.BoundingBox.ScaleFormat.ZERO_ONE)

    assert out.scale == Model4Mat.BoundingBox.ScaleFormat.ZERO_ONE
    np.testing.assert_allclose(
        out.data,
        np.array([[0.1, 0.2, 0.3, 0.5]], dtype=np.float32),
    )