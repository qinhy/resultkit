# pcd_backend_cpu.py

from pcd_utils import (
    ColoredPointCloud,
    SGBMDisparityPredictor,
    StereoRgbCalibration,
    points_left_to_rgb_depth,
    read_image,
    rectified_left_to_original_left,
    rgb8,
    rgb_depth_to_points_rgb,
    save_point_cloud,
    transform_points,
)

from pcd_yolo_utils import split_cloud


def image_gpu(img):
    return img