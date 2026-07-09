"""ROS 2 message helpers for YOLO-seg 3D point-cloud segments.

This module is intentionally ROS-only at the boundary:
- It does NOT run YOLO.
- It does NOT compute stereo disparity/depth.
- It takes the ROS-independent result returned by a function like
  `yolo_result_to_pcd_segments(...)` and converts it to ROS 2 messages.

Recommended ROS topics:
    /perception/segments/cloud        sensor_msgs/msg/PointCloud2
    /perception/segments/instance_map sensor_msgs/msg/Image, encoding 32UC1
    /perception/segments/depth_rgb    sensor_msgs/msg/Image, encoding 32FC1, optional
    /perception/segments/info_json    std_msgs/msg/String, JSON fallback metadata
    /perception/segments/markers      visualization_msgs/msg/MarkerArray, optional RViz debug

PointCloud2 fields produced by `segmented_result_to_pointcloud2`:
    x, y, z         float32, meters
    rgb             float32, PCL/RViz packed RGB
    instance_id     uint32, 0 is background; segment points start at 1
    class_id        int32, YOLO class id
    confidence      float32, YOLO confidence
    u, v            uint32, original RGB image pixel coordinates

The cloud is sparse/unordered by design:
    cloud.height = 1
    cloud.width  = number_of_valid_segment_points

The `u` and `v` fields preserve the RGB pixel mapping, while the separate
`instance_map` preserves the dense RGB-sized segmentation layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import colorsys
import json
from typing import Any

import numpy as np

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class SegmentRosMessages:
    """Convenience container for the main ROS messages for one frame."""

    cloud: PointCloud2
    instance_map: Image
    info_json: String
    depth_rgb: Image | None = None
    markers: MarkerArray | None = None


def make_header(frame_id: str, stamp: Any | None = None) -> Header:
    """Create a std_msgs/Header.

    Args:
        frame_id: TF frame for the message, e.g. "camera_rgb_optical_frame".
        stamp: Optional ROS stamp. Accepts:
            - None: leaves stamp at zero
            - builtin_interfaces.msg.Time
            - rclpy.time.Time, using .to_msg()
            - any object already shaped like a ROS stamp with sec/nanosec
    """
    header = Header()
    header.frame_id = str(frame_id)

    if stamp is None:
        return header

    if hasattr(stamp, "to_msg"):
        header.stamp = stamp.to_msg()
    elif isinstance(stamp, Time):
        header.stamp = stamp
    elif hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        header.stamp.sec = int(stamp.sec)
        header.stamp.nanosec = int(stamp.nanosec)
    else:
        raise TypeError(
            "stamp must be None, builtin_interfaces.msg.Time, "
            "an rclpy Time with .to_msg(), or an object with sec/nanosec"
        )

    return header


def _as_rgb8(colors: Any) -> np.ndarray:
    """Return colors as Nx3 uint8 RGB."""
    c = np.asarray(colors)
    if c.ndim != 2 or c.shape[1] < 3:
        raise ValueError(f"colors must be Nx3 or Nx4, got {c.shape}")
    c = c[:, :3]
    if np.issubdtype(c.dtype, np.floating) and c.size and np.nanmax(c) <= 1.0:
        c = c * 255.0
    return np.clip(np.rint(c), 0, 255).astype(np.uint8, copy=False)


def rgb8_to_pcl_float32(colors_rgb: Any) -> np.ndarray:
    """Pack RGB uint8 colors into the common PCL/RViz float32 rgb field."""
    c = _as_rgb8(colors_rgb)
    packed = (
        (c[:, 0].astype(np.uint32) << 16)
        | (c[:, 1].astype(np.uint32) << 8)
        | c[:, 2].astype(np.uint32)
    )
    return packed.astype("<u4", copy=False).view("<f4")


def rgb8_to_uint32(colors_rgb: Any) -> np.ndarray:
    """Pack RGB uint8 colors into uint32 0xRRGGBB."""
    c = _as_rgb8(colors_rgb)
    return (
        (c[:, 0].astype(np.uint32) << 16)
        | (c[:, 1].astype(np.uint32) << 8)
        | c[:, 2].astype(np.uint32)
    )


def _require_1d(name: str, value: Any, n: int, dtype: Any) -> np.ndarray:
    a = np.asarray(value, dtype=dtype).reshape(-1)
    if len(a) != n:
        raise ValueError(f"{name} must have length {n}, got {len(a)}")
    return a


def _require_points(points: Any) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {p.shape}")
    return np.ascontiguousarray(p)


def _require_pixels(pixels_rgb: Any, n: int) -> np.ndarray:
    uv = np.asarray(pixels_rgb, dtype=np.uint32)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) != n:
        raise ValueError(f"pixels_rgb must be Nx2 with length {n}, got {uv.shape}")
    return np.ascontiguousarray(uv)


def segmented_arrays_to_pointcloud2(
    *,
    points_m: Any,
    colors_rgb: Any,
    instance_ids: Any,
    class_ids: Any,
    confidences: Any,
    pixels_rgb: Any,
    frame_id: str,
    stamp: Any | None = None,
) -> PointCloud2:
    """Build one sparse segmented PointCloud2 from arrays.

    Args:
        points_m: Nx3 xyz points in meters.
        colors_rgb: Nx3 RGB uint8 colors, or float colors in 0..1.
        instance_ids: N instance IDs. Usually 1..K; 0 is background.
        class_ids: N YOLO class IDs.
        confidences: N YOLO confidences.
        pixels_rgb: Nx2 RGB image coordinates [u, v].
        frame_id: TF frame, usually "camera_rgb_optical_frame".
        stamp: Optional ROS timestamp, commonly self.get_clock().now().
    """
    points = _require_points(points_m)
    n = len(points)
    colors = _as_rgb8(colors_rgb)
    if len(colors) != n:
        raise ValueError(f"colors_rgb must have length {n}, got {len(colors)}")

    instance_ids = _require_1d("instance_ids", instance_ids, n, np.uint32)
    class_ids = _require_1d("class_ids", class_ids, n, np.int32)
    confidences = _require_1d("confidences", confidences, n, np.float32)
    pixels = _require_pixels(pixels_rgb, n)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("rgb", "<f4"),
            ("instance_id", "<u4"),
            ("class_id", "<i4"),
            ("confidence", "<f4"),
            ("u", "<u4"),
            ("v", "<u4"),
        ]
    )

    cloud_np = np.empty(n, dtype=dtype)
    cloud_np["x"] = points[:, 0]
    cloud_np["y"] = points[:, 1]
    cloud_np["z"] = points[:, 2]
    cloud_np["rgb"] = rgb8_to_pcl_float32(colors)
    cloud_np["instance_id"] = instance_ids
    cloud_np["class_id"] = class_ids
    cloud_np["confidence"] = confidences
    cloud_np["u"] = pixels[:, 0]
    cloud_np["v"] = pixels[:, 1]

    fields = [
        PointField(name="x", offset=dtype.fields["x"][1], datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=dtype.fields["y"][1], datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=dtype.fields["z"][1], datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=dtype.fields["rgb"][1], datatype=PointField.FLOAT32, count=1),
        PointField(name="instance_id", offset=dtype.fields["instance_id"][1], datatype=PointField.UINT32, count=1),
        PointField(name="class_id", offset=dtype.fields["class_id"][1], datatype=PointField.INT32, count=1),
        PointField(name="confidence", offset=dtype.fields["confidence"][1], datatype=PointField.FLOAT32, count=1),
        PointField(name="u", offset=dtype.fields["u"][1], datatype=PointField.UINT32, count=1),
        PointField(name="v", offset=dtype.fields["v"][1], datatype=PointField.UINT32, count=1),
    ]

    msg = PointCloud2()
    msg.header = make_header(frame_id, stamp)
    msg.height = 1
    msg.width = int(n)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = int(dtype.itemsize)
    msg.row_step = int(dtype.itemsize * n)
    msg.data = cloud_np.tobytes(order="C")
    msg.is_dense = bool(np.isfinite(points).all())
    return msg


def segmented_result_to_pointcloud2(
    result: Any,
    *,
    frame_id: str = "camera_rgb_optical_frame",
    stamp: Any | None = None,
) -> PointCloud2:
    """Build a PointCloud2 from a YoloSegments3DResult-like object."""
    return segmented_arrays_to_pointcloud2(
        points_m=result.points_m,
        colors_rgb=result.colors_rgb,
        instance_ids=result.instance_ids,
        class_ids=result.class_ids,
        confidences=result.confidences,
        pixels_rgb=result.pixels_rgb,
        frame_id=frame_id,
        stamp=stamp,
    )


def array_to_image_msg(
    array: Any,
    *,
    frame_id: str,
    stamp: Any | None = None,
    encoding: str | None = None,
) -> Image:
    """Convert a contiguous numpy array into a sensor_msgs/Image.

    Supported automatic encodings:
        uint8  HxW      -> mono8
        uint16 HxW      -> 16UC1
        uint32 HxW      -> 32UC1
        int32  HxW      -> 32SC1
        float32 HxW     -> 32FC1
        uint8  HxWx3    -> rgb8
        uint8  HxWx4    -> rgba8
    """
    a = np.asarray(array)
    if a.ndim not in (2, 3):
        raise ValueError(f"Image array must be HxW or HxWxC, got {a.shape}")

    if encoding is None:
        if a.ndim == 2 and a.dtype == np.uint8:
            encoding = "mono8"
        elif a.ndim == 2 and a.dtype == np.uint16:
            encoding = "16UC1"
        elif a.ndim == 2 and a.dtype == np.uint32:
            encoding = "32UC1"
        elif a.ndim == 2 and a.dtype == np.int32:
            encoding = "32SC1"
        elif a.ndim == 2 and a.dtype == np.float32:
            encoding = "32FC1"
        elif a.ndim == 3 and a.dtype == np.uint8 and a.shape[2] == 3:
            encoding = "rgb8"
        elif a.ndim == 3 and a.dtype == np.uint8 and a.shape[2] == 4:
            encoding = "rgba8"
        else:
            raise ValueError(f"Cannot infer ROS image encoding for shape={a.shape}, dtype={a.dtype}")

    a = np.ascontiguousarray(a)
    msg = Image()
    msg.header = make_header(frame_id, stamp)
    msg.height = int(a.shape[0])
    msg.width = int(a.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False
    channels = int(a.shape[2]) if a.ndim == 3 else 1
    msg.step = int(a.shape[1] * channels * a.dtype.itemsize)
    msg.data = a.tobytes(order="C")
    return msg


def instance_map_to_image_msg(
    instance_map: Any,
    *,
    frame_id: str = "camera_rgb_optical_frame",
    stamp: Any | None = None,
) -> Image:
    """Publish RGB-sized instance map as 32UC1 image.

    Pixel value convention:
        0 = background
        1..K = YOLO instance IDs
    """
    m = np.asarray(instance_map, dtype=np.uint32)
    if m.ndim != 2:
        raise ValueError(f"instance_map must be HxW, got {m.shape}")
    return array_to_image_msg(m, frame_id=frame_id, stamp=stamp, encoding="32UC1")


def depth_rgb_to_image_msg(
    depth_rgb_m: Any,
    *,
    frame_id: str = "camera_rgb_optical_frame",
    stamp: Any | None = None,
) -> Image:
    """Publish RGB-aligned depth in meters as 32FC1 image."""
    d = np.asarray(depth_rgb_m, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth_rgb_m must be HxW, got {d.shape}")
    return array_to_image_msg(d, frame_id=frame_id, stamp=stamp, encoding="32FC1")


def _numpy_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def segment_to_dict(segment: Any) -> dict[str, Any]:
    """Convert one YoloSegment3D-like object into JSON-safe metadata."""
    points = np.asarray(getattr(segment, "points_m", np.empty((0, 3))), dtype=float)
    return {
        "instance_id": int(getattr(segment, "instance_id")),
        "class_id": int(getattr(segment, "class_id")),
        "class_name": str(getattr(segment, "class_name")),
        "confidence": float(getattr(segment, "confidence")),
        "bbox_xyxy_rgb": [float(v) for v in getattr(segment, "bbox_xyxy_rgb")],
        "mask_area_px": int(getattr(segment, "mask_area_px", 0)),
        "point_count": int(len(points)),
        "centroid_m": _numpy_list(getattr(segment, "centroid_m", np.full(3, np.nan))),
        "aabb_min_m": _numpy_list(getattr(segment, "aabb_min_m", np.full(3, np.nan))),
        "aabb_max_m": _numpy_list(getattr(segment, "aabb_max_m", np.full(3, np.nan))),
        "output_frame": str(getattr(segment, "output_frame", "unknown")),
        "pcd_path": getattr(segment, "pcd_path", None),
        "pixels_path": getattr(segment, "pixels_path", None),
        "meta_path": getattr(segment, "meta_path", None),
    }


def segments_result_to_json_dict(
    result: Any,
    *,
    frame_id: str = "camera_rgb_optical_frame",
) -> dict[str, Any]:
    """Convert a YoloSegments3DResult-like object into JSON-safe metadata."""
    points = np.asarray(getattr(result, "points_m", np.empty((0, 3))), dtype=float)
    segments = [segment_to_dict(s) for s in getattr(result, "segments", [])]
    return {
        "frame_id": frame_id,
        "output_frame": str(getattr(result, "output_frame", "unknown")),
        "point_count": int(len(points)),
        "segment_count": int(len(segments)),
        "segments": segments,
    }


def segments_result_to_json_msg(
    result: Any,
    *,
    frame_id: str = "camera_rgb_optical_frame",
    stamp: Any | None = None,
    pretty: bool = False,
) -> String:
    """Create std_msgs/String JSON metadata.

    This is useful before you create custom messages such as SegmentInfoArray.
    The stamp is not embedded as a ROS Header because std_msgs/String has no header.
    If you need synchronized metadata, prefer a custom message with a Header.
    """
    d = segments_result_to_json_dict(result, frame_id=frame_id)
    if stamp is not None:
        h = make_header(frame_id, stamp)
        d["stamp"] = {"sec": int(h.stamp.sec), "nanosec": int(h.stamp.nanosec)}
    msg = String()
    msg.data = json.dumps(d, indent=2 if pretty else None)
    return msg


def _color_for_instance(instance_id: int) -> tuple[float, float, float]:
    """Deterministic bright-ish color for RViz markers."""
    hue = (int(instance_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return float(r), float(g), float(b)


def _marker_common(
    *,
    frame_id: str,
    stamp: Any | None,
    ns: str,
    marker_id: int,
    marker_type: int,
) -> Marker:
    m = Marker()
    m.header = make_header(frame_id, stamp)
    m.ns = ns
    m.id = int(marker_id)
    m.type = int(marker_type)
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def segments_result_to_marker_array(
    result: Any,
    *,
    frame_id: str = "camera_rgb_optical_frame",
    stamp: Any | None = None,
    ns: str = "yolo_segments_3d",
    include_delete_all: bool = True,
    cube_alpha: float = 0.22,
    text_alpha: float = 1.0,
    text_scale: float = 0.06,
    min_box_size_m: float = 0.01,
) -> MarkerArray:
    """Create RViz markers: one 3D AABB cube and one text label per segment."""
    arr = MarkerArray()

    if include_delete_all:
        clear = Marker()
        clear.header = make_header(frame_id, stamp)
        clear.ns = ns
        clear.id = 0
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

    for segment in getattr(result, "segments", []):
        instance_id = int(getattr(segment, "instance_id"))
        class_name = str(getattr(segment, "class_name"))
        conf = float(getattr(segment, "confidence"))
        point_count = int(len(getattr(segment, "points_m", [])))

        aabb_min = np.asarray(getattr(segment, "aabb_min_m"), dtype=float).reshape(3)
        aabb_max = np.asarray(getattr(segment, "aabb_max_m"), dtype=float).reshape(3)
        centroid = np.asarray(getattr(segment, "centroid_m"), dtype=float).reshape(3)

        if not (np.isfinite(aabb_min).all() and np.isfinite(aabb_max).all() and np.isfinite(centroid).all()):
            continue

        center = 0.5 * (aabb_min + aabb_max)
        size = np.maximum(aabb_max - aabb_min, float(min_box_size_m))
        r, g, b = _color_for_instance(instance_id)

        cube = _marker_common(
            frame_id=frame_id,
            stamp=stamp,
            ns=f"{ns}/boxes",
            marker_id=instance_id,
            marker_type=Marker.CUBE,
        )
        cube.pose.position.x = float(center[0])
        cube.pose.position.y = float(center[1])
        cube.pose.position.z = float(center[2])
        cube.scale.x = float(size[0])
        cube.scale.y = float(size[1])
        cube.scale.z = float(size[2])
        cube.color.r = r
        cube.color.g = g
        cube.color.b = b
        cube.color.a = float(cube_alpha)
        arr.markers.append(cube)

        text = _marker_common(
            frame_id=frame_id,
            stamp=stamp,
            ns=f"{ns}/labels",
            marker_id=instance_id,
            marker_type=Marker.TEXT_VIEW_FACING,
        )
        text.pose.position.x = float(centroid[0])
        text.pose.position.y = float(centroid[1])
        text.pose.position.z = float(aabb_max[2] + max(float(size[2]) * 0.15, float(text_scale)))
        text.scale.z = float(text_scale)
        text.color.r = r
        text.color.g = g
        text.color.b = b
        text.color.a = float(text_alpha)
        text.text = f"{instance_id}: {class_name} {conf:.2f} ({point_count} pts)"
        arr.markers.append(text)

    return arr


def build_segment_ros_messages(
    result: Any,
    *,
    frame_id: str = "camera_rgb_optical_frame",
    stamp: Any | None = None,
    include_depth: bool = False,
    include_markers: bool = True,
    pretty_json: bool = False,
) -> SegmentRosMessages:
    """Build the standard ROS message bundle for a segmented 3D frame."""
    cloud = segmented_result_to_pointcloud2(result, frame_id=frame_id, stamp=stamp)
    instance_map = instance_map_to_image_msg(result.instance_map, frame_id=frame_id, stamp=stamp)
    info_json = segments_result_to_json_msg(
        result,
        frame_id=frame_id,
        stamp=stamp,
        pretty=pretty_json,
    )

    depth_msg = None
    if include_depth and getattr(result, "depth_rgb_m", None) is not None:
        depth_msg = depth_rgb_to_image_msg(result.depth_rgb_m, frame_id=frame_id, stamp=stamp)

    markers = None
    if include_markers:
        markers = segments_result_to_marker_array(result, frame_id=frame_id, stamp=stamp)

    return SegmentRosMessages(
        cloud=cloud,
        instance_map=instance_map,
        info_json=info_json,
        depth_rgb=depth_msg,
        markers=markers,
    )


__all__ = [
    "SegmentRosMessages",
    "make_header",
    "rgb8_to_pcl_float32",
    "rgb8_to_uint32",
    "segmented_arrays_to_pointcloud2",
    "segmented_result_to_pointcloud2",
    "array_to_image_msg",
    "instance_map_to_image_msg",
    "depth_rgb_to_image_msg",
    "segment_to_dict",
    "segments_result_to_json_dict",
    "segments_result_to_json_msg",
    "segments_result_to_marker_array",
    "build_segment_ros_messages",
]


if __name__ == "__main__":
    # It supports:
    # PointCloud2:
    #   x, y, z
    #   rgb
    #   instance_id
    #   class_id
    #   confidence
    #   u, v

    # Image:
    #   instance_map as 32UC1
    #   optional RGB-aligned depth as 32FC1

    # String:
    #   JSON metadata fallback

    # MarkerArray:
    #   RViz boxes and labels

    # The main function is:
    # from pcd_ros2_utils import build_segment_ros_messages

    # msgs = build_segment_ros_messages(
    #     seg3d,
    #     frame_id="camera_rgb_optical_frame",
    #     stamp=self.get_clock().now(),
    #     include_depth=True,
    #     include_markers=True,
    # )

    # Then publish:
    # self.segment_cloud_pub.publish(msgs.cloud)
    # self.instance_map_pub.publish(msgs.instance_map)
    # self.info_json_pub.publish(msgs.info_json)

    # if msgs.depth_rgb is not None:
    #     self.depth_rgb_pub.publish(msgs.depth_rgb)

    # if msgs.markers is not None:
    #     self.markers_pub.publish(msgs.markers)

    # Suggested publishers:
    # from sensor_msgs.msg import PointCloud2, Image
    # from std_msgs.msg import String
    # from visualization_msgs.msg import MarkerArray

    # self.segment_cloud_pub = self.create_publisher(
    #     PointCloud2,
    #     "/perception/segments/cloud",
    #     10,
    # )

    # self.instance_map_pub = self.create_publisher(
    #     Image,
    #     "/perception/segments/instance_map",
    #     10,
    # )

    # self.depth_rgb_pub = self.create_publisher(
    #     Image,
    #     "/perception/segments/depth_rgb",
    #     10,
    # )

    # self.info_json_pub = self.create_publisher(
    #     String,
    #     "/perception/segments/info_json",
    #     10,
    # )

    # self.markers_pub = self.create_publisher(
    #     MarkerArray,
    #     "/perception/segments/markers",
    #     10,
    # )

    # The most important design choice is that `/perception/segments/cloud` is a **single sparse combined cloud**, not one topic per object. Each point carries `instance_id`, `class_id`, `confidence`, and original RGB pixel coordinates `u, v`, so downstream ROS nodes can filter objects easily.
    pass