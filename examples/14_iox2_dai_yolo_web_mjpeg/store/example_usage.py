from pathlib import Path
import time

from custom_record_store import CustomStore


# 2026-07-08T14:35:10.000000000Z
TIMESTAMP_NS = time.time_ns()

store = CustomStore("recording")
record = store.add_record(
    mode="rgb_stereo",
    timestamp_ns_utc=TIMESTAMP_NS,
    field_id="field_01",
    sequence=1,
    exist_ok=True,
)

# Use Pillow images, NumPy arrays, encoded bytes, or existing file paths.
# This example uses Pillow if available.
try:
    from PIL import Image

    rgb = Image.new("RGB", (1920, 1080))
    left = Image.new("L", (1280, 720))
    right = Image.new("L", (1280, 720))
    disparity = Image.new("I;16", (1280, 720))

    record.add_image("cam_c", "rgb", rgb)
    record.add_image("cam_c", "left", left)
    record.add_image("cam_c", "right", right)

    record.add_disparity(
        "cam_c",
        disparity,
        {
            "algorithm": "sgbm",
            "scale": 16.0,
            "invalid_value": 0,
            "rectified": True,
        },
    )
except ImportError:
    print("Install pillow to run the image-writing part of this example.")

record.add_calibration(
    "cam_c",
    {
        "camera_id": "cam_c",
        "camera_type": "rgb_stereo",
        "streams": {
            "rgb": {"width": 1920, "height": 1080, "intrinsics": {}, "distortion": {}},
            "left": {"width": 1280, "height": 720, "intrinsics": {}, "distortion": {}},
            "right": {"width": 1280, "height": 720, "intrinsics": {}, "distortion": {}},
        },
        "extrinsics": {"rgb_to_left": {}, "left_to_right": {}},
        "rectification": {"left": {}, "right": {}},
    },
)

record.add_gis(
    "location",
    {
        "timestamp_ns": TIMESTAMP_NS,
        "frame": "wgs84",
        "latitude": 35.681236,
        "longitude": 139.767125,
        "altitude_m": 42.1,
        "source": "rtk",
    },
)

record.add_gis(
    "pose",
    {
        "timestamp_ns": TIMESTAMP_NS,
        "frame": "local",
        "position_m": [1.2, 0.4, 0.0],
        "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "source": "slam",
    },
)

record.add_gis(
    "coordinate_system",
    {
        "frame": "local",
        "origin": "first_frame",
        "unit": "meters",
    },
)

record.add_yolo(
    camera_id="cam_c",
    stream="rgb",
    detections=[
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.92,
            "bbox_xyxy": [120.4, 80.1, 300.2, 420.8],
            "track_id": 17,
        }
    ],
    model_info={
        "name": "yolov8n",
        "version": "8.x",
        "classes": {"0": "person", "1": "car"},
    },
)

record.add_point_cloud(
    "cam_c",
    cloud=[
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.1],
        [0.0, 0.1, 0.9],
    ],
    metadata={"algorithm": "disparity_to_pointcloud"},
)

record.add_object_point_cloud(
    class_id=0,
    class_name="person",
    object_index=1,
    cloud=[
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.1],
    ],
    metadata={"source_detection": "yolo/cam_c/rgb.json"},
)

issues = record.close(validate=True)
print(f"Record path: {record.path}")
print("Validation issues:", issues)
print("All records:")
for path in store.list_records(mode="rgb_stereo", date="2026-07-08", field_id="field_01"):
    print(" -", path)
