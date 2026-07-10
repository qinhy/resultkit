"""Custom synchronized single-frame record store.

This module implements the directory format discussed in the conversation:

    recording/<mode>/<date_utc>/<field_id>/<record_id>/...

Each leaf directory is one synchronized capture record.

Supported modes:
    - dual_rgb
    - rgb_stereo

The implementation intentionally uses only the Python standard library for core
JSON/path/validation behavior. Camera image writing stores MJPEG/JPEG frames as .jpg files and supports
bytes, existing files, Pillow images, and NumPy arrays when Pillow is installed.
Disparity images remain PNG. PCD writing supports
bytes, existing files, and simple Nx3/Nx4/Nx6/Nx7 point arrays/sequences.
"""

from __future__ import annotations

import calendar
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, List, Literal, Mapping, Sequence, Union, cast
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


BytesLike = Union[bytes, bytearray, memoryview]
RecordMode = Literal["dual_rgb", "rgb_stereo"]
ImageStream = Literal["rgb", "left", "right"]
GISKind = Literal[
    "location",
    "pose",
    "coordinate_system",
    "geofences",
    "map_notes",
]


# parser = argparse.ArgumentParser(description="Custom synchronized single-frame record store")
# parser.add_argument("--dual-rgb-cam0", default="cam_a")
# parser.add_argument("--dual-rgb-cam1", default="cam_b")
# parser.add_argument("--rgb-stereo-cam", default="cam_c")
# args = parser.parse_args()

SCHEMA_VERSION = "1.0"
NS_PER_SECOND = 1_000_000_000

# Camera streams come from MJPEG. Each single-frame capture is stored as one
# JPEG frame using .jpg. Disparity is intentionally kept as .png.
CAMERA_IMAGE_EXTENSION = ".jpg"
CAMERA_IMAGE_ENCODING = "mjpeg"
JPEG_LIKE_SUFFIXES = {".jpg", ".jpeg", ".mjpg", ".mjpeg"}

RECORD_ID_RE = re.compile(
    r"^(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\."
    r"(?P<nsec>\d{9})JST$"
)

VALID_MODES: set[str] = {"dual_rgb", "rgb_stereo"}
GIS_EXTENSIONS: dict[GISKind, str] = {
    "location": ".json",
    "pose": ".json",
    "coordinate_system": ".json",
    "geofences": ".geojson",
    "map_notes": ".geojson",
}

DUAL_RGB_CAM_NAME_0 = "cam_a" # args.dual_rgb_cam0
DUAL_RGB_CAM_NAME_1 = "cam_b" # args.dual_rgb_cam1
RGB_STEREO_CAM_NAME = "cam_c" # args.rgb_stereo_cam

REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "dual_rgb": (
        "record.json",
        f"calib/{DUAL_RGB_CAM_NAME_0}.json",
        f"calib/{DUAL_RGB_CAM_NAME_1}.json",
        "calib/extrinsics.json",
        "gis/location.json",
        "gis/pose.json",
        "gis/coordinate_system.json",
        f"imgs/{DUAL_RGB_CAM_NAME_0}/rgb.jpg",
        f"imgs/{DUAL_RGB_CAM_NAME_1}/rgb.jpg",
    ),
    "rgb_stereo": (
        "record.json",
        f"calib/{RGB_STEREO_CAM_NAME}.json",
        "gis/location.json",
        "gis/pose.json",
        "gis/coordinate_system.json",
        f"imgs/{RGB_STEREO_CAM_NAME}/rgb.jpg",
        f"imgs/{RGB_STEREO_CAM_NAME}/left.jpg",
        f"imgs/{RGB_STEREO_CAM_NAME}/right.jpg",
        f"depth/{RGB_STEREO_CAM_NAME}/disparity.png",
        f"depth/{RGB_STEREO_CAM_NAME}/disparity.json",
    ),
}

MODE_CAMERAS: dict[str, tuple[str, ...]] = {
    "dual_rgb": (DUAL_RGB_CAM_NAME_0, DUAL_RGB_CAM_NAME_1),
    "rgb_stereo": (RGB_STEREO_CAM_NAME,),
}

MODE_STREAMS: dict[str, dict[str, tuple[str, ...]]] = {
    "dual_rgb": {DUAL_RGB_CAM_NAME_0: ("rgb",), DUAL_RGB_CAM_NAME_1: ("rgb",)},
    "rgb_stereo": {RGB_STEREO_CAM_NAME: ("rgb", "left", "right")},
}


@dataclass(frozen=True)
class RecordTimestamp:
    """Convenience view of a timestamp and its filesystem identifiers."""

    timestamp_ns_utc: int
    datetime_utc: str
    date_utc: str
    record_id: str


def datetime_utc_from_timestamp_ns(timestamp_ns_utc: int) -> str:
    """Return an ISO-like UTC timestamp with nanosecond precision."""
    seconds, nsec = divmod(int(timestamp_ns_utc), NS_PER_SECOND)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{nsec:09d}Z"


def date_utc_from_timestamp_ns(timestamp_ns_utc: int) -> str:
    """Return YYYY-MM-DD in UTC."""
    seconds = int(timestamp_ns_utc) // NS_PER_SECOND
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d")


JST = timezone(timedelta(hours=9), name="JST")
def record_id_from_timestamp_ns(timestamp_ns_utc: int, sequence: int = 1) -> str:
    """Return HHMMSS.NNNNNNNNNJST for a UTC nanosecond timestamp shown in JST."""
    # if sequence < 0:
    #     raise ValueError("sequence must be non-negative")
    seconds, nsec = divmod(int(timestamp_ns_utc), NS_PER_SECOND)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(JST)
    return f"{dt:%H%M%S}.{nsec:09d}JST"


def timestamp_ns_from_date_and_record_id(date_utc: str, record_id: str) -> int:
    """Parse timestamp ns from a date folder and record_id."""
    match = RECORD_ID_RE.match(record_id)
    if not match:
        raise ValueError(f"Invalid record_id: {record_id!r}")

    year, month, day = map(int, date_utc.split("-"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    nsec = int(match.group("nsec"))

    epoch_seconds = calendar.timegm((year, month, day, hour, minute, second))
    return epoch_seconds * NS_PER_SECOND + nsec


def timestamp_info(timestamp_ns_utc: int, sequence: int = 1) -> RecordTimestamp:
    """Return date, datetime string, and record_id for a UTC ns timestamp."""
    return RecordTimestamp(
        timestamp_ns_utc=int(timestamp_ns_utc),
        datetime_utc=datetime_utc_from_timestamp_ns(timestamp_ns_utc),
        date_utc=date_utc_from_timestamp_ns(timestamp_ns_utc),
        record_id=record_id_from_timestamp_ns(timestamp_ns_utc, sequence),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp_path = Path(f.name)
        f.write(text)
        f.flush()
    tmp_path.replace(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
        tmp_path = Path(f.name)
        f.write(data)
        f.flush()
    tmp_path.replace(path)


def _write_json(path: Path, data: Any) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _is_existing_file(value: Any) -> bool:
    if isinstance(value, Path):
        return value.is_file()
    if isinstance(value, str):
        try:
            return Path(value).is_file()
        except OSError:
            return False
    return False


def _write_image_png(path: Path, image: Any) -> None:
    """Write an image as PNG.

    Supported input types:
        - bytes/bytearray/memoryview containing encoded PNG data
        - path-like object pointing to an existing file
        - object with .save(path)
        - NumPy array, if Pillow is installed
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, (bytes, bytearray, memoryview)):
        _atomic_write_bytes(path, bytes(image))
        return

    if _is_existing_file(image):
        _copy_file(Path(image), path)
        return

    if hasattr(image, "save"):
        image.save(path)
        return

    # NumPy array support without requiring NumPy as a hard dependency.
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise TypeError(
                "NumPy image arrays require Pillow to be installed. "
                "Install pillow or pass encoded bytes/a file path."
            ) from exc

        arr = image
        pil_image = Image.fromarray(arr)
        pil_image.save(path)
        return

    raise TypeError(
        "Unsupported image type. Pass encoded bytes, an existing file path, "
        "a Pillow image, or a NumPy array."
    )


def _jpeg_ready_image(image: Any) -> Any:
    """Return a Pillow image converted to a JPEG-compatible mode when needed."""
    mode = getattr(image, "mode", None)
    if mode in {"RGBA", "LA", "P"}:
        return image.convert("RGB")
    return image


def _write_mjpeg_frame(path: Path, image: Any) -> None:
    """Write a camera frame as a JPEG file.

    MJPEG camera streams provide individual JPEG-compressed frames. For bytes-like
    input, this function writes those encoded frame bytes directly to .jpg without
    decoding/re-encoding. Pillow images and NumPy arrays are encoded as JPEG for
    convenience in tests or non-camera call sites.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, (bytes, bytearray, memoryview)):
        _atomic_write_bytes(path, bytes(image))
        return

    if _is_existing_file(image):
        src = Path(image)
        if src.suffix.lower() in JPEG_LIKE_SUFFIXES:
            _copy_file(src, path)
            return

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise TypeError(
                "Only JPEG/MJPEG frame files can be copied directly. "
                "Install pillow to transcode other image files to JPEG, or pass "
                "encoded MJPEG/JPEG frame bytes."
            ) from exc

        with Image.open(src) as im:
            _jpeg_ready_image(im).save(path, format="JPEG")
        return

    if hasattr(image, "save"):
        _jpeg_ready_image(image).save(path, format="JPEG")
        return

    # NumPy array support without requiring NumPy as a hard dependency.
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise TypeError(
                "NumPy image arrays require Pillow to be installed. "
                "Install pillow or pass encoded MJPEG/JPEG frame bytes/a file path."
            ) from exc

        pil_image = Image.fromarray(image)
        _jpeg_ready_image(pil_image).save(path, format="JPEG")
        return

    raise TypeError(
        "Unsupported image type. Pass encoded MJPEG/JPEG frame bytes, an existing "
        "JPEG file path, a Pillow image, or a NumPy array."
    )


def _normalize_points(cloud: Any) -> list[list[float]]:
    """Convert a sequence/array of points to a list of numeric rows."""
    if hasattr(cloud, "tolist"):
        cloud = cloud.tolist()

    if not isinstance(cloud, Sequence) or isinstance(cloud, (str, bytes, bytearray)):
        raise TypeError("Point cloud must be bytes, a file path, or a sequence/array of points")

    points: list[list[float]] = []
    for row in cloud:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError("Each point must be a numeric sequence")
        values = [float(v) for v in row]
        if len(values) not in (3, 4, 6, 7):
            raise ValueError(
                "Each point must have 3, 4, 6, or 7 values: "
                "xyz, xyzi, xyzrgb, or xyzrgbi"
            )
        points.append(values)

    return points


def _pcd_header(fields: list[str], point_count: int) -> str:
    sizes = ["4"] * len(fields)
    types = ["F"] * len(fields)
    counts = ["1"] * len(fields)
    return "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            f"FIELDS {' '.join(fields)}",
            f"SIZE {' '.join(sizes)}",
            f"TYPE {' '.join(types)}",
            f"COUNT {' '.join(counts)}",
            f"WIDTH {point_count}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {point_count}",
            "DATA ascii",
        ]
    ) + "\n"


def _write_pcd(path: Path, cloud: Any) -> None:
    """Write a PCD file.

    Supported input types:
        - bytes/bytearray/memoryview containing an already-encoded PCD
        - path-like object pointing to an existing PCD file
        - sequence/array of points with 3, 4, 6, or 7 values per point
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(cloud, (bytes, bytearray, memoryview)):
        _atomic_write_bytes(path, bytes(cloud))
        return

    if _is_existing_file(cloud):
        _copy_file(Path(cloud), path)
        return

    points = _normalize_points(cloud)

    if not points:
        fields = ["x", "y", "z"]
    else:
        width = len(points[0])
        if width == 3:
            fields = ["x", "y", "z"]
        elif width == 4:
            fields = ["x", "y", "z", "intensity"]
        elif width == 6:
            fields = ["x", "y", "z", "r", "g", "b"]
        elif width == 7:
            fields = ["x", "y", "z", "r", "g", "b", "intensity"]
        else:  # _normalize_points already checks this.
            raise ValueError("Unsupported point width")

    lines = [_pcd_header(fields, len(points))]
    lines.extend(" ".join(f"{v:.9g}" for v in row) + "\n" for row in points)
    _atomic_write_text(path, "".join(lines))


def _sanitize_name(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "object"


def _with_schema(data: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    result.setdefault("schema_version", SCHEMA_VERSION)
    return result


class CustomRecord(BaseModel):
    """One synchronized single-frame capture record."""

    root_path: Path
    mode: RecordMode
    field_id: str
    record_id: str
    timestamp_ns_utc: int
    date_utc: str | None = None

    @staticmethod
    def empty() -> "CustomRecord":
        return CustomRecord(
            root_path=Path("."),
            mode="dual_rgb",
            field_id="null",
            record_id="000000.000000000JST",
            timestamp_ns_utc=0,
            date_utc=None,
        )
    
    def is_empty(self) -> bool:
        return self.field_id == "null"

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, mode: RecordMode) -> RecordMode:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode!r}")
        return mode

    @field_validator("record_id")
    @classmethod
    def _validate_record_id(cls, record_id: str) -> str:
        if not RECORD_ID_RE.match(record_id):
            raise ValueError(f"Invalid record_id: {record_id!r}")
        return record_id

    @field_validator("field_id")
    @classmethod
    def _validate_field_id(cls, field_id: str) -> str:
        if not field_id:
            raise ValueError("field_id must not be empty")
        return field_id

    @model_validator(mode="after")
    def _fill_date_utc(self) -> "CustomRecord":
        if self.date_utc is None:
            self.date_utc = date_utc_from_timestamp_ns(self.timestamp_ns_utc)
        return self

    @computed_field
    @property
    def path(self) -> Path:
        """Return the leaf record directory path."""
        return (
            self.root_path
            / self.mode
            / self.date_utc
            / self.field_id
            / self.record_id
        )

    @computed_field
    @property
    def datetime_utc(self) -> str:
        """Return this record timestamp as an ISO-like UTC string."""
        return datetime_utc_from_timestamp_ns(self.timestamp_ns_utc)

    @classmethod
    def from_path(cls, path: str | Path) -> "CustomRecord":
        """
        Construct a CustomRecord from a leaf record directory path.

        Expected layout:
            <root_path>/<mode>/<date_utc>/<field_id>/<record_id>
        """
        path = Path(path)

        if len(path.parts) < 4:
            raise ValueError(f"Path is too short to be a record path: {path}")

        mode = path.parts[-4]
        date_utc = path.parts[-3]
        field_id = path.parts[-2]
        record_id = path.parts[-1]

        root_path = Path(*path.parts[:-4]) if path.parts[:-4] else Path(".")

        timestamp_ns_utc = timestamp_ns_from_date_and_record_id(
            date_utc,
            record_id,
        )

        return cls(
            root_path=root_path,
            mode=cast(RecordMode, mode),
            field_id=field_id,
            record_id=record_id,
            timestamp_ns_utc=timestamp_ns_utc,
            date_utc=date_utc,
        )
    
    @property
    def calib_path(self) -> Path:
        return self.path / "calib"
    
    @property
    def image_path(self) -> Path:
        return self.path / "imgs"
    
    @property
    def depth_path(self) -> Path:
        return self.path / "depth"
    
    @property
    def pcd_path(self) -> Path:
        return self.path / "pcd"
    
    @property
    def listup_rgb_image_paths(self) -> List[Path]:
        return list(self.image_path.rglob(f"rgb{CAMERA_IMAGE_EXTENSION}"))
    
    @property
    def listup_rgb_image_parent_paths(self) -> List[Path]:
        return [p.parent for p in self.listup_rgb_image_paths]
    
    @property
    def listup_left_image_paths(self) -> List[Path]:
        return list(self.image_path.rglob(f"left{CAMERA_IMAGE_EXTENSION}"))
    
    @property
    def listup_left_image_parent_paths(self) -> List[Path]:
        return [p.parent for p in self.listup_left_image_paths]
    
    @property
    def listup_right_image_paths(self) -> List[Path]:
        return list(self.image_path.rglob(f"right{CAMERA_IMAGE_EXTENSION}"))
    
    @property
    def is_stereo(self) -> bool:
        return self.mode == "rgb_stereo"

    @property
    def expect_disparity_path(self) -> Path | None:
        if self.is_stereo:
            return self.path / "depth" / f"{RGB_STEREO_CAM_NAME}" / "disparity.png"
        else:
            return None
        
    @property
    def expect_disparity_json_path(self) -> Path | None:
        if self.is_stereo:
            return self.path / "depth" / f"{RGB_STEREO_CAM_NAME}" / "disparity.json"
        else:
            return None
        
    @property
    def expect_stereo_calib(self) -> Path | None:
        if self.is_stereo:
            return self.path / "calib" / f"{RGB_STEREO_CAM_NAME}.json"
        else:
            return None
    @property
    def expect_pcd_path(self) -> Path | None:
        if self.is_stereo:
            return self.path / "pcd" / f"{RGB_STEREO_CAM_NAME}.pcd"
        else:
            return None
    
    @property
    def expect_pcd_seg_dir(self) -> Path | None:
        if self.is_stereo:
            return self.path / "pcd" / f"{RGB_STEREO_CAM_NAME}_segs"
        else:
            return None

    def mkdirs(self) -> None:
        """Create the leaf record directory."""
        self.path.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        """Return a POSIX relative path from this record root."""
        return _rel(path, self.path)

    def add_calibration(self, camera_id: str, data: dict[str, Any]) -> None:
        """Write calib/{camera_id}.json."""
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        _write_json(self.path / "calib" / f"{camera_id}.json", _with_schema(data))

    def add_extrinsics(self, data: dict[str, Any]) -> None:
        """Write calib/extrinsics.json."""
        _write_json(self.path / "calib" / "extrinsics.json", _with_schema(data))

    def add_gis(self, kind: GISKind, data: Any) -> None:
        """Write one GIS file using the canonical extension mapping."""
        if kind not in GIS_EXTENSIONS:
            raise ValueError(f"Unsupported GIS kind: {kind!r}")
        suffix = GIS_EXTENSIONS[kind]
        out = self.path / "gis" / f"{kind}{suffix}"
        if isinstance(data, Mapping):
            data = _with_schema(data)
        _write_json(out, data)

    def add_image(self, camera_id: str, stream: ImageStream, image: Any) -> None:
        """Write imgs/{camera_id}/{stream}.jpg from an MJPEG/JPEG frame."""
        if stream not in {"rgb", "left", "right"}:
            raise ValueError(f"Unsupported image stream: {stream!r}")
        _write_mjpeg_frame(
            self.path / "imgs" / camera_id / f"{stream}{CAMERA_IMAGE_EXTENSION}",
            image,
        )
    
    def add_mjpeg_image(self, camera_id: str, stream: ImageStream, image_bytes: Any) -> None:
        """Backward-compatible alias for add_image()."""
        self.add_image(camera_id, stream, image_bytes)

    def add_yolo(
        self,
        camera_id: str,
        stream: str,
        detections: list[dict[str, Any]],
        model_info: dict[str, Any],
        overlay_image: Any | None = None,
    ) -> None:
        """Write yolo/{camera_id}/{stream}.json and optional overlay JPG."""
        if not stream:
            raise ValueError("stream must not be empty")

        source_image = self.path / "imgs" / camera_id / f"{stream}{CAMERA_IMAGE_EXTENSION}"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_ns": self.timestamp_ns_utc,
            "camera": camera_id,
            "stream": stream,
            "source_image": source_image.relative_to(self.path).as_posix(),
            "model": model_info,
            "detections": detections,
        }
        _write_json(self.path / "yolo" / camera_id / f"{stream}.json", payload)

        if overlay_image is not None:
            _write_mjpeg_frame(
                self.path / "yolo" / camera_id / f"{stream}_overlay{CAMERA_IMAGE_EXTENSION}",
                overlay_image,
            )

    def add_disparity(
        self,
        camera_id: str,
        disparity_image: Any,
        metadata: dict[str, Any],
    ) -> None:
        """Write depth/{camera_id}/disparity.png and disparity.json."""
        out_dir = self.path / "depth" / camera_id
        _write_image_png(out_dir / "disparity.png", disparity_image)

        default_metadata = {
            "schema_version": SCHEMA_VERSION,
            "source_streams": [
                f"imgs/{camera_id}/left{CAMERA_IMAGE_EXTENSION}",
                f"imgs/{camera_id}/right{CAMERA_IMAGE_EXTENSION}",
            ],
            "unit": "pixels",
            "encoding": "uint16_png",
            "scale": 16.0,
            "invalid_value": 0,
            "algorithm": "unknown",
            "rectified": True,
        }
        default_metadata.update(metadata)
        default_metadata.setdefault("schema_version", SCHEMA_VERSION)
        _write_json(out_dir / "disparity.json", default_metadata)

    def add_point_cloud(
        self,
        camera_id: str,
        cloud: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write pcd/{camera_id}.pcd and optional pcd/{camera_id}.json."""
        _write_pcd(self.path / "pcd" / f"{camera_id}.pcd", cloud)
        if metadata is not None:
            default_metadata = {
                "schema_version": SCHEMA_VERSION,
                "camera": camera_id,
                "source_disparity": f"depth/{camera_id}/disparity.png",
                "source_disparity_metadata": f"depth/{camera_id}/disparity.json",
                "source_streams": [
                    f"imgs/{camera_id}/left{CAMERA_IMAGE_EXTENSION}",
                    f"imgs/{camera_id}/right{CAMERA_IMAGE_EXTENSION}",
                ],
                "frame": f"{camera_id}_left",
                "unit": "meters",
            }
            default_metadata.update(metadata)
            _write_json(self.path / "pcd" / f"{camera_id}.json", default_metadata)

    def add_object_point_cloud(
        self,
        class_id: int,
        class_name: str,
        object_index: int,
        cloud: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write pcd/objects/classXX_name_NNN.pcd and update pcd/objects.json."""
        if class_id < 0:
            raise ValueError("class_id must be non-negative")
        if object_index < 1:
            raise ValueError("object_index must be >= 1")

        safe_class_name = _sanitize_name(class_name)
        pcd_rel = f"pcd/objects/class{class_id:02d}_{safe_class_name}_{object_index:03d}.pcd"
        _write_pcd(self.path / pcd_rel, cloud)

        objects_json = self.path / "pcd" / "objects.json"
        if objects_json.exists():
            payload = _read_json(objects_json)
            if not isinstance(payload, dict):
                payload = {"schema_version": SCHEMA_VERSION, "objects": []}
            payload.setdefault("schema_version", SCHEMA_VERSION)
            payload.setdefault("objects", [])
        else:
            payload = {"schema_version": SCHEMA_VERSION, "objects": []}

        entry: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "object_index": object_index,
            "pcd": pcd_rel,
        }
        if metadata:
            entry.update(metadata)

        # Replace duplicate entry for the same path; otherwise append.
        objects = payload.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        objects = [obj for obj in objects if not isinstance(obj, dict) or obj.get("pcd") != pcd_rel]
        objects.append(entry)
        payload["objects"] = objects
        _write_json(objects_json, payload)

    def write_record_json(self) -> None:
        """Scan the record directory and write record.json manifest."""
        self.mkdirs()

        files = self._scan_files()
        cameras = self._scan_cameras(files)
        streams = self._scan_streams(files)
        image_metadata = self._scan_image_metadata(files)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "record_id": self.record_id,
            "timestamp_ns_utc": self.timestamp_ns_utc,
            "datetime_utc": self.datetime_utc,
            "date_utc": self.date_utc,
            "field_id": self.field_id,
            "cameras": cameras,
            "streams": streams,
            "files": files,
            "images": image_metadata,
        }
        _write_json(self.path / "record.json", payload)

    def validate(self) -> list[str]:
        """Validate required files and common references. Returns issues."""
        issues: list[str] = []
        record_path = self.path

        if not record_path.exists():
            return [f"ERROR: record path does not exist: {record_path}"]

        # record_id/date/timestamp consistency.
        try:
            parsed_ns = timestamp_ns_from_date_and_record_id(self.date_utc, self.record_id)
            if parsed_ns != self.timestamp_ns_utc:
                issues.append(
                    "ERROR: timestamp_ns_utc does not match date folder + record_id: "
                    f"expected {parsed_ns}, got {self.timestamp_ns_utc}"
                )
        except Exception as exc:
            issues.append(f"ERROR: could not parse record_id/date: {exc}")

        # Required files.
        for rel_path in REQUIRED_FILES[self.mode]:
            if not (record_path / rel_path).exists():
                issues.append(f"ERROR: missing required file: {rel_path}")

        # record.json content checks.
        record_json_path = record_path / "record.json"
        if record_json_path.exists():
            try:
                manifest = _read_json(record_json_path)
                if manifest.get("record_id") != self.record_id:
                    issues.append("ERROR: record.json record_id does not match directory name")
                if manifest.get("mode") != self.mode:
                    issues.append("ERROR: record.json mode does not match record mode")
                if manifest.get("field_id") != self.field_id:
                    issues.append("ERROR: record.json field_id does not match field directory")
                if manifest.get("date_utc") != self.date_utc:
                    issues.append("ERROR: record.json date_utc does not match date directory")
                if manifest.get("timestamp_ns_utc") != self.timestamp_ns_utc:
                    issues.append("ERROR: record.json timestamp_ns_utc does not match record")
            except Exception as exc:
                issues.append(f"ERROR: could not read record.json: {exc}")

        # Raw MJPEG stream extensions are not used for single extracted frames.
        # Store extracted MJPEG/JPEG frames as .jpg instead.
        for file in record_path.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".mjpeg", ".mjpg"}:
                issues.append(
                    f"WARNING: store single MJPEG frames as .jpg, not {file.suffix}: {self.rel(file)}"
                )

        # YOLO source_image references.
        for yolo_json in (record_path / "yolo").glob("*/*.json") if (record_path / "yolo").exists() else []:
            try:
                payload = _read_json(yolo_json)
                source_image = payload.get("source_image")
                if source_image and not (record_path / source_image).exists():
                    issues.append(
                        f"ERROR: YOLO source_image does not exist in {self.rel(yolo_json)}: {source_image}"
                    )
                if payload.get("timestamp_ns") not in (None, self.timestamp_ns_utc):
                    issues.append(f"WARNING: YOLO timestamp differs from record: {self.rel(yolo_json)}")
            except Exception as exc:
                issues.append(f"ERROR: could not read YOLO JSON {self.rel(yolo_json)}: {exc}")

        # Disparity source streams.
        for disparity_json in (record_path / "depth").glob("*/disparity.json") if (record_path / "depth").exists() else []:
            try:
                payload = _read_json(disparity_json)
                for source in payload.get("source_streams", []):
                    if not (record_path / source).exists():
                        issues.append(
                            f"ERROR: disparity source stream does not exist in "
                            f"{self.rel(disparity_json)}: {source}"
                        )
            except Exception as exc:
                issues.append(f"ERROR: could not read disparity JSON {self.rel(disparity_json)}: {exc}")

        # Point cloud metadata references.
        for pcd_json in (record_path / "pcd").glob("*.json") if (record_path / "pcd").exists() else []:
            if pcd_json.name == "objects.json":
                continue
            try:
                payload = _read_json(pcd_json)
                for key in ("source_disparity", "source_disparity_metadata"):
                    ref = payload.get(key)
                    if ref and not (record_path / ref).exists():
                        issues.append(f"ERROR: {key} does not exist in {self.rel(pcd_json)}: {ref}")
            except Exception as exc:
                issues.append(f"ERROR: could not read PCD metadata {self.rel(pcd_json)}: {exc}")

        # Object PCD references.
        objects_json = record_path / "pcd" / "objects.json"
        if objects_json.exists():
            try:
                payload = _read_json(objects_json)
                for obj in payload.get("objects", []):
                    if not isinstance(obj, dict):
                        issues.append("ERROR: object entry is not a JSON object")
                        continue
                    pcd_ref = obj.get("pcd")
                    if pcd_ref and not (record_path / pcd_ref).exists():
                        issues.append(f"ERROR: object PCD does not exist: {pcd_ref}")
                    source_detection = obj.get("source_detection")
                    if source_detection and not (record_path / source_detection).exists():
                        issues.append(f"ERROR: object source_detection does not exist: {source_detection}")
            except Exception as exc:
                issues.append(f"ERROR: could not read objects.json: {exc}")

        return issues

    def close(self, validate: bool = False) -> list[str]:
        """Finalize the record by writing record.json. Optionally validate."""
        self.write_record_json()
        return self.validate() if validate else []

    def _scan_files(self) -> dict[str, list[str]]:
        base = self.path

        def glob(pattern: str) -> list[str]:
            return sorted(_rel(p, base) for p in base.glob(pattern) if p.is_file())

        def glob_many(*patterns: str) -> list[str]:
            files: list[str] = []
            for pattern in patterns:
                files.extend(glob(pattern))
            return sorted(files)

        return {
            "calibration": glob("calib/*.json"),
            "gis": glob("gis/*.json") + glob("gis/*.geojson"),
            "images": glob_many("imgs/*/*.jpg", "imgs/*/*.jpeg"),
            "yolo": glob("yolo/*/*.json"),
            "yolo_overlays": glob_many("yolo/*/*_overlay.jpg", "yolo/*/*_overlay.jpeg"),
            "disparity": glob("depth/*/disparity.png") + glob("depth/*/disparity.json"),
            "point_clouds": [
                p for p in glob("pcd/*.pcd") if not p.startswith("pcd/objects/")
            ],
            "point_cloud_metadata": [
                p for p in glob("pcd/*.json") if p != "pcd/objects.json"
            ],
            "object_point_clouds": glob("pcd/objects/*.pcd"),
            "object_metadata": glob("pcd/objects.json"),
            "logs": glob("logs/*.log"),
        }

    def _scan_cameras(self, files: dict[str, list[str]]) -> list[str]:
        cameras: set[str] = set()
        for rel_path in files.get("images", []):
            parts = Path(rel_path).parts
            if len(parts) >= 3 and parts[0] == "imgs":
                cameras.add(parts[1])
        if not cameras:
            cameras.update(MODE_CAMERAS[self.mode])
        return sorted(cameras)

    def _scan_streams(self, files: dict[str, list[str]]) -> dict[str, list[str]]:
        streams: dict[str, set[str]] = {}
        for rel_path in files.get("images", []):
            parts = Path(rel_path).parts
            if len(parts) >= 3 and parts[0] == "imgs":
                camera = parts[1]
                stream = Path(parts[2]).stem
                streams.setdefault(camera, set()).add(stream)
        if not streams:
            return {camera: list(streams_) for camera, streams_ in MODE_STREAMS[self.mode].items()}
        return {camera: sorted(values) for camera, values in sorted(streams.items())}

    def _scan_image_metadata(self, files: dict[str, list[str]]) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        for rel_path in files.get("images", []):
            parts = Path(rel_path).parts
            if len(parts) < 3:
                continue
            image_path = self.path / rel_path
            item: dict[str, Any] = {
                "camera": parts[1],
                "stream": Path(parts[2]).stem,
                "path": rel_path,
                "timestamp_ns": self.timestamp_ns_utc,
                "encoding": CAMERA_IMAGE_ENCODING,
            }
            # Width/height are optional and only available if Pillow exists.
            try:
                from PIL import Image  # type: ignore

                with Image.open(image_path) as im:
                    item["width"], item["height"] = im.size
                    mode = im.mode
                    item["color_space"] = {
                        "RGB": "rgb",
                        "RGBA": "rgba",
                        "L": "gray",
                        "I;16": "gray16",
                    }.get(mode, mode.lower())
            except Exception:
                pass
            metadata.append(item)
        return metadata


class CustomStore:
    """Store for synchronized single-frame capture records."""

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)

    def add_record(
        self,
        mode: RecordMode,
        timestamp_ns_utc: int,
        field_id: str = "field_01",
        sequence: int = 1,
        exist_ok: bool = False,
    ) -> CustomRecord:
        """Create a new single-frame record directory and return it."""
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode!r}")
        info = timestamp_info(timestamp_ns_utc, sequence)
        record = CustomRecord(
            root_path=self.root_path,
            mode=mode,
            field_id=field_id,
            record_id=info.record_id,
            timestamp_ns_utc=timestamp_ns_utc,
            date_utc=info.date_utc,
        )
        if record.path.exists() and not exist_ok:
            raise FileExistsError(f"Record already exists: {record.path}")
        record.mkdirs()
        return record

    def get_record(
        self,
        mode: RecordMode,
        date: str,
        field_id: str,
        record_id: str,
    ) -> CustomRecord:
        """Load an existing record."""
        path = self.root_path / mode / date / field_id / record_id
        if not path.exists():
            raise FileNotFoundError(f"Record does not exist: {path}")

        record_json = path / "record.json"
        if record_json.exists():
            payload = _read_json(record_json)
            timestamp_ns_utc = int(payload["timestamp_ns_utc"])
        else:
            timestamp_ns_utc = timestamp_ns_from_date_and_record_id(date, record_id)

        return CustomRecord(
            root_path=self.root_path,
            mode=mode,
            field_id=field_id,
            record_id=record_id,
            timestamp_ns_utc=timestamp_ns_utc,
            date_utc=date,
        )

    def list_records(
        self,
        mode: RecordMode | None = None,
        date: str | None = None,
        field_id: str | None = None,
    ) -> list[Path]:
        """List leaf record directories, optionally filtered by mode/date/field."""
        modes = [mode] if mode is not None else sorted(VALID_MODES)
        results: list[Path] = []

        for mode_name in modes:
            if mode_name not in VALID_MODES:
                raise ValueError(f"Unsupported mode: {mode_name!r}")
            mode_root = self.root_path / mode_name
            if not mode_root.exists():
                continue

            date_dirs = [mode_root / date] if date else [p for p in mode_root.iterdir() if p.is_dir()]
            for date_dir in date_dirs:
                if not date_dir.exists() or not date_dir.is_dir():
                    continue
                field_dirs = [date_dir / field_id] if field_id else [p for p in date_dir.iterdir() if p.is_dir()]
                for field_dir in field_dirs:
                    if not field_dir.exists() or not field_dir.is_dir():
                        continue
                    for record_dir in field_dir.iterdir():
                        if record_dir.is_dir() and RECORD_ID_RE.match(record_dir.name):
                            results.append(record_dir)

        return sorted(results)

    def validate_record(
        self,
        mode: RecordMode,
        date: str,
        field_id: str,
        record_id: str,
    ) -> list[str]:
        """Validate one record."""
        return self.get_record(mode, date, field_id, record_id).validate()


__all__ = [
    "CustomRecord",
    "CustomStore",
    "CAMERA_IMAGE_ENCODING",
    "CAMERA_IMAGE_EXTENSION",
    "GIS_EXTENSIONS",
    "GISKind",
    "ImageStream",
    "RecordMode",
    "RecordTimestamp",
    "date_utc_from_timestamp_ns",
    "datetime_utc_from_timestamp_ns",
    "record_id_from_timestamp_ns",
    "timestamp_info",
    "timestamp_ns_from_date_and_record_id",
]
