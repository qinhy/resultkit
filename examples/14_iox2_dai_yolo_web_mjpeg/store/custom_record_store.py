"""Filesystem-backed synchronized single-frame record store.

Directory layout::

    <root>/<mode>/<date_utc>/<field_id>/<record_id>/
        calib/<camera_id>.json
        imgs/<camera_id>/{rgb,left,right}.jpg
        gnss/<kind>.json
        pcd/...
        yolo/<camera_id>/<stream>.json
        _SUCCESS                         # optional record commit marker

The same logical layout is supported on local filesystems and fsspec URLs
(e.g. ``s3://bucket/prefix``).

``record_id`` stores the capture clock time in JST while ``date_utc`` stores the
UTC calendar date. The conversion helpers in this module preserve that mixed
convention when round-tripping timestamps.

Dependencies:
    - Pydantic v2
    - NumPy
    - OpenCV (only for decoding/encoding image arrays and non-JPEG image files)
    - fsspec
    - s3fs (optional; required for s3:// and S3-compatible services such as MinIO)

The data-model classes use Pydantic v2 models while still representing a live
filesystem object graph. Parent/child references and filesystem connection objects
are excluded from normal model serialization to avoid recursive wire payloads.
"""
from __future__ import annotations

import calendar
import contextlib
import errno
import json
import os
import posixpath
import re
import threading
from datetime import date, datetime, timedelta, timezone
from functools import total_ordering
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, ClassVar, Dict, Iterator, List, Literal, Mapping, Sequence, cast

import cv2
import fsspec
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator


RecordMode = Literal["dual_rgb", "rgbd_hand"]
ImageStream = Literal["rgb", "left", "right"]
PCDKind = Literal["file", "folder"]
PCDEncoding = Literal["ascii", "binary"]

SCHEMA_VERSION = "1.0"
WIRE_SCHEMA = "custom-record/v1"
NS_PER_SECOND = 1_000_000_000
JST = timezone(timedelta(hours=9), name="JST")

CAMERA_IMAGE_EXTENSION = ".jpg"
CAMERA_IMAGE_ENCODING = "mjpeg"
JPEG_LIKE_SUFFIXES = frozenset({".jpg", ".jpeg", ".mjpg", ".mjpeg"})
VALID_MODES = frozenset({"dual_rgb", "rgbd_hand"})
VALID_IMAGE_STREAMS = frozenset({"rgb", "left", "right"})

RECORD_ID_RE = re.compile(
    r"^(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\."
    r"(?P<nsec>\d{9})JST$"
)


# ---------------------------------------------------------------------------
# Pydantic model bases
# ---------------------------------------------------------------------------


class RecordModel(BaseModel):
    """Base model for the live record-store object graph.

    Pydantic validates construction, while arbitrary filesystem/backend objects are
    allowed. Parent references are excluded on individual fields so ``model_dump``
    does not recurse through the object graph. A small positional-argument shim
    preserves the constructor style of the previous dataclass API.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
        validate_default=True,
        revalidate_instances="never",
    )

    __positional_fields__: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *args: Any, **data: Any) -> None:
        if len(args) > len(self.__positional_fields__):
            raise TypeError(
                f"{type(self).__name__}() takes at most "
                f"{len(self.__positional_fields__)} positional arguments "
                f"but {len(args)} were given"
            )
        for name, value in zip(self.__positional_fields__, args):
            if name in data:
                raise TypeError(
                    f"{type(self).__name__}() got multiple values for argument {name!r}"
                )
            data[name] = value
        super().__init__(**data)


class FrozenRecordModel(RecordModel):
    """Immutable Pydantic model used for path-value objects."""

    model_config = ConfigDict(**RecordModel.model_config, frozen=True)


# ---------------------------------------------------------------------------
# Storage abstraction (local filesystems + fsspec remote filesystems)
# ---------------------------------------------------------------------------


class _ConnState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.generation = 0


class RecordFS:
    def __init__(
        self,
        root_url: str | Path,
        storage_options: Mapping[str, Any] | None = None,
        *,
        keepalive: int = 30,
    ) -> None:
        self.storage_options = dict(storage_options or {})
        self.keepalive = keepalive
        self._state = _ConnState()

        url = str(root_url)
        if "://" not in url:
            url = str(Path(url).absolute())

        if url.split("://", 1)[0] in {"sftp", "ssh"}:
            self.storage_options.setdefault("skip_instance_cache", True)

        self.fs, root = fsspec.core.url_to_fs(url, **self.storage_options)
        self.root_key = root.rstrip("/")

        protocols = self._protocols()
        self.is_local = bool({"file", "local"} & protocols) or bool(
            getattr(self.fs, "local_file", False)
        )
        self.root_url = self.display(self.root_key)
        self._configure()

    def _protocols(self) -> set[str]:
        p = self.fs.protocol
        return set(map(str, p)) if isinstance(p, (tuple, list, set)) else {str(p)}

    @property
    def _is_sftp(self) -> bool:
        return bool({"sftp", "ssh"} & self._protocols())

    def _configure(self) -> None:
        if not self._is_sftp:
            return
        transport = self.fs.client.get_transport()
        if transport and self.keepalive > 0:
            transport.set_keepalive(self.keepalive)

    @staticmethod
    def _connection_error(exc: BaseException) -> bool:
        errnos = {
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ECONNABORTED,
            errno.ENOTCONN,
            errno.ETIMEDOUT,
        }
        messages = (
            "socket is closed",
            "socket closed",
            "connection reset",
            "connection lost",
            "broken pipe",
            "server connection dropped",
            "session is not active",
            "channel closed",
            "transport is not active",
        )

        while exc:
            if isinstance(exc, EOFError):
                return True
            if isinstance(exc, OSError) and exc.errno in errnos:
                return True
            if any(x in str(exc).lower() for x in messages):
                return True
            exc = exc.__cause__ or exc.__context__

        return False

    def _reconnect(self, generation: int | None = None) -> None:
        if not self._is_sftp:
            return

        with self._state.lock:
            if generation is not None and generation != self._state.generation:
                return

            with contextlib.suppress(Exception):
                self.fs.ftp.close()
            with contextlib.suppress(Exception):
                self.fs.client.close()

            self.fs._connect()
            self._configure()
            self._state.generation += 1

    def reconnect(self) -> None:
        self._reconnect()

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self.fs, method)

        if not self._is_sftp:
            return fn(*args, **kwargs)

        generation = self._state.generation

        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not self._connection_error(exc):
                raise

        self._reconnect(generation)
        return getattr(self.fs, method)(*args, **kwargs)

    def read_bytes(self, key: str) -> bytes:
        return self._call("cat_file", key)

    def info(self, key: str) -> dict[str, Any]:
        return self._call("info", key)

    def exists(self, key: str) -> bool:
        try:
            self.info(key)
            return True
        except (FileNotFoundError,):
            return False
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return False
            raise

    def ls(self, key: str, **kwargs: Any) -> Any:
        return self._call("ls", key, **kwargs)

    def glob(self, pattern: str, **kwargs: Any) -> Any:
        return self._call("glob", pattern, **kwargs)

    def find(self, key: str, **kwargs: Any) -> Any:
        return self._call("find", key, **kwargs)

    def _join_key(self, *parts: str) -> str:
        parts = [str(x).strip("/") for x in parts if str(x)]
        return posixpath.join(self.root_key, *parts) if self.root_key else posixpath.join(*parts)

    def path(self, *parts: str) -> "RecordPath":
        return RecordPath(self, self._join_key(*parts))

    def from_key(self, key: str) -> "RecordPath":
        return RecordPath(self, key.rstrip("/"))

    def subtree(self, root_key: str) -> "RecordFS":
        clone = RecordFS.__new__(RecordFS)
        clone.fs = self.fs
        clone._state = self._state
        clone.root_key = root_key.rstrip("/")
        clone.storage_options = dict(self.storage_options)
        clone.keepalive = self.keepalive
        clone.is_local = self.is_local
        clone.root_url = clone.display(clone.root_key)
        return clone

    def _display_remote(self, key: str) -> str:
        if self._is_sftp and hasattr(self.fs, "host"):
            port = (getattr(self.fs, "ssh_kwargs", {}) or {}).get("port")
            authority = f"{self.fs.host}:{port}" if port else str(self.fs.host)
            return f"sftp://{authority}/{key.lstrip('/')}"
        return self.fs.unstrip_protocol(key)

    def display(self, key: str) -> str:
        return str(Path(key)) if self.is_local else self._display_remote(key)


@total_ordering
class RecordPath(FrozenRecordModel):
    """Small pathlib-like path wrapper backed by :class:`RecordFS`.

    The wrapper intentionally implements only the subset used by this module.
    This keeps existing record APIs path-shaped while allowing the same object
    model to work with object stores and other fsspec backends.
    """

    __positional_fields__ = ("storage", "key")

    storage: Any = Field(repr=False, exclude=True)
    key: str

    @field_validator("storage")
    @classmethod
    def _validate_storage(cls, value: Any) -> RecordFS:
        if not isinstance(value, RecordFS):
            raise ValueError("storage must be a RecordFS instance")
        return value

    def __str__(self) -> str:
        return self.storage.display(self.key)

    def __repr__(self) -> str:
        return f"RecordPath({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RecordPath):
            return self.key == other.key
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, RecordPath):
            return self.key < other.key
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.key)

    def __fspath__(self) -> str:
        if not self.storage.is_local:
            raise TypeError("Remote RecordPath does not have a local os.PathLike representation")
        return self.key

    def to_local_path(self) -> Path:
        """Return a pathlib.Path for local storage; reject remote paths."""
        return Path(os.fspath(self))

    def __truediv__(self, part: str | Path) -> "RecordPath":
        return RecordPath(self.storage, posixpath.join(self.key, str(part).strip("/")))

    @property
    def name(self) -> str:
        return posixpath.basename(self.key.rstrip("/"))

    @property
    def parent(self) -> "RecordPath":
        return RecordPath(self.storage, posixpath.dirname(self.key.rstrip("/")))
    
    @property
    def parents(self) -> tuple["RecordPath", ...]:
        """Return all parent paths, from nearest parent to filesystem root."""
        result: list[RecordPath] = []
        current = self

        while True:
            parent = current.parent

            # We reached the filesystem root.
            if parent.key == current.key:
                break

            result.append(parent)
            current = parent

        return tuple(result)
    
    @property
    def suffix(self) -> str:
        return posixpath.splitext(self.name)[1]

    @property
    def stem(self) -> str:
        return posixpath.splitext(self.name)[0]
    
    def with_suffix(self, suffix: str) -> "RecordPath":
        suffix = _validate_suffix(suffix)
        stem = posixpath.splitext(self.name)[0]
        return self.parent / f"{stem}{suffix}"

    def as_posix(self) -> str:
        return str(self)

    def exists(self) -> bool:
        return self.storage.fs.exists(self.key)

    def is_file(self) -> bool:
        return self.storage.fs.isfile(self.key)

    def is_dir(self) -> bool:
        return self.storage.fs.isdir(self.key)

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        # fsspec makedirs already behaves recursively; preserving the pathlib
        # flags keeps the caller-facing API familiar.
        if not exist_ok and self.exists():
            raise FileExistsError(str(self))
        if parents:
            self.storage.fs.makedirs(self.key, exist_ok=True)
        else:
            self.storage.fs.mkdir(self.key, create_parents=False)

    def open(self, mode: str = "r", encoding: str | None = None):
        kwargs: dict[str, Any] = {}
        if "b" not in mode and encoding is not None:
            kwargs["encoding"] = encoding
        return self.storage.fs.open(self.key, mode=mode, **kwargs)
    
    def write_text(
        self,
        data: str,
        encoding: str = "utf-8",
    ) -> int:
        """Write text to this path and return the number of characters written."""
        if not isinstance(data, str):
            raise TypeError(f"data must be str, not {type(data).__name__}")

        if encoding.lower().replace("-", "") != "utf8":
            # _atomic_write_text currently always uses UTF-8.
            with self.open("w", encoding=encoding) as handle:
                return handle.write(data)

        _atomic_write_text(self, data)
        return len(data)

    def iterdir(self) -> Iterator["RecordPath"]:
        if not self.is_dir():
            raise NotADirectoryError(str(self))
        entries = self.storage.fs.ls(self.key, detail=True)
        for entry in entries:
            key = entry["name"] if isinstance(entry, dict) else str(entry)
            yield RecordPath(self.storage, key.rstrip("/"))

    def glob(self, pattern: str) -> list["RecordPath"]:
        pattern_key = posixpath.join(self.key, pattern)
        return [
            RecordPath(self.storage, str(key).rstrip("/"))
            for key in self.storage.fs.glob(pattern_key)
        ]

    def relative_to(self, base: "RecordPath") -> "RelativeRecordPath":
        if self.storage is not base.storage:
            raise ValueError("Cannot compare paths from different filesystems")
        prefix = base.key.rstrip("/") + "/"
        if self.key == base.key:
            rel = "."
        elif self.key.startswith(prefix):
            rel = self.key[len(prefix):]
        else:
            raise ValueError(f"{self} is not under {base}")
        return RelativeRecordPath(rel)
    

class RelativeRecordPath(FrozenRecordModel):
    __positional_fields__ = ("value",)

    value: str

    def as_posix(self) -> str:
        return self.value


# RecordFS is defined before these models, so resolve its deferred annotation now.
RecordPath.model_rebuild()
RelativeRecordPath.model_rebuild()


# ---------------------------------------------------------------------------
# Validation and timestamp helpers
# ---------------------------------------------------------------------------


def _validate_path_component(value: str, *, name: str) -> str:
    """Validate a single safe filesystem path component."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if value in {".", ".."}:
        raise ValueError(f"{name} must not be {value!r}")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be a single path component: {value!r}")
    return value


def _validate_suffix(value: str, *, name: str = "suffix") -> str:
    if not value.startswith(".") or len(value) == 1:
        raise ValueError(f"{name} must look like '.ext': {value!r}")
    if "/" in value or "\\" in value:
        raise ValueError(f"{name} must not contain path separators: {value!r}")
    return value


def _parse_utc_date(value: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid UTC date {value!r}; expected YYYY-MM-DD") from exc
    return parsed


def _validate_record_id(record_id: str) -> re.Match[str]:
    match = RECORD_ID_RE.fullmatch(record_id)
    if match is None:
        raise ValueError(f"Invalid record_id: {record_id!r}")

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"Invalid record_id clock time: {record_id!r}")
    return match


def datetime_utc_from_timestamp_ns(timestamp_ns_utc: int) -> str:
    """Return an ISO-like UTC timestamp with nanosecond precision."""
    seconds, nsec = divmod(int(timestamp_ns_utc), NS_PER_SECOND)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{nsec:09d}Z"


def date_utc_from_timestamp_ns(timestamp_ns_utc: int) -> str:
    """Return the UTC calendar date (YYYY-MM-DD) for a nanosecond timestamp."""
    seconds = int(timestamp_ns_utc) // NS_PER_SECOND
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d")


def record_id_from_timestamp_ns(timestamp_ns_utc: int, sequence: int = 1) -> str:
    """Return ``HHMMSS.NNNNNNNNNJST`` for a UTC nanosecond timestamp.

    ``sequence`` is retained for source compatibility with the previous API.
    The on-disk record-id format has no sequence field, so it does not affect
    the generated ID.
    """
    if sequence < 0:
        raise ValueError("sequence must be non-negative")

    seconds, nsec = divmod(int(timestamp_ns_utc), NS_PER_SECOND)
    dt_jst = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(JST)
    return f"{dt_jst:%H%M%S}.{nsec:09d}JST"


def timestamp_ns_from_date_and_record_id(date_utc: str, record_id: str) -> int:
    """Reconstruct a UTC nanosecond timestamp from the record directory names.

    The directory date is UTC, while ``record_id`` contains a JST clock time.
    JST is nine hours ahead of UTC, so a JST clock time from 00:00 through
    08:59 belongs to the day *after* the UTC date directory. Times from 09:00
    through 23:59 belong to the same calendar date.
    """
    utc_date = _parse_utc_date(date_utc)
    match = _validate_record_id(record_id)

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    nsec = int(match.group("nsec"))

    jst_date = utc_date + timedelta(days=1 if hour < 9 else 0)
    dt_jst = datetime(
        jst_date.year,
        jst_date.month,
        jst_date.day,
        hour,
        minute,
        second,
        tzinfo=JST,
    )
    dt_utc = dt_jst.astimezone(timezone.utc)
    epoch_seconds = calendar.timegm(dt_utc.utctimetuple())
    timestamp_ns_utc = epoch_seconds * NS_PER_SECOND + nsec

    # Defensive invariant: the reconstructed timestamp must reproduce both
    # directory components exactly.
    if date_utc_from_timestamp_ns(timestamp_ns_utc) != date_utc:
        raise ValueError(
            f"date_utc={date_utc!r} and record_id={record_id!r} are inconsistent"
        )
    if record_id_from_timestamp_ns(timestamp_ns_utc) != record_id:
        raise ValueError(
            f"date_utc={date_utc!r} and record_id={record_id!r} are inconsistent"
        )

    return timestamp_ns_utc


def timestamp_info(timestamp_ns_utc: int, sequence: int = 1) -> tuple[str, str]:
    """Return ``(record_id, date_utc)`` for a UTC nanosecond timestamp."""
    timestamp_ns_utc = int(timestamp_ns_utc)
    return (
        record_id_from_timestamp_ns(timestamp_ns_utc, sequence),
        date_utc_from_timestamp_ns(timestamp_ns_utc),
    )


# ---------------------------------------------------------------------------
# Atomic/local and streaming/remote file helpers
# ---------------------------------------------------------------------------


PathLike = Path | RecordPath


def _atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Write bytes, preserving atomic replacement on local filesystems.

    Remote object stores do not generally provide POSIX rename semantics, so
    remote writes are uploaded directly. Use ``CustomRecord.commit()`` when a
    record-level visibility/commit marker is required.
    """
    if isinstance(path, RecordPath):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.storage.is_local:
            with path.open("wb") as handle:
                handle.write(data)
            return
        local_path = Path(path.key)
    else:
        local_path = Path(path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=local_path.parent, delete=False) as handle:
            tmp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(local_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_text(path: PathLike, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _open_binary_source(src: str | Path | RecordPath):
    if isinstance(src, RecordPath):
        return src.open("rb")
    src_text = str(src)
    if "://" in src_text:
        return fsspec.open(src_text, mode="rb").open()
    return Path(src).open("rb")


def _source_suffix(src: str | Path | RecordPath) -> str:
    if isinstance(src, RecordPath):
        return src.suffix.lower()
    return Path(str(src).split("?", 1)[0]).suffix.lower()


def _atomic_copy_file(src: str | Path | RecordPath, dst: PathLike) -> None:
    """Copy a local or fsspec-readable file to ``dst``."""
    if isinstance(src, RecordPath) and isinstance(dst, RecordPath):
        if src.storage is dst.storage and src.key == dst.key:
            return

    with _open_binary_source(src) as source:
        # Copying through bytes keeps the implementation backend-neutral. JPEG
        # frames are typically small; large PCD files are read through _read_pcd.
        _atomic_write_bytes(dst, source.read())


def _write_json(path: PathLike, data: Any) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(path, text)


def _read_json(path: PathLike) -> Any:
    path_obj = path if isinstance(path, RecordPath) else Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(str(path_obj))
    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rel(path: PathLike, base: PathLike) -> str:
    if isinstance(path, RecordPath) and isinstance(base, RecordPath):
        return path.relative_to(base).as_posix()
    return Path(path).relative_to(Path(base)).as_posix()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(CAMERA_IMAGE_EXTENSION, image)
    if not ok:
        raise IOError("OpenCV failed to encode JPEG image")
    return encoded.tobytes()


def _write_rgb_array(path: PathLike, image: Any) -> PathLike:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got dtype={array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3), got {array.shape}")

    image_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    _atomic_write_bytes(path, _encode_jpeg(image_bgr))
    return path


def _write_gray_array(path: PathLike, image: Any) -> PathLike:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 grayscale image, got dtype={array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale image shape (H, W), got {array.shape}")

    _atomic_write_bytes(path, _encode_jpeg(array))
    return path


def _write_encoded_jpeg(path: PathLike, image: Any) -> PathLike:
    """Write an encoded frame, local/remote image path, or Pillow image as JPEG."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        _atomic_write_bytes(path, bytes(image))
        return path

    if isinstance(image, (str, Path, RecordPath)):
        if _source_suffix(image) in JPEG_LIKE_SUFFIXES:
            _atomic_copy_file(image, path)
            return path

        try:
            with _open_binary_source(image) as source:
                encoded = np.frombuffer(source.read(), dtype=np.uint8)
        except FileNotFoundError:
            raise
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Could not decode image file: {image}")
        _atomic_write_bytes(path, _encode_jpeg(decoded))
        return path

    if hasattr(image, "save"):
        buffer = BytesIO()
        mode = getattr(image, "mode", None)
        if mode in {"RGBA", "LA", "P"}:
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        _atomic_write_bytes(path, buffer.getvalue())
        return path

    raise TypeError(
        "Unsupported image type. Pass encoded JPEG/MJPEG bytes, an existing image "
        "file path, a Pillow image, or a NumPy array through CustomRecord.add_image()."
    )

def _read_encoded_jpeg(path: str | Path | RecordPath) -> bytes:
    """Read an encoded JPEG/MJPEG frame from local or remote storage.

    The returned bytes are the original encoded data and can be:

    - sent directly through HTTP
    - passed to cv2.imdecode()
    - written to another JPEG file
    - returned as an MJPEG frame
    """
    with _open_binary_source(path) as source:
        data = source.read()

    if not data:
        raise ValueError(f"Empty JPEG file: {path}")

    return data

# ---------------------------------------------------------------------------
# PCD reader
# ---------------------------------------------------------------------------


def _pcd_scalar_dtype(type_code: str, size: int) -> np.dtype[Any]:
    lookup: dict[tuple[str, int], str] = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "<i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "<u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    try:
        return np.dtype(lookup[(type_code.upper(), size)])
    except KeyError as exc:
        raise ValueError(f"Unsupported PCD scalar type: TYPE={type_code!r}, SIZE={size}") from exc


def _read_pcd(path: str | Path | RecordPath) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ + packed RGB from a local or remote ASCII/binary PCD file."""
    path_obj: Path | RecordPath
    if isinstance(path, RecordPath):
        path_obj = path
    elif "://" in str(path):
        fs, key = fsspec.core.url_to_fs(str(path))
        temp_storage = RecordFS.__new__(RecordFS)
        temp_storage.fs = fs
        temp_storage.root_key = posixpath.dirname(key)
        temp_storage.root_url = fs.unstrip_protocol(temp_storage.root_key)
        temp_storage.storage_options = {}
        temp_storage.is_local = bool(getattr(fs, "local_file", False))
        path_obj = RecordPath(temp_storage, key)
    else:
        path_obj = Path(path)
    header: dict[str, list[str]] = {}

    with path_obj.open("rb") as handle:
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError("Invalid PCD file: DATA line not found")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("Invalid PCD header: expected ASCII text") from exc

            if not line or line.startswith("#"):
                continue

            key, *values = line.split()
            key = key.upper()
            header[key] = values
            if key == "DATA":
                if not values:
                    raise ValueError("Invalid PCD header: DATA has no encoding")
                data_encoding = values[0].lower()
                break

        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields:
            raise ValueError("Invalid PCD header: missing FIELDS")
        for required in ("x", "y", "z", "rgb"):
            if required not in fields:
                raise ValueError(f"Expected PCD field {required!r}; got {fields}")

        sizes = [int(v) for v in header.get("SIZE", [])]
        types = [v.upper() for v in header.get("TYPE", [])]
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError("Invalid PCD header: FIELDS/SIZE/TYPE/COUNT lengths differ")

        if "POINTS" in header:
            num_points = int(header["POINTS"][0])
        elif "WIDTH" in header and "HEIGHT" in header:
            num_points = int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
        else:
            raise ValueError("Invalid PCD header: missing POINTS or WIDTH/HEIGHT")

        for required in ("x", "y", "z", "rgb"):
            idx = fields.index(required)
            if counts[idx] != 1:
                raise ValueError(f"PCD field {required!r} must have COUNT 1")

        if data_encoding == "binary":
            dtype_fields: list[tuple[Any, ...]] = []
            for name, size, type_code, count in zip(fields, sizes, types, counts):
                scalar_dtype = _pcd_scalar_dtype(type_code, size)
                if count == 1:
                    dtype_fields.append((name, scalar_dtype))
                else:
                    dtype_fields.append((name, scalar_dtype, (count,)))

            raw = handle.read()
            data = np.frombuffer(raw, dtype=np.dtype(dtype_fields), count=num_points)
            if len(data) != num_points:
                raise ValueError(
                    f"PCD contains {len(data)} binary points; expected {num_points}"
                )
            points_m = np.column_stack((data["x"], data["y"], data["z"])).astype(
                np.float32, copy=False
            )
            rgb_values = np.asarray(data["rgb"])
        elif data_encoding == "ascii":
            data = np.loadtxt(handle, dtype=np.float64, ndmin=2)
            if data.shape[0] != num_points:
                raise ValueError(
                    f"PCD contains {data.shape[0]} ASCII points; expected {num_points}"
                )
            field_index = {name: index for index, name in enumerate(fields)}
            points_m = data[
                :, [field_index["x"], field_index["y"], field_index["z"]]
            ].astype(np.float32)
            rgb_values = data[:, field_index["rgb"]]
        else:
            raise ValueError(f"Unsupported PCD DATA format: {data_encoding!r}")

    rgb_index = fields.index("rgb")
    rgb_type = types[rgb_index]
    rgb_size = sizes[rgb_index]
    if rgb_size != 4:
        raise ValueError(f"Packed RGB field must be 4 bytes; got SIZE={rgb_size}")

    if rgb_type == "F":
        packed = np.asarray(rgb_values, dtype=np.float32).view(np.uint32)
    elif rgb_type == "U":
        packed = np.asarray(rgb_values, dtype=np.uint32)
    elif rgb_type == "I":
        packed = np.asarray(rgb_values, dtype=np.int32).view(np.uint32)
    else:
        raise ValueError(f"Unsupported packed RGB TYPE={rgb_type!r}")

    colors_rgb = np.column_stack(
        ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)
    ).astype(np.uint8)
    return points_m, colors_rgb


# ---------------------------------------------------------------------------
# Record components
# ---------------------------------------------------------------------------


class ImageRecord(RecordModel):
    """One camera image within a record."""

    __positional_fields__ = ("parent", "filename")

    parent: CameraRecord | None = Field(default=None, repr=False, exclude=True)
    filename: str = "null.jpg"

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        return _validate_path_component(value, name="filename")

    def expected_path(self) -> RecordPath:
        if self.parent is None:
            raise ValueError("ImageRecord.parent is not set")
        return self.parent.expected_path() / self.filename

    def exists(self) -> bool:
        return self.expected_path().is_file()

    def write_rgb(self, image: Any) -> RecordPath:
        return _write_rgb_array(self.expected_path(), image)

    def write_gray(self, image: Any) -> RecordPath:
        return _write_gray_array(self.expected_path(), image)

    def write_mjpeg_frame(self, data: Any) -> RecordPath:
        return _write_encoded_jpeg(self.expected_path(), data)

    def load(self):
        return _read_encoded_jpeg(self.expected_path())

    def yolo_record(self) -> YoloRecord:
        return YoloRecord(img_parent=self)


class CameraRecord(RecordModel):
    """Filesystem view of one camera inside a capture record."""

    __positional_fields__ = ("parent", "camera_id", "calib")

    parent: CustomRecord = Field(repr=False, exclude=True)
    camera_id: str = "cam_a"
    calib: CalibrationRecord | None = Field(default=None, repr=False, exclude=True)
    images: dict[str, ImageRecord] = Field(default_factory=dict, repr=False, exclude=True)

    @field_validator("camera_id")
    @classmethod
    def _validate_camera_id(cls, value: str) -> str:
        return _validate_path_component(value, name="camera_id")

    def model_post_init(self, __context: Any) -> None:
        self.images = {
            stream: ImageRecord(parent=self, filename=f"{stream}{CAMERA_IMAGE_EXTENSION}")
            for stream in cast(Sequence[str], sorted(VALID_IMAGE_STREAMS))
        }

    def expected_parent_path(self) -> RecordPath:
        return self.parent.expected_image_path

    def expected_path(self) -> RecordPath:
        return self.expected_parent_path() / self.camera_id

    def expected_image_path(self, name: str) -> RecordPath:
        try:
            image = self.images[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown image type {name!r}; available: {sorted(self.images)}"
            ) from exc
        return image.expected_path()

    def expected_rgb_path(self) -> RecordPath:
        return self.expected_image_path("rgb")

    def expected_left_path(self) -> RecordPath:
        return self.expected_image_path("left")

    def expected_right_path(self) -> RecordPath:
        return self.expected_image_path("right")

    def mkdirs(self) -> RecordPath:
        path = self.expected_path()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_image(self, name: str) -> bool:
        return self.expected_image_path(name).is_file()

    def has_rgb(self) -> bool:
        return self.has_image("rgb")

    def has_left(self) -> bool:
        return self.has_image("left")

    def has_right(self) -> bool:
        return self.has_image("right")

    def add_calibration(self, data: Mapping[str, Any]) -> RecordPath:
        self.calib = CalibrationRecord(parent=self.parent, cam_parent=self, data=dict(data))
        return self.calib.write()

    def get_calibration(self):
        self.calib = CalibrationRecord(parent=self.parent, cam_parent=self)
        return self.calib.load()


class CalibrationRecord(RecordModel):
    """Calibration JSON associated with one camera."""

    __positional_fields__ = ("parent", "cam_parent", "data")

    parent: CustomRecord = Field(repr=False, exclude=True)
    cam_parent: CameraRecord = Field(repr=False, exclude=True)
    data: dict[str, Any] = Field(default_factory=dict)

    def expected_data_path(self) -> RecordPath:
        return self.cam_parent.parent.expected_calib_path / f"{self.cam_parent.camera_id}.json"

    def exists(self) -> bool:
        return self.expected_data_path().is_file()

    def load(self) -> dict[str, Any]:
        data = _read_json(self.expected_data_path())
        if not isinstance(data, dict):
            raise ValueError("Calibration JSON root must be an object")
        self.data = data
        return data

    def write(self, data: Mapping[str, Any] | None = None) -> RecordPath:
        if data is not None:
            self.data = dict(data)
        path = self.expected_data_path()
        _write_json(path, self.data)
        return path


class GnssRecord(RecordModel):
    """One GNSS JSON file within a capture record."""

    __positional_fields__ = ("parent", "kind", "suffix")

    parent: CustomRecord = Field(repr=False, exclude=True)
    kind: str = "baselink"
    suffix: str = ".json"

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        return _validate_path_component(value, name="kind")

    @field_validator("suffix")
    @classmethod
    def _validate_suffix_field(cls, value: str) -> str:
        return _validate_suffix(value)

    def expected_root_path(self) -> RecordPath:
        return self.parent.expected_gnss_path

    def expected_path(self) -> RecordPath:
        return self.expected_root_path() / f"{self.kind}{self.suffix}"

    def exists(self) -> bool:
        return self.expected_path().is_file()

    def mkdirs(self) -> RecordPath:
        path = self.expected_root_path()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, data: Any) -> RecordPath:
        path = self.expected_path()
        _write_json(path, data)
        return path

    def load(self) -> Any:
        return _read_json(self.expected_path())


class ArmRecord(RecordModel):
    """One Arm JSON file within a capture record."""

    __positional_fields__ = ("parent", "kind", "suffix")

    parent: CustomRecord = Field(repr=False, exclude=True)
    kind: str = "arm_result"
    suffix: str = ".json"

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        return _validate_path_component(value, name="kind")

    @field_validator("suffix")
    @classmethod
    def _validate_suffix_field(cls, value: str) -> str:
        return _validate_suffix(value)

    def expected_root_path(self) -> RecordPath:
        return self.parent.expected_arm_path

    def exists(self) -> bool:
        return self.expected_path().is_file()

    def mkdirs(self) -> RecordPath:
        path = self.expected_root_path()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_result(self, run_id: str, data: Dict):
        run_dir = self.expected_root_path() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{run_id}_{self.kind}{self.suffix}"
        _write_json(path, data)

    def get_result(self, run_id: str) -> Dict:
        path = (
            self.expected_root_path()
            / run_id
            / f"{run_id}_{self.kind}{self.suffix}"
        )
        return _read_json(path)


    def list_results(self) -> List[Dict]:
        root = self.expected_root_path()
        results = []
        if not root.exists(): return results

        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir(): continue

            run_id = run_dir.name
            result_path = run_dir / f"{run_id}_{self.kind}{self.suffix}"
            if not result_path.is_file(): continue

            results.append(self.get_result(run_id))

        return results
    

PCD_SEGMENT_RE = re.compile(
    r"^(?P<obj_id>\d+)"
    r"_class(?P<class_id>\d+)"
    r"_(?P<class_name>.+?)"
    r"_(?P<score>\d+(?:\.\d+)?)"
    r"_(?P<num_points>\d+)pts"
    r"\.pcd$",
    re.IGNORECASE,
)


class PCDSegment(RecordModel):
    """A segmented point cloud whose metadata is encoded in its filename."""

    __positional_fields__ = ("filepath",)

    filepath: RecordPath | Path | str
    obj_id: int = -1
    class_id: int = -1
    class_name: str = "null"
    score: float = -1.0
    num_points: int = -1

    @model_validator(mode="after")
    def _parse_filename_metadata(self) -> "PCDSegment":
        filename = (
            self.filepath.name
            if hasattr(self.filepath, "name")
            else posixpath.basename(str(self.filepath))
        )
        match = PCD_SEGMENT_RE.fullmatch(filename)
        if match is None:
            self.obj_id = -1
            self.class_id = -1
            self.class_name = "null"
            self.score = -1.0
            self.num_points = -1
            return self
        self.obj_id = int(match.group("obj_id"))
        self.class_id = int(match.group("class_id"))
        self.class_name = match.group("class_name")
        self.score = float(match.group("score"))
        self.num_points = int(match.group("num_points"))
        return self

    def is_valid(self) -> bool:
        return self.obj_id >= 0

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return _read_pcd(self.filepath)


class PCDRecord(RecordModel):
    """Filesystem view of a full point cloud and optional segmented clouds."""

    __positional_fields__ = ("parent", "source_name", "kind", "suffix", "encode")

    parent: CustomRecord = Field(repr=False, exclude=True)
    source_name: str = "cam_a"
    kind: PCDKind = "file"
    suffix: str = ".pcd"
    encode: PCDEncoding = "binary"

    @field_validator("source_name")
    @classmethod
    def _validate_source_name(cls, value: str) -> str:
        return _validate_path_component(value, name="source_name")

    @field_validator("suffix")
    @classmethod
    def _validate_suffix_field(cls, value: str) -> str:
        return _validate_suffix(value)

    def expected_parent_path(self) -> RecordPath:
        return self.parent.expected_pcd_path

    def expected_full_pcd_path(self) -> RecordPath:
        if self.kind == "file":
            return self.expected_parent_path() / f"{self.source_name}{self.suffix}"
        return self.expected_parent_path() / self.source_name / f"full{self.suffix}"

    def exists(self) -> bool:
        return self.expected_full_pcd_path().is_file()

    def mkdirs(self) -> RecordPath:
        path = self.expected_full_pcd_path().parent
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_pcd_segments(self, class_name=None, score=None) -> list[PCDSegment]:
        if self.kind != "folder":
            raise ValueError("PCD segments are only available when kind='folder'")

        folder = self.expected_parent_path() / self.source_name
        if not folder.is_dir():
            return []

        segments = [PCDSegment(path) for path in folder.glob(f"*{self.suffix}")]
        valid_segments = [segment for segment in segments if segment.is_valid()]
        if class_name:
            valid_segments = [
                segment for segment in valid_segments if segment.class_name == class_name
            ]
        if score is not None:
            valid_segments = [segment for segment in valid_segments if segment.score >= score]
        return sorted(valid_segments, key=lambda segment: segment.obj_id)

    def get_pcd_segment_paths(self) -> list[RecordPath | Path | str]:
        return [segment.filepath for segment in self.get_pcd_segments()]

    def num_segments(self) -> int:
        return len(self.get_pcd_segments())

    def has_segments(self) -> bool:
        return self.num_segments() > 0


class YoloRecord(RecordModel):
    """YOLO detection metadata associated with one image."""

    __positional_fields__ = ("img_parent", "parent", "data")

    img_parent: ImageRecord = Field(repr=False, exclude=True)
    parent: CustomRecord | None = Field(default=None, repr=False, exclude=True)
    data: dict[str, Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.parent is None and self.img_parent.parent is not None:
            self.parent = self.img_parent.parent.parent

    def expected_image_path(self) -> RecordPath:
        return self.img_parent.expected_path()

    def expected_data_path(self) -> RecordPath:
        if self.img_parent.parent is None:
            raise ValueError("ImageRecord.parent is not set")
        camera = self.img_parent.parent
        return (
            camera.parent.expected_yolo_path
            / camera.camera_id
            / f"{posixpath.splitext(self.img_parent.filename)[0]}.json"
        )

    def exists(self) -> bool:
        return self.expected_data_path().is_file()

    def load(self) -> dict[str, Any]:
        data = _read_json(self.expected_data_path())
        if not isinstance(data, dict):
            raise ValueError("YOLO JSON root must be an object")
        self.data = data
        return data

    def write(self, data: Mapping[str, Any] | None = None) -> RecordPath:
        if data is not None:
            self.data = dict(data)
        if self.data is None:
            raise ValueError("No YOLO data to write")
        path = self.expected_data_path()
        _write_json(path, self.data)
        return path

    def get_detections(self, class_name=None, score=None) -> list[Any]:
        if self.data is None:
            self.load()
        assert self.data is not None
        detections = self.data.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError("YOLO 'detections' must be a list")
        if class_name:
            detections = [det for det in detections if det.get("class_name") == class_name]
        if score is not None:
            detections = [
                det
                for det in detections
                if det.get("confidence", float("-inf")) >= score
            ]
        return detections

    def get_bbox_xyxy(self, class_name=None, score=None) -> list[Any]:
        detections = self.get_detections(class_name=class_name, score=score)
        return [item.get("bbox_xyxy", []) for item in detections if isinstance(item, dict)]

    def get_mask(self, class_name=None, score=None) -> list[Any]:
        detections = self.get_detections(class_name=class_name, score=score)
        return [item.get("mask", []) for item in detections if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Capture record and store
# ---------------------------------------------------------------------------


class CustomRecord(RecordModel):
    """One synchronized single-frame capture record on local or remote storage."""

    __positional_fields__ = (
        "root_path",
        "mode",
        "field_id",
        "record_id",
        "timestamp_ns_utc",
        "date_utc",
        "storage_options",
        "_storage",
    )

    root_path: Path | str
    mode: RecordMode
    field_id: str
    record_id: str
    timestamp_ns_utc: int
    date_utc: str | None = None
    storage_options: dict[str, Any] = Field(default_factory=dict, repr=False)
    storage_backend: Any | None = Field(
        default=None,
        alias="_storage",
        repr=False,
        exclude=True,
    )

    @field_validator("storage_backend")
    @classmethod
    def _validate_storage_backend(cls, value: Any | None) -> RecordFS | None:
        if value is not None and not isinstance(value, RecordFS):
            raise ValueError("_storage must be a RecordFS instance or None")
        return value

    @field_validator("field_id")
    @classmethod
    def _validate_field_id(cls, value: str) -> str:
        return _validate_path_component(value, name="field_id")

    @field_validator("record_id")
    @classmethod
    def _validate_record_id_field(cls, value: str) -> str:
        _validate_record_id(value)
        return value

    @model_validator(mode="after")
    def _validate_timestamp_layout(self) -> "CustomRecord":
        expected_date = date_utc_from_timestamp_ns(self.timestamp_ns_utc)
        if self.date_utc is None:
            self.date_utc = expected_date
        else:
            _parse_utc_date(self.date_utc)
            if self.date_utc != expected_date:
                raise ValueError(
                    f"date_utc={self.date_utc!r} does not match timestamp date "
                    f"{expected_date!r}"
                )

        expected_record_id = record_id_from_timestamp_ns(self.timestamp_ns_utc)
        if self.record_id != expected_record_id:
            raise ValueError(
                f"record_id={self.record_id!r} does not match timestamp; "
                f"expected {expected_record_id!r}"
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        if self.storage_backend is None:
            self.storage_backend = RecordFS(self.root_path, self.storage_options)
        else:
            self.storage_options = dict(self.storage_backend.storage_options)
        self.root_path = self.storage_backend.root_url

    @property
    def _storage(self) -> RecordFS | None:
        """Compatibility alias for the previous dataclass's internal field."""
        return self.storage_backend

    @_storage.setter
    def _storage(self, value: RecordFS | None) -> None:
        self.storage_backend = value

    @property
    def storage(self) -> RecordFS:
        assert self._storage is not None
        return self._storage

    @staticmethod
    def empty() -> "CustomRecord":
        timestamp_ns_utc = 0
        return CustomRecord(
            root_path=Path("."),
            mode="dual_rgb",
            field_id="null",
            record_id=record_id_from_timestamp_ns(timestamp_ns_utc),
            timestamp_ns_utc=timestamp_ns_utc,
        )

    def is_empty(self) -> bool:
        return self.field_id == "null"

    @property
    def path(self) -> RecordPath:
        assert self.date_utc is not None
        return self.storage.path(self.mode, self.date_utc, self.field_id, self.record_id)

    @property
    def datetime_utc(self) -> str:
        return datetime_utc_from_timestamp_ns(self.timestamp_ns_utc)

    def to_dict(
        self,
        *,
        root_path: str | Path | None = None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a complete JSON-safe wire representation of this record.

        The payload contains both the filesystem URL and ``storage_options`` so a
        remote client can reconstruct a working :class:`CustomRecord` directly
        with ``CustomRecord.from_dict(payload)``.

        ``root_path`` and ``storage_options`` may be overridden by a REST server
        that discovers records locally but wants clients to access them through a
        different backend, for example SFTP.
        """
        wire_root = str(self.root_path if root_path is None else root_path)
        wire_storage_options = dict(
            self.storage_options if storage_options is None else storage_options
        )
        return {
            "wire_schema": WIRE_SCHEMA,
            "object_type": "CustomRecord",
            "root_path": wire_root,
            "storage_options": wire_storage_options,
            "mode": self.mode,
            "field_id": self.field_id,
            "record_id": self.record_id,
            "timestamp_ns_utc": self.timestamp_ns_utc,
            "date_utc": self.date_utc,
            "datetime_utc": self.datetime_utc,
        }

    def to_json(
        self,
        *,
        root_path: str | Path | None = None,
        storage_options: Mapping[str, Any] | None = None,
        indent: int | None = None,
    ) -> str:
        """Serialize :meth:`to_dict` as JSON text, including connection info."""
        return json.dumps(
            self.to_dict(root_path=root_path, storage_options=storage_options),
            ensure_ascii=False,
            indent=indent,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        root_path: str | Path | None = None,
        storage_options: Mapping[str, Any] | None = None,
        _storage: RecordFS | None = None,
    ) -> "CustomRecord":
        """Reconstruct a working record from its REST/wire representation.

        By default both ``root_path`` and ``storage_options`` are read from the
        payload, so ``CustomRecord.from_dict(record.to_dict())`` is a complete
        round trip. Explicit keyword arguments override the values in the payload.
        """
        if not isinstance(data, Mapping):
            raise TypeError("CustomRecord JSON payload must be an object")

        wire_schema = data.get("wire_schema")
        if wire_schema is not None and wire_schema != WIRE_SCHEMA:
            raise ValueError(
                f"Unsupported CustomRecord wire schema: {wire_schema!r}; "
                f"expected {WIRE_SCHEMA!r}"
            )

        object_type = data.get("object_type")
        if object_type is not None and object_type != "CustomRecord":
            raise ValueError(
                f"Expected object_type='CustomRecord'; got {object_type!r}"
            )

        required = (
            "mode",
            "field_id",
            "record_id",
            "timestamp_ns_utc",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "CustomRecord JSON payload is missing required fields: "
                + ", ".join(missing)
            )

        selected_root = root_path if root_path is not None else data.get("root_path")
        if selected_root is None:
            raise ValueError(
                "CustomRecord JSON payload has no root_path; provide root_path=... "
                "on the client"
            )

        date_value = data.get("date_utc")
        if date_value is not None and not isinstance(date_value, str):
            raise ValueError("date_utc must be a string or null")

        payload_storage_options = data.get("storage_options", {})
        if payload_storage_options is None:
            payload_storage_options = {}
        if not isinstance(payload_storage_options, Mapping):
            raise ValueError("storage_options must be a JSON object")
        selected_storage_options = (
            dict(payload_storage_options)
            if storage_options is None
            else dict(storage_options)
        )

        return cls(
            root_path=selected_root,
            mode=cast(RecordMode, str(data["mode"])),
            field_id=str(data["field_id"]),
            record_id=str(data["record_id"]),
            timestamp_ns_utc=int(data["timestamp_ns_utc"]),
            date_utc=date_value,
            storage_options=selected_storage_options,
            _storage=_storage,
        )

    @classmethod
    def from_json(
        cls,
        text: str | bytes | bytearray,
        *,
        root_path: str | Path | None = None,
        storage_options: Mapping[str, Any] | None = None,
        _storage: RecordFS | None = None,
    ) -> "CustomRecord":
        """Reconstruct a working record from JSON text."""
        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode("utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("CustomRecord JSON must contain one JSON object")
        return cls.from_dict(
            data,
            root_path=root_path,
            storage_options=storage_options,
            _storage=_storage,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path | RecordPath,
        storage_options: Mapping[str, Any] | None = None,
    ) -> "CustomRecord":
        if isinstance(path, RecordPath):
            record_path = path
        else:
            path_text = str(path)
            if "://" not in path_text:
                path_text = str(Path(path_text).absolute())
            fs, key = fsspec.core.url_to_fs(path_text, **dict(storage_options or {}))
            protocol = fs.protocol
            protocols = set(protocol) if isinstance(protocol, (tuple, list)) else {protocol}
            is_local = bool({"file", "local"} & protocols) or bool(
                getattr(fs, "local_file", False)
            )
            temp = RecordFS.__new__(RecordFS)
            temp.fs = fs
            temp.root_key = key.rstrip("/")
            temp.storage_options = dict(storage_options or {})
            temp.is_local = is_local
            temp.root_url = temp.display(temp.root_key)
            record_path = RecordPath(temp, key.rstrip("/"))

        # <root>/<mode>/<date>/<field_id>/<record_id>
        record_id = record_path.name
        field_path = record_path.parent
        date_path = field_path.parent
        mode_path = date_path.parent
        root_path_obj = mode_path.parent

        field_id = field_path.name
        date_utc = date_path.name
        mode_text = mode_path.name
        if not all((record_id, field_id, date_utc, mode_text)):
            raise ValueError(f"Path is too short to be a record path: {path}")
        if mode_text not in VALID_MODES:
            raise ValueError(f"Unsupported mode in path: {mode_text!r}")

        timestamp_ns_utc = timestamp_ns_from_date_and_record_id(date_utc, record_id)
        storage = record_path.storage.subtree(root_path_obj.key)
        return cls(
            root_path=storage.root_url,
            mode=cast(RecordMode, mode_text),
            field_id=field_id,
            record_id=record_id,
            timestamp_ns_utc=timestamp_ns_utc,
            date_utc=date_utc,
            storage_options=dict(storage.storage_options),
            _storage=storage,
        )
    
    @property
    def expected_calib_path(self) -> RecordPath:
        return self.path / "calib"

    @property
    def expected_image_path(self) -> RecordPath:
        return self.path / "imgs"

    @property
    def expected_depth_path(self) -> RecordPath:
        return self.path / "depth"

    @property
    def expected_pcd_path(self) -> RecordPath:
        return self.path / "pcd"

    @property
    def expected_yolo_path(self) -> RecordPath:
        return self.path / "yolo"

    @property
    def expected_gnss_path(self) -> RecordPath:
        return self.path / "gnss"
    
    @property
    def expected_arm_path(self) -> RecordPath:
        return self.path / "arm"

    @property
    def expected_commit_path(self) -> RecordPath:
        return self.path / "_SUCCESS"

    @property
    def camera_paths(self) -> list[RecordPath]:
        if not self.expected_image_path.is_dir():
            return []
        return sorted(
            (path for path in self.expected_image_path.iterdir() if path.is_dir()),
            key=lambda path: path.key,
        )

    def image_paths(self, stream: ImageStream) -> list[RecordPath]:
        if stream not in VALID_IMAGE_STREAMS:
            raise ValueError(f"Unsupported image stream: {stream!r}")
        return [path / f"{stream}{CAMERA_IMAGE_EXTENSION}" for path in self.camera_paths]

    # Compatibility aliases for the previous public property names.
    @property
    def listup_cam_paths(self) -> list[RecordPath]:
        return self.camera_paths

    @property
    def listup_rgb_image_paths(self) -> list[RecordPath]:
        return self.image_paths("rgb")

    @property
    def listup_rgb_image_parent_paths(self) -> list[RecordPath]:
        return [path.parent for path in self.listup_rgb_image_paths]

    @property
    def listup_left_image_paths(self) -> list[RecordPath]:
        return self.image_paths("left")

    @property
    def listup_left_image_parent_paths(self) -> list[RecordPath]:
        return [path.parent for path in self.listup_left_image_paths]

    @property
    def listup_right_image_paths(self) -> list[RecordPath]:
        return self.image_paths("right")

    @property
    def expected_yolo_rgb_paths(self) -> list[RecordPath]:
        return [self.expected_yolo_path / path.name / "rgb.json" for path in self.camera_paths]

    def mkdirs(self) -> RecordPath:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def rel(self, path: RecordPath | Path) -> str:
        return _rel(path, self.path)

    @property
    def is_remote(self) -> bool:
        return not self._storage.is_local
    
    def bind_as_sftp(self, host: str, prefix: str = "",
        storage_options: Mapping[str, Any] | None = None,
    ) -> "CustomRecord":
        """Return the same logical record bound to an SFTP backend."""
        if self.is_remote:return self
        # is_win = str(self.root_path).contains(":\\")

        host = host.removeprefix("sftp://").rstrip("/")
        prefix = prefix.strip("/")
        local_root = str(self.root_path).lstrip("/")
        parts = [p for p in (prefix, local_root) if p]
        remote_root = "/".join(parts)
        root_url = f"sftp://{host}/{remote_root}"

        return CustomRecord(
            root_path=root_url,
            mode=self.mode,
            field_id=self.field_id,
            record_id=self.record_id,
            timestamp_ns_utc=self.timestamp_ns_utc,
            date_utc=self.date_utc,
            storage_options=dict(storage_options or {}),
        )
    
    def bind_as_local(self) -> "CustomRecord":
        """Return the same logical record bound to a local filesystem."""
        if not self.is_remote: return self        
        remote_path = self.root_path
        local_root = "/" + "/".join(remote_path.split("://")[1].split("/")[1:])
        return CustomRecord(root_path=local_root,
            mode=self.mode,
            field_id=self.field_id,
            record_id=self.record_id,
            timestamp_ns_utc=self.timestamp_ns_utc,
            date_utc=self.date_utc,
            storage_options={},
        )

    def commit(self) -> RecordPath:
        """Mark the record complete after all component uploads succeed.

        Object stores do not have POSIX atomic directory rename.  A small
        ``_SUCCESS`` object gives readers a record-level commit boundary.
        """
        _write_json(
            self.expected_commit_path,
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp_ns_utc": self.timestamp_ns_utc,
                "datetime_utc": self.datetime_utc,
            },
        )
        return self.expected_commit_path

    def is_committed(self) -> bool:
        return self.expected_commit_path.is_file()

    def get_camera(self, camera_id: str) -> CameraRecord:
        return CameraRecord(parent=self, camera_id=camera_id)

    def get_camera_list(self) -> List[CameraRecord]:
        return [CameraRecord(parent=self, camera_id=i.name) for i in self.listup_cam_paths]

    def get_yolo_list(self) -> list[YoloRecord]:
        imgs: list[ImageRecord] = []
        for camera in self.get_camera_list():
            imgs.extend(camera.images.values())
        records = [YoloRecord(parent=self, img_parent=image) for image in imgs]
        records = [record for record in records if record.exists()]
        for record in records: record.load()
        return records

    def get_pcd(self, camera_id: str, kind: PCDKind = "folder") -> PCDRecord:
        return PCDRecord(parent=self, source_name=camera_id, kind=kind)

    def get_pcd_list(self) -> List[PCDRecord]:
        pcds_fs = [
            PCDRecord(parent=self, source_name=i.name, kind="file")
            for i in self.listup_cam_paths
        ]
        pcds_dirs = [
            PCDRecord(parent=self, source_name=i.name, kind="folder")
            for i in self.listup_cam_paths
        ]
        return [p for p in pcds_fs + pcds_dirs if p.exists()]

    def get_calibration(self, camera_id: str) -> Dict:
        return self.get_camera(camera_id).get_calibration()
    
    def add_calibration(self, camera_id: str, data: Mapping[str, Any]) -> RecordPath:
        return self.get_camera(camera_id).add_calibration(data)

    def add_gnss(self, data: Any, kind: str = "baselink") -> RecordPath:
        return GnssRecord(parent=self, kind=kind).write(data)
        
    def get_gnss(self, kind: str = "baselink"):
        return GnssRecord(parent=self, kind=kind).load()
            
    def get_arm(self):
        return ArmRecord(parent=self)

    def add_image(self, camera_id: str, stream: ImageStream, image: Any) -> RecordPath:
        if stream not in VALID_IMAGE_STREAMS:
            raise ValueError(f"Unsupported image stream: {stream!r}")

        image_record = self.get_camera(camera_id).images[stream]
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return image_record.write_gray(image)
            if image.ndim == 3:
                return image_record.write_rgb(image)
            raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
        return image_record.write_mjpeg_frame(image)

    def add_mjpeg_image(
        self, camera_id: str, stream: ImageStream, image_bytes: Any
    ) -> RecordPath:
        """Backward-compatible alias for :meth:`add_image`."""
        return self.add_image(camera_id, stream, image_bytes)

    def get_image_bytes(self, camera_id: str, stream: ImageStream):
        image_record = self.get_camera(camera_id).images[stream]
        return image_record.load()
        
    def add_yolo(self, camera_id: str, stream: ImageStream, data: Mapping[str, Any]) -> RecordPath:
        image = self.get_camera(camera_id).images[stream]
        return YoloRecord(parent=self, img_parent=image, data=dict(data)).write()

    def get_yolo(self, camera_id: str, stream: ImageStream):
        image = self.get_camera(camera_id).images[stream]
        return YoloRecord(parent=self, img_parent=image).load()

class CustomStore(RecordModel):
    """Store for synchronized capture records on local or fsspec storage."""

    __positional_fields__ = ("root_path", "storage_options", "committed_only")

    root_path: Path | str
    storage_options: dict[str, Any] = Field(default_factory=dict, repr=False)
    committed_only: bool = False
    _storage: RecordFS = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._storage = RecordFS(self.root_path, self.storage_options)
        self.root_path = self._storage.root_url

    @property
    def storage(self) -> RecordFS:
        return self._storage

    @property
    def is_remote(self) -> bool:
        return not self._storage.is_local

    def record_from_dict(self, data: Mapping[str, Any]) -> CustomRecord:
        """Reconstruct a REST record using this store's root and credentials.

        Reusing the store's :class:`RecordFS` also reuses the underlying fsspec
        filesystem/connection, which is preferable when decoding many records.
        The payload's advertised ``root_path`` is intentionally overridden by
        this store.
        """
        return CustomRecord.from_dict(
            data,
            root_path=self.root_path,
            storage_options=self.storage_options,
            _storage=self._storage,
        )

    def record_from_json(self, text: str | bytes | bytearray) -> CustomRecord:
        """JSON-text counterpart of :meth:`record_from_dict`."""
        return CustomRecord.from_json(
            text,
            root_path=self.root_path,
            storage_options=self.storage_options,
            _storage=self._storage,
        )

    def records_from_dicts(
        self, data: Sequence[Mapping[str, Any]]
    ) -> list[CustomRecord]:
        """Reconstruct many REST records while sharing one filesystem."""
        return [self.record_from_dict(item) for item in data]

    def add_record(
        self,
        mode: RecordMode,
        timestamp_ns_utc: int,
        field_id: str = "field_all",
        sequence: int = 1,
        exist_ok: bool = False,
    ) -> CustomRecord:
        record_id, date_utc = timestamp_info(timestamp_ns_utc, sequence)
        record = CustomRecord(
            root_path=self.root_path,
            mode=mode,
            field_id=field_id,
            record_id=record_id,
            timestamp_ns_utc=timestamp_ns_utc,
            date_utc=date_utc,
            storage_options=dict(self.storage_options),
            _storage=self._storage,
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
        committed_only: bool | None = None,
    ) -> CustomRecord:
        path = self._storage.path(mode, date, field_id, record_id)
        if not path.is_dir():
            raise FileNotFoundError(f"Record does not exist: {path}")
        record = CustomRecord.from_path(path)
        require_commit = self.committed_only if committed_only is None else committed_only
        if require_commit and not record.is_committed():
            raise FileNotFoundError(f"Record exists but is not committed: {path}")
        return record

    def list_records(
        self,
        mode: RecordMode | None = None,
        date: str | None = None,
        field_id: str | None = "field_all",
        committed_only: bool | None = None,
    ) -> list[RecordPath]:
        """List leaf record directories, optionally filtering committed records."""
        if mode is not None and mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode!r}")
        if date is not None:
            _parse_utc_date(date)
        if field_id is not None:
            _validate_path_component(field_id, name="field_id")

        require_commit = self.committed_only if committed_only is None else committed_only
        modes = [mode] if mode is not None else sorted(VALID_MODES)
        results: list[RecordPath] = []

        for mode_name in modes:
            mode_root = self._storage.path(cast(str, mode_name))
            if not mode_root.is_dir():
                continue

            date_dirs = [mode_root / date] if date else sorted(
                (path for path in mode_root.iterdir() if path.is_dir()),
                key=lambda path: path.key,
            )
            for date_dir in date_dirs:
                if not date_dir.is_dir():
                    continue

                field_dirs = [date_dir / field_id] if field_id else sorted(
                    (path for path in date_dir.iterdir() if path.is_dir()),
                    key=lambda path: path.key,
                )
                for field_dir in field_dirs:
                    if not field_dir.is_dir():
                        continue
                    for record_dir in field_dir.iterdir():
                        if not record_dir.is_dir() or not RECORD_ID_RE.fullmatch(record_dir.name):
                            continue
                        if require_commit and not (record_dir / "_SUCCESS").is_file():
                            continue
                        results.append(record_dir)

        return sorted(results, key=lambda path: path.key)

    def list_record_objects(
        self,
        mode: RecordMode | None = None,
        date: str | None = None,
        field_id: str | None = "field_all",
        committed_only: bool | None = None,
    ) -> list[CustomRecord]:
        """Return :class:`CustomRecord` objects for all matching directories."""
        return [
            CustomRecord.from_path(path)
            for path in self.list_records(
                mode=mode,
                date=date,
                field_id=field_id,
                committed_only=committed_only,
            )
        ]

    def find_record_objects(
        self,
        mode: RecordMode,
        start_time_ns: int,
        end_time_ns: int,
        field_id: str = "field_all",
        committed_only: bool | None = None,
    ) -> list[CustomRecord]:
        """Return records whose UTC timestamps fall within an inclusive range."""
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode!r}")

        field_id = _validate_path_component(field_id, name="field_id")
        start_time_ns = int(start_time_ns)
        end_time_ns = int(end_time_ns)
        if start_time_ns > end_time_ns:
            raise ValueError("start_time_ns must be less than or equal to end_time_ns")

        start_date = datetime.fromtimestamp(
            start_time_ns // NS_PER_SECOND,
            tz=timezone.utc,
        ).date()
        end_date = datetime.fromtimestamp(
            end_time_ns // NS_PER_SECOND,
            tz=timezone.utc,
        ).date()

        records: list[CustomRecord] = []
        current_date = start_date
        while current_date <= end_date:
            for path in self.list_records(
                mode=mode,
                date=current_date.isoformat(),
                field_id=field_id,
                committed_only=committed_only,
            ):
                record = CustomRecord.from_path(path)
                if start_time_ns <= record.timestamp_ns_utc <= end_time_ns:
                    records.append(record)
            current_date += timedelta(days=1)

        return sorted(records, key=lambda record: record.timestamp_ns_utc)


# Resolve the circular parent/child annotations after every model is defined.
for _model in (
    ImageRecord,
    CameraRecord,
    CalibrationRecord,
    GnssRecord,
    ArmRecord,
    PCDSegment,
    PCDRecord,
    YoloRecord,
    CustomRecord,
    CustomStore,
):
    _model.model_rebuild()


def jst_datetime_to_time_ns(date_str: str, time_str: str) -> int:
    """Convert a JST date/time string to Unix time in nanoseconds.

    Args:
        date_str:
            JST calendar date in ``YYYY-MM-DD`` format.

        time_str:
            JST time in one of these formats:

            - ``HH:MM:SS``
            - ``HH:MM:SS.NNNNNNNNN``
            - ``HHMMSS``
            - ``HHMMSS.NNNNNNNNN``
            - ``HHMMSS.NNNNNNNNNJST``

    Returns:
        Unix timestamp in nanoseconds.

    Example:
        >>> jst_datetime_to_time_ns(
        ...     "2026-08-19",
        ...     "160200.123456789JST",
        ... )
    """
    time_str = time_str.removesuffix("JST")

    match = re.fullmatch(
        r"(?P<hour>\d{2}):?"
        r"(?P<minute>\d{2}):?"
        r"(?P<second>\d{2})"
        r"(?:\.(?P<nsec>\d{1,9}))?",
        time_str,
    )
    if match is None:
        raise ValueError(f"Invalid JST time string: {time_str!r}")

    try:
        date_value = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date string: {date_str!r}; expected YYYY-MM-DD"
        ) from exc

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))

    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"Invalid JST time string: {time_str!r}")

    # Right-pad fractional seconds to exactly nanoseconds.
    nsec_text = match.group("nsec") or ""
    nsec = int(nsec_text.ljust(9, "0"))

    dt_jst = datetime(
        date_value.year,
        date_value.month,
        date_value.day,
        hour,
        minute,
        second,
        tzinfo=JST,
    )

    dt_utc = dt_jst.astimezone(timezone.utc)
    epoch_seconds = calendar.timegm(dt_utc.utctimetuple())

    return epoch_seconds * NS_PER_SECOND + nsec


if __name__ == "__main__":
    # fs = RecordFS(
    #     "sftp://localhost:2222/",
    #     storage_options={
    #         "username": "myuser",
    #         "password": "mypassword",
    #     },
    #     keepalive=30,
    # )

    # print(fs.ls(fs.root_key))
    pass