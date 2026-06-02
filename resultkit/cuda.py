import ctypes
from typing import Any, ClassVar, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
import torch
import iceoryx2 as iox2
try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
except ImportError:
    print("warning: pycuda is not available")
    cuda = None
    gpuarray = None
    

class ImageMatCUDAPubSub(BaseModel):
    """CUDA IPC backed ImageMat publisher/subscriber using PyCUDA only.

    iceoryx2 carries a tiny control message.  The image bytes stay in
    CUDA device memory and are shared with subscribers through CUDA IPC.

    Notes:
    - Create/push a PyCUDA CUDA context before constructing this model.
    - The publisher process must stay alive while subscribers read its
        CUDA IPC memory handle.
    - Subscribers receive a ``pycuda.gpuarray.GPUArray`` view.  Use
        ``copy=True`` in ``sub`` to detach the returned GPU array from the
        publisher ring slot.
    """

    CUDA_IPC_MEM_HANDLE_BYTES: ClassVar[int] = 64
    CUDA_IPC_EVENT_HANDLE_BYTES: ClassVar[int] = 64
    DTYPE_STR_BYTES: ClassVar[int] = 16
    MAX_DIMS: ClassVar[int] = 8

    class Iox2CUDAIPCFrameSignal(ctypes.Structure):
        """Small fixed-size iceoryx2 payload announcing the latest CUDA frame."""

        payload_type = "CudaIpcFrameSignalV3PyCUDA"

        _fields_ = [
            ("magic", ctypes.c_char * 8),       # b"CUDAPY3"
            ("version", ctypes.c_uint32),
            ("slot_index", ctypes.c_uint32),

            ("num_slots", ctypes.c_uint32),
            ("ndim", ctypes.c_uint32),
            ("dtype_str", ctypes.c_char * 16),  # numpy dtype.str, e.g. '|u1', '<f4'
            ("itemsize", ctypes.c_uint32),
            ("device_id", ctypes.c_int32),
            ("producer_pid", ctypes.c_uint32),

            ("sequence", ctypes.c_uint64),
            ("frame_bytes", ctypes.c_uint64),
            ("total_bytes", ctypes.c_uint64),

            ("shape", ctypes.c_int64 * 8),
            ("strides", ctypes.c_int64 * 8),

            ("mem_handle_len", ctypes.c_uint64),
            ("cuda_ipc_mem_handle", ctypes.c_uint8 * 64),

            ("event_handle_len", ctypes.c_uint64),
            ("cuda_ipc_event_handle", ctypes.c_uint8 * 64),
        ]

        @staticmethod
        def type_name() -> str:
            return "CudaIpcFrameSignalV3PyCUDA"

        @classmethod
        def new(
            cls,
            *,
            sequence: int,
            slot_index: int,
            num_slots: int,
            shape: Tuple[int, ...],
            strides: Tuple[int, ...],
            dtype_str: str,
            itemsize: int,
            frame_bytes: int,
            total_bytes: int,
            mem_handle: bytes,
            event_handle: bytes,
            device_id: int,
            producer_pid: int,
        ) -> "ImageMatCUDAPubSub.Iox2CUDAIPCFrameSignal":
            if len(shape) > 8:
                raise ValueError(f"CUDA IPC signal supports at most 8 dimensions, got {shape}")
            if len(strides) != len(shape):
                raise ValueError(f"strides length must match shape length: {strides} vs {shape}")
            if len(mem_handle) > 64:
                raise ValueError("CUDA IPC memory handle too large")
            if len(event_handle) > 64:
                raise ValueError("CUDA IPC event handle too large")

            dtype_bytes = dtype_str.encode("ascii")
            if len(dtype_bytes) >= 16:
                raise ValueError(f"dtype string too long for CUDA IPC signal: {dtype_str!r}")

            msg = cls()
            msg.magic = b"CUDAPY3"
            msg.version = 3
            msg.slot_index = int(slot_index)
            msg.num_slots = int(num_slots)
            msg.ndim = len(shape)
            msg.dtype_str = dtype_bytes
            msg.itemsize = int(itemsize)
            msg.device_id = int(device_id)
            msg.producer_pid = int(producer_pid)
            msg.sequence = int(sequence)
            msg.frame_bytes = int(frame_bytes)
            msg.total_bytes = int(total_bytes)

            for i, value in enumerate(shape):
                msg.shape[i] = int(value)
            for i, value in enumerate(strides):
                msg.strides[i] = int(value)

            msg.mem_handle_len = len(mem_handle)
            msg.cuda_ipc_mem_handle[: len(mem_handle)] = mem_handle
            msg.event_handle_len = len(event_handle)
            msg.cuda_ipc_event_handle[: len(event_handle)] = event_handle
            return msg

        def mem_handle_bytes(self) -> bytes:
            return bytes(self.cuda_ipc_mem_handle[: self.mem_handle_len])

        def event_handle_bytes(self) -> bytes:
            return bytes(self.cuda_ipc_event_handle[: self.event_handle_len])

        def dtype(self) -> np.dtype:
            raw = bytes(self.dtype_str).split(b"\x00", 1)[0]
            if not raw:
                raise ValueError("empty dtype in CUDA IPC signal")
            return np.dtype(raw.decode("ascii"))

        def shape_tuple(self) -> Tuple[int, ...]:
            return tuple(int(self.shape[i]) for i in range(int(self.ndim)))

        def strides_tuple(self) -> Tuple[int, ...]:
            return tuple(int(self.strides[i]) for i in range(int(self.ndim)))

        def validate(self) -> None:
            if bytes(self.magic).rstrip(b"\x00") != b"CUDAPY3":
                raise ValueError(f"bad magic: {bytes(self.magic)!r}")
            if self.version != 3:
                raise ValueError(f"unsupported CUDA IPC signal version: {self.version}")
            if not (0 <= self.slot_index < self.num_slots):
                raise ValueError(f"slot_index out of range: {self.slot_index}/{self.num_slots}")
            if self.ndim == 0 or self.ndim > 8:
                raise ValueError(f"invalid ndim: {self.ndim}")
            if self.mem_handle_len != 64:
                raise ValueError(f"bad CUDA IPC memory handle length: {self.mem_handle_len}")
            # Some PyCUDA builds, especially on Windows, expose CUDA IPC
            # memory handles but do not expose event_from_ipc_handle.
            # In that portable mode the publisher synchronizes before
            # sending the signal and the event handle is intentionally empty.
            if self.event_handle_len not in (0, 64):
                raise ValueError(f"bad CUDA IPC event handle length: {self.event_handle_len}")
            if self.frame_bytes <= 0 or self.total_bytes <= 0:
                raise ValueError(
                    f"invalid CUDA IPC byte sizes: frame={self.frame_bytes}, total={self.total_bytes}"
                )
            if self.frame_bytes * self.num_slots > self.total_bytes:
                raise ValueError(
                    f"ring layout exceeds allocation: frame={self.frame_bytes}, "
                    f"slots={self.num_slots}, total={self.total_bytes}"
                )

    class _OffsetPointer(cuda.PointerHolderBase):
        """PointerHolderBase that keeps the owner allocation alive and adds a byte offset.

        PyCUDA's kernel argument builder accepts GPUArray objects and many
        built-in pointer holders, but it does not recognize every custom
        PointerHolderBase subclass when the object itself is passed as a
        kernel argument.  Expose a ``gpudata`` integer so both of these
        call styles work::

            kernel(frame, ...)
            kernel(frame.gpudata, ...)

        The integer value is the final device pointer including offset.
        """

        def __init__(self, owner, offset: int = 0):
            super().__init__()
            self.owner = owner
            self.offset = int(offset)

        def get_pointer(self):
            return int(self.owner) + self.offset

        @property
        def gpudata(self):
            return int(self)

        @property
        def ptr(self):
            return int(self)

    lib: str = "pycuda"
    num_slots: int = Field(default=2, ge=1)
    sequence: int = 0
    device_id: int = 0
    producer_pid: int = 0
    # Keep this True by default for portability. PyCUDA IPC memory is
    # widely exposed, but CUDA IPC events are missing from some PyCUDA
    # wheels/builds. Synchronizing before publishing guarantees the
    # subscriber sees a complete frame without opening an IPC event.
    sync_before_send: bool = True

    data: Any = Field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.uint8),
        exclude=True,
    )

    _gpu_ring: Any = PrivateAttr(default=None)
    _gpu_ring_shape: Any = PrivateAttr(default=None)
    _gpu_ring_dtype: Any = PrivateAttr(default=None)
    _gpu_mem_handle: Optional[bytes] = PrivateAttr(default=None)
    _cuda_event: Any = PrivateAttr(default=None)
    _cuda_event_handle: Optional[bytes] = PrivateAttr(default=None)
    _frame_bytes: int = PrivateAttr(default=0)
    _total_bytes: int = PrivateAttr(default=0)

    _remote_mem_handle: Optional[bytes] = PrivateAttr(default=None)
    _remote_event_handle: Optional[bytes] = PrivateAttr(default=None)
    _remote_mem: Any = PrivateAttr(default=None)
    _remote_event: Any = PrivateAttr(default=None)
    _remote_shape: Any = PrivateAttr(default=None)
    _remote_strides: Any = PrivateAttr(default=None)
    _remote_dtype: Any = PrivateAttr(default=None)
    _remote_total_bytes: int = PrivateAttr(default=0)

    @staticmethod
    def _current_device_id() -> int:
        try:
            return int(cuda.Context.get_device().ordinal)
        except Exception:
            # PyCUDA Device objects expose ordinal on recent versions;
            # if not available, keep the value as diagnostic metadata.
            return 0

    @staticmethod
    def _as_bytearray(handle: bytes) -> bytearray:
        # PyCUDA's IPC open helpers expect bytearray, not bytes.
        return bytearray(handle)

    @staticmethod
    def _is_gpuarray(data) -> bool:
        return isinstance(data, gpuarray.GPUArray)

    @staticmethod
    def _require_contiguous(arr):
        expected = tuple(np.ndarray(arr.shape, dtype=arr.dtype).strides)
        if tuple(arr.strides) != expected:
            raise ValueError(
                f"ImageMatCUDAPubSub requires C-contiguous GPUArray data, "
                f"got strides={arr.strides}, expected={expected}"
            )
        return arr

    @classmethod
    def _as_gpuarray(cls, data):
        if isinstance(data, gpuarray.GPUArray):
            return cls._require_contiguous(data)
        if isinstance(data, torch.Tensor):
            # PyCUDA does not consume torch CUDA tensors directly here.
            # Copy through host for correctness without CuPy/DLPack.
            data = data.detach().cpu().numpy()
        return gpuarray.to_gpu(np.ascontiguousarray(data))

    def _get_service(self):
        if self._service is None:
            self._service = (
                self._get_node()
                .service_builder(iox2.ServiceName.new(self.topic_name()))
                .publish_subscribe(self.Iox2CUDAIPCFrameSignal)
                .open_or_create()
            )
        return self._service

    def get_pub(self):
        if self._pub is None:
            self._pub = self._get_service().publisher_builder().create()
        return self._pub

    def get_sub(self):
        if self._sub is None:
            self._sub = self._get_service().subscriber_builder().create()
        return self._sub

    def _ensure_event(self):
        """Best-effort CUDA IPC event creation.

        This is optional. Some PyCUDA builds do not expose the matching
        event_from_ipc_handle opener, so pub/sub correctness must not
        depend on it. The default path uses Context.synchronize() before
        sending the iceoryx2 signal instead.
        """
        if self._cuda_event is None:
            flags = int(cuda.event_flags.DISABLE_TIMING) | int(cuda.event_flags.INTERPROCESS)
            self._cuda_event = cuda.Event(flags=flags)
            self._cuda_event_handle = bytes(self._cuda_event.ipc_handle())
        return self._cuda_event

    def _ensure_gpu_ring(self, arr):
        shape = tuple(int(v) for v in arr.shape)
        dtype = np.dtype(arr.dtype)

        if len(shape) == 0:
            raise ValueError("ImageMatCUDAPubSub cannot publish scalar CUDA data")
        if len(shape) > self.MAX_DIMS:
            raise ValueError(f"CUDA IPC signal supports at most {self.MAX_DIMS} dimensions, got {shape}")

        needs_alloc = (
            self._gpu_ring is None
            or self._gpu_ring_shape != shape
            or self._gpu_ring_dtype != dtype
            or int(self._gpu_ring.shape[0]) != int(self.num_slots)
        )

        if needs_alloc:
            self.device_id = self._current_device_id()
            self.producer_pid = int(__import__("os").getpid())
            self._gpu_ring = gpuarray.empty((int(self.num_slots),) + shape, dtype=dtype)
            self._gpu_ring_shape = shape
            self._gpu_ring_dtype = dtype
            self._frame_bytes = int(self._gpu_ring.strides[0])
            self._total_bytes = int(self._gpu_ring.nbytes)
            self._gpu_mem_handle = bytes(cuda.mem_get_ipc_handle(int(self._gpu_ring.gpudata)))
            self._cuda_event_handle = b""

        return self._gpu_ring

    def _slot_view(self, slot_index: int):
        ptr = self._OffsetPointer(self._gpu_ring.gpudata, int(slot_index) * int(self._frame_bytes))
        return gpuarray.GPUArray(
            shape=self._gpu_ring_shape,
            dtype=self._gpu_ring_dtype,
            gpudata=ptr,
            base=self._gpu_ring,
            strides=tuple(np.ndarray(self._gpu_ring_shape, dtype=self._gpu_ring_dtype).strides),
        )
    
    def _make_signal(self, slot_index: int):
        shape = tuple(int(v) for v in self._gpu_ring_shape)
        strides = tuple(np.ndarray(shape, dtype=self._gpu_ring_dtype).strides)
        return self.Iox2CUDAIPCFrameSignal.new(
            sequence=int(self.sequence),
            slot_index=int(slot_index),
            num_slots=int(self.num_slots),
            shape=shape,
            strides=strides,
            dtype_str=np.dtype(self._gpu_ring_dtype).str,
            itemsize=int(np.dtype(self._gpu_ring_dtype).itemsize),
            frame_bytes=int(self._frame_bytes),
            total_bytes=int(self._total_bytes),
            mem_handle=self._gpu_mem_handle,
            event_handle=self._cuda_event_handle or b"",
            device_id=int(self.device_id),
            producer_pid=int(self.producer_pid),
        )

    def get_data(self):
        return self.data

    def pub(self, data=None, edit_func=None):
        """Publish one CUDA image frame using PyCUDA.

        ``data`` may be a PyCUDA GPUArray, NumPy array, or torch Tensor.
        The published ring slot is passed to ``edit_func`` when provided,
        so callers can fill the GPU slot directly.
        """
        self.is_pub = True
        arr = self._as_gpuarray(self.data if data is None else data)
        # self._update_image_metadata_from_array(arr)
        # self.validate()

        self._ensure_gpu_ring(arr)
        slot_index = int(self.sequence % int(self.num_slots))
        out = self._slot_view(slot_index)

        if data is not None or edit_func is None:
            cuda.memcpy_dtod(int(out.gpudata), int(arr.gpudata), int(out.nbytes))
        if edit_func is not None:
            edit_func(out)

        if self.sync_before_send:
            # Portable synchronization path. This avoids relying on
            # pycuda.driver.event_from_ipc_handle, which is absent in
            # some PyCUDA builds while IPC memory still works.
            cuda.Context.synchronize()

        sample = self.get_pub().loan_uninit()
        sample = sample.write_payload(self._make_signal(slot_index))
        sample.send()

        self.data = out
        self.sequence += 1
        return self

    def _open_remote_memory_if_needed(self, signal):
        mem_handle = signal.mem_handle_bytes()
        dtype = signal.dtype()
        shape = signal.shape_tuple()
        strides = signal.strides_tuple()

        layout_changed = (
            self._remote_mem_handle != mem_handle
            or self._remote_total_bytes != int(signal.total_bytes)
            or self._remote_dtype != dtype
            or self._remote_shape != shape
            or self._remote_strides != strides
        )

        if layout_changed:
            if self._remote_mem is not None:
                self._remote_mem.close()
            flags = cuda.ipc_mem_flags.LAZY_ENABLE_PEER_ACCESS
            self._remote_mem = cuda.IPCMemoryHandle(self._as_bytearray(mem_handle), flags)
            self._remote_mem_handle = mem_handle
            self._remote_total_bytes = int(signal.total_bytes)
            self._remote_dtype = dtype
            self._remote_shape = shape
            self._remote_strides = strides

    def _open_remote_event_if_needed(self, signal):
        event_handle = signal.event_handle_bytes()
        if not event_handle:
            self._remote_event = None
            self._remote_event_handle = b""
            return
        if not hasattr(cuda, "event_from_ipc_handle"):
            # PyCUDA build has no IPC event opener. Publisher-side
            # synchronization is the portable fallback.
            self._remote_event = None
            self._remote_event_handle = event_handle
            return
        if self._remote_event_handle != event_handle:
            self._remote_event = cuda.event_from_ipc_handle(self._as_bytearray(event_handle))
            self._remote_event_handle = event_handle

    def _remote_slot_view(self, signal: Iox2CUDAIPCFrameSignal):
        offset = int(signal.slot_index) * int(signal.frame_bytes)
        ptr = self._OffsetPointer(self._remote_mem, offset)
        return gpuarray.GPUArray(
            shape=signal.shape_tuple(),
            dtype=signal.dtype(),
            gpudata=ptr,
            base=self._remote_mem,
            strides=signal.strides_tuple(),
        )

    def sub(self, copy=False, sync=True):
        """Receive one CUDA image frame as a PyCUDA GPUArray.

        ``copy=False`` returns a view into the publisher ring buffer.
        ``copy=True`` makes a GPU copy so later publisher writes cannot
        overwrite the subscriber's array.
        """
        self.is_pub = False
        sample = self.get_sub().receive()
        if sample is None:
            return self

        signal = sample.payload().contents
        # signal.validate()

        self._open_remote_memory_if_needed(signal)
        self._open_remote_event_if_needed(signal)

        if sync and self._remote_event is not None:
            self._remote_event.synchronize()

        view = self._remote_slot_view(signal)
        self.data = view.copy() if copy else view
        self.sequence = int(signal.sequence)
        self.device_id = int(signal.device_id)
        self.producer_pid = int(signal.producer_pid)
        # self._update_image_metadata_from_array(self.data)
        return self

    def get_data_numpy(self):
        data = self.get_data()
        if isinstance(data, gpuarray.GPUArray):
            return data.get()
        return data

    def close(self):
        if self._remote_mem is not None:
            self._remote_mem.close()
            self._remote_mem = None
        self._remote_event = None
        self._cuda_event = None
        return self

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
