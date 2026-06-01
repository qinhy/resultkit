
import ctypes
import os
import numpy as np

CUDA_IPC_MEM_HANDLE_BYTES = 64
CUDA_IPC_EVENT_HANDLE_BYTES = 64

STREAM_NAME_BYTES = 128
LAYOUT_BYTES = 32
MAX_DIMS = 8

class Iox2CUDAIPCFrameSignal(ctypes.Structure):
    """
    Minimal CUDA IPC frame signal.

    Publishes:
    - CUDA IPC memory handle
    - CUDA IPC event handle
    - sequence number
    - ring-buffer slot index
    - frame_bytes for pointer offset calculation
    """

    payload_type = "CudaIpcFrameSignalV1"

    _fields_ = [
        ("magic", ctypes.c_char * 8),       # b"CUDAFS1"
        ("version", ctypes.c_uint32),
        ("slot_index", ctypes.c_uint32),

        ("sequence", ctypes.c_uint64),
        ("frame_bytes", ctypes.c_uint64),

        ("mem_handle_len", ctypes.c_uint64),
        ("cuda_ipc_mem_handle", ctypes.c_uint8 * CUDA_IPC_MEM_HANDLE_BYTES),

        ("event_handle_len", ctypes.c_uint64),
        ("cuda_ipc_event_handle", ctypes.c_uint8 * CUDA_IPC_EVENT_HANDLE_BYTES),
    ]

    @classmethod
    def new(
        cls,
        *,
        sequence: int,
        slot_index: int,
        frame_bytes: int,
        mem_handle: bytes,
        event_handle: bytes,
    ) -> "Iox2CUDAIPCFrameSignal":
        msg = cls()
        msg.magic = b"CUDAFS1"
        msg.version = 1
        msg.slot_index = int(slot_index)
        msg.sequence = int(sequence)
        msg.frame_bytes = int(frame_bytes)

        if len(mem_handle) > CUDA_IPC_MEM_HANDLE_BYTES:
            raise ValueError("CUDA IPC memory handle too large")
        if len(event_handle) > CUDA_IPC_EVENT_HANDLE_BYTES:
            raise ValueError("CUDA IPC event handle too large")

        msg.mem_handle_len = len(mem_handle)
        msg.cuda_ipc_mem_handle[: len(mem_handle)] = mem_handle

        msg.event_handle_len = len(event_handle)
        msg.cuda_ipc_event_handle[: len(event_handle)] = event_handle

        return msg

    def mem_handle_bytes(self) -> bytes:
        return bytes(self.cuda_ipc_mem_handle[: self.mem_handle_len])

    def event_handle_bytes(self) -> bytes:
        return bytes(self.cuda_ipc_event_handle[: self.event_handle_len])

    def validate(self) -> None:
        if bytes(self.magic).rstrip(b"\x00") != b"CUDAFS1":
            raise ValueError(f"bad magic: {bytes(self.magic)!r}")
        if self.version != 1:
            raise ValueError(f"unsupported version: {self.version}")
        if self.mem_handle_len > CUDA_IPC_MEM_HANDLE_BYTES:
            raise ValueError("bad CUDA IPC memory handle length")
        if self.event_handle_len > CUDA_IPC_EVENT_HANDLE_BYTES:
            raise ValueError("bad CUDA IPC event handle length")


class Iox2CUDAIPCStreamInfo(ctypes.Structure):
    """
    Static CUDA IPC stream discovery message.

    This is usually published once, or periodically as a heartbeat.

    It tells subscribers:
    - which CUDA IPC memory handle to open
    - how many ring-buffer slots exist
    - how to interpret each slot as a matrix/tensor
    """

    payload_type = "CudaIpcStreamInfoV1"

    _fields_ = [
        ("magic", ctypes.c_char * 8),          # b"CUDASI1"
        ("version", ctypes.c_uint32),

        # Device that produced/exported the CUDA allocation.
        # Usually useful for debugging and validation.
        ("device_id", ctypes.c_uint32),

        ("producer_pid", ctypes.c_uint64),

        # Ring-buffer layout.
        ("num_slots", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),

        ("shape", ctypes.c_uint64 * MAX_DIMS),
        ("strides", ctypes.c_uint64 * MAX_DIMS),

        # Data type.
        ("dtype_code", ctypes.c_uint32),
        ("itemsize", ctypes.c_uint32),

        # Size info.
        ("frame_bytes", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),

        # Human-readable metadata.
        ("stream_name", ctypes.c_char * STREAM_NAME_BYTES),
        ("layout", ctypes.c_char * LAYOUT_BYTES),   # b"MAT", b"HWC", b"CHW", etc.

        # CUDA IPC memory handle for the whole ring buffer.
        ("mem_handle_len", ctypes.c_uint64),
        ("cuda_ipc_mem_handle", ctypes.c_uint8 * CUDA_IPC_MEM_HANDLE_BYTES),
    ]

    @classmethod
    def new(
        cls,
        *,
        stream_name: str,
        shape: tuple[int, ...],
        dtype: np.dtype | type,
        num_slots: int,
        mem_handle: bytes,
        device_id: int = 0,
        layout: str = "MAT",
        producer_pid: int | None = None,
    ) -> "Iox2CUDAIPCStreamInfo":
        dtype = np.dtype(dtype)
        shape = tuple(int(x) for x in shape)

        if len(shape) > MAX_DIMS:
            raise ValueError(f"shape has too many dims: {len(shape)} > {MAX_DIMS}")

        if len(mem_handle) > CUDA_IPC_MEM_HANDLE_BYTES:
            raise ValueError("CUDA IPC memory handle too large")

        frame_bytes = int(np.prod(shape)) * dtype.itemsize
        total_bytes = frame_bytes * int(num_slots)

        msg = cls()
        msg.magic = b"CUDASI1"
        msg.version = 1
        msg.device_id = int(device_id)
        msg.producer_pid = int(producer_pid if producer_pid is not None else os.getpid())

        msg.num_slots = int(num_slots)
        msg.ndim = len(shape)

        for i, dim in enumerate(shape):
            msg.shape[i] = dim

        # C-contiguous byte strides.
        strides = np.empty(shape, dtype=dtype).strides
        for i, stride in enumerate(strides):
            msg.strides[i] = int(stride)

        msg.dtype_code = cls.dtype_to_code(dtype)
        msg.itemsize = int(dtype.itemsize)

        msg.frame_bytes = frame_bytes
        msg.total_bytes = total_bytes

        msg.stream_name = stream_name.encode("utf-8")[: STREAM_NAME_BYTES - 1]
        msg.layout = layout.encode("utf-8")[: LAYOUT_BYTES - 1]

        msg.mem_handle_len = len(mem_handle)
        for i, b in enumerate(mem_handle):
            msg.cuda_ipc_mem_handle[i] = b

        return msg

    def validate(self) -> None:
        if bytes(self.magic).rstrip(b"\x00") != b"CUDASI1":
            raise ValueError(f"bad magic: {bytes(self.magic)!r}")

        if self.version != 1:
            raise ValueError(f"unsupported version: {self.version}")

        if self.ndim == 0 or self.ndim > MAX_DIMS:
            raise ValueError(f"bad ndim: {self.ndim}")

        if self.num_slots == 0:
            raise ValueError("num_slots must be > 0")

        if self.mem_handle_len > CUDA_IPC_MEM_HANDLE_BYTES:
            raise ValueError("bad CUDA IPC memory handle length")

        if self.frame_bytes == 0:
            raise ValueError("frame_bytes must be > 0")

        if self.total_bytes < self.frame_bytes * self.num_slots:
            raise ValueError("total_bytes is inconsistent with frame_bytes * num_slots")

    def shape_tuple(self) -> tuple[int, ...]:
        return tuple(int(self.shape[i]) for i in range(self.ndim))

    def strides_tuple(self) -> tuple[int, ...]:
        return tuple(int(self.strides[i]) for i in range(self.ndim))

    def stream_name_str(self) -> str:
        return bytes(self.stream_name).split(b"\x00", 1)[0].decode("utf-8")

    def layout_str(self) -> str:
        return bytes(self.layout).split(b"\x00", 1)[0].decode("utf-8")

    def mem_handle_bytes(self) -> bytes:
        return bytes(self.cuda_ipc_mem_handle[: self.mem_handle_len])

    def dtype(self) -> np.dtype:
        return self.code_to_dtype(int(self.dtype_code))

    @staticmethod
    def dtype_to_code(dtype: np.dtype) -> int:
        dtype = np.dtype(dtype)

        if dtype == np.dtype(np.float32):
            return 1
        if dtype == np.dtype(np.float64):
            return 2
        if dtype == np.dtype(np.uint8):
            return 3
        if dtype == np.dtype(np.int32):
            return 4
        if dtype == np.dtype(np.int64):
            return 5
        if dtype == np.dtype(np.float16):
            return 6

        raise ValueError(f"unsupported dtype: {dtype}")

    @staticmethod
    def code_to_dtype(code: int) -> np.dtype:
        if code == 1:
            return np.dtype(np.float32)
        if code == 2:
            return np.dtype(np.float64)
        if code == 3:
            return np.dtype(np.uint8)
        if code == 4:
            return np.dtype(np.int32)
        if code == 5:
            return np.dtype(np.int64)
        if code == 6:
            return np.dtype(np.float16)

        raise ValueError(f"unsupported dtype code: {code}")


import os
import ctypes
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray


cuda.init()


class PyCUDACudaIpcProducer:
    def __init__(
        self,
        *,
        device_id: int,
        shape: tuple[int, ...],
        dtype=np.float32,
        num_slots: int = 4,
    ):
        self.device_id = int(device_id)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.num_slots = int(num_slots)

        # Important: use a real PyCUDA context.
        # If integrating with CUDA Runtime users, consider retain_primary_context().
        self.dev = cuda.Device(self.device_id)
        self.ctx = self.dev.make_context()

        self.stream = cuda.Stream()

        self.frame_bytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.total_bytes = self.frame_bytes * self.num_slots

        # One CUDA ring buffer.
        self.dev_mem = cuda.mem_alloc(self.total_bytes)

        # Export memory handle.
        self.mem_handle = cuda.mem_get_ipc_handle(self.dev_mem)

        # One IPC event per slot.
        flags = cuda.event_flags.INTERPROCESS | cuda.event_flags.DISABLE_TIMING
        self.events = [cuda.Event(flags) for _ in range(self.num_slots)]
        self.event_handles = [evt.ipc_handle() for evt in self.events]

        self.sequence = 0

    def stream_info_msg(self):
        return Iox2CUDAIPCStreamInfo.new(
            stream_name="cuda-mat",
            shape=self.shape,
            dtype=self.dtype,
            num_slots=self.num_slots,
            mem_handle=self.mem_handle,
            device_id=self.device_id,
            layout="MAT",
            producer_pid=os.getpid(),
        )

    def publish_gpuarray(self, src: gpuarray.GPUArray, stream_info_pub, frame_pub):
        """
        Copy PyCUDA GPUArray into the IPC ring buffer and publish signal.
        """
        if src.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {src.shape}")
        if np.dtype(src.dtype) != self.dtype:
            raise ValueError(f"expected dtype {self.dtype}, got {src.dtype}")
        if src.nbytes != self.frame_bytes:
            raise ValueError(f"expected {self.frame_bytes} bytes, got {src.nbytes}")

        slot = self.sequence % self.num_slots
        dst_ptr = int(self.dev_mem) + slot * self.frame_bytes

        cuda.memcpy_dtod_async(
            dst_ptr,
            int(src.gpudata),
            self.frame_bytes,
            self.stream,
        )

        # Event is recorded after the device-to-device copy.
        self.events[slot].record(self.stream)

        frame_msg = Iox2CUDAIPCFrameSignal.new(
            sequence=self.sequence,
            slot_index=slot,
            event_handle=self.event_handles[slot],
        )

        # Optional: republish stream info occasionally for late subscribers.
        if self.sequence == 0:
            stream_info_pub.send_copy(self.stream_info_msg())

        frame_pub.send_copy(frame_msg)
        self.sequence += 1

    def close(self):
        self.ctx.pop()
        self.ctx.detach()































