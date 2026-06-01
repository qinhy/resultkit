import ctypes
import ctypes.util
import os


CUDA_IPC_HANDLE_BYTES = 64
CUDA_IPC_EVENT_HANDLE_BYTES = 64

cudaSuccess = 0
cudaMemcpyDeviceToDevice = 3
cudaIpcMemLazyEnablePeerAccess = 1

cudaEventDisableTiming = 0x02
cudaEventInterprocess = 0x04


class CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_HANDLE_BYTES)]


class CudaIpcEventHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * CUDA_IPC_EVENT_HANDLE_BYTES)]


class CUDART:
    def __init__(self):
        lib = ctypes.util.find_library("cudart") or "libcudart.so"
        self.rt = ctypes.CDLL(lib)

        self.rt.cudaSetDevice.argtypes = [ctypes.c_int]
        self.rt.cudaSetDevice.restype = ctypes.c_int

        self.rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.rt.cudaMalloc.restype = ctypes.c_int

        self.rt.cudaFree.argtypes = [ctypes.c_void_p]
        self.rt.cudaFree.restype = ctypes.c_int

        self.rt.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.rt.cudaMemcpyAsync.restype = ctypes.c_int

        self.rt.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(CudaIpcMemHandle),
            ctypes.c_void_p,
        ]
        self.rt.cudaIpcGetMemHandle.restype = ctypes.c_int

        self.rt.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        self.rt.cudaIpcOpenMemHandle.restype = ctypes.c_int

        self.rt.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self.rt.cudaIpcCloseMemHandle.restype = ctypes.c_int

        self.rt.cudaEventCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.rt.cudaEventCreateWithFlags.restype = ctypes.c_int

        self.rt.cudaEventRecord.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.rt.cudaEventRecord.restype = ctypes.c_int

        self.rt.cudaIpcGetEventHandle.argtypes = [
            ctypes.POINTER(CudaIpcEventHandle),
            ctypes.c_void_p,
        ]
        self.rt.cudaIpcGetEventHandle.restype = ctypes.c_int

        self.rt.cudaIpcOpenEventHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            CudaIpcEventHandle,
        ]
        self.rt.cudaIpcOpenEventHandle.restype = ctypes.c_int

        self.rt.cudaStreamWaitEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.rt.cudaStreamWaitEvent.restype = ctypes.c_int

        self.rt.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.rt.cudaEventDestroy.restype = ctypes.c_int

    def check(self, err: int, where: str):
        if err != cudaSuccess:
            raise RuntimeError(f"{where} failed with CUDA error code {err}")

    def set_device(self, device: int):
        self.check(self.rt.cudaSetDevice(device), "cudaSetDevice")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self.check(self.rt.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return int(ptr.value)

    def free(self, ptr: int):
        if ptr:
            self.check(self.rt.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def ipc_get_mem_handle(self, ptr: int) -> bytes:
        h = CudaIpcMemHandle()
        self.check(
            self.rt.cudaIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)),
            "cudaIpcGetMemHandle",
        )
        return bytes(h.reserved)

    def ipc_open_mem_handle(self, handle_bytes: bytes) -> int:
        h = CudaIpcMemHandle()
        h.reserved = handle_bytes[:CUDA_IPC_HANDLE_BYTES]

        ptr = ctypes.c_void_p()
        self.check(
            self.rt.cudaIpcOpenMemHandle(
                ctypes.byref(ptr),
                h,
                cudaIpcMemLazyEnablePeerAccess,
            ),
            "cudaIpcOpenMemHandle",
        )
        return int(ptr.value)

    def ipc_close_mem_handle(self, ptr: int):
        if ptr:
            self.check(
                self.rt.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr)),
                "cudaIpcCloseMemHandle",
            )

    def create_ipc_event(self) -> int:
        event = ctypes.c_void_p()
        flags = cudaEventDisableTiming | cudaEventInterprocess
        self.check(
            self.rt.cudaEventCreateWithFlags(ctypes.byref(event), flags),
            "cudaEventCreateWithFlags",
        )
        return int(event.value)

    def event_record(self, event: int, stream: int = 0):
        self.check(
            self.rt.cudaEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream)),
            "cudaEventRecord",
        )

    def ipc_get_event_handle(self, event: int) -> bytes:
        h = CudaIpcEventHandle()
        self.check(
            self.rt.cudaIpcGetEventHandle(ctypes.byref(h), ctypes.c_void_p(event)),
            "cudaIpcGetEventHandle",
        )
        return bytes(h.reserved)

    def ipc_open_event_handle(self, handle_bytes: bytes) -> int:
        h = CudaIpcEventHandle()
        h.reserved = handle_bytes[:CUDA_IPC_EVENT_HANDLE_BYTES]

        event = ctypes.c_void_p()
        self.check(
            self.rt.cudaIpcOpenEventHandle(ctypes.byref(event), h),
            "cudaIpcOpenEventHandle",
        )
        return int(event.value)

    def stream_wait_event(self, stream: int, event: int):
        self.check(
            self.rt.cudaStreamWaitEvent(
                ctypes.c_void_p(stream),
                ctypes.c_void_p(event),
                0,
            ),
            "cudaStreamWaitEvent",
        )

    def event_destroy(self, event: int):
        if event:
            self.check(self.rt.cudaEventDestroy(ctypes.c_void_p(event)), "cudaEventDestroy")


import numpy as np
import cupy as cp
CUDA_IPC_HANDLE_BYTES = 64
CUDA_IPC_EVENT_HANDLE_BYTES = 64
STREAM_NAME_BYTES = 128
LAYOUT_BYTES = 32
MAX_DIMS = 8



class Iox2PayloadMixin:
    """Common header behavior for all fixed-size iceoryx2 payload structs."""

    payload_type: ClassVar[str]

    @classmethod
    def type_name(cls) -> str:
        return cls.payload_type

    def init_header(self) -> None:
        self.magic = IOX2_MAGIC
        self.version = IOX2_VERSION

    def validate_header(self) -> None:
        if bytes(self.magic) != IOX2_MAGIC:
            raise Iox2PayloadError(
                f"bad {self.type_name()} magic: {bytes(self.magic)!r}"
            )

        if int(self.version) != IOX2_VERSION:
            raise Iox2PayloadError(
                f"bad {self.type_name()} version: {int(self.version)}"
            )
        
class Iox2CUDAIPCStreamInfo(Iox2PayloadMixin, ctypes.Structure):
    """
    Published once, or periodically as heartbeat/discovery.

    This describes the producer-owned CUDA ring buffer.
    """

    payload_type = "CudaIpcStreamInfoV1"

    _fields_ = [
        ("magic", ctypes.c_char * 8),          # b"CUDAIPC"
        ("version", ctypes.c_uint32),

        ("stream_name", ctypes.c_char * STREAM_NAME_BYTES),
        ("layout", ctypes.c_char * LAYOUT_BYTES),  # b"HWC", b"CHW", b"MAT", etc.

        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("num_slots", ctypes.c_uint32),

        ("dtype_code", ctypes.c_uint32),       # your enum
        ("itemsize", ctypes.c_uint64),
        ("frame_bytes", ctypes.c_uint64),
        ("total_bytes", ctypes.c_uint64),

        ("handle_len", ctypes.c_uint64),
        ("producer_pid", ctypes.c_uint64),

        # CUDA IPC memory handle for the whole ring buffer.
        ("cuda_ipc_handle", ctypes.c_uint8 * CUDA_IPC_HANDLE_BYTES),
    ]


class Iox2CUDAIPCFrameInfo(Iox2PayloadMixin, ctypes.Structure):
    """
    Published once per ready frame.

    This does NOT contain the frame data.
    It tells subscribers which GPU slot is ready and what CUDA event to wait on.
    """

    payload_type = "CudaIpcFrameInfoV1"

    _fields_ = [
        ("magic", ctypes.c_char * 8),          # b"CUDAFRM"
        ("version", ctypes.c_uint32),

        ("sequence", ctypes.c_uint64),
        ("slot_index", ctypes.c_uint32),

        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * MAX_DIMS),
        ("strides", ctypes.c_uint64 * MAX_DIMS),

        ("dtype_code", ctypes.c_uint32),
        ("itemsize", ctypes.c_uint64),
        ("frame_bytes", ctypes.c_uint64),

        # Event recorded by producer after GPU write/copy into slot.
        ("event_handle_len", ctypes.c_uint64),
        ("cuda_ipc_event_handle", ctypes.c_uint8 * CUDA_IPC_EVENT_HANDLE_BYTES),
    ]

class CudaMatProducer:
    def __init__(
        self,
        topic: str,
        shape: tuple[int, ...],
        dtype=np.float32,
        num_slots: int = 4,
        device_id: int = 0,
    ):
        self.topic = topic
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.num_slots = int(num_slots)
        self.device_id = int(device_id)

        self.cudart = CUDART()
        self.cudart.set_device(self.device_id)

        self.frame_bytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.total_bytes = self.frame_bytes * self.num_slots

        self.base_ptr = self.cudart.malloc(self.total_bytes)
        self.mem_handle = self.cudart.ipc_get_mem_handle(self.base_ptr)

        self.events = [self.cudart.create_ipc_event() for _ in range(self.num_slots)]
        self.event_handles = [
            self.cudart.ipc_get_event_handle(e) for e in self.events
        ]

        self.sequence = 0

        # You would create these exactly like your existing iceoryx2 services.
        self.stream_pub = self._make_pub(f"{topic}/stream", Iox2CUDAIPCStreamInfo)
        self.frame_pub = self._make_pub(f"{topic}/frame", Iox2CUDAIPCFrameInfo)

    def _make_pub(self, topic_name: str, payload_type):
        service = (
            iox2.NodeBuilder.new()
            .create(iox2.ServiceType.Ipc)
            .service_builder(iox2.ServiceName.new(topic_name))
            .publish_subscribe(payload_type)
            .open_or_create()
        )
        return service.publisher_builder().create()

    def publish_stream_info(self):
        msg = Iox2CUDAIPCStreamInfo()
        msg.magic = b"CUDAIPC"
        msg.version = 1

        msg.stream_name = self.topic.encode()[:STREAM_NAME_BYTES]
        msg.layout = b"MAT"

        # For 2D matrix: height, width.
        # For HWC image: height, width, channels.
        if len(self.shape) == 2:
            msg.height = self.shape[0]
            msg.width = self.shape[1]
            msg.channels = 1
        elif len(self.shape) == 3:
            msg.height = self.shape[0]
            msg.width = self.shape[1]
            msg.channels = self.shape[2]
        else:
            raise ValueError("StreamInfo example supports 2D or 3D shapes")

        msg.num_slots = self.num_slots
        msg.dtype_code = self._dtype_code(self.dtype)
        msg.itemsize = self.dtype.itemsize
        msg.frame_bytes = self.frame_bytes
        msg.total_bytes = self.total_bytes
        msg.handle_len = CUDA_IPC_HANDLE_BYTES
        msg.producer_pid = os.getpid()

        msg.cuda_ipc_handle[:] = self.mem_handle

        self.stream_pub.send_copy(msg)

    def publish_cupy(self, arr: cp.ndarray, stream: cp.cuda.Stream | None = None):
        """
        Copy a CuPy array into the next ring slot, record an IPC event,
        then publish a tiny frame-ready message.
        """
        arr = cp.asarray(arr)
        if arr.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {arr.shape}")
        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype, copy=False)
        if not arr.flags.c_contiguous:
            arr = cp.ascontiguousarray(arr)

        stream = stream or cp.cuda.get_current_stream()
        stream_ptr = int(stream.ptr)

        slot = self.sequence % self.num_slots
        dst_ptr = self.base_ptr + slot * self.frame_bytes

        self.cudart.check(
            self.cudart.rt.cudaMemcpyAsync(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(int(arr.data.ptr)),
                self.frame_bytes,
                cudaMemcpyDeviceToDevice,
                ctypes.c_void_p(stream_ptr),
            ),
            "cudaMemcpyAsync D2D into IPC slot",
        )

        self.cudart.event_record(self.events[slot], stream_ptr)

        frame = Iox2CUDAIPCFrameInfo()
        frame.magic = b"CUDAFRM"
        frame.version = 1
        frame.sequence = self.sequence
        frame.slot_index = slot

        frame.ndim = len(self.shape)
        for i, dim in enumerate(self.shape):
            frame.shape[i] = dim

        # C-contiguous strides in bytes.
        strides = np.empty(self.shape, dtype=self.dtype).strides
        for i, stride in enumerate(strides):
            frame.strides[i] = stride

        frame.dtype_code = self._dtype_code(self.dtype)
        frame.itemsize = self.dtype.itemsize
        frame.frame_bytes = self.frame_bytes

        frame.event_handle_len = CUDA_IPC_EVENT_HANDLE_BYTES
        frame.cuda_ipc_event_handle[:] = self.event_handles[slot]

        self.frame_pub.send_copy(frame)
        self.sequence += 1

    def _dtype_code(self, dtype: np.dtype) -> int:
        if dtype == np.float32:
            return 1
        if dtype == np.float64:
            return 2
        if dtype == np.uint8:
            return 3
        if dtype == np.int32:
            return 4
        raise ValueError(f"unsupported dtype: {dtype}")

    def close(self):
        # Important: producer must not free while subscribers still have the IPC memory open.
        self.cudart.free(self.base_ptr)
        self.base_ptr = 0


class CudaMatSubscriber:
    def __init__(self, topic: str, device_id: int = 0):
        self.topic = topic
        self.device_id = int(device_id)

        self.cudart = CUDART()
        self.cudart.set_device(self.device_id)

        self.imported_base_ptr: int | None = None
        self.stream_info: Iox2CUDAIPCStreamInfo | None = None
        self.event_cache: dict[bytes, int] = {}

        self.stream_sub = self._make_sub(f"{topic}/stream", Iox2CUDAIPCStreamInfo)
        self.frame_sub = self._make_sub(f"{topic}/frame", Iox2CUDAIPCFrameInfo)

    def _make_sub(self, topic_name: str, payload_type):
        service = (
            iox2.NodeBuilder.new()
            .create(iox2.ServiceType.Ipc)
            .service_builder(iox2.ServiceName.new(topic_name))
            .publish_subscribe(payload_type)
            .open_or_create()
        )
        return service.subscriber_builder().create()

    def receive_stream_info(self) -> bool:
        sample = self.stream_sub.receive()
        if sample is None:
            return False

        info = sample.payload()
        handle = bytes(info.cuda_ipc_handle[:info.handle_len])

        if self.imported_base_ptr is None:
            self.imported_base_ptr = self.cudart.ipc_open_mem_handle(handle)
            self.stream_info = info

        return True

    def receive_frame(self, stream: cp.cuda.Stream | None = None) -> cp.ndarray | None:
        if self.imported_base_ptr is None:
            self.receive_stream_info()
            if self.imported_base_ptr is None:
                return None

        sample = self.frame_sub.receive()
        if sample is None:
            return None

        frame = sample.payload()

        event_handle = bytes(
            frame.cuda_ipc_event_handle[:frame.event_handle_len]
        )

        event = self.event_cache.get(event_handle)
        if event is None:
            event = self.cudart.ipc_open_event_handle(event_handle)
            self.event_cache[event_handle] = event

        stream = stream or cp.cuda.get_current_stream()
        self.cudart.stream_wait_event(int(stream.ptr), event)

        shape = tuple(int(frame.shape[i]) for i in range(frame.ndim))
        dtype = self._dtype_from_code(frame.dtype_code)

        slot = int(frame.slot_index)
        ptr = self.imported_base_ptr + slot * int(frame.frame_bytes)

        # CuPy view over imported CUDA IPC memory.
        mem = cp.cuda.UnownedMemory(
            ptr,
            int(frame.frame_bytes),
            owner=self,
        )
        memptr = cp.cuda.MemoryPointer(mem, 0)

        return cp.ndarray(shape, dtype=dtype, memptr=memptr)

    def _dtype_from_code(self, code: int) -> np.dtype:
        if code == 1:
            return np.dtype(np.float32)
        if code == 2:
            return np.dtype(np.float64)
        if code == 3:
            return np.dtype(np.uint8)
        if code == 4:
            return np.dtype(np.int32)
        raise ValueError(f"unsupported dtype code: {code}")

    def close(self):
        for event in self.event_cache.values():
            self.cudart.event_destroy(event)
        self.event_cache.clear()

        if self.imported_base_ptr:
            self.cudart.ipc_close_mem_handle(self.imported_base_ptr)
            self.imported_base_ptr = None




import ctypes


CUDA_IPC_MEM_HANDLE_BYTES = 64
CUDA_IPC_EVENT_HANDLE_BYTES = 64


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