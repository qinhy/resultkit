# from https://github.com/qinhy/singleton-key-value-storage.git
import ctypes
import enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr
import torch
import requests
from PIL import Image
from io import BytesIO
import iceoryx2 as iox2

try:    
    from cuda import ImageMatCUDAPubSub
    from mat import to_ctypes_type, to_np_type
    from BasicModel import Controller4Basic, Model4Basic, BasicStore
    from mat import DataType, MatOps, NumpyMatOps, TorchMatOps, MatLib, MatDevice
except Exception as e:    
    from .cuda import ImageMatCUDAPubSub
    from .mat import to_ctypes_type, to_np_type
    from .BasicModel import Controller4Basic, Model4Basic, BasicStore
    from .mat import DataType, MatOps, NumpyMatOps, TorchMatOps, MatLib, MatDevice

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
except ImportError:
    print("warning: pycuda is not available")
    cuda = None
    gpuarray = None
    
class Controller4Mat:
    class AbstractObjController(Controller4Basic.AbstractObjController):pass        
    class AbstractGroupController(Controller4Basic.AbstractGroupController):pass  
    class MatController(AbstractGroupController):pass        
    class ImageMatController(MatController):pass 
    class ImageMatViewController(ImageMatController):pass        
    class BoundingBoxController(MatController):pass

class Model4Mat:
    class AbstractObj(Model4Basic.AbstractObj):
        def model_dump_json_dict(self): return self
        def init_controller(self,store):
            self.controller = self._get_controller_class(Controller4Mat)(store,self)

    class AbstractGroup(Model4Basic.AbstractGroup, AbstractObj):pass

    class Mat(AbstractGroup):
        lib: MatLib = MatLib.NUMPY
        _ops: MatOps = None
        dtype: DataType = DataType.FLOAT32
        device: MatDevice = MatDevice.CPU
        data: Union[np.ndarray,torch.Tensor] = Field(default_factory=lambda:np.random.rand(5,5), exclude=True)
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def model_post_init(self, context):
            self.init()
            self.validate()
            return super().model_post_init(context)

        def get_data(self):
            return self.data
        
        def get_ops(self):
            if self._ops is None:
                self._ops = self.get_mat_ops(self.lib)
            return self._ops

        def init(self):            
            self.lib = MatLib.which(self.get_data())
            self.device = MatDevice.which(self.get_data())
            self.dtype = DataType.which(self.get_data())
            return self
        
        def validate(self):
            pass

        def shape(self):return self.data.shape

        def update(self, **kwargs):
            return self.controller.update(**kwargs).model

        def safe_update_data(self,data:Union[np.ndarray,torch.Tensor]):
            model = self.__class__(**{**self.model_dump(exclude=['lib','device','dtype']),'data':data})
            model.validate()
            return self.controller.update(**{**model.model_dump(),'data':model.data}).model

        def unsafe_update_data(self,data:Union[np.ndarray,torch.Tensor]):
            self.data = data
            return self
    
        @staticmethod
        def get_mat_ops(lib: Union[str, MatLib]) -> MatOps:
            mat_lib = MatLib(lib)
            if mat_lib == MatLib.NUMPY:
                return NumpyMatOps()
            if mat_lib == MatLib.TORCH:
                return TorchMatOps()
            raise ValueError(f"Unsupported matrix library: {lib}")


    class MatPubSub(Mat):
        lib: str = "iceoryx2"
        dtype: DataType = DataType.FLOAT32
        device: MatDevice = MatDevice.CPU

        data: Union[np.ndarray] = Field(
            default_factory=lambda: np.random.rand(5, 5).astype(np.float32),
            exclude=True,
        )
        is_pub: bool = Field(default=False)

        model_config = ConfigDict(arbitrary_types_allowed=True)

        _ops: MatOps = PrivateAttr(default=None)
        _node: Any = PrivateAttr(default=None)
        _service: Any = PrivateAttr(default=None)
        _pub: Any = PrivateAttr(default=None)
        _sub: Any = PrivateAttr(default=None)
        _np_type: Any = PrivateAttr(default=None)
        _c_type: Any = PrivateAttr(default=None)
        _n_elements: int = PrivateAttr(default=0)
        _last_sample: Any = PrivateAttr(default=None)

        def model_post_init(self, context):
            self._refresh_pubsub_layout()
            return super().model_post_init(context)

        def _refresh_pubsub_layout(self):
            """Refresh cached slice metadata from ``self.data``.

            This makes ``obj.set_id(...).init()`` safe after construction and
            keeps the iceoryx2 slice length/dtype aligned with the current
            NumPy array.  Use ``self.data`` directly here; calling
            ``get_data()`` would loan a publish sample when ``is_pub=True``.
            """
            if not isinstance(self.data, np.ndarray):
                # CUDA subclasses may be constructed with a CuPy array before
                # their own ``init`` runs.  They publish a fixed ctypes control
                # message, not a CPU slice, so there is no CPU slice layout to
                # refresh here.
                if hasattr(self.data, "__cuda_array_interface__"):
                    return self
                raise TypeError(f"MatPubSub expects a NumPy array, got {type(self.data)!r}")
            if not self.data.flags.c_contiguous:
                self.data = np.ascontiguousarray(self.data)
            self._n_elements = int(np.prod(self.data.shape))
            self._np_type = None
            self._c_type = None
            return self

        def init(self):
            self._refresh_pubsub_layout()
            return super().init()

        def topic_name(self) -> str:
            return self.get_id().replace(":", "/")

        def topic2id(self, topic: str) -> str:
            return topic.replace("/", ":")

        def _np_dtype(self) -> np.dtype:
            if self._np_type is None:
                self._np_type = np.dtype(to_np_type(self.dtype))
            return self._np_type

        def _ctypes_type(self):
            if self._c_type is None:
                self._c_type = to_ctypes_type(self.dtype)
            return self._c_type

        def _slice_cls(self):
            return iox2.Slice[self._ctypes_type()]

        def _slice_to_numpy(self, payload) -> np.ndarray:
            """
            Return a NumPy view over an iceoryx2 Slice payload.

            Important:
            - The returned array is only valid while the sample/loan is alive.
            - Use `shape`, not payload.len(), as the source of truth.
            """
            c_type = self._ctypes_type()
            n = self._n_elements

            ptr = ctypes.cast(
                int(payload.as_ptr()),
                ctypes.POINTER(c_type),
            )

            arr = np.ctypeslib.as_array(ptr, shape=(n,))
            return arr.reshape(self.shape())

        def _get_node(self):
            if self._node is None:
                self._node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
            return self._node

        def _get_service(self):
            if self._service is None:
                self._service = (
                    self._get_node()
                    .service_builder(iox2.ServiceName.new(self.topic_name()))
                    .publish_subscribe(self._slice_cls())
                    .open_or_create()
                )
            return self._service

        def get_pub(self):
            if self._pub is None:
                max_len = self._n_elements

                self._pub = (
                    self._get_service()
                    .publisher_builder()
                    .initial_max_slice_len(max_len)
                    .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo)
                    .create()
                )

            return self._pub

        def get_sub(self):
            if self._sub is None:
                self._sub = (
                    self._get_service()
                    .subscriber_builder()
                    .create()
                )

            return self._sub

        def get_data(self):
            if not self.is_pub:
                return self.data
            
            sample = self.get_pub().loan_slice_uninit(self._n_elements)
            self.data = self._slice_to_numpy(sample.payload())
            return sample,self.data
        
        def _coerce_publish_data(self, data) -> np.ndarray:
            arr = np.asarray(data, dtype=self._np_dtype(), order="C")
            if arr.shape != self.shape():
                raise ValueError(
                    f"published data shape must be {self.shape()}, got {arr.shape}. "
                    "Create publisher/subscriber with the same shape on both sides."
                )
            return arr

        def pub(self, data=None, edit_func=None):
            """Publish one NumPy matrix through iceoryx2 shared memory.

            Supported fast path used by your demo::

                img.pub(edit_func=draw)

            ``edit_func`` receives the borrowed shared-memory NumPy array.  It
            may edit in-place and may also return an array; a non-``None``
            return value is copied back into the shared-memory loan.
            """
            self.is_pub = True
            sample = self.get_pub().loan_slice_uninit(self._n_elements)
            out = self._slice_to_numpy(sample.payload())

            if data is not None:
                out[...] = self._coerce_publish_data(data)
            elif edit_func is None:
                # No edit function means "publish the current model data".
                out[...] = self._coerce_publish_data(self.data)

            if edit_func is not None:
                edited = edit_func(out)
                if edited is not None and edited is not out:
                    out[...] = self._coerce_publish_data(edited)

            sample.assume_init().send()
            return self

        def sub(self, copy=False):
            """Receive one matrix.

            ``copy=False`` keeps the received iceoryx2 sample alive on the model
            so ``img.sub().get_data()`` returns a valid NumPy view for OpenCV.
            Use ``copy=True`` when you want a detached array that remains valid
            after the next ``sub()`` call.
            """
            self.is_pub = False
            sample = self.get_sub().receive()
            if sample is None:
                return self

            # Keep the sample alive while exposing a zero-copy NumPy view.
            # Without this, the local `sample` can be destroyed before cv2.imshow
            # consumes `img.sub().get_data()`.
            self._last_sample = sample
            view = self._slice_to_numpy(sample.payload())
            self.unsafe_update_data(view.copy() if copy else view)
            return self

    class ImageMat(Mat):
        class ColorFormat(str, enum.Enum):
            RGB = "RGB"
            BGR = "BGR"
            GRAY = "GRAY"
            BAYER = "BAYER"
            UNKNOWN = "UNKNOWN"

            @classmethod
            def channels(cls, color_format) -> int:
                color_format = cls(color_format)
                channel_counts = {
                    cls.RGB: 3,
                    cls.BGR: 3,
                    cls.GRAY: 1,
                    cls.BAYER: 1,
                }
                if color_format not in channel_counts:
                    raise ValueError(f"Unknown color format: {color_format}")
                return channel_counts[color_format]

        class ShapeType(str, enum.Enum):
            BCHW = "BCHW"  # torch, float, [0, 1]
            HW = "HW"      # numpy, uint8, grayscale
            BHW = "BHW"    # numpy, uint8, grayscale batch
            HWC = "HWC"    # numpy, uint8, channels-last, C>1
            BHWC = "BHWC"  # numpy, uint8, batch + channels-last, C>1
            UNKNOWN = "UNKNOWN"

            @staticmethod
            def to_bchw(shape_type, data) -> tuple[int, int, int, int]:
                shape = tuple(int(v) for v in data.shape)
                key = shape_type.value if isinstance(shape_type, enum.Enum) else str(shape_type)

                expected_ndim = {
                    "BCHW": 4,
                    "HW": 2,
                    "BHW": 3,
                    "HWC": 3,
                    "BHWC": 4,
                }
                if key not in expected_ndim:
                    raise ValueError(f"Cannot convert unknown shape type to BCHW: {shape_type}")

                if len(shape) != expected_ndim[key]:
                    raise ValueError(
                        f"Shape type {shape_type} expects {expected_ndim[key]} dimensions, "
                        f"got shape {shape}"
                    )

                if key == "BCHW":
                    b, c, h, w = shape
                elif key == "HW":
                    h, w = shape
                    b, c = 1, 1
                elif key == "BHW":
                    b, h, w = shape
                    c = 1
                elif key == "HWC":
                    h, w, c = shape
                    b = 1
                else:  # BHWC
                    b, h, w, c = shape

                return b, c, h, w

        color_format: ColorFormat = ColorFormat.GRAY
        shape_type: ShapeType = ShapeType.UNKNOWN
        dtype: DataType = DataType.UNKNOWN
        BCHW: tuple[int, int, int, int] = Field(default=(0, 0, 0, 0))
        path: Optional[str] = None

        data: Union[np.ndarray, torch.Tensor] = Field(
            default_factory=lambda: np.zeros((1, 1), dtype=np.uint8),
            exclude=True,
        )


        def safe_update_data(self,data:Union[np.ndarray,torch.Tensor]):
            model = self.__class__(**{'color_format':Model4Mat.ImageMat.ColorFormat.UNKNOWN,'data':data})
            return self.controller.update(**{**model.model_dump(),'data':model.data}).model
        
        def init(self):
            super().init()

            cls = type(self)
            self.color_format = cls.ColorFormat(self.color_format)
            self.shape_type = cls.ShapeType(self.shape_type)

            if self.shape_type == cls.ShapeType.UNKNOWN:
                self.shape_type = self._infer_shape_type()

            B,C,H,W = self.BCHW = cls.ShapeType.to_bchw(self.shape_type, self.get_data())

            if self.color_format == cls.ColorFormat.UNKNOWN:
                if C == 1:
                    self.color_format = cls.ColorFormat.GRAY # or Bayer
                elif C == 3:
                    self.color_format = cls.ColorFormat.RGB # or BGR
                else:
                    raise ValueError(f"Unsupported number of channels: {C}")
                
            return self

        def _infer_shape_type(self):
            cls = type(self)
            shape = self.shape()
            ndim = len(shape)

            if ndim == 2:
                return cls.ShapeType.HW

            if self.lib == MatLib.TORCH:
                if ndim == 4:
                    return cls.ShapeType.BCHW
                raise ValueError(f"Torch image data must be BCHW, got shape {shape}")

            if self.lib != MatLib.NUMPY:
                raise TypeError(f"Unsupported image backend: {self.lib}")

            if ndim == 3:
                if self.color_format == cls.ColorFormat.UNKNOWN:
                    raise ValueError(
                        "Cannot infer 3-D NumPy image layout. "
                        "Set shape_type to BHW or HWC and set color_format."
                    )

                expected_channels = cls.ColorFormat.channels(self.color_format)
                if shape[-1] == expected_channels:
                    return cls.ShapeType.HWC
                if expected_channels == 1:
                    return cls.ShapeType.BHW

                raise ValueError(
                    f"Cannot infer 3-D NumPy image layout for shape {shape} "
                    f"and color_format={self.color_format}"
                )

            if ndim == 4:
                return cls.ShapeType.BHWC

            raise ValueError(f"Unsupported image shape: {shape}")

        @staticmethod
        def _to_scalar(value) -> float:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            return float(value)

        def validate(self):
            cls = type(self)
            b, c, h, w = self.BCHW

            if min(b, c, h, w) <= 0:
                raise ValueError(f"Invalid BCHW dimensions: {self.BCHW}")

            if self.color_format == cls.ColorFormat.UNKNOWN:
                raise TypeError("color_format must be RGB, BGR, GRAY, or BAYER")

            expected_channels = cls.ColorFormat.channels(self.color_format)
            if c != expected_channels:
                raise TypeError(
                    f"Expected {expected_channels} channels for {self.color_format}, "
                    f"got {c} from shape_type={self.shape_type} and shape={self.shape()}"
                )

            if self.lib == MatLib.TORCH:
                if self.shape_type != cls.ShapeType.BCHW:
                    raise TypeError(f"Expected BCHW shape for torch data, got {self.shape_type}")
                if self.dtype not in {DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16}:
                    raise TypeError(f"Expected float dtype for torch data, got {self.dtype}")

                if not bool(torch.isfinite(self.get_data()).all()):
                    raise ValueError("Torch image data contains NaN or infinite values")

                mi = self._to_scalar(self.get_ops().min(self.get_ops().flatten(self.get_data())))
                ma = self._to_scalar(self.get_ops().max(self.get_ops().flatten(self.get_data())))
                if mi < 0.0 or ma > 1.0:
                    raise ValueError(f"Expected torch image values in [0, 1], got {mi}~{ma}")

            elif self.lib == MatLib.NUMPY:
                if self.shape_type == cls.ShapeType.BCHW:
                    raise TypeError(f"Expected HW, BHW, HWC, or BHWC for NumPy data, got {self.shape_type}")
                if self.dtype != DataType.UINT8:
                    raise TypeError(f"Expected uint8 dtype for NumPy data, got {self.dtype}")

            else:
                raise TypeError(f"Unsupported matrix backend: {self.lib}")

        def size(self):
            B,C,Height,Width = self.BCHW
            return Width, Height
        
        
        # local edit
        def to_numpy(self, tmp=False):
            if self.lib == MatLib.NUMPY:return self
            B,C,H,W = self.BCHW
            data = (self.data*255.0).to(dtype=torch.uint8).detach().cpu()
            shape_type = self.ShapeType.UNKNOWN
            if C==1:
                data = data.squeeze(1)
                shape_type = self.ShapeType.BHW
                if data.shape[0]==1:
                    data = data.squeeze(0)
                    shape_type = self.ShapeType.HW
            else:
                data = data.permute(0,2,3,1)
                shape_type = self.ShapeType.BHWC
                if data.shape[0]==1:
                    data = data.squeeze(0)
                    shape_type = self.ShapeType.HWC

            data = data.numpy()
            if not tmp:
                self.shape_type = shape_type
                return self.safe_update_data(data)
            else:
                return self.model_copy(update={'data':data, 'shape_type':shape_type})
        
        def to_torch(self,device=None,dtype=torch.float32, tmp=False):
            if self.lib == MatLib.TORCH:return self
            B,C,H,W = self.BCHW
            data = torch.from_numpy(self.data.copy()).to(device=device).div(255.0).to(dtype=dtype)
            if self.shape_type == self.ShapeType.HWC:
                data = data.permute(2,0,1).unsqueeze(0)
            elif self.shape_type == self.ShapeType.BHWC:
                data = data.permute(0,3,1,2)
            elif self.shape_type == self.ShapeType.BHW:
                data = data.unsqueeze(1)
            elif self.shape_type == self.ShapeType.HW:
                data = data.unsqueeze(0).unsqueeze(0)
            
            if not tmp:
                self.shape_type = self.ShapeType.BCHW
                return self.safe_update_data(data)
            else:
                return self.model_copy(update={'data':data, 'shape_type':self.ShapeType.BCHW})
            
        @staticmethod
        def from_url(url:str,color_format=ColorFormat.RGB):
            if url.startswith("http"):
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(url)
            data = np.asarray(img)
            return Model4Mat.ImageMat(
                color_format=color_format,
                data=data)
        
        def pil_show(self,title=None):
            tmp = self.to_numpy(tmp=True)
            img = Image.fromarray(tmp.data)
            img.show(title=title)

        def crop_bbox(self,bbox:"Model4Mat.BoundingBox", copy=True):
            xyxy = bbox.get_raw_xyxy()
            x1,y1,x2,y2 = xyxy.T
            ops = self.get_ops()
            res = []
            for i in range(len(xyxy)):
                crop = self.get_data()[int(y1[i]):int(y2[i]),int(x1[i]):int(x2[i])]
                crop = ops.copy_mat(crop) if copy else crop
                img = Model4Mat.ImageMat(**{'color_format':self.color_format,
                                        'shape_type':self.shape_type,
                                        'dtype':self.dtype,
                                        'BCHW':self.BCHW,
                                        'data':crop})
                res.append(img)
            return res
        
        def crop_by_children(self)->List["Model4Mat.ImageMat"]:
            res = []
            for child, depth in self.yield_children_recursive():
                if isinstance(child, Model4Mat.BoundingBox):
                    res += self.crop_bbox(child)
            return res

    class ImageMatView(ImageMat):
        class Mode(str, enum.Enum):
            HWxyxy = "HWxyxy"
            HWxyhw = "HWxyhw"
            ALL = "ALL"

        class ScaleFormat(str, enum.Enum):
            ZERO_ONE = "01"
            RAW = "raw"
        
        target_img_id: str=None
        view2target3x3: list = Field(default_factory=list)
        target2view3x3: list = Field(default_factory=list)
        # axis info , target mat=a[:,:,:], data = [(from_a,to_a),(from_b,to_b),(from_c,to_c)...]
        data: Union[np.ndarray, torch.Tensor] = Field(
            default_factory=lambda: np.zeros((1, 2), dtype=np.int32),
            exclude=True,
        )
        mode:Mode = Mode.HWxyxy
        scale:ScaleFormat = ScaleFormat.RAW
        _ranges:Dict = None

        def model_post_init(self, context):
            self.target_img_id = self.controller.model.get_id()
            self._ranges={}
            return super().model_post_init(context)
        
        def calc_hw_range(self,data,img_size):
            width, height = img_size

            if self.mode == self.Mode.HWxyxy:
                if self.scale == self.ScaleFormat.ZERO_ONE:
                    (H_from, W_from), (H_to, W_to) = data
                    H_from *= height
                    H_to *= height
                    W_from *= width
                    W_to *= width
                else:
                    (H_from, W_from), (H_to, W_to) = data

            elif self.mode == self.Mode.HWxyhw:
                if self.scale == self.ScaleFormat.ZERO_ONE:
                    (H_from, W_from), (H_len, W_len) = data
                    H_to = H_from + H_len * height
                    W_to = W_from + W_len * width
                else:
                    (H_from, W_from), (H_len, W_len) = data
                    H_to = H_from + H_len
                    W_to = W_from + W_len

            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            res = (int(H_from), int(H_to), int(W_from), int(W_to))
            return res

        def unsafe_get_data(self):
            img:Model4Mat.ImageMat = self.controller.storage().get(self.target_img_id)
            res_func,H_from,H_to,W_from,W_to = self._ranges["last"]
            return res_func(img.get_data(),H_from,H_to,W_from,W_to)
        
        def get_data(self):
            img:Model4Mat.ImageMat = self.controller.storage().get(self.target_img_id)
            self.color_format = img.color_format
            self.shape_type = img.shape_type
            self.dtype = img.dtype
            self.BCHW = img.BCHW

            if self.mode == self.Mode.ALL:
                return self.view_from_ranges(img.get_data(), self.data)
        
            (a,b),(c,d) = self.data
            data = ((a,b),(c,d))
            key = (img.shape_type, img.shape(), self.mode, data)
            res = self._ranges.get(key)
            if res is not None:
                res_func,H_from,H_to,W_from,W_to = res
                self.target2view3x3 = [
                    [1, 0, -W_from],
                    [0, 1, -H_from],
                    [0, 0, 1]]
                self.view2target3x3 = [
                    [1, 0, W_from],
                    [0, 1, H_from],
                    [0, 0, 1]]                
                self._ranges["last"] = res
                return res_func(img.get_data(),H_from,H_to,W_from,W_to)

            H_from,H_to,W_from,W_to = self.calc_hw_range(self.data, img.size())
            self.target2view3x3 = [
                [1, 0, -W_from],
                [0, 1, -H_from],
                [0, 0, 1]]
            self.view2target3x3 = [
                [1, 0, W_from],
                [0, 1, H_from],
                [0, 0, 1]]
            
            if img.shape_type == Model4Mat.ImageMat.ShapeType.BCHW:
                res_func = lambda data,H_from,H_to,W_from,W_to:data[:,:,H_from:H_to,W_from:W_to]
            elif img.shape_type in [Model4Mat.ImageMat.ShapeType.HWC,Model4Mat.ImageMat.ShapeType.HW]:
                res_func = lambda data,H_from,H_to,W_from,W_to:data[H_from:H_to,W_from:W_to]
            elif img.shape_type in [Model4Mat.ImageMat.ShapeType.BHWC,Model4Mat.ImageMat.ShapeType.BHW]:
                res_func = lambda data,H_from,H_to,W_from,W_to:data[:,H_from:H_to,W_from:W_to]
            else:
                raise ValueError(f"Unsupported shape type: {img.shape_type}")
            
            self._ranges["last"] = self._ranges[key] = res_func,H_from,H_to,W_from,W_to
            return res_func(img.get_data(),H_from,H_to,W_from,W_to)
            
        @staticmethod
        def view_from_ranges(mat, idx):
            """
            idx: sequence of (start, stop) pairs, one per axis.
            Returns a NumPy view when only slices are used.
            """
            key = tuple(slice(start, stop) for start, stop in idx)
            return mat[key]
        
        def to_numpy(self, tmp=False):raise ValueError("ImageMatView not supported!")        
        def to_torch(self,device=None,dtype=torch.float32, tmp=False):raise ValueError("ImageMatView not supported!")        
        @staticmethod
        def from_url(url:str,color_format=None):raise ValueError("ImageMatView not supported!")
        
        def get_data_numpy(self):
            data = self.get_data()
            if isinstance(data,np.ndarray):return data
            B,C,H,W = self.BCHW
            data = (data*255.0).to(dtype=torch.uint8).detach().cpu()
            if C==1:
                data = data.squeeze(1)
                if data.shape[0]==1:
                    data = data.squeeze(0)
            else:
                data = data.permute(0,2,3,1)
                if data.shape[0]==1:
                    data = data.squeeze(0)
            return data.numpy()
        
        def pil_show(self,title=None):
            tmp = self.get_data_numpy()
            img = Image.fromarray(tmp)
            img.show(title=title)

    class ImageMatPubSub(MatPubSub,ImageMat):
        pass

    try:
        class ImageMatCUDAPubSub(ImageMatCUDAPubSub,ImageMatPubSub):
            dtype: DataType = DataType.FLOAT32
            device: MatDevice = MatDevice.CUDA0
            model_config = ConfigDict(arbitrary_types_allowed=True)
            def init(self):
                if isinstance(self.data, gpuarray.GPUArray):
                    self.lib = "pycuda"
                    self.device = MatDevice.CUDA if hasattr(MatDevice, "CUDA") else self.device
                    self._update_image_metadata_from_array(self.data)
                    self.validate()
                    return self
                return super().init()

            def validate(self):
                if not isinstance(self.data, gpuarray.GPUArray):
                    return super().validate()
                
                cls = Model4Mat.ImageMat
                b, c, h, w = self.BCHW
                if min(b, c, h, w) <= 0:
                    raise ValueError(f"Invalid BCHW dimensions: {self.BCHW}")

                expected_channels = cls.ColorFormat.channels(self.color_format)
                if c != expected_channels:
                    raise TypeError(
                        f"Expected {expected_channels} channels for {self.color_format}, "
                        f"got {c} from shape_type={self.shape_type} and shape={self.shape()}"
                    )

                np_dtype = np.dtype(self.data.dtype)
                if self.shape_type == cls.ShapeType.BCHW:
                    if np_dtype not in {np.dtype("float16"), np.dtype("float32")}:
                        raise TypeError(f"Expected float dtype for BCHW PyCUDA image data, got {np_dtype}")
                else:
                    if np_dtype != np.dtype("uint8"):
                        raise TypeError(f"Expected uint8 dtype for PyCUDA image data, got {np_dtype}")

            def _update_image_metadata_from_array(self, arr):
                cls = Model4Mat.ImageMat
                self.color_format = cls.ColorFormat(self.color_format)
                self.shape_type = cls.ShapeType(self.shape_type)

                if self.shape_type == cls.ShapeType.UNKNOWN:
                    ndim = int(len(arr.shape))
                    if ndim == 2:
                        self.shape_type = cls.ShapeType.HW
                    elif ndim == 3:
                        if self.color_format == cls.ColorFormat.UNKNOWN:
                            raise ValueError(
                                "Cannot infer 3-D PyCUDA image layout. "
                                "Set shape_type to BHW or HWC and set color_format."
                            )
                        expected = cls.ColorFormat.channels(self.color_format)
                        if int(arr.shape[-1]) == expected:
                            self.shape_type = cls.ShapeType.HWC
                        elif expected == 1:
                            self.shape_type = cls.ShapeType.BHW
                        else:
                            raise ValueError(
                                f"Cannot infer 3-D PyCUDA layout for shape={arr.shape} "
                                f"and color_format={self.color_format}"
                            )
                    elif ndim == 4:
                        self.shape_type = cls.ShapeType.BHWC
                    else:
                        raise ValueError(f"Unsupported PyCUDA image shape: {arr.shape}")

                b, c, h, w = self.BCHW = cls.ShapeType.to_bchw(self.shape_type, arr)

                if self.color_format == cls.ColorFormat.UNKNOWN:
                    if c == 1:
                        self.color_format = cls.ColorFormat.GRAY
                    elif c == 3:
                        self.color_format = cls.ColorFormat.RGB
                    else:
                        raise ValueError(f"Unsupported number of channels: {c}")

                try:
                    self.dtype = DataType.which(np.empty((0,), dtype=np.dtype(arr.dtype)))
                except Exception:
                    pass

    except Exception:
        pass

    class BoundingBox(Mat):        
        class AxisFormat(str, enum.Enum):
            XYXY = "xyxy"
            XYWH = "xywh"
            CXCYWH = "cxcywh"
        class ScaleFormat(str, enum.Enum):
            ZERO_ONE = "01"
            RAW = "raw"
        
        data: Union[np.ndarray,torch.Tensor] = Field(default_factory=lambda:np.random.rand(1,4), exclude=True)        
        labels_id:Optional[Union[np.ndarray,torch.Tensor]] = None
        scores:Optional[Union[np.ndarray,torch.Tensor]] = None
        labels:List[str] = Field(default_factory=list)

        format: AxisFormat = AxisFormat.XYXY
        scale: ScaleFormat = ScaleFormat.ZERO_ONE
        image_size: Optional[Tuple[int, int]] = None  # width, height        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

        def validate(self):
            shape = self.shape()
            if len(shape) != 2: raise ValueError(f"Expected bounding box shape dim is 2, got {shape}")
            if shape[1] != 4: raise ValueError(f"Expected bounding box shape is (n, 4), got {shape}")
            mi = self.get_ops().min(self.get_ops().flatten(self.data))
            ma = self.get_ops().max(self.get_ops().flatten(self.data))
            if self.scale==self.ScaleFormat.ZERO_ONE and (mi < 0.0 or ma > 1.0):
                raise ValueError(f"Expected bounding box values in [0, 1], got {mi}~{ma}")
            if self.scale==self.ScaleFormat.RAW and ma <= 1.0:
                raise ValueError(f"Expected bounding box values in raw pixels, got {mi}~{ma}")
                
        def get_raw_abcd(self, data, type):
            if self.format==type:return self.data
            ops = self.get_ops()
            a,b,c,d = getattr(ops,f"from_{self.format.value}_to_{type.value}")(data)
            return ops.stack([a,b,c,d],dim=1)
        
        def get_raw_xywh(self):return self.get_raw_abcd(self.data,self.AxisFormat.XYWH)
        def get_raw_xyxy(self):return self.get_raw_abcd(self.data,self.AxisFormat.XYXY)
        def get_raw_cxcywh(self):return self.get_raw_abcd(self.data,self.AxisFormat.CXCYWH)

        # local edit
        def to_abcd(self, data, type):
            data = self.get_raw_abcd(data, type)
            self.format = type
            return self.controller.update(**{**self.model_dump(),'data':data}).model
        
        def to_xywh(self):return self.to_abcd(self.data,self.AxisFormat.XYWH)
        def to_xyxy(self):return self.to_abcd(self.data,self.AxisFormat.XYXY)
        def to_cxcywh(self):return self.to_abcd(self.data,self.AxisFormat.CXCYWH)

        def to_scale(self,scale):
            if self.scale==scale:return self
            if self.image_size is None:
                raise ValueError("image_size must be set before scaling")
            if scale not in [self.ScaleFormat.ZERO_ONE,self.ScaleFormat.RAW]:                
                raise ValueError(f"Unknown scale format: {scale}")      
            
            f = self.format
            model:Model4Mat.BoundingBox = self.to_xyxy()
            xyxy = model.data
            width, height = self.image_size

            if scale==self.ScaleFormat.ZERO_ONE:
                xyxy[:, [0, 2]] /= float(width)
                xyxy[:, [1, 3]] /= float(height)
            elif scale==self.ScaleFormat.RAW:
                xyxy[:, [0, 2]] *= float(width)
                xyxy[:, [1, 3]] *= float(height)
                
            model.data = xyxy
            model.scale = scale
            model = model.to_abcd(model.data,f)
            return self.controller.update(**{**self.model_dump(),'data':model.data}).model

        def area(self) -> np.ndarray:
            """Return the area of each box in the current scale."""
            x1, y1, x2, y2 = self.get_ops().__dict__[f"from_{self.format}_to_xyxy"](self.data)
            return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        
        def iou(self, other: "Model4Mat.BoundingBox") -> np.ndarray:
            """Pairwise IoU matrix with another Box object."""
            a = self.get_ops().__dict__[f"from_{self.format}_to_xyxy"](self.data)
            b = other.get_ops().__dict__[f"from_{other.format}_to_xyxy"](other.data)

            if self.scale != other.scale:
                raise ValueError("cannot compute IoU between different scales")
            a_area = self.area()[:, None]
            b_area = other.area()[None, :]
            lt = np.maximum(a[:, None, :2], b[None, :, :2])
            rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
            wh = np.clip(rb - lt, 0, None)
            inter = wh[:, :, 0] * wh[:, :, 1]
            union = a_area + b_area - inter
            return np.where(union > 0, inter / union, 0.0).astype(np.float32)
    
class MatStore(BasicStore):
    MODEL_CLASS_GROUP = Model4Mat
    def _get_class(self, id: str, modelclass=MODEL_CLASS_GROUP):
        class_type = id.split(':')[0]
        res = [i for k,i in modelclass.__dict__.items() if '_' not in k]
        res = {c.__name__:c for c in res}
        res = res.get(class_type, None)
        if res is None: raise ValueError(f'No such class of {class_type}')
        return res
        
    def _get_as_obj(self,id,data_dict)->MODEL_CLASS_GROUP.AbstractObj:
        if data_dict is None : return None
        if isinstance(data_dict,dict):
            obj:Model4Basic.AbstractObj = self._get_class(id)(**data_dict)
        else:
            obj:Model4Basic.AbstractObj = data_dict
            obj.set_id(None,ast=False)
        obj.set_id(id).init_controller(self)
        return obj
    
    def find(self,id:str, fa:bool=True) -> MODEL_CLASS_GROUP.AbstractObj:
        if self.exists(id): return self._get_as_obj(id, self.get(id) )
        res = self.find_all(f'*:{id}') if fa else []
        return res[0] if len(res) == 1 else None
    