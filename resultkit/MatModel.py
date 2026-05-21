# from https://github.com/qinhy/singleton-key-value-storage.git
import enum
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from pydantic import ConfigDict, Field
import torch
import requests
from PIL import Image
from io import BytesIO

try:    
    from BasicModel import Controller4Basic, Model4Basic, BasicStore
    from mat import DataType, MatOps, NumpyMatOps, TorchMatOps, MatLib, MatDevice
except Exception as e:    
    from .BasicModel import Controller4Basic, Model4Basic, BasicStore
    from .mat import DataType, MatOps, NumpyMatOps, TorchMatOps, MatLib, MatDevice

class Controller4Mat:
    class AbstractObjController(Controller4Basic.AbstractObjController):pass        
    class AbstractGroupController(Controller4Basic.AbstractGroupController):pass  
    class MatController(AbstractGroupController):pass        
    class ImageMatController(MatController):pass        
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
            self.init(self.data)
            self.validate()
            return super().model_post_init(context)

        def get_ops(self):
            if self._ops is None:
                self._ops = self.get_mat_ops(self.lib)
            return self._ops

        def init(self, data):            
            self.lib = MatLib.which(data)
            self.device = MatDevice.which(data)
            self.dtype = DataType.which(data)
            return self
        
        def validate(self):
            pass

        def shape(self):return self.data.shape

        def update(self, **kwargs):
            return self.controller.update(**kwargs).model

        def safe_update_data(self,data:Union[np.ndarray,torch.Tensor]):
            model = self.__class__(**{**self.model_dump(),'data':data})
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
        path: Optional[str] = None
        data: Union[np.ndarray, torch.Tensor] = Field(
            default_factory=lambda: np.zeros((1, 1), dtype=np.uint8),
            exclude=True,
        )
        BCHW: tuple[int, int, int, int] = Field(default=(0, 0, 0, 0))

        def init(self, data):
            super().init(data)

            cls = type(self)
            self.color_format = cls.ColorFormat(self.color_format)
            self.shape_type = cls.ShapeType(self.shape_type)

            if self.shape_type == cls.ShapeType.UNKNOWN:
                self.shape_type = self._infer_shape_type()

            B,C,H,W = self.BCHW = cls.ShapeType.to_bchw(self.shape_type, self.data)

            if self.color_format == cls.ColorFormat.UNKNOWN:
                if C == 1:
                    self.color_format = cls.ColorFormat.GRAY
                elif C == 3:
                    self.color_format = cls.ColorFormat.RGB
                else:
                    raise ValueError(f"Unsupported number of channels: {C}")
                
            self.data = data
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

                if not bool(torch.isfinite(self.data).all()):
                    raise ValueError("Torch image data contains NaN or infinite values")

                mi = self._to_scalar(self.get_ops().min(self.get_ops().flatten(self.data)))
                ma = self._to_scalar(self.get_ops().max(self.get_ops().flatten(self.data)))
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
            if C==1:
                data = data.squeeze(1).numpy()
            else:
                data = data.permute(0,2,3,1).numpy()

            self.shape_type = self.ShapeType.UNKNOWN
            if tmp:
                return self.safe_update_data(data)
            else:
                res = self.model_copy()
                res.data = data
                return res
        
        def to_torch(self,device=None,dtype=torch.float32, tmp=False):
            if self.lib == MatLib.TORCH:return self
            B,C,H,W = self.BCHW
            data = torch.from_numpy(self.data).to(device=device).div(255.0).to(dtype=dtype)
            if self.shape_type == self.ShapeType.HWC:
                data = data.permute(2,0,1).unsqueeze(0)
            elif self.shape_type == self.ShapeType.BHWC:
                data = data.permute(0,3,1,2)
            elif self.shape_type == self.ShapeType.BHW:
                data = data.unsqueeze(1)
            elif self.shape_type == self.ShapeType.HW:
                data = data.unsqueeze(0).unsqueeze(0)
            
            self.shape_type = self.ShapeType.UNKNOWN
            if tmp:
                return self.safe_update_data(data)
            else:
                res = self.model_copy()
                res.data = data
                return res
        
        @staticmethod
        def from_url(url,color_format=ColorFormat.RGB):
            if url.startswith("http"):
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(url)
            data = np.asarray(img)
            return Model4Mat.ImageMat(
                color_format=color_format,
                data=data)
        
        def pil_show(self):
            tmp = self.to_numpy(tmp=True)
            img = Image.fromarray(tmp.data)
            img.show()

        def crop_bbox(self,bbox:"Model4Mat.BoundingBox", copy=True):
            xyxy = bbox.get_raw_xyxy()
            x1,y1,x2,y2 = xyxy.T
            ops = self.get_ops()
            res = []
            for i in range(len(xyxy)):
                img = self.model_copy()
                crop = self.data[int(y1[i]):int(y2[i]),int(x1[i]):int(x2[i])]
                img.init(ops.copy_mat(crop) if copy else crop)
                img.validate()
                res.append(img)
            return res
        
        def crop_by_children(self):
            res = []
            for child, depth in self.yield_children_recursive():
                if isinstance(child, Model4Mat.BoundingBox):
                    res += self.crop_bbox(child)
            return res


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
    
if __name__ == '__main__':
    store = MatStore()
    mat = store.add_new_obj(Model4Mat.ImageMat(
                color_format=Model4Mat.ImageMat.ColorFormat.GRAY,
                data=np.ones((4,4),dtype=np.uint8)))
    
    print(mat.get_id())
    print(mat.data)

    mat = mat.to_torch()
    print(mat.get_id())
    print(mat.data)

    mat = mat.to_numpy()
    print(mat.get_id())
    print(mat.data)

    cmat = store.add_new_obj(Model4Mat.BoundingBox())
    mat.controller.add_child(cmat.get_id())
    print(cmat.get_ops())
    cmat.to_cxcywh()
    print(store.get(mat.get_id()))
    print(store.get(cmat.get_id()))

    
    # BoundingBox, Keypoints, ResultNode, ResultSet, TextSpan, Vector
    lib = MatStore()
    BoundingBox = Model4Mat.BoundingBox
    ColorFormat = Model4Mat.ImageMat.ColorFormat

    img = lib.add_new_obj(Model4Mat.ImageMat.from_url("./examples/img1.jpg",
                                                    color_format=ColorFormat.RGB))

    person_bbox = lib.add_new_obj(BoundingBox(data=np.array([[10, 20, 220, 440]], dtype=np.float32),
                                            labels=['person'],
                                            labels_id=np.array([0], dtype=np.int32),
                                            scores=np.array([0.98], dtype=np.float32),
                                            scale=BoundingBox.ScaleFormat.RAW,
                                            format=BoundingBox.AxisFormat.XYXY,
                                            image_size=img.size()))

    img.controller.add_child(person_bbox.get_id())

    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    results = yolo.predict(img.data)[0] #one image

    person_bbox.update(data=results.boxes.xyxy,
                    labels_id=results.boxes.cls,
                    scores=results.boxes.conf,
                    labels=[results.names[int(i)] for i in results.boxes.cls],
                    )
