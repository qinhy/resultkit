
import os
import sys
import numpy as np
from ultralytics import YOLO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resultkit.MatModel import ColorFormat, Model4Mat, MatStore

store = MatStore.build()
BoundingBox = Model4Mat.BoundingBox

img = store.add_new_obj(Model4Mat.ImageMat.from_url("./examples/img1.jpg",
                                                color_format=ColorFormat.RGB))
imgv = store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0, 0],[0.5, 0.5]]),
                                                scale=Model4Mat.ImageMatView.MatScaleFormat.ZERO_ONE,
                                                mode=Model4Mat.ImageMatView.MatViewMode.HWxyxy,
                                                controller=img.controller))

det_bbox = store.add_new_obj(BoundingBox(data=np.array([[10, 20, 220, 440]], dtype=np.float32),
                                          labels_id=np.array([0], dtype=np.int32),
                                          scores=np.array([0.98], dtype=np.float32),
                                          scale=BoundingBox.ScaleFormat.RAW,
                                          format=BoundingBox.AxisFormat.XYXY,
                                          image_size=img.size()))
img.controller.add_child(det_bbox.get_id())

imgv = [store.add_new_obj(imgv.model_copy(update={'data':np.array([[0.0, 0.0],[0.5, 0.5]])})),
        store.add_new_obj(imgv.model_copy(update={'data':np.array([[0.5, 0.5],[1.0, 1.0]])})),
        store.add_new_obj(imgv.model_copy(update={'data':np.array([[0.0, 0.5],[0.5, 1.0]])})),
        store.add_new_obj(imgv.model_copy(update={'data':np.array([[0.5, 0.0],[1.0, 0.5]])})),
        ]
det_bbox = [store.add_new_obj(det_bbox.model_copy(update={'image_size':i.size()})) for i in imgv]

for i,b in zip(imgv,det_bbox):
    i.controller.add_child(b.get_id())

# [r.pil_show(f"split {i}") for i,r in enumerate(imgv)] #split into quarters
print(imgv)

def AI_detetions(imgs=imgv,det_bbox=det_bbox,yolo = YOLO("yolov8n.pt")):
    data = [i.unsafe_get_data() for i in imgs]
    res = yolo(data)
    for r,i,b in zip(res,imgs,det_bbox):
        b.update(data=r.boxes.xyxy.cpu().numpy(),
                labels_id=r.boxes.cls.cpu().numpy().astype(np.int32),
                scores=r.boxes.conf.cpu().numpy())
    return det_bbox
    
AI_detetions(imgv,det_bbox)

for i in imgv:
    [r.pil_show() for r in i.crop_by_children()]


# res = img.crop_bbox(person_bbox)
# [r.pil_show() for r in res]

# res = img.crop_bbox(face_bbox)
# [r.pil_show() for r in res]
