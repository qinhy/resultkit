
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resultkit.MatModel import Model4Mat, MatStore

store = MatStore.build()
BoundingBox = Model4Mat.BoundingBox
ColorFormat = Model4Mat.ImageMat.ColorFormat

img = store.add_new_obj(Model4Mat.ImageMat.from_url("./examples/img1.jpg",
                                                color_format=ColorFormat.RGB))

imgv = store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0, 0],[0.8, 0.9]]),
                                                scale=Model4Mat.ImageMatView.ScaleFormat.ZERO_ONE,
                                                mode=Model4Mat.ImageMatView.Mode.HWxyxy,
                                                controller=img.controller))

img = img.to_torch()

person_bbox = store.add_new_obj(BoundingBox(data=np.array([[10, 20, 220, 440]], dtype=np.float32),
                                          labels=['person'],
                                          labels_id=np.array([0], dtype=np.int32),
                                          scores=np.array([0.98], dtype=np.float32),
                                          scale=BoundingBox.ScaleFormat.RAW,
                                          format=BoundingBox.AxisFormat.XYWH,
                                          image_size=img.size()))

face_bbox = store.add_new_obj(person_bbox.model_copy())

person_bbox.controller.add_child(face_bbox.get_id())
imgv.controller.add_child(person_bbox.get_id())

def AI_detetion(person_bbox=person_bbox,face_bbox=face_bbox):
    person_bbox.update(data=np.array([[80,992,3488,2608]]),
                    labels_id=np.array([0]),
                    scores=np.array([0.9]),
                    labels=["person"],)
    face_bbox.update(data=np.array([[1416,1312,584,792]]),
                    labels_id=np.array([0]),
                    scores=np.array([0.9]),
                    labels=["face"],)
    return person_bbox,face_bbox

AI_detetion(person_bbox,face_bbox)

img.pil_show()
imgv.pil_show()

# print(np.shares_memory(img.get_data(),imgv.get_data()))

# res = imgv.crop_by_children()
# [r.pil_show() for r in res]

# res = img.crop_bbox(person_bbox)
# [r.pil_show() for r in res]

# res = img.crop_bbox(face_bbox)
# [r.pil_show() for r in res]
