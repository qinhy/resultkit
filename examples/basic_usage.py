
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resultkit.MatModel import Model4Mat, MatStore
# BoundingBox, Keypoints, ResultNode, ResultSet, TextSpan, Vector
store = MatStore.build()
BoundingBox = Model4Mat.BoundingBox
ColorFormat = Model4Mat.ImageMat.ColorFormat

img = store.add_new_obj(Model4Mat.ImageMat.from_url("./examples/img1.jpg",
                                                color_format=ColorFormat.RGB))

person_bbox = store.add_new_obj(BoundingBox(data=np.array([[10, 20, 220, 440]], dtype=np.float32),
                                          labels=['person'],
                                          labels_id=np.array([0], dtype=np.int32),
                                          scores=np.array([0.98], dtype=np.float32),
                                          scale=BoundingBox.ScaleFormat.RAW,
                                          format=BoundingBox.AxisFormat.XYWH,
                                          image_size=img.size()))

face_bbox = store.add_new_obj(person_bbox.model_copy())

person_bbox.controller.add_child(face_bbox.get_id())
img.controller.add_child(person_bbox.get_id())

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
res = img.crop_by_children()
[r.pil_show() for r in res]

# res = img.crop_bbox(person_bbox)
# [r.pil_show() for r in res]

# res = img.crop_bbox(face_bbox)
# [r.pil_show() for r in res]

# person_bbox.update

# person.add_child(
#     ResultNode(
#         kind="part",
#         label="face",
#         score=0.95,
#         payload=BoundingBox.xyxy([60, 35, 150, 140], image_size=(640, 480)),
#     )
# )

# person.add_child(
#     ResultNode(
#         kind="pose",
#         label="body_pose",
#         payload=Keypoints(
#             points=[[90, 80, 0.99], [80, 180, 0.91], [120, 180, 0.88]],
#             names=["nose", "left_shoulder", "right_shoulder"],
#             image_size=(640, 480),
#         ),
#     )
# )

# ocr = ResultNode(
#     kind="ocr",
#     label="invoice_number",
#     score=0.93,
#     payload=TextSpan(text="INV-2026-001", start=0, end=12),
# )

# embedding = ResultNode(
#     kind="embedding",
#     label="image_embedding",
#     payload=Vector(data=[0.1, 0.2, 0.3]),
# )

# results = ResultSet(items=[person, ocr, embedding], metadata={"source": "demo.jpg"})

# print(results.model_dump_json(indent=2))
# print(results.to_flat_rows())
