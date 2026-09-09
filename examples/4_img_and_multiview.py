
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resultkit.MatModel import ColorFormat, MatScaleFormat, MatViewMode, Model4Mat, MatStore

store = MatStore.build()
BoundingBox = Model4Mat.BoundingBox

img = store.add_new_obj(Model4Mat.ImageMat.from_url("./examples/img1.jpg",
                                                color_format=ColorFormat.RGB))

imgv = [store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0, 0],[0.5, 0.5]]),
                                                scale=MatScaleFormat.ZERO_ONE,
                                                mode=MatViewMode.HWxyxy,
                                                controller=img.controller)),
        store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0.5, 0.5],[1.0, 1.0]]),
                                                scale=MatScaleFormat.ZERO_ONE,
                                                mode=MatViewMode.HWxyxy,
                                                controller=img.controller)),
        store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0, 0.5],[0.5, 1.0]]),
                                                scale=MatScaleFormat.ZERO_ONE,
                                                mode=MatViewMode.HWxyxy,
                                                controller=img.controller)),
        store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0.5, 0],[1.0, 0.5]]),
                                                scale=MatScaleFormat.ZERO_ONE,
                                                mode=MatViewMode.HWxyxy,
                                                controller=img.controller)),
        ]

img.pil_show()
[r.pil_show() for r in imgv] #split into quarters

# update image
img.safe_update_data((np.random.rand(512, 512, 3)*255.0).astype(np.uint8))
img = img.to_torch()
[r.pil_show() for r in imgv]