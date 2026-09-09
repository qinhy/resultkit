
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resultkit.MatModel import ColorFormat, MatScaleFormat, MatViewMode, Model4Mat, MatStore

store = MatStore.build()
BoundingBox = Model4Mat.BoundingBox

img = store.add_new_obj(Model4Mat.ImageMat.from_url("./examples/img1.jpg",
                                                color_format=ColorFormat.RGB))

imgv = store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0.25, 0.25],[0.75, 0.75]]),
                                                scale=MatScaleFormat.ZERO_ONE,
                                                mode=MatViewMode.HWxyxy,
                                                controller=img.controller))

img.pil_show()
imgv.pil_show()

