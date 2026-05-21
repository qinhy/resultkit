
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

imgv = store.add_new_obj(Model4Mat.ImageMatView(data=np.array([[0.25, 0.25],[0.75, 0.75]]),
                                                scale=Model4Mat.ImageMatView.ScaleFormat.ZERO_ONE,
                                                mode=Model4Mat.ImageMatView.Mode.HWxyxy,
                                                controller=img.controller))

img.pil_show()
imgv.pil_show()

