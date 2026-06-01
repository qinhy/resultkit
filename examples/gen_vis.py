import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import Model4Mat
# from resultkit.gen import FrameGenerator
# from resultkit.vis import FrameGeneratorVisualizer

def draw(frame):
    frame.fill(0)
    idx = np.random.choice(len(frame), int(0.1 * len(frame)), replace=False)
    idx1 = np.random.choice(len(frame), int(0.1 * len(frame)), replace=False)
    idx2 = np.random.choice(len(frame), int(0.1 * len(frame)), replace=False)
    frame[idx, :, 0] = 125
    frame[idx1, :, 1] = 125
    frame[idx2, :, 2] = 125
    return frame

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic FrameGenerator output.")
    # parser.add_argument("--mode", choices=["color", "gray", "left", "right"], default="color")
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sub", action="store_true")
    args = parser.parse_args()

    img = Model4Mat.ImageMatPubSub(
        color_format=Model4Mat.ImageMat.ColorFormat.BGR,
        data=np.zeros((args.height, args.width, 3), dtype=np.uint8),
    )
    img.set_id("ImageMatPubSub:test").init()

    if not args.sub:
        img.is_pub=True
        fps = 0.0
        frame_idx = 0
        st = time.time()
        while True:
            # iox2_sample,frame = img.get_data()
            # draw(frame)
            # iox2_sample.assume_init().send()
            img.pub(edit_func=draw)
            frame_idx += 1
            te = time.time() - st
            if frame_idx % 1000 == 0 and te>0:
                fps = frame_idx / te
                print(f"FPS : {fps:.2f}")
    else:
        while True:
            frame = img.sub().get_data()
            try:
                cv2.imshow(f"resultkit gen", frame)
            except cv2.error:
                pass
            if cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF == ord("q"):
                break           


if __name__ == "__main__":
    main()
