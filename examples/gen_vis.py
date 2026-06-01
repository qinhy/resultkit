import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import Model4Mat
from resultkit.gen import FrameGenerator
from resultkit.vis import FrameGeneratorVisualizer


def build_image(width: int, height: int, mode: str):
    if mode in {"gray", "left", "right"}:
        res = Model4Mat.ImageMatPubSub(
            color_format=Model4Mat.ImageMat.ColorFormat.GRAY,
            data=np.zeros((height, width), dtype=np.uint8),
        )
    else:
        res = Model4Mat.ImageMatPubSub(
            color_format=Model4Mat.ImageMat.ColorFormat.BGR,
            data=np.zeros((height, width, 3), dtype=np.uint8),
        )
    res.set_id("ImageMatPubSub:test").init()
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic FrameGenerator output.")
    parser.add_argument("--mode", choices=["color", "gray", "left", "right"], default="color")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sub", action="store_true")
    args = parser.parse_args()

    img = build_image(args.width, args.height, args.mode)

    if not args.sub:
        img.is_pub=True
        gen = FrameGenerator(
            img=img,
            name=args.mode,
            fps=args.fps,
            overlay_prefix="ResultKit GEN",
        )
        fps = 0.0
        cnt = 0
        st = time.time()
        while True:
            gen.read().assume_init().send()
            cnt += 1
            te = time.time() - st
            if cnt % 1000 == 0 and te>0:
                fps = cnt / te
                print(f"FPS : {fps:.2f}")
    else:
        vis = FrameGeneratorVisualizer(
            generator=img,
            window_name=f"resultkit gen: {args.mode}",
            delay_ms=max(1, int(1000 / args.fps)),
        )
        vis.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
