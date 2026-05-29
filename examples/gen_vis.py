import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import Model4Mat
from resultkit.gen import FrameGenerator
from resultkit.vis import FrameGeneratorVisualizer


def build_image(width: int, height: int, mode: str) -> Model4Mat.ImageMat:
    if mode in {"gray", "left", "right"}:
        return Model4Mat.ImageMat(
            color_format=Model4Mat.ImageMat.ColorFormat.GRAY,
            data=np.zeros((height, width), dtype=np.uint8),
        )

    return Model4Mat.ImageMat(
        color_format=Model4Mat.ImageMat.ColorFormat.BGR,
        data=np.zeros((height, width, 3), dtype=np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic FrameGenerator output.")
    parser.add_argument("--mode", choices=["color", "gray", "left", "right"], default="color")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    img = build_image(args.width, args.height, args.mode)
    gen = FrameGenerator(
        img=img,
        name=args.mode,
        fps=args.fps,
        overlay_prefix="ResultKit GEN",
    )

    vis = FrameGeneratorVisualizer(
        generator=gen,
        window_name=f"resultkit gen: {args.mode}",
        delay_ms=max(1, int(1000 / args.fps)),
    )
    vis.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
