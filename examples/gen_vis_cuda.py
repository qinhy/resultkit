from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


CUDA_ROOT = os.environ.get("CUDA_PATH")
if CUDA_ROOT and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(CUDA_ROOT, "bin"))

from OpenGL.GL import *
from OpenGL.GLUT import *

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resultkit.MatModel import ColorFormat, ImageShapeType, Model4Mat
from resultkit.cudavis import ImageMatCudaGlViewer
from resultkit.mat import DataType, MatDevice


_DRAW_MODULE = None
_DRAW_KERNEL = None

DRAW_SRC = r'''
extern "C" __global__ void draw_bgr_u8(
    unsigned char *frame,
    int width,
    int height,
    unsigned int seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (i >= n) return;

    int x = i % width;
    int y = i / width;

    // Cheap deterministic row hashes. This gives a random-looking
    // 10% subset per channel without creating GPU RNG state.
    unsigned int h0 = (unsigned int)y * 1664525u + seed + 1013904223u;
    unsigned int h1 = (unsigned int)y * 22695477u + seed * 3u + 1u;
    unsigned int h2 = (unsigned int)y * 1103515245u + seed * 7u + 12345u;

    int base = (y * width + x) * 3;
    frame[base + 0] = ((h0 >> 8) % 10u == 0u) ? 125u : 0u;  // B
    frame[base + 1] = ((h1 >> 8) % 10u == 0u) ? 125u : 0u;  // G
    frame[base + 2] = ((h2 >> 8) % 10u == 0u) ? 125u : 0u;  // R
}
'''


def get_draw_kernel():
    global _DRAW_MODULE, _DRAW_KERNEL
    if _DRAW_KERNEL is not None:
        return _DRAW_KERNEL
    _DRAW_MODULE = SourceModule(DRAW_SRC, options=["-O3"])
    _DRAW_KERNEL = _DRAW_MODULE.get_function("draw_bgr_u8")
    return _DRAW_KERNEL


def draw(frame):
    """Edit one HWC/BGR uint8 PyCUDA GPUArray frame in-place on the GPU."""
    if not isinstance(frame, gpuarray.GPUArray):
        raise TypeError(f"expected pycuda.gpuarray.GPUArray, got {type(frame)!r}")
    if frame.dtype != np.dtype(np.uint8) or len(frame.shape) != 3 or frame.shape[2] != 3:
        raise TypeError(f"expected HWC/BGR uint8 GPUArray, got shape={frame.shape}, dtype={frame.dtype}")

    height, width, _ = frame.shape
    block = (256, 1, 1)
    grid = ((width * height + block[0] - 1) // block[0], 1, 1)
    seed = np.uint32(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

    # IPC slot views may use a custom pointer holder. Passing the plain uintptr
    # avoids PyCUDA's "invalid type on parameter #0" kernel-argument failure.
    frame_ptr = np.uintp(int(frame.gpudata))

    get_draw_kernel()(
        frame_ptr,
        np.int32(width),
        np.int32(height),
        seed,
        block=block,
        grid=grid,
    )
    return frame


def make_image(args):
    if not hasattr(Model4Mat, "ImageMatCUDAPubSub"):
        raise SystemExit(
            "Model4Mat.ImageMatCUDAPubSub is not available. "
            "Use the PyCUDA patched MatModel.py that defines ImageMatCUDAPubSub."
        )

    width = int(args.width)
    height = int(args.height)
    num_slots = int(args.num_slots)
    topic_id = getattr(args, "topic_id", "ImageMatCUDAPubSub:test")

    data = gpuarray.empty((height, width, 3), dtype=np.uint8)
    img = Model4Mat.ImageMatCUDAPubSub(
        color_format=ColorFormat.BGR,
        shape_type=ImageShapeType.HWC,
        dtype=DataType.UINT8,
        device=MatDevice.CUDA0,
        data=data,
        num_slots=num_slots,
    )
    img.set_id(topic_id).init()
    return img


def run_publisher(args) -> None:
    cuda.init()
    ctx = cuda.Device(args.device).make_context()
    img = None
    try:
        img = make_image(args)
        frame_idx = 0
        st = time.time()
        while True:
            img.pub(edit_func=draw)
            frame_idx += 1
            te = time.time() - st
            if frame_idx % 1000 == 0 and te > 0:
                print(f"FPS : {frame_idx / te:.2f}", flush=True)
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass
        ctx.pop()
        ctx.detach()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ImageMatCUDAPubSub PyCUDA publisher + OpenGL/PBO subscriber viewer"
    )
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--fps", type=float, default=60.0, help="Kept for CLI compatibility; GLUT redraws continuously.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sub", action="store_true", help="Run subscriber GL viewer instead of publisher.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-slots", type=int, default=3)
    parser.add_argument("--topic-id", default="ImageMatCUDAPubSub:test")
    parser.add_argument("--flip-y", action="store_true", default=True)
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sub:
        run_publisher(args)
    else:
        vis = ImageMatCudaGlViewer(                    
            width=int(args.width),
            height=int(args.height),
            fps=args.fps,
            device=args.device,
            flip_y=args.flip_y,
            max_frames=args.max_frames,
        )
        vis.init()
        vis.run(img=make_image(args))


if __name__ == "__main__":
    main()
