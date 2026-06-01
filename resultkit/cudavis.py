import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

CUDA_ROOT = os.environ.get("CUDA_PATH")
if CUDA_ROOT and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(CUDA_ROOT, "bin"))

from OpenGL.GL import *
from OpenGL.GLUT import *

import pycuda.driver as cuda
import pycuda.gl as cudagl
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

_GL_COPY_MODULE = None
_GL_COPY_KERNEL = None

GL_COPY_SRC = r'''
extern "C" __global__
void image_u8_to_rgba_pbo(
    const unsigned char *src,
    unsigned char *dst,
    int width,
    int height,
    int channels,
    int layout,
    int color_order,
    int flip_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_y = flip_y ? (height - 1 - y) : y;
    int pix = src_y * width + x;
    int plane = width * height;

    unsigned char r = 255, g = 0, b = 255, a = 255;

    // layout:
    //   0 = HW
    //   1 = HWC
    //   2 = BHW, first batch
    //   3 = BHWC, first batch
    //   4 = BCHW, first batch, channels-first
    if (layout == 0 || layout == 2) {
        unsigned char v = src[pix];
        r = v; g = v; b = v;
    } else if (layout == 4) {
        if (channels == 1) {
            unsigned char v = src[pix];
            r = v; g = v; b = v;
        } else {
            unsigned char c0 = src[0 * plane + pix];
            unsigned char c1 = src[1 * plane + pix];
            unsigned char c2 = src[2 * plane + pix];
            unsigned char c3 = channels >= 4 ? src[3 * plane + pix] : 255;
            if (color_order == 1) { // BGR/BGRA
                b = c0; g = c1; r = c2;
            } else {                // RGB/RGBA
                r = c0; g = c1; b = c2;
            }
            a = channels >= 4 ? c3 : 255;
        }
    } else {
        int base = pix * channels;
        if (channels == 1) {
            unsigned char v = src[base];
            r = v; g = v; b = v;
        } else {
            unsigned char c0 = src[base + 0];
            unsigned char c1 = src[base + 1];
            unsigned char c2 = src[base + 2];
            unsigned char c3 = channels >= 4 ? src[base + 3] : 255;
            if (color_order == 1) { // BGR/BGRA
                b = c0; g = c1; r = c2;
            } else {                // RGB/RGBA
                r = c0; g = c1; b = c2;
            }
            a = channels >= 4 ? c3 : 255;
        }
    }

    int dst_i = 4 * (y * width + x);
    dst[dst_i + 0] = r;
    dst[dst_i + 1] = g;
    dst[dst_i + 2] = b;
    dst[dst_i + 3] = a;
}
'''


def get_gl_copy_kernel():
    global _GL_COPY_MODULE, _GL_COPY_KERNEL
    if _GL_COPY_KERNEL is not None:
        return _GL_COPY_KERNEL
    _GL_COPY_MODULE = SourceModule(GL_COPY_SRC, options=["-O3", "--use_fast_math"])
    _GL_COPY_KERNEL = _GL_COPY_MODULE.get_function("image_u8_to_rgba_pbo")
    return _GL_COPY_KERNEL


@dataclass(frozen=True, slots=True)
class GlViewerConfig:
    width: int = 4000
    height: int = 3000
    fps: float = 60.0
    device: int = 0
    num_slots: int = 3
    topic_id: str = "ImageMatCUDAPubSub:test"
    flip_y: bool = True
    max_frames: Optional[int] = None


class ImageMatCudaGlViewer:
    """Model4Mat.ImageMatCUDAPubSub subscriber rendered through OpenGL/PBO."""

    def __init__(self, config: GlViewerConfig):
        self.config = config
        self.img = None
        self.tex: Optional[int] = None
        self.pbo: Optional[int] = None
        self.cuda_pbo = None
        self.cuda_ctx = None
        self.copy_kernel = None
        self.closing = False
        self.frame_idx = 0
        self.last_status_time = 0.0
        self.last_sequence: Optional[int] = None

    def init_gl_window(self) -> None:
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
        glutInitWindowSize(self.config.width, self.config.height)
        glutCreateWindow(b"resultkit ImageMatCUDAPubSub PyCUDA -> OpenGL")

        vendor = glGetString(GL_VENDOR)
        renderer = glGetString(GL_RENDERER)
        version = glGetString(GL_VERSION)
        print("[GL] vendor  :", vendor.decode(errors="replace") if vendor else None)
        print("[GL] renderer:", renderer.decode(errors="replace") if renderer else None)
        print("[GL] version :", version.decode(errors="replace") if version else None)

        glViewport(0, 0, self.config.width, self.config.height)
        glDisable(GL_DEPTH_TEST)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def init_gl_resources(self) -> None:
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA8,
            self.config.width,
            self.config.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            None,
        )

        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(
            GL_PIXEL_UNPACK_BUFFER,
            self.config.width * self.config.height * 4,
            None,
            GL_STREAM_DRAW,
        )
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def init_cuda_after_gl_context_exists(self) -> None:
        if self.pbo is None:
            raise RuntimeError("OpenGL PBO has not been created")
        cuda.init()
        self.cuda_ctx = cudagl.make_context(cuda.Device(self.config.device))
        self.copy_kernel = get_gl_copy_kernel()
        self.cuda_pbo = cudagl.RegisteredBuffer(
            int(self.pbo),
            cudagl.graphics_map_flags.WRITE_DISCARD,
        )

    @staticmethod
    def _layout_code(shape_type) -> int:
        st = str(shape_type.value if hasattr(shape_type, "value") else shape_type)
        mapping = {
            "HW": 0,
            "HWC": 1,
            "BHW": 2,
            "BHWC": 3,
            "BCHW": 4,
        }
        if st not in mapping:
            raise ValueError(f"Unsupported shape_type for GL viewer: {shape_type}")
        return mapping[st]

    @staticmethod
    def _color_order(color_format) -> int:
        cf = str(color_format.value if hasattr(color_format, "value") else color_format).upper()
        # 0 = RGB/RGBA/GRAY, 1 = BGR/BGRA
        return 1 if cf == "BGR" else 0

    def copy_frame_to_pbo(self, frame: gpuarray.GPUArray) -> bool:
        if not isinstance(frame, gpuarray.GPUArray):
            raise TypeError(f"expected pycuda.gpuarray.GPUArray, got {type(frame)!r}")
        if np.dtype(frame.dtype) != np.dtype(np.uint8):
            raise TypeError(f"GL viewer currently expects uint8 GPUArray, got {frame.dtype}")

        _, channels, height, width = self.img.BCHW
        width = int(width)
        height = int(height)
        channels = int(channels)
        if width != self.config.width or height != self.config.height:
            raise RuntimeError(
                f"Frame size {width}x{height} does not match GL window "
                f"{self.config.width}x{self.config.height}. Start viewer with matching --width/--height."
            )

        mapping = self.cuda_pbo.map()
        try:
            pbo_ptr, pbo_size = mapping.device_ptr_and_size()
            needed = self.config.width * self.config.height * 4
            if pbo_size < needed:
                raise RuntimeError(f"PBO too small: {pbo_size} < {needed}")

            block = (16, 16, 1)
            grid = ((self.config.width + 15) // 16, (self.config.height + 15) // 16, 1)
            self.copy_kernel(
                np.uintp(int(frame.gpudata)),
                np.uintp(int(pbo_ptr)),
                np.int32(self.config.width),
                np.int32(self.config.height),
                np.int32(channels),
                np.int32(self._layout_code(self.img.shape_type)),
                np.int32(self._color_order(self.img.color_format)),
                np.int32(1 if self.config.flip_y else 0),
                block=block,
                grid=grid,
            )
            cuda.Context.synchronize()
        finally:
            mapping.unmap()
        return True

    def upload_pbo_to_texture(self) -> None:
        if self.tex is None or self.pbo is None:
            raise RuntimeError("OpenGL texture/PBO has not been created")
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            self.config.width,
            self.config.height,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def draw_fullscreen_quad(self) -> None:
        if self.tex is None:
            return
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0)
        glEnd()
        glutSwapBuffers()

    def receive_and_render_once(self) -> bool:
        self.img.sub(copy=False, sync=False)
        # No signal has arrived yet; keep the last texture on screen.
        if getattr(self.img, "_remote_mem", None) is None:
            return False

        frame = self.img.get_data()
        if self.copy_frame_to_pbo(frame):
            self.upload_pbo_to_texture()
            self.frame_idx += 1
            self.last_sequence = int(getattr(self.img, "sequence", self.frame_idx))
            return True
        return False

    def display(self) -> None:
        if self.closing:
            return
        try:
            self.receive_and_render_once()
            self.draw_fullscreen_quad()
            self.print_status_if_due()
        except Exception as exc:
            print("[gl-viewer] display error:", repr(exc), flush=True)
            self.request_close()
            return

        if self.config.max_frames is not None and self.frame_idx >= self.config.max_frames:
            self.request_close()
            return

        if not self.closing:
            glutPostRedisplay()

    def print_status_if_due(self) -> None:
        now = time.time()
        if now - self.last_status_time < 1.0:
            return
        print(
            f"[gl-viewer] displayed={self.frame_idx} latest_seq={self.last_sequence}",
            flush=True,
        )
        self.last_status_time = now

    def keyboard(self, key, x, y) -> None:
        if key in (b"q", b"\x1b"):
            self.request_close()

    def request_close(self) -> None:
        self.closing = True
        self.cleanup()
        try:
            glutLeaveMainLoop()
        except Exception:
            os._exit(0)

    def cleanup(self) -> None:
        try:
            if self.cuda_pbo is not None:
                self.cuda_pbo.unregister()
                self.cuda_pbo = None
        except Exception as exc:
            print("[cleanup] cuda_pbo:", exc, flush=True)

        try:
            if self.img is not None:
                self.img.close()
                self.img = None
        except Exception as exc:
            print("[cleanup] img:", exc, flush=True)

        try:
            if self.pbo is not None:
                glDeleteBuffers(1, [self.pbo])
                self.pbo = None
        except Exception as exc:
            print("[cleanup] pbo:", exc, flush=True)

        try:
            if self.tex is not None:
                glDeleteTextures([self.tex])
                self.tex = None
        except Exception as exc:
            print("[cleanup] tex:", exc, flush=True)

        try:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
                self.cuda_ctx = None
        except Exception as exc:
            print("[cleanup] cuda_ctx:", exc, flush=True)

    def init(self):        
        self.init_gl_window()
        self.init_gl_resources()
        self.init_cuda_after_gl_context_exists()

    def run(self,img=None) -> None:
        if img is not None:
            self.img = img
        if self.img is None:
            raise RuntimeError("ImageMat subscriber has not been created")
        if self.cuda_pbo is None or self.copy_kernel is None:
            raise RuntimeError("CUDA/OpenGL resources are not initialized")
        try:
            glutDisplayFunc(self.display)
            glutKeyboardFunc(self.keyboard)
            try:
                glutCloseFunc(self.request_close)
            except Exception:
                pass

            print("[gl-viewer] running. Press q or Esc to quit.", flush=True)
            glutMainLoop()
        except Exception:
            self.cleanup()
            raise

