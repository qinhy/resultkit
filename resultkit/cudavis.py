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
from pycuda.compiler import SourceModule


def make_specialized_gl_copy_src(
    *,
    width: int,
    height: int,
    channels: int,
    layout: int,
    color_order: int,
    flip_y: bool,
) -> str:
    """Generate a branch-minimized CUDA kernel specialized for one image format."""

    plane = width * height

    if flip_y:
        src_y_code = f"int src_y = {height - 1} - y;"
    else:
        src_y_code = "int src_y = y;"

    # layout:
    #   0 = HW
    #   1 = HWC
    #   2 = BHW
    #   3 = BHWC
    #   4 = BCHW
    #
    # color_order:
    #   0 = RGB/RGBA/GRAY
    #   1 = BGR/BGRA

    if layout in (0, 2):
        load_store_code = """
    unsigned char v = src[pix];

    dst[dst_i + 0] = v;
    dst[dst_i + 1] = v;
    dst[dst_i + 2] = v;
    dst[dst_i + 3] = 255;
"""

    elif layout == 4:
        if channels == 1:
            load_store_code = """
    unsigned char v = src[pix];

    dst[dst_i + 0] = v;
    dst[dst_i + 1] = v;
    dst[dst_i + 2] = v;
    dst[dst_i + 3] = 255;
"""
        elif color_order == 1:
            # BCHW + BGR/BGRA
            if channels == 4:
                load_store_code = f"""
    dst[dst_i + 0] = src[2 * {plane} + pix];
    dst[dst_i + 1] = src[1 * {plane} + pix];
    dst[dst_i + 2] = src[0 * {plane} + pix];
    dst[dst_i + 3] = src[3 * {plane} + pix];
"""
            else:
                load_store_code = f"""
    dst[dst_i + 0] = src[2 * {plane} + pix];
    dst[dst_i + 1] = src[1 * {plane} + pix];
    dst[dst_i + 2] = src[0 * {plane} + pix];
    dst[dst_i + 3] = 255;
"""
        else:
            # BCHW + RGB/RGBA
            if channels == 4:
                load_store_code = f"""
    dst[dst_i + 0] = src[0 * {plane} + pix];
    dst[dst_i + 1] = src[1 * {plane} + pix];
    dst[dst_i + 2] = src[2 * {plane} + pix];
    dst[dst_i + 3] = src[3 * {plane} + pix];
"""
            else:
                load_store_code = f"""
    dst[dst_i + 0] = src[0 * {plane} + pix];
    dst[dst_i + 1] = src[1 * {plane} + pix];
    dst[dst_i + 2] = src[2 * {plane} + pix];
    dst[dst_i + 3] = 255;
"""

    else:
        # HWC or BHWC
        if channels == 1:
            load_store_code = """
    unsigned char v = src[pix];

    dst[dst_i + 0] = v;
    dst[dst_i + 1] = v;
    dst[dst_i + 2] = v;
    dst[dst_i + 3] = 255;
"""
        elif color_order == 1:
            # HWC/BHWC + BGR/BGRA
            if channels == 4:
                load_store_code = f"""
    int base = pix * {channels};

    dst[dst_i + 0] = src[base + 2];
    dst[dst_i + 1] = src[base + 1];
    dst[dst_i + 2] = src[base + 0];
    dst[dst_i + 3] = src[base + 3];
"""
            else:
                load_store_code = f"""
    int base = pix * {channels};

    dst[dst_i + 0] = src[base + 2];
    dst[dst_i + 1] = src[base + 1];
    dst[dst_i + 2] = src[base + 0];
    dst[dst_i + 3] = 255;
"""
        else:
            # HWC/BHWC + RGB/RGBA
            if channels == 4:
                load_store_code = f"""
    int base = pix * {channels};

    dst[dst_i + 0] = src[base + 0];
    dst[dst_i + 1] = src[base + 1];
    dst[dst_i + 2] = src[base + 2];
    dst[dst_i + 3] = src[base + 3];
"""
            else:
                load_store_code = f"""
    int base = pix * {channels};

    dst[dst_i + 0] = src[base + 0];
    dst[dst_i + 1] = src[base + 1];
    dst[dst_i + 2] = src[base + 2];
    dst[dst_i + 3] = 255;
"""

    return f'''
extern "C" __global__
void image_u8_to_rgba_pbo(
    const unsigned char *src,
    unsigned char *dst)
{{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= {width} || y >= {height}) return;

    {src_y_code}
    int pix = src_y * {width} + x;
    int dst_i = 4 * (y * {width} + x);

{load_store_code}
}}
'''


@dataclass(frozen=True, slots=True)
class GlViewerConfig:
    width: int = 4000
    height: int = 3000
    fps: float = 60.0
    device: int = 0
    flip_y: bool = True
    max_frames: Optional[int] = None

    # Keep False for speed. True is useful only while debugging async CUDA errors.
    debug_cuda_sync: bool = False


class ImageMatCudaGlViewer:
    """Fast PyCUDA/OpenGL PBO viewer for ImageMatCUDAPubSub-like images."""

    def __init__(self, 
                width, height,
                fps: float = 60.0,
                device: int = 0,
                flip_y: bool = True,
                max_frames: Optional[int] = None,
    ):
        self.config = GlViewerConfig(
            width=width,
            height=height,
            fps=fps,
            device=device,
            flip_y=flip_y,
            max_frames=max_frames,
        )
        self.img = None

        self.tex = None
        self.pbo = None
        self.cuda_pbo = None
        self.cuda_ctx = None

        self.copy_module = None
        self.copy_kernel = None

        self.block = (16, 16, 1)
        self.grid = (
            (self.config.width + 15) // 16,
            (self.config.height + 15) // 16,
            1,
        )

        self.frame_idx = 0
        self.last_sequence = None
        self.last_status_time = 0.0
        self.closing = False

    @staticmethod
    def _enum_string(value) -> str:
        return str(value.value if hasattr(value, "value") else value).upper()

    @classmethod
    def _layout_code(cls, shape_type) -> int:
        st = cls._enum_string(shape_type)
        mapping = {
            "HW": 0,
            "HWC": 1,
            "BHW": 2,
            "BHWC": 3,
            "BCHW": 4,
        }
        return mapping[st]

    @classmethod
    def _color_order(cls, color_format) -> int:
        cf = cls._enum_string(color_format)
        return 1 if cf in {"BGR", "BGRA"} else 0

    def _prepare_image_format(self) -> tuple[int, int, int, int]:
        _, channels, height, width = self.img.BCHW

        channels = int(channels)
        height = int(height)
        width = int(width)

        if width != self.config.width or height != self.config.height:
            raise RuntimeError(
                f"Frame size {width}x{height} does not match viewer size "
                f"{self.config.width}x{self.config.height}"
            )

        if channels not in (1, 3, 4):
            raise RuntimeError(
                f"Unsupported channel count: {channels}. Supported: 1, 3, 4."
            )

        layout = self._layout_code(self.img.shape_type)
        color_order = self._color_order(self.img.color_format)

        return channels, layout, color_order, width * height * channels

    def init_gl_window(self) -> None:
        glutInit(sys.argv)

        try:
            glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        except Exception:
            pass

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
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def init_gl_resources(self) -> None:
        self.tex = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        try:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
        except Exception:
            pass

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
        glBindTexture(GL_TEXTURE_2D, 0)

    def init_cuda_after_gl_context_exists(self) -> None:
        cuda.init()

        device = cuda.Device(self.config.device)
        print("[CUDA] device :", device.name())

        self.cuda_ctx = cudagl.make_context(device)

        self.cuda_pbo = cudagl.RegisteredBuffer(
            int(self.pbo),
            cudagl.graphics_map_flags.WRITE_DISCARD,
        )

    def compile_specialized_kernel(self) -> None:
        channels, layout, color_order, _ = self._prepare_image_format()

        src = make_specialized_gl_copy_src(
            width=self.config.width,
            height=self.config.height,
            channels=channels,
            layout=layout,
            color_order=color_order,
            flip_y=self.config.flip_y,
        )

        self.copy_module = SourceModule(src, options=["-O3", "--use_fast_math"])
        self.copy_kernel = self.copy_module.get_function("image_u8_to_rgba_pbo")

        print(
            "[CUDA] specialized kernel:",
            f"layout={layout}",
            f"channels={channels}",
            f"color_order={color_order}",
            f"flip_y={self.config.flip_y}",
            flush=True,
        )

    def init(self) -> None:
        self.init_gl_window()
        self.init_gl_resources()
        self.init_cuda_after_gl_context_exists()

    def attach_img_and_compile(self, img) -> None:
        self.img = img
        self.compile_specialized_kernel()

    def copy_frame_to_pbo(self, frame) -> None:
        mapping = self.cuda_pbo.map()

        try:
            pbo_ptr, _ = mapping.device_ptr_and_size()

            self.copy_kernel(
                np.uintp(int(frame.gpudata)),
                np.uintp(int(pbo_ptr)),
                block=self.block,
                grid=self.grid,
            )

            if self.config.debug_cuda_sync:
                cuda.Context.synchronize()

        finally:
            mapping.unmap()

    def upload_pbo_to_texture(self) -> None:
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
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)

        glBegin(GL_QUADS)

        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)

        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, -1.0)

        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)

        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, 1.0)

        glEnd()
        glutSwapBuffers()

    def receive_and_render_once(self) -> None:
        self.img.sub(copy=False, sync=False)

        # # This is the main remaining necessary per-frame check:
        # # no remote frame yet means keep showing the previous texture.
        # if getattr(self.img, "_remote_mem", None) is None:
        #     return

        frame = self.img.get_data()

        self.copy_frame_to_pbo(frame)
        self.upload_pbo_to_texture()

        self.frame_idx += 1
        self.last_sequence = getattr(self.img, "sequence", self.frame_idx)

    def display(self) -> None:        
        if self.closing:return
        self.receive_and_render_once()
        self.draw_fullscreen_quad()
        self.print_status_if_due()

    def display_limited(self) -> None:
        if self.closing:return
        self.receive_and_render_once()
        self.draw_fullscreen_quad()
        self.print_status_if_due()

        if self.frame_idx >= self.config.max_frames:
            self.request_close()

    def timer(self, value: int = 0) -> None:
        glutPostRedisplay()
        glutTimerFunc(self._timer_interval_ms, self.timer, 0)

    def idle(self) -> None:
        glutPostRedisplay()

    def reshape(self, width: int, height: int) -> None:
        glViewport(0, 0, max(1, width), max(1, height))

    def keyboard(self, key, x, y) -> None:
        if key in (b"q", b"\x1b"):
            self.request_close()

    def print_status_if_due(self) -> None:
        now = time.time()

        if now - self.last_status_time < 1.0:
            return

        print(
            f"[gl-viewer] displayed={self.frame_idx} latest_seq={self.last_sequence}",
            flush=True,
        )
        self.last_status_time = now

    def request_close(self) -> None:
        if self.closing:
            return

        self.closing = True

        try:
            glutIdleFunc(None)
        except Exception:
            pass

        self.cleanup()

        try:
            glutLeaveMainLoop()
        except Exception:
            pass

    def cleanup(self) -> None:
        try:
            if self.cuda_pbo is not None:
                self.cuda_pbo.unregister()
                self.cuda_pbo = None
        except Exception as exc:
            print("[cleanup] cuda_pbo:", exc, flush=True)

        try:
            if self.img is not None and hasattr(self.img, "close"):
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
                glDeleteTextures(1, [self.tex])
                self.tex = None
        except Exception as exc:
            print("[cleanup] tex:", exc, flush=True)

        self.copy_kernel = None
        self.copy_module = None

        try:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
                self.cuda_ctx = None
        except Exception as exc:
            print("[cleanup] cuda_ctx:", exc, flush=True)

    def run(self, img=None) -> None:
        if img is not None:
            self.attach_img_and_compile(img)
        elif self.img is not None:
            self.compile_specialized_kernel()
        else:
            raise RuntimeError("ImageMat subscriber has not been provided")

        glutDisplayFunc(self.display_limited if self.config.max_frames else self.display)
        glutKeyboardFunc(self.keyboard)
        glutReshapeFunc(self.reshape)

        try:
            glutCloseFunc(self.request_close)
        except Exception:
            pass

        if self.config.fps > 0:
            self._timer_interval_ms = max(1, int(round(1000.0 / self.config.fps)))
            glutTimerFunc(0, self.timer, 0)
            print(f"[gl-viewer] running at target fps={self.config.fps}", flush=True)
        else:
            glutIdleFunc(self.idle)
            print("[gl-viewer] running as fast as possible", flush=True)

        try:
            glutMainLoop()
        finally:
            self.cleanup()