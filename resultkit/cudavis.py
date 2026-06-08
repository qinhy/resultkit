from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cuda import ImageMatCUDAPubSub

CUDA_ROOT = os.environ.get("CUDA_PATH")
if CUDA_ROOT and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(CUDA_ROOT, "bin"))

import glfw
from OpenGL.GL import *

import pycuda.driver as cuda
import pycuda.gl as cudagl
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


# GLFW has process-global init/terminate state. Keep a tiny reference count so
# sequential viewers can safely init -> terminate -> init again, and so nested
# use does not terminate GLFW too early.
_GLFW_USERS = 0


def _acquire_glfw() -> None:
    global _GLFW_USERS

    if _GLFW_USERS == 0:
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

    _GLFW_USERS += 1


def _release_glfw() -> None:
    global _GLFW_USERS

    if _GLFW_USERS <= 0:
        return

    _GLFW_USERS -= 1
    if _GLFW_USERS == 0:
        glfw.terminate()


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

    # True limits swap_buffers to the display refresh rate. For benchmarking,
    # set this False.
    vsync: bool = True

    # Keep False for speed. True is useful while debugging async CUDA errors.
    debug_cuda_sync: bool = False

    # Resource ownership policy. False is safer when the subscriber is owned by
    # caller code and may be reused. True preserves the behavior of your old
    # cleanup() method, which closed self.img.
    close_img_on_cleanup: bool = False

    title: str = "resultkit ImageMatCUDAPubSub PyCUDA -> OpenGL"


class ImageMatCudaGlViewer:
    """Fast PyCUDA/OpenGL PBO viewer using GLFW.

    This version is designed for safe sequential use:

        viewer = ImageMatCudaGlViewer(width, height)
        viewer.run(img1)   # show, close, cleanup
        viewer.run(img2)   # show again with a fresh GL + CUDA interop context

    Important lifecycle rule:
        OpenGL context current
        -> create/register GL resources with CUDA
        -> map/unmap per frame
        -> unregister CUDA resources
        -> delete GL resources
        -> detach CUDA context
        -> destroy GLFW window/context
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 60.0,
        device: int = 0,
        flip_y: bool = True,
        max_frames: Optional[int] = None,
        *,
        vsync: bool = True,
        debug_cuda_sync: bool = False,
        close_img_on_cleanup: bool = False,
        title: str = "resultkit ImageMatCUDAPubSub PyCUDA -> OpenGL",
    ):
        self.config = GlViewerConfig(
            width=width,
            height=height,
            fps=fps,
            device=device,
            flip_y=flip_y,
            max_frames=max_frames,
            vsync=vsync,
            debug_cuda_sync=debug_cuda_sync,
            close_img_on_cleanup=close_img_on_cleanup,
            title=title,
        )

        self.img: Optional[ImageMatCUDAPubSub] = None

        self.window = None
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
        self.last_sequence: Optional[int] = None
        self.last_status_time = 0.0
        self.closing = False

        self._timer_target_ms = self._fps_to_interval_ms(self.config.fps)
        self._timer_interval_ms = self._timer_target_ms
        self._timer_min_ms = 1
        self._timer_max_ms = max(self._timer_target_ms * 4, 100)

        self.same_sequence_polls = 0
        self.no_remote_polls = 0
        self.sequence_jumps = 0

        self._initialized = False
        self._cleaned = True
        self._glfw_acquired = False

    @staticmethod
    def _fps_to_interval_ms(fps: float) -> int:
        if fps <= 0:
            return 1
        return max(1, int(round(1000.0 / float(fps))))

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
        try:
            return mapping[st]
        except KeyError as exc:
            raise RuntimeError(f"Unsupported shape_type: {shape_type!r}") from exc

    @classmethod
    def _color_order(cls, color_format) -> int:
        cf = cls._enum_string(color_format)
        return 1 if cf in {"BGR", "BGRA"} else 0

    def _prepare_image_format(self) -> tuple[int, int, int, int]:
        if self.img is None:
            raise RuntimeError("ImageMat subscriber has not been provided")

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

    def _reset_run_state(self) -> None:
        self.frame_idx = 0
        self.last_sequence = None
        self.last_status_time = 0.0
        self.closing = False

        self._timer_target_ms = self._fps_to_interval_ms(self.config.fps)
        self._timer_interval_ms = self._timer_target_ms
        self._timer_min_ms = 1
        self._timer_max_ms = max(self._timer_target_ms * 4, 100)

        self.same_sequence_polls = 0
        self.no_remote_polls = 0
        self.sequence_jumps = 0

        self._cleaned = False

    def init_gl_window(self) -> None:
        _acquire_glfw()
        self._glfw_acquired = True

        # Ask for a compatibility-style context. The immediate-mode textured quad
        # below intentionally matches your original GLUT renderer.
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)

        self.window = glfw.create_window(
            self.config.width,
            self.config.height,
            self.config.title,
            None,
            None,
        )
        if self.window is None:
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.config.vsync else 0)

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_window_close_callback(self.window, self._on_window_close)
        glfw.set_framebuffer_size_callback(self.window, self._on_framebuffer_size)

        # Be explicit for repeat runs. A newly created window should normally
        # have this flag False, but clearing it here makes the lifecycle robust
        # against wrapper/platform quirks and accidental reuse during debugging.
        glfw.set_window_should_close(self.window, False)
        self.closing = False

        vendor = glGetString(GL_VENDOR)
        renderer = glGetString(GL_RENDERER)
        version = glGetString(GL_VERSION)

        print("[GL] vendor  :", vendor.decode(errors="replace") if vendor else None)
        print("[GL] renderer:", renderer.decode(errors="replace") if renderer else None)
        print("[GL] version :", version.decode(errors="replace") if version else None)

        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.reshape(fb_width, fb_height)

        glDisable(GL_DEPTH_TEST)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        # Keep fixed-function matrices identity for the fullscreen quad.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def init_gl_resources(self) -> None:
        self._require_gl_context()

        self.tex = glGenTextures(1)
        self.tex = int(self.tex)

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
        self.pbo = int(self.pbo)
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
        if self.pbo is None:
            raise RuntimeError("PBO must exist before registering it with CUDA")

        self._require_gl_context()

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
        """Create one fresh GLFW window, OpenGL context, and CUDA/GL interop context."""
        if self._initialized:
            return

        self.init_gl_window()
        self.init_gl_resources()
        self.init_cuda_after_gl_context_exists()
        self._initialized = True

    def attach_img_and_compile(self, img: ImageMatCUDAPubSub) -> None:
        self.img = img
        if self._initialized:
            self.compile_specialized_kernel()

    def _require_gl_context(self) -> None:
        if self.window is None:
            raise RuntimeError("GLFW window has not been created")

        current = glfw.get_current_context()
        if current != self.window:
            glfw.make_context_current(self.window)

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            self.request_close()

    def _on_window_close(self, window) -> None:
        self.request_close()

    def _on_framebuffer_size(self, window, width: int, height: int) -> None:
        self.reshape(width, height)

    def copy_frame_to_pbo(self, frame: gpuarray.GPUArray) -> None:
        if self.cuda_pbo is None:
            raise RuntimeError("CUDA PBO is not registered")
        if self.copy_kernel is None:
            raise RuntimeError("CUDA copy kernel has not been compiled")

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
        self._require_gl_context()
        if self.tex is None or self.pbo is None:
            raise RuntimeError("OpenGL texture/PBO has not been initialized")

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
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw_fullscreen_quad(self) -> None:
        self._require_gl_context()
        if self.tex is None:
            return

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
        glBindTexture(GL_TEXTURE_2D, 0)

        if self.window is not None:
            glfw.swap_buffers(self.window)

    @staticmethod
    def _safe_int_sequence(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _current_sequence(self) -> Optional[int]:
        return self._safe_int_sequence(getattr(self.img, "sequence", None))

    def adapt_timer_after_poll(self, *, rendered: bool, seq_gap: int = 1) -> None:
        """Adapt polling interval using producer sequence feedback."""
        if self.config.fps <= 0:
            return

        if not rendered:
            self._timer_interval_ms = min(
                self._timer_max_ms,
                self._timer_interval_ms + 1,
            )
            return

        if seq_gap > 1:
            self.sequence_jumps += 1
            self._timer_interval_ms = max(
                self._timer_min_ms,
                max(1, self._timer_interval_ms // 2),
            )
            return

        # Normal new frame: gently converge back to the target interval.
        if self._timer_interval_ms < self._timer_target_ms:
            self._timer_interval_ms += 1
        elif self._timer_interval_ms > self._timer_target_ms:
            self._timer_interval_ms -= 1

    def receive_and_render_once(self) -> bool:
        if self.img is None:
            return False

        self.img.sub(copy=False, sync=False)

        # No remote frame yet. Keep drawing previous texture / black screen,
        # but slow down polling.
        if getattr(self.img, "_remote_mem", None) is None:
            self.no_remote_polls += 1
            self.adapt_timer_after_poll(rendered=False)
            return False

        seq = self._current_sequence()

        # Same producer frame again. Avoid repeating CUDA PBO copy and texture upload.
        if seq is not None and seq == self.last_sequence:
            self.same_sequence_polls += 1
            self.adapt_timer_after_poll(rendered=False)
            return False

        prev_seq = self.last_sequence

        frame: gpuarray.GPUArray = self.img.get_data()
        self.copy_frame_to_pbo(frame)
        self.upload_pbo_to_texture()

        # Count only unique producer frames actually uploaded to the GL texture.
        self.frame_idx += 1

        if seq is None:
            self.last_sequence = self.frame_idx
            seq_gap = 1
        else:
            self.last_sequence = seq
            seq_gap = 1 if prev_seq is None else max(1, seq - prev_seq)

        self.adapt_timer_after_poll(rendered=True, seq_gap=seq_gap)
        return True

    def display_once(self) -> None:
        if self.closing:
            return

        self.receive_and_render_once()
        self.draw_fullscreen_quad()
        self.print_status_if_due()

        if self.config.max_frames is not None and self.frame_idx >= self.config.max_frames:
            self.request_close()

    def reshape(self, width: int, height: int) -> None:
        self._require_gl_context()
        glViewport(0, 0, max(1, int(width)), max(1, int(height)))

    def print_status_if_due(self) -> None:
        now = time.time()

        if now - self.last_status_time < 1.0:
            return

        print(
            f"[gl-viewer] displayed={self.frame_idx} "
            f"latest_seq={self.last_sequence} "
            f"timer_ms={self._timer_interval_ms} "
            f"same_seq_polls={self.same_sequence_polls} "
            f"seq_jumps={self.sequence_jumps}",
            flush=True,
        )
        self.last_status_time = now

    def request_close(self) -> None:
        if self.closing:
            return

        self.closing = True

        if self.window is not None:
            try:
                glfw.set_window_should_close(self.window, True)
            except Exception:
                pass

        try:
            glfw.post_empty_event()
        except Exception:
            pass

    def stop(self) -> None:
        """Public API to stop the viewer. Cleanup happens in run() finally."""
        self.request_close()

    def _window_should_run(self) -> bool:
        if self.closing or self.window is None:
            return False
        return not glfw.window_should_close(self.window)

    def _window_stop_reason(self) -> str:
        if self.closing:
            return "self.closing is True"
        if self.window is None:
            return "self.window is None"
        try:
            if glfw.window_should_close(self.window):
                return "glfw.window_should_close(self.window) is True"
        except Exception as exc:
            return f"glfw.window_should_close raised: {exc!r}"
        return "window should run"

    def _run_event_loop(self) -> None:
        if not self._window_should_run():
            print(f"[gl-viewer] event loop not started: {self._window_stop_reason()}", flush=True)
            return

        if self.config.fps > 0:
            print(
                f"[gl-viewer] adaptive timer target_fps={self.config.fps} "
                f"target_ms={self._timer_target_ms} "
                f"min_ms={self._timer_min_ms} "
                f"max_ms={self._timer_max_ms}",
                flush=True,
            )

            # Run first frame immediately.
            next_frame_time = 0.0

            while self._window_should_run():
                now = time.monotonic()

                if now >= next_frame_time:
                    glfw.poll_events()
                    self.display_once()
                    next_frame_time = time.monotonic() + self._timer_interval_ms / 1000.0
                    continue

                timeout = max(0.0, next_frame_time - now)
                # Wake occasionally even without UI events, so stop() from another
                # thread is noticed promptly and the adaptive timer stays responsive.
                glfw.wait_events_timeout(min(timeout, 0.05))

        else:
            print("[gl-viewer] running as fast as possible", flush=True)

            while self._window_should_run():
                glfw.poll_events()
                self.display_once()

    def cleanup(self) -> None:
        """Release CUDA/GL/GLFW resources exactly once, in safe interop order."""
        if self._cleaned:
            return

        self._cleaned = True
        print("[ImageMatCUDAPubSub] cleanup", flush=True)

        # The GL context must still be current while deleting GL resources.
        try:
            if self.window is not None:
                glfw.make_context_current(self.window)
        except Exception as exc:
            print("[cleanup] make_context_current:", exc, flush=True)

        # CUDA/GL interop resource must be unregistered before deleting the GL PBO.
        try:
            if self.cuda_pbo is not None:
                try:
                    cuda.Context.synchronize()
                except Exception:
                    pass
                self.cuda_pbo.unregister()
                self.cuda_pbo = None
        except Exception as exc:
            print("[cleanup] cuda_pbo:", exc, flush=True)

        # Optional: only close the image subscriber if this viewer owns it.
        try:
            if (
                self.config.close_img_on_cleanup
                and self.img is not None
                and hasattr(self.img, "close")
            ):
                self.img.close()
                self.img = None
        except Exception as exc:
            print("[cleanup] img:", exc, flush=True)

        try:
            if self.pbo is not None:
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                glDeleteBuffers(1, [int(self.pbo)])
                self.pbo = None
        except Exception as exc:
            print("[cleanup] pbo:", exc, flush=True)

        try:
            if self.tex is not None:
                glBindTexture(GL_TEXTURE_2D, 0)
                glDeleteTextures(1, [int(self.tex)])
                self.tex = None
        except Exception as exc:
            print("[cleanup] tex:", exc, flush=True)

        # Drop CUDA module/function references before detaching the context.
        self.copy_kernel = None
        self.copy_module = None

        try:
            if self.cuda_ctx is not None:
                try:
                    cuda.Context.synchronize()
                except Exception:
                    pass
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
                self.cuda_ctx = None
        except Exception as exc:
            print("[cleanup] cuda_ctx:", exc, flush=True)

        try:
            if self.window is not None:
                glfw.make_context_current(None)
                glfw.destroy_window(self.window)
                self.window = None
        except Exception as exc:
            print("[cleanup] window:", exc, flush=True)

        if self._glfw_acquired:
            try:
                _release_glfw()
            finally:
                self._glfw_acquired = False

        self._initialized = False

    def run(self, img: Optional[ImageMatCUDAPubSub] = None) -> None:
        """Show the viewer window and block until it closes.

        This method is safe to call again after it returns. Each call creates a
        new GLFW window/context and a new CUDA/OpenGL interop context.
        """
        # If caller accidentally reuses a still-live viewer, close the old run first.
        if self._initialized or self.window is not None or self.cuda_ctx is not None:
            self.request_close()
            self.cleanup()

        self._reset_run_state()

        if img is not None:
            self.img = img
        elif self.img is None:
            raise RuntimeError("ImageMat subscriber has not been provided")

        try:
            self.init()
            self.compile_specialized_kernel()

            # Start each run from an explicit open state. If you click close
            # during initialization/compilation, the first poll will close it
            # again; this line prevents stale close flags from old runs.
            self.closing = False
            if self.window is not None:
                glfw.set_window_should_close(self.window, False)

            self._run_event_loop()
        finally:
            self.cleanup()
