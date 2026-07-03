import cv2
import torch
import torch.multiprocessing as mp
import time
import queue as py_queue
import traceback

from resultkit.dai.rgb_stereo_generator import DepthAIPoeRGBStereoTorchGenerator


def to_small_cv(mat, s=10, rgb_to_bgr=True):
    x = mat.detach()

    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)

    x = x[::s, ::s]

    if x.dtype.is_floating_point:
        if float(x.max()) <= 1.5:
            x = x * 255.0
        x = x.clamp(0, 255).to(torch.uint8)
    else:
        x = x.to(torch.uint8)

    arr = x.cpu().numpy()

    if rgb_to_bgr and arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[:, :, ::-1].copy()

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    return arr


def build_dai_gen(decoder_backend="gst-nvivafilter"):
    return DepthAIPoeRGBStereoTorchGenerator(
        uuid="OkadCam:CamA",
        sources=["169.254.1.222"],
        color_types=[],
        rgb_width=4032,
        rgb_height=3040,
        stereo_width=1280,
        stereo_height=800,
        capture_fps=15,
        rgb_codec="h265",
        stereo_codec="h265",
        rgb_bitrate_kbps=60000,
        stereo_bitrate_kbps=6000,
        decoder_backend=decoder_backend,
        gst_nvivafilter_so="./libdepthai_cuda_preprocess.so",
        gst_nvivafilter_dtype="fp16",
        gst_nvivafilter_channel_order="rgba",
        decoder_output_color="rgbp",
        stereo_decoder_output_color="rgbp",
        rgb_camera_socket="CAM_A",
        left_camera_socket="CAM_B",
        right_camera_socket="CAM_C",
        normalize_rgb=True,
        normalize_stereo=True,
        show_rgb_preview=False,
        show_stereo_preview=False,
        fps=0,
    )


def put_latest(queue, item):
    """
    Keep only latest frames.
    Avoid blocking the producer when the consumer is slow.
    """
    try:
        queue.put_nowait(item)
        return
    except py_queue.Full:
        pass

    try:
        queue.get_nowait()
    except py_queue.Empty:
        pass

    try:
        queue.put_nowait(item)
    except py_queue.Full:
        pass


def prepare_for_queue(t:torch.Tensor):
    t = t.detach().clone()
    return t.share_memory_()


def start_rgb_stereo_steam(decoder_backend, frame_queue, stop_event, preview=True):
    print("start_rgb_stereo_steam process started.")
    gen = None

    try:
        gen = build_dai_gen(decoder_backend=decoder_backend)

        for i, mats in enumerate(gen):
            if stop_event.is_set():
                break

            packed = mats[0].data
            rgb, stereo, left, right = gen.unpack_packed_tensor(packed)

            payload = (
                prepare_for_queue(rgb),
                prepare_for_queue(stereo),
                prepare_for_queue(left),
                prepare_for_queue(right),
            )
            put_latest(frame_queue, payload)

            if preview:
                # rgb_valid = rgb[:, :, :gen.stereo_payload_start_row, :]
                small_rgb = to_small_cv(rgb[0], s=10, rgb_to_bgr=True)
                small_left = to_small_cv(left[0], s=4, rgb_to_bgr=False)
                small_right = to_small_cv(right[0], s=4, rgb_to_bgr=False)

                cv2.imshow("small_rgb", small_rgb)
                cv2.imshow("small_left", small_left)
                cv2.imshow("small_right", small_right)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break

    except KeyboardInterrupt:
        stop_event.set()

    except Exception:
        traceback.print_exc()
        stop_event.set()

    finally:
        try:
            if gen is not None:
                gen.release()
        except Exception:
            traceback.print_exc()

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("start_rgb_stereo_steam process stopping.")


def main():
    decoder_backend = "pynvvideocodec"

    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=3)
    stop_event = ctx.Event()

    p = ctx.Process(
        target=start_rgb_stereo_steam,
        args=(decoder_backend, frame_queue, stop_event),
        daemon=False,
    )

    p.start()

    print("Main program consumer loop starting...")

    try:
        while p.is_alive():
            try:
                rgb, stereo, left, right = frame_queue.get(timeout=0.1)
            except py_queue.Empty:
                continue

            print(f"Received stereo tensor shape: {stereo.shape} on device: {stereo.device}")

    except KeyboardInterrupt:
        print("Main program stopping.")
        stop_event.set()

    finally:
        stop_event.set()

        try:
            while True:
                frame_queue.get_nowait()
        except py_queue.Empty:
            pass

        p.join(timeout=5)

        if p.is_alive():
            print("Child did not exit cleanly; terminating.")
            p.terminate()
            p.join(timeout=2)

        frame_queue.close()
        frame_queue.join_thread()


if __name__ == "__main__":
    mp.freeze_support()
    while True:
        main()