import argparse
import logging
import os
from pathlib import Path
import sys
import time
import requests


# EXAMPLE_DIR = Path(__file__).absolute().parent/"servers"
# if EXAMPLE_DIR not in sys.path:
#     sys.path.append(str(EXAMPLE_DIR))
# from servers.server_pcd import ToPcdParams, ToYoloSegmentsParams
# from store.custom_record_store import CustomRecord, jst_datetime_to_time_ns
# from iox2redis import redis_for

API_URL = "http://127.0.0.1:8000"

def call_method(controller: str, method: str, params: dict = None) -> dict | None:
    if params is None:
        params = {}
    url = f"{API_URL}/controllers/{controller}/{method}"
    try:
        response = requests.post(url, json=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response body: {e.response.text}")
        return None

def refleshapi():    
    # Force a refresh on the API gateway to ensure it discovers the latest services
    logging.info("Refreshing API gateway controllers...")
    try:
        requests.get(f"{API_URL}/refresh")
    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not reach API gateway at {API_URL}. Is it running? Error: {e}")
        return

def opencam(name):
    logging.info(f"\n=== {name}.open ===")
    call_method(name, "open", {})

def closecam(name):
    logging.info(f"\n=== {name}.close ===")
    call_method(name, "close", {})

def checkcam(name):
    logging.info(f"\n=== {name}.status ===")
    status = call_method(name, "status", {})
    return bool(status and status.get("opened", False))

def store_watch(sn="store_dual",sts=[
        "rgbd_left",
        "rgbd_right"
    ]):
    logging.info(f"\n=== {sn}.watch ===")
    store_watch = call_method(sn, "watch", {
    "service": "jrpc",
    "stream_ids":sts
    })
    logging.info(store_watch)

def store_capture(sn="store_dual",sts=[
        "rgbd_left",
        "rgbd_right"
    ],to_yolo=False,to_pcd=False):
    logging.info(f"\n=== {sn}.capture ===")
    params = {
        "service": "jrpc",
        "stream_ids": sts,
        "field_id": "field_all",
        "meta": {
            # "gnss":{"the_data":"xxxxxxxxx"},
            # "arm":{"run_id":"UUIDXXXX","data":"xxxxxxxxx"}
        },
        "capture_timeout_s": None,
        "fresh_frame": True,
        "hook_urls": [[]]
    }
    if to_yolo:
        params["hook_urls"][0].append(
            "http://localhost:8000/controllers/yolo/start"
        )
    if to_pcd:        
        params["hook_urls"][0].append(
            # "http://localhost:8000/controllers/pcd/to_pcd"
            "http://localhost:8000/controllers/pcd/detect_segments_to_pcd"
        )
    store_capture = call_method(sn, "capture", params)
    logging.info(store_capture)

def store_dual_watch():
    store_watch(sn="store_dual",sts=[
        "rgbd_left",
        "rgbd_right"
    ])

def store_dual_capture():    
    store_capture(sn="store_dual",sts=[
        "rgbd_left",
        "rgbd_right"
    ],to_yolo=True,to_pcd=False)

def store_hand_watch():
    store_watch(sn="store_hand",sts=[
        "rgbd_hand"
    ])

def store_hand_capture():    
    store_capture(sn="store_hand",sts=[
        "rgbd_hand"
    ],to_yolo=True,to_pcd=True)

def yolo_start(params=
  {"db_record": {
    "root_path": ".",
    "mode": "dual_rgb",
    "field_id": "null",
    "record_id": "000000.000000000JST",
    "timestamp_ns_utc": 0,
    "date_utc": "1970-01-01",
    "path": "dual_rgb/1970-01-01/null/000000.000000000JST",
    "datetime_utc": "1970-01-01T00:00:00.000000000Z"
  }}):
    res = call_method("yolo", "start", params)
    logging.info(res)

def yolo_set_model(params={"model_name": "yolo11l-seg.pt",
    #Width:  1280 + 3×864 = 3872
    #Height: 1280 + 2×864 = 3008
    # 1280,2144,3008,3872
    # "detection_bbox_xyxy" : [0,0,0,0]
    }):
    res = call_method("yolo", "set_model", params)
    logging.info(res)

def set_dnn_pcd():
    logging.info(call_method("pcd", "set_backend",{
        "backend": "dnn",
        "repo_dir": "./examples/14_iox2_dai_yolo_web_mjpeg/fast-foundationstereo",
        "model_path": "weights/23-36-37/model_best_bp2_serialize.pth",
        "model_dir": None,
        "device": "cuda",
        "valid_iters": 8,
        "max_disp": 192,
        "hiera": False,
        "model_scale": 1,
        "stereo_input_color_order": "RGB",
        "remove_invisible": True
    }))
    
def pcd_set_backend(backend="sgbm"):
    logging.info(call_method("pcd", "set_backend",{
        "backend": backend,
        "valid_iters": 8,
        "max_disp": 192,
        "hiera": False,
        "model_scale": 1,
        "stereo_input_color_order": "RGB",
        "remove_invisible": True
    }))

def pcd_to_pcd(params):
    res = call_method("pcd", "to_pcd", params)
    logging.info(res)

def pcd_detect_segments_to_pcd(params):
    res = call_method("pcd", "detect_segments_to_pcd", params)
    logging.info(res)

def open_cams(sts = ["rgbd_left","rgbd_right"]):
    for s in sts:
        opencam(s)    
    for s in sts:
        while not checkcam(s):
            time.sleep(2)

def close_cams(sts = ["rgbd_left","rgbd_right"]):
    for s in sts:
        closecam(s)    
    for s in sts:
        while checkcam(s):
            time.sleep(2)

def open_dual_cam():
    open_cams(["rgbd_left","rgbd_right"])

def close_dual_cam():
    close_cams(["rgbd_left","rgbd_right"])

def open_hand_cam():
    open_cams(["rgbd_hand"])

def close_hand_cam():
    close_cams(["rgbd_hand"])

def init_dual_mode():
    open_dual_cam()
    store_dual_watch()

def init_hand_mode():
    pcd_set_backend(backend="sgbm")
    open_hand_cam()
    store_hand_watch()

def close_all_cams():
    close_cams(["rgbd_left","rgbd_right","rgbd_hand"])

def capture_for_seconds(capture_func, duration, delay=0):
    if delay > 0:
        logging.info(f"Waiting {delay} seconds before capture...")
        time.sleep(delay)

    logging.info(f"Starting capture for {duration} seconds")
    start_time = time.monotonic()
    for i in range(duration):
        target_time = start_time + i

        # Wait until the scheduled capture time
        sleep_time = target_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

        logging.info(f"Capture {i + 1}/{duration}")
        capture_func()

    logging.info("Capture finished")


def capture_dual_for_seconds(duration, delay=0):
    capture_for_seconds(
        store_dual_capture,
        duration=duration,
        delay=delay
    )


def capture_hand_for_seconds(duration, delay=0):
    capture_for_seconds(
        store_hand_capture,
        duration=duration,
        delay=delay
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAK AI PCD test")

    parser.add_argument(
        "--close",
        action="store_true",
        help="Close all cameras"
    )

    parser.add_argument(
        "--dual",
        action="store_true",
        help="Initialize dual camera mode"
    )

    parser.add_argument(
        "--hand",
        action="store_true",
        help="Initialize hand camera mode"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=1,
        help="Capture duration in seconds (1 capture per second)"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Delay before starting capture in seconds"
    )

    parser.add_argument(
        "--capture-dual",
        action="store_true",
        help="Capture dual cameras"
    )

    parser.add_argument(
        "--capture-hand",
        action="store_true",
        help="Capture hand camera"
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh API gateway controllers"
    )

    parser.add_argument(
        "--backend",
        choices=["sgbm", "dnn"],
        default=None,
        help="Set PCD backend"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Set YOLO model, for example yolo11l-seg.pt"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of captures"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Interval between captures in seconds"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.refresh:
        refleshapi()

    if args.backend:
        if args.backend == "dnn":
            set_dnn_pcd()
        else:
            pcd_set_backend(args.backend)

    if args.model:
        yolo_set_model({
            "model_name": args.model
        })

    if args.dual:
        init_dual_mode()

    if args.hand:
        init_hand_mode()

    if args.capture_dual:
        capture_dual_for_seconds(
            duration=args.duration,
            delay=args.delay
        )

    if args.capture_hand:
        capture_hand_for_seconds(
            duration=args.duration,
            delay=args.delay
        )

    if args.close:
        close_all_cams()

#     # main()
#     # # pcd_set_backend(backend="sgbm")
#     # # set_dnn_pcd()
#     # for i in range(100):
#     #     store_dual_capture()
#     #     time.sleep(0.1)

#     # refleshapi()
#     yolo_set_model({"model_name": "yolo11l-seg.pt",
#         "tile_batch_size":6,

#         # 0,864,1728,2592                
#         # 1280,2144,3008,3872
#         "detection_bbox_xyxy" : [864,0,3008,3008],
#         # "detection_bbox_xyxy" : [1728,1728,3872,3008],
#         # "tile_batch_size":3,
#     })
#     pcd_set_backend(backend="sgbm")
#     open_hand_cam()
#     store_hand_watch()
#     for i in range(10):
#         store_hand_capture()
#         time.sleep(0.1)
    pass


# # 1. Reset everything
# python3 auto_test --refresh

# # 2. Refresh API
# python3 auto_test --close

# # 3. Set YOLO model
# python3 auto_test --model yolo11l-seg.pt

# # 4. Initialize dual camera mode
# python3 auto_test --dual

# # 5. Dual camera basic capture test
# python3 auto_test --capture-dual --duration 5

# # 6. Dual camera delayed capture test
# python3 auto_test --capture-dual --delay 3 --duration 10

# # 7. Close cameras before switching mode
# python3 auto_test --close

# # 8. Initialize hand camera with SGBM
# python3 auto_test --backend sgbm --hand

# # 9. Hand camera basic capture test
# python3 auto_test --capture-hand --duration 5

# # 10. Hand camera delayed capture test
# python3 auto_test --capture-hand --delay 3 --duration 10

# # 11. Longer hand capture test
# python3 auto_test --capture-hand --duration 30

# # 12. Final cleanup
# python3 auto_test --close