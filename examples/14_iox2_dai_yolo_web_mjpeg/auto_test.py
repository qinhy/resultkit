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
    return status.get("opened",False)

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
        "meta": {},
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

def yolo_set_model(params={"model_name": "weed_yolo_seg_1280.pt",}):
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

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    time.sleep(5)
    refleshapi()

    sts = ["rgbd_left","rgbd_right"]
    for s in sts:
        opencam(s)
    
    for s in sts:
        while not checkcam(s):
            time.sleep(5)

    store_watch()
    time.sleep(3)

    store_capture(to_yolo=True,to_pcd=True)

    # # get decodepub status
    # logging.info("\n=== decodepub.status ===")
    # res_decode_status = call_method("decodepub", "status")
    # logging.info(res_decode_status)

    # # start yolo
    # logging.info("\n=== yolo.start ===")
    # res_yolo_start = call_method("yolo", "start", {})
    # logging.info(res_yolo_start)

    # time.sleep(5)

    # # get yolo status
    # logging.info("\n=== yolo.status ===")
    # res_yolo_status = call_method("yolo", "status")
    # logging.info(res_yolo_status)

    # # change yolo model
    # logging.info("\n=== yolo.set_model ===")
    # r = redis_for(host="/iox2redis/", decode_responses=True)
    # while not r.ping():
    #     time.sleep(1)
    #     logging.info("Waiting for iox2redis...")

    # default_yolo_settings = {
    #     "model_name": "yolov8s.pt",  # changing to a different model as an example
    #     "confidence": 0.9,
    #     "iou": 0.45,
    #     "max_detections": 100,
    #     "stride": 32,
    # }
    # last_yolo_settings = r.get_json("yolo_settings")
    # if last_yolo_settings is None:
    #     last_yolo_settings = default_yolo_settings
        
    # res_yolo_model = call_method("yolo", "set_model", last_yolo_settings)
    # logging.info(res_yolo_model)

    # time.sleep(5)

    # # get yolo status
    # logging.info("\n=== yolo.status ===")
    # res_yolo_status2 = call_method("yolo", "status")
    # logging.info(res_yolo_status2)

    # # wait for quit
    # input("Press Enter to quit...")
    

if __name__ == "__main__":
    # main()
    # # pcd_set_backend(backend="sgbm")
    # # set_dnn_pcd()
    # for i in range(100):
    #     store_dual_capture()
    #     time.sleep(0.1)

    refleshapi()
    pcd_set_backend(backend="dnn")
    open_hand_cam()
    store_hand_watch()
    for i in range(10):
        store_hand_capture()
        time.sleep(0.1)
    pass