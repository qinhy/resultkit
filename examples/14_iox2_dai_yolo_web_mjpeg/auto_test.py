import logging
import time
import requests
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

def checkcam(name):
    logging.info(f"\n=== {name}.status ===")
    status = call_method(name, "status", {})
    return status.get("opened",False)

def store_watch(sts=[
        "rgbd_left",
        "rgbd_right"
    ]):
    logging.info("\n=== store.watch ===")
    store_watch = call_method("store", "watch", {
    "service": "jrpc",
    "stream_ids":sts
    })
    logging.info(store_watch)

def store_capture(sts=[
        "rgbd_left",
        "rgbd_right"
    ],to_yolo=True,to_pcd=False):
    logging.info("\n=== store.capture ===")
    params = {
        "service": "jrpc",
        "stream_ids": sts,
        "field_id": "field_01",
        "gis": None,
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
            "http://localhost:8000/controllers/pcd/to_pcd"
        )
    store_capture = call_method("store", "capture", params)
    logging.info(store_capture)


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
    
def set_sgbm_pcd():
    logging.info(call_method("pcd", "set_backend",{
        "backend": "sgbm",
        "valid_iters": 8,
        "max_disp": 192,
        "hiera": False,
        "model_scale": 1,
        "stereo_input_color_order": "RGB",
        "remove_invisible": True
    }))

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    time.sleep(5)
    refleshapi()

    sts = [
        "rgbd_left",
        "rgbd_right"
    ]
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
    # set_sgbm_pcd()
    set_dnn_pcd()
    store_capture(to_yolo=True,to_pcd=True)
    pass