start uv run .\examples\13_iox2_jrpc_yolo_web_mjpeg\servers\server_decodepub.py
start uv run .\examples\13_iox2_jrpc_yolo_web_mjpeg\servers\server_yolo.py
start uv run .\examples\13_iox2_jrpc_yolo_web_mjpeg\servers\server_glshow.py
start uv run .\examples\13_iox2_jrpc_yolo_web_mjpeg\servers\server_iox2redis.py /iox2redis --store-file ./iox2redis-store.json
start uv run .\examples\13_iox2_jrpc_yolo_web_mjpeg\cli.py api
start uv run .\examples\13_iox2_jrpc_yolo_web_mjpeg\auto_test.py
