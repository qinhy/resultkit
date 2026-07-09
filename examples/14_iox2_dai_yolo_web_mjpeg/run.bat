start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_rgb_stereo.py --device 169.254.1.222
@REM start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_dual_rgb.py  --devices 169.254.1.221,169.254.1.222
start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_store.py
start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_yolo.py
@REM start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_iox2redis.py /iox2redis --store-file ./iox2redis-store.json
start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\cli.py api
@REM start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\auto_test.py
