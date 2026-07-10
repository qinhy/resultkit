@REM .\examples\14_iox2_dai_yolo_web_mjpeg\cls.bat
start "rgbd_left"  uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_rgb_stereo.py --controller-name rgbd_left  --device 169.254.1.221
start "rgbd_right" uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_rgb_stereo.py --controller-name rgbd_right --device 169.254.1.222
start "store_dual" uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_store.py --rgbd-controller rgbd_left,rgbd_right

@REM start "rgbd_hand" uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_rgb_stereo.py --device 169.254.1.222
@REM start "store_hand" uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_store.py

start "yolo" uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_yolo.py
start "pcd" uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_pcd.py
@REM start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\servers\server_iox2redis.py /iox2redis --store-file ./iox2redis-store.json
start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\cli.py api
@REM start uv run .\examples\14_iox2_dai_yolo_web_mjpeg\auto_test.py
