@REM ./examples/14_iox2_dai_yolo_web_mjpeg/cls.bat
@REM start "rgbd_left"  uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_rgb_stereo.py --controller-name rgbd_left  --device 169.254.1.221
@REM start "rgbd_right" uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_rgb_stereo.py --controller-name rgbd_right --device 169.254.1.222
@REM start "store_dual" uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_store.py --record-mode dual_rgb --rgbd-controller rgbd_left,rgbd_right

start "rgbd_hand" uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_rgb_stereo.py --device 169.254.1.222
start "store_hand" uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_store.py --record-mode rgbd_hand --rgbd-controller rgbd_hand

start "yolo" uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_yolo.py
start "pcd" uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_pcd.py
@REM start uv run ./examples/14_iox2_dai_yolo_web_mjpeg/servers/server_iox2redis.py /iox2redis --store-file ./iox2redis-store.json
start "web" uv run ./examples/14_iox2_dai_yolo_web_mjpeg/cli.py api
@REM start uv run ./examples/14_iox2_dai_yolo_web_mjpeg/auto_test.py
