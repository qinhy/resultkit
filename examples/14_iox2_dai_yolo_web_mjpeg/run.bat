@echo off
setlocal

REM ============================================================================
REM Configuration
REM ============================================================================

set "APP_DIR=examples\14_iox2_dai_yolo_web_mjpeg"
set "SERVER_DIR=%APP_DIR%\servers\server_"

set "DEVICE_LEFT=169.254.1.221"
set "DEVICE_RIGHT=169.254.1.222"
set "DEVICE_HAND=%DEVICE_RIGHT%"


REM ============================================================================
REM RGB-D cameras Data storage
REM ============================================================================

start "rgbd_left" uv run "%SERVER_DIR%rgb_stereo.py" --controller-name rgbd_left --device "%DEVICE_LEFT%"
start "rgbd_right" uv run "%SERVER_DIR%rgb_stereo.py" --controller-name rgbd_right --device "%DEVICE_RIGHT%"
start "store_dual" uv run "%SERVER_DIR%store.py" --controller-name store_dual --record-mode dual_rgb --rgbd-controller rgbd_left,rgbd_right

start "rgbd_hand" uv run "%SERVER_DIR%rgb_stereo.py" --controller-name rgbd_hand --device "%DEVICE_HAND%"
start "store_hand" uv run "%SERVER_DIR%store.py" --controller-name store_hand --record-mode rgbd_hand --rgbd-controller rgbd_hand


REM ============================================================================
REM Processing services
REM ============================================================================

start "yolo" uv run "%SERVER_DIR%yolo.py"
start "pcd" uv run "%SERVER_DIR%pcd.py"
start "web" uv run "%APP_DIR%\cli.py" api

REM ============================================================================
REM Optional tests
REM ============================================================================

REM start "auto_test" ^
REM     uv run "%APP_DIR%\auto_test.py"

endlocal