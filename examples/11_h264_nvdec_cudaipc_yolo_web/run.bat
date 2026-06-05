start "" cmd /k "uv run examples\11_h264_nvdec_cudaipc_yolo_web\cil.py web --width 1280 --height 720 --fps 30 --image-topic ImageMatCUDAPubSub:yolo --monitor-width 640"

@REM "ImageMatCUDAPubSub:h264FileDemo" "ImageMatCUDAPubSub:yolo"

start "" cmd /k "uv run examples\11_h264_nvdec_cudaipc_yolo_web\cil.py decode-pub --input examples\demo.h264 --width 1280 --height 720 --fps 30 --loop"

start "" cmd /k "uv run examples\11_h264_nvdec_cudaipc_yolo_web\cil.py torch"

