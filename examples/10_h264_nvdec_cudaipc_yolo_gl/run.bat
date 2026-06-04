start "" cmd /k "uv run examples\10_h264_nvdec_cudaipc_yolo_gl\cil.py show --width 1280 --height 720 --fps 30 --image-topic ImageMatCUDAPubSub:yolo"

@REM "ImageMatCUDAPubSub:h264FileDemo" "ImageMatCUDAPubSub:yolo"

start "" cmd /k "uv run examples\10_h264_nvdec_cudaipc_yolo_gl\cil.py decode-pub --input examples\demo.h264 --width 1280 --height 720 --fps 30 --loop"

start "" cmd /k "uv run examples\10_h264_nvdec_cudaipc_yolo_gl\cil.py torch"

