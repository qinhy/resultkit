# fastapi_multi_mjpeg.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import threading
import time
import math
import asyncio

app = FastAPI()


class MJPEGPublisher:
    def __init__(self, stream_id: int, fps: int = 30):
        self.stream_id = stream_id
        self.fps = fps

        self.latest_jpeg: bytes | None = None
        self.lock = threading.Lock()

        self.running = True
        self.viewers = 0

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add_viewer(self):
        with self.lock:
            self.viewers += 1
            print(f"/cam/{self.stream_id}: viewer joined, viewers={self.viewers}")

    def remove_viewer(self):
        with self.lock:
            self.viewers = max(0, self.viewers - 1)
            print(f"/cam/{self.stream_id}: viewer left, viewers={self.viewers}")

    def has_viewers(self) -> bool:
        with self.lock:
            return self.viewers > 0

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def _make_simulated_frame(self, t: float):
        width, height = 640, 360
        img = np.zeros((height, width, 3), dtype=np.uint8)

        phase = self.stream_id * 1.5
        x = int((math.sin(t + phase) * 0.5 + 0.5) * (width - 80)) + 40
        y = height // 2

        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
        ]

        color = colors[self.stream_id % len(colors)]

        cv2.circle(img, (x, y), 40, color, -1)

        cv2.putText(
            img,
            f"FastAPI /cam/{self.stream_id}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            img,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

        cv2.putText(
            img,
            f"Viewers: {self.viewers}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

        return img

    def _run(self):
        t = 0.0
        delay = 1.0 / self.fps

        while self.running:
            if not self.has_viewers():
                time.sleep(0.1)
                continue

            img = self._make_simulated_frame(t)

            ok, jpeg = cv2.imencode(
                ".jpg",
                img,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            if ok:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()

            t += 0.08
            time.sleep(delay)


publishers = {
    0: MJPEGPublisher(0),
    1: MJPEGPublisher(1),
    2: MJPEGPublisher(2),
}


async def mjpeg_generator(pub: MJPEGPublisher):
    pub.add_viewer()

    try:
        while True:
            frame = pub.get_jpeg()

            if frame is None:
                await asyncio.sleep(0.01)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                b"\r\n" + frame + b"\r\n"
            )

            await asyncio.sleep(0.001)

    finally:
        # Runs when browser closes tab, leaves page, or img.src is cleared.
        pub.remove_viewer()


# @app.get("/", response_class=HTMLResponse)
# async def index():
#     return """
#     <!doctype html>
#     <html>
#       <head>
#         <title>FastAPI MJPEG Demo</title>
#         <style>
#           body {
#             font-family: sans-serif;
#             background: #111;
#             color: white;
#           }

#           .cam {
#             display: inline-block;
#             margin: 12px;
#           }

#           img {
#             width: 420px;
#             border: 2px solid #555;
#             background: black;
#           }

#           button {
#             margin-top: 6px;
#             padding: 6px 12px;
#           }
#         </style>
#       </head>

#       <body>
#         <h1>FastAPI Multiple MJPEG Publishers</h1>

#         <div class="cam">
#           <h3>/cam/0</h3>
#           <img id="cam0" src="/cam/0">
#           <br>
#           <button onclick="stopCam('cam0')">Stop cam 0</button>
#           <button onclick="startCam('cam0', '/cam/0')">Start cam 0</button>
#         </div>

#         <div class="cam">
#           <h3>/cam/1</h3>
#           <img id="cam1" src="/cam/1">
#           <br>
#           <button onclick="stopCam('cam1')">Stop cam 1</button>
#           <button onclick="startCam('cam1', '/cam/1')">Start cam 1</button>
#         </div>

#         <div class="cam">
#           <h3>/cam/2</h3>
#           <img id="cam2" src="/cam/2">
#           <br>
#           <button onclick="stopCam('cam2')">Stop cam 2</button>
#           <button onclick="startCam('cam2', '/cam/2')">Start cam 2</button>
#         </div>

#         <script>
#           function stopCam(id) {
#             const img = document.getElementById(id);
#             img.src = "";
#           }

#           function startCam(id, url) {
#             const img = document.getElementById(id);
#             img.src = url + "?t=" + Date.now();
#           }

#           window.addEventListener("beforeunload", () => {
#             document.querySelectorAll("img").forEach(img => {
#               img.src = "";
#             });
#           });
#         </script>
#       </body>
#     </html>
#     """


@app.get("/cam/{stream_id}")
async def cam(stream_id: int):
    pub = publishers.get(stream_id)

    if pub is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        mjpeg_generator(pub),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "close",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app=app,
        factory=False,
        host="0.0.0.0",
        port=8080,
        reload=False,
    )