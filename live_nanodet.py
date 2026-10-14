from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2
from nanodet import NanoDet

ROOT = Path(__file__).resolve().parent
LABELS = {0: "person", 15: "cat", 16: "dog"}

cv2.setNumThreads(2)
model = NanoDet(
    modelPath=str(ROOT / "models/nanodet.onnx"),
    prob_threshold=0.40
)

camera = Picamera2()
camera.configure(camera.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 10}
))
app = Flask(__name__)
lock = Lock()

def frames():
    while True:
        with lock:
            frame = camera.capture_array("main")
            height, width = frame.shape[:2]

            # Preserve proportions and pad to the model's 416x416 input.
            scale = min(416 / width, 416 / height)
            new_w = round(width * scale)
            new_h = round(height * scale)
            left = (416 - new_w) // 2
            top = (416 - new_h) // 2
            resized = cv2.resize(frame, (new_w, new_h))
            padded = np.zeros((416, 416, 3), dtype=np.uint8)
            padded[top:top + new_h, left:left + new_w] = resized
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

            start = time.perf_counter()
            predictions = model.infer(rgb)
            elapsed = time.perf_counter() - start

            for prediction in predictions:
                class_id = int(prediction[-1])
                score = float(prediction[-2])
                if class_id not in LABELS:
                    continue

                x1, y1, x2, y2 = prediction[:4]
                x1 = int(np.clip((x1 - left) * width / new_w, 0, width - 1))
                x2 = int(np.clip((x2 - left) * width / new_w, 0, width - 1))
                y1 = int(np.clip((y1 - top) * height / new_h, 0, height - 1))
                y2 = int(np.clip((y2 - top) * height / new_h, 0, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                colour = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    frame, f"{LABELS[class_id]}: {score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2
                )

            cv2.putText(
                frame, f"NanoDet inference: {elapsed:.2f}s",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ok:
                continue
            image = encoded.tobytes()

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + image + b"\r\n"
        )
        time.sleep(0.02)

@app.route("/")
def home():
    return (
        "<title>Bertie: NanoDet</title>"
        "<h2>Bertie: NanoDet dog detection</h2>"
        "<p>No photographs saved. Robot stays stationary.</p>"
        '<img src="/video" style="max-width:100%">'
    )

@app.route("/video")
def video():
    return Response(
        frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"}
    )

if __name__ == "__main__":
    try:
        camera.start()
        time.sleep(2)
        app.run(host="0.0.0.0", port=9001, threaded=True,
                debug=False, use_reloader=False)
    finally:
        camera.close()
