from pathlib import Path
from threading import Lock, Condition, Thread
import time
import os
import traceback

import cv2
import numpy as np
from flask import Flask, Response, make_response
from picamera2 import Picamera2
from nanodet import NanoDet
from picarx import Picarx

# Preserve existing initialization and parameters
px = Picarx()
px.stop()
px.set_cam_pan_angle(0)

pan_angle = 0.0
dog_streak = 0
PAN_DIRECTION = 1  # Change to -1 if it turns away from the dog.

ROOT = Path(__file__).resolve().parent
LABELS = {0: "person", 15: "cat", 16: "dog"}

# Optional timing logs - enable by setting TRACKER_TIMINGS=1 in the environment
LOG_TIMINGS = os.getenv("TRACKER_TIMINGS") == "1"

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

# Synchronization primitives for worker <-> viewers
cam_lock = Lock()          # Protect camera capture if needed
frame_cond = Condition()   # Notify viewers of a new frame

# Shared state published by the worker
_latest_jpeg = None        # bytes
_frame_seq = 0             # monotonically increasing sequence number
_worker_running = False
_worker_thread = None
_worker_exception = None


def _worker_loop():
    global _latest_jpeg, _frame_seq, _worker_running, _worker_exception
    global pan_angle, dog_streak

    try:
        while _worker_running:
            t0 = time.perf_counter()
            # Capture
            with cam_lock:
                frame = camera.capture_array("main")
            capture_t = time.perf_counter() - t0

            # Preprocessing (preserve existing pipeline)
            t1 = time.perf_counter()
            height, width = frame.shape[:2]

            scale = min(416 / width, 416 / height)
            new_w = round(width * scale)
            new_h = round(height * scale)
            left = (416 - new_w) // 2
            top = (416 - new_h) // 2
            resized = cv2.resize(frame, (new_w, new_h))
            padded = np.zeros((416, 416, 3), dtype=np.uint8)
            padded[top:top + new_h, left:left + new_w] = resized
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            preprocess_t = time.perf_counter() - t1

            # Inference
            t2 = time.perf_counter()
            predictions = model.infer(rgb)
            infer_t = time.perf_counter() - t2

            # Tracking: select dogs and update pan when stable
            dogs = [
                p for p in predictions
                if int(p[-1]) == 16 and float(p[-2]) >= 0.50
            ]
            dog_streak = dog_streak + 1 if dogs else 0

            if dog_streak >= 3:
                target = max(
                    dogs,
                    key=lambda p: (p[2] - p[0]) * (p[3] - p[1])
                )
                centre_x = (
                    ((target[0] + target[2]) / 2 - left)
                    * width / new_w
                )
                error = (centre_x - width / 2) / (width / 2)

                if abs(error) > 0.15:
                    step = float(np.clip(error * 3, -2, 2))
                    pan_angle = float(np.clip(
                        pan_angle + PAN_DIRECTION * step, -20, 20
                    ))
                    # Send servo command (hardware)
                    px.set_cam_pan_angle(pan_angle)

            # Drawing and encode (annotate original frame)
            t3 = time.perf_counter()
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
                frame, f"NanoDet inference: {infer_t:.2f}s",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ok:
                # Skip publishing this frame
                if LOG_TIMINGS:
                    print("[worker] JPEG encode failed, skipping frame")
                time.sleep(0.01)
                continue

            image = encoded.tobytes()
            drawencode_t = time.perf_counter() - t3

            # Publish latest frame and sequence number
            with frame_cond:
                _latest_jpeg = image
                _frame_seq += 1
                frame_cond.notify_all()

            if LOG_TIMINGS:
                print(f"[worker] timings capture={capture_t:.3f}s preprocess={preprocess_t:.3f}s infer={infer_t:.3f}s draw+encode={drawencode_t:.3f}s seq={_frame_seq}")

            # Small sleep to avoid tight loop; original loop slept 0.02s after yield
            time.sleep(0.02)

    except Exception as e:
        # Record exception and notify viewers that worker is stopping
        _worker_exception = e
        print("[worker] exception:")
        traceback.print_exc()
        with frame_cond:
            _worker_running = False
            frame_cond.notify_all()


def start_worker():
    global _worker_thread, _worker_running, _worker_exception
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_exception = None
    _worker_running = True
    _worker_thread = Thread(target=_worker_loop, name="tracker-worker", daemon=True)
    _worker_thread.start()


def stop_worker(timeout=None):
    """Signal the worker to stop and join.

    Returns True if the worker thread exited, False if it is still alive after timeout.
    """
    global _worker_thread, _worker_running
    _worker_running = False
    # Wake any waiting viewers so they can exit
    with frame_cond:
        frame_cond.notify_all()

    if _worker_thread is None:
        return True

    _worker_thread.join(timeout)
    alive = _worker_thread.is_alive()
    if not alive:
        # Only clear the reference once the thread has fully exited
        _worker_thread = None
    return not alive


def frames():
    """Generator for streaming latest frames to browser viewers.

    Viewers wait for a new frame sequence; duplicates are not resent.
    A viewer connecting before the first frame will wait until a JPEG
    exists and its sequence differs from the viewer's last seen sequence,
    or until the worker stops.
    """
    last_seq = -1
    while True:
        with frame_cond:
            # Wait until there is a new frame (and we have a JPEG) or the worker stops
            while ((_frame_seq == last_seq) or (_latest_jpeg is None)) and _worker_running:
                frame_cond.wait()

            # If there was an exception in the worker, log and stop
            if _worker_exception is not None:
                print("[frames] worker exception detected, stopping viewer")
                break

            # If worker stopped and there is no new frame, exit
            if ((_frame_seq == last_seq) or (_latest_jpeg is None)) and not _worker_running:
                break

            image = _latest_jpeg
            last_seq = _frame_seq

        if image is None:
            # No image to send (worker may have stopped); end generator
            break

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + image + b"\r\n"
        )


@app.route("/")
def home():
    return (
        "<title>Bertie: NanoDet</title>"
        "<h2>Bertie: dog head tracking</h2>"
        "<p>No photographs saved. Robot stays stationary.</p>"
        "<p>Tracking runs independently of browser viewers.</p>"
        '<img src="/video" style="max-width:100%">'
    )


@app.route("/video")
def video():
    return Response(
        frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"}
    )


@app.route("/status")
def status():
    """Simple health endpoint exposing worker state and last-frame availability.

    - 200: worker running and frames available
    - 503: worker running but no frame yet
    - 500: worker exception or worker stopped with error
    """
    if _worker_exception is not None:
        resp = make_response(f"worker_exception: {_worker_exception}\n", 500)
        return resp
    if _worker_running and _latest_jpeg is not None:
        return "ok\n", 200
    if _worker_running and _latest_jpeg is None:
        return "starting\n", 503
    return "stopped\n", 503


if __name__ == "__main__":
    try:
        # Start camera and allow warm-up BEFORE the worker starts (preserve original behaviour)
        camera.start()
        time.sleep(2)

        # Start the background worker which owns capture, inference and servo updates
        start_worker()

        app.run(host="0.0.0.0", port=9001, threaded=True,
                debug=False, use_reloader=False)
    finally:
        # Clean shutdown: stop worker, join, then stop robot and camera only after worker exit
        try:
            joined = stop_worker(timeout=5)
            if not joined:
                # Worker did not exit in timeout; attempt a longer join and do not proceed to cleanup
                print("[main] worker did not exit within 5s, waiting an additional 10s")
                joined = stop_worker(timeout=10)
                if not joined:
                    print("[main] Worker still alive; aborting hardware cleanup to avoid concurrent access")
                    # Do not proceed to cleanup because worker may still be using hardware
                    raise SystemExit(1)
        finally:
            try:
                px.stop()
                px.set_cam_pan_angle(0)
            finally:
                camera.close()
