# PiCar-X Dog Tracker

Dog detection and horizontal camera tracking using Python, OpenCV and
NanoDet on a Raspberry Pi-powered SunFounder PiCar-X.

I'm an AWS Platform Engineer exploring AI, computer vision and robotics
through practical experiments with my dogs, Scout and Charlie.

## Current capabilities

- Pretrained NanoDet object detection.
- Horizontal camera tracking while the robot stays stationary.
- Browser video stream with bounding boxes and confidence scores.
- Largest detected dog selected as the target.
- Three consecutive dog detections required before movement.
- Dead zone to reduce twitching, movement steps capped at 2 degrees,
  and pan limited to ±20 degrees.
- No photographs or recordings saved.

The model detects the generic dog class. It does not identify individual dogs.

## Hardware and verified software

| Component | Version |
|---|---|
| Hardware | Raspberry Pi 4, PiCar-X v2.0 |
| OpenCV | 4.10.0 |
| NumPy | 2.2.4 |
| Flask | 3.1.3 |
| Picamera2 | 0.3.37 |
| PiCar-X library | 2.1.0a1 |

Camera capture is configured for 640 × 480 at 10 FPS.
Model input is padded to 416 × 416 while preserving aspect ratio.
Configured capture rate is not end-to-end processing speed.

## Files

- `live_tracking.py`: detection, camera pan tracking and browser streaming. (Updated: now uses a single background worker for capture, inference, servo updates, and publishing the latest encoded JPEG to viewers.)
- `live_nanodet.py`: detection and streaming without servo control.
- `nanodet.py`: inference helper adapted from OpenCV Zoo.
- `live_detector.py`: earlier MobileNet SSD experiment.
- `models/`: local model assets; binary weights are excluded from Git.

## Setup and running

A working Raspberry Pi camera, OpenCV, Picamera2, Flask, NumPy and
SunFounder robot libraries are required.

Clone: `git clone https://github.com/eglebudina-arch/picarx-dog-tracker.git`

Enter the folder: `cd picarx-dog-tracker`

Download the model below and save it as `models/nanodet.onnx`:

https://huggingface.co/opencv/opencv_zoo/resolve/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx

Expected SHA-256:

`4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`

The model URL follows upstream main; verify the checksum before use.

Start tracking: `ROBOT_HAT_GPIOCHIP=0 python3 live_tracking.py`

Open `http://<PI-IP-ADDRESS>:9001` on your local network.
Opening the video stream starts tracking. Press Ctrl+C to stop.

Detection only: `python3 live_nanodet.py`

Run one application and use one browser viewer at a time.
The server has no authentication and is intended for a trusted local network.

## Results so far

NanoDet inference took approximately 0.21 seconds for a test frame on
the Pi 4. This excludes capture, application preprocessing and streaming.

Scout's lighter coat was detected and tracked successfully in informal tests.
Charlie’s dark coat was harder to detect in low light, with occasional
cat or person classifications. Brighter lighting helped.

These are observations, not a measured accuracy benchmark.

## Limitations and behaviour changes (updated)

- Pan tracking only; tilt and autonomous driving are not implemented.
- Tracking now runs independently of browser viewers (background worker).
- Each browser viewer no longer starts a separate processing loop; viewers
  now consume the latest JPEG produced by the single worker.
- The previous behaviour where tracking stopped when no viewers were
  connected has been removed: tracking continues even with zero viewers.
- Hardware initialises at module import.
- The helper uses class-agnostic non-maximum suppression.
- Reliable shutdown after streaming still needs verification on real
  hardware; the code attempts to ensure the worker exits before
  hardware cleanup, but this must be validated on a Pi.

## GPIO troubleshooting

Robot HAT selected gpiochip512 automatically on the tested Pi.
Setting `ROBOT_HAT_GPIOCHIP=0` resolved startup without sudo.
Other setups may require a different chip number.

## Roadmap

- Separate detection, tracking and streaming into small modules.
- Add configuration, logging and a dry-run mode.
- Test target selection and movement limits without hardware.
- Share one processing loop across browser viewers. (Implemented in this PR.)
- Measure performance under different lighting conditions.
- Explore tilt tracking and individual dog recognition.
- Later investigate navigation, mapping and voice interaction.

## Validation (hardware-free tests added)

These tests can be run without moving servos or a Pi camera by mocking
`camera.capture_array` and `px.set_cam_pan_angle` in unit tests. They
help validate the new worker/viewer coordination and shutdown ordering.

1) Connection before first frame
- Start the application but prevent the worker from publishing frames
  initially (mock `_latest_jpeg` as None). Connect a viewer and assert
  the viewer blocks until `_latest_jpeg` becomes non-None and the
  sequence changes.

2) Two viewers sharing frames
- Connect two viewers; ensure both receive the same `_frame_seq` values
  in order and that only one worker `model.infer` call occurs per frame
  (mock `model.infer` and count calls).

3) Waiting viewers wake on stop/failure
- While viewers are waiting, cause the worker to set `_worker_running`
  to False and optionally set `_worker_exception`. Verify viewers are
  woken and generators exit.

4) Shutdown ordering
- Simulate a slow worker by delaying its exit. Call `stop_worker(timeout=5)`
  and verify that hardware cleanup does not proceed until the worker has
  exited or the process aborts as a safety measure.

## Attribution and licensing

The NanoDet helper and model originate from OpenCV Zoo:

https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet

The helper includes local NumPy array-construction adaptations.
The upstream NanoDet license is Apache 2.0; see `third_party/nanodet-LICENSE`.

Hardware control uses SunFounder's PiCar-X library:

https://github.com/sunfounder/picar-x

Third-party components retain their respective licenses.
Project licensing and the earlier MobileNet experiment's attribution
are being finalised before public code upload.
