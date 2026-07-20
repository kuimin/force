#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Open an Intel RealSense camera and display its color stream.

Examples:

    python src/lerobot/scripts/lerobot_view_realsense.py
    python src/lerobot/scripts/lerobot_view_realsense.py --serial-number 0123456789
    python src/lerobot/scripts/lerobot_view_realsense.py --device-index 6 --width 640 --height 480 --fps 30

Press ``q`` or ``Esc`` in the preview window to exit.
"""

import argparse
import time

import cv2

from lerobot.cameras import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


def _find_serial_number() -> str:
    cameras = RealSenseCamera.find_cameras()
    if not cameras:
        raise RuntimeError("No RealSense camera found. Check the USB connection and camera permissions.")

    if len(cameras) > 1:
        camera_list = ", ".join(f"{camera['name']} ({camera['id']})" for camera in cameras)
        raise RuntimeError(
            f"Found multiple RealSense cameras: {camera_list}. Select one with --serial-number."
        )

    camera = cameras[0]
    print(f"Using RealSense camera: {camera['name']} ({camera['id']})")
    return str(camera["id"])


def _show_frame(window_name: str, frame, frame_count: int, started_at: float) -> bool:
    elapsed = time.perf_counter() - started_at
    measured_fps = frame_count / elapsed if elapsed > 0 else 0.0
    cv2.putText(
        frame,
        f"FPS: {measured_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF in (ord("q"), 27)


def view_video_device(device_index: int, width: int | None, height: int | None, fps: int | None) -> None:
    if any(value is not None for value in (width, height, fps)) and not all(
        value is not None for value in (width, height, fps)
    ):
        raise ValueError("--width, --height, and --fps must be provided together.")
    capture = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open /dev/video{device_index}.")

    print(f"Using OpenCV video device /dev/video{device_index}. Press q or Esc to exit.")
    frame_count = 0
    started_at = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read a frame from /dev/video{device_index}.")
            frame_count += 1
            if _show_frame(f"Video device {device_index}", frame, frame_count, started_at):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()


def view_realsense(serial_number: str | None, width: int | None, height: int | None, fps: int | None) -> None:
    if any(value is not None for value in (width, height, fps)) and not all(
        value is not None for value in (width, height, fps)
    ):
        raise ValueError("--width, --height, and --fps must be provided together.")

    camera = RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name=serial_number or _find_serial_number(),
            width=width,
            height=height,
            fps=fps,
            # OpenCV imshow expects BGR, so avoid a conversion in the display loop.
            color_mode=ColorMode.BGR,
        )
    )

    window_name = f"RealSense {camera.serial_number}"
    frame_count = 0
    started_at = time.perf_counter()

    try:
        camera.connect(warmup=True)
        print("Preview started. Press q or Esc to exit.")

        while True:
            frame = camera.read()
            frame_count += 1
            if _show_frame(window_name, frame, frame_count, started_at):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if camera.is_connected:
            camera.disconnect()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display live video from an Intel RealSense camera.")
    parser.add_argument("--serial-number", help="Camera serial number. Auto-detected when only one is connected.")
    parser.add_argument("--device-index", type=int, help="OpenCV device index, for example 6 for /dev/video6.")
    parser.add_argument("--width", type=int, help="Color stream width; requires --height and --fps.")
    parser.add_argument("--height", type=int, help="Color stream height; requires --width and --fps.")
    parser.add_argument("--fps", type=int, help="Color stream FPS; requires --width and --height.")
    args = parser.parse_args()
    if args.device_index is not None and args.serial_number is not None:
        parser.error("Use either --device-index or --serial-number, not both.")

    # Keep the original `--serial-number 6` command useful: short numbers denote /dev/videoN.
    device_index = args.device_index
    if (
        device_index is None
        and args.serial_number is not None
        and args.serial_number.isdigit()
        and len(args.serial_number) <= 2
    ):
        device_index = int(args.serial_number)

    if device_index is not None:
        view_video_device(device_index, args.width, args.height, args.fps)
    else:
        view_realsense(args.serial_number, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
