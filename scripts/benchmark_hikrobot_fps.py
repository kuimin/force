#!/usr/bin/env python

import argparse
import time
from threading import Event, Thread

from lerobot.cameras.hikrobot import HikrobotCamera, HikrobotCameraConfig


def count_camera_frames(camera: HikrobotCamera, stop_event: Event) -> tuple[int, list[float]]:
    timestamps: list[float] = []
    while not stop_event.is_set():
        try:
            camera.async_read(timeout_ms=1000)
        except TimeoutError:
            continue
        timestamps.append(time.perf_counter())
    return len(timestamps), timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hikrobot camera capture FPS without robot control.")
    parser.add_argument("--serials", nargs="+", required=True, help="Hikrobot serial numbers to open.")
    parser.add_argument("--sdk-path", required=True, help="Path containing MvCameraControl_class.py.")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    cameras = [
        HikrobotCamera(
            HikrobotCameraConfig(
                serial_number=serial,
                width=args.width,
                height=args.height,
                fps=args.fps,
                sdk_path=args.sdk_path,
            )
        )
        for serial in args.serials
    ]

    stop_event = Event()
    results: dict[str, list[float]] = {}

    def worker(serial: str, camera: HikrobotCamera) -> None:
        _, timestamps = count_camera_frames(camera, stop_event)
        results[serial] = timestamps

    try:
        for camera in cameras:
            camera.connect()

        threads = [
            Thread(target=worker, args=(serial, camera), name=f"benchmark_{serial}")
            for serial, camera in zip(args.serials, cameras, strict=True)
        ]
        start = time.perf_counter()
        for thread in threads:
            thread.start()
        time.sleep(args.seconds)
        stop_event.set()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - start

        for serial in args.serials:
            timestamps = results.get(serial, [])
            measured_fps = len(timestamps) / elapsed if elapsed > 0 else 0.0
            print(f"{serial}: {len(timestamps)} frames in {elapsed:.2f}s -> {measured_fps:.1f} FPS")
    finally:
        stop_event.set()
        for camera in cameras:
            if camera.is_connected:
                camera.disconnect()


if __name__ == "__main__":
    main()
