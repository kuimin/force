#!/usr/bin/env python

import logging
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FORCE_NAMES = ("fx", "fy", "fz", "mx", "my", "mz")


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class DmTacImageSensors:
    def __init__(
        self,
        *,
        dev_ids: list[int | str],
        sdk_dir: Path | None,
        backend: str,
        mode: str,
        max_fps: int,
        show_fps: bool,
        enable_images: bool,
        enable_force: bool,
        image_shape: tuple[int, int, int],
        force_dim: int,
        auto_count: int,
        cpu_fallback: bool,
        silent_sdk: bool,
        wait_timeout_ms: int,
        remote_addr: str,
        pc_host: str,
        pc_port: int,
    ) -> None:
        self.dev_ids = dev_ids
        self.sdk_dir = sdk_dir
        self.backend = backend
        self.mode = mode
        self.max_fps = max_fps
        self.show_fps = show_fps
        self.enable_images = enable_images
        self.enable_force = enable_force
        self.image_shape = image_shape
        self.force_dim = force_dim
        self.auto_count = auto_count
        self.cpu_fallback = cpu_fallback
        self.silent_sdk = silent_sdk
        self.wait_timeout_ms = wait_timeout_ms
        self.remote_addr = remote_addr
        self.pc_host = pc_host
        self.pc_port = pc_port
        self._sensors: list[Any] = []
        self._active_backends: list[str] = []
        self._feature_sensor_count = self._initial_sensor_count()
        self._last_frame_ids = [-1] * self._feature_sensor_count
        self._put_arrows_on_image = None

    def _initial_sensor_count(self) -> int:
        return self.auto_count if self._uses_auto_discovery(self.dev_ids) else len(self.dev_ids)

    @staticmethod
    def _uses_auto_discovery(dev_ids: list[int | str]) -> bool:
        return any(str(dev_id).strip().lower() == "auto" for dev_id in dev_ids)

    @property
    def features(self) -> dict[str, type | tuple]:
        features = {}
        for index in range(self._feature_sensor_count):
            if self.enable_images:
                features[f"dmtac_{index}_ui_infer"] = self.image_shape
                features[f"dmtac_{index}_ui_deformation"] = self.image_shape
            if self.enable_force:
                for force_index, channel in enumerate(self._force_channel_names()):
                    features[f"dmtac_{index}_force.{channel}"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return len(self._sensors) == self._feature_sensor_count

    def connect(self) -> None:
        if self.is_connected:
            return
        if self.sdk_dir is not None:
            sdk_path = str(self.sdk_dir)
            if sdk_path not in sys.path:
                sys.path.insert(0, sdk_path)

        from dmrobotics import Mode, Sensor, SensorOptions, listConnectedDevIDs
        from dmrobotics.utils import put_arrows_on_image

        self.dev_ids = self._resolve_dev_ids(listConnectedDevIDs)
        self._put_arrows_on_image = put_arrows_on_image
        sensor_mode = Mode.HIGH if self.mode.lower() == "high" else Mode.STANDARD
        self._sensors = []
        self._active_backends = []
        self._last_frame_ids = [-1] * len(self.dev_ids)
        for dev_id in self.dev_ids:
            with _suppress_output(self.silent_sdk):
                sensor, active_backend = self._connect_one_sensor(dev_id, sensor_mode, Sensor, SensorOptions)
            self._sensors.append(sensor)
            self._active_backends.append(active_backend)
        logger.info("Connected %s DM-Tac sensor(s): %s", len(self._sensors), self.dev_ids)

    def _resolve_dev_ids(self, list_connected_dev_ids) -> list[int | str]:
        if not self._uses_auto_discovery(self.dev_ids):
            return [self._normalize_dev_id(dev_id) for dev_id in self.dev_ids]

        with _suppress_output(self.silent_sdk):
            connected = list(list_connected_dev_ids())
        if len(connected) < self.auto_count:
            raise RuntimeError(
                f"DM-Tac auto discovery found {len(connected)} sensor(s), expected {self.auto_count}: {connected}"
            )
        resolved = [self._normalize_dev_id(dev_id) for dev_id in connected[: self.auto_count]]
        logger.info("Auto-discovered DM-Tac sensor(s): %s", resolved)
        return resolved

    def _normalize_dev_id(self, dev_id) -> int | str:
        if isinstance(dev_id, dict):
            if "device_index" in dev_id:
                return int(dev_id["device_index"])
            if "serial" in dev_id:
                return str(dev_id["serial"])
        if isinstance(dev_id, str):
            stripped = dev_id.strip()
            return int(stripped) if stripped.isdigit() else stripped
        return dev_id

    def _connect_one_sensor(self, dev_id, sensor_mode, sensor_cls, options_cls):
        try:
            return self._make_sensor(dev_id, self.backend, sensor_mode, sensor_cls, options_cls), self.backend
        except Exception as exc:
            if self.backend.lower() == "cpu" or not self.cpu_fallback:
                raise
            logger.warning(
                "%s backend failed for DM-Tac dev_id=%s: %s. Falling back to CPU.",
                self.backend,
                dev_id,
                exc,
            )
            return self._make_sensor(dev_id, "cpu", sensor_mode, sensor_cls, options_cls), "cpu"

    def _make_sensor(self, dev_id, backend: str, sensor_mode, sensor_cls, options_cls):
        options = options_cls(
            dev_id=dev_id,
            backend=backend,
            mode=sensor_mode,
            show_fps=self.show_fps,
            max_fps=self.max_fps,
            enable_raw=False,
            enable_deformation=self.enable_force or self.enable_images,
            enable_depth=False,
            enable_shear=False,
            enable_force=self.enable_force,
            remote_addr=self.remote_addr,
            pc_host=self.pc_host,
            pc_port=self.pc_port,
        )
        return sensor_cls(options)

    def read(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        for index, sensor in enumerate(self._sensors):
            if self._active_backends[index].lower() == "flux":
                sensor.getEvents()
            if int(sensor.getDevStatus()) != 0:
                if self.enable_images:
                    obs[f"dmtac_{index}_ui_infer"] = self._blank_image()
                    obs[f"dmtac_{index}_ui_deformation"] = self._blank_image()
                if self.enable_force:
                    self._write_force(obs, index, self._blank_force())
                continue

            sensor.wait_for_new(self._last_frame_ids[index], timeout_ms=self.wait_timeout_ms)
            frame_ids = []
            if self.enable_images:
                infer_frame_id, infer = sensor.getInferImg()
                deformation_frame_id, deformation = sensor.getDeformation2D()
                obs[f"dmtac_{index}_ui_infer"] = self._image_to_array(infer)
                obs[f"dmtac_{index}_ui_deformation"] = self._deformation_to_ui_image(deformation)
                frame_ids.extend([int(infer_frame_id), int(deformation_frame_id)])
            if self.enable_force:
                force_frame_id, force = self._split_frame_result(sensor.getForce())
                self._write_force(obs, index, self._force_to_array(force))
                if force_frame_id is not None:
                    frame_ids.append(int(force_frame_id))
            if frame_ids:
                self._last_frame_ids[index] = max(frame_ids)
        return obs

    def disconnect(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect DM-Tac sensor cleanly: %s", exc)
        self._sensors = []

    def _blank_image(self) -> np.ndarray:
        return np.zeros(self.image_shape, dtype=np.uint8)

    def _blank_force(self) -> np.ndarray:
        return np.zeros((self.force_dim,), dtype=np.float32)

    def _force_channel_names(self) -> tuple[str, ...]:
        if self.force_dim == len(DEFAULT_FORCE_NAMES):
            return DEFAULT_FORCE_NAMES
        return tuple(f"v{index}" for index in range(self.force_dim))

    def _write_force(self, obs: dict[str, np.ndarray], sensor_index: int, force: np.ndarray) -> None:
        for channel, value in zip(self._force_channel_names(), force, strict=True):
            obs[f"dmtac_{sensor_index}_force.{channel}"] = float(value)

    def _image_to_array(self, image: Any) -> np.ndarray:
        img = getattr(image, "img", image)
        if img is None:
            return self._blank_image()
        arr = np.asarray(img)
        return self._array_to_image_shape(arr)

    def _deformation_to_ui_image(self, deformation: Any) -> np.ndarray:
        if deformation is None or self._put_arrows_on_image is None:
            return self._blank_image()
        deformation_arr = np.asarray(deformation)
        if deformation_arr.ndim < 2:
            return self._blank_image()
        canvas = np.zeros(deformation_arr.shape[:2] + (3,), dtype=np.uint8)
        return self._array_to_image_shape(self._put_arrows_on_image(canvas, deformation_arr, step=16, scale=20.0))

    def _array_to_image_shape(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2 and self.image_shape[2] == 3:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[2] == 1 and self.image_shape[2] == 3:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape != self.image_shape:
            logger.warning("DM-Tac image shape %s does not match configured %s", arr.shape, self.image_shape)
            return np.resize(arr, self.image_shape).astype(np.uint8, copy=False)
        return arr.astype(np.uint8, copy=False)

    def _split_frame_result(self, result):
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result
        return None, result

    def _force_to_array(self, force: Any) -> np.ndarray:
        wrench = np.asarray(force, dtype=np.float32).reshape(-1)
        if wrench.size < self.force_dim:
            logger.warning(
                "DM-Tac force shape %s is smaller than configured dim %s",
                np.asarray(force).shape,
                self.force_dim,
            )
            return np.resize(wrench, (self.force_dim,)).astype(np.float32, copy=False)
        return wrench[: self.force_dim].astype(np.float32, copy=False)
