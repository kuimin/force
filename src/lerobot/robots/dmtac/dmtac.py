#!/usr/bin/env python

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_dmtac import DmTacConfig

logger = logging.getLogger(__name__)


class DmTac(Robot):
    config_class = DmTacConfig
    name = "dmtac"

    def __init__(self, config: DmTacConfig):
        super().__init__(config)
        self.config = config
        self._sensor = None
        self._last_frame_id = -1
        self._active_backend = config.backend

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {"tactile.status": int}
        image_shape = (self.config.image_height, self.config.image_width, self.config.image_channels)
        map_shape = (self.config.map_height, self.config.map_width)
        vector_shape = (self.config.map_height, self.config.map_width, 2)

        if self.config.enable_infer:
            features["tactile.infer"] = image_shape
        if self.config.enable_raw:
            features["tactile.raw"] = image_shape
        if self.config.enable_deformation:
            features["tactile.deformation"] = vector_shape
        if self.config.enable_depth:
            features["tactile.depth"] = map_shape
        if self.config.enable_shear:
            features["tactile.shear"] = vector_shape
        if self.config.enable_force:
            features["tactile.force"] = (self.config.force_dim,)
        if self.config.enable_contact_area:
            features["tactile.contact_area"] = float
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._sensor is not None

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self._sensor = self._make_sensor(self.config.backend)
            self._active_backend = self.config.backend
        except Exception as exc:
            if self.config.backend.lower() == "cpu" or not self.config.cpu_fallback:
                raise
            logger.warning(
                "%s backend failed during DM-Tac initialization: %s. Falling back to CPU.",
                self.config.backend,
                exc,
            )
            self._sensor = self._make_sensor("cpu")
            self._active_backend = "cpu"

        logger.info("%s connected with %s backend.", self, self._active_backend)

    def _make_sensor(self, backend: str):
        if self.config.sdk_dir is not None:
            sdk_path = str(Path(self.config.sdk_dir))
            if sdk_path not in sys.path:
                sys.path.insert(0, sdk_path)

        from dmrobotics import Mode, Sensor, SensorOptions

        mode = Mode.HIGH if self.config.mode.lower() == "high" else Mode.STANDARD
        options = SensorOptions(
            dev_id=self.config.dev_id,
            backend=backend,
            mode=mode,
            show_fps=self.config.show_fps,
            max_fps=self.config.max_fps,
            enable_raw=self.config.enable_raw,
            enable_deformation=self.config.enable_deformation or self.config.enable_force,
            enable_depth=self.config.enable_depth,
            enable_shear=self.config.enable_shear,
            enable_force=self.config.enable_force,
            remote_addr=self.config.remote_addr,
            pc_host=self.config.pc_host,
            pc_port=self.config.pc_port,
        )
        return Sensor(options)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._active_backend.lower() == "flux":
            self._sensor.getEvents()

        obs: RobotObservation = {"tactile.status": int(self._sensor.getDevStatus())}
        if obs["tactile.status"] != 0:
            return obs

        self._sensor.wait_for_new(self._last_frame_id, timeout_ms=self.config.wait_timeout_ms)

        if self.config.enable_infer:
            self._read_image(obs, "tactile.infer", self._sensor.getInferImg)
        if self.config.enable_raw:
            self._read_image(obs, "tactile.raw", self._sensor.getRawImg)
        if self.config.enable_deformation:
            self._read_array(obs, "tactile.deformation", self._sensor.getDeformation2D)
        if self.config.enable_depth:
            self._read_array(obs, "tactile.depth", self._sensor.getDepth)
        if self.config.enable_shear:
            self._read_array(obs, "tactile.shear", self._sensor.getShear)
        if self.config.enable_force:
            self._read_force(obs)
        if self.config.enable_contact_area:
            obs["tactile.contact_area"] = float(self._sensor.getContactArea())
        return obs

    def _read_image(self, obs: RobotObservation, key: str, getter) -> None:
        result = getter()
        if isinstance(result, (tuple, list)) and len(result) == 2:
            frame_id, image = result
        else:
            frame_id, image = None, result
        img = getattr(image, "img", image)
        if img is not None:
            obs[key] = np.asarray(img)
            if frame_id is not None:
                self._last_frame_id = int(frame_id)

    def _read_array(self, obs: RobotObservation, key: str, getter) -> None:
        result = getter()
        if isinstance(result, (tuple, list)) and len(result) == 2:
            frame_id, value = result
        else:
            frame_id, value = None, result
        if value is not None:
            obs[key] = np.asarray(value)
            if frame_id is not None:
                self._last_frame_id = int(frame_id)

    def _read_force(self, obs: RobotObservation) -> None:
        result = self._sensor.getForce()
        if isinstance(result, (tuple, list)) and len(result) == 2:
            frame_id, value = result
        else:
            frame_id, value = None, result

        wrench = np.asarray(value, dtype=np.float32).reshape(-1)
        if wrench.size < self.config.force_dim:
            raise ValueError(
                f"Expected at least {self.config.force_dim} force values, got shape={np.asarray(value).shape}"
            )
        obs["tactile.force"] = wrench[: self.config.force_dim]
        if frame_id is not None:
            self._last_frame_id = int(frame_id)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        return {}

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def reset(self) -> None:
        if self._sensor is not None:
            self._sensor.reset()

    def disconnect(self) -> None:
        if self._sensor is None:
            return
        try:
            self._sensor.disconnect()
        finally:
            self._sensor = None
        logger.info("%s disconnected.", self)
