#!/usr/bin/env python

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RobotiqUsbGripper:
    """Small adapter for Robotiq 2F grippers connected through USB-RS485."""

    def __init__(
        self,
        port: str = "auto",
        device_id: int = 9,
        speed: int = 255,
        force: int = 255,
        activate_on_connect: bool = True,
        open_on_connect: bool = False,
    ) -> None:
        self.port = port
        self.device_id = device_id
        self.speed = speed
        self.force = force
        self.activate_on_connect = activate_on_connect
        self.open_on_connect = open_on_connect
        self._driver = None
        self._last_position = None

    def connect(self) -> None:
        try:
            from pyrobotiqgripper import RobotiqGripper
        except ImportError as exc:
            raise ImportError(
                "pyrobotiqgripper is required for TJ robotiq_usb gripper control. "
                "Install it in the same Python environment used to run LeRobot."
            ) from exc

        self._driver = RobotiqGripper(
            com_port=self.port,
            device_id=self.device_id,
            connection_type="RTU",
        )
        self._driver.connect()
        if self.activate_on_connect:
            self._driver.activate()
        if self.open_on_connect:
            self._driver.open(speed=self.speed, force=self.force)
            self._last_position = 0
        logger.info("Connected Robotiq USB gripper on %s with device_id=%s", self.port, self.device_id)

    def move_norm(self, value: float) -> int:
        if self._driver is None:
            raise RuntimeError("Robotiq USB gripper is not connected")
        position = int(round(float(np.clip(value, 0.0, 1.0)) * 255.0))
        if position == self._last_position:
            return position
        self._driver.move(position, speed=self.speed, force=self.force)
        self._last_position = position
        return position

    def disconnect(self) -> None:
        if self._driver is None:
            return
        try:
            self._driver.disconnect()
        finally:
            self._driver = None
            self._last_position = None
