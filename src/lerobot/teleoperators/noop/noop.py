#!/usr/bin/env python

from typing import Any

from lerobot.types import RobotAction

from ..teleoperator import Teleoperator
from .config_noop import NoopTeleopConfig


class NoopTeleop(Teleoperator):
    config_class = NoopTeleopConfig
    name = "noop"

    def __init__(self, config: NoopTeleopConfig):
        super().__init__(config)
        self._is_connected = False

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self) -> None:
        self._is_connected = True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> RobotAction:
        return {}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        self._is_connected = False
