#!/usr/bin/env python

import glob
import importlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_gello import GelloTeleopConfig

logger = logging.getLogger(__name__)


def _lock_file_for_port(port: str) -> Path:
    real_port = os.path.realpath(port)
    lock_name = os.path.basename(real_port).replace("/", "_")
    return Path(tempfile.gettempdir()) / f"gello_lock_{lock_name}"


class _PortLock:
    """Small lock file guard so two processes do not open the same GELLO serial port."""

    def __init__(self, port: str):
        self.port = port
        self.path = _lock_file_for_port(port)
        self._locked = False

    def acquire(self) -> None:
        if self.path.exists():
            try:
                pid = int(self.path.read_text(encoding="utf-8").strip())
                if Path(f"/proc/{pid}").exists():
                    raise RuntimeError(f"GELLO port {self.port} is already locked by process {pid}")
            except ValueError:
                pass
            self.path.unlink(missing_ok=True)

        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
        except FileExistsError as exc:
            raise RuntimeError(f"GELLO port {self.port} is already locked: {self.path}") from exc
        self._locked = True

    def release(self) -> None:
        if not self._locked:
            return
        try:
            if self.path.exists() and self.path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self.path.unlink()
        finally:
            self._locked = False


def _import_gello_agent(extra_path: Path | None):
    if extra_path is not None:
        import sys

        sys.path.insert(0, str(extra_path))

    import_errors = []
    for module_name in (
        "gello.agents.gello_agent",
        "utils.gello_software.gello.agents.gello_agent",
    ):
        try:
            module = importlib.import_module(module_name)
            return module.GelloAgent
        except Exception as exc:  # noqa: BLE001 - keep the import fallback readable for hardware setup.
            import_errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "Cannot import GELLO GelloAgent. Install/copy the GELLO software package, "
        "or pass --teleop.gello_software_path=/path/to/gello_software. Tried: "
        + "; ".join(import_errors)
    )


def _discover_ports(count: int) -> list[str]:
    ports = sorted(glob.glob("/dev/serial/by-id/*"))
    if len(ports) < count:
        raise RuntimeError(f"Need {count} GELLO serial port(s), found {len(ports)}: {ports}")
    return ports[:count]


class GelloTeleop(Teleoperator):
    """LeRobot teleoperator wrapper for GELLO.

    Single-arm mode returns actions matching the UR5e wrapper:
    ``shoulder_pan.pos`` ... ``wrist_3.pos``.  The raw GELLO 7th value can be
    exposed as ``gripper.pos`` by setting ``include_gripper=True``.
    """

    config_class = GelloTeleopConfig
    name = "gello"

    def __init__(self, config: GelloTeleopConfig):
        super().__init__(config)
        self.config = config
        self._agent_cls = None
        self._left_agent = None
        self._right_agent = None
        self._single_agent = None
        self._locks: list[_PortLock] = []
        self.logs: dict[str, Any] = {}

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.bimanual:
            features = {
                f"{prefix}{joint}.pos": float
                for prefix in (self.config.left_prefix, self.config.right_prefix)
                for joint in self.config.joint_names
            }
            if self.config.include_gripper:
                features[f"{self.config.left_prefix}{self.config.gripper_name}.pos"] = float
                features[f"{self.config.right_prefix}{self.config.gripper_name}.pos"] = float
            return features

        features = {f"{joint}.pos": float for joint in self.config.joint_names}
        if self.config.include_gripper:
            features[f"{self.config.gripper_name}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        if self.config.bimanual:
            return self._left_agent is not None and self._right_agent is not None
        return self._single_agent is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._agent_cls = _import_gello_agent(self.config.gello_software_path)

        if self.config.bimanual:
            left_port, right_port = (
                _discover_ports(2) if self.config.auto_discover_ports else [self.config.left_port, self.config.right_port]
            )
            assert left_port is not None and right_port is not None
            self._lock_ports([left_port, right_port])
            self._left_agent = self._agent_cls(port=left_port)
            self._right_agent = self._agent_cls(port=right_port)
            logger.info("%s connected to GELLO ports left=%s right=%s", self, left_port, right_port)
        else:
            (single_port,) = _discover_ports(1) if self.config.auto_discover_ports else [self.config.single_port]
            assert single_port is not None
            self._lock_ports([single_port])
            self._single_agent = self._agent_cls(port=single_port)
            logger.info("%s connected to GELLO port %s", self, single_port)

    def _lock_ports(self, ports: list[str]) -> None:
        try:
            for port in ports:
                lock = _PortLock(port)
                lock.acquire()
                self._locks.append(lock)
        except Exception:
            self._unlock_ports()
            raise

    def _unlock_ports(self) -> None:
        for lock in self._locks:
            lock.release()
        self._locks = []

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _read_agent(self, agent) -> np.ndarray:
        values = np.asarray(agent.act(), dtype=np.float64).reshape(-1)
        if values.size < self.config.num_joints_per_arm:
            raise RuntimeError(
                f"GELLO returned {values.size} values, expected at least {self.config.num_joints_per_arm}"
            )
        return values

    def _commands_to_action(self, commands: np.ndarray, prefix: str = "") -> RobotAction:
        action: RobotAction = {}
        joints = commands[: self.config.num_joints_per_arm]
        for joint, value in zip(self.config.joint_names, joints, strict=True):
            action[f"{prefix}{joint}.pos"] = float(value)

        if self.config.include_gripper:
            gripper_value = float(commands[self.config.num_joints_per_arm]) if commands.size > self.config.num_joints_per_arm else 0.0
            action[f"{prefix}{self.config.gripper_name}.pos"] = gripper_value
        return action

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        if self.config.bimanual:
            left = self._read_agent(self._left_agent)
            right = self._read_agent(self._right_agent)
            action = {
                **self._commands_to_action(left, self.config.left_prefix),
                **self._commands_to_action(right, self.config.right_prefix),
            }
        else:
            action = self._commands_to_action(self._read_agent(self._single_agent))
        self.logs["read_pos_dt_s"] = time.perf_counter() - start
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        for agent in (self._single_agent, self._left_agent, self._right_agent):
            if agent is not None and hasattr(agent, "stop"):
                agent.stop()
        self._single_agent = None
        self._left_agent = None
        self._right_agent = None
        self._unlock_ports()
        logger.info("%s disconnected.", self)
