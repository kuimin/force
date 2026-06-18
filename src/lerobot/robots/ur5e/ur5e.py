#!/usr/bin/env python

import logging
import socket
import threading
import time
from collections import OrderedDict
from functools import cached_property

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_ur5e import UR5eRobotConfig

logger = logging.getLogger(__name__)


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
TCP_POSE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
TCP_SPEED_NAMES = ["tcp_vx", "tcp_vy", "tcp_vz", "tcp_wx", "tcp_wy", "tcp_wz"]


class _RobotiqGripper:
    ACT = "ACT"
    ATR = "ATR"
    FLT = "FLT"
    FOR = "FOR"
    GTO = "GTO"
    OBJ = "OBJ"
    POS = "POS"
    SPE = "SPE"
    STA = "STA"

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.socket: socket.socket | None = None
        self.command_lock = threading.Lock()

    def connect(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(2.0)
        self.socket.connect((self.ip, self.port))

    def disconnect(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _send_recv(self, command: str) -> bytes:
        if self.socket is None:
            raise RuntimeError("Robotiq gripper is not connected")
        with self.command_lock:
            self.socket.sendall(command.encode("UTF-8"))
            return self.socket.recv(1024)

    def get_var(self, variable: str) -> int:
        data = self._send_recv(f"GET {variable}\n")
        name, value = data.decode("UTF-8").split()
        if name != variable:
            raise RuntimeError(f"Unexpected Robotiq response: {data!r}")
        return int(value)

    def set_vars(self, values: OrderedDict[str, int]) -> None:
        command = "SET" + "".join(f" {name} {value}" for name, value in values.items()) + "\n"
        data = self._send_recv(command)
        if data != b"ack":
            raise RuntimeError(f"Robotiq did not ack command {command!r}: {data!r}")

    def activate(self) -> None:
        if self.get_var(self.STA) == 3:
            return
        self.reset()
        self.set_vars(OrderedDict([(self.ACT, 1)]))
        time.sleep(1.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.get_var(self.ACT) == 1 and self.get_var(self.STA) == 3:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for Robotiq activation")

    def reset(self) -> None:
        self.set_vars(OrderedDict([(self.ACT, 0), (self.ATR, 0)]))
        time.sleep(0.5)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.get_var(self.ACT) == 0 and self.get_var(self.STA) == 0:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for Robotiq reset")

    def get_position(self) -> int:
        return self.get_var(self.POS)

    def move(self, position: int, speed: int, force: int) -> int:
        clip_pos = int(np.clip(position, 0, 255))
        clip_speed = int(np.clip(speed, 0, 255))
        clip_force = int(np.clip(force, 0, 255))
        self.set_vars(
            OrderedDict(
                [
                    (self.POS, clip_pos),
                    (self.SPE, clip_speed),
                    (self.FOR, clip_force),
                    (self.GTO, 1),
                ]
            )
        )
        return clip_pos

    def move_and_wait_for_recv(self, position: int, speed: int, force: int) -> tuple[int, int]:
        self.move(position, speed, force)
        return self.get_var(self.POS), self.get_var(self.OBJ)


class UR5eRobot(Robot):
    """LeRobot wrapper for a UR5e arm through ``ur_rtde``.

    This is a lightweight adaptation of the Embodied-RL UR5EJP controller: it uses
    ``RTDEReceiveInterface`` for feedback and ``servoJ``/``moveJ`` for joint commands,
    but exposes the standard LeRobot ``Robot`` interface.
    """

    config_class = UR5eRobotConfig
    name = "ur5e"

    def __init__(self, config: UR5eRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.rtde_c = None
        self.rtde_r = None
        self.gripper: _RobotiqGripper | None = None
        self._last_gripper_position: int | None = None
        self._last_gripper_value: float | None = None

    @property
    def _joint_pos_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINT_NAMES}

    @property
    def _joint_vel_ft(self) -> dict[str, type]:
        return {f"{joint}.vel": float for joint in JOINT_NAMES}

    @property
    def _tcp_pose_ft(self) -> dict[str, type]:
        return {name: float for name in TCP_POSE_NAMES}

    @property
    def _tcp_speed_ft(self) -> dict[str, type]:
        return {name: float for name in TCP_SPEED_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = {
            **self._joint_pos_ft,
            **self._joint_vel_ft,
            **self._tcp_pose_ft,
            **self._tcp_speed_ft,
            **self._cameras_ft,
        }
        if self.config.use_gripper:
            features[f"{self.config.gripper_name}.pos"] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = dict(self._joint_pos_ft)
        if self.config.use_gripper:
            features[f"{self.config.gripper_name}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        robot_connected = self.rtde_c is not None and self.rtde_r is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )
        if self.config.use_gripper:
            return robot_connected and self.gripper is not None and self.gripper.socket is not None
        return robot_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise ImportError("UR5e support requires 'ur_rtde'. Install it with: pip install ur_rtde") from exc

        self.rtde_c = RTDEControlInterface(self.config.ip)
        self.rtde_r = RTDEReceiveInterface(self.config.ip)
        self.configure()

        if self.config.joints_init is not None:
            target = np.asarray(self.config.joints_init, dtype=np.float64)
            self.rtde_c.moveJ(target, self.config.joints_init_speed, self.config.max_joint_acc)

        if self.config.reset_joints is not None:
            target = np.asarray(self.config.reset_joints, dtype=np.float64)
            self.rtde_c.moveJ(target, self.config.joints_init_speed, self.config.max_joint_acc)

        for cam in self.cameras.values():
            cam.connect()

        if self.config.use_gripper:
            gripper_ip = self.config.gripper_ip or self.config.ip
            self.gripper = _RobotiqGripper(gripper_ip, self.config.gripper_port)
            self.gripper.connect()
            if self.config.gripper_activate_on_connect:
                self.gripper.activate()
            self._last_gripper_position = None
            self._last_gripper_value = None
            logger.info("%s connected to Robotiq gripper at %s:%s", self, gripper_ip, self.config.gripper_port)

        logger.info("%s connected to UR5e at %s", self, self.config.ip)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        if self.rtde_c is None:
            return
        if self.config.tcp_offset_pose is not None:
            self.rtde_c.setTcp(np.asarray(self.config.tcp_offset_pose, dtype=np.float64))
        if self.config.payload_mass is not None:
            if self.config.payload_cog is not None:
                self.rtde_c.setPayload(
                    self.config.payload_mass,
                    np.asarray(self.config.payload_cog, dtype=np.float64),
                )
            else:
                self.rtde_c.setPayload(self.config.payload_mass)

    def _read_joints(self) -> np.ndarray:
        if self.rtde_r is None:
            raise RuntimeError("UR5e is not connected")
        return np.asarray(self.rtde_r.getActualQ(), dtype=np.float64)

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        if self.config.joint_limits is not None:
            limits = np.asarray(self.config.joint_limits, dtype=np.float64)
            target = np.clip(target, limits[0], limits[1])
        if self.config.max_relative_target is not None:
            current = self._read_joints()
            lower = current - self.config.max_relative_target
            upper = current + self.config.max_relative_target
            clipped = np.clip(target, lower, upper)
            if not np.allclose(clipped, target):
                logger.warning("UR5e target joints clipped by max_relative_target=%s", self.config.max_relative_target)
            target = clipped
        return target

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        assert self.rtde_r is not None
        obs: RobotObservation = {}

        actual_q = np.asarray(self.rtde_r.getActualQ(), dtype=np.float64)
        actual_qd = np.asarray(self.rtde_r.getActualQd(), dtype=np.float64)
        tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64)
        tcp_speed = np.asarray(self.rtde_r.getActualTCPSpeed(), dtype=np.float64)

        for joint, value in zip(JOINT_NAMES, actual_q, strict=True):
            obs[f"{joint}.pos"] = float(value)
        for joint, value in zip(JOINT_NAMES, actual_qd, strict=True):
            obs[f"{joint}.vel"] = float(value)
        for name, value in zip(TCP_POSE_NAMES, tcp_pose, strict=True):
            obs[name] = float(value)
        for name, value in zip(TCP_SPEED_NAMES, tcp_speed, strict=True):
            obs[name] = float(value)

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        if self.config.use_gripper:
            obs[f"{self.config.gripper_name}.pos"] = float(
                self._last_gripper_value if self._last_gripper_value is not None else 0.0
            )

        return obs

    def normalized_to_gripper_position(self, value: float) -> int:
        """Map a continuous GELLO gripper value in [0, 1] to Robotiq POS."""
        normalized = float(np.clip(value, 0.0, 1.0))
        open_pos = self.config.gripper_open_position
        closed_pos = self.config.gripper_closed_position
        return int(round(open_pos + normalized * (closed_pos - open_pos)))

    def _send_gripper_action(self, action: RobotAction) -> float | None:
        if not self.config.use_gripper:
            return None
        if self.gripper is None:
            raise RuntimeError("Robotiq gripper is enabled but not connected")

        key = f"{self.config.gripper_name}.pos"
        if key not in action:
            return None

        gripper_value = float(np.clip(float(action[key]), 0.0, 1.0))
        target_position = self.normalized_to_gripper_position(gripper_value)
        if (
            self._last_gripper_position is None
            or abs(target_position - self._last_gripper_position) >= self.config.gripper_command_deadband
        ):
            self.gripper.move(
                target_position,
                self.config.gripper_speed,
                self.config.gripper_force,
            )
            self._last_gripper_position = target_position
        self._last_gripper_value = gripper_value
        return gripper_value

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self.rtde_c is not None
        target = np.array([action[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=np.float64)
        target = self._clip_target(target)

        if self.config.control_mode == "servoJ":
            self.rtde_c.servoJ(
                target,
                self.config.max_joint_speed,
                self.config.max_joint_acc,
                self.config.servo_time,
                self.config.lookahead_time,
                self.config.gain,
            )
        else:
            self.rtde_c.moveJ(target, self.config.max_joint_speed, self.config.max_joint_acc)

        sent_action = {f"{joint}.pos": float(value) for joint, value in zip(JOINT_NAMES, target, strict=True)}
        gripper_value = self._send_gripper_action(action)
        if gripper_value is not None:
            sent_action[f"{self.config.gripper_name}.pos"] = gripper_value
        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.rtde_c is not None:
            try:
                if self.config.control_mode == "servoJ":
                    self.rtde_c.servoStop()
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            finally:
                self.rtde_c = None
        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
            finally:
                self.rtde_r = None
        for cam in self.cameras.values():
            cam.disconnect()
        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            finally:
                self.gripper = None
        logger.info("%s disconnected.", self)
