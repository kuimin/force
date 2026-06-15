#!/usr/bin/env python

import logging
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
        return {
            **self._joint_pos_ft,
            **self._joint_vel_ft,
            **self._tcp_pose_ft,
            **self._tcp_speed_ft,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joint_pos_ft

    @property
    def is_connected(self) -> bool:
        return self.rtde_c is not None and self.rtde_r is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )

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

        return obs

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

        return {f"{joint}.pos": float(value) for joint, value in zip(JOINT_NAMES, target, strict=True)}

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
        logger.info("%s disconnected.", self)
