#!/usr/bin/env python

import importlib.util
import logging
import time
from functools import cached_property
from pathlib import Path
from types import ModuleType

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_tj import TJRobotConfig
from .robotiq_usb_gripper import RobotiqUsbGripper

logger = logging.getLogger(__name__)


def _default_sdk_python_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "third_party" / "tj_marvin_sdk" / "SDK_PYTHON"


def _load_marvin_sdk(sdk_python_dir: Path) -> ModuleType:
    sdk_file = sdk_python_dir / "fx_robot.py"
    if not sdk_file.is_file():
        raise ImportError(f"TJ SDK file not found: {sdk_file}")
    spec = importlib.util.spec_from_file_location("lerobot_tj_fx_robot", sdk_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TJ SDK from {sdk_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TJRobot(Robot):
    config_class = TJRobotConfig
    name = "tj"

    def __init__(self, config: TJRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.sdk = None
        self.robot = None
        self.dcss = None
        self.gripper = None
        self._last_commanded_joints = None
        self._last_gripper_value = None
        self._last_gripper_state = None
        self._warned_gripper_unconfigured = False
        self._last_send_log_s = 0.0

    @property
    def _arm_index(self) -> int:
        return 0 if self.config.arm == "A" else 1

    @property
    def _joint_pos_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joint_names}

    @property
    def _joint_vel_ft(self) -> dict[str, type]:
        return {f"{joint}.vel": float for joint in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joint_pos_ft, **self._joint_vel_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = dict(self._joint_pos_ft)
        if self.config.enable_gripper_action:
            features[f"{self.config.gripper_name}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self.robot is not None and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        sdk_python_dir = self.config.sdk_python_dir or _default_sdk_python_dir()
        self.sdk = _load_marvin_sdk(sdk_python_dir)
        self.robot = self.sdk.Marvin_Robot()
        self.dcss = self.sdk.DCSS()

        if not self.robot.connect(self.config.ip):
            self.robot = None
            self.dcss = None
            raise ConnectionError(f"Failed to connect TJ robot at {self.config.ip}")

        try:
            if self.config.check_error_on_connect:
                self.robot.check_error_and_clear(self.dcss)
            self._check_frame_updates()
            self.configure()
            self._connect_gripper()
        except Exception:
            if self.gripper is not None:
                self.gripper.disconnect()
                self.gripper = None
            self.robot.release_robot()
            self.robot = None
            self.dcss = None
            self.sdk = None
            raise

        for cam in self.cameras.values():
            cam.connect()
        logger.info("%s connected to TJ robot arm %s at %s", self, self.config.arm, self.config.ip)

    def _connect_gripper(self) -> None:
        if not self.config.enable_gripper_action or self.config.gripper_backend != "robotiq_usb":
            return
        self.gripper = RobotiqUsbGripper(
            port=self.config.robotiq_usb_port,
            device_id=self.config.robotiq_usb_device_id,
            speed=self.config.robotiq_usb_speed,
            force=self.config.robotiq_usb_force,
            activate_on_connect=self.config.robotiq_usb_activate_on_connect,
            open_on_connect=self.config.robotiq_usb_open_on_connect,
        )
        self.gripper.connect()

    def _check_frame_updates(self) -> None:
        if self.config.connect_frame_checks == 0:
            return
        frame = None
        updates = 0
        for _ in range(self.config.connect_frame_checks):
            sub_data = self._subscribe()
            next_frame = sub_data["outputs"][self._arm_index]["frame_serial"]
            if next_frame != 0 and next_frame != frame:
                updates += 1
                frame = next_frame
            time.sleep(self.config.connect_frame_check_dt_s)
        if updates == 0:
            raise ConnectionError("TJ robot connected, but subscription frames did not update")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        if self.robot is None:
            return
        self.robot.log_switch("1" if self.config.log_switch else "0")
        self.robot.local_log_switch("1" if self.config.local_log_switch else "0")
        self.robot.clear_set()
        self.robot.set_state(arm=self.config.arm, state=1)
        self.robot.set_vel_acc(
            arm=self.config.arm,
            velRatio=self.config.velocity_percent,
            AccRatio=self.config.acceleration_percent,
        )
        if not self.robot.send_cmd():
            raise RuntimeError(f"Failed to configure TJ arm {self.config.arm} in position mode")
        time.sleep(0.1)

    def _subscribe(self) -> dict:
        if self.robot is None or self.dcss is None:
            raise RuntimeError("TJ robot is not connected")
        sub_data = self.robot.subscribe(self.dcss)
        if not sub_data:
            raise RuntimeError("TJ robot subscription returned no data")
        return sub_data

    def _read_joints(self) -> np.ndarray:
        sub_data = self._subscribe()
        joints = np.asarray(sub_data["outputs"][self._arm_index]["fb_joint_pos"], dtype=np.float64)
        if joints.shape != (7,):
            raise RuntimeError(f"TJ SDK returned {joints.shape} joint positions, expected (7,)")
        return joints

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        if self.config.joint_limits is not None:
            limits = np.asarray(self.config.joint_limits, dtype=np.float64)
            target = np.clip(target, limits[0], limits[1])
        if self.config.max_relative_target is not None:
            basis = self._last_commanded_joints
            if basis is None:
                basis = self._read_joints()
            clipped = np.clip(target, basis - self.config.max_relative_target, basis + self.config.max_relative_target)
            if not np.allclose(clipped, target):
                logger.warning("TJ target joints clipped by max_relative_target=%s", self.config.max_relative_target)
            target = clipped
        return target

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        sub_data = self._subscribe()
        output = sub_data["outputs"][self._arm_index]
        actual_q = np.asarray(output["fb_joint_pos"], dtype=np.float64)
        actual_qd = np.asarray(output["fb_joint_vel"], dtype=np.float64)

        obs: RobotObservation = {}
        for joint, value in zip(self.config.joint_names, actual_q, strict=True):
            obs[f"{joint}.pos"] = float(value)
        for joint, value in zip(self.config.joint_names, actual_qd, strict=True):
            obs[f"{joint}.vel"] = float(value)
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.robot is None:
            raise RuntimeError("TJ robot is not connected")
        target = np.asarray([action[f"{joint}.pos"] for joint in self.config.joint_names], dtype=np.float64)
        target = self._clip_target(target)
        gripper_key = f"{self.config.gripper_name}.pos"
        gripper_value = None
        if self.config.enable_gripper_action and gripper_key in action:
            gripper_value = float(np.clip(float(action[gripper_key]), 0.0, 1.0))

        self.robot.clear_set()
        ok_set = self.robot.set_joint_cmd_pose(arm=self.config.arm, joints=target.tolist())
        if self.config.wait_response_on_action:
            send_response = self.robot.send_cmd_wait_response(self.config.action_response_timeout_ms)
        else:
            send_response = self.robot.send_cmd()
        if not ok_set or send_response <= 0:
            raise RuntimeError(
                f"Failed to send TJ joint command to arm {self.config.arm}: "
                f"ok_set={ok_set}, send_response={send_response}"
            )
        self._last_commanded_joints = target.copy()
        if gripper_value is not None:
            self._send_gripper_action(gripper_value)
        now = time.monotonic()
        if now - self._last_send_log_s >= 0.5:
            logger.info("TJ sent q(deg)=%s", [round(float(value), 3) for value in target])
            self._last_send_log_s = now
        sent = {f"{joint}.pos": float(value) for joint, value in zip(self.config.joint_names, target, strict=True)}
        if gripper_value is not None:
            sent[gripper_key] = gripper_value
        return sent

    def _send_gripper_action(self, value: float) -> None:
        if self.robot is None:
            return
        if (
            self._last_gripper_value is not None
            and abs(value - self._last_gripper_value) < self.config.gripper_send_deadband
        ):
            return

        if self.config.gripper_backend == "robotiq_usb":
            if self.gripper is None:
                raise RuntimeError("TJ Robotiq USB gripper backend is enabled but the gripper is not connected")
            position = self.gripper.move_norm(value)
            logger.info(
                "TJ sent Robotiq USB gripper command (%s.pos=%.3f -> %s/255)",
                self.config.gripper_name,
                value,
                position,
            )
            self._last_gripper_value = value
            self._last_gripper_state = str(position)
            return

        close = value >= self.config.gripper_close_threshold
        state = "close" if close else "open"
        if state == self._last_gripper_state:
            self._last_gripper_value = value
            return

        hex_command = self.config.gripper_close_hex if close else self.config.gripper_open_hex
        channel = self.config.gripper_command_channel
        if channel is None or not hex_command:
            if not self._warned_gripper_unconfigured:
                logger.warning(
                    "Received %s.pos but no TJ gripper protocol is configured; "
                    "set gripper_command_channel plus gripper_open_hex/gripper_close_hex to move hardware",
                    self.config.gripper_name,
                )
                self._warned_gripper_unconfigured = True
            self._last_gripper_value = value
            self._last_gripper_state = state
            return

        payload = bytes.fromhex(hex_command.replace(" ", ""))
        if hasattr(self.robot, "set_ch_data"):
            response = self.robot.set_ch_data(self.config.arm, payload, len(payload), channel)
        else:
            response = self.robot.set_485_data(self.config.arm, payload, len(payload), channel)
        if response <= 0:
            raise RuntimeError(
                f"Failed to send TJ gripper {state} command on channel {channel}: response={response}"
            )
        logger.info("TJ sent gripper %s command (%s.pos=%.3f)", state, self.config.gripper_name, value)
        self._last_gripper_value = value
        self._last_gripper_state = state

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                if self.config.disable_on_disconnect:
                    self.robot.clear_set()
                    self.robot.set_state(arm=self.config.arm, state=0)
                    self.robot.send_cmd()
                self.robot.release_robot()
            finally:
                if self.gripper is not None:
                    self.gripper.disconnect()
                    self.gripper = None
                self.robot = None
                self.dcss = None
                self.sdk = None
                self._last_commanded_joints = None
                self._last_gripper_value = None
                self._last_gripper_state = None
                self._warned_gripper_unconfigured = False
                self._last_send_log_s = 0.0
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)
