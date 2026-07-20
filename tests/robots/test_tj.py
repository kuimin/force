#!/usr/bin/env python

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.tj import TJRobot, TJRobotConfig
from lerobot.robots.tj.robotiq_usb_gripper import RobotiqUsbGripper
from lerobot.robots.utils import make_robot_from_config


class FakeDCSS:
    pass


class FakeMarvinRobot:
    def __init__(self):
        self.connected = False
        self.target = [0.0] * 7
        self.send_cmd_count = 0
        self.send_cmd_wait_response_count = 0

    def connect(self, _ip):
        self.connected = True
        return 1

    def check_error_and_clear(self, _dcss):
        return None

    def subscribe(self, _dcss):
        return {
            "outputs": [
                {"frame_serial": 1, "fb_joint_pos": [1, 2, 3, 4, 5, 6, 7], "fb_joint_vel": [0] * 7},
                {"frame_serial": 1, "fb_joint_pos": [11, 12, 13, 14, 15, 16, 17], "fb_joint_vel": [1] * 7},
            ]
        }

    def log_switch(self, _flag):
        return 1

    def clear_set(self):
        return 1

    def set_state(self, arm, state):
        self.state = (arm, state)
        return 1

    def set_vel_acc(self, arm, velRatio, AccRatio):
        self.vel_acc = (arm, velRatio, AccRatio)
        return 1

    def send_cmd(self):
        self.send_cmd_count = getattr(self, "send_cmd_count", 0) + 1
        return 1

    def send_cmd_wait_response(self, _timeout_ms):
        self.send_cmd_wait_response_count = getattr(self, "send_cmd_wait_response_count", 0) + 1
        return 1

    def set_joint_cmd_pose(self, arm, joints):
        self.target = joints
        self.target_arm = arm
        return 1

    def release_robot(self):
        self.connected = False
        return 1


@pytest.fixture
def fake_sdk():
    return SimpleNamespace(Marvin_Robot=FakeMarvinRobot, DCSS=FakeDCSS)


def test_tj_config_registered_and_factory_creates_robot():
    robot = make_robot_from_config(TJRobotConfig(ip="127.0.0.1"))

    assert isinstance(robot, TJRobot)


def test_tj_connect_get_observation_and_disconnect(fake_sdk):
    with patch("lerobot.robots.tj.tj._load_marvin_sdk", return_value=fake_sdk):
        robot = TJRobot(TJRobotConfig(ip="127.0.0.1", connect_frame_checks=0))
        robot.connect()

    assert robot.is_connected
    assert robot.robot.state == ("A", 1)
    assert robot.robot.vel_acc == ("A", 10, 10)

    obs = robot.get_observation()
    assert obs["joint_1.pos"] == 1.0
    assert obs["joint_7.vel"] == 0.0

    robot.disconnect()
    assert not robot.is_connected


def test_tj_send_action_to_selected_arm(fake_sdk):
    with patch("lerobot.robots.tj.tj._load_marvin_sdk", return_value=fake_sdk):
        robot = TJRobot(TJRobotConfig(ip="127.0.0.1", arm="B", max_relative_target=None, connect_frame_checks=0))
        robot.connect()

    action = {f"joint_{idx}.pos": float(idx) for idx in range(1, 8)}
    sent = robot.send_action(action)

    assert sent == action
    assert robot.robot.target_arm == "B"
    assert robot.robot.target == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert robot.robot.send_cmd_wait_response_count == 0


def test_tj_dmtac_image_features_are_one_raw_image_per_sensor():
    robot = TJRobot(
        TJRobotConfig(
            ip="127.0.0.1",
            enable_dmtac_images=True,
            dmtac_dev_ids=[0, 1],
        )
    )

    assert robot.observation_features["dmtac_0_raw"] == (480, 640, 3)
    assert robot.observation_features["dmtac_1_raw"] == (480, 640, 3)
    assert len(robot.observation_features) == 16


def test_tj_get_observation_includes_dmtac_images():
    robot = TJRobot(TJRobotConfig(ip="127.0.0.1"))
    robot.robot = FakeMarvinRobot()
    robot.dcss = FakeDCSS()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    robot.dmtac = SimpleNamespace(
        is_connected=True,
        read=lambda: {
            "dmtac_0_raw": image,
        },
    )

    obs = robot.get_observation()

    assert np.array_equal(obs["dmtac_0_raw"], image)


def test_tj_robotiq_usb_gripper_maps_norm_to_raw_position():
    robot = TJRobot(
        TJRobotConfig(
            ip="127.0.0.1",
            gripper_backend="robotiq_usb",
            gripper_send_deadband=0.0,
        )
    )
    robot.robot = object()
    robot.gripper = MagicMock()
    robot.gripper.move_norm.return_value = 128

    robot._send_gripper_action(0.5)

    robot.gripper.move_norm.assert_called_once_with(0.5)
    assert robot._last_gripper_value == 0.5
    assert robot._last_gripper_state == "128"


def test_robotiq_usb_gripper_move_norm_uses_raw_position_range():
    gripper = RobotiqUsbGripper()
    gripper._driver = MagicMock()

    assert gripper.move_norm(0.0) == 0
    assert gripper.move_norm(0.5) == 128
    assert gripper.move_norm(1.0) == 255

    positions = [call.args[0] for call in gripper._driver.move.call_args_list]
    assert positions == [0, 128, 255]


def test_tj_connect_reports_failure(fake_sdk):
    fake_robot = MagicMock()
    fake_robot.connect.return_value = 0
    fake_sdk.Marvin_Robot = MagicMock(return_value=fake_robot)

    with patch("lerobot.robots.tj.tj._load_marvin_sdk", return_value=fake_sdk):
        robot = TJRobot(TJRobotConfig(ip="127.0.0.1"))
        with pytest.raises(ConnectionError, match="Failed to connect TJ robot"):
            robot.connect()
